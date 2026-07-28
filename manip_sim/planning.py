"""Constrained motion planning: mink IK + OMPL ProjectedStateSpace on TSRs.

This is the CBiRRT role in the architecture, assembled from off-the-shelf
parts rather than reimplemented:

    goal configs   sampled TSR poses (tsr.sample_intersection)
                     -> mink differential IK, multi-seed      (MinkIK)
    path manifold  path-TSR excess as an OMPL ob.Constraint   (TSRConstraint)
                     -> ob.ProjectedStateSpace + og.RRTConnect
                        == bidirectional RRT with Newton projection onto
                           {q : excess(FK(q)) = 0}, i.e. CBiRRT

Everything operates on a SCRATCH MjData (never the live simulation), so
planning cannot perturb physics state. The attached object is kinematic:
T0_body(q) = FK_ee(q) @ T_ee_body, with T_ee_body the measured grasp
transform (in sim: relative pose read once at grasp completion).

Known v1 limits, on purpose:
  * collision checking covers everything IN THE MODEL (robot self, table,
    free objects) but the attached object's mesh is not yet welded into the
    model, so attached-object collisions are unchecked. Wiring a mocap-weld
    or geom-reparent for the grasped object is the follow-up.
  * the constraint residual is the clamped TSR excess: zero (with zero
    Jacobian) everywhere inside the bounds, so OMPL's Newton projection is
    a no-op for satisfied states and pulls boundary violations back — the
    same semantics as CBiRRT's TSR projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .tsr import TSR, make_pose

# --------------------------------------------------------------- kinematics


def _raw_model(env):
    """robosuite's MjSim wraps mujoco.MjModel; unwrap defensively."""
    m = env.sim.model
    return getattr(m, "_model", m)


class ArmKinematics:
    """FK + collision queries for one arm on a scratch MjData."""

    def __init__(self, env, eef_site: str = "gripper0_right_grip_site",
                 joint_prefix: str = "robot0_"):
        self.model = _raw_model(env)
        self.data = mujoco.MjData(self.model)
        # start scratch state from the live sim so free objects sit where
        # they really are (matters for collision checking); clamp into joint
        # limits -- robosuite leaves gripper fingers epsilon outside their
        # ranges at reset, which otherwise makes mink warn on every solve
        self.data.qpos[:] = env.sim.data.qpos
        for j in range(self.model.njnt):
            if self.model.jnt_limited[j]:
                adr = self.model.jnt_qposadr[j]
                lo, hi = self.model.jnt_range[j]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr], lo, hi)
        mujoco.mj_forward(self.model, self.data)

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, eef_site)
        assert self.site_id >= 0, f"site '{eef_site}' not in model"
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
        ]
        self.arm_joint_ids = [
            j for j, n in enumerate(self.joint_names)
            if n and n.startswith(joint_prefix)
            and self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                           mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        self.qpos_ids = np.array(
            [self.model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.joint_range = np.array(
            [self.model.jnt_range[j] for j in self.arm_joint_ids])
        self.n = len(self.qpos_ids)

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """-> 4x4 world pose of the eef site at arm config q."""
        self.set_q(q)
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T

    def in_collision(self, q: np.ndarray, depth_tol: float = 1e-4,
                     robot_prefixes=("robot0", "gripper0", "mount0"),
                     allowed_prefix_pairs=(("gripper0", "gripper0"),)) -> bool:
        """True if any contact INVOLVING THE ROBOT penetrates deeper than
        depth_tol. Contacts between free objects and the scene (mug resting
        on the table, etc.) are configuration-independent facts about the
        world, not collisions the arm caused, and are ignored -- counting
        them would veto every configuration. Pairs matching
        allowed_prefix_pairs (default: gripper internal contacts) are also
        ignored."""
        self.set_q(q)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist > -depth_tol:
                continue
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[con.geom1]) or ""
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[con.geom2]) or ""
            if not (b1.startswith(robot_prefixes) or b2.startswith(robot_prefixes)):
                continue                       # scene<->scene contact
            if any((b1.startswith(p) and b2.startswith(q_)) or
                   (b1.startswith(q_) and b2.startswith(p))
                   for p, q_ in allowed_prefix_pairs):
                continue
            return True
        return False


@dataclass
class AttachedObject:
    """Kinematic attachment: object body pose rides the eef."""

    T_ee_body: np.ndarray            # eef -> object body (measured at grasp)

    def body_pose(self, T0_ee: np.ndarray) -> np.ndarray:
        return T0_ee @ self.T_ee_body


# ---------------------------------------------------------------------- IK


class MinkIK:
    """Multi-seed differential IK to an eef site pose, via mink."""

    def __init__(self, kin: ArmKinematics, eef_site: str = "gripper0_right_grip_site",
                 solver: str = "daqp"):
        import mink
        self._mink = mink
        self.kin = kin
        self.config = mink.Configuration(kin.model)
        self.task = mink.FrameTask(frame_name=eef_site, frame_type="site",
                                   position_cost=1.0, orientation_cost=1.0)
        self.posture = mink.PostureTask(kin.model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(kin.model)]
        self.solver = solver

    def solve(self, T0_ee_target: np.ndarray, q_seed: np.ndarray,
              iters: int = 120, dt: float = 0.05,
              pos_tol: float = 2e-3, rot_tol: float = 8e-3):
        """-> (q[6], ok). ok iff position within pos_tol (m) and rotation
        within rot_tol (rad) at the returned config."""
        mink = self._mink
        qfull = self.kin.data.qpos.copy()
        qfull[self.kin.qpos_ids] = q_seed
        self.config.update(qfull)
        self.posture.set_target(qfull)
        self.task.set_target(mink.SE3.from_matrix(T0_ee_target))
        for _ in range(iters):
            v = mink.solve_ik(self.config, [self.task, self.posture],
                              dt, self.solver, limits=self.limits)
            self.config.integrate_inplace(v, dt)
        q = self.config.q[self.kin.qpos_ids].copy()
        T = self.kin.fk(q)
        pos_err = float(np.linalg.norm(T[:3, 3] - T0_ee_target[:3, 3]))
        rot_err = float(np.linalg.norm(
            R.from_matrix(T[:3, :3].T @ T0_ee_target[:3, :3]).as_rotvec()))
        return q, (pos_err <= pos_tol and rot_err <= rot_tol)

    def solve_multiseed(self, T0_ee_target: np.ndarray, seeds, **kw):
        for s in seeds:
            q, ok = self.solve(T0_ee_target, np.asarray(s, float), **kw)
            if ok:
                return q, True
        return None, False


# ---------------------------------------------------------------- planning

try:
    from ompl import base as ob
    from ompl import geometric as og
    _HAVE_OMPL = True
except ImportError:                                    # pragma: no cover
    _HAVE_OMPL = False


if _HAVE_OMPL:

    class TSRConstraint(ob.Constraint):
        """OMPL constraint: the attached object's pose must have zero excess
        w.r.t. every path TSR. One SCALAR residual per TSR (norm of the
        6-vector excess): the excess is zero on the whole feasible region,
        so a full 6-dim residual would declare a 0-dim manifold, which OMPL
        rejects. Newton projection with the finite-difference Jacobian pulls
        violating states back to the region boundary — CBiRRT's TSR
        projection semantics."""

        def __init__(self, kin: ArmKinematics, attached: AttachedObject,
                     path_tsrs: list[TSR], tolerance: float = 2e-3):
            assert len(path_tsrs) < kin.n, "need manifold dim > 0"
            super().__init__(kin.n, len(path_tsrs))
            self.kin, self.attached, self.tsrs = kin, attached, path_tsrs
            self.setTolerance(tolerance)

        def function(self, x, out):
            T_body = self.attached.body_pose(self.kin.fk(np.asarray(x)))
            for i, t in enumerate(self.tsrs):
                out[i] = np.linalg.norm(t.excess(T_body))

        def jacobian(self, x, out):
            """Analytic d|excess|/dq, replacing OMPL's finite differences
            (which cost ~250 ms/projection through Python FK). This is the
            CBiRRT Jacobian: MuJoCo site Jacobians mapped into each TSR's
            w frame, with the extrinsic-xyz rpy-rate matrix for the
            rotational rows. Valid away from the rpy singularity
            (|pitch| = pi/2), which the path corridors keep us clear of."""
            kin = self.kin
            q = np.asarray(x)
            T0_ee = kin.fk(q)                       # also sets scratch data
            T_body = self.attached.body_pose(T0_ee)
            nv = kin.model.nv
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(kin.model, kin.data, jacp, jacr, kin.site_id)
            dof = [kin.model.jnt_dofadr[j] for j in kin.arm_joint_ids]
            Jp = jacp[:, dof]                        # (3, n) eef-site linear
            Jr = jacr[:, dof]                        # (3, n) angular (rigid)
            p_site = kin.data.site_xpos[kin.site_id]

            for i, t in enumerate(self.tsrs):
                e = t.excess(T_body)
                norm = np.linalg.norm(e)
                if norm < 1e-12:
                    out[i, :] = 0.0
                    continue
                R_A = t._T0_w_inv[:3, :3]            # world -> w frame
                # constrained feature point: origin of T_body @ Tw_e^-1
                p_feat = (T_body @ t._Tw_e_inv)[:3, 3]
                r = p_feat - p_site
                J_feat = Jp - np.cross(r, Jr, axisa=0, axisb=0).T
                d_trans = R_A @ J_feat               # d(displacement xyz)/dq
                # rpy rates: omega in w frame -> extrinsic-xyz angle rates
                d = t.displacement(T_body)
                rr, pp, yy = d[3], d[4], d[5]
                cx, sx = np.cos(rr), np.sin(rr)
                cy, sy = np.cos(pp), np.sin(pp)
                cz, sz = np.cos(yy), np.sin(yy)
                # omega_w = E @ [rdot, pdot, ydot], E cols = Rz Ry x, Rz y, z
                E = np.array([[cz * cy, -sz, 0.0],
                              [sz * cy,  cz, 0.0],
                              [-sy,     0.0, 1.0]])
                d_rot = np.linalg.pinv(E) @ (R_A @ Jr)
                de = np.vstack([d_trans, d_rot])     # (6, n) displacement jac
                active = (np.abs(e) > 1e-12)
                out[i, :] = (e[active] / norm) @ de[active]


@dataclass
class PlanResult:
    ok: bool
    path: np.ndarray | None = None            # (n_waypoints, n_joints)
    reason: str = ""
    solve_time: float = 0.0
    max_excess: float = np.nan                # over interpolated waypoints
    stats: dict = field(default_factory=dict)


def _state_to_np(state, n: int) -> np.ndarray:
    return np.array([state[i] for i in range(n)])


# ---------------------------------------------------------------- projection


def project_config(kin: ArmKinematics, attached: AttachedObject,
                   tsrs: list[TSR], q: np.ndarray,
                   tol: float = 2e-3, max_iters: int = 30,
                   step_cap: float = 0.25, damping: float = 1e-4):
    """CBiRRT's projection operator: pull config q onto {q : attached object
    pose inside every TSR}. Task-space, not residual-Newton: clamp the pose
    into the TSR (tsr.project), then take damped-least-squares joint steps
    toward the clamped pose using the body-frame Jacobian from mj_jacSite.

    -> (q_projected, ok)."""
    q = np.asarray(q, float).copy()
    lo, hi = kin.joint_range[:, 0], kin.joint_range[:, 1]
    nv = kin.model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    dof = [kin.model.jnt_dofadr[j] for j in kin.arm_joint_ids]
    for _ in range(max_iters):
        T_body = attached.body_pose(kin.fk(q))
        worst = max(tsrs, key=lambda t: t.distance(T_body))
        if worst.distance(T_body) <= tol:
            return q, True
        T_tgt = worst.project(T_body)
        dp = T_tgt[:3, 3] - T_body[:3, 3]
        w = R.from_matrix(T_tgt[:3, :3] @ T_body[:3, :3].T).as_rotvec()
        # body-origin Jacobian from the eef site (rigid attachment)
        mujoco.mj_jacSite(kin.model, kin.data, jacp, jacr, kin.site_id)
        Jp, Jr = jacp[:, dof], jacr[:, dof]
        r_off = T_body[:3, 3] - kin.data.site_xpos[kin.site_id]
        J = np.vstack([Jp - np.cross(r_off, Jr, axisa=0, axisb=0).T, Jr])
        twist = np.concatenate([dp, w])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)
        n = np.linalg.norm(dq)
        if n > step_cap:
            dq *= step_cap / n
        q = np.clip(q + dq, lo, hi)
    T_body = attached.body_pose(kin.fk(q))
    return q, max(t.distance(T_body) for t in tsrs) <= tol


def _constrained_segment(kin, attached, tsrs, qa, qb, eps, tol,
                         check_collision, max_steps=200):
    """CBiRRT ConstrainedExtend: step from qa toward qb in eps-sized joint
    moves, projecting each step onto the manifold, requiring monotone
    progress and (optionally) collision-freedom. Returns the list of
    configs reached (possibly stopping short of qb), starting AFTER qa."""
    out = []
    q = np.asarray(qa, float).copy()
    dist_prev = np.linalg.norm(qb - q)
    for _ in range(max_steps):
        d = qb - q
        n = np.linalg.norm(d)
        if n <= eps:                              # arrival
            q_new, ok = project_config(kin, attached, tsrs, qb, tol=tol)
            if ok and not (check_collision and kin.in_collision(q_new)):
                out.append(q_new)
            return out
        q_step = q + d * (eps / n)
        q_new, ok = project_config(kin, attached, tsrs, q_step, tol=tol)
        if not ok:
            return out
        if check_collision and kin.in_collision(q_new):
            return out
        dist_new = np.linalg.norm(qb - q_new)
        if dist_new >= dist_prev or np.linalg.norm(q_new - q) > 3 * eps:
            return out                            # stuck / projection lurch
        out.append(q_new)
        q, dist_prev = q_new, dist_new
    return out


def plan_constrained(
    kin: ArmKinematics,
    attached: AttachedObject,
    path_tsrs: list[TSR],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    timeout: float = 10.0,
    n_interp: int = 100,
    constraint_tol: float = 2e-3,
    check_collision: bool = True,
    eps: float = 0.12,
    n_shortcut: int = 80,
    rng: np.random.Generator | None = None,
) -> PlanResult:
    """CBiRRT (Berenson et al.): bidirectional RRT whose every extension is
    projected onto the TSR manifold by project_config.

    Implemented directly rather than via OMPL's ProjectedStateSpace: the
    TSR residual is zero (with zero Jacobian) on a full-dimensional region,
    and the pip wheel's Newton-projection machinery produces no tree growth
    for that constraint class (plus discreteGeodesic is unbound, so it
    cannot be diagnosed from Python). The task-space projection here is
    also the faster and more faithful operator.
    """
    import time as _time
    rng = rng or np.random.default_rng()
    t_end = _time.time() + timeout

    def proj(q):
        return project_config(kin, attached, path_tsrs, q, tol=constraint_tol)

    q_start_p, ok_s = proj(q_start)
    q_goal_p, ok_g = proj(q_goal)
    if not (ok_s and ok_g):
        return PlanResult(False, reason="start/goal projection failed")
    for name, q in (("start", q_start_p), ("goal", q_goal_p)):
        if check_collision and kin.in_collision(q):
            return PlanResult(False, reason=f"{name} in collision after projection")

    lo, hi = kin.joint_range[:, 0], kin.joint_range[:, 1]
    lo_s, hi_s = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)
    trees = [{"q": [q_start_p], "parent": [-1]},
             {"q": [q_goal_p], "parent": [-1]}]

    def nearest(tree, q):
        d = np.linalg.norm(np.array(tree["q"]) - q, axis=1)
        return int(np.argmin(d))

    def grow(tree, q_target):
        i = nearest(tree, q_target)
        seg = _constrained_segment(kin, attached, path_tsrs,
                                   tree["q"][i], q_target, eps,
                                   constraint_tol, check_collision)
        for q in seg:
            tree["q"].append(q)
            tree["parent"].append(i)
            i = len(tree["q"]) - 1
        return i, (len(seg) > 0 and
                   np.linalg.norm(tree["q"][i] - q_target) <= eps)

    a, b = 0, 1
    bridge = None
    while _time.time() < t_end:
        q_rand = rng.uniform(lo_s, hi_s)
        ia, _ = grow(trees[a], q_rand)
        ib, reached = grow(trees[b], trees[a]["q"][ia])
        if reached:
            bridge = (a, ia, b, ib)
            break
        a, b = b, a

    if bridge is None:
        return PlanResult(False, reason=f"no path in {timeout}s",
                          solve_time=timeout,
                          stats={"tree_sizes": [len(t["q"]) for t in trees]})

    def backtrack(tree, i):
        out = []
        while i != -1:
            out.append(tree["q"][i])
            i = tree["parent"][i]
        return out

    a, ia, b, ib = bridge
    half_a = backtrack(trees[a], ia)[::-1]
    half_b = backtrack(trees[b], ib)
    path = half_a + half_b
    if a == 1:                                    # tree a rooted at goal
        path = path[::-1]
    path = np.array(path)

    # shortcut smoothing (constrained: shortcuts are projected segments)
    for _ in range(n_shortcut):
        if len(path) < 4 or _time.time() > t_end + 5.0:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        seg = _constrained_segment(kin, attached, path_tsrs,
                                   path[i], path[j], eps, constraint_tol,
                                   check_collision)
        if seg and np.linalg.norm(seg[-1] - path[j]) <= 1e-9 and \
           len(seg) < (j - i):
            path = np.vstack([path[: i + 1], seg[:-1], path[j:]])

    # densify to n_interp and verify the manifold held
    dense = [path[0]]
    for qa, qb in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(qb - qa) / eps)))
        for k in range(1, n + 1):
            dense.append(qa + (qb - qa) * k / n)
    idx = np.linspace(0, len(dense) - 1, min(n_interp, len(dense))).astype(int)
    qs = np.array(dense)[idx]

    max_exc = 0.0
    for q in qs:
        T_body = attached.body_pose(kin.fk(q))
        for t in path_tsrs:
            max_exc = max(max_exc, float(np.max(np.abs(t.excess(T_body)))))

    return PlanResult(ok=True, path=qs, solve_time=timeout - (t_end - _time.time()),
                      max_excess=max_exc,
                      stats={"n_waypoints": len(qs),
                             "tree_sizes": [len(t["q"]) for t in trees]})