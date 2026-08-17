"""Physical execution of planned joint paths — the friction-based layer.

Everything upstream (TSR sampling, IK funnels, CBiRRT) is kinematic and
runs on a SCRATCH MjData; this module is the one place that touches the
LIVE simulation. The handoff contract:

  plan time   the grasp is a kinematic attachment; T_ee_body is assumed
              rigid and collision between the attached object and the
              world is unchecked (the known v1 gap).
  execution   the grasp is Coulomb friction between the Robotiq pads and
              the handle's collision hulls. T_ee_body is MEASURED from
              settled physics after the fingers close (the plan-time
              value is a prediction of this measurement), and nothing
              re-guarantees rigidity afterwards — the SlipMonitor turns
              that non-guarantee into a first-class metric by comparing
              the object's measured pose against the rigid prediction
              every control step.

Controller: the arm part of the BASIC composite config is REPLACED
wholesale with a JOINT_POSITION dict (README pin #4: mutating the part
`type` in place leaves OSC keys that collide with the joint controller's
kwargs in robosuite 1.5.2). Planned paths are joint paths, so joint-space
tracking is the faithful executor — OSC would re-route through task space
and reintroduce exactly the IK ambiguity the planner already resolved.
The nested {"gripper": {"type": "GRIP"}} subdict must survive the
replacement; the gripper controller lives inside the arm part.

Tracking scheme: densify the plan so consecutive targets differ by at
most `max_joint_step` per joint, then per target command the clipped
joint error as a delta action (the controller's own PD does the rest),
advancing on tolerance or a step budget. Simple, stiff, and — unlike an
interpolator-based scheme — the commanded configurations are exactly the
planner's manifold-projected ones, so execution-time TSR excess measures
tracking + slip, not command shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .planning import _raw_model


# ------------------------------------------------------------- controller


def joint_position_arm_part(
    output_max: float = 0.05, kp: float = 150.0
) -> dict:
    """The replacement arm part dict: JOINT_POSITION with delta inputs.
    output_max bounds the per-step commanded delta (rad) — 0.05 at 20 Hz
    caps commanded joint speed at 1 rad/s."""
    return {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": output_max,
        "output_min": -output_max,
        "kp": kp,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "input_type": "delta",
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},          # gripper rides the arm part
    }


def tracking_controller_config(robot: str, **part_kw) -> dict:
    """BASIC composite config with the right-arm part swapped for
    joint-position tracking. Pass to make_env(controller_configs=...)."""
    from robosuite.controllers import load_composite_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    cfg["body_parts"]["right"] = joint_position_arm_part(**part_kw)
    return cfg


# ------------------------------------------------------------ live access


def live_handles(env):
    """(raw MjModel, raw MjData) of the LIVE simulation — not a scratch
    copy. Mutating this data mutates physics; read-only use intended."""
    m = _raw_model(env)
    d = env.sim.data
    return m, getattr(d, "_data", d)


class LiveArm:
    """Arm joint indexing + eef pose on the live simulation (the
    ArmKinematics naming logic, minus the scratch MjData)."""

    def __init__(self, env, eef_site: str = "gripper0_right_grip_site",
                 joint_prefix: str = "robot0_"):
        self.env = env
        self.model, self.data = live_handles(env)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, eef_site)
        assert self.site_id >= 0, f"site '{eef_site}' not in model"
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(self.model.njnt)]
        arm = [j for j, n in enumerate(names)
               if n and n.startswith(joint_prefix)
               and self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                              mujoco.mjtJoint.mjJNT_SLIDE)]
        self.qpos_ids = np.array([self.model.jnt_qposadr[j] for j in arm])
        self.n = len(self.qpos_ids)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def ee_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T

    def body_pose(self, name_prefix: str) -> np.ndarray:
        """Measured world pose of the first body whose name starts with
        name_prefix (robosuite object naming, e.g. 'teapot' -> 'teapot_main')."""
        for b in range(self.model.nbody):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if n and n.startswith(name_prefix):
                T = np.eye(4)
                T[:3, :3] = self.data.xmat[b].reshape(3, 3)
                T[:3, 3] = self.data.xpos[b]
                return T
        raise KeyError(f"no body with prefix '{name_prefix}'")

    def nearest_pad_object_gap(self, obj_prefix: str,
                               pad_keyword: str = "inner_finger",
                               distmax: float = 1.0):
        """Minimum geom-geom distance from the finger pads to the object,
        plus the site-frame offset to the nearest object geom — the
        closed-on-nothing post-mortem: distinguishes 'bar is 30 mm to the
        -x of the pads' (aim/convention error) from 'nearest hull is 40 mm
        away' (hull<->visual divergence) at a glance."""
        pad_bodies = [b for b in range(self.model.nbody)
                      if pad_keyword in (mujoco.mj_id2name(
                          self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
        obj_bodies = [b for b in range(self.model.nbody)
                      if (mujoco.mj_id2name(self.model,
                          mujoco.mjtObj.mjOBJ_BODY, b) or ""
                          ).startswith(obj_prefix)]
        pad_geoms = [g for g in range(self.model.ngeom)
                     if self.model.geom_bodyid[g] in pad_bodies]
        obj_geoms = [g for g in range(self.model.ngeom)
                     if self.model.geom_bodyid[g] in obj_bodies
                     and self.model.geom_contype[g]]
        if not pad_geoms or not obj_geoms:
            return None, None
        best, best_g = distmax, None
        fromto = np.zeros(6)
        for gp in pad_geoms:
            for go in obj_geoms:
                dist = mujoco.mj_geomDistance(
                    self.model, self.data, gp, go, distmax, fromto)
                if dist < best:
                    best, best_g = dist, go
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        off = np.linalg.inv(T) @ np.append(self.data.geom_xpos[best_g], 1.0)
        return best, off[:3]

    def contacts_between(self, prefix_a: str, prefix_b: str,
                         depth_tol: float = 0.0) -> int:
        """Count live contacts between bodies matching the two prefixes
        (order-free). depth_tol > 0 requires actual penetration."""
        n = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist > -depth_tol:
                continue
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[con.geom1]) or ""
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[con.geom2]) or ""
            if (b1.startswith(prefix_a) and b2.startswith(prefix_b)) or \
               (b1.startswith(prefix_b) and b2.startswith(prefix_a)):
                n += 1
        return n


# --------------------------------------------------------------- tracking


GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


@dataclass
class TrackStats:
    n_targets: int = 0
    n_env_steps: int = 0
    max_track_err: float = 0.0        # per-joint, over advance moments
    mean_track_err: float = 0.0
    stalled: bool = False             # aborted: arm could not keep station

    def merge(self, other: "TrackStats") -> None:
        self.stalled = self.stalled or other.stalled
        tot = self.n_targets + other.n_targets
        if tot:
            self.mean_track_err = (
                self.mean_track_err * self.n_targets
                + other.mean_track_err * other.n_targets) / tot
        self.n_targets = tot
        self.n_env_steps += other.n_env_steps
        self.max_track_err = max(self.max_track_err, other.max_track_err)


def densify(path: np.ndarray, max_joint_step: float) -> np.ndarray:
    """Insert waypoints so consecutive targets differ by <= max_joint_step
    on every joint. Endpoints (the planner's projected configs) are kept."""
    out = [path[0]]
    for qa, qb in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / max_joint_step)))
        for k in range(1, n + 1):
            out.append(qa + (qb - qa) * k / n)
    return np.asarray(out)


class Tracker:
    """Delta-action joint tracking through the JOINT_POSITION controller."""

    def __init__(self, env, arm: LiveArm, output_max: float = 0.05):
        self.env, self.arm, self.output_max = env, arm, output_max

    def _step(self, q_target: np.ndarray, gripper: float):
        err = q_target - self.arm.q()
        a = np.zeros(self.env.action_dim)
        a[: self.arm.n] = np.clip(err / self.output_max, -1.0, 1.0)
        a[-1] = gripper
        self.env.step(a)
        return err

    def track(self, path: np.ndarray, gripper: float,
              max_joint_step: float = 0.02, tol: float = 0.015,
              steps_per_target: int = 6, on_step=None,
              stall_err: float = 0.10, stall_patience: int = 40) -> TrackStats:
        """stall_err/stall_patience: the runaway guard. The per-target step
        budget makes advancement OPEN-LOOP: if the arm is physically
        obstructed (attached object jammed on the scene), targets keep
        marching while the arm lags, and the controller then drags the
        object through whatever stopped it. If the residual after the
        budget exceeds stall_err (nominal tracking peaks ~0.012 rad; the
        threshold is ~8x that), the target stream HOLDS — same target,
        stall_patience extra steps — and if the arm still cannot close to
        stall_err the segment aborts with stats.stalled=True. The caller
        owns the policy, matching the typed-failure convention."""
        stats = TrackStats(n_targets=0)
        errs = []
        targets = densify(path, max_joint_step)
        for q_t in targets:
            for _ in range(steps_per_target):
                err = self._step(q_t, gripper)
                stats.n_env_steps += 1
                if on_step:
                    on_step()
                if np.max(np.abs(err)) <= tol:
                    break
            e = float(np.max(np.abs(q_t - self.arm.q())))
            if e > stall_err:
                for _ in range(stall_patience):
                    err = self._step(q_t, gripper)
                    stats.n_env_steps += 1
                    if on_step:
                        on_step()
                    if np.max(np.abs(err)) <= stall_err:
                        break
                e = float(np.max(np.abs(q_t - self.arm.q())))
            errs.append(e)
            stats.n_targets += 1
            if e > stall_err:
                stats.stalled = True
                print(f"[track] STALL: residual {e:.3f} rad after "
                      f"{stall_patience} hold steps at target "
                      f"{stats.n_targets}/{len(targets)} — aborting the "
                      "segment instead of dragging through the obstruction")
                break
        stats.max_track_err = float(np.max(errs)) if errs else 0.0
        stats.mean_track_err = float(np.mean(errs)) if errs else 0.0
        return stats

    def hold(self, q_target: np.ndarray, gripper: float, steps: int,
             on_step=None) -> None:
        for _ in range(steps):
            self._step(q_target, gripper)
            if on_step:
                on_step()

    def close_gripper(self, q_hold: np.ndarray, obj_prefix: str,
                      steps: int = 40, settle_steps: int = 15,
                      on_step=None) -> int:
        """Hold the arm at q_hold and close on the object; keep squeezing
        for settle_steps after first contact so the pads seat. Returns the
        final finger<->object contact count (0 = the grasp missed)."""
        first_contact = None
        for k in range(steps):
            self._step(q_hold, GRIPPER_CLOSE)
            if on_step:
                on_step()
            n = self.arm.contacts_between("gripper0", obj_prefix)
            if n and first_contact is None:
                first_contact = k
            if first_contact is not None and k - first_contact >= settle_steps:
                break
        return self.arm.contacts_between("gripper0", obj_prefix)


# ------------------------------------------------------------ slip metric


@dataclass
class SlipMonitor:
    """Rigid-attachment residual: how far the object's MEASURED pose has
    drifted from the pose the frozen T_ee_body predicts from the measured
    eef pose. Zero for a perfect friction grasp; growth localizes slip in
    time (the per-stage maxima matter more than the average)."""

    T_ee_body: np.ndarray
    max_dpos: float = 0.0
    max_drot: float = 0.0
    history: list = field(default_factory=list)   # (dpos, drot) per update

    def update(self, T_ee_meas: np.ndarray, T_body_meas: np.ndarray):
        T_pred = T_ee_meas @ self.T_ee_body
        dpos = float(np.linalg.norm(T_body_meas[:3, 3] - T_pred[:3, 3]))
        drot = float(np.linalg.norm(R.from_matrix(
            T_pred[:3, :3].T @ T_body_meas[:3, :3]).as_rotvec()))
        self.max_dpos = max(self.max_dpos, dpos)
        self.max_drot = max(self.max_drot, drot)
        self.history.append((dpos, drot))
        return dpos, drot