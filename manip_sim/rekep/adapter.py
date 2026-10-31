"""What ReKep's solvers need from a simulator, answered from MuJoCo.

Upstream gets these from OmniGibson and Lula. The solvers themselves are
unchanged (vendor.py), so every difference from the paper's setup is either
here or in a program:

  frame       upstream reads the gripper's approach as +x (Fetch); the grip
              site's is +z. The solvers see a frame with +x = approach, and the
              conversion happens only here (T_SITE_RK).
  IK          ReKepIK returns Lula's result fields. It stops at convergence,
              because the objective's reachability cost is the iteration count
              and it runs on every evaluation (MinkIK is about 25x slower per
              call and always runs a fixed count).
  SDF         scene_sdf from the collision hulls: per hull, the largest face-
              plane distance; over hulls, the union. Overlapping hulls stay
              correct. Outside distances are a lower bound near corners, which
              errs toward caution in a soft cost.
  collision   collision_points: gripper, wrist and held object, as upstream.
              Other arm links are not in ReKep's cost (see the planner's report).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as R

# rekep x = site z (approach), rekep y = site y, rekep z = -site x (closing)
T_SITE_RK = np.eye(4)
T_SITE_RK[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
T_RK_SITE = np.linalg.inv(T_SITE_RK)

ROBOT_PREFIXES = ("robot0", "gripper0", "fixed_mount0", "mount0")
COLLISION_POINT_PREFIXES = ("gripper0", "robot0_wrist")


def site_to_rekep(T_site: np.ndarray) -> np.ndarray:
    """4x4 grip site pose -> upstream's [x, y, z, qx, qy, qz, qw] EE pose."""
    T = T_site @ T_SITE_RK
    return np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()]


def rekep_to_site(pose: np.ndarray) -> np.ndarray:
    """Upstream's 7-vector EE pose -> 4x4 grip site pose."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T @ T_RK_SITE


def farthest_point_sampling(pc: np.ndarray, num_points: int) -> np.ndarray:
    """Greedy farthest-point downsampling, starting from the first point."""
    selected = [0]
    distances = np.linalg.norm(pc - pc[0], axis=1)
    for _ in range(1, min(num_points, len(pc))):
        selected.append(int(np.argmax(distances)))
        distances = np.minimum(distances, np.linalg.norm(pc - pc[selected[-1]], axis=1))
    return pc[selected]


# ---------------------------------------------------------------------- IK


@dataclass
class IKResult:
    """The fields of Lula's CyclicCoordDescentIkResult the solvers read."""

    success: bool
    position_error: float
    num_descents: int
    cspace_position: np.ndarray


class ReKepIK:
    """Damped least squares on MuJoCo Jacobians, on its own scratch MjData."""

    def __init__(self, kin, reset_q: np.ndarray, damping: float = 1e-3,
                 max_step: float = 0.4):
        self.kin, self.model = kin, kin.model
        self.data = mujoco.MjData(kin.model)
        self.data.qpos[:] = kin.data.qpos
        self.reset_q = np.asarray(reset_q, float)
        self.dof = np.array([kin.model.jnt_dofadr[j] for j in kin.arm_joint_ids])
        self.damping, self.max_step = damping, max_step

    def solve(self, target_pose_homo, position_tolerance=0.01,
              orientation_tolerance=0.05, max_iterations=150,
              initial_joint_pos=None, **_) -> IKResult:
        """Upstream's call: the target is in the frame the solvers see."""
        return self.solve_site(target_pose_homo @ T_RK_SITE, position_tolerance,
                               orientation_tolerance, max_iterations, initial_joint_pos)

    def solve_site(self, T_site, position_tolerance=0.01, orientation_tolerance=0.05,
                   max_iterations=150, initial_joint_pos=None) -> IKResult:
        k, m, d = self.kin, self.model, self.data
        q = self.reset_q.copy() if initial_joint_pos is None \
            else np.array(initial_joint_pos, float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        reg = self.damping * np.eye(6)
        for i in range(max_iterations + 1):
            d.qpos[k.qpos_ids] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            Rm = d.site_xmat[k.site_id].reshape(3, 3)
            ep = T_site[:3, 3] - d.site_xpos[k.site_id]
            er = Rm @ R.from_matrix(Rm.T @ T_site[:3, :3]).as_rotvec()
            pos_err = float(np.linalg.norm(ep))
            if pos_err <= position_tolerance and np.max(np.abs(er)) <= orientation_tolerance:
                return IKResult(True, pos_err, i, q)
            if i == max_iterations:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, k.site_id)
            J = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])
            dq = J.T @ np.linalg.solve(J @ J.T + reg, np.r_[ep, er])
            q = np.clip(q + np.clip(dq, -self.max_step, self.max_step),
                        k.joint_range[:, 0], k.joint_range[:, 1])
        return IKResult(False, pos_err, max_iterations, q)


# ---------------------------------------------------------------- geometry


def _body_name(model, geom: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ""


def _collides(model, geom: int) -> bool:
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def body_pose(kin, prefix: str) -> np.ndarray:
    """World pose of the first body whose name starts with prefix."""
    for b in range(kin.model.nbody):
        name = mujoco.mj_id2name(kin.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith(prefix):
            T = np.eye(4)
            T[:3, :3] = kin.data.xmat[b].reshape(3, 3)
            T[:3, 3] = kin.data.xpos[b]
            return T
    raise KeyError(f"no body with prefix '{prefix}'")


def geom_vertices(model, data, geom: int) -> np.ndarray:
    """World-frame vertices of a mesh geom or corners of a box geom."""
    Rm = data.geom_xmat[geom].reshape(3, 3)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[adr: adr + num].astype(float)
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        local = signs * model.geom_size[geom]
    else:
        raise ValueError(f"geom {geom} on '{_body_name(model, geom)}': only mesh and "
                         f"box geoms are supported, got type {model.geom_type[geom]}")
    return data.geom_xpos[geom] + local @ Rm.T


def collision_vertices(kin, prefixes) -> np.ndarray:
    """World-frame collision vertices of every body matching prefixes."""
    m, d = kin.model, kin.data
    verts = [geom_vertices(m, d, g) for g in range(m.ngeom)
             if _collides(m, g) and _body_name(m, g).startswith(tuple(prefixes))]
    if not verts:
        raise KeyError(f"no collision geoms under {prefixes}")
    return np.vstack(verts)


def hull_planes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) with A x <= b inside the convex hull of vertices; A unit normals."""
    try:
        eq = ConvexHull(vertices).equations
    except QhullError:                       # nearly flat hull
        eq = ConvexHull(vertices, qhull_options="QJ").equations
    return eq[:, :3], -eq[:, 3]


def hull_distance(points: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest face-plane distance: negative inside the hull, positive outside."""
    return np.max(points @ A.T - b, axis=1)


def sdf_shape(bounds_min, bounds_max, voxel: float) -> tuple[int, int, int]:
    return tuple(np.ceil((np.asarray(bounds_max) - bounds_min) / voxel).astype(int))


def scene_sdf(kin, bounds_min, bounds_max, voxel: float = 0.01,
              exclude=(), margin: float = 0.25) -> np.ndarray:
    """Upstream's SDF grid: positive inside obstacles, on the linspace grid the
    solvers interpolate over. Bodies under ROBOT_PREFIXES and exclude (the held
    object) are left out. Cells farther than margin from a hull read -margin;
    the solvers' cost thresholds are 0.10 and 0.20, so those never contribute."""
    m, d = kin.model, kin.data
    skip = tuple(ROBOT_PREFIXES) + tuple(exclude)
    axes = [np.linspace(lo, hi, n) for lo, hi, n in
            zip(bounds_min, bounds_max, sdf_shape(bounds_min, bounds_max, voxel))]
    sdf = np.full([len(a) for a in axes], -float(margin))
    for g in range(m.ngeom):
        if not _collides(m, g) or _body_name(m, g).startswith(skip) \
                or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        verts = geom_vertices(m, d, g)
        lo, hi = verts.min(0) - margin, verts.max(0) + margin
        ranges = [np.arange(np.searchsorted(a, l), np.searchsorted(a, h))
                  for a, l, h in zip(axes, lo, hi)]
        if not all(len(r) for r in ranges):
            continue
        A, b = hull_planes(verts)
        ix = np.ix_(*ranges)
        pts = np.stack(np.meshgrid(*[a[r] for a, r in zip(axes, ranges)], indexing="ij"),
                       axis=-1).reshape(-1, 3)
        sdf[ix] = np.maximum(sdf[ix], -hull_distance(pts, A, b).reshape(sdf[ix].shape))
    return sdf
