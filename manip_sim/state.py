"""Real-time state access for manip-sim.

Three layers, matching the sim-study design:

  PoseReader       ground-truth SE(3) poses straight from mjData — the
                   critical-path pose source in simulation. Zero cost,
                   exact, available every physics step.

  NoisyPoseSensor  first-class perception-ablation arm: wraps a PoseReader
                   and returns what a real estimator would — Gaussian
                   translation/rotation noise, latency (pose lag), dropout
                   (occlusion-driven tracking loss with hold-last-pose).
                   Sim ground truth hides exactly the failure modes the
                   architecture targets; this layer puts them back in,
                   parameterized.

  camera geometry  intrinsics/extrinsics/depth->point-cloud helpers for the
                   eventual estimated-pose arm (GenPose++ / FoundationPose
                   consume RGB-D or point clouds).

All quaternions here are wxyz (MuJoCo convention) unless noted.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import robosuite.utils.camera_utils as CU
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- gt
class PoseReader:
    """Ground-truth SE(3) poses of named bodies, read live from mjData.

    Works with any robosuite env. Body resolution accepts either an exact
    mjcf body name or a prefix (robosuite prefixes object bodies, e.g.
    "mug_main").
    """

    def __init__(self, env, names: list[str]):
        self.env = env
        self.body_ids = {n: self._resolve(n) for n in names}

    def _resolve(self, name: str) -> int:
        model = self.env.sim.model
        try:
            return model.body_name2id(name)
        except Exception:
            matches = [bn for bn in model.body_names if bn and bn.startswith(name)]
            if not matches:
                raise KeyError(f"No body matching '{name}'. Bodies: {model.body_names}")
            return model.body_name2id(matches[0])

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """-> (pos[3], quat_wxyz[4]) in world frame, current physics step."""
        bid = self.body_ids[name]
        d = self.env.sim.data
        return d.body_xpos[bid].copy(), d.body_xquat[bid].copy()

    def pose_matrix(self, name: str) -> np.ndarray:
        """-> 4x4 homogeneous world-frame pose."""
        pos, quat = self.pose(name)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
        T[:3, 3] = pos
        return T

    def velocity(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """-> (linear[3], angular[3]) world-frame velocities."""
        bid = self.body_ids[name]
        d = self.env.sim.data
        # cvel is [ang, lin] in body-centered world-aligned frame
        v = d.cvel[bid]
        return v[3:].copy(), v[:3].copy()


# ------------------------------------------------------------------ noisy layer
@dataclass
class NoiseConfig:
    trans_std: float = 0.005          # m
    rot_std_deg: float = 2.0          # deg, axis-angle
    latency_steps: int = 2            # pose is this many control steps old
    dropout_prob: float = 0.05        # per-query tracking-loss probability
    dropout_min_steps: int = 3        # once lost, stay lost at least this long
    seed: int | None = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)


class NoisyPoseSensor:
    """What a pose estimator returns, not what the simulator knows.

    Call .step() once per control step (after env.step), then .pose(name).
    Returns (pos, quat_wxyz, valid). When tracking is lost (dropout), valid
    is False and the last successfully "estimated" pose is returned frozen —
    the hold-last-pose behavior that produces the in-hand drift and stale
    -target failures the hybrid-frame design is meant to survive.
    """

    def __init__(self, reader: PoseReader, cfg: NoiseConfig | None = None):
        self.reader = reader
        self.cfg = cfg or NoiseConfig()
        n = self.cfg.latency_steps + 1
        self._buffers = {name: deque(maxlen=n) for name in reader.body_ids}
        self._lost_until = {name: 0 for name in reader.body_ids}
        self._held = {}
        self._t = 0

    def step(self):
        self._t += 1
        for name in self.reader.body_ids:
            self._buffers[name].append(self.reader.pose(name))

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray, bool]:
        cfg = self.cfg
        buf = self._buffers[name]
        if not buf:
            self.step()
            buf = self._buffers[name]

        # dropout state machine
        if self._t >= self._lost_until[name] and cfg.rng.random() < cfg.dropout_prob:
            self._lost_until[name] = self._t + cfg.dropout_min_steps
        if self._t < self._lost_until[name] and name in self._held:
            pos, quat = self._held[name]
            return pos.copy(), quat.copy(), False

        # latency: oldest pose in the buffer
        pos, quat = buf[0]
        # noise
        pos = pos + cfg.rng.normal(0.0, cfg.trans_std, 3)
        axis = cfg.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(cfg.rng.normal(0.0, cfg.rot_std_deg))
        dR = R.from_rotvec(axis * angle)
        quat = (dR * R.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True)

        self._held[name] = (pos, quat)
        return pos.copy(), quat.copy(), True


# ------------------------------------------------------------- camera geometry
def camera_intrinsics(env, camera: str, h: int, w: int) -> np.ndarray:
    return CU.get_camera_intrinsic_matrix(env.sim, camera, h, w)


def camera_extrinsics(env, camera: str) -> np.ndarray:
    """4x4 camera-to-world transform."""
    return CU.get_camera_extrinsic_matrix(env.sim, camera)


def depth_to_pointcloud(env, camera: str, depth_norm: np.ndarray, world: bool = True):
    """robosuite depth obs (normalized, flipped) -> Nx3 point cloud.

    Note robosuite camera images/depths arrive vertically flipped relative
    to intrinsics; this handles the flip and the metric conversion.
    """
    h, w = depth_norm.shape[:2]
    # depth = CU.get_real_depth_map(env.sim, depth_norm)[::-1].squeeze()
    depth_norm = np.nan_to_num(
        np.asarray(depth_norm, dtype=np.float32), nan=1.0, posinf=1.0, neginf=0.0
    )
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth = CU.get_real_depth_map(env.sim, depth_norm)[::-1].squeeze()
    K = camera_intrinsics(env, camera, h, w)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - K[0, 2]) * depth / K[0, 0]
    y = (vs - K[1, 2]) * depth / K[1, 1]
    pts_cam = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
    if not world:
        return pts_cam
    # NB: robosuite's get_camera_extrinsic_matrix already bakes in the
    # MuJoCo(-z fwd, +y up) -> CV(+z fwd, +y down) axis correction, so the
    # CV-convention points above transform directly. Do NOT flip y/z again.
    T = camera_extrinsics(env, camera)
    return (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
