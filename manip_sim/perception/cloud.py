"""Depth-camera point clouds from the robosuite sim.

Bridges the sim to the AnyGrasp sidecar: renders RGB-D from a named MuJoCo
camera, deprojects to a metric cloud in the CV camera frame (z forward,
y down — robosuite's get_camera_extrinsic_matrix already applies the
MuJoCo->CV axis correction, so T_world_cam here composes directly with
poses returned in the cloud frame), and returns the world-frame copy for
masking/sanity checks.

Conventions worth pinning down once:
  * sim.render() output is OpenGL bottom-up; both rgb and depth are
    flipped vertically here so pixel (0,0) is top-left, matching the
    intrinsics from camera_utils (principal point at the image center,
    y down). Skipping the flip warps the deprojection whenever the
    principal point isn't dead center of the scene.
  * MuJoCo depth is normalized [0,1]; get_real_depth_map converts to
    meters using the model's znear/zfar.
  * The offscreen render context is created lazily, so the scene factory
    (make_env with has_offscreen_renderer=False) needs no changes.
"""

from __future__ import annotations

import numpy as np
from robosuite.utils import camera_utils as CU


def _ensure_offscreen(sim, width: int, height: int) -> None:
    if sim._render_context_offscreen is None:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen
        sim.add_render_context(
            MjRenderContextOffscreen(sim, device_id=-1,
                                     max_width=width, max_height=height))


def render_cloud(
    env,
    camera: str = "agentview",
    width: int = 640,
    height: int = 480,
    workspace: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Render one RGB-D frame and deproject.

    workspace  optional (lo[3], hi[3]) WORLD-frame AABB crop; pass a box
               around the object plus a slab of table so AnyGrasp's
               collision check sees the support surface.

    Returns (pts_cam, pts_world, colors, T_world_cam):
        pts_cam     (N,3) float32, CV camera frame — feed this to AnyGrasp
        pts_world   (N,3) float32, same points in world (row-aligned)
        colors      (N,3) float32 in [0,1]
        T_world_cam 4x4 camera pose, CV convention
    """
    sim = env.sim
    _ensure_offscreen(sim, width, height)
    rgb, depth = sim.render(width=width, height=height,
                            camera_name=camera, depth=True)
    rgb = np.flipud(rgb).copy()
    depth = np.flipud(depth).copy()

    z = CU.get_real_depth_map(sim, depth)
    if z.ndim == 3:
        z = z[..., 0]
    K = CU.get_camera_intrinsic_matrix(sim, camera, height, width)
    T_world_cam = CU.get_camera_extrinsic_matrix(sim, camera)

    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    valid = np.isfinite(pts_cam[:, 2]) & (pts_cam[:, 2] > 1e-4)
    pts_cam, colors = pts_cam[valid], colors[valid]
    pts_world = (T_world_cam[:3, :3] @ pts_cam.T).T + T_world_cam[:3, 3]

    if workspace is not None:
        lo = np.asarray(workspace[0], dtype=float)
        hi = np.asarray(workspace[1], dtype=float)
        m = np.all((pts_world >= lo) & (pts_world <= hi), axis=1)
        pts_cam, pts_world, colors = pts_cam[m], pts_world[m], colors[m]

    return (pts_cam.astype(np.float32), pts_world.astype(np.float32),
            colors, T_world_cam)


def object_workspace(
    obj_pos: np.ndarray,
    table_z: float,
    xy_margin: float = 0.25,
    z_top: float = 0.35,
    table_slab: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """WORLD AABB around an object: xy box centered on it, z from just
    below the table top (keep the surface for collision reasoning) to
    z_top above it."""
    lo = np.array([obj_pos[0] - xy_margin, obj_pos[1] - xy_margin,
                   table_z - table_slab])
    hi = np.array([obj_pos[0] + xy_margin, obj_pos[1] + xy_margin,
                   table_z + z_top])
    return lo, hi