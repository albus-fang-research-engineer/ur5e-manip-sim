"""Capture a real RGB-D "frame packet" from the TableTop sim for perception tests.

The point: test the perception sidecars (SAM3 / FoundationPose / TRELLIS.2 /
cuRobo) with EXACTLY the interface they see on hardware — an RGB-D frame plus
intrinsics/extrinsics — but with simulation ground truth riding along so the
tests can score the outputs. No ROS2 anywhere: the packet is a plain .npz you
copy next to the ur5e-manip-hardware test/ dir, and the tests talk msgpack-ZMQ
straight to the sidecars, i.e. the same wire protocol the ROS bridge nodes use.

Everything in the packet is in the CV convention a RealSense would produce:

  rgb            HxWx3 uint8, row 0 = top of image
  depth          HxW float32 METERS, 0 = invalid (far plane / clipped), aligned
                 to rgb (same camera, same K — sim luxury)
  K              3x3 intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
  T_world_cam    4x4 camera-to-world; CV camera axes (+z fwd, +y down).
                 robosuite's get_camera_extrinsic_matrix already bakes the
                 MuJoCo->CV axis correction (see manip_sim.state note).

Ground truth riding along:

  mask_robot / mask_table / mask_<obj>   HxW bool, occlusion-aware (rendered
                                         element segmentation, grouped by body)
  world_T_<obj> / cam_T_<obj>            4x4 pose of mjcf body '<obj>_object' —
                                         the SAME frame frames.json and the
                                         exported visual mesh live in, so
                                         FoundationPose(cam_T_mesh) is directly
                                         comparable to cam_T_<obj>
  T_world_base                           robot base body pose
  qpos_arm / joint_names_arm             6 arm joint angles + names with the
                                         "robot0_" prefix stripped (matches
                                         ur_description / cuRobo names)
  meta (json)                            camera, sizes, table geometry, objects

Self-validation: each object's GT position is projected through K/T_world_cam
and must land inside that object's (dilated) mask — this catches any raster
flip / intrinsics / extrinsics inconsistency at capture time instead of as a
mystery test failure two repos away. Debug PNG overlays are written next to
the npz for eyeballing.

Usage (sim container or bare venv, headless):

    MUJOCO_GL=egl PYTHONPATH=. python scripts/capture_rgbd_packet.py
    PYTHONPATH=. python scripts/capture_rgbd_packet.py \
        --camera agentview --height 480 --width 640 \
        --objects teapot mug --out outputs/frame_packets/agentview

Then copy the whole output dir to ur5e-manip-hardware/test/data/packet/ (or
point FRAME_PACKET at packet.npz).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

DEPTH_MAX = 3.0  # meters; beyond this -> 0 (invalid), mirrors RealSense range
                 # and the curobo server's depth_maximum_distance
SETTLE_STEPS = 60


# --------------------------------------------------------------------- helpers
def body_group(model, body_id) -> str:
    """Classify a body (by walking its ancestor chain) into
    'robot' / 'table' / '<object name prefix>' / 'other'."""
    def _name(b):
        try:
            return model.body_id2name(b) or ""
        except AttributeError:
            return model.body_names[b] or ""

    bid = body_id
    names = []
    while bid > 0:
        names.append(_name(bid))
        bid = model.body_parentid[bid]
    for n in names:
        if n.startswith(("robot0", "gripper0", "mount0")):
            return "robot"
    for n in names:
        if "table" in n:
            return "table"
    return names[-1] if names else "other"  # topmost non-world ancestor name


def masks_from_segmentation(env, seg, object_names):
    """seg: HxWx2 (objtype, objid) raw render. Returns dict of HxW bool masks
    keyed 'robot', 'table', and each object name."""
    model = env.sim.model
    geom_ids = seg[..., 1].astype(np.int64)
    valid = geom_ids >= 0
    ngeom = model.ngeom

    # geom id -> group label, computed once
    group_of = {}
    for gid in np.unique(geom_ids[valid]):
        if gid >= ngeom:
            group_of[gid] = "other"
            continue
        group_of[gid] = body_group(model, model.geom_bodyid[gid])

    masks = {k: np.zeros(geom_ids.shape, bool) for k in ["robot", "table"] + list(object_names)}
    for gid, grp in group_of.items():
        key = None
        if grp == "robot":
            key = "robot"
        elif grp == "table":
            key = "table"
        else:
            for name in object_names:
                if grp.startswith(name):
                    key = name
                    break
        if key is not None:
            masks[key] |= geom_ids == gid
    return masks


def body_pose_matrix(env, body_name):
    from scipy.spatial.transform import Rotation as R

    bid = env.sim.model.body_name2id(body_name)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(env.sim.data.body_xquat[bid], scalar_first=True).as_matrix()
    T[:3, 3] = env.sim.data.body_xpos[bid]
    return T


def resolve_object_body(env, name):
    """Prefer '<name>_object' (the frame frames.json + meshes are defined in);
    fall back to the first body with the object's prefix."""
    model = env.sim.model
    target = f"{name}_object"
    if target in model.body_names:
        return target
    matches = [b for b in model.body_names if b and b.startswith(name)]
    if not matches:
        raise KeyError(f"no body for object '{name}'; bodies: {model.body_names}")
    return matches[0]


def project(K, T_world_cam, p_world):
    """world point -> (u, v) pixel via CV camera. Returns None if behind cam."""
    T_cw = np.linalg.inv(T_world_cam)
    p = T_cw[:3, :3] @ p_world + T_cw[:3, 3]
    if p[2] <= 0:
        return None
    return np.array([K[0, 0] * p[0] / p[2] + K[0, 2],
                     K[1, 1] * p[1] / p[2] + K[1, 2]])


def dilate(mask, it=6):
    m = mask.copy()
    for _ in range(it):
        m[1:] |= m[:-1]; m[:-1] |= m[1:]
        m[:, 1:] |= m[:, :-1]; m[:, :-1] |= m[:, 1:]
    return m


def save_overlay(path, rgb, mask, color=(255, 0, 0)):
    from PIL import Image

    img = rgb.copy()
    edge = mask & ~dilate(mask, 0)  # cheap: just tint the whole mask
    img[mask] = (0.5 * img[mask] + 0.5 * np.array(color)).astype(np.uint8)
    del edge
    Image.fromarray(img).save(path)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--objects", nargs="+", default=["teapot", "mug"])
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/frame_packets/packet")
    args = ap.parse_args()

    import robosuite as suite
    import robosuite.utils.camera_utils as CU
    from robosuite.environments.base import register_env

    from manip_sim.envs.tabletop import TableTop

    try:
        register_env(TableTop)
    except AssertionError:
        pass

    np.random.seed(args.seed)
    object_xmls = {n: f"assets/objects/{n}/{n}.xml" for n in args.objects}
    make_kwargs = dict(
        env_name="TableTop",
        robots=args.robot,
        object_xmls=object_xmls,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        camera_depths=True,
        control_freq=20,
        ignore_done=True,
    )
    try:
        env = suite.make(seed=args.seed, **make_kwargs)
    except TypeError:  # robosuite build without a seed kwarg
        env = suite.make(**make_kwargs)

    obs = env.reset()
    zeros = np.zeros(env.action_spec[0].shape)
    for _ in range(SETTLE_STEPS):  # let objects come to rest on the table
        obs, *_ = env.step(zeros)

    cam, h, w = args.camera, args.height, args.width

    # ---- rasters (robosuite obs arrive vertically flipped w.r.t. K -> [::-1])
    rgb = np.asarray(obs[f"{cam}_image"], np.uint8)[::-1].copy()
    depth_norm = np.asarray(obs[f"{cam}_depth"], np.float32)
    depth_norm = np.clip(np.nan_to_num(depth_norm, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    depth = CU.get_real_depth_map(env.sim, depth_norm)[::-1].squeeze().astype(np.float32).copy()
    depth[depth > DEPTH_MAX] = 0.0

    try:
        seg = CU.get_camera_segmentation(sim=env.sim, camera_name=cam,
                                         camera_height=h, camera_width=w)
    except TypeError:  # older/newer camera_utils signature
        seg = env.sim.render(camera_name=cam, height=h, width=w,
                             segmentation=True)
    seg = np.asarray(seg)[::-1]

    K = CU.get_camera_intrinsic_matrix(env.sim, cam, h, w).astype(np.float64)
    T_world_cam = CU.get_camera_extrinsic_matrix(env.sim, cam).astype(np.float64)

    masks = masks_from_segmentation(env, seg, args.objects)

    # ---- GT poses
    world_T = {n: body_pose_matrix(env, resolve_object_body(env, n)) for n in args.objects}
    T_cam_world = np.linalg.inv(T_world_cam)
    cam_T = {n: T_cam_world @ world_T[n] for n in args.objects}
    T_world_base = body_pose_matrix(env, env.robots[0].robot_model.root_body)

    # ---- projection self-check: GT centroid must land in the (dilated) mask.
    # If it only works with the seg flipped the other way, flip the masks: the
    # rgb/depth path is anchored by this same check via the object masks.
    def check(mks):
        ok = True
        for n in args.objects:
            uv = project(K, T_world_cam, world_T[n][:3, 3])
            if uv is None:
                return False
            u, v = int(round(uv[0])), int(round(uv[1]))
            if not (0 <= v < h and 0 <= u < w and dilate(mks[n])[v, u]):
                ok = False
        return ok

    if not check(masks):
        flipped = {k: m[::-1].copy() for k, m in masks.items()}
        if check(flipped):
            print("[capture] NOTE: segmentation flip convention differed; auto-corrected.")
            masks = flipped
        else:
            raise RuntimeError(
                "GT projection does not land inside object masks under either "
                "flip — raster/K/extrinsics conventions are inconsistent; do "
                "not use this packet.")

    # ---- arm state (names stripped to ur_description convention for cuRobo)
    model = env.sim.model
    ref = env.robots[0]._ref_joint_pos_indexes
    raw_names = [model.joint_names[jid] for jid in env.robots[0]._ref_joint_indexes]
    qpos_arm = env.sim.data.qpos[ref].copy()
    joint_names_arm = [n.replace("robot0_", "") for n in raw_names]

    # ---- write packet
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        rgb=rgb, depth=depth, K=K, T_world_cam=T_world_cam,
        T_world_base=T_world_base, qpos_arm=qpos_arm,
        mask_robot=masks["robot"], mask_table=masks["table"],
    )
    for n in args.objects:
        arrays[f"mask_{n}"] = masks[n]
        arrays[f"world_T_{n}"] = world_T[n]
        arrays[f"cam_T_{n}"] = cam_T[n]

    meta = dict(
        camera=cam, height=h, width=w, robot=args.robot, seed=args.seed,
        objects=list(args.objects),
        joint_names_arm=joint_names_arm,
        joint_names_arm_raw=raw_names,
        depth_units="meters", depth_invalid=0.0, depth_max=DEPTH_MAX,
        table_offset=list(map(float, env.table_offset)),
        table_full_size=list(map(float, env.table_full_size)),
        convention="CV: row0=top, +z fwd, +y down; T_world_cam is cam-to-world",
    )
    np.savez_compressed(out / "packet.npz", meta=json.dumps(meta), **arrays)

    # ---- companion assets: visual meshes + frames.json (FoundationPose /
    # TRELLIS metric / sphere-fit tests). Meshes are in body-'object' frame.
    for n in args.objects:
        src = Path(f"assets/objects/{n}/meshes/{n}_visual.obj")
        if src.exists():
            shutil.copy(src, out / f"{n}.obj")
        else:
            print(f"[capture] WARNING: {src} missing (gitignored?) — "
                  f"FoundationPose/mesh tests for '{n}' will skip.")
        fj = Path(f"assets/objects/{n}/frames.json")
        if fj.exists():
            shutil.copy(fj, out / f"{n}_frames.json")

    # ---- debug overlays
    try:
        save_overlay(out / "overlay_robot.png", rgb, masks["robot"], (255, 60, 60))
        for i, n in enumerate(args.objects):
            save_overlay(out / f"overlay_{n}.png", rgb, masks[n], (60, 255, 60))
        from PIL import Image
        d = depth.copy()
        d[d == 0] = np.nan
        dn = (255 * (np.nan_to_num(d, nan=DEPTH_MAX) / DEPTH_MAX)).astype(np.uint8)
        Image.fromarray(dn).save(out / "depth.png")
    except Exception as e:  # PIL optional
        print(f"[capture] overlay skipped: {e}")

    cov = {k: float(m.mean()) for k, m in masks.items()}
    print(f"[capture] wrote {out/'packet.npz'}  ({w}x{h}, {cam})")
    print(f"[capture] mask coverage: {cov}")
    print(f"[capture] arm q ({', '.join(joint_names_arm)}) = {np.round(qpos_arm, 3)}")
    env.close()


if __name__ == "__main__":
    main()
