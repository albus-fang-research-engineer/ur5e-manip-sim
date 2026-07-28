"""Render a saved plan (outputs/plans/*.npz) to an mp4 -- no display needed.

Works headless via software or EGL rendering, so it sidesteps X11/GLX
entirely. On a machine with a GPU, MUJOCO_GL=egl is much faster:

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_plan.py
    MUJOCO_GL=egl    PYTHONPATH=. python scripts/render_plan.py --camera agentview

Overlays (toggle with --no-markers):
    red sphere     spout tip (follows the attached teapot)
    red dot trail  spout-tip trace over the whole path so far
    blue sphere    mug opening center (the transport subgoal target)
    orange sphere  teapot body origin (visible even without meshes)

The teapot rides the gripper kinematically via the saved T_ee_body; if the
teapot asset's meshes are converted, the real teapot is teleported too.
"""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
import robosuite as suite  # noqa: F401  (env built via the scene factory)
from scipy.spatial.transform import Rotation as R

from manip_sim.frames import load_symbols
from manip_sim.planning import ArmKinematics, AttachedObject
from manip_sim.tsr import make_pose
from scripts.demos.demo_pour_tea import MUG_XY, make_env

RGBA = {
    "tip": (0.9, 0.15, 0.15, 1.0),
    "trace": (0.9, 0.15, 0.15, 0.55),
    "opening": (0.2, 0.4, 0.95, 0.9),
    "body": (0.95, 0.6, 0.1, 0.9),
}


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0], float),
        np.asarray(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32))
    scene.ngeom += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="outputs/plans/transport_path.npz")
    ap.add_argument("--out", default="outputs/videos/transport.mp4")
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--interp", type=int, default=3,
                    help="extra frames between waypoints")
    ap.add_argument("--no-markers", action="store_true")
    args = ap.parse_args()

    plan = np.load(args.plan)
    path, T_ee_body = plan["path"], plan["T_ee_body"]

    env, objs = make_env(robot="UR5e", has_renderer=False)
    kin = ArmKinematics(env)          # raw model handle + arm qpos ids
    model, data = kin.model, kin.data
    attached = AttachedObject(T_ee_body)

    spout = load_symbols("assets/objects/teapot").frame("spout_tip", "pour_axis")
    if "mug" in objs:
        from manip_sim.state import PoseReader
        from manip_sim.tsr import pose_from_pos_quat_wxyz
        T0_mug = pose_from_pos_quat_wxyz(*PoseReader(env, ["mug"]).pose("mug"))
    else:
        T0_mug = make_pose([*MUG_XY, 0.86])
    opening_w = (T0_mug @ load_symbols("assets/objects/mug")
                 .frame("opening_center", "up_axis").T())[:3, 3]

    teapot_qadr = None
    if "teapot" in objs:
        jname = env.objects["teapot"].joints[0]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        teapot_qadr = model.jnt_qposadr[jid]

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id < 0:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.1, 0.9]
        cam.distance, cam.azimuth, cam.elevation = 2.2, 160.0, -18.0
    else:
        cam = args.camera

    # densify for smooth video
    dense = [path[0]]
    for qa, qb in zip(path[:-1], path[1:]):
        for k in range(1, args.interp + 1):
            dense.append(qa + (qb - qa) * k / args.interp)

    frames, trace = [], []
    for q in dense:
        data.qpos[kin.qpos_ids] = q
        T_body = attached.body_pose(kin.fk(q))     # fk() runs mj_forward
        if teapot_qadr is not None:
            quat = R.from_matrix(T_body[:3, :3]).as_quat(scalar_first=True)
            data.qpos[teapot_qadr: teapot_qadr + 7] = \
                np.concatenate([T_body[:3, 3], quat])
            mujoco.mj_forward(model, data)
        tip = (T_body @ spout.T())[:3, 3]
        trace.append(tip.copy())

        renderer.update_scene(data, camera=cam)
        if not args.no_markers:
            scene = renderer.scene
            for p in trace[:-1][::2]:
                add_sphere(scene, p, 0.006, RGBA["trace"])
            add_sphere(scene, tip, 0.014, RGBA["tip"])
            add_sphere(scene, opening_w, 0.018, RGBA["opening"])
            if teapot_qadr is None:
                add_sphere(scene, T_body[:3, 3], 0.03, RGBA["body"])
        frames.append(renderer.render().copy())

    # hold the last frame for a second
    frames.extend([frames[-1]] * args.fps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8)
    print(f"[render_plan] {len(frames)} frames -> {out}")
    env.close()


if __name__ == "__main__":
    main()