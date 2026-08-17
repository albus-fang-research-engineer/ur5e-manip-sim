"""Render the combined three-stage plan to an mp4 — headless, same
plumbing as render_test_plan.py.

The plan carries an emission-ablation arm stamp (see manip_sim.provenance)
saying whether its task frames were hand-authored or VLM-selected; the
video is filed under that arm and never overwrites the other one:

    outputs/videos/hand/pour_tea_full_hand.mp4
    outputs/videos/vlm/pour_tea_full_vlm.mp4

Stage-aware playback:
  stage 1 (grasp)      the teapot stays at its settled table pose; the arm
                       reaches, descends the approach, and "grasps"
  stages 2-3           the teapot rides the gripper kinematically via the
                       measured T_ee_body frozen at grasp completion

Overlays (toggle with --no-markers):
  green sphere    handle center (stage-1 target), while ungrasped
  red sphere      spout tip (follows the teapot)
  dot trail       spout-tip trace, colored by stage
                  (grey = grasp, red = transport, purple = pour)
  blue sphere     mug opening center (the transport subgoal target)
  orange sphere   teapot body origin (visible even without meshes)

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py --arm vlm
    MUJOCO_GL=egl    PYTHONPATH=. python scripts/render_full_plan.py --camera agentview

--arm defaults to 'auto': it reads the stamp, and only asks when both
arms have a plan on disk. Passing --arm on a stamped plan is a CHECK, not
an override — a mismatch is an error rather than a mislabeled video.
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
from manip_sim.provenance import (add_arm_flag, announce, resolve_arm,
                                  resolve_plan_path, video_path)
from scripts.demos.demo_pour_tea import make_env

RGBA = {
    "tip": (0.9, 0.15, 0.15, 1.0),
    "handle": (0.15, 0.75, 0.25, 0.95),
    "opening": (0.2, 0.4, 0.95, 0.9),
    "body": (0.95, 0.6, 0.1, 0.9),
    "trace1": (0.55, 0.55, 0.55, 0.5),
    "trace2": (0.9, 0.15, 0.15, 0.55),
    "trace3": (0.6, 0.2, 0.85, 0.6),
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
    ap.add_argument("--plan", default=None,
                    help="plan artifact; default is the one plan present "
                         "under outputs/plans/<arm>/")
    ap.add_argument("--out", default=None,
                    help="mp4; default "
                         "outputs/videos/<arm>/pour_tea_full_<arm>.mp4")
    add_arm_flag(ap)
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--interp", type=int, default=3,
                    help="extra frames between waypoints")
    ap.add_argument("--no-markers", action="store_true")
    args = ap.parse_args()

    plan_file = resolve_plan_path(args.plan, args.arm)
    plan = np.load(plan_file)
    arm_tag = resolve_arm(args.arm, plan, plan_file)
    announce(arm_tag, plan, plan_file)
    out = Path(args.out) if args.out else video_path("pour_tea_full", arm_tag)

    path, stage_ids = plan["path"], plan["stage_ids"]
    T_ee_body, T0_teapot_init = plan["T_ee_body"], plan["T0_teapot_init"]
    T0_mug = plan["T0_mug"]

    env, objs = make_env(robot="UR5e", has_renderer=False)
    kin = ArmKinematics(env)
    model, data = kin.model, kin.data
    attached = AttachedObject(T_ee_body)

    teapot_sym = load_symbols("assets/objects/teapot")
    spout = teapot_sym.frame("spout_tip", "pour_axis")
    handle_w = (T0_teapot_init @
                teapot_sym.frame("handle_center", "handle_axis").T())[:3, 3]
    opening_w = (T0_mug @ load_symbols("assets/objects/mug")
                 .frame("opening_center", "up_axis").T())[:3, 3]

    teapot_qadr = None
    if "teapot" in objs:
        jname = env.objects["teapot"].joints[0]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        teapot_qadr = model.jnt_qposadr[jid]

    def set_teapot(T_body):
        if teapot_qadr is None:
            return
        quat = R.from_matrix(T_body[:3, :3]).as_quat(scalar_first=True)
        data.qpos[teapot_qadr: teapot_qadr + 7] = \
            np.concatenate([T_body[:3, 3], quat])
        mujoco.mj_forward(model, data)

    # hide robosuite's group-0 collision hulls, as its own viewer does
    vis_opt = mujoco.MjvOption()
    vis_opt.geomgroup[0] = 0

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

    # densify per waypoint, carrying each segment's stage id
    dense, dense_stage = [path[0]], [stage_ids[0]]
    for (qa, sa), (qb, sb) in zip(zip(path[:-1], stage_ids[:-1]),
                                  zip(path[1:], stage_ids[1:])):
        for k in range(1, args.interp + 1):
            dense.append(qa + (qb - qa) * k / args.interp)
            dense_stage.append(sb)

    frames, trace = [], []          # trace entries: (pos, stage)
    for q, stage in zip(dense, dense_stage):
        data.qpos[kin.qpos_ids] = q
        if stage >= 2:
            T_body = attached.body_pose(kin.fk(q))   # fk runs mj_forward
        else:
            kin.fk(q)
            T_body = np.asarray(T0_teapot_init)
        set_teapot(T_body)
        tip = (T_body @ spout.T())[:3, 3]
        if stage >= 2:
            trace.append((tip.copy(), stage))

        renderer.update_scene(data, camera=cam, scene_option=vis_opt)
        if not args.no_markers:
            scene = renderer.scene
            for p, s in trace[:-1][::2]:
                add_sphere(scene, p, 0.006, RGBA[f"trace{s}"])
            add_sphere(scene, tip, 0.014, RGBA["tip"])
            add_sphere(scene, opening_w, 0.018, RGBA["opening"])
            if stage == 1:
                add_sphere(scene, handle_w, 0.015, RGBA["handle"])
            if teapot_qadr is None:
                add_sphere(scene, T_body[:3, 3], 0.03, RGBA["body"])
        frames.append(renderer.render().copy())

    frames.extend([frames[-1]] * args.fps)      # hold the final pour pose

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8)
    n1 = int((stage_ids == 1).sum())
    n2 = int((stage_ids == 2).sum())
    n3 = int((stage_ids == 3).sum())
    print(f"[render_full_plan] [{arm_tag}] {len(frames)} frames "
          f"(grasp {n1} | transport {n2} | pour {n3} waypoints) -> {out}")
    env.close()


if __name__ == "__main__":
    main()