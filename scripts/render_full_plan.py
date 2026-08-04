"""Render the combined three-stage plan (outputs/plans/pour_tea_full.npz)
to an mp4 — headless, same plumbing as render_test_plan.py.

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
  blue sphere     mug opening center, lifted to the middle of the
                  stage-2 standoff band (--opening-lift 0 for the raw rim)
  orange sphere   teapot body origin (visible even without meshes)

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py
    MUJOCO_GL=egl    PYTHONPATH=. python scripts/render_full_plan.py --camera agentview
"""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
import robosuite as suite  # noqa: F401  (env built via the scene factory)
from scipy.spatial.transform import Rotation as R

from manip_sim.planning import ArmKinematics, AttachedObject
from manip_sim.viz import DEFAULT_OPENING_LIFT, InteractionMarkers
from scripts.demos.demo_pour_tea import make_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="outputs/plans/pour_tea_full.npz")
    ap.add_argument("--out", default="outputs/videos/pour_tea_full.mp4")
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--interp", type=int, default=3,
                    help="extra frames between waypoints")
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--opening-lift", type=float,
                    default=DEFAULT_OPENING_LIFT,
                    help="raise the blue mug-opening marker this far (m) "
                         "along the mug up_axis; default is the center of "
                         "the stage-2 standoff band, 0.0 draws the raw "
                         "calibrated rim symbol. Drawing offset only.")
    args = ap.parse_args()

    plan = np.load(args.plan)
    path, stage_ids = plan["path"], plan["stage_ids"]
    T_ee_body, T0_teapot_init = plan["T_ee_body"], plan["T0_teapot_init"]
    T0_mug = plan["T0_mug"]

    env, objs = make_env(robot="UR5e", has_renderer=False)
    kin = ArmKinematics(env)
    model, data = kin.model, kin.data
    attached = AttachedObject(T_ee_body)

    markers = InteractionMarkers(opening_lift=args.opening_lift)

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

    frames = []
    for q, stage in zip(dense, dense_stage):
        data.qpos[kin.qpos_ids] = q
        if stage >= 2:
            T_body = attached.body_pose(kin.fk(q))   # fk runs mj_forward
        else:
            kin.fk(q)
            T_body = np.asarray(T0_teapot_init)
        set_teapot(T_body)
        if stage >= 2:
            markers.push_trail(markers.spout_tip(T_body), stage)

        renderer.update_scene(data, camera=cam, scene_option=vis_opt)
        if not args.no_markers:
            markers.draw(renderer.scene, T_body, T0_mug, stage=stage,
                         show_handle=(stage == 1),
                         show_body=(teapot_qadr is None))
        frames.append(renderer.render().copy())

    frames.extend([frames[-1]] * args.fps)      # hold the final pour pose

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8)
    n1 = int((stage_ids == 1).sum())
    n2 = int((stage_ids == 2).sum())
    n3 = int((stage_ids == 3).sum())
    print(f"[render_full_plan] {len(frames)} frames "
          f"(grasp {n1} | transport {n2} | pour {n3} waypoints) -> {out}")
    env.close()


if __name__ == "__main__":
    main()