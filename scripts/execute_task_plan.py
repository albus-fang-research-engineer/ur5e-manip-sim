"""Physically execute a plan_task.py plan in MuJoCo, or play it back
kinematically. Same execution layer as execute_pour_tea.py (joint-position
tracking controller, friction grasp, no teleports, no qpos pinning), minus
everything pour-specific (re-anchoring, the pour pair, the spout report).

  physics (default)
    1  track the reach + approach with the gripper OPEN, hold, CLOSE on the
       grasped object (Coulomb friction between pads and collision hulls);
       T_ee_body is MEASURED at closure and frozen
    2  track every later segment closed (lift, then each planned stage),
       with the slip monitor on the measured object pose; a stalled segment
       (arm obstructed) aborts instead of dragging the object through
    3  --release (default): open the gripper and let the object settle
    report: per-segment tracking, slip, grasp contacts, and where the object
       ended relative to the final stage's passive point: long-axis tilt from
       the passive's up, the lowest point's depth below the plane through the
       point and its distance from the axis through it
  --kinematic
       no physics: set the arm to each waypoint and carry the object rigidly
       with the plan's T_ee_body after the grasp (the plan as planned)

    MUJOCO_GL=egl PYTHONPATH=. python scripts/execute_task_plan.py \\
        --scene scenes/insert_marker.json --grounding outputs/grounding/insert_marker_gt_opus1 \\
        --plan outputs/plans/vlm/insert_marker_opusplan_opus.npz [--kinematic]

Writes outputs/videos/vlm/<plan stem>_{exec,kin}.mp4 and, for physics,
outputs/metrics/vlm/<plan stem>_exec.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from manip_sim.execution import (GRIPPER_CLOSE, GRIPPER_OPEN, LiveArm,
                                 SlipMonitor, Tracker,
                                 tracking_controller_config)
from manip_sim.frames import load_symbols
from manip_sim.planning import ArmKinematics, AttachedObject
from manip_sim.proposal import load_obj
from manip_sim.scene import add_scene_arg, load_scene, make_env
from manip_sim.task_plan import lowest_end, transform_points

ARM_OUTPUT_MAX = 0.05             # as execute_pour_tea: rad/step delta cap


class Camera:
    """Offscreen renderer on a model/data pair (collision hulls hidden)."""

    def __init__(self, model, data, camera, width, height):
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.data = data
        self.r = mujoco.Renderer(model, height, width)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.0, 0.0, 0.9]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.4, 90.0, -25.0
        else:
            self.cam = camera
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0
        self.frames: list[np.ndarray] = []

    def shot(self):
        self.r.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        self.frames.append(self.r.render().copy())


def end_state(T_obj, V_obj, T_pas, target_body, up_body) -> dict:
    """Where the object ended relative to the passive's target point (the
    final stage's passive point): long-axis tilt from the passive's up, and
    the lowest point's depth below the target plane and distance from the
    target axis (task_plan.lowest_end). Plain numbers; what counts as
    success is a separate decision."""
    up = T_pas[:3, :3] @ np.asarray(up_body, float)
    up /= np.linalg.norm(up)
    Vw = transform_points(T_obj, V_obj)
    axis = np.linalg.svd(Vw - Vw.mean(0), full_matrices=False)[2][0]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(axis @ up)))))
    c = transform_points(T_pas, np.asarray(target_body, float)[None])[0]
    low = lowest_end(T_obj, V_obj, {}, up_world=up, target=(c, up))
    return {"long_axis_tilt_from_passive_up_deg": round(tilt, 1),
            "lowest_point_below_target_plane_m": low["depth_below_target_m"],
            "lowest_point_from_target_axis_m": low["radial_from_target_axis_m"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--kinematic", action="store_true", help="playback, no physics")
    ap.add_argument("--passive", default=None,
                    help="object the end state is measured against; default: the "
                         "plan's final-stage passive, else the scene's only other object")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-every", type=int, default=2, help="physics: render every Nth step")
    ap.add_argument("--interp", type=int, default=3, help="kinematic: frames between waypoints")
    ap.add_argument("--close-steps", type=int, default=50)
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene, getattr(args, "grounding", None))
    plan_file = Path(args.plan)
    plan = np.load(plan_file)
    path, ids, labels = plan["path"], plan["stage_ids"], [str(s) for s in plan["stage_labels"]]
    obj = str(plan["grasped"])
    segs = [(labels[i], path[ids == i]) for i in range(len(labels))]
    others = [n for n in scene.objects if n != obj]
    passive = args.passive or (str(plan["target_object"]) if "target_object" in plan.files
                               else others[0] if len(others) == 1 else None)
    meshes = {n: load_obj(Path(scene.asset_dirs[n]) / "meshes" / f"{n}_visual.obj")[0]
              for n in [obj] + ([passive] if passive else [])}
    mode = "kin" if args.kinematic else "exec"
    out_video = Path("outputs/videos/vlm") / f"{plan_file.stem}_{mode}.mp4"
    print(f"[execute] {plan_file} ({mode}): segments "
          + ", ".join(f"{l} {len(s)}" for l, s in segs))

    if args.kinematic:
        env, objs = make_env(scene, robot=args.robot, has_renderer=False)
        kin = ArmKinematics(env)
        cam = Camera(kin.model, kin.data, args.camera, args.width, args.height)
        attached = AttachedObject(plan["T_ee_body"])
        jid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, env.objects[obj].joints[0])
        qadr = kin.model.jnt_qposadr[jid]
        T_init = plan[f"T0_{obj}"]
        for k, (label, seg) in enumerate(segs):
            carried = k > 0                      # segment 0 = reach + approach
            for qa, qb in zip(seg[:-1], seg[1:]):
                for t in range(1, args.interp + 1):
                    q = qa + (qb - qa) * t / args.interp
                    T_ee = kin.fk(q)
                    T_b = attached.body_pose(T_ee) if carried else T_init
                    kin.data.qpos[qadr: qadr + 7] = np.r_[
                        T_b[:3, 3], R.from_matrix(T_b[:3, :3]).as_quat(scalar_first=True)]
                    mujoco.mj_forward(kin.model, kin.data)
                    cam.shot()
        cam.frames += [cam.frames[-1]] * args.fps
        out_video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out_video, cam.frames, fps=args.fps, quality=8)
        print(f"[execute] {len(cam.frames)} frames -> {out_video}")
        env.close()
        return

    cfg = tracking_controller_config(args.robot, output_max=ARM_OUTPUT_MAX)
    env, objs = make_env(scene, robot=args.robot, has_renderer=False, controller_configs=cfg)
    arm = LiveArm(env)
    tracker = Tracker(env, arm, output_max=ARM_OUTPUT_MAX)
    cam = Camera(arm.model, arm.data, args.camera, args.width, args.height)
    tick = {"k": 0}

    def tap():
        tick["k"] += 1
        if tick["k"] % args.frame_every == 0:
            cam.shot()

    metrics: dict = {"plan": str(plan_file), "grasped": obj, "passive": passive,
                     "segments": {}}
    reach_label, reach = segs[0]
    st = tracker.track(reach, GRIPPER_OPEN, on_step=tap)
    metrics["segments"][reach_label] = vars(st)
    n_c = tracker.close_gripper(reach[-1], obj, steps=args.close_steps, on_step=tap)
    metrics["grasp_contacts"] = n_c
    print(f"[grasp] reach tracked (max err {st.max_track_err:.4f} rad); closed: "
          f"{n_c} finger<->{obj} contacts")
    if n_c == 0:
        metrics["failure"] = "grasp closed on nothing"
    else:
        T_ee_body = np.linalg.inv(arm.ee_pose()) @ arm.body_pose(obj)
        d = float(np.linalg.norm(T_ee_body[:3, 3] - plan["T_ee_body"][:3, 3]))
        metrics["T_ee_body_delta_mm"] = round(d * 1000, 1)
        print(f"[grasp] measured T_ee_body vs planned: {d * 1000:.1f} mm")
        slip = SlipMonitor(T_ee_body)
        for label, seg in segs[1:]:
            def on_step():
                tap()
                slip.update(arm.ee_pose(), arm.body_pose(obj))
            st = tracker.track(seg, GRIPPER_CLOSE, on_step=on_step)
            metrics["segments"][label] = {**vars(st), "slip_max_mm": round(slip.max_dpos * 1000, 1),
                                          "slip_max_deg": round(float(np.degrees(slip.max_drot)), 1)}
            print(f"[{label}] tracked {st.n_targets} targets, max err {st.max_track_err:.4f} rad, "
                  f"slip so far {slip.max_dpos * 1000:.1f} mm / {np.degrees(slip.max_drot):.1f} deg"
                  + ("  STALLED" if st.stalled else ""))
            if st.stalled:
                metrics["failure"] = f"{label}: stalled (arm obstructed)"
                break
        if passive:
            if "target_point_body" in plan.files:
                tgt, up = plan["target_point_body"], plan["target_up_body"]
            else:                        # older plan files: top-centre of the passive
                up = load_symbols(scene.asset_dirs[passive]).axes["up_axis"]
                tgt = up * float(np.max(meshes[passive] @ up))
            metrics["target_point_body"] = np.round(tgt, 4).tolist()
            metrics["held_end"] = end_state(arm.body_pose(obj), meshes[obj],
                                            arm.body_pose(passive), tgt, up)
            metrics["held_contacts"] = arm.contacts_between("gripper0", obj)
            print(f"[end, held] {metrics['held_end']}")
            if not args.no_release and "failure" not in metrics:
                tracker.hold(arm.q(), GRIPPER_OPEN, int(args.settle_seconds * 20), on_step=tap)
                metrics["released_end"] = end_state(arm.body_pose(obj), meshes[obj],
                                                    arm.body_pose(passive), tgt, up)
                print(f"[end, released] {metrics['released_end']}")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_video, cam.frames, fps=args.fps, quality=8)
    out_m = Path("outputs/metrics/vlm") / f"{plan_file.stem}_exec.json"
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(f"[execute] {len(cam.frames)} frames -> {out_video}; metrics -> {out_m}")
    env.close()


if __name__ == "__main__":
    main()
