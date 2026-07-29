"""Physically execute the saved three-stage plan — friction grasp, no
teleports, no qpos pinning. The planning artifact is the input; this
script is the execution layer:

  stage 1  track the reach + approach with the gripper OPEN, then hold
           the grasp config and CLOSE — the grasp is Coulomb friction
           between the Robotiq pads and the handle's collision hulls.
           T_ee_body is MEASURED from settled physics at closure (the
           planned value was a prediction of this measurement) and
           frozen; the plan-vs-measured delta is printed.
  stage 2  track the transport path closed. Every step, the SlipMonitor
           compares the teapot's measured pose against the rigid
           prediction, and the transport path TSR is evaluated on the
           MEASURED body pose — execution-time excess = tracking error +
           slip, the number the plan-time manifold guarantee does not
           cover.
  stage 3  track the pour to the tilt target and dwell. The pour pair is
           rebuilt FROZEN AT THE MEASURED ENTRY (same convention as
           planning: relative to where the motion actually is, not where
           it was aimed). Slip here is the experiment, not a failure:
           gravity gets a lever arm on the grasp as the pot tilts.

Final report: per-stage tracking error, contact counts, slip maxima,
measured tilt vs target, spout-tip-to-opening distance, per-stage max
measured TSR excess; metrics also saved as json next to the video.

Requires converted meshes (a friction grasp needs geometry to rub on);
--allow-meshfree runs the arm trajectory + controller stack without the
objects as a smoke test.

    PYTHONPATH=. python scripts/execute_pour_tea.py
    MUJOCO_GL=egl PYTHONPATH=. python scripts/execute_pour_tea.py --camera agentview
"""

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
import robosuite as suite  # noqa: F401  (env built via the scene factory)

from scripts.demos.demo_pour_tea import make_env
from manip_sim.execution import (
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
    LiveArm,
    SlipMonitor,
    TrackStats,
    Tracker,
    tracking_controller_config,
)
from manip_sim.frames import load_symbols
from manip_sim.pour_stages import pour_pair, transport_pair

ARM_OUTPUT_MAX = 0.05             # rad/step commanded delta cap (1 rad/s @20Hz)


class VideoTap:
    """on_step hook: render the LIVE sim every `every` control steps."""

    def __init__(self, arm: LiveArm, camera: str, width: int, height: int,
                 every: int, enabled: bool):
        self.enabled = enabled
        self.frames: list[np.ndarray] = []
        if not enabled:
            return
        self.arm, self.every, self.k = arm, every, 0
        m = arm.model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(m, height, width)
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.0, 0.1, 0.9]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = \
                2.2, 160.0, -18.0
        else:
            self.cam = camera
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0             # hide collision hulls

    def __call__(self):
        if not self.enabled:
            return
        self.k += 1
        if self.k % self.every:
            return
        self.renderer.update_scene(self.arm.data, camera=self.cam,
                                   scene_option=self.opt)
        self.frames.append(self.renderer.render().copy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="outputs/plans/pour_tea_full.npz")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--out-video", default="outputs/videos/pour_tea_exec.mp4")
    ap.add_argument("--out-metrics", default="outputs/metrics/pour_tea_exec.json")
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-every", type=int, default=2,
                    help="render every Nth control step")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--close-steps", type=int, default=50)
    ap.add_argument("--dwell-seconds", type=float, default=2.0)
    ap.add_argument("--allow-meshfree", action="store_true",
                    help="run the trajectory without objects (controller "
                         "smoke test; no grasp, no slip metrics)")
    args = ap.parse_args()

    import os, time as _time
    age_min = (_time.time() - os.path.getmtime(args.plan)) / 60.0
    print(f"[execute] plan artifact: {args.plan} "
          f"(written {age_min:.0f} min ago)")
    if age_min > 60:
        print("[execute] WARNING: this plan is over an hour old — if "
              "planning just failed, you are about to execute a STALE "
              "plan; check the planner output for a SystemExit.")
    plan = np.load(args.plan)
    path, stage_ids = plan["path"], plan["stage_ids"]
    T_ee_body_plan = plan["T_ee_body"]
    T0_teapot_plan, T0_mug_plan = plan["T0_teapot_init"], plan["T0_mug"]
    tilt_target = float(plan["tilt_target"])
    p1 = path[stage_ids == 1]
    p2 = path[stage_ids == 2]
    p3 = path[stage_ids == 3]

    # scene with the joint-position tracking controller injected through
    # the single factory (controller choice does not move the scene)
    cfg = tracking_controller_config(args.robot, output_max=ARM_OUTPUT_MAX)
    env, objs = make_env(robot=args.robot, has_renderer=False,
                         controller_configs=cfg)
    have_teapot = "teapot" in objs
    if not have_teapot and not args.allow_meshfree:
        raise SystemExit("[execute] friction grasping needs the converted "
                         "teapot meshes; rerun scripts/convert_asset.py, or "
                         "pass --allow-meshfree for a no-object smoke test.")

    arm = LiveArm(env)
    tracker = Tracker(env, arm, output_max=ARM_OUTPUT_MAX)
    tap = VideoTap(arm, args.camera, args.width, args.height,
                   args.frame_every, enabled=not args.no_video)
    metrics: dict = {"plan": str(args.plan), "meshfree": not have_teapot}

    teapot_sym = load_symbols("assets/objects/teapot")
    mug_sym = load_symbols("assets/objects/mug")
    spout_tip = teapot_sym.frame("spout_tip", "pour_axis")
    tilt_frame = teapot_sym.frame("spout_tip", "tilt_axis",
                                  secondary="pour_axis")
    opening = mug_sym.frame("opening_center", "up_axis")

    if have_teapot:
        T0_teapot = arm.body_pose("teapot")
        drift = float(np.linalg.norm(T0_teapot[:3, 3] - T0_teapot_plan[:3, 3]))
        print(f"[execute] settled teapot vs plan: {drift * 1000:.1f} mm")
        if drift > 0.02:
            print("[execute] WARNING: scene drifted >2 cm from plan time — "
                  "the grasp config aims at the planned pose; expect a "
                  "misaligned close. Replan for the fresh scene if it misses.")

    # ================================================= stage 1: reach + grasp
    print("\n================ stage 1: grasp ================")
    s1 = tracker.track(p1, GRIPPER_OPEN, on_step=tap)
    print(f"[grasp] tracked {s1.n_targets} targets in {s1.n_env_steps} steps; "
          f"joint err max {s1.max_track_err:.4f} mean {s1.mean_track_err:.4f} rad")
    metrics["stage1_track"] = vars(s1)

    n_contacts = tracker.close_gripper(p1[-1], "teapot",
                                       steps=args.close_steps, on_step=tap) \
        if have_teapot else 0
    if have_teapot:
        print(f"[grasp] closed: {n_contacts} finger<->teapot contacts")
        if n_contacts == 0:
            gap, off = arm.nearest_pad_object_gap("teapot")
            if gap is not None:
                print(f"[grasp] post-mortem: nearest pad<->teapot hull gap "
                      f"{gap * 1000:.1f} mm; nearest hull sits at "
                      f"{np.round(off * 1000, 1)} mm in the grip-site frame "
                      "(x = closing axis, z = approach). Large |x| means "
                      "the bar missed the pad plane; large gap everywhere "
                      "means the collision hulls diverge from the visual "
                      "mesh the symbols were calibrated on.")
            raise SystemExit("[grasp] the fingers closed on nothing — check "
                             "handle symbol calibration and hull geometry "
                             "around the handle bar.")
        # grasp completion: MEASURE the in-hand transform from live physics
        T_ee_body = np.linalg.inv(arm.ee_pose()) @ arm.body_pose("teapot")
        dpos = float(np.linalg.norm(
            T_ee_body[:3, 3] - T_ee_body_plan[:3, 3]))
        print(f"[grasp] measured T_ee_body vs planned: {dpos * 1000:.1f} mm "
              f"translation delta (measured value frozen for stages 2-3)")
        slip = SlipMonitor(T_ee_body)
        metrics["grasp_contacts"] = n_contacts
        metrics["T_ee_body_delta_mm"] = dpos * 1000
    else:
        tracker.hold(p1[-1], GRIPPER_CLOSE, args.close_steps, on_step=tap)
        slip = None

    # execution-time constraint frames, built from MEASURED state
    if have_teapot:
        tpair = transport_pair(
            T0_mug_body=T0_mug_plan, mug_opening=opening, spout_tip=spout_tip,
            teapot_body_pos_now=arm.body_pose("teapot")[:3, 3],
            rim_margin=mug_sym.quantities.get("rim_radius", 0.04) * 0.5,
            z_corridor=(-0.02, 0.45))

    def staged_run(label, p, path_tsr):
        stats = TrackStats()
        max_excess = 0.0

        def on_step():
            nonlocal max_excess
            tap()
            if slip is not None:
                T_body = arm.body_pose("teapot")
                slip.update(arm.ee_pose(), T_body)
                if path_tsr is not None:
                    max_excess = max(max_excess, float(
                        np.max(np.abs(path_tsr.excess(T_body)))))

        stats.merge(tracker.track(p, GRIPPER_CLOSE, on_step=on_step))
        print(f"[{label}] tracked {stats.n_targets} targets in "
              f"{stats.n_env_steps} steps; joint err max "
              f"{stats.max_track_err:.4f} rad" +
              (f"; measured path-TSR excess max {max_excess:.4f}"
               if slip is not None and path_tsr is not None else ""))
        if slip is not None:
            print(f"[{label}] slip so far: {slip.max_dpos * 1000:.1f} mm / "
                  f"{np.rad2deg(slip.max_drot):.1f} deg")
        metrics[f"{label}_track"] = vars(stats)
        metrics[f"{label}_measured_excess"] = max_excess

    # ==================================================== stage 2: transport
    print("\n============== stage 2: transport ==============")
    staged_run("transport", p2, tpair.path if have_teapot else None)

    # ========================================================= stage 3: pour
    print("\n================= stage 3: pour ================")
    ppair = None
    if have_teapot:
        # frozen at the MEASURED entry — where transport actually ended
        ppair = pour_pair(arm.body_pose("teapot"), tilt_frame,
                          tilt_target=tilt_target)
    staged_run("pour", p3, ppair.path if ppair is not None else None)
    dwell_steps = int(args.dwell_seconds * 20)
    tracker.hold(p3[-1], GRIPPER_CLOSE, dwell_steps, on_step=tap)

    # ---------------------------------------------------------- final report
    print("\n================= execution report =============")
    if have_teapot:
        T_body = arm.body_pose("teapot")
        tilt_meas = ppair.subgoal.displacement(T_body)[3]
        tip = (T_body @ spout_tip.T())[:3, 3]
        opening_w = (T0_mug_plan @ opening.T())[:3, 3]
        tip_err = float(np.linalg.norm(tip - opening_w))
        held = arm.contacts_between("gripper0", "teapot") > 0
        print(f"  measured tilt      {np.rad2deg(tilt_meas):6.1f} deg "
              f"(target {np.rad2deg(tilt_target):.0f})")
        print(f"  spout tip <-> opening  {tip_err * 1000:6.1f} mm")
        print(f"  slip max           {slip.max_dpos * 1000:6.1f} mm / "
              f"{np.rad2deg(slip.max_drot):5.1f} deg")
        print(f"  grasp retained     {held}")
        metrics.update(tilt_measured_deg=float(np.rad2deg(tilt_meas)),
                       tilt_target_deg=float(np.rad2deg(tilt_target)),
                       tip_to_opening_mm=tip_err * 1000,
                       slip_max_mm=slip.max_dpos * 1000,
                       slip_max_deg=float(np.rad2deg(slip.max_drot)),
                       grasp_retained=bool(held))
    else:
        print("  (mesh-free smoke test: trajectory + controller only)")

    mpath = Path(args.out_metrics)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"  metrics -> {mpath}")
    if tap.enabled and tap.frames:
        vpath = Path(args.out_video)
        vpath.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(vpath, tap.frames, fps=args.fps, quality=8)
        print(f"  video   -> {vpath} ({len(tap.frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()