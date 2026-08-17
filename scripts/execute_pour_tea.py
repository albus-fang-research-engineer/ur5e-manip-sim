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

Both land under the plan's emission-ablation arm (see
manip_sim.provenance), so a hand-authored run and a VLM run never
overwrite each other:

    outputs/videos/<arm>/pour_tea_exec_<arm>.mp4
    outputs/metrics/<arm>/pour_tea_exec_<arm>.json

The arm is also written into the metrics json, alongside the selections
artifact it came from — the run is self-describing even out of context.

Requires converted meshes (a friction grasp needs geometry to rub on);
--allow-meshfree runs the arm trajectory + controller stack without the
objects as a smoke test.

    PYTHONPATH=. python scripts/execute_pour_tea.py
    PYTHONPATH=. python scripts/execute_pour_tea.py --arm vlm
    MUJOCO_GL=egl PYTHONPATH=. python scripts/execute_pour_tea.py --camera agentview

--arm defaults to 'auto' (read the stamp); passing it explicitly is a
CHECK against the stamp, and a mismatch is an error.
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
from manip_sim.provenance import (add_arm_flag, announce, metrics_path,
                                  read_selections, resolve_arm,
                                  resolve_plan_path, video_path)
from manip_sim.replan import (ReplanError, TaskFrames, replan_from_stage,
                              slip_exceeds)
from manip_sim.viz import DebugOverlay

ARM_OUTPUT_MAX = 0.05             # rad/step commanded delta cap (1 rad/s @20Hz)


class VideoTap:
    """on_step hook: render the LIVE sim every `every` control steps."""

    def __init__(self, arm: LiveArm, camera: str, width: int, height: int,
                 every: int, enabled: bool, draw=None):
        self.enabled = enabled
        self.draw = draw                  # scene -> None, called per frame
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
        if self.draw is not None:
            self.draw(self.renderer.scene)
        self.frames.append(self.renderer.render().copy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None,
                    help="plan artifact; default is the one plan present "
                         "under outputs/plans/<arm>/")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--out-video", default=None,
                    help="mp4; default "
                         "outputs/videos/<arm>/pour_tea_exec_<arm>.mp4")
    ap.add_argument("--out-metrics", default=None,
                    help="json; default "
                         "outputs/metrics/<arm>/pour_tea_exec_<arm>.json")
    add_arm_flag(ap)
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-every", type=int, default=2,
                    help="render every Nth control step")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-overlay", action="store_true",
                    help="drop the frame/TSR/slip overlay from the video")
    ap.add_argument("--close-steps", type=int, default=50)
    ap.add_argument("--dwell-seconds", type=float, default=2.0)
    ap.add_argument("--allow-meshfree", action="store_true",
                    help="run the trajectory without objects (controller "
                         "smoke test; no grasp, no slip metrics)")
    ap.add_argument("--no-reanchor", action="store_true",
                    help="disable slip-triggered stage re-anchoring (the "
                         "stale-plan baseline arm of the re-plan ablation)")
    args = ap.parse_args()

    import os, time as _time
    plan_file = resolve_plan_path(args.plan, args.arm)
    plan = np.load(plan_file)
    arm_tag = resolve_arm(args.arm, plan, plan_file)
    announce(arm_tag, plan, plan_file)
    out_video = Path(args.out_video) if args.out_video \
        else video_path("pour_tea_exec", arm_tag)
    out_metrics = Path(args.out_metrics) if args.out_metrics \
        else metrics_path("pour_tea_exec", arm_tag)

    age_min = (_time.time() - os.path.getmtime(plan_file)) / 60.0
    print(f"[execute] plan written {age_min:.0f} min ago")
    if age_min > 60:
        print("[execute] WARNING: this plan is over an hour old — if "
              "planning just failed, you are about to execute a STALE "
              "plan; check the planner output for a SystemExit.")
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
    metrics: dict = {"plan": str(plan_file), "ablation_arm": arm_tag,
                     "selections": read_selections(plan) or "",
                     "meshfree": not have_teapot}

    teapot_sym = load_symbols("assets/objects/teapot")
    mug_sym = load_symbols("assets/objects/mug")
    spout_tip = teapot_sym.frame("spout_tip", "pour_axis")
    tilt_frame = teapot_sym.frame("spout_tip", "tilt_axis",
                                  secondary="pour_axis")
    opening = mug_sym.frame("opening_center", "up_axis")
    handle = teapot_sym.frame("handle_center", "handle_axis")
    rim_radius = mug_sym.quantities.get("rim_radius", 0.044)

    # overlay: a readout of the frames and TSRs in force this step. `live`
    # is mutated as the run advances so the video always shows the CURRENT
    # constraint, including a re-anchored pour pair.
    dbg = DebugOverlay(spout_tip, tilt_frame, opening, handle=handle,
                       rim_radius=rim_radius)
    live: dict = {"tsrs": [], "T_ee_body": None}

    def _draw(scene):
        if not have_teapot:
            return
        T_body = arm.body_pose("teapot")
        T_pred = (arm.ee_pose() @ live["T_ee_body"]
                  if live["T_ee_body"] is not None else None)
        dbg.draw(scene, T_body, T0_mug_plan, T_body_pred=T_pred,
                 tsrs=live["tsrs"])

    tap = VideoTap(arm, args.camera, args.width, args.height,
                   args.frame_every, enabled=not args.no_video,
                   draw=None if args.no_overlay else _draw)

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
        live["T_ee_body"] = T_ee_body
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
            rim_margin=rim_radius * 0.5,
            # match plan_pour_tea.py: the standoff band the goals were
            # actually sampled from, not pour_stages' default
            height=(0.04, 0.10),
            z_corridor=(-0.02, 0.45))
        live["tsrs"] = [(tpair.subgoal, "tsr_transport")]

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

    # ------------- stage boundary 2->3: slip-triggered re-anchor -----------
    # The one interior boundary before a precision stage in this task; the
    # mechanism (replan_from_stage + slip_exceeds) is stage-generic. The
    # STANDING residual — where the pot settled, not the transient max —
    # is compared against what the pour subgoal's own B^w tolerates; the
    # constraints are unchanged (slip is a perception update, not an
    # authoring error), only the stale T_ee_body is re-measured.
    ppair = None
    if have_teapot:
        r2 = dbg.report(arm.body_pose("teapot"), T0_mug_plan)
        print(f"[boundary 2->3] spout tip: lateral "
              f"{r2['tip_lateral_mm']:.1f} mm (rim {r2['rim_radius_mm']:.1f}), "
              f"standoff {r2['tip_standoff_mm']:.1f} mm, "
              f"over rim {r2['tip_over_rim']}")
        print(f"[boundary 2->3] transport subgoal contains the measured "
              f"pose: {tpair.subgoal.contains(arm.body_pose('teapot'), tol=1e-3)}")
        metrics["transport_end"] = r2
    if have_teapot and not args.no_reanchor and slip.history:
        standing = slip.history[-1]
        probe = pour_pair(arm.body_pose("teapot"), tilt_frame,
                          tilt_target=tilt_target)
        fire, tol_pos, tol_rot = slip_exceeds(standing, probe.subgoal)
        print(f"[boundary 2->3] standing slip {standing[0] * 1000:.1f} mm / "
              f"{np.rad2deg(standing[1]):.1f} deg vs pour-subgoal tolerance "
              f"{tol_pos * 1000:.1f} mm / {np.rad2deg(tol_rot):.1f} deg")
        metrics["boundary_standing_slip"] = [standing[0] * 1000,
                                             float(np.rad2deg(standing[1]))]
        metrics["boundary_tolerance"] = [tol_pos * 1000,
                                         float(np.rad2deg(tol_rot))]
        if fire:
            print("[boundary 2->3] slip exceeds the pour subgoal's "
                  "tolerance -> re-anchor: re-measure T_ee_body, re-plan "
                  "stage 3 from the measured entry")
            metrics["transport_slip_final"] = {
                "max_mm": slip.max_dpos * 1000,
                "max_deg": float(np.rad2deg(slip.max_drot))}
            T_ee_body = np.linalg.inv(arm.ee_pose()) @ arm.body_pose("teapot")
            slip = SlipMonitor(T_ee_body)      # re-frozen at the new grip
            live["T_ee_body"] = T_ee_body
            try:
                rr = replan_from_stage(
                    3, env=env, q_now=arm.q(), T_ee_body=T_ee_body,
                    T_teapot_now=arm.body_pose("teapot"),
                    T0_mug=T0_mug_plan,
                    frames=TaskFrames(
                        spout_tip=spout_tip, tilt_frame=tilt_frame,
                        opening=opening,
                        rim_margin=mug_sym.quantities.get(
                            "rim_radius", 0.04) * 0.5),
                    tilt_target=tilt_target,
                    object_joint=env.objects["teapot"].joints[0])
                p3 = rr.path                   # replaces the stale stage 3
                ppair = rr.pairs[3]            # frozen at the SAME entry
                if 2 in rr.pairs:
                    live["tsrs"] = [(rr.pairs[2].subgoal, "tsr_transport")]
                metrics["reanchor"] = {"fired": True, "ok": True}
                print(f"[boundary 2->3] re-planned pour: {len(p3)} waypoints")
            except ReplanError as e:
                metrics["reanchor"] = {"fired": True, "ok": False,
                                       "stage": e.stage, "reason": e.reason}
                print(f"[boundary 2->3] re-plan failed ({e.reason}); "
                      "falling through to the stale stage-3 plan")
        else:
            metrics["reanchor"] = {"fired": False}

    # ========================================================= stage 3: pour
    print("\n================= stage 3: pour ================")
    if have_teapot and ppair is None:
        # frozen at the MEASURED entry — where transport actually ended
        ppair = pour_pair(arm.body_pose("teapot"), tilt_frame,
                          tilt_target=tilt_target)
    if have_teapot:
        live["tsrs"] = live["tsrs"] + [(ppair.subgoal, "tsr_pour")]
    staged_run("pour", p3, ppair.path if ppair is not None else None)
    dwell_steps = int(args.dwell_seconds * 20)
    tracker.hold(p3[-1], GRIPPER_CLOSE, dwell_steps, on_step=tap)

    # ---------------------------------------------------------- final report
    print("\n================= execution report =============")
    if have_teapot:
        T_body = arm.body_pose("teapot")
        tilt_meas = ppair.subgoal.displacement(T_body)[3]
        rep = dbg.report(T_body, T0_mug_plan)
        held = arm.contacts_between("gripper0", "teapot") > 0
        sl = rep["stream_lateral_mm"]
        print(f"  measured tilt      {np.rad2deg(tilt_meas):6.1f} deg "
              f"(target {np.rad2deg(tilt_target):.0f})")
        print(f"  spout declination  {rep['spout_declination_deg']:6.1f} deg "
              "from straight down")
        print(f"  tip lateral        {rep['tip_lateral_mm']:6.1f} mm "
              f"(rim {rep['rim_radius_mm']:.1f} mm) -> over rim "
              f"{rep['tip_over_rim']}")
        print(f"  tip standoff       {rep['tip_standoff_mm']:6.1f} mm "
              "(subgoal band 40-100)")
        print("  stream landing     " +
              (f"{sl:6.1f} mm from center -> in mug "
               f"{rep['stream_lands_in_mug']}" if sl is not None
               else "  n/a  spout is not pointing at the opening plane"))
        print(f"  tip <-> opening    {rep['tip_to_opening_mm']:6.1f} mm "
              "(3-D; includes the intended standoff — not a miss metric)")
        print(f"  slip max           {slip.max_dpos * 1000:6.1f} mm / "
              f"{np.rad2deg(slip.max_drot):5.1f} deg")
        print(f"  grasp retained     {held}")
        metrics.update(tilt_measured_deg=float(np.rad2deg(tilt_meas)),
                       tilt_target_deg=float(np.rad2deg(tilt_target)),
                       slip_max_mm=slip.max_dpos * 1000,
                       slip_max_deg=float(np.rad2deg(slip.max_drot)),
                       grasp_retained=bool(held), **rep)
    else:
        print("  (mesh-free smoke test: trajectory + controller only)")

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"  arm     [{arm_tag}]")
    print(f"  metrics -> {out_metrics}")
    if tap.enabled and tap.frames:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out_video, tap.frames, fps=args.fps, quality=8)
        print(f"  video   -> {out_video} ({len(tap.frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()