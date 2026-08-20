"""Kinematic TSR playback for the pour scene -- validate frames and TSRs
against the real meshes BEFORE any robot, IK, or planner is involved.

No robot motion. The teapot is teleported (qpos pinned each step) through:

  phase A  a handful of samples from the STAGE-2 intersection
           (transport subgoal INTERSECT upright path) -- the spout tip
           should visibly hover over the mug opening, teapot upright,
           at varied approach azimuths;
  phase B  a sweep of the STAGE-3 pour path TSR, roll 0 -> tilt_target --
           the teapot should rotate about its SPOUT TIP (tip visually
           pinned over the mug) with the spout dipping toward the opening.

What to look for (this run IS the calibration check):
  * a wrong spout_tip point  -> the pivot floats off the spout in phase B;
  * a flipped tilt_axis sign -> phase B tips the spout UP/away: negate the
    axis in assets/objects/teapot/frames.json;
  * a wrong opening_center   -> phase A hovers off the rim.
Calibrate by double-click-selecting geometry in this same viewer, editing
frames.json, and re-running until statuses can be flipped to "calibrated".

Run INSIDE the container (same X11 setup as view_pour_tea.py):

  PYTHONPATH=. python scripts/play_tsr_pour.py
  PYTHONPATH=. python scripts/play_tsr_pour.py --n-goals 8 --hold 60
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

import manip_sim  # noqa: F401
from manip_sim.frames import load_symbols
from manip_sim.pour_stages import pour_pair, transport_pair
from manip_sim.tsr import displacement_to_pose, pose_from_pos_quat_wxyz, sample_intersection
from manip_sim.scene import add_scene_arg, load_scene, make_env


def pose_to_qpos(T: np.ndarray) -> np.ndarray:
    quat = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)  # wxyz
    return np.concatenate([T[:3, 3], quat])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=os.environ.get("ROBOT", "UR5e"))
    ap.add_argument("--n-goals", type=int, default=6,
                    help="stage-2 intersection samples to display")
    ap.add_argument("--hold", type=int, default=50,
                    help="render steps to hold each sampled goal")
    ap.add_argument("--tilt-deg", type=float, default=95.0)
    ap.add_argument("--sweep-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene)

    if not os.environ.get("DISPLAY"):
        raise SystemExit("[play_tsr_pour] $DISPLAY is empty -- see view_pour_tea.py header.")

    # ---- symbols (grounding layer) and composed task frames (DSL layer) ----
    teapot_sym = load_symbols(scene.asset_dirs["teapot"])
    mug_sym = load_symbols(scene.asset_dirs["mug"])
    spout_tip = teapot_sym.frame("spout_tip", "pour_axis")
    tilt_frame = teapot_sym.frame("spout_tip", "tilt_axis", secondary="pour_axis")
    opening = mug_sym.frame("opening_center", "up_axis")

    # ---- scene: manifest placement via the single factory -----------------
    env, _ = make_env(scene, robot=args.robot, has_renderer=True)

    # ---- read settled world poses (ground truth: the sim-study pose source)
    from manip_sim.state import PoseReader
    reader = PoseReader(env, ["teapot", "mug"])
    T0_teapot = pose_from_pos_quat_wxyz(*reader.pose("teapot"))
    T0_mug = pose_from_pos_quat_wxyz(*reader.pose("mug"))

    teapot_joint = env.objects["teapot"].joints[0]

    def pin(T_body, steps):
        q = pose_to_qpos(T_body)
        for _ in range(steps):
            env.sim.data.set_joint_qpos(teapot_joint, q)
            env.sim.data.set_joint_qvel(teapot_joint, np.zeros(6))
            env.sim.forward()
            try:
                env.render()
            except Exception:
                return False        # viewer closed
        return True

    rng = np.random.default_rng(args.seed)

    # ---- phase A: stage-2 intersection samples ----------------------------
    pair = transport_pair(
        T0_mug_body=T0_mug,
        mug_opening=opening,
        spout_tip=spout_tip,
        teapot_body_pos_now=T0_teapot[:3, 3],
        rim_margin=mug_sym.quantities.get("rim_radius", 0.04) * 0.5,
    )
    rep = sample_intersection(pair.subgoal, [pair.path], n=args.n_goals, rng=rng)
    print(f"[play_tsr_pour] stage-2 intersection: {rep.summary()}")
    if not rep.accepted:
        raise SystemExit("[play_tsr_pour] empty intersection -- the TSR pair is "
                         "inconsistent (this is the diagnostic doing its job); "
                         "check frames.json axes and bounds.")
    opening_w = (T0_mug @ opening.T())[:3, 3]
    for i, T_body in enumerate(rep.accepted):
        tip_w = (T_body @ spout_tip.T())[:3, 3]
        print(f"  goal {i}: spout tip {np.round(tip_w, 3)} "
              f"(opening {np.round(opening_w, 3)}, "
              f"dz {tip_w[2] - opening_w[2]:+.3f})")
        pin(T_body, args.hold)

    # ---- phase B: stage-3 pour sweep about the spout tip -------------------
    T_entry = rep.accepted[-1]                     # pour from the last goal
    ppair = pour_pair(T0_body_at_entry=T_entry, tilt_frame=tilt_frame,
                      tilt_target=np.deg2rad(args.tilt_deg))
    print(f"[play_tsr_pour] pour sweep 0 -> {args.tilt_deg:.0f} deg about "
          f"tilt axis at the spout tip; the tip should stay pinned.")
    for k in range(args.sweep_steps + 1):
        roll = np.deg2rad(args.tilt_deg) * k / args.sweep_steps
        T_body = ppair.path.T0_w @ displacement_to_pose(
            np.array([0, 0, 0, roll, 0, 0])) @ ppair.path.Tw_e
        assert ppair.path.contains(T_body, tol=1e-6)
        pin(T_body, 1)
    assert ppair.subgoal.contains(T_body, tol=1e-6), \
        "sweep endpoint should satisfy the pour subgoal TSR"
    print("[play_tsr_pour] endpoint inside pour subgoal TSR. Holding; Esc quits.")
    try:
        pin(T_body, 100000)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()