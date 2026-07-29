"""End-to-end AnyGrasp smoke test against the canonical pour-tea scene:

    render RGB-D -> deproject cloud -> ZMQ round trip to the AnyGrasp
    sidecar -> convert to grip-site proposals -> grasp-TSR classifier
    -> (optional) IK + collision on the survivors.

No planning, no probe — this isolates the perception seam so each failure
mode names itself:

    cloud centroid far from the teapot   -> deprojection / flip / extrinsic bug
    server returns n=0                   -> crop too tight, steering too strict,
                                            or cloud in the wrong frame
    grasps returned, all rejected        -> frame-convention or tcp_offset
                                            error (look at the per-axis
                                            displacement it prints), or
                                            AnyGrasp genuinely proposing
                                            outside the TSR elevation band
                                            (steer with --steer, or widen
                                            rp_tol/wrap_rot)
    survivors, none IK-reachable         -> same story as the synthetic
                                            pipeline; not a perception bug

Run inside the sim container with the sidecar up:

    docker compose --profile grasp up -d grasp
    docker compose run --rm sim bash -lc \\
        "PYTHONPATH=. python scripts/test_anygrasp.py --steer --ik"

Saves cloud + raw grasps + converted proposals to outputs/anygrasp/ for
offline inspection.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from scripts.demos.demo_pour_tea import TABLE_TOP_Z, make_env
from manip_sim.frames import load_symbols
from manip_sim.grasping import classify_grasps, handle_grasp_tsr, wrist_flip
from manip_sim.perception.anygrasp_proposals import (anygrasp_proposals,
                                                     steer_toward)
from manip_sim.perception.cloud import object_workspace, render_cloud
from manip_sim.perception.grasp_client import GraspClient
from manip_sim.tsr import pose_from_pos_quat_wxyz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--addr", default=None,
                    help="override ANYGRASP_ADDR (e.g. tcp://localhost:5666)")
    ap.add_argument("--tcp-offset", type=float, default=0.0,
                    help="grip-site shift along approach, m; calibrate via "
                         "the printed TSR displacements")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--max-n", type=int, default=50,
                    help="top-N raw grasps to convert")
    ap.add_argument("--steer", action="store_true",
                    help="pass the TSR nominal approach as approach_steering")
    ap.add_argument("--steer-thresh-deg", type=float, default=60.0)
    ap.add_argument("--grasp-elevation", type=float, default=35.0)
    ap.add_argument("--tsr-tol", type=float, default=1e-6,
                    help="classifier containment tol (loosen to e.g. 5e-3 "
                         "while calibrating tcp_offset)")
    ap.add_argument("--ik", action="store_true",
                    help="also run IK + collision on the survivors")
    ap.add_argument("--out", default="outputs/anygrasp/last_query.npz")
    args = ap.parse_args()

    # ---- scene (headless) + ground-truth poses ----------------------------
    env, objs = make_env(robot=args.robot, has_renderer=False)
    if "teapot" not in objs:
        raise SystemExit("[anygrasp] teapot mesh not converted — AnyGrasp "
                         "needs real geometry to see (assets/objects/teapot).")
    from manip_sim.state import PoseReader
    reader = PoseReader(env, list(objs))
    T0_teapot = pose_from_pos_quat_wxyz(*reader.pose("teapot"))

    teapot_sym = load_symbols("assets/objects/teapot")
    handle = teapot_sym.frame("handle_center", "handle_axis")
    a_body = -handle.point.copy()
    a_h = handle.T()[:3, :3].T @ a_body
    gtsr = handle_grasp_tsr(T0_teapot, handle, a_h,
                            elevation=np.deg2rad(args.grasp_elevation))

    # ---- render + deproject ----------------------------------------------
    ws = object_workspace(T0_teapot[:3, 3], TABLE_TOP_Z)
    pts_cam, pts_world, colors, T_wc = render_cloud(
        env, camera=args.camera, width=args.width, height=args.height,
        workspace=ws)
    if len(pts_cam) < 500:
        raise SystemExit(f"[anygrasp] only {len(pts_cam)} points in the crop "
                         f"— is the {args.camera} camera looking at the "
                         "table? Try --camera frontview / birdview.")
    centroid = pts_world.mean(axis=0)
    err = np.linalg.norm(centroid[:2] - T0_teapot[:2, 3])
    print(f"[cloud] {len(pts_cam)} pts from '{args.camera}'; world centroid "
          f"{np.round(centroid, 3)} vs teapot {np.round(T0_teapot[:3, 3], 3)} "
          f"(xy err {err:.3f} m — should be small; large means the "
          "deprojection chain is wrong)")

    # steer AnyGrasp's grasp REGION to the object (not the table slab kept
    # for its collision check): world-z above the tabletop
    region = pts_world[:, 2] > TABLE_TOP_Z + 0.01
    print(f"[cloud] region_steering marks {int(region.sum())} object pts")

    # ---- server round trip -----------------------------------------------
    kw = dict(region_steering=region)
    if args.steer:
        a_world = (gtsr.T0_w @ gtsr.Tw_e)[:3, 2]      # nominal approach, +z
        kw["approach_steering"] = steer_toward(a_world, T_wc)
        kw["approach_thresh"] = np.deg2rad(args.steer_thresh_deg)
        print(f"[server] steering approach toward {np.round(a_world, 2)} "
              f"(world), thresh {args.steer_thresh_deg:.0f} deg")
    client = GraspClient(addr=args.addr)
    t0 = time.time()
    rep = client.get_grasps(pts_cam, colors, lims=None, **kw)
    print(f"[server] {rep['n']} grasps in {time.time() - t0:.2f}s")
    if rep["n"] == 0:
        raise SystemExit("[server] empty reply — loosen steering, widen the "
                         "workspace crop, or check the cloud frame.")
    k = min(5, rep["n"])
    print(f"[server] top-{k}: scores {np.round(rep['scores'][:k], 3)}, "
          f"widths {np.round(rep['widths'][:k], 3)}")

    # ---- convert + classify ----------------------------------------------
    proposals = anygrasp_proposals(rep, T_wc, tcp_offset=args.tcp_offset,
                                   min_score=args.min_score, max_n=args.max_n)
    survivors, tally = classify_grasps(gtsr, proposals, tol=args.tsr_tol)
    survivors.sort(key=lambda p: p.tsr_distance)
    print(f"[classifier] {len(proposals)} proposals (both wrist branches) "
          f"-> {len(survivors)} kept  {tally}")
    best = min(proposals, key=lambda p: p.tsr_distance)
    d = gtsr.displacement(best.T0_ee)
    print(f"[classifier] nearest proposal TSR distance "
          f"{best.tsr_distance:.4f}; per-axis displacement "
          f"xyz {np.round(d[:3], 4)} m, rpy {np.round(np.rad2deg(d[3:]), 1)} "
          f"deg  (systematic x/y/z offset across proposals -> tune "
          f"--tcp-offset; big roll/pitch -> elevation band mismatch, "
          f"try --steer)")

    # ---- optional IK + collision on survivors ----------------------------
    if args.ik and survivors:
        from manip_sim.planning import ArmKinematics, MinkIK
        kin = ArmKinematics(env)
        ik = MinkIK(kin)
        q_home = env.sim.data.qpos[kin.qpos_ids].copy()
        seeds = [q_home]
        for dj0 in (-1.0, -0.5, 0.5, 1.0):
            s0 = q_home.copy(); s0[0] += dj0
            seeds.append(s0)
        allowed = ((("gripper0", "gripper0"), ("gripper0", "teapot")))
        n_ok = 0
        for p in survivors:
            for T in (p.T0_ee, wrist_flip(p.T0_ee)):
                q, ok = ik.solve_multiseed(T, seeds)
                if ok and not kin.in_collision(q, allowed_prefix_pairs=allowed):
                    n_ok += 1
                    break
        print(f"[ik] {n_ok}/{len(survivors)} survivors reachable & "
              "collision-free")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pts_cam=pts_cam, pts_world=pts_world, colors=colors,
             T_world_cam=T_wc, T0_teapot=T0_teapot,
             translations=rep["translations"], rotations=rep["rotations"],
             scores=rep["scores"], widths=rep["widths"],
             depths=rep["depths"],
             proposal_poses=np.stack([p.T0_ee for p in proposals]),
             proposal_dists=np.array([p.tsr_distance for p in proposals]))
    print(f"[anygrasp] saved query artifact -> {out}")
    env.close()


if __name__ == "__main__":
    main()