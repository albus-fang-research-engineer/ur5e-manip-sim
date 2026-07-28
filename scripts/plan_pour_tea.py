"""All three stages of the pour-tea task, end to end, one plan artifact:

  STAGE 1  GRASP      proposals (AnyGrasp stand-in) -> grasp-TSR CLASSIFIER
                      -> IK + collision (wrist-flip symmetry) -> LOOKAHEAD
                      FEASIBILITY PROBE over stages 2-3 -> best grasp ->
                      free plan home -> pre-grasp, straight approach ->
                      T_ee_body MEASURED at grasp completion (frozen).
  STAGE 2  TRANSPORT  subgoal (spout tip @ mug opening) INTERSECT upright
                      path -> eef goals via T_ee_body -> IK funnel ->
                      CBiRRT on the upright path manifold.
  STAGE 3  POUR       pour pair FROZEN AT STAGE ENTRY (w = tilt axis at the
                      spout tip where transport actually ended) -> subgoal
                      samples -> IK funnel -> CBiRRT on the pivot corridor
                      -> dwell at pouring attitude (termination is
                      perceptual, not a pose event; the dwell stands in).

The printed FUNNELS (proposed -> classified -> IK -> collision-free ->
probe-ranked; sampled -> IK -> collision-free -> contained) are the
architecture's per-stage diagnostics — attrition at each step is expected
and informative; a zero anywhere names the failing layer.

Runs headless, with or without converted meshes (mesh-free falls back to
synthetic object poses at the canonical scene coordinates; collision
checking against the objects is then vacuous, same caveat as
plan_transport.py).

    PYTHONPATH=. python scripts/plan_pour_tea.py
    PYTHONPATH=. python scripts/plan_pour_tea.py --tilt-deg 95 --seed 3

Render the saved plan:

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import robosuite as suite  # noqa: F401  (env built via the scene factory)
from scipy.spatial.transform import Rotation as R

from scripts.demos.demo_pour_tea import MUG_XY, TEAPOT_XY, make_env
from manip_sim.frames import load_symbols
from manip_sim.grasping import (
    ProbeContext,
    classify_grasps,
    free_tsr,
    handle_grasp_tsr,
    lookahead_probe,
    propose_handle_grasps,
    wrist_flip,
)
from manip_sim.planning import ArmKinematics, AttachedObject, MinkIK, plan_constrained
from manip_sim.pour_stages import pour_pair, transport_pair
from manip_sim.tsr import make_pose, pose_from_pos_quat_wxyz, sample_intersection

TABLE_TOP_Z = 0.8
PREGRASP_STANDOFF = 0.08          # meters back along the approach axis
APPROACH_STEPS = 12               # joint-interp steps pre-grasp -> grasp


def _synthetic_object_poses(teapot_sym):
    """Mesh-free fallback: the canonical scene poses, teapot yawed so the
    spout (frames.json pour_axis, not the retired SPOUT_YAW_OFFSET
    constant) faces the mug — same geometry the factory commands."""
    bearing = MUG_XY - TEAPOT_XY
    pour = teapot_sym.axes["pour_axis"]
    yaw = float(np.arctan2(bearing[1], bearing[0])) - \
        float(np.arctan2(pour[1], pour[0]))
    z0 = TABLE_TOP_Z + 0.06
    T0_teapot = make_pose([*TEAPOT_XY, z0], R.from_euler("z", yaw).as_matrix())
    T0_mug = make_pose([*MUG_XY, z0])
    return T0_teapot, T0_mug


def _park_teapot(env, kin, objs):
    """The teapot is 'in hand' from stage 2 on: park its free body far
    ABOVE the planner's scratch world so it is not a phantom obstacle at
    its old table spot (above, not below — see plan_transport.py)."""
    if "teapot" not in objs:
        return
    jname = env.objects["teapot"].joints[0]
    jid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    adr = kin.model.jnt_qposadr[jid]
    kin.data.qpos[adr: adr + 3] = [0.0, 0.0, 5.0]
    mujoco.mj_forward(kin.model, kin.data)


def _goal_funnel(rep, ik, kin, attached, seeds, containment, label,
                 q_ref, ik_kw=None):
    """Shared stage-2/3 funnel: sampled body poses -> IK -> collision ->
    containment of the ACHIEVED config; sorted nearest q_ref first."""
    goals, n_ik, n_col = [], 0, 0
    t0 = time.time()
    for T_body in rep.accepted:
        T_ee = T_body @ np.linalg.inv(attached.T_ee_body)
        q, ok = ik.solve_multiseed(T_ee, seeds, **(ik_kw or {}))
        if not ok:
            continue
        n_ik += 1
        if kin.in_collision(q):
            continue
        n_col += 1
        T_ach = attached.body_pose(kin.fk(q))
        if all(t.contains(T_ach, tol=5e-3) for t in containment):
            goals.append(q)
    print(f"[{label}] goal funnel: {len(rep.accepted)} sampled -> "
          f"{n_ik} IK-feasible -> {n_col} collision-free -> "
          f"{len(goals)} contained  ({time.time() - t0:.1f}s)")
    goals.sort(key=lambda q: float(np.linalg.norm(q - q_ref)))
    return goals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-proposals", type=int, default=80)
    ap.add_argument("--max-grasp-candidates", type=int, default=6,
                    help="classified grasps to carry through IK + probe")
    ap.add_argument("--n-goal-samples", type=int, default=30)
    ap.add_argument("--n-probe", type=int, default=8,
                    help="shared body-pose probes per downstream stage")
    ap.add_argument("--tilt-deg", type=float, default=95.0)
    ap.add_argument("--dwell", type=int, default=15,
                    help="waypoints holding the pour attitude")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--out", default="outputs/plans/pour_tea_full.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- THE canonical scene ----------------------------------------------
    env, objs = make_env(robot=args.robot, has_renderer=False)

    teapot_sym = load_symbols("assets/objects/teapot")
    mug_sym = load_symbols("assets/objects/mug")
    handle = teapot_sym.frame("handle_center", "handle_axis")
    spout_tip = teapot_sym.frame("spout_tip", "pour_axis")
    tilt_frame = teapot_sym.frame("spout_tip", "tilt_axis",
                                  secondary="pour_axis")
    opening = mug_sym.frame("opening_center", "up_axis")

    if objs:
        from manip_sim.state import PoseReader
        reader = PoseReader(env, list(objs))
        T0_teapot = pose_from_pos_quat_wxyz(*reader.pose("teapot"))
        T0_mug = pose_from_pos_quat_wxyz(*reader.pose("mug"))
        print(f"[pour_tea] settled poses from sim: teapot "
              f"{np.round(T0_teapot[:3, 3], 3)}, mug {np.round(T0_mug[:3, 3], 3)}")
    else:
        T0_teapot, T0_mug = _synthetic_object_poses(teapot_sym)
        print(f"[pour_tea] meshes absent -> synthetic poses: teapot "
              f"{np.round(T0_teapot[:3, 3], 3)}, mug {np.round(T0_mug[:3, 3], 3)}")

    kin = ArmKinematics(env)
    ik = MinkIK(kin)
    q_home = env.sim.data.qpos[kin.qpos_ids].copy()
    lo, hi = kin.joint_range[:, 0], kin.joint_range[:, 1]
    lo_s, hi_s = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)
    seeds = [q_home]
    for dj0 in (-1.0, -0.5, 0.5, 1.0):        # base-joint sweeps: cheap
        s0 = q_home.copy()                     # coverage of azimuth branches
        s0[0] += dj0
        seeds.append(s0)
    seeds += [rng.uniform(lo_s, hi_s) for _ in range(8)]
    # fingers around the handle inevitably contact the teapot's collision
    # hulls at the grasp config; that contact IS the grasp, not a fault
    grasp_allowed = ((("gripper0", "gripper0"), ("gripper0", "teapot")))

    # ======================================================== STAGE 1: GRASP
    print("\n================ stage 1: grasp ================")
    # approach: horizontal component of (handle -> body center), pitched
    # down inside the TSR nominal (UR5e reach favors oblique-from-above)
    a_body = -handle.point.copy()
    a_h = handle.T()[:3, :3].T @ a_body
    gtsr = handle_grasp_tsr(T0_teapot, handle, a_h)

    proposals = propose_handle_grasps(
        gtsr, rng, n=args.n_proposals,
        junk_points=[(T0_teapot @ spout_tip.T())[:3, 3],
                     T0_teapot[:3, 3] + [0.0, 0.0, 0.12]])
    survivors, tally = classify_grasps(gtsr, proposals)
    print(f"[grasp] classifier: {len(proposals)} proposed -> "
          f"{len(survivors)} kept "
          f"(handle {tally['handle_kept']}/{tally['handle_kept'] + tally['handle_rejected']}, "
          f"junk {tally['junk_kept']}/{tally['junk_kept'] + tally['junk_rejected']})")
    if not survivors:
        raise SystemExit("[grasp] classifier kept nothing — widen the "
                         "proposal spread or check handle frame symbols.")
    survivors.sort(key=lambda p: p.tsr_distance)

    # IK + collision, trying the parallel-jaw wrist flip on both branches
    candidates = []                     # (q_grasp, T0_ee_achieved)
    for p in survivors:
        if len(candidates) >= args.max_grasp_candidates:
            break
        for T in (p.T0_ee, wrist_flip(p.T0_ee)):
            q, ok = ik.solve_multiseed(T, seeds)
            if ok and not kin.in_collision(
                    q, allowed_prefix_pairs=grasp_allowed):
                candidates.append((q, kin.fk(q)))
                break
    print(f"[grasp] IK + collision: {len(survivors)} classified -> "
          f"{len(candidates)} reachable candidates")
    if not candidates:
        raise SystemExit("[grasp] no reachable grasp — add IK seeds or "
                         "revisit the approach elevation.")

    # lookahead feasibility probe: shared downstream samples, one score per
    # candidate. Inter-stage coupling lives HERE, not in the grasp TSR.
    tpair_probe = transport_pair(
        T0_mug_body=T0_mug, mug_opening=opening, spout_tip=spout_tip,
        teapot_body_pos_now=T0_teapot[:3, 3],
        rim_margin=mug_sym.quantities.get("rim_radius", 0.04) * 0.5,
        upright_tol=np.deg2rad(5.0),
        z_corridor=(-0.02, 0.45))
    rep2p = sample_intersection(tpair_probe.subgoal, [tpair_probe.path],
                                n=args.n_probe, rng=rng)
    # pour probes are built over SEVERAL candidate entries, not one: the
    # pour's feasibility depends strongly on the transport approach azimuth
    # (the UR5e reach scan shows only a band of azimuths admits the full
    # tilt), and a single lucky entry would zero every candidate's score
    pour_samples = []
    for entry_probe in rep2p.accepted[: max(1, args.n_probe // 2)]:
        ppair_probe = pour_pair(entry_probe, tilt_frame,
                                tilt_target=np.deg2rad(args.tilt_deg))
        rep3p = sample_intersection(ppair_probe.subgoal, [ppair_probe.path],
                                    n=2, rng=rng)
        pour_samples.extend(rep3p.accepted)
    ctx = ProbeContext(transport_samples=rep2p.accepted,
                       pour_samples=pour_samples, seeds=seeds)
    scored = []
    t0 = time.time()
    for q_g, T_g in candidates:
        rep = lookahead_probe(ik, kin, T_g, T0_teapot, ctx)
        scored.append((rep.score, q_g, T_g, rep))
    scored.sort(key=lambda s: (-s[0], float(np.linalg.norm(s[1] - q_home))))
    print(f"[grasp] lookahead probe over {len(candidates)} candidates "
          f"({time.time() - t0:.1f}s):")
    for sc, q_g, _, rep in scored:
        print(f"    {rep.summary()}")
    best_score, q_grasp, T0_ee_grasp, best_rep = scored[0]
    if best_score <= 0.0:
        raise SystemExit("[grasp] every candidate scored 0 on the lookahead "
                         "probe — downstream stages unreachable from any "
                         "surviving grasp; widen wrap_rot or move the scene.")
    print(f"[grasp] selected grasp: {best_rep.summary()}")

    # pre-grasp back along the approach axis; free plan home -> pre-grasp
    T_pre = T0_ee_grasp.copy()
    T_pre[:3, 3] -= PREGRASP_STANDOFF * T0_ee_grasp[:3, 2]
    q_pre, ok = ik.solve_multiseed(T_pre, [q_grasp] + seeds)
    if not ok or kin.in_collision(q_pre):
        raise SystemExit("[grasp] pre-grasp pose infeasible.")
    attached_none = AttachedObject(np.eye(4))
    res1 = plan_constrained(kin, attached_none, [free_tsr()], q_home, q_pre,
                            timeout=args.timeout)
    if not res1.ok:
        raise SystemExit(f"[grasp] reach planning failed: {res1.reason}")
    # straight approach segment, teapot contact allowed at the fingers
    approach = [q_pre + (q_grasp - q_pre) * k / APPROACH_STEPS
                for k in range(1, APPROACH_STEPS + 1)]
    for q in approach:
        if kin.in_collision(q, allowed_prefix_pairs=grasp_allowed):
            raise SystemExit("[grasp] approach segment collides — increase "
                             "the standoff or pick the next candidate.")
    path1 = np.vstack([res1.path, approach])
    print(f"[grasp] reach {res1.stats['n_waypoints']} waypoints in "
          f"{res1.solve_time:.2f}s + {APPROACH_STEPS} approach steps")

    # grasp completion: MEASURE the in-hand transform, freeze it, attach
    T_ee_body = np.linalg.inv(kin.fk(q_grasp)) @ T0_teapot
    attached = AttachedObject(T_ee_body)
    _park_teapot(env, kin, objs)
    print(f"[grasp] measured T_ee_body translation "
          f"{np.round(T_ee_body[:3, 3], 3)} (frozen for stages 2-3)")

    # ==================================================== STAGE 2: TRANSPORT
    print("\n============== stage 2: transport ==============")
    pair = transport_pair(
        T0_mug_body=T0_mug, mug_opening=opening, spout_tip=spout_tip,
        teapot_body_pos_now=T0_teapot[:3, 3],
        rim_margin=mug_sym.quantities.get("rim_radius", 0.04) * 0.5,
        # start sits at the frame origin: corridor must contain z = 0
        upright_tol=np.deg2rad(2.0),
        z_corridor=(-0.02, 0.45))
    rep2 = sample_intersection(pair.subgoal, [pair.path],
                               n=args.n_goal_samples, rng=rng)
    print(f"[transport] intersection: {rep2.summary()}")
    goals2 = _goal_funnel(rep2, ik, kin, attached, [q_grasp] + seeds,
                          [pair.path, pair.subgoal], "transport", q_grasp)
    if not goals2:
        raise SystemExit("[transport] no feasible goal configs.")
    # stage-2 -> stage-3 lookahead: rank transport goals by whether the
    # pour subgoal is IK-reachable FROM THAT ENTRY (the pour pair is frozen
    # wherever transport ends, so approach azimuth decides pour
    # feasibility; nearest-config ranking alone routinely picks entries
    # from which the tilt is out of the arm's reach)
    t0 = time.time()
    ranked2 = []
    for q in goals2[: min(len(goals2), 10)]:
        T_ent = attached.body_pose(kin.fk(q))
        pp = pour_pair(T_ent, tilt_frame,
                       tilt_target=np.deg2rad(args.tilt_deg))
        qp, ok = ik.solve_multiseed(
            pp.subgoal.zero() @ np.linalg.inv(attached.T_ee_body),
            [q] + seeds, iters=200)
        pour_ok = ok and not kin.in_collision(qp)
        ranked2.append((not pour_ok, float(np.linalg.norm(q - q_grasp)), q))
    ranked2.sort(key=lambda r: r[:2])
    n_pourable = sum(1 for r in ranked2 if not r[0])
    print(f"[transport] pour lookahead: {n_pourable}/{len(ranked2)} goal "
          f"entries admit the {args.tilt_deg:.0f} deg pour "
          f"({time.time() - t0:.1f}s)")
    if n_pourable == 0:
        print("[transport] WARNING: no probed entry admits the pour — "
              "proceeding with nearest goal; expect stage 3 to fail "
              "(lower --tilt-deg or raise --n-goal-samples).")
    res2 = plan_constrained(kin, attached, [pair.path], q_grasp,
                            ranked2[0][2], timeout=args.timeout)
    if not res2.ok:
        raise SystemExit(f"[transport] planning failed: {res2.reason}")
    print(f"[transport] path: {res2.stats['n_waypoints']} waypoints in "
          f"{res2.solve_time:.2f}s; max path-TSR excess {res2.max_excess:.4f}")
    q2_end = res2.path[-1]

    # ========================================================= STAGE 3: POUR
    print("\n================= stage 3: pour ================")
    # pair FROZEN AT ENTRY: w = tilt axis at the spout tip where transport
    # actually ended (not where it was nominally aimed)
    T_entry = attached.body_pose(kin.fk(q2_end))
    ppair = pour_pair(T_entry, tilt_frame,
                      tilt_target=np.deg2rad(args.tilt_deg))
    rep3 = sample_intersection(ppair.subgoal, [ppair.path],
                               n=args.n_goal_samples, rng=rng)
    print(f"[pour] intersection: {rep3.summary()}")
    goals3 = _goal_funnel(rep3, ik, kin, attached, [q2_end] + seeds,
                          [ppair.path, ppair.subgoal], "pour", q2_end,
                          ik_kw={"iters": 200})
    if not goals3:
        raise SystemExit("[pour] no feasible pour configs — try a smaller "
                         "--tilt-deg or a different grasp seed.")
    # the pivot corridor is thin (5 mm / 3 deg): smaller extension steps
    res3 = plan_constrained(kin, attached, [ppair.path], q2_end, goals3[0],
                            timeout=args.timeout, eps=0.05)
    if not res3.ok:
        raise SystemExit(f"[pour] planning failed: {res3.reason}")
    print(f"[pour] path: {res3.stats['n_waypoints']} waypoints in "
          f"{res3.solve_time:.2f}s; max path-TSR excess {res3.max_excess:.4f}")
    tilt_achieved = ppair.subgoal.displacement(
        attached.body_pose(kin.fk(res3.path[-1])))[3]
    print(f"[pour] achieved tilt {np.rad2deg(tilt_achieved):.1f} deg "
          f"(target {args.tilt_deg:.0f}); holding {args.dwell} dwell "
          f"waypoints (pour termination is perceptual, not a pose event)")
    dwell = np.repeat(res3.path[-1][None, :], args.dwell, axis=0)

    # ---- combined artifact -------------------------------------------------
    path = np.vstack([path1, res2.path, res3.path, dwell])
    stage_ids = np.concatenate([
        np.full(len(path1), 1), np.full(len(res2.path), 2),
        np.full(len(res3.path) + args.dwell, 3)])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, path=path, stage_ids=stage_ids, q_home=q_home,
             q_grasp=q_grasp, T_ee_body=T_ee_body,
             T0_teapot_init=T0_teapot, T0_mug=T0_mug,
             tilt_target=np.deg2rad(args.tilt_deg))
    print(f"\n[pour_tea] {len(path)} waypoints "
          f"(grasp {len(path1)} | transport {len(res2.path)} | "
          f"pour {len(res3.path)}+{args.dwell} dwell) -> {out}")
    env.close()


if __name__ == "__main__":
    main()