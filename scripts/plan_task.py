"""Generic multi-stage planner: a stage plan + call #2 selections + call #3
emissions -> one joint-space plan, for any task the contract binds. No API
calls: it consumes the artifacts run_pipeline wrote.

    GRASP     the grasped object's selected part (a line or ring fit) ->
              grasp-TSR classifier over the sim grasp stand-in's proposals
              (top-down, GRASP_ELEVATION_RAD; slide sampled over the part's
              fitted length) -> IK + collision (wrist-flip pairs) ->
              LOOKAHEAD over every downstream stage -> free reach to
              pre-grasp, straight approach -> T_ee_body MEASURED -> lift.
              The emitted grasp TSR is not used, as in pour-tea (a gripper
              nominal is not in the emission vocabulary).
    per mover stage, in bound order
              compile the emission at the ACTUAL entry pose (compile_stage,
              the same call the emit gate made at its nominal entry) ->
              [synthesized start: the subgoal nominal raised along the
              passive's +up until the mover clears it, asserted inside the
              stage's path TSR, reached by a free-space plan] ->
              subgoal INTERSECT path samples -> IK -> collision ->
              containment -> CBiRRT on the path TSR, walking ranked goals.

Every failure is typed (manip_sim.task_plan.TaskPlanError) and written to
the report with its evidence; nothing is skipped silently. pour-tea keeps
scripts/plan_pour_tea.py; this driver does not touch it.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/plan_task.py \\
        --scene scenes/insert_marker.json --grounding outputs/grounding/insert_marker \\
        --stage-plan outputs/stage_plan/insert_marker.marks.json \\
        --selections outputs/selections/insert_marker.json \\
        --emissions outputs/emissions/insert_marker.json

Writes outputs/plans/vlm/<scene>_task_vlm.npz (path, stage ids/labels,
q_home, q_grasp, T_ee_body, initial object poses, provenance) and a
.report.json beside it (per stage: entry kind, funnel counts, collision
offenders, reorientation split, which end of the mover is lowest at the
goal). On failure only the report is written, with status "failed".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from manip_sim.compile_tsr import CompileError, compile_stage
from manip_sim.frames import Frame, load_symbols
from manip_sim.grasping import (classify_grasps, free_tsr, handle_grasp_tsr,
                                propose_handle_grasps, wrist_flip)
from manip_sim.planning import (ArmKinematics, AttachedArmKinematics,
                                AttachedObject, MinkIK, plan_constrained)
from manip_sim.proposal import load_obj
from manip_sim.provenance import stamp, tagged
from manip_sim.scene import add_scene_arg, load_scene, make_env
from manip_sim.selection import (load_pool, load_selections, selected_points,
                                 selection_objects)
from manip_sim.task_plan import (GRASP_ELEVATION_RAD, START_CONTAINMENT_TOL,
                                 SYNTH_START_CLEARANCE_M, TaskPlanError,
                                 decomposition, grasp_part, lowest_end,
                                 mover_stages, probe_fractions,
                                 rotation_angle_deg, sample_goals,
                                 synthesized_start, topdown_approach_h,
                                 walk_goals)
from manip_sim.tsr import pose_from_pos_quat_wxyz, sample_intersection
from manip_sim.vlm import load_emissions
from scripts.emit_constraints import bound_stages
from scripts.plan_pour_tea import (GOAL_SAMPLE_MAX, MIN_ACCEPT_RATE,
                                   _funnel_offender)
from scripts.plan_stages import load_bindings

PREGRASP_STANDOFF = 0.08          # same as plan_pour_tea
LIFT_HEIGHT = 0.05
APPROACH_STEPS = 12
GOALS_TRIED = 10                  # ranked goals walked per stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-plan", required=True, metavar="JSON")
    ap.add_argument("--selections", required=True, metavar="JSON")
    ap.add_argument("--emissions", required=True, metavar="JSON")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-proposals", type=int, default=80)
    ap.add_argument("--max-grasp-candidates", type=int, default=6)
    ap.add_argument("--n-goal-samples", type=int, default=120)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="default outputs/plans/vlm/<scene>_task_vlm.npz")
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene, getattr(args, "grounding", None))
    out = Path(args.out) if args.out else tagged("outputs/plans", f"{scene.name}_task", ".npz", "vlm")
    report = {"scene": args.scene, "grounding": getattr(args, "grounding", None),
              "stage_plan": args.stage_plan, "selections": args.selections,
              "emissions": args.emissions, "seed": args.seed,
              "constants": {"SYNTH_START_CLEARANCE_M": SYNTH_START_CLEARANCE_M,
                            "GRASP_ELEVATION_RAD (sim stand-in)": GRASP_ELEVATION_RAD},
              "status": "running", "grasp": {}, "stages": []}
    try:
        plan(args, scene, out, report)
        report["status"] = "ok"
    except TaskPlanError as e:
        report["status"] = "failed"
        report["failure"] = e.as_dict()
        print(f"\n[plan-task] FAILED {e}")
        for k, v in e.detail.items():
            print(f"    {k}: {v}")
    rp = out.with_suffix(".report.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    print(f"[plan-task] report -> {rp}")
    sys.exit(0 if report["status"] == "ok" else 2)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def plan(args, scene, out: Path, report: dict) -> None:
    rng = np.random.default_rng(args.seed)
    asset_dirs = scene.asset_dirs
    symbols = {n: load_symbols(d) for n, d in asset_dirs.items()}
    frames_docs = {n: json.loads((Path(d) / "frames.json").read_text())
                   for n, d in asset_dirs.items()}
    meshes = {}
    for n, d in asset_dirs.items():
        m = Path(d) / "meshes" / f"{n}_visual.obj"
        if not m.exists():
            raise TaskPlanError("compile", "setup", f"{m} missing: the planner "
                                "needs every object's converted mesh")
        meshes[n] = load_obj(m)[0]

    # ---- the artifacts --------------------------------------------------
    b = load_bindings(args.stage_plan)
    by_idx = {s.index: s for s in b.plan.stages}
    staged = bound_stages(b, by_idx)
    ems = load_emissions(args.emissions)
    em_doc = json.loads(Path(args.emissions).read_text())
    report["emit_model"] = em_doc.get("model")
    report["emit_arm"] = em_doc.get("arm")
    movers = mover_stages(staged, ems, b.entry_synthesized)
    sels = load_selections(args.selections)
    objects = selection_objects(args.selections, sels)
    role_index = {r: st.index for r, (_, st) in b.roles.items()}
    points = {role: selected_points(role, stage, sels, objects, role_index,
                                    asset_dirs, symbols) for stage, role in staged}
    grasp_role = next((r for st, r in staged if st.passive is None), None)
    if grasp_role is None or grasp_role not in sels:
        raise TaskPlanError("grasp_part", "grasp", "no grasp stage/selection bound")
    obj = objects[grasp_role]
    print(f"[plan-task] stages: {[r for _, r in staged]}; mover stages "
          f"{[(m.role, 'synthesized' if m.synthesized else 'chained') for m in movers]}")

    # ---- the world ------------------------------------------------------
    env, objs = make_env(scene, robot=args.robot, has_renderer=False)
    from manip_sim.state import PoseReader
    reader = PoseReader(env, list(objs))
    T0 = {n: pose_from_pos_quat_wxyz(*reader.pose(n)) for n in objs}
    kin = ArmKinematics(env)
    ik = MinkIK(kin)
    q_home = env.sim.data.qpos[kin.qpos_ids].copy()
    lo, hi = kin.joint_range[:, 0], kin.joint_range[:, 1]
    seeds = [q_home] + [q_home + np.r_[d, np.zeros(5)] for d in (-1.0, -0.5, 0.5, 1.0)]
    seeds += [rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi)) for _ in range(8)]

    # ---- GRASP ----------------------------------------------------------
    print("\n================ grasp ================")
    pool = load_pool(asset_dirs[obj])
    cand = pool.get(int(sels[grasp_role].candidate_id), {})
    gp = grasp_part(frames_docs[obj], cand.get("part"))
    sec = np.array([0.0, 0.0, 1.0]) if abs(gp.axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    gframe = Frame(f"{obj}.{gp.part}", gp.center, gp.axis, sec)
    T0_w = T0[obj] @ gframe.T()
    a_h = topdown_approach_h(T0_w)
    gtsr = handle_grasp_tsr(T0[obj], gframe, a_h, elevation=GRASP_ELEVATION_RAD,
                            slide=max(gp.half_length, 1e-3))
    proposals = propose_handle_grasps(gtsr, rng, n=args.n_proposals,
                                      slide_span=max(gp.half_length, 1e-3))
    survivors, tally = classify_grasps(gtsr, proposals)
    survivors.sort(key=lambda p: p.tsr_distance)
    grasp_allowed = (("gripper0", "gripper0"), ("gripper0", obj))
    cands = []
    for p in survivors:
        if len(cands) >= args.max_grasp_candidates:
            break
        for T in (p.T0_ee, wrist_flip(p.T0_ee)):
            q, ok = ik.solve_multiseed(T, seeds)
            if ok and not kin.in_collision(q, allowed_prefix_pairs=grasp_allowed):
                cands.append((q, kin.fk(q)))
    report["grasp"] = {"part": gp.part, "primitive": gp.primitive,
                       "slide_half_length_m": gp.half_length,
                       "proposed": len(proposals), "classified": len(survivors),
                       "reachable": len(cands)}
    print(f"[grasp] part {gp.part} ({gp.primitive}, slide +-{gp.half_length * 1000:.0f} mm): "
          f"{len(proposals)} proposed -> {len(survivors)} classified -> {len(cands)} reachable")
    if not cands:
        raise TaskPlanError("grasp", "grasp", "no reachable, collision-free grasp",
                            report["grasp"])

    # lookahead over every mover stage, compiled at its nominal entry the way
    # the emit gate did (spawn, then the previous stage's subgoal nominal)
    probe_samples, T_ent = [], T0[obj]
    for m in movers:
        w_pt, e_pt = points[m.role]
        try:
            cs = compile_stage(m.emission, symbols, {m.stage.active: T_ent,
                               m.stage.passive: T0[m.stage.passive]},
                               w_point=w_pt, e_point=e_pt, entry_synthesized=m.synthesized)
        except CompileError as e:
            raise TaskPlanError("compile", m.role, str(e), {"slot": e.slot})
        probe_samples.append(sample_intersection(cs.subgoal, [cs.path],
                                                 n=args.n_probe, rng=rng).accepted)
        T_ent = cs.subgoal.nominal()
    scored = []
    for q_g, T_g in cands:
        fr, sc = probe_fractions(ik, kin, T_g, T0[obj], probe_samples, seeds)
        scored.append((sc, fr, q_g, T_g))
    scored.sort(key=lambda s: (-s[0], float(np.linalg.norm(s[2] - q_home))))
    axis_w = T0[obj][:3, :3] @ gp.axis
    centre_w = (T0[obj] @ np.r_[gp.center, 1.0])[:3]
    report["grasp"]["lookahead"] = [
        {"score": round(sc, 3), "fractions": [round(f, 3) for f in fr],
         "slide_offset_mm": round(1000 * float((T_g[:3, 3] - centre_w) @ axis_w), 1)}
        for sc, fr, _, T_g in scored]
    for row in report["grasp"]["lookahead"]:
        print(f"    probe {row}")

    def track_line(kin_, ik_, p_to, q_seed, T_rot, steps, allowed=None):
        qs, q = [], np.asarray(q_seed, float)
        p_from = kin_.fk(q)[:3, 3]
        for k in range(1, steps + 1):
            T = T_rot.copy()
            T[:3, 3] = p_from + (p_to - p_from) * k / steps
            q_new, ok = ik_.solve(T, q, iters=60)
            if not ok or np.linalg.norm(q_new - q) > 0.35:
                return None, f"IK lost track at step {k}/{steps}"
            coll = (kin_.in_collision(q_new, allowed_prefix_pairs=allowed)
                    if allowed is not None else kin_.in_collision(q_new))
            if coll and not (allowed is None and k <= 2):
                return None, f"collides at step {k}/{steps} ({_funnel_offender(kin_)})"
            qs.append(q_new)
            q = q_new
        return qs, None

    chosen, skips = None, []
    for sc, fr, q_grasp, T_g in scored:
        if sc <= 0.0:
            break
        p_pre = T_g[:3, 3] - PREGRASP_STANDOFF * T_g[:3, 2]
        retreat, why = track_line(kin, ik, p_pre, q_grasp, T_g, APPROACH_STEPS, grasp_allowed)
        if retreat is None:
            skips.append(f"approach corridor: {why}")
            continue
        res1 = plan_constrained(kin, AttachedObject(np.eye(4)), [free_tsr()],
                                q_home, retreat[-1], timeout=args.timeout)
        if not res1.ok:
            skips.append(f"reach: {res1.reason}")
            continue
        chosen = (q_grasp, T_g, res1, retreat[-2::-1] + [q_grasp], sc)
        break
    if chosen is None:
        raise TaskPlanError("grasp", "grasp", "no candidate with a positive lookahead "
                            "score survived approach + reach",
                            {"skips": skips, "lookahead": report["grasp"]["lookahead"]})
    q_grasp, T_g, res1, approach, sc = chosen
    report["grasp"].update(chosen_score=round(sc, 3), chosen_slide_offset_mm=round(
        1000 * float((T_g[:3, 3] - centre_w) @ axis_w), 1), skips=skips)
    segments = [("grasp:reach", np.vstack([res1.path, approach]))]

    T_ee_body = np.linalg.inv(kin.fk(q_grasp)) @ T0[obj]
    attached = AttachedObject(T_ee_body)
    kin_att = AttachedArmKinematics(env, attached, env.objects[obj].joints[0], obj)
    ik_att = MinkIK(kin_att)
    T_up = kin.fk(q_grasp)
    p_up = T_up[:3, 3] + np.array([0.0, 0.0, LIFT_HEIGHT])
    lift, why = track_line(kin_att, ik_att, p_up, q_grasp, T_up, APPROACH_STEPS)
    if lift is None:
        raise TaskPlanError("lift", "grasp", f"post-grasp lift infeasible: {why}")
    segments.append(("grasp:lift", np.array(lift)))
    q_cur = lift[-1]

    # ---- MOVER STAGES ---------------------------------------------------
    for m in movers:
        print(f"\n================ {m.role}: {m.stage.name} ================")
        mover, passive = m.stage.active, m.stage.passive
        w_pt, e_pt = points[m.role]
        T_entry = attached.body_pose(kin_att.fk(q_cur))
        rec = {"role": m.role, "name": m.stage.name,
               "entry": "synthesized" if m.synthesized else "chained"}
        report["stages"].append(rec)
        try:
            cs = compile_stage(m.emission, symbols, {mover: T_entry, passive: T0[passive]},
                               w_point=w_pt, e_point=e_pt, entry_synthesized=m.synthesized)
        except CompileError as e:
            raise TaskPlanError("compile", m.role, str(e), {"slot": e.slot})
        rec["compile_notes"] = list(cs.notes)
        q_stage, T_stage = q_cur, T_entry
        if m.synthesized:
            T_start, shift = synthesized_start(cs.subgoal.nominal(), meshes[mover],
                                               T0[passive], meshes[passive],
                                               symbols[passive].axes["up_axis"])
            rec["synthesized_start"] = {"raise_m": round(shift, 4),
                                        "pos": np.round(T_start[:3, 3], 4).tolist()}
            dist = cs.path.distance(T_start)
            if dist > START_CONTAINMENT_TOL:
                raise TaskPlanError("synthesized_start_outside_path", m.role,
                                    "the synthesized start is not inside the stage's path TSR",
                                    {"excess": np.round(cs.path.excess(T_start), 4).tolist()})
            q_s, ok = ik_att.solve_multiseed(T_start @ np.linalg.inv(T_ee_body),
                                             [q_cur] + seeds)
            if not ok or kin_att.in_collision(q_s):
                raise TaskPlanError("synthesized_start_unreachable", m.role,
                                    "no collision-free IK at the synthesized start",
                                    {"ik": bool(ok), "offender": _funnel_offender(kin_att) if ok else None})
            r = plan_constrained(kin_att, attached, [free_tsr()], q_cur, q_s,
                                 timeout=args.timeout)
            if not r.ok:
                raise TaskPlanError("free_reach", m.role, f"free-space reach to the "
                                    f"synthesized start failed: {r.reason}")
            segments.append((f"{m.role}:synthesized_reach", r.path))
            rec["reorient_free_reach_deg"] = round(rotation_angle_deg(T_entry[:3, :3],
                                                                      T_start[:3, :3]), 1)
            q_stage, T_stage = q_s, attached.body_pose(kin_att.fk(q_s))

        goals, funnel = sample_goals(cs.subgoal, cs.path, args.n_goal_samples,
                                     GOAL_SAMPLE_MAX, MIN_ACCEPT_RATE, rng, ik_att,
                                     kin_att, attached, [q_stage] + seeds,
                                     _funnel_offender, q_stage)
        rec["goal_funnel"] = funnel.as_dict()
        print(f"[{m.role}] goal funnel: {funnel.as_dict()}")
        if not goals:
            raise TaskPlanError("empty_goal_set", m.role,
                                "no collision-free, contained goal configuration",
                                funnel.as_dict())
        res, k, fails = walk_goals(plan_constrained, kin_att, attached, [cs.path],
                                   q_stage, goals, GOALS_TRIED, args.timeout, m.role)
        rec["planning"] = {"goal_rank_used": k, "failed_before": fails,
                           "n_waypoints": len(res.path),
                           "max_path_excess": round(float(res.max_excess), 4)}
        segments.append((m.role, res.path))
        q_cur = res.path[-1]
        T_end = attached.body_pose(kin_att.fk(q_cur))
        rec["reorient_in_stage_deg"] = round(rotation_angle_deg(T_stage[:3, :3],
                                                                T_end[:3, :3]), 1)
        rec["lowest_end_at_goal"] = lowest_end(T_end, meshes[mover], symbols[mover].points)
        print(f"[{m.role}] planned: {rec['planning']}; lowest end {rec['lowest_end_at_goal']}")

    report["decomposition"] = decomposition(report["stages"])
    print(f"\n[plan-task] decomposition: {report['decomposition']}")
    path = np.vstack([s for _, s in segments])
    stage_ids = np.concatenate([np.full(len(s), i) for i, (_, s) in enumerate(segments)])
    labels = np.array([lab for lab, _ in segments])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, path=path, stage_ids=stage_ids, stage_labels=labels, q_home=q_home,
             q_grasp=q_grasp, T_ee_body=T_ee_body, grasped=np.array(obj),
             **{f"T0_{n}": T for n, T in T0.items()},
             **stamp("vlm", args.selections, args.emissions))
    report["plan"] = {"path": str(out), "n_waypoints": int(len(path)),
                      "segments": [(lab, int(len(s))) for lab, s in segments]}
    print(f"[plan-task] {len(path)} waypoints -> {out}")
    env.close()


if __name__ == "__main__":
    main()
