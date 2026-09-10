"""VLM touchpoint-#3 driver: one live emit_constraints call per pour
stage, gated by the grounding compiler — the closure plan_stages.py's
exit message has been waiting for. The wrapper (Client.emit_constraints),
prompt, parser and its numeric-literal guardrails already exist and are
tested; what has never existed is a driver that makes the calls and
confronts each emission with compile_tsr, so a licitly-parsed schema
that cannot ground (mixed alignment, off-w anchors, empty row
intersections) fails HERE with a slot-named CompileError, not inside
the planner.

Per stage (grasp / transport / pour: the fixed pour-tea structure
below, or with --stage-plan the stages call #1 bound to those planner
roles — same binding select_frames.py uses):

  1. emit    Client.emit_constraints(StageSpec, vocab[, selection,
             views, frames]). Two arms of the emission-modality
             ablation, same system text, same anchor points:
               schema-only  (default) text only;
               framed       (--frames) the stage's render_stage_frames
                            views — w and the licensed canonical triads
                            drawn at the call-#2 points — plus
                            vlm.frames_legend in the user turn and the
                            #2 candidate id.
             --selections supplies the ANCHOR POINTS (w's origin, the
             active feature) to both arms; it does not pick the arm.
  2. ground  compile_tsr.compile_stage at the manifest's spawn poses,
             then check_pair_consistency (the subgoal must lie on the
             path manifold — the planner's own sample_intersection test,
             run here so a path that forbids its subgoal comes back as
             typed repair text, not as "no grasp survived" downstream)
             (upright, teapot facing the mug — every rule-table gate is
             an attitude question, so the spawn attitude exercises
             exactly what an offline gate can). w is the passive
             object's canonical frame at its selected point; the mover's
             selected point is the feature Tw_e pins. Geometric
             feasibility (IK, non-empty subgoal INTERSECT path under the
             real scene) remains plan_pour_tea's job.
  3. report  PASS/FAIL per stage with the compiler's provenance notes
             and the parser's flagged numeric literals.

The artifact (all raw emissions + compiled B^w rows) and the .log.json
(Client.logs: attempts, rejections, flags) are written BEFORE the gate
verdict — a failing emission is evidence, not garbage. Any FAIL exits
nonzero.

Requires ANTHROPIC_API_KEY.

    PYTHONPATH=. python scripts/emit_constraints.py
    PYTHONPATH=. python scripts/emit_constraints.py \
        --stage-plan outputs/stage_plan/pour_tea.marks.json
    PYTHONPATH=. python scripts/emit_constraints.py \
        --selections outputs/selections/pour_tea.json     # #2 anchors
    PYTHONPATH=. python scripts/emit_constraints.py \
        --selections outputs/selections/pour_tea.json \
        --frames outputs/frames/manifest.json             # framed arm
    PYTHONPATH=. python scripts/emit_constraints.py \
        --out outputs/emissions/other.json

Then: plan_pour_tea.py --emissions <artifact> swaps pour_stages.* for
compile_stage at the transport and pour construction sites (stage 1's
grasp stays on handle_grasp_tsr: the gripper nominal is not expressible
in the emission vocabulary).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from manip_sim.compile_tsr import (ALIGN_TOL_RAD, CompileError, check_pair_consistency, check_stage_seam,
                                   compile_stage, stage_accounting)
from manip_sim.frames import Symbols, load_symbols
from manip_sim.vlm import PROMPT_DELTAS, Client, StageSpec, Vocabulary, entry_line
from manip_sim.scene import add_scene_arg, load_scene

COMPILE_RETRIES = 2      # re-emissions per stage on a CompileError (stopgap
                         # for touchpoint #5); attempts = 1 + retries

OUT = Path("outputs/emissions/pour_tea.json")

# fixed pour-tea stage structure (mirrors select_frames.py's table);
# `role` keys the --selections and --frames artifacts.
STAGES = (
    (StageSpec(index=1, name="grasp", active="teapot", passive=None,
               parts={"teapot": ("handle",)}), "grasp"),
    (StageSpec(index=2, name="transport", active="teapot", passive="mug",
               parts={"teapot": ("spout",), "mug": ("rim",)}), "transport_active"),
    (StageSpec(index=3, name="pour", active="teapot", passive="mug",
               parts={"teapot": ("spout",), "mug": ("rim",)}), "pour"),
)

def _spawn_poses(scene) -> dict[str, np.ndarray]:
    """Compile-gate poses: the manifest's spawn poses (upright, scene
    yaw). Attitude is what the rule-table gates ask about; the spawn
    attitude is the canonical one for every scene, so the gate needs no
    hand-typed constants."""
    from manip_sim.tsr import pose_from_pos_quat_wxyz
    return {n: pose_from_pos_quat_wxyz(*pq) for n, pq in scene.fixed_poses().items()}


def _framed_inputs(role: str, manifest: dict, views=None
                   ) -> tuple[list[Path], dict]:
    """(view paths, render record) for a stage from a render_stage_frames
    manifest; the record feeds vlm.frames_legend. Missing role is a
    hard error: an arm that silently degrades to schema-only would
    corrupt the modality ablation."""
    roles = manifest.get("roles", {})
    if role not in roles:
        raise SystemExit(f"[emit] role {role!r} absent from the --frames "
                         f"manifest (has {sorted(roles)}); re-run "
                         "render_stage_frames.py on the same selections")
    rec = roles[role]
    order = views or manifest.get("views") or sorted(rec["views"])
    paths = [Path(rec["views"][v]) for v in order]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"[emit] --frames views missing on disk: {missing}")
    return paths, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=None, metavar="JSON",
                    help="touchpoint-#2 artifact: the anchor points w is "
                         "rooted on and Tw_e pins (both arms)")
    ap.add_argument("--frames", default=None, metavar="JSON",
                    help="render_stage_frames.py manifest; switches on "
                         "the framed (image-conditioned) arm: the stage's "
                         "views + legend + #2 candidate id in the prompt. "
                         "Requires --selections")
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    ap.add_argument("--stage-plan", default=None, metavar="JSON",
                    help="plan_stages.py artifact; the stages bound to the "
                         "grasp / transport_active / pour roles replace STAGES")
    add_scene_arg(ap)
    args = ap.parse_args()
    if args.frames and not args.selections:
        raise SystemExit("[emit] --frames needs --selections (the frames "
                         "were rendered at those points)")
    scene = load_scene(args.scene, getattr(args, "grounding", None))
    asset_dirs = scene.asset_dirs
    poses = _spawn_poses(scene)
    stages = STAGES
    if args.stage_plan:
        from scripts.plan_stages import load_bindings
        b = load_bindings(args.stage_plan)
        by_idx = {s.index: s for s in b.plan.stages}   # full parts, both objects
        stages = tuple((by_idx[b.roles[r][1].index], r)
                       for r in ("grasp", "transport_active", "pour"))

    vocab = Vocabulary.from_asset_dirs(asset_dirs)
    symbols = {n: load_symbols(d) for n, d in asset_dirs.items()}
    # per-stage anchor points for the compiler: w's origin on the w-owning
    # object and the mover's feature point — from the call-#2 selections
    # when given (the only source under runtime grounding — authored
    # symbol names do not exist there), else the authored symbols
    if args.selections:
        from manip_sim.selection import (load_selections, selected_points,
                                         selection_objects)
        sels = load_selections(args.selections)
        objects = selection_objects(args.selections, sels)
        if args.stage_plan:
            role_index = {r: st.index for r, (_, st) in b.roles.items()}
        else:
            from scripts.select_frames import ROLES
            role_index = {r: st.index for r, (_, st) in ROLES.items()}
        points = {role: selected_points(role, stage, sels, objects,
                                        role_index, asset_dirs, symbols)
                  for stage, role in stages}
    else:
        if "spout_tip" not in symbols.get("teapot", Symbols("x", {}, {})).points:
            raise SystemExit("[emit] no anchor points: pass --selections "
                             "(runtime grounding has no authored spout_tip)")
        tp, mg = symbols["teapot"].points, symbols["mug"].points
        points = {"grasp": (tp["handle_center"], None),
                  "transport_active": (mg["opening_center"], tp["spout_tip"]),
                  "pour": (mg["opening_center"], tp["spout_tip"])}
    manifest = json.loads(Path(args.frames).read_text()) if args.frames else None
    client = Client()

    emissions, gate, framed = [], [], {}
    # `poses` are ENTRY poses: the compiler freezes the goal attitude on
    # them and the path must admit them. Stage k's mover enters stage k+1
    # at stage k's subgoal center, so chain it; once a stage fails the
    # gate there is no honest entry for the next, so stop there rather
    # than compile it against the spawn pose.
    prev_cs, prev_active = None, None   # last grounded OBJECT-mover stage
    chained: set[str] = set()           # objects whose pose is a chained goal center
    for stage, role in stages:
        sel = views = rec = None
        if manifest is not None:            # framed arm
            views, rec = _framed_inputs(role, manifest)
            sel = sels[role]
            framed[role] = [str(v) for v in views]
        w_point, e_point = points[role]     # keyed by ROLE: stage names are free text
        # factual entry line, both arms: the attitude the compiler freezes
        # the goal on (spawn for the first stage, the chained goal center
        # after an object-mover stage)
        entry = entry_line(stage.active, poses[stage.active],
                           symbols[stage.active].axes,
                           chained=stage.active in chained, tol_rad=ALIGN_TOL_RAD)
        rejections: list[tuple[str, str]] = []   # (raw emission, reason)
        err: dict | None = None
        for attempt in range(1 + COMPILE_RETRIES):
            em = client.emit_constraints(stage, vocab, selection=sel,
                                         view_paths=views,
                                         rejections=rejections, frames=rec,
                                         entry=entry)
            print(f"[emit] stage {em.stage} ({em.name}) attempt {attempt}")
            try:
                cs = compile_stage(em, symbols, poses, w_point=w_point,
                                   e_point=e_point)
                check_pair_consistency(cs)   # subgoal on the path manifold?
                if prev_cs is not None and em.passive and em.active == prev_active:
                    for n_ in check_stage_seam(prev_cs, cs, em):
                        print(f"         {n_}")
            except CompileError as e:
                rejections.append((client.logs[-1].raw, e.text()))
                print(f"         compile rejected: {e.text()}")
                err = {"slot": e.slot, "reason": e.reason,
                       "others": [{"slot": o.slot, "reason": o.reason}
                                  for o in e.others]}
                continue
            for n in cs.notes:
                print(f"         {n}")
            rows = {k: np.round(getattr(cs, k).Bw, 4).tolist()
                    for k in ("path", "subgoal")}
            entry = {n: np.round(T, 6).tolist() for n, T in poses.items()}
            acct = {**stage_accounting(cs, em, symbols, poses),
                    "attempts": attempt + 1,
                    "rejection_slots": [r.split(":")[0] for _, r in rejections],
                    "notes": list(cs.notes)}
            gate.append((em.name, True, rows, None, entry, acct))
            if em.passive:                  # object mover: exits at the subgoal center
                poses = {**poses, em.active: cs.subgoal.nominal()}
                chained.add(em.active)
                prev_cs, prev_active = cs, em.active
            break
        else:
            gate.append((em.name, False, None,
                         {**err, "rejections": [r for _, r in rejections]},
                         {n: np.round(T, 6).tolist() for n, T in poses.items()},
                         {"attempts": 1 + COMPILE_RETRIES,
                          "rejection_slots": [r.split(":")[0] for _, r in rejections]}))
        emissions.append(em)
        if not gate[-1][1]:
            remaining = [st.name for st, _ in stages[len(gate):]]
            if remaining:
                print(f"[emit] stage {em.name} failed the gate; not emitting "
                      f"{remaining} (no entry pose for them)")
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": b.plan.task if args.stage_plan else "pour tea from the teapot into the mug",
        "stage_plan": args.stage_plan,
        "selections": args.selections,
        "frames": args.frames,
        "roles": [r for _, r in stages],
        "arm": "framed" if args.frames else "schema-only",
        "prompt_deltas": list(PROMPT_DELTAS),
        "views": framed,
        "emissions": [asdict(e) for e in emissions],
        "compiled": [{"stage": n, "grounded": ok, "Bw": rows,
                      "error": err, "entry_poses": entry, **acct}
                     for n, ok, rows, err, entry, acct in gate],
        "stages_reached": len(gate),
        "stages_total": len(stages),
    }, indent=2) + "\n")
    log = out.with_suffix(".log.json")
    log.write_text(json.dumps([asdict(l) for l in client.logs], indent=2,
                              default=str) + "\n")
    print(f"[emit] wrote {out} (+ {log})")

    print("\n[emit] compile gate:")
    failed = []
    for name, ok, _, err, _, _ in gate:
        if ok:
            print(f"  PASS  {name}: grounded to B^w")
        else:
            failed.append(name)
            print(f"  FAIL  {name}: {err['slot']}: {err['reason']}")
    flags = [f for l in client.logs for f in l.flags]
    if flags:
        print(f"[emit] {len(flags)} flagged numeric literal(s) in bound "
              f"expressions: {flags}")

    if failed:
        sys.exit(f"[emit] compile gate FAILED for {failed} — slot-named "
                 "CompileErrors above are the typed input the repair "
                 "touchpoint (#5) will consume")
    print("\n[emit] all stages grounded — next: plan_pour_tea.py "
          "--emissions to swap pour_stages.* for these at the planner's "
          "three construction sites")


if __name__ == "__main__":
    main()
