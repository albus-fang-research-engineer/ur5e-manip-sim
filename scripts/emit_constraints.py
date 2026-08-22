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
             views]) — two-pass mode (--selections) attaches the
             touchpoint-#2 selection for the stage's active role plus
             the marked renders, per the emission-modality ablation's
             two-pass arm; default is single-pass schema-only.
  2. ground  compile_tsr.compile_stage at the manifest's spawn poses
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
        --selections outputs/selections/pour_tea.json     # two-pass
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

from manip_sim.compile_tsr import CompileError, compile_stage
from manip_sim.frames import Symbols, load_symbols
from manip_sim.vlm import Client, StageSpec, Vocabulary
from manip_sim.scene import add_scene_arg, load_scene

COMPILE_RETRIES = 2      # re-emissions per stage on a CompileError (stopgap
                         # for touchpoint #5); attempts = 1 + retries

OUT = Path("outputs/emissions/pour_tea.json")
VLM_DIR = Path("outputs/candidates/vlm")

# fixed pour-tea stage structure (mirrors select_frames.py's table);
# `role` keys the --selections artifact for the two-pass arm.
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


def _two_pass_inputs(role: str, sel_path: Path):
    """(PointAxisSelection, view paths) for a stage's active role from a
    touchpoint-#2 artifact + the --vlm render manifest, or (None, None)
    with a warning when either is missing."""
    from manip_sim.vlm import PointAxisSelection
    sels = json.loads(sel_path.read_text())
    if role not in sels:
        print(f"[emit] WARNING: role {role!r} absent from {sel_path}; "
              "falling back to single-pass for this stage")
        return None, None
    d = sels[role]
    sel = PointAxisSelection(
        candidate_id=d["candidate_id"], axis=d["axis"], sign=d["sign"],
        secondary=d.get("secondary"), rationale=d.get("rationale", ""))
    obj = d.get("object") or ("mug" if role == "transport_passive"
                              else "teapot")
    mpath = VLM_DIR / obj / "manifest.json"
    if not mpath.exists():
        print(f"[emit] WARNING: no render manifest {mpath}; "
              "two-pass views unavailable, sending selection only")
        return sel, None
    manifest = json.loads(mpath.read_text())
    # manifest["views"] is {view_name: {path, visible_ids}} (render_candidates)
    views = [Path(v["path"]) for _, v in sorted(manifest.get("views", {}).items())]
    return sel, views or None


def _selected_points(role: str, stage: StageSpec, sels: dict,
                     role_index: dict[str, int], asset_dirs: dict,
                     symbols: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """(w_point, e_point) for a stage from the role-keyed call-#2
    selections: w's origin is the point most recently selected on the
    w-owning object at or before this stage (the passive's interaction
    point, e.g. the mug opening serves transport and pour alike); the
    feature is this role's own selection when it lies on the mover
    (None when the mover is the gripper)."""
    from manip_sim.selection import load_pool, resolve_selection
    w_obj = stage.passive or stage.active
    mover = stage.active if stage.passive else None

    def obj_of(r):
        return sels[r].axis.partition(".")[0]

    def point(r):
        o = obj_of(r)
        return resolve_selection(sels[r], load_pool(asset_dirs[o]),
                                 symbols[o]).frame.point

    k = role_index[role]
    on_w = [r for r in sels if obj_of(r) == w_obj and role_index.get(r, -1) <= k]
    if not on_w:
        raise SystemExit(f"[emit] no call-#2 selection on {w_obj!r} at or "
                         f"before stage {k} to root w on (roles {sorted(sels)})")
    w_role = max(on_w, key=lambda r: (role_index[r], r == role))
    e_point = point(role) if mover and obj_of(role) == mover else None
    return point(w_role), e_point


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=None, metavar="JSON",
                    help="touchpoint-#2 artifact; enables the two-pass "
                         "emission arm (selection + marked renders in "
                         "the prompt)")
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    ap.add_argument("--stage-plan", default=None, metavar="JSON",
                    help="plan_stages.py artifact; the stages bound to the "
                         "grasp / transport_active / pour roles replace STAGES")
    add_scene_arg(ap)
    args = ap.parse_args()
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
        from manip_sim.selection import load_selections
        sels = load_selections(args.selections)
        if args.stage_plan:
            role_index = {r: st.index for r, (_, st) in b.roles.items()}
        else:
            from scripts.select_frames import ROLES
            role_index = {r: st.index for r, (_, st) in ROLES.items()}
        points = {role: _selected_points(role, stage, sels, role_index,
                                         asset_dirs, symbols)
                  for stage, role in stages}
    else:
        if "spout_tip" not in symbols.get("teapot", Symbols("x", {}, {})).points:
            raise SystemExit("[emit] no anchor points: pass --selections "
                             "(runtime grounding has no authored spout_tip)")
        tp, mg = symbols["teapot"].points, symbols["mug"].points
        points = {"grasp": (tp["handle_center"], None),
                  "transport_active": (mg["opening_center"], tp["spout_tip"]),
                  "pour": (mg["opening_center"], tp["spout_tip"])}
    client = Client()

    emissions, gate = [], []
    for stage, role in stages:
        sel = views = None
        if args.selections:
            sel, views = _two_pass_inputs(role, Path(args.selections))
        w_point, e_point = points[role]     # keyed by ROLE: stage names are free text
        rejections: list[tuple[str, str]] = []   # (raw emission, reason)
        err: dict | None = None
        for attempt in range(1 + COMPILE_RETRIES):
            em = client.emit_constraints(stage, vocab, selection=sel,
                                         view_paths=views,
                                         rejections=rejections)
            print(f"[emit] stage {em.stage} ({em.name}) attempt {attempt}")
            try:
                cs = compile_stage(em, symbols, poses, w_point=w_point,
                                   e_point=e_point)
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
            gate.append((em.name, True, rows, None))
            break
        else:
            gate.append((em.name, False, None,
                         {**err, "rejections": [r for _, r in rejections]}))
        emissions.append(em)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": b.plan.task if args.stage_plan else "pour tea from the teapot into the mug",
        "stage_plan": args.stage_plan,
        "selections": args.selections,
        "roles": [r for _, r in stages],
        "arm": "two-pass" if args.selections else "schema-only",
        "emissions": [asdict(e) for e in emissions],
        "compiled": [{"stage": n, "grounded": ok, "Bw": rows,
                      "error": err}
                     for n, ok, rows, err in gate],
    }, indent=2) + "\n")
    log = out.with_suffix(".log.json")
    log.write_text(json.dumps([asdict(l) for l in client.logs], indent=2,
                              default=str) + "\n")
    print(f"[emit] wrote {out} (+ {log})")

    print("\n[emit] compile gate:")
    failed = []
    for name, ok, _, err in gate:
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
