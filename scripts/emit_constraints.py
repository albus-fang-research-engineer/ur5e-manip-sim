"""VLM touchpoint-#3 driver: one live emit_constraints call per pour
stage, gated by the grounding compiler — the closure plan_stages.py's
exit message has been waiting for. The wrapper (Client.emit_constraints),
prompt, parser and its numeric-literal guardrails already exist and are
tested; what has never existed is a driver that makes the calls and
confronts each emission with compile_tsr, so a licitly-parsed schema
that cannot ground (mixed alignment, off-w anchors, empty row
intersections) fails HERE with a slot-named CompileError, not inside
the planner.

Per stage (grasp / transport / pour, the fixed pour-tea structure — as
in select_frames.py, call #1's StagePlan substitutes for this table in
the full orchestrator; call #3 is what is under test):

  1. emit    Client.emit_constraints(StageSpec, vocab[, selection,
             views]) — two-pass mode (--selections) attaches the
             touchpoint-#2 selection for the stage's active role plus
             the marked renders, per the emission-modality ablation's
             two-pass arm; default is single-pass schema-only.
  2. ground  compile_tsr.compile_stage at the canonical upright scene
             attitude (identity body rotations — positions shift only
             T0_w translation, and every rule-table gate is an attitude
             question, so identity poses exercise exactly what an
             offline gate can). Geometric feasibility (IK, non-empty
             subgoal INTERSECT path under the real scene) remains
             plan_pour_tea's job.
  3. report  PASS/FAIL per stage with the compiler's provenance notes
             and the parser's flagged numeric literals.

The artifact (all raw emissions + compiled B^w rows) and the .log.json
(Client.logs: attempts, rejections, flags) are written BEFORE the gate
verdict — a failing emission is evidence, not garbage. Any FAIL exits
nonzero.

Requires ANTHROPIC_API_KEY.

    PYTHONPATH=. python scripts/emit_constraints.py
    PYTHONPATH=. python scripts/emit_constraints.py \
        --selections outputs/selections/pour_tea.json     # two-pass
    PYTHONPATH=. python scripts/emit_constraints.py \
        --out outputs/emissions/other.json

Next (not this script): plan_pour_tea.py --emissions, swapping
pour_stages.* for compile_stage at its three construction sites.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from manip_sim.compile_tsr import CompileError, compile_stage
from manip_sim.frames import load_symbols
from manip_sim.tsr import make_pose
from manip_sim.vlm import Client, StageSpec, Vocabulary
from manip_sim.scene import add_scene_arg, load_scene

OUT = Path("outputs/emissions/pour_tea.json")
VLM_DIR = Path("outputs/candidates/vlm")

# fixed pour-tea stage structure (mirrors select_frames.py's table);
# `role` keys the --selections artifact for the two-pass arm.
STAGES = (
    (StageSpec(index=1, name="grasp", active="teapot", passive=None,
               parts=("handle",)), "grasp"),
    (StageSpec(index=2, name="transport", active="teapot", passive="mug",
               parts=("spout", "rim")), "transport_active"),
    (StageSpec(index=3, name="pour", active="teapot", passive="mug",
               parts=("spout", "rim")), "pour"),
)

# canonical upright scene attitude for the compile gate (see docstring)
POSES = {"teapot": make_pose((0.3, 0.1, 0.05)),
         "mug": make_pose((0.55, -0.2, 0.02))}


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
    views = [Path(v["path"]) for v in manifest.get("views", [])]
    return sel, views or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=None, metavar="JSON",
                    help="touchpoint-#2 artifact; enables the two-pass "
                         "emission arm (selection + marked renders in "
                         "the prompt)")
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    add_scene_arg(ap)
    args = ap.parse_args()
    asset_dirs = load_scene(args.scene).asset_dirs

    vocab = Vocabulary.from_asset_dirs(asset_dirs)
    symbols = {n: load_symbols(d) for n, d in asset_dirs.items()}
    spout_tip = symbols["teapot"].frame("spout_tip", "pour_axis")
    client = Client()

    emissions, gate = [], []
    for stage, role in STAGES:
        sel = views = None
        if args.selections:
            sel, views = _two_pass_inputs(role, Path(args.selections))
        em = client.emit_constraints(stage, vocab, selection=sel,
                                     view_paths=views)
        emissions.append(em)
        print(f"[emit] stage {em.stage} ({em.name}): "
              f"w = frame({em.w_origin}, {em.w_axis})")

        # feature binding for the gate: stages 2-3 pin the spout tip;
        # the grasp TSR constrains the gripper frame directly (Tw_e=I).
        feat = spout_tip if stage.name in ("transport", "pour") else None
        try:
            cs = compile_stage(em, symbols, POSES, e_feature=feat)
            for n in cs.notes:
                print(f"         {n}")
            rows = {k: np.round(getattr(cs, k).Bw, 4).tolist()
                    for k in ("path", "subgoal")}
            gate.append((em.name, True, rows, None))
        except CompileError as e:
            gate.append((em.name, False, None,
                         {"slot": e.slot, "reason": e.reason}))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": "pour tea from the teapot into the mug",
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
