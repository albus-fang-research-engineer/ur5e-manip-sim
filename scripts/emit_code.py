"""Call #3 in a code or selection medium (authoring ablation A3 / A4 / A5): one live call per
object-mover stage, gated the way emit_constraints.py gates the schema.

    --medium constrained-code   A3: author() over the helper API; w, anchors
                                and feature as in the schema; rotations via
                                points_to / perpendicular_to only; numeric
                                magnitudes. Compiled by compile_stage.
    --medium bounds-code        A4: author(scene) computing B^w from the grounded
                                values; w and e are given (the canonical
                                frames at call #2's points, as the schema's).
    --medium bounds-json        A5: the same B^w written directly as one JSON
                                object (no code).
    --medium free-code          the pilot's A4 (code computed T0_w / Tw_e);
                                kept to re-plan the pilot artifacts only.

Same inputs as the schema run it is compared with: the stage plan, the
call-#2 selections (anchor points), the entry line, and with
--render-chained the same renders and legend at the chained entry pose
(the schema's default modality). Same budgets: the client's parse retries
(a rejected code block) and COMPILE_RETRIES re-emissions on a gate failure
(a run error or a CompileError), each fed back verbatim. Same chaining: the
next stage enters at this stage's subgoal center; a stage that fails the
gate stops the run.

The grasp stage (no passive object) is not authored: plan_task.py and
plan_pour_tea.py take the grasp from the grasp stand-in or --grasp-program
and never read an emitted grasp TSR, so a call there would buy nothing.
The artifact lists it under "skipped".

The artifact keeps the code verbatim (the authored object), the compiled
B^w at the nominal entry, the gate verdicts, the run settings and the
git state; .log.json keeps every call's attempts and rejections. Then:

    plan_task.py --emissions <artifact>        (re-runs the code at the
                                               actual entry pose)

Requires ANTHROPIC_API_KEY. Each stage costs 1-3 calls (more with parse
retries).

    PYTHONPATH=. python scripts/emit_code.py --medium constrained-code \\
        --scene scenes/insert_marker.json --grounding outputs/grounding/insert_marker_gt_opus1 \\
        --stage-plan outputs/runs/gt_opus1/stage_plan.json \\
        --selections outputs/runs/gt_opus1/selections.json \\
        --render-chained outputs/frames/code_a3 --out outputs/emissions/code/a3.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from manip_sim import code_authoring as ca
from manip_sim.compile_tsr import (ALIGN_TOL_RAD, CompileError, check_pair_consistency,
                                   stage_accounting)
from manip_sim.frames import load_symbols
from manip_sim.scene import add_scene_arg, load_scene
from manip_sim.vlm import TEMPERATURE as VLM_TEMPERATURE
from manip_sim.vlm import Client, VLMError, Vocabulary, entry_line
from scripts.emit_constraints import (COMPILE_RETRIES, STAGES, _point_token,
                                      _spawn_poses, bound_stages)


def git_state() -> dict:
    """Commit and the tracked files changed since it: what produced this."""
    def run(*a):
        return subprocess.run(["git", "-c", "safe.directory=*", *a], capture_output=True,
                              text=True).stdout.strip()
    commit = run("rev-parse", "HEAD")
    if not commit:          # e.g. a container without the repository's .git mounted
        return {"commit": None, "note": "git unavailable; run via scripts/authoring_docker.sh"}
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("status", "--porcelain", "--untracked-files=no").splitlines()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medium", required=True, choices=ca.MEDIA)
    ap.add_argument("--selections", required=True, metavar="JSON")
    ap.add_argument("--stage-plan", default=None, metavar="JSON",
                    help="plan_stages.py artifact (default: the fixed pour-tea stages)")
    ap.add_argument("--render-chained", default=None, metavar="DIR",
                    help="attach the stage renders + legend at the chained entry pose, "
                         "as the schema's framed-chained run does; omit for text only")
    ap.add_argument("--out", required=True, metavar="JSON")
    add_scene_arg(ap)
    args = ap.parse_args()

    scene = load_scene(args.scene, getattr(args, "grounding", None))
    asset_dirs = scene.asset_dirs
    poses = _spawn_poses(scene)
    stages, synthesized = STAGES, frozenset()
    from manip_sim.selection import load_selections, selected_points, selection_objects
    sels = load_selections(args.selections)
    objects = selection_objects(args.selections, sels)
    if args.stage_plan:
        from scripts.plan_stages import load_bindings
        b = load_bindings(args.stage_plan)
        synthesized = b.entry_synthesized
        stages = bound_stages(b, {s.index: s for s in b.plan.stages})
        role_index = {r: st.index for r, (_, st) in b.roles.items()}
        task = b.plan.task
    else:
        from scripts.select_frames import ROLES
        role_index = {r: st.index for r, (_, st) in ROLES.items()}
        task = "pour tea from the teapot into the mug"
    vocab = Vocabulary.from_asset_dirs(asset_dirs)
    symbols = {n: load_symbols(d) for n, d in asset_dirs.items()}
    points = {role: selected_points(role, stage, sels, objects, role_index,
                                    asset_dirs, symbols) for stage, role in stages}
    if args.render_chained:
        from scripts.render_stage_frames import compute_stage_frames, render_role, scene_extent
        render_dir = Path(args.render_chained)
    client = Client(max_tokens=ca.MAX_TOKENS)

    emissions, compiled, roles, skipped, views_by_role = [], [], [], [], {}
    prev_cs, prev_active, chained, entry_from = None, None, set(), "spawn"
    failed = None
    for stage, role in stages:
        if stage.passive is None:
            skipped.append({"role": role, "name": stage.name,
                            "why": "grasp stage: the planners take the grasp from the "
                                   "grasp stand-in or --grasp-program"})
            continue
        views = rec = None
        if args.render_chained:
            _, radius = scene_extent(scene, poses)
            sf = compute_stage_frames(((stage, role),), sels, objects, role_index,
                                      asset_dirs, symbols, poses, radius)[role]
            rec = render_role(scene, role, sf, poses, render_dir, entry_from)
            views = [Path(rec["views"][v]) for v in rec["views"]]
            views_by_role[role] = [str(v) for v in views]
        w_point, e_point = points[role]
        synth = role in synthesized
        e_tok = _point_token(stage.active, symbols, e_point) or f"{stage.active}.anchor"
        w_tok = _point_token(stage.passive, symbols, w_point) or f"{stage.passive}.anchor"
        entry = None if synth else entry_line(
            stage.active, poses[stage.active], symbols[stage.active].axes,
            chained=stage.active in chained, tol_rad=ALIGN_TOL_RAD,
            anchor=(e_tok, e_point), w_anchor=(w_tok, w_point, poses[stage.passive]))
        facts = ca.stage_facts(stage.active, stage.passive, symbols, poses, w_point, e_point)
        rejections: list[tuple[str, str]] = []
        err, em, attempts = None, None, 0
        for attempt in range(1 + COMPILE_RETRIES):
            attempts = attempt + 1
            try:
                # the selection id goes with the renders, as in the schema run
                em = ca.emit_code(client, stage, vocab, args.medium, facts,
                                  selection=sels[role] if views else None,
                                  view_paths=views, frames=rec,
                                  entry=entry, rejections=rejections)
            except VLMError as e:
                if "parse retry budget exhausted" not in str(e):
                    raise                  # transport / API: no result, rerun later
                err = {"slot": "parse", "reason": str(e), "others": []}
                break                      # the model never wrote valid code
            print(f"[emit-code] {role} ({stage.name}) attempt {attempt}")
            try:
                cs = ca.compile_any(em, symbols, poses, w_point=w_point, e_point=e_point,
                                    entry_synthesized=synth, vocab=vocab)
                check_pair_consistency(cs)
                notes = list(cs.notes)
                if prev_cs is not None and em.active == prev_active:
                    notes += ca.check_seam(prev_cs, cs)
            except CompileError as e:
                rejections.append((client.logs[-1].raw, e.text()))
                client.logs[-1].gate_rejection = e.text()
                print(f"         rejected: {e.text()}")
                err = {"slot": e.slot, "reason": e.reason,
                       "others": [{"slot": o.slot, "reason": o.reason} for o in e.others]}
                continue
            for n in notes:
                print(f"         {n}")
            acct = stage_accounting(cs, em, symbols, poses)
            compiled.append({
                "stage": em.name, "role": role, "grounded": True, "error": None,
                "Bw": {k: np.round(getattr(cs, k).Bw, 4).tolist() for k in ("path", "subgoal")},
                "T0_w": {k: np.round(getattr(cs, k).T0_w, 6).tolist() for k in ("path", "subgoal")},
                "Tw_e": {k: np.round(getattr(cs, k).Tw_e, 6).tolist() for k in ("path", "subgoal")},
                "entry_poses": {n: np.round(T, 6).tolist() for n, T in poses.items()},
                "attempts": attempt + 1, "notes": notes,
                "rejections": [r for _, r in rejections], **acct,
                **({"entry_from": "synthesized"} if synth else {})})
            poses = {**poses, em.active: cs.subgoal.nominal()}
            chained.add(em.active)
            entry_from = f"{em.name}.subgoal.nominal"
            prev_cs, prev_active = cs, em.active
            break
        if em is not None:              # a failing emission is evidence too
            emissions.append(em)
            roles.append(role)
        if not compiled or compiled[-1]["role"] != role:
            compiled.append({"stage": stage.name, "role": role, "grounded": False,
                             "error": err, "Bw": None,
                             "entry_poses": {n: np.round(T, 6).tolist() for n, T in poses.items()},
                             "attempts": attempts,
                             "rejections": [r for _, r in rejections]})
            failed = role
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    movers = [r for st, r in stages if st.passive is not None]
    out.write_text(json.dumps({
        "task": task, "method": ca.LABEL[args.medium], "medium": args.medium,
        "stage_plan": args.stage_plan, "selections": args.selections,
        "scene": args.scene, "grounding": getattr(args, "grounding", None),
        "views_mode": "framed-chained" if args.render_chained else "text-only",
        "render_dir": str(render_dir) if args.render_chained else None,
        "model": client.model, "temperature": VLM_TEMPERATURE,
        "max_tokens": client.max_tokens, "prompt_version": ca.PROMPT_VERSIONS[args.medium],
        "arm": "framed-chained" if args.render_chained else "text-only",  # legacy modality key plan_task reports
        "git": git_state(), "roles": roles, "skipped": skipped, "views": views_by_role,
        "emissions": [ca.as_record(e) for e in emissions], "compiled": compiled,
        # A0's semantics: every stage counts, the unauthored grasp as reached
        "stages_reached": len(compiled) + len(skipped), "stages_total": len(stages),
        "mover_stages_reached": len(compiled), "mover_stages_total": len(movers),
    }, indent=2) + "\n")
    log = out.with_suffix(".log.json")
    log.write_text(json.dumps([vars(l) for l in client.logs], indent=2, default=str) + "\n")
    print(f"[emit-code] wrote {out} (+ {log})")
    for c in compiled:
        print(f"  {'PASS' if c['grounded'] else 'FAIL'}  {c['role']}"
              + ("" if c["grounded"] else f": {c['error']['slot']}: {c['error']['reason']}"))
    if failed:
        sys.exit(f"[emit-code] gate failed at {failed}")


if __name__ == "__main__":
    main()
