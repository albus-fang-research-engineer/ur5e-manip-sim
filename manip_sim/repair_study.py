"""Run records and scoring for the repair study (docs/deferred.md D9). Pure
library: the loop driver writes one record per (fault, condition, repeat);
this module turns records into the per-fault table. No planning, no VLM.

Record (JSON, one file per run):

    fault        fault_injection.FaultRecord (asdict): fault_id, task, category,
                 role, stage, pointer, before, after, expect, unfixable_by
    condition    "typed" | "render_check" | "untyped"
    repeat, mode, model, temperature, call_cap, round_budget, n_calls
    start        paths of the known-good specification's selections / emissions
    start_sha    hash of those two files (sha())
    rounds       one entry per model step, in order:
                   kind       critic | repair | check | regenerate | reemit
                   stage      the stage acted on (check / regenerate), or None
                   calls      [{touchpoint, attempts, input_tokens, output_tokens}]
                   edits      [{target, action, token, before, after, direction}]
                   labels     [{stage, label}] (check)
                   stage_sha  {stage: hash of its emission + point after this step}
                   compiled   {stage: true | error text}
                   planned    null (not planned this step) | {ok, failure}
    final        {accepted, planned_in_loop, grades: [{seed, success}], error}:
                 accepted = the loop ended on an accept / all-success;
                 grades = the final specification planned and graded per seed
                 after the loop (empty if it did not plan)
    diff         fault_injection.spec_diff(start, final): rows / points that
                 differ from the known-good specification

Scoring rules, fixed before any run:
    fixed           the final specification passes the grader in a majority of
                    planner seeds (seeds 0-2). A run that stops early, errors, or
                    spends its budget is "not fixed"; stopping early never counts
                    as fewer calls.
    calls           API requests, parse retries included.
    calls per fix   all calls of a condition on a fault / runs fixed.
    localization    the first step's stage (and, for typed edits, the pointer)
                    equals the fault's.
    collateral      rows / points changed away from the known-good specification
                    other than the fault's own pointer (a re-emitted or
                    regenerated stage counts every changed row).
    oscillation     a target edited again in the opposite direction or to a value
                    it had before, or a stage regenerated to an emission it had
                    before.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .fault_injection import CONDITIONS

OUTCOMES = ("fixed", "planned_grader_failed", "accepted_plan_failed",
            "budget_spent", "error")
N_SEEDS = 3


def sha(*paths_or_text) -> str:
    h = hashlib.sha256()
    for x in paths_or_text:
        h.update(Path(x).read_bytes() if isinstance(x, Path) else str(x).encode())
    return h.hexdigest()[:16]


def load_records(paths) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def n_calls(rec: dict) -> int:
    return sum(c.get("attempts", 1) for r in rec["rounds"] for c in r.get("calls", []))


def n_tokens(rec: dict) -> tuple[int, int]:
    calls = [c for r in rec["rounds"] for c in r.get("calls", [])]
    return (sum(c.get("input_tokens") or 0 for c in calls),
            sum(c.get("output_tokens") or 0 for c in calls))


def outcome(rec: dict) -> str:
    fin = rec.get("final") or {}
    if fin.get("error"):
        return "error"
    grades = fin.get("grades") or []
    if grades:
        wins = sum(bool(g.get("success")) for g in grades)
        return "fixed" if wins * 2 > len(grades) else "planned_grader_failed"
    if fin.get("accepted"):
        return "accepted_plan_failed"
    return "budget_spent"


def first_step(rec: dict) -> dict | None:
    """The first step that changed the specification (an edit, a re-emit or a
    regeneration), or None if the run never changed it."""
    for r in rec["rounds"]:
        if r.get("edits"):
            e = r["edits"][0]
            return {"stage": _stage_of(e["target"], rec), "pointer": e["target"]}
        if r["kind"] in ("regenerate", "reemit") and r.get("stage"):
            return {"stage": r["stage"], "pointer": None}
    return None


def _stage_of(pointer: str, rec: dict) -> str:
    """Stage name of a critic pointer. Stage names may contain dots, so match
    against the stage names the run's rounds recorded."""
    names = {s for r in rec["rounds"] for s in (r.get("stage_sha") or {})}
    names.add(rec["fault"]["stage"])
    hits = [n for n in names if pointer == n or pointer.startswith(n + ".")]
    return max(hits, key=len) if hits else pointer


def localized(rec: dict) -> tuple[bool | None, bool | None]:
    """(stage hit, pointer hit); pointer hit is None when the first step was a
    whole-stage step (no row named)."""
    fs = first_step(rec)
    if fs is None:
        return None, None
    stage_hit = fs["stage"] == rec["fault"]["stage"]
    if fs["pointer"] is None or fs["pointer"] == fs["stage"]:
        return stage_hit, None
    return stage_hit, fs["pointer"] == rec["fault"]["pointer"]


def collateral(rec: dict) -> int:
    return sum(d["pointer"] != rec["fault"]["pointer"] for d in rec.get("diff", []))


def oscillations(rec: dict) -> int:
    n = 0
    seen: dict[str, list[str]] = defaultdict(list)      # target -> values it had
    last_dir: dict[str, int] = {}
    for r in rec["rounds"]:
        for e in r.get("edits", []):
            t = e["target"]
            if e["after"] in seen[t] or \
                    (e.get("direction") and last_dir.get(t) == -e["direction"]):
                n += 1
            seen[t] += [e["before"], e["after"]]
            if e.get("direction"):
                last_dir[t] = e["direction"]
    hist: dict[str, list[str]] = defaultdict(list)       # stage -> emission hashes
    for r in rec["rounds"]:
        for stage, h in (r.get("stage_sha") or {}).items():
            if r["kind"] in ("regenerate", "reemit") and stage == r.get("stage") \
                    and h in hist[stage]:
                n += 1
            if not hist[stage] or hist[stage][-1] != h:
                hist[stage].append(h)
    return n


def summarize(records: list[dict], call_cap: int = 8) -> dict:
    """{(fault_id, condition): cell} plus per-category and per-condition
    totals; a cell is marked structural when the condition cannot fix the
    fault by construction."""
    cells: dict[tuple[str, str], dict] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for rec in records:
        f = rec["fault"]
        groups[(f["fault_id"], rec["condition"])].append(rec)
        meta[f["fault_id"]] = f
    for (fid, cond), recs in groups.items():
        outs = [outcome(r) for r in recs]
        fixed = outs.count("fixed")
        calls = [n_calls(r) for r in recs]
        loc = [localized(r) for r in recs]
        stage_hits = [s for s, _ in loc if s is not None]
        row_hits = [p for _, p in loc if p is not None]
        cells[(fid, cond)] = {
            "fault": fid, "category": meta[fid]["category"], "condition": cond,
            "structural": cond in meta[fid].get("unfixable_by", ()),
            "runs": len(recs), "fixed": fixed,
            "outcomes": {o: outs.count(o) for o in OUTCOMES if outs.count(o)},
            "calls_total": sum(calls), "calls_median": median(calls),
            "calls_per_fix": (sum(calls) / fixed) if fixed else None,
            "tokens_in_out": [sum(n_tokens(r)[0] for r in recs),
                              sum(n_tokens(r)[1] for r in recs)],
            "rounds_median": median(len(r["rounds"]) for r in recs),
            "stage_localized": f"{sum(stage_hits)}/{len(stage_hits)}" if stage_hits else "-",
            "row_localized": f"{sum(row_hits)}/{len(row_hits)}" if row_hits else "-",
            "collateral_median": median(collateral(r) for r in recs),
            "oscillations": sum(oscillations(r) for r in recs),
            "fixed_within": [sum(1 for r, o in zip(recs, outs)
                                 if o == "fixed" and n_calls(r) <= k)
                             for k in range(1, call_cap + 1)],
        }
    return {"cells": list(cells.values())}


def table_md(summary: dict) -> str:
    """Rows: fault (grouped by category); columns: conditions present.
    A cell is "fixed/runs, calls per fix"; structural cells say so."""
    cells = {(c["fault"], c["condition"]): c for c in summary["cells"]}
    conds = [c for c in CONDITIONS if any(k[1] == c for k in cells)]
    faults = sorted({(c["category"], c["fault"]) for c in summary["cells"]})
    head = "| category | fault | " + " | ".join(conds) + " |"
    lines = [head, "|" + "---|" * (2 + len(conds))]
    for cat, fid in faults:
        row = []
        for cond in conds:
            c = cells.get((fid, cond))
            if c is None:
                row.append("-")
                continue
            per = f"{c['calls_per_fix']:.1f}" if c["calls_per_fix"] else "-"
            s = f"{c['fixed']}/{c['runs']}, {per} calls/fix"
            row.append(s + (" (cannot fix by construction)" if c["structural"] else ""))
        lines.append(f"| {cat} | {fid} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("A cell is: runs fixed / runs, then VLM calls per repaired specification "
                 "(all calls / runs fixed). Fixed = passes the grader in a majority of "
                 f"{N_SEEDS} planner seeds; stopping early counts as not fixed.")
    return "\n".join(lines)
