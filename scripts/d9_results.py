"""D9 results table (free): every current-loop run record -> one table per fault
and condition (runs fixed / runs, VLM calls per run), grouped by fault class, plus
a CSV. To append new runs, add their folder to RUN_GLOBS (and a new fault to
FAULTS); rerun and paste the table into docs/d9_status.md.

    python3 scripts/d9_results.py            # prints the markdown table
    -> outputs/repair_study/results_current_loop.csv

Current loop = upstream-first rule + the word-ranges call #3 prompt, critic memory
off (records with critic_memory true are skipped: they are a separate check).
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manip_sim import repair_study as rs  # noqa: E402

RUN_GLOBS = (
    "outputs/repair_study/scale/set*/*/*/record.json",   # 2026-11-15: shared + Langfold-only
    "outputs/repair_study/angle/r*/*/*/record.json",     # 2026-11-18: angle (OmniManip's class)
)
CSV_OUT = Path("outputs/repair_study/results_current_loop.csv")

# fault id -> (group, row label); groups print in this order
GROUPS = ("shared", "angle", "Langfold-only")
GROUP_TITLE = {
    "shared": "Shared: OmniManip makes it, its retry cannot repair it (point or distance)",
    "angle": "Angle: OmniManip makes it and its retry is built to repair it (direction)",
    "Langfold-only": "Langfold-only: exists because Langfold's specifications are richer",
}
FAULTS = {
    "pour_wrong_anchor_both": ("shared", "wrong anchor, both stages"),
    "pour_wrong_anchor": ("shared", "wrong anchor, carry only"),
    "pour_height_contact": ("shared", "height contact"),
    "pour_height_small": ("shared", "height small"),
    "pour_clearance_contact": ("shared", "clearance contact"),
    "pour_spout_up": ("angle", "spout up (pot tipped backward)"),
    "pour_no_tilt": ("angle", "no tilt (spout held level)"),
    "pour_handoff_high": ("Langfold-only", "handoff"),
    "pour_contradictory_upright": ("Langfold-only", "contradictory rows"),
}
CONDITIONS = (("typed", "Langfold"), ("untyped", "Withheld"),
              ("render_check", "OmniManip-style"))


def load() -> dict:
    cells = defaultdict(list)
    for pattern in RUN_GLOBS:
        for p in sorted(glob.glob(pattern)):
            r = json.loads(Path(p).read_text())
            if r.get("critic_memory") or not r.get("fault"):
                continue
            fid = r["fault"]["fault_id"]
            if fid not in FAULTS:
                raise SystemExit(f"{p}: fault {fid} has no entry in FAULTS")
            cells[(fid, r["condition"])].append((rs.outcome(r) == "fixed", rs.n_calls(r)))
    return cells


def _cell(runs) -> tuple[str, str]:
    if not runs:
        return "-", "-"
    calls = sum(c for _, c in runs) / len(runs)
    return f"{sum(f for f, _ in runs)}/{len(runs)}", f"{calls:.1f}".rstrip("0").rstrip(".")


def table(cells) -> str:
    head = ["Fault"] + [f"{n} {k}" for _, n in CONDITIONS for k in ("fixed", "calls")]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    everything = defaultdict(list)
    for g in GROUPS:
        out.append(f"| *{GROUP_TITLE[g]}* |" + " |" * (len(head) - 1))
        total = defaultdict(list)
        for fid, (grp, label) in FAULTS.items():
            if grp != g:
                continue
            row = [label]
            for c, _ in CONDITIONS:
                runs = cells.get((fid, c), [])
                total[c] += runs
                everything[c] += runs
                row += _cell(runs)
            out.append("| " + " | ".join(row) + " |")
        out.append("| " + " | ".join([f"**{g} total**"] + [f"**{x}**" for c, _ in CONDITIONS
                                                            for x in _cell(total[c])]) + " |")
    out.append("| " + " | ".join(["**All faults**"] + [f"**{x}**" for c, _ in CONDITIONS
                                                       for x in _cell(everything[c])]) + " |")
    return "\n".join(out)


def write_csv(cells) -> None:
    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fault", "group", "label", "condition", "runs", "fixed", "calls_total",
                    "calls_per_run"])
        for fid, (grp, label) in FAULTS.items():
            for c, _ in CONDITIONS:
                runs = cells.get((fid, c), [])
                if runs:
                    calls = sum(n for _, n in runs)
                    w.writerow([fid, grp, label, c, len(runs), sum(x for x, _ in runs), calls,
                                round(calls / len(runs), 2)])


if __name__ == "__main__":
    cells = load()
    write_csv(cells)
    print(table(cells))
    print(f"\n-> {CSV_OUT}")
