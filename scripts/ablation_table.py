"""Ablation table from emission artifacts — fields only, no recompile.

    PYTHONPATH=. python scripts/ablation_table.py outputs/emissions/ablation-modality-v3

One row per artifact: gate result, stages reached, attempts per stage,
and for each object-mover stage the goal attitude class of the mover's
canonical axes and the entry->goal sweep axis (as a mover direction).
Artifacts written before step 10 (no accounting fields) print '-'.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def row(path: Path) -> str:
    d = json.load(open(path))
    cells = [path.name.removesuffix(".json").ljust(14),
             f"{d.get('stages_reached', '-')}/{d.get('stages_total', '-')}",
             ",".join(d.get("prompt_deltas", [])) or "-"]
    for c in d["compiled"]:
        if c["stage"] == "grasp":
            continue
        att = c.get("attempts", "-")
        gc = c.get("goal_class")
        sw = c.get("sweep")
        cells.append(f"{c['stage'][:9]}: {'ok' if c['grounded'] else 'FAIL'} "
                     f"a={att} "
                     + (f"front={gc['+front']} up={gc['+up']} sweep={sw['axis']}@{sw['angle_deg']:.0f} "
                        f"x={c.get('x_source', '')}" if gc and sw else "-"))
    return " | ".join(cells)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs/emissions"
    files = sorted(Path(p) for p in glob.glob(f"{root}/*[0-9].json"))
    if not files:
        raise SystemExit(f"no *N.json artifacts under {root}")
    for f in files:
        print(row(f))


if __name__ == "__main__":
    main()
