"""Qualify a repair-study fault (FREE: no model call): inject it into the
starting specification, compile and render the preview, then plan and grade
the faulty specification on planner seeds 0-2 (in parallel).

A fault is kept when no stage fails to compile (a compile-time fault is
caught by our compiler in every condition, so it does not separate them)
and the faulty specification passes the grader on 0 of the seeds (it
breaks the task, whether the planner fails or the plan fails the grader).

    scripts/authoring_docker.sh python scripts/qualify_fault.py --task pour_tea \\
        --selections S.json --emissions E.json --fault F --out OUT

Writes OUT/qualify.json (per-stage compile result, per-seed plan and grade,
verdict) and OUT/round_1/images (the renders a critic or checker would see).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from manip_sim import repair_loop as rl
from scripts.run_repair_case import Case, _w

SEEDS = (0, 1, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--emissions", required=True)
    ap.add_argument("--fault", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    case = Case(SimpleNamespace(**vars(a), condition="typed", mode="dry_run", budget=2,
                                call_cap=0, repeat=0, responses=None, critic_memory=False))
    chain, _, _ = case.preview(case.out / "round_1")
    compiled = {c.emission.name: (True if c.compiled is not None else c.error)
                for c in chain if c.emission.passive is not None}
    rec = {"fault": a.fault, "compiled": compiled, "grades": []}
    if any(v is not True for v in compiled.values()):
        rec["verdict"] = "compile-time (not planned): " + "; ".join(
            f"{k}: {v}" for k, v in compiled.items() if v is not True)
    else:
        d = case.write_spec("spec_qualify")
        with ThreadPoolExecutor(len(SEEDS)) as ex:
            rec["grades"] = list(ex.map(
                lambda s: {"seed": s, **rl.plan_and_grade(a.task, d, s, case.stage_names(),
                                                          case.mover_roles())}, SEEDS))
        passed = sum(1 for g in rec["grades"] if g.get("planned") and g.get("success"))
        rec["passed"] = f"{passed}/{len(SEEDS)}"
        rec["verdict"] = ("keep" if passed == 0 else
                          f"drop: the faulty specification passes on {passed} seed(s)")
    _w(case.out / "qualify.json", rec)
    print(f"[qualify] {a.fault}: {rec['verdict']}")
    for g in rec["grades"]:
        print(f"[qualify]   seed {g['seed']}: " + (
            ("planned, grader " + ("pass" if g.get("success") else "fail"))
            if g.get("planned") else f"no plan: {(g.get('failure') or {}).get('type')}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
