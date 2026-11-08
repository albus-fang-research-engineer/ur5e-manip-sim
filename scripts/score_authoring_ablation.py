"""Plan and grade every emission of an authoring-ablation batch, each task with its own
planner, grader and spec. Makes no model calls; run_authoring_ablation.py stays the only
script that does.

    insert_marker   plan_task.py (fixed grasp program)   eval_plan.py   tasks/specs/insert_marker.json
    pour_tea        plan_pour_tea.py                     eval_pour.py   tasks/specs/pour_tea.json

Nothing here judges a plan: the graders do, and "success" is each task's existing
definition (eval_pour.success for pour tea; goal, path and no penetration for insert
marker, as in compare_plans.py). Each run's files land beside its emission:

    <method>.<i>.plan.npz / .plan.out / .plan.eval.json / .eval.out

and the batch gets scores.json plus scores.md (one table per task, that grader's columns).
A run whose .plan.eval.json, or whose failed .plan.out, already exists is not redone.

    scripts/authoring_docker.sh python scripts/score_authoring_ablation.py --tag pilot
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_authoring_ablation import METHODS, TASKS  # noqa: E402

INSERT_GRASP = "manip_sim/omnimanip/programs/insert_marker.json"
SPECS = {"insert_marker": "tasks/specs/insert_marker.json", "pour_tea": "tasks/specs/pour_tea.json"}
GRADERS = {"insert_marker": "scripts/eval_plan.py", "pour_tea": "scripts/eval_pour.py"}
NAMES = {"A0": "A0 framed-chained (default)", "A3": "A3 constrained code",
         "A4": "A4 unconstrained code"}
COLLISION_TOL_MM = 1.0      # eval_pour.COLLISION_TOL_MM; insert marker uses "no penetration"


def plan_command(task: str, emission: Path, plan: Path, seed: int) -> list[str]:
    t = TASKS[task]
    common = ["--scene", t["scene"], "--selections", t["selections"],
              "--emissions", str(emission), "--seed", str(seed), "--out", str(plan)]
    if task == "insert_marker":
        return [sys.executable, "scripts/plan_task.py", *common, "--grounding", t["grounding"],
                "--stage-plan", t["stage_plan"], "--grasp-program", INSERT_GRASP]
    return [sys.executable, "scripts/plan_pour_tea.py", *common]


def grade_command(task: str, plan: Path) -> list[str]:
    return [sys.executable, GRADERS[task], "--scene", TASKS[task]["scene"],
            "--spec", SPECS[task], "--plan", str(plan)]


def plan_failure(out: Path) -> str:
    """The planner's own failure line, verbatim (its last FAILED / 'no feasible' line)."""
    lines = [l.strip() for l in out.read_text(errors="replace").splitlines() if l.strip()]
    hits = [l for l in lines if "FAILED" in l or "no feasible" in l or "Error" in l]
    return (hits or lines or ["(no output)"])[-1][:300]


def run(cmd: list[str], log: Path) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def score_one(task: str, emission: Path, seed: int) -> dict:
    stem = emission.with_suffix("")                       # .../A0.0
    plan, plan_out = Path(f"{stem}.plan.npz"), Path(f"{stem}.plan.out")
    ev_json, ev_out = Path(f"{stem}.plan.eval.json"), Path(f"{stem}.eval.out")
    method, rep = stem.name.split(".")
    row = {"task": task, "method": method, "rep": int(rep), "emission": str(emission)}
    art = json.loads(emission.read_text())
    if art.get("stages_reached") != art.get("stages_total"):
        return {**row, "stage": "emission", "why": f"gate passed {art.get('stages_reached')}"
                f"/{art.get('stages_total')} stages"}
    if not ev_json.exists():
        if not plan.exists():
            if not plan_out.exists():
                run(plan_command(task, emission, plan, seed), plan_out)
            if not plan.exists():
                return {**row, "stage": "plan", "why": plan_failure(plan_out)}
        run(grade_command(task, plan), ev_out)
        if not ev_json.exists():
            return {**row, "stage": "grade", "why": plan_failure(ev_out)}
    ev = json.loads(ev_json.read_text())
    pen = max(ev["max_penetration_mm"].values()) if ev["max_penetration_mm"] else 0.0
    free = pen <= COLLISION_TOL_MM if task == "pour_tea" else pen == 0.0
    return {**row, "stage": "graded", "goal_pass": ev["goal_pass"], "path_pass": ev["path_pass"],
            "collision_free": free, "success": ev["goal_pass"] and ev["path_pass"] and free,
            "eval": str(ev_json), "detail": detail(task, ev)}


def detail(task: str, ev: dict) -> dict:
    """The grader's own numbers that its pass/fail is decided on."""
    if task == "insert_marker":
        return {**ev["goal"], "inserted_max_tilt_deg": ev["inserted"].get("max_tilt_deg")}
    return {"carry_max_tilt_deg": ev["carrying"]["max_tilt_deg"],
            "pour_max_off_axis_mm": ev["pouring"]["max_off_axis_mm"],
            "end_lateral_mm": ev["end"]["lateral_mm"], "end_tilt_deg": ev["end"]["tilt_deg"],
            "spout_off_down_deg": ev["end"]["spout_off_down_deg"]}


def tables(rows: list[dict]) -> str:
    out = []
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        if not rs:
            continue
        out.append(f"## {task} (spec {SPECS[task]}, grader {GRADERS[task]})\n")
        out.append("| method | success | planned | goal | path | collision-free |")
        out.append("|---|---|---|---|---|---|")
        for m in METHODS:
            mr = [r for r in rs if r["method"] == m]
            if not mr:
                continue
            g = [r for r in mr if r["stage"] == "graded"]
            n = len(mr)
            c = lambda k: f"{sum(r[k] for r in g)}/{n}"
            out.append(f"| {NAMES[m]} | {c('success')} | {len(g)}/{n} | {c('goal_pass')} | "
                       f"{c('path_pass')} | {c('collision_free')} |")
        out.append("\n| run | outcome | grader numbers |")
        out.append("|---|---|---|")
        for r in sorted(rs, key=lambda r: (r["method"], r["rep"])):
            if r["stage"] != "graded":
                out.append(f"| {NAMES[r['method']]}, run {r['rep']} | {r['stage']} failed: {r['why']} | |")
            else:
                verdict = "success" if r["success"] else ", ".join(
                    k for k in ("goal_pass", "path_pass", "collision_free") if not r[k]) + " failed"
                out.append(f"| {NAMES[r['method']]}, run {r['rep']} | {verdict} | "
                           + ", ".join(f"{k} {v}" for k, v in r["detail"].items()) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0, help="planner seed, the same for every run")
    ap.add_argument("--jobs", type=int, default=6, help="runs planned at once")
    args = ap.parse_args()
    root = Path("outputs/emissions") / f"authoring-{args.tag}"
    todo = [(task, e) for task in TASKS for e in sorted((root / task).glob("*.json"))
            if re.fullmatch(r"[A-Z]\d+\.\d+\.json", e.name)]   # <method>.<i>.json only
    print(f"[score] {len(todo)} emissions under {root}, seed {args.seed}, {args.jobs} at once")
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda te: score_one(*te, args.seed), todo))
    (root / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = f"# Authoring ablation scores: {args.tag}\n\n" + tables(rows)
    (root / "scores.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
