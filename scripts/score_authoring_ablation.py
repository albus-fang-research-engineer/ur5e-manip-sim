"""Plan and grade every emission of an authoring-ablation batch, each task with its own
planner, grader and spec. Makes no model calls; run_authoring_ablation.py stays the only
script that does.

    insert_marker   plan_task.py (fixed grasp program)   eval_plan.py   tasks/specs/insert_marker.json
    pour_tea        plan_pour_tea.py                     eval_pour.py   tasks/specs/pour_tea.json

Nothing here judges a plan: the graders do. Success is each task's existing definition:
eval_pour.success for pour tea; goal, path and no penetration for insert marker (as in
compare_plans.py).

Each emission is planned once per seed (--seeds, default 0 1 2), so a planner miss is
not read as an authoring failure; the planners have a wall-clock search budget, so
fewer jobs at once (--jobs, default 3) keeps load from deciding outcomes. Each
(emission, seed) writes beside its emission

    <method>.<i>.s<seed>.plan.npz / .plan.out / .plan.eval.json / .eval.out / .score.json

and the batch gets scores.json and scores.md (one table per task, that grader's
columns). A finished (emission, seed) is reused only when its .score.json was made at
the same known commit with no tracked changes and from the same emission and input
files (hashed); anything else, or --force, is redone.

Outcomes beyond the grader's: an emission that failed the gate is not planned
(A0's failures on its grasp stage, which no planner reads, are labeled as such);
a plan failure whose start pose lies outside the next stage's path is counted
apart from other planning failures.

    scripts/authoring_docker.sh python scripts/score_authoring_ablation.py --tag pilot
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_authoring_ablation import METHODS, TASKS  # noqa: E402

INSERT_GRASP = "manip_sim/omnimanip/programs/insert_marker.json"
SPECS = {"insert_marker": "tasks/specs/insert_marker.json", "pour_tea": "tasks/specs/pour_tea.json"}
GRADERS = {"insert_marker": "scripts/eval_plan.py", "pour_tea": "scripts/eval_pour.py"}
NAMES = {"A0": "A0 framed-chained (default)", "A3": "A3 constrained code",
         "A4": "A4 unconstrained code", "A5": "A5 direct bounds",
         "A4-pilot": "A4-pilot (pilot design: code computed the frames)"}
START_OUTSIDE = ("entry pose lies outside", "entry attitude lies outside",
                 "must admit the entry")
NO_GOAL = ("empty_goal_set", "no feasible goal", "no collision-free")


def code_version() -> str:
    """HEAD, marked dirty when tracked files differ: what planned and graded."""
    def git(*a):
        return subprocess.run(["git", "-c", "safe.directory=*", *a], capture_output=True,
                              text=True).stdout.strip()
    head = git("rev-parse", "HEAD") or "unknown"
    return head + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def input_hash(task: str, emission: Path) -> str:
    """sha256 over the emission and every input file the planner reads for
    this task (scene, selections, stage plan, grounding tables): outputs/ is
    untracked, so a changed input would not show in code_version()."""
    import hashlib
    t = TASKS[task]
    files = [emission] + [Path(t[k]) for k in ("scene", "selections", "stage_plan") if t[k]]
    if t["grounding"]:
        files += sorted(f for f in Path(t["grounding"]).rglob("*.json") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(str(f).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


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


def last_failure(out: Path) -> str:
    """The tool's own failure line, verbatim (its last FAILED / 'no feasible' / Error line)."""
    lines = [l.strip() for l in out.read_text(errors="replace").splitlines() if l.strip()]
    hits = [l for l in lines if "FAILED" in l or "no feasible" in l or "Error" in l]
    return (hits or lines or ["(no output)"])[-1][:300]


def plan_category(why: str) -> str:
    if any(k in why for k in START_OUTSIDE):
        return "start outside the next path"
    if any(k in why for k in NO_GOAL):
        return "no reachable goal"
    return "other planning failure"


def run(cmd: list[str], log: Path) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def gate_failure(art: dict) -> str | None:
    """None if every authored stage passed the gate; else what failed."""
    reached, total = art.get("stages_reached"), art.get("stages_total")
    if not isinstance(reached, int) or not isinstance(total, int):
        return "emission artifact has no stage counts"
    bad = next((c for c in art.get("compiled", []) if not c.get("grounded")), None)
    if reached == total and bad is None:
        return None
    bad = bad or {}
    name = bad.get("stage", "?")
    grasp = any(e.get("name") == name and e.get("passive") is None
                for e in art.get("emissions", []))
    return ("gate failed on the grasp stage (unused by the planners)" if grasp
            else f"gate failed on {name!r} ({reached}/{total} stages)")


def success(task: str, ev: dict) -> tuple[bool, bool]:
    """(collision_free, success) by the task's existing definition."""
    if task == "pour_tea":
        from scripts.eval_pour import collision_free, success as pour_success
        return collision_free(ev), pour_success(ev)
    free = not any(ev["max_penetration_mm"].values())        # compare_plans.py
    return free, bool(ev["goal_pass"] and ev["path_pass"] and free)


def score_one(task: str, emission: Path, seed: int, version: str, inputs: str,
              force: bool) -> dict:
    stem = Path(f"{emission.with_suffix('')}.s{seed}")      # .../A0.0.s1
    plan, plan_out = Path(f"{stem}.plan.npz"), Path(f"{stem}.plan.out")
    ev_json, ev_out = Path(f"{stem}.plan.eval.json"), Path(f"{stem}.eval.out")
    cache = Path(f"{stem}.score.json")
    reusable = not force and not version.endswith("+dirty") and not version.startswith("unknown")
    if cache.exists() and reusable:
        old = json.loads(cache.read_text())
        if old.get("version") == version and old.get("inputs") == inputs:
            return old
    prefix, rep = emission.with_suffix("").name.split(".")
    art_doc = json.loads(emission.read_text())
    from manip_sim.code_authoring import LABEL           # medium -> variant (free-code = A4-pilot)
    method = LABEL.get(art_doc.get("medium")) or art_doc.get("method") or prefix   # A0: prefix
    row = {"task": task, "method": method, "rep": int(rep), "seed": seed,
           "emission": str(emission), "version": version, "inputs": inputs}
    for f in (plan, plan_out, ev_json, ev_out):
        f.unlink(missing_ok=True)
    why = gate_failure(json.loads(emission.read_text()))
    if why:
        row |= {"stage": "emission", "why": why}
    else:
        run(plan_command(task, emission, plan, seed), plan_out)
        if not plan.exists():
            why = last_failure(plan_out)
            row |= {"stage": "plan", "category": plan_category(why), "why": why}
        else:
            run(grade_command(task, plan), ev_out)
            if not ev_json.exists():
                row |= {"stage": "grade", "why": last_failure(ev_out)}
            else:
                ev = json.loads(ev_json.read_text())
                free, ok = success(task, ev)
                row |= {"stage": "graded", "goal_pass": bool(ev["goal_pass"]),
                        "path_pass": bool(ev["path_pass"]), "collision_free": free,
                        "success": ok, "eval": str(ev_json), "detail": detail(task, ev)}
    transient = row["stage"] == "grade" or row.get("category") == "other planning failure"
    if not transient:                       # a crash or OOM is retried next time
        cache.write_text(json.dumps(row, indent=2) + "\n")
    else:
        cache.unlink(missing_ok=True)
    return row


def detail(task: str, ev: dict) -> dict:
    """The grader's own numbers that its pass/fail is decided on."""
    if task == "insert_marker":
        return {**ev["goal"], "inserted_max_tilt_deg": ev["inserted"].get("max_tilt_deg")}
    return {"carry_max_tilt_deg": ev["carrying"]["max_tilt_deg"],
            "pour_max_off_axis_mm": ev["pouring"]["max_off_axis_mm"],
            "end_lateral_mm": ev["end"]["lateral_mm"], "end_tilt_deg": ev["end"]["tilt_deg"],
            "spout_off_down_deg": ev["end"]["spout_off_down_deg"]}


def outcome(r: dict) -> str:
    if r["stage"] == "emission":
        return f"not planned: {r['why']}"
    if r["stage"] == "plan":
        return f"plan failed ({r['category']}): {r['why']}"
    if r["stage"] == "grade":
        return f"grader failed: {r['why']}"
    if r["success"]:
        return "success"
    return ", ".join(k for k in ("goal_pass", "path_pass", "collision_free") if not r[k]) + " failed"


def tables(rows: list[dict]) -> str:
    out = []
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        if not rs:
            continue
        seeds = sorted({r["seed"] for r in rs})
        out.append(f"## {task} (spec {SPECS[task]}, grader {GRADERS[task]}; "
                   f"planner seeds {seeds}; a run is one emission, the unit of comparison; "
                   "(run, seed) columns are detail, not independent samples)\n")
        out.append("| method | runs | succeeded in a majority of seeds | in any seed | "
                   "(run, seed) success | planned | goal | path | collision-free | "
                   "not planned / plan failures |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in list(METHODS) + sorted({r["method"] for r in rs} - set(METHODS)):
            mr = [r for r in rs if r["method"] == m]
            if not mr:
                continue
            g = [r for r in mr if r["stage"] == "graded"]
            n = len(mr)
            c = lambda k: f"{sum(r[k] for r in g)}/{n}"
            fails = Counter("gate: " + r["why"] if r["stage"] == "emission"
                            else r["category"] for r in mr if r["stage"] in ("emission", "plan"))
            per_run = Counter()
            for r in g:
                per_run[r["rep"]] += r["success"]
            runs = sorted({r["rep"] for r in mr})
            k = len(seeds)
            major = sum(per_run[i] * 2 > k for i in runs)
            anyok = sum(per_run[i] > 0 for i in runs)
            out.append(f"| {NAMES.get(m, m)} | {len(runs)} | {major}/{len(runs)} | {anyok}/{len(runs)} | "
                       f"{c('success')} | {len(g)}/{n} | {c('goal_pass')} | "
                       f"{c('path_pass')} | {c('collision_free')} | "
                       + ("; ".join(f"{k} x{v}" for k, v in fails.most_common()) or "-") + " |")
        out.append("\n| run | seed | outcome | grader numbers |")
        out.append("|---|---|---|---|")
        for r in sorted(rs, key=lambda r: (r["method"], r["rep"], r["seed"])):
            nums = ", ".join(f"{k} {v}" for k, v in r.get("detail", {}).items())
            out.append(f"| {NAMES.get(r['method'], r['method'])}, run {r['rep']} | {r['seed']} | "
                       f"{outcome(r)} | {nums} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="planner seeds; every emission is planned once per seed")
    ap.add_argument("--jobs", type=int, default=3, help="plans run at once")
    ap.add_argument("--force", action="store_true", help="redo every (emission, seed)")
    args = ap.parse_args()
    root = Path("outputs/emissions") / f"authoring-{args.tag}"
    emissions = [(task, e) for task in TASKS for e in sorted((root / task).glob("*.json"))
                 if re.fullmatch(r"[A-Z]\d+\.\d+\.json", e.name)]   # <method>.<i>.json only
    version = code_version()
    hashes = {e: input_hash(task, e) for task, e in emissions}
    todo = [(task, e, s, hashes[e]) for task, e in emissions for s in args.seeds]
    print(f"[score] {len(emissions)} emissions x {len(args.seeds)} seeds under {root}, "
          f"{args.jobs} at once, code {version}")
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda t: score_one(t[0], t[1], t[2], version, t[3],
                                                 args.force), todo))
    (root / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = f"# Authoring ablation scores: {args.tag}\n\ncode {version}\n\n" + tables(rows)
    (root / "scores.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
