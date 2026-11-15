"""Plan and grade every emission of an authoring-ablation batch, each task with its own
planner, grader and spec. Makes no model calls; run_authoring_ablation.py stays the only
script that does.

    insert_marker   plan_task.py (fixed grasp program)   eval_plan.py   tasks/specs/insert_marker.json
    pour_tea        plan_pour_tea.py                     eval_pour.py   tasks/specs/pour_tea_ablation.json
    (the ablation's specs: pour carry tilt per axis + 1 deg, spout <= 30 deg; insert final
    tilt <= 30 deg, tilt inside the cup reported. The representation study's
    tasks/specs/{pour_tea,insert_marker}.json are unchanged)

Nothing here judges a plan: the graders do. Success is each task's existing definition:
eval_pour.success for pour tea; goal, path and no penetration for insert marker (as in
compare_plans.py).

Each emission is planned once per seed (--seeds, default 0 1 2), so a planner miss is
not read as an authoring failure; the planners have a wall-clock search budget, so
fewer jobs at once (--jobs, default 3) keeps load from deciding outcomes. Each
(emission, seed) writes beside its emission

    <method>.<i>.s<seed>.plan.npz / .plan.out / .plan.eval.json / .eval.out / .score.json

and the batch gets scores.json and scores.md (one table per task, that grader's
columns). Two cache layers, keyed on code CONTENT: a saved plan is reused when the
planner-side code (CODE_PATHS minus the graders and specs) and the emission and input
files (hashed) are unchanged; it is then re-graded only if the graders, this scorer or the
task's spec changed. --tasks / --methods rescore a subset (the rest is read from cache);
--force redoes everything.

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
SPECS = {"insert_marker": "tasks/specs/insert_marker_ablation.json",          # final tilt 30 deg,
         "insert_marker_upright": "tasks/specs/insert_marker_ablation.json",  # path tilt reported
         "pour_tea": "tasks/specs/pour_tea_ablation.json"}          # carry tilt per axis + 1 deg
GRADERS = {"insert_marker": "scripts/eval_plan.py", "insert_marker_upright": "scripts/eval_plan.py",
           "pour_tea": "scripts/eval_pour.py"}
NAMES = {"A0": "A0 framed-chained (default)", "A3": "A3 constrained code",
         "A4": "A4 unconstrained code", "A5": "A5 direct bounds",
         "A4-pilot": "A4-pilot (pilot design: code computed the frames)"}
START_OUTSIDE = ("entry pose lies outside", "entry attitude lies outside",
                 "must admit the entry")
NO_GOAL = ("empty_goal_set", "no feasible goal", "no collision-free")
NO_HANDOFF = ("no_admitted_handoff",)
SEARCH_LIMIT = ("iteration budget exhausted",)   # deterministic for a seed (B1): cached
NO_GRASP = ("no candidate survived", "no reachable grasp", "classifier kept nothing")   # cached


REPO = Path(__file__).resolve().parents[1]
# Two cache layers, each keyed on CONTENT (git's blob hashes), so a docs-only commit, a
# rebase or an amend of identical code keeps every result:
#   plan key  = everything the planners can read: CODE_PATHS minus the graders and specs
#   grade key = the graders, this scorer (success rules) and the task's spec
# A grader or spec change re-grades the saved plans (seconds); only a planner-side change
# re-plans.
CODE_PATHS = ("manip_sim", "scripts", "tasks", "scenes", "assets", "docker", "requirements.txt")
GRADE_FILES = ("scripts/eval_plan.py", "scripts/eval_pour.py", "scripts/score_authoring_ablation.py")


def _git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), "-c", "safe.directory=*", *a],
                          capture_output=True, text=True).stdout.strip()


def _blobs(rev: str) -> dict[str, str]:
    """path -> blob sha for every tracked file under CODE_PATHS at rev."""
    out = {}
    for line in _git("ls-tree", "-r", rev, "--", *CODE_PATHS).splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def _digest(items) -> str:
    import hashlib
    return hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(items)).encode()).hexdigest()[:16]


def _in_plan_layer(path: str) -> bool:
    return path not in GRADE_FILES and not path.startswith("tasks/")


def _invocation() -> str:
    """How this scorer invokes the planners (plan_command, gate_failure, INSERT_GRASP): part
    of the plan key, although this file is otherwise a grade-layer file."""
    import inspect
    return _digest([("plan_command", inspect.getsource(plan_command)),
                    ("gate_failure", inspect.getsource(gate_failure)),
                    ("INSERT_GRASP", INSERT_GRASP)])


def plan_key_at(rev: str) -> str | None:
    """The plan key of the code at rev (None if git or rev is unavailable): what planned."""
    b = _blobs(rev)
    if not b:
        return None
    return "plan-" + _digest([(k, v) for k, v in b.items() if _in_plan_layer(k)]
                             + [("invocation", _invocation())])


def grade_key(spec: str) -> str | None:
    b = _blobs("HEAD")
    if not b:
        return None
    return "grade-" + _digest((k, b.get(k, "missing")) for k in (*GRADE_FILES, spec))


def changed_files() -> list[str]:
    """Tracked files with uncommitted changes, and untracked files, under CODE_PATHS."""
    out = _git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    return [line[3:].split(" -> ")[-1] for line in out.splitlines()]


def dirty(paths) -> bool:
    return any(f == p or f.startswith(p.rstrip("/") + "/") for f in changed_files() for p in paths)


def _legacy_code_key(rev: str) -> str:
    """The 'code-' key that commit 8682d88's scorer recorded, recomputed at rev."""
    import hashlib
    paths = ("manip_sim", "scripts", "tasks", "scenes", "assets", "docker", "requirements.txt")
    objs = [f"{p}:{_git('rev-parse', f'{rev}:{p}')}" for p in paths]
    return "code-" + hashlib.sha256("\n".join(objs).encode()).hexdigest()[:16]


def code_version() -> str:
    """For the tables' header: the plan key, marked dirty when tracked code has
    uncommitted changes."""
    k = plan_key_at("HEAD") or "unknown"
    return k + ("+dirty" if dirty(CODE_PATHS) else "")


def input_hash(task: str, emission: Path) -> str:
    """sha256 over the emission and every input file the planner reads for this
    task (scene, selections, stage plan, the grounding dir's files, followed through
    symlinks): outputs/ is not in code_version()'s key."""
    import hashlib
    t = TASKS[task]
    files = [emission] + [Path(t[k]) for k in ("scene", "selections", "stage_plan") if t[k]]
    if t["grounding"]:
        files += sorted(f for f in Path(t["grounding"]).rglob("*") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(str(f).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def plan_command(task: str, emission: Path, plan: Path, seed: int) -> list[str]:
    t = TASKS[task]
    common = ["--scene", t["scene"], "--selections", t["selections"],
              "--emissions", str(emission), "--seed", str(seed), "--out", str(plan)]
    if task.startswith("insert_marker"):
        grasp = (["--grasp-source", "antipodal"] if t.get("grasp") == "antipodal"
                 else ["--grasp-program", INSERT_GRASP])
        return [sys.executable, "scripts/plan_task.py", *common, "--grounding", t["grounding"],
                "--stage-plan", t["stage_plan"], *grasp]
    return [sys.executable, "scripts/plan_pour_tea.py", *common]


def grade_command(task: str, plan: Path) -> list[str]:
    return [sys.executable, GRADERS[task], "--scene", TASKS[task]["scene"],
            "--spec", SPECS[task], "--plan", str(plan)]


def last_failure(out: Path) -> str:
    """The tool's own failure line, verbatim: its last FAILED line, else its last
    'no feasible' / Error line (detail lines printed after FAILED may contain 'Error')."""
    lines = [l.strip() for l in out.read_text(errors="replace").splitlines() if l.strip()]
    for keys in (("FAILED",), ("no feasible", "Error")):
        hits = [i for i, l in enumerate(lines) if any(k in l for k in keys)]
        if hits:
            i = hits[-1]
            # plan_task prints the reasons histogram on the line after FAILED
            nxt = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].startswith("reasons") else ""
            return (lines[i] + (" " + nxt if nxt else ""))[:600]
    return (lines or ["(no output)"])[-1][:300]


def plan_category(why: str) -> str:
    if any(k in why for k in NO_HANDOFF):
        return "no goal the next stage can start from"
    reasons = re.findall(r"'([^']*)'", why)       # plan_task's per-goal reasons histogram
    if (all(any(k in r for k in SEARCH_LIMIT) for r in reasons) if reasons
            else any(k in why for k in SEARCH_LIMIT)):  # plan_pour_tea: one unquoted reason
        return "planner search limit"                # every goal ran out of iterations
    if any(k in why for k in NO_GRASP):
        return "no usable grasp"
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


def handoff_of(task: str, emission: Path) -> dict:
    """Robot-free handoff consistency, once per emission (manip_sim.spec_scores):
    worst stage pair's rejected fraction, and whether any pair admits none / all."""
    from manip_sim.spec_scores import handoff_scores
    t = TASKS[task]
    try:
        rows = handoff_scores(t["scene"], t["grounding"], t["stage_plan"], t["selections"],
                              str(emission))
    except (Exception, SystemExit) as e:   # the loaders refuse e.g. a gate-failed
        return {"error": f"{type(e).__name__}: {e}"[:200]}   # emission with SystemExit
    pairs = [r for r in rows if "tested" in r and r["tested"]]
    if not pairs:
        return {"pairs": rows}
    return {"pairs": rows,
            "rejected_max": max(r["rejected"] / r["tested"] for r in pairs),
            "none_admitted": any(r["none_admitted"] for r in pairs),
            "all_admitted": all(r["all_admitted"] for r in pairs)}


_LEGACY: dict[str, str | None] = {}


def _old_plan_key(old: dict) -> str | None:
    """The plan key an existing row was planned with: recorded, or (rows from before the
    split) recomputed from the commit sha it recorded."""
    if old.get("plan_key"):
        return old["plan_key"]
    raw = old.get("version") or ""
    v = raw.split("+")[0]
    if "+dirty" in raw or not v:
        return None
    if v not in _LEGACY:
        if re.fullmatch(r"[0-9a-f]{40}", v):
            _LEGACY[v] = plan_key_at(v)
        elif v.startswith("code-"):              # 8682d88's key: find the commit that made it
            revs = _git("log", "-30", "--format=%H").split()
            _LEGACY[v] = next((plan_key_at(r) for r in revs if _legacy_code_key(r) == v), None)
        else:
            _LEGACY[v] = None
    return _LEGACY[v]


def score_one(task: str, emission: Path, seed: int, keys: tuple, inputs: str,
              force: bool) -> dict:
    """keys = (plan_key, grade_key, plan_clean, grade_clean)."""
    plan_k, grade_k, plan_clean, grade_clean = keys
    stem = Path(f"{emission.with_suffix('')}.s{seed}")      # .../A0.0.s1
    plan, plan_out = Path(f"{stem}.plan.npz"), Path(f"{stem}.plan.out")
    ev_json, ev_out = Path(f"{stem}.plan.eval.json"), Path(f"{stem}.eval.out")
    report = Path(f"{stem}.plan.report.json")              # plan_task / plan_pour_tea
    cache = Path(f"{stem}.score.json")
    old = json.loads(cache.read_text()) if cache.exists() and not force else None
    known = plan_k is not None and grade_k is not None
    plan_ok = (known and old is not None and plan_clean and old.get("inputs") == inputs
               and _old_plan_key(old) == plan_k
               and (old["stage"] in ("emission", "plan") or plan.exists()))
    if plan_ok and old["stage"] in ("emission", "plan"):   # labels re-derived: grade layer
        row = old | {"plan_key": plan_k, "grade_key": grade_k, "version": f"{plan_k}/{grade_k}"}
        if old["stage"] == "plan":
            row["category"] = plan_category(old.get("why") or "")
        else:
            row["why"] = gate_failure(json.loads(emission.read_text())) or old.get("why")
        return row
    if plan_ok and grade_clean and old.get("grade_key") == grade_k:
        return old | {"version": f"{plan_k}/{grade_k}"}     # nothing to redo
    prefix, rep = emission.with_suffix("").name.split(".")
    art_doc = json.loads(emission.read_text())
    from manip_sim.code_authoring import LABEL           # medium -> variant (free-code = A4-pilot)
    method = LABEL.get(art_doc.get("medium")) or art_doc.get("method") or prefix   # A0: prefix
    row = {"task": task, "method": method, "rep": int(rep), "seed": seed,
           "emission": str(emission), "inputs": inputs, "plan_key": plan_k, "grade_key": grade_k,
           "version": f"{plan_k}/{grade_k}"}
    if plan_ok:                                          # same plan: re-grade only
        row["grasp_source"] = old.get("grasp_source")
        for f in (ev_json, ev_out):
            f.unlink(missing_ok=True)
    else:
        for f in (plan, plan_out, ev_json, ev_out, report):
            f.unlink(missing_ok=True)
        why = gate_failure(json.loads(emission.read_text()))
        if why:
            row |= {"stage": "emission", "why": why}
        else:
            run(plan_command(task, emission, plan, seed), plan_out)
            rep_doc = json.loads(report.read_text()) if report.exists() else {}
            row["grasp_source"] = (rep_doc.get("grasp") or {}).get("source")
            if not plan.exists():
                why = last_failure(plan_out)
                row |= {"stage": "plan", "category": plan_category(why), "why": why}
    if "stage" not in row:
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
    if known and plan_clean and grade_clean and not transient:
        cache.write_text(json.dumps(row, indent=2) + "\n")
    elif transient and plan_ok:
        pass                       # a re-grade crashed: keep the plan layer's cache; retried next time
    else:                          # uncommitted code, unknown keys or a planning crash: no cache
        cache.unlink(missing_ok=True)
    return row


def detail(task: str, ev: dict) -> dict:
    """The grader's own numbers that its pass/fail is decided on."""
    if task.startswith("insert_marker"):
        return {**ev["goal"], "inserted_max_tilt_deg": ev["inserted"].get("max_tilt_deg")}
    return {"carry_max_axis_tilt_deg": ev["carrying"].get("max_axis_tilt_deg"),   # graded
            "carry_max_tilt_deg": ev["carrying"]["max_tilt_deg"],                  # total, reported
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
        sources = sorted({r["grasp_source"] for r in rs if r.get("grasp_source")})
        out.append(f"## {task} (spec {SPECS[task]}, grader {GRADERS[task]}; "
                   f"planner seeds {seeds}; a run is one emission, the unit of comparison; "
                   "(run, seed) columns are detail, not independent samples; grasp source "
                   + (", ".join(sources) or "unknown")
                   + (" -- MIXED: these rows are not comparable" if len(sources) > 1 else "")
                   + ")\n")
        out.append("| method | runs | succeeded in a majority of seeds | in any seed | "
                   "(run, seed) success | planned | goal | path | collision-free | "
                   "handoff rejected (mean) / runs with a pair admitting none | not planned / plan failures |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
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
            hr = [next((r["handoff_rejected"] for r in mr if r["rep"] == i
                        and r.get("handoff_rejected") is not None), None) for i in runs]
            hr = [h for h in hr if h is not None]
            none_runs = sum(1 for i in runs if next((r["handoff"].get("none_admitted")
                                                     for r in mr if r["rep"] == i), False))
            hr_s = (f"{100 * sum(hr) / len(hr):.0f}%" if hr else "-") + f" / {none_runs}"
            out.append(f"| {NAMES.get(m, m)} | {len(runs)} | {major}/{len(runs)} | {anyok}/{len(runs)} | "
                       f"{c('success')} | {len(g)}/{n} | {c('goal_pass')} | "
                       f"{c('path_pass')} | {c('collision_free')} | {hr_s} | "
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
    ap.add_argument("--tasks", nargs="+", default=None, help="score only these tasks; "
                    "the others are read from their cached rows")
    ap.add_argument("--methods", nargs="+", default=None, help="score only these method "
                    "prefixes (A0 A3 ...); the others are read from their cached rows")
    args = ap.parse_args()
    root = Path("outputs/emissions") / f"authoring-{args.tag}"
    emissions = [(task, e) for task in TASKS for e in sorted((root / task).glob("*.json"))
                 if re.fullmatch(r"[A-Z]\d+\.\d+\.json", e.name)]   # <method>.<i>.json only

    def selected(task, e):
        return ((args.tasks is None or task in args.tasks) and
                (args.methods is None or e.name.split(".")[0] in args.methods))

    version = code_version()
    changed = changed_files()
    plan_k = plan_key_at("HEAD")
    plan_clean = not any(_in_plan_layer(f) for f in changed)
    keys = {task: (plan_k, grade_key(SPECS[task]), plan_clean,
                   not any(f in (*GRADE_FILES, SPECS[task]) for f in changed)) for task in TASKS}
    if changed:
        print(f"[score] uncommitted changes under {CODE_PATHS}: {changed[:5]}"
              f"{' ...' if len(changed) > 5 else ''}; results in the changed layer are not cached")
    for name, picked, valid in (("--tasks", args.tasks, set(TASKS)),
                                ("--methods", args.methods, {e.name.split(".")[0] for _, e in emissions})):
        if picked and not set(picked) <= valid:
            sys.exit(f"[score] {name} {sorted(set(picked) - valid)} not among {sorted(valid)}")
    hashes = {e: input_hash(task, e) for task, e in emissions}
    with ThreadPoolExecutor(args.jobs) as pool:          # robot-free, once per emission
        handoffs = dict(zip([e for _, e in emissions],
                            pool.map(lambda te: handoff_of(*te), emissions)))
    todo = [(task, e, s, hashes[e]) for task, e in emissions for s in args.seeds
            if selected(task, e)]
    kept = []                                            # outside the filter: cached rows
    for task, e in emissions:
        if not selected(task, e):
            for sd in args.seeds:
                c = Path(f"{e.with_suffix('')}.s{sd}.score.json")
                if c.exists():
                    kept.append(json.loads(c.read_text()))
                else:
                    print(f"[score] outside the filter and not cached: {e.name} s{sd} (left out)")
    stale = [r for r in kept if r.get("plan_key") != plan_k
             or r.get("grade_key") != keys[r["task"]][1]]
    if stale:
        print(f"[score] WARNING: {len(stale)} cached rows outside the filter were scored with "
              "other code or specs; rerun without the filter to bring them up to date")
    print(f"[score] {len(emissions)} emissions x {len(args.seeds)} seeds under {root}, "
          f"{args.jobs} at once, code {version}")
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda t: score_one(t[0], t[1], t[2], keys[t[0]], t[3],
                                                 args.force), todo)) + kept
    for r in rows:
        h = handoffs.get(Path(r["emission"]), {})
        r["handoff"] = h
        r["handoff_rejected"] = h.get("rejected_max")
    (root / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = f"# Authoring ablation scores: {args.tag}\n\ncode {version}\n\n" + tables(rows)
    (root / "scores.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
