"""Authoring ablation, call #3 only: tasks x methods x repetitions, on frozen
call #1 / #2 artifacts, one emission artifact per run.

    A0   the token schema         scripts/emit_constraints.py --render-chained
    A3   constrained code         scripts/emit_code.py --medium constrained-code
    A4   free code                scripts/emit_code.py --medium free-code

Every method sees the same stage plan, selections, entry line, renders and
legend (framed-chained), and the same model and temperature, set here and
recorded in each artifact. Runs go to

    outputs/emissions/authoring-<tag>/<task>/<method>.<i>.json   (+ .log.json, .out)

and an index.json beside them (command, exit code, seconds per run). A run
whose artifact already exists is skipped, so an interrupted batch resumes.

Without --live it prints the plan and the call count and makes no call.
Run it in the sim image through the wrapper (mounts the repository's .git so
each artifact records its commit, passes ANTHROPIC_API_KEY through):

    scripts/authoring_docker.sh python scripts/run_authoring_ablation.py --tag v1 --reps 5
    scripts/authoring_docker.sh python scripts/run_authoring_ablation.py --tag v1 --reps 5 --live
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASKS = {
    # the schema run Opus made on GT grounding (README, representation study)
    "insert_marker": {"scene": "scenes/insert_marker.json",
                      "grounding": "outputs/grounding/insert_marker_gt_opus1",
                      "stage_plan": "outputs/runs/gt_opus1/stage_plan.json",
                      "selections": "outputs/runs/gt_opus1/selections.json",
                      "authored_stages": {"A0": 3, "A3": 2, "A4": 2}},
    # authored sidecars and the fixed pour-tea stages, as plan_pour_tea.py uses
    "pour_tea": {"scene": "scenes/pour_tea.json", "grounding": None,
                 "stage_plan": None, "selections": "outputs/selections/pour_tea.json",
                 "authored_stages": {"A0": 3, "A3": 2, "A4": 2}},
}
METHODS = ("A0", "A3", "A4")
MODEL = "claude-opus-5"        # hardware's model (ur5e-manip-hardware .env)
TEMPERATURE = "none"           # hardware omits the key; the API default applies
MAX_TOKENS = "4000"            # one cap for every method (code replies are longer)
PARSE_ATTEMPTS, GATE_ATTEMPTS = 3, 3   # vlm.MAX_PARSE_RETRIES + 1, COMPILE_RETRIES + 1


def command(task: str, method: str, out: Path) -> list[str]:
    t = TASKS[task]
    common = ["--scene", t["scene"], "--selections", t["selections"], "--out", str(out),
              "--render-chained", str(out.with_suffix("")) + ".frames"]
    if t["grounding"]:
        common += ["--grounding", t["grounding"]]
    if t["stage_plan"]:
        common += ["--stage-plan", t["stage_plan"]]
    if method == "A0":
        return [sys.executable, "scripts/emit_constraints.py", *common]
    medium = {"A3": "constrained-code", "A4": "free-code"}[method]
    return [sys.executable, "scripts/emit_code.py", "--medium", medium, *common]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--start", type=int, default=0, help="first repetition index")
    ap.add_argument("--live", action="store_true", help="make the calls")
    args = ap.parse_args()

    root = Path("outputs/emissions") / f"authoring-{args.tag}"
    runs = [(t, m, i, root / t / f"{m}.{i}.json") for t in args.tasks for m in args.methods
            for i in range(args.start, args.start + args.reps)]
    todo = [r for r in runs if not r[3].exists()]
    lo = sum(TASKS[t]["authored_stages"][m] for t, m, _, _ in todo)
    hi = lo * PARSE_ATTEMPTS * GATE_ATTEMPTS
    print(f"[ablation] {len(runs)} runs ({len(runs) - len(todo)} already done), "
          f"model {MODEL}, temperature {TEMPERATURE}, max_tokens {MAX_TOKENS} (all methods)")
    print(f"[ablation] calls: {lo} if every stage passes first time, at most {hi} "
          "(parse and gate retries)")
    for t, m, i, out in todo:
        print(f"  {t:14s} {m}.{i}  -> {out}")
    if not args.live:
        print("[ablation] dry run: add --live to make the calls")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[ablation] ANTHROPIC_API_KEY is not set")
    env = {**os.environ, "MANIP_VLM_MODEL": MODEL, "MANIP_VLM_TEMPERATURE": TEMPERATURE,
           "MANIP_VLM_MAX_TOKENS": MAX_TOKENS, "PYTHONPATH": "."}
    index_p = root / "index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else []
    for t, m, i, out in todo:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = command(t, m, out)
        print(f"\n[ablation] {t} {m}.{i}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.perf_counter()
        with open(out.with_suffix(".out"), "w") as log:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        index.append({"task": t, "method": m, "rep": i, "artifact": str(out),
                      "exit": code, "seconds": round(time.perf_counter() - t0, 1),
                      "model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                      "command": cmd[1:]})
        index_p.write_text(json.dumps(index, indent=2) + "\n")
        print(f"[ablation]   exit {code} ({'gate passed' if code == 0 else 'see .out'})")


if __name__ == "__main__":
    main()
