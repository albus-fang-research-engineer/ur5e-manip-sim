"""Scene picture + task query -> plan, end to end. Sequences the existing
stage scripts as subprocesses (each owns its own GL / API-key needs) and
passes artifacts between them by path; nothing is computed here.

    pool     scripts/propose_interaction_points.py --write, only if an
             asset has no candidates.json yet
    mark     scripts/mark_scene.py         scene RGB + numbered marks
    plan     scripts/plan_stages.py        VLM call #1 (image + query)
             -> stage plan, object parts, role bindings   [verified]
    render   scripts/render_candidates.py  --vlm menus for the objects
             and parts call #1 named
    select   scripts/select_frames.py      VLM call #2 per bound role
             -> selections artifact                        [resolved]
    preview  scripts/preview_selections.py selected frames rendered on
             the objects, before any planner touches them
    path     scripts/plan_pour_tea.py      --selections -> plan npz
    video    scripts/render_full_plan.py   kinematic playback mp4
    exec     scripts/execute_pour_tea.py   physics execution mp4 + metrics

    export ANTHROPIC_API_KEY=...
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/run_pipeline.py --task "pour tea"
    ... --from select                  # resume after a step's artifacts exist
    ... --until path                   # stop early (no video / execution)
    ... --image rgb.png --masks DIR    # marks from another provider (SAM dump)
    ... --text-only                    # call #1 without the image (ablation arm)

Every step's artifacts stay where its script writes them (the paths each
script's docstring documents); the run additionally writes
outputs/runs/<stamp>/run.json listing the commands issued and the
artifact paths, so a run can be replayed or its failing step named.

What is and is not automatic here, stated once:

  * Call #1 decides the stage list, which marks matter, and the part
    names; call #2 decides the frame on each part. Both are live.
  * The stage CONSTRAINTS are still pour_stages.* (grasp / transport /
    pour compilers) — call #1's stages are routed to those three by the
    task contract's role bindings (tasks/<task>.json). A plan whose
    stages do not bind to the contract stops at `plan` with the failing
    check named; it never reaches the planner. Swapping pour_stages.*
    for compiled call-#3 emissions is the `--emissions` item, not this.
  * Part names are grounded against candidates.json, which is built
    from the asset's frames.json. That is sim-native grounding; the
    hardware path (part name -> mask -> primitive fit -> frames.json)
    is a separate item.
  * A raw photograph with no masks needs a segmenter: sim uses the
    MuJoCo segmentation buffer; on hardware pass --image/--masks from
    the SAM node.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from manip_sim.scene import add_scene_arg, load_scene

STEPS = ("pool", "mark", "plan", "render", "select", "preview", "path",
         "video", "exec")
PY = [sys.executable]


ISSUED: list[str] = []


def sh(args: list[str], gl: str | None = None) -> None:
    env = dict(os.environ, PYTHONPATH=".")
    if gl:
        env["MUJOCO_GL"] = gl
        env["PYOPENGL_PLATFORM"] = gl
    line = " ".join(shlex.quote(a) for a in args)
    print(f"\n$ {line}", flush=True)
    ISSUED.append(line)
    r = subprocess.run(PY + args, env=env)
    if r.returncode:
        _write_run(status=f"failed at {args[0]} (exit {r.returncode})")
        sys.exit(f"[pipeline] step failed ({args[0]}, exit {r.returncode}); "
                 f"fix and resume with --from <step>")


RUN: dict = {}


def _write_run(status: str) -> None:
    d = Path(RUN["dir"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(
        {**RUN, "status": status, "commands": ISSUED}, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_arg(ap)
    ap.add_argument("--task", default=None,
                    help="language query; default: the scene manifest's")
    ap.add_argument("--from", dest="start", choices=STEPS, default="pool")
    ap.add_argument("--until", dest="stop", choices=STEPS, default="exec")
    ap.add_argument("--image", default=None, metavar="RGB",
                    help="with --masks: ingest external marks instead of "
                         "rendering the sim camera")
    ap.add_argument("--masks", default=None, metavar="DIR")
    ap.add_argument("--text-only", action="store_true",
                    help="call #1 on object names + symbols, no image")
    ap.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "osmesa"))
    ap.add_argument("--plan-args", default="",
                    help="extra args for plan_pour_tea.py, e.g. "
                         "'--seed 3 --tilt-deg 95'")
    args = ap.parse_args()
    if bool(args.image) != bool(args.masks):
        ap.error("--image and --masks go together")
    scene = load_scene(args.scene)
    task = args.task or scene.task
    mode = "text" if args.text_only else "marks"
    marks_dir = Path(f"outputs/marks/{scene.name}")
    stage_plan = Path(f"outputs/stage_plan/{scene.name}.{mode}.json")
    selections = Path(f"outputs/selections/{scene.name}.json")
    do = lambda step: STEPS.index(args.start) <= STEPS.index(step) <= STEPS.index(args.stop)
    RUN.update(dir=f"outputs/runs/{time.strftime('%Y%m%d-%H%M%S')}",
               scene=args.scene, task=task, mode=mode,
               artifacts={"marks": str(marks_dir), "stage_plan": str(stage_plan),
                          "selections": str(selections),
                          "preview": str(selections.with_suffix(".preview.png")),
                          "plan": "outputs/plans/vlm/pour_tea_full_vlm.npz",
                          "video_plan": "outputs/videos/vlm/pour_tea_full_vlm.mp4",
                          "video_exec": "outputs/videos/vlm/pour_tea_exec_vlm.mp4",
                          "metrics": "outputs/metrics/vlm/pour_tea_exec_vlm.json"})
    if (do("plan") or do("select")) and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[pipeline] ANTHROPIC_API_KEY unset (calls #1 and #2 are live)")

    if do("pool"):
        missing = [n for n, d in scene.asset_dirs.items()
                   if not (Path(d) / "candidates.json").exists()]
        for n in missing:
            sh(["scripts/propose_interaction_points.py", "--scene", args.scene,
                "--object", n, "--write"])

    if do("mark") and not args.text_only:
        cmd = ["scripts/mark_scene.py", "--scene", args.scene, "--out", str(marks_dir)]
        if args.image:
            cmd += ["--from-masks", args.image, args.masks]
        sh(cmd, gl=None if args.image else args.gl)

    if do("plan"):
        cmd = ["scripts/plan_stages.py", "--scene", args.scene, "--task", task,
               "--out", str(stage_plan)]
        if not args.text_only:
            cmd += ["--marks", str(marks_dir)]
        sh(cmd)                      # exits nonzero if the contract fails

    if do("render"):
        parts = json.loads(stage_plan.read_text())["object_parts"]
        for obj, ps in parts.items():
            if ps:
                sh(["scripts/render_candidates.py", "--scene", args.scene,
                    "--object", obj, "--vlm", "--parts", *ps], gl=args.gl)

    if do("select"):
        sh(["scripts/select_frames.py", "--scene", args.scene,
            "--stage-plan", str(stage_plan), "--out", str(selections)])

    if do("preview"):
        sh(["scripts/preview_selections.py", str(selections), "--scene", args.scene,
            "--render", str(selections.with_suffix(".preview.png"))], gl=args.gl)

    if do("path"):
        sh(["scripts/plan_pour_tea.py", "--scene", args.scene,
            "--selections", str(selections), *shlex.split(args.plan_args)],
           gl=args.gl)

    if do("video"):
        sh(["scripts/render_full_plan.py", "--arm", "vlm"],
           gl=args.gl)

    if do("exec"):
        sh(["scripts/execute_pour_tea.py", "--scene", args.scene, "--arm", "vlm"],
           gl=args.gl)

    _write_run(status="ok")
    print(f"\n[pipeline] task {task!r} -> {RUN['dir']}/run.json")
    for k, v in RUN["artifacts"].items():
        print(f"  {k:11s} {v}" + ("" if Path(v).exists() else "   (not produced)"))


if __name__ == "__main__":
    main()
