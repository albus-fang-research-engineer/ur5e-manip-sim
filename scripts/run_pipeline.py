"""Scene picture + task query -> plan, end to end. Sequences the existing
stage scripts as subprocesses (each owns its own GL / API-key needs) and
passes artifacts between them by path; nothing is computed here.

    pool     scripts/propose_interaction_points.py --write, only if an
             asset has no candidates.json yet
    mark     scripts/mark_scene.py         scene RGB + numbered marks
    plan     scripts/plan_stages.py        VLM call #1 (image + query)
             -> stage plan, object parts, role bindings   [verified]
    ground   scripts/ground_parts.py       call #1's part names -> masks
             -> lifted points -> primitives -> runtime frames.json +
             candidates.json under outputs/grounding/<scene>; every
             later step reads them via --grounding. Then plan_stages
             --replay re-verifies the plan against the GROUNDED pools.
             (--no-ground: authored sidecars, the ground-truth arm)
    render   scripts/render_candidates.py  --vlm menus for the objects
             and parts call #1 named
    select   scripts/select_frames.py      VLM call #2 per bound role
             -> selections artifact                        [resolved]
    preview  scripts/preview_selections.py selected frames rendered on
             the objects, before any planner touches them
    emit     scripts/emit_constraints.py   VLM call #3 per bound stage,
             compile-gated -> emissions artifact (--no-emit: pour_stages
             hand compilers, the schema ablation's authored arm)
    path     scripts/plan_pour_tea.py      --selections [--emissions]
             -> plan npz
    video    scripts/render_full_plan.py   kinematic playback mp4
    exec     scripts/execute_pour_tea.py   physics execution mp4 + metrics

    export ANTHROPIC_API_KEY=...
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/run_pipeline.py --task "pour tea"
    ... --from select                  # resume after a step's artifacts exist
    ... --until path                   # stop early (no video / execution)
    ... --image rgb.png --masks DIR    # marks from another provider (SAM dump)
    ... --text-only                    # call #1 without the image (ablation arm)
    ... --ground-provider masks --masks-root DIR   # SAM part masks instead of the oracle
    ... --no-ground --no-emit          # authored symbols + hand compilers

Every step's artifacts stay where its script writes them (the paths each
script's docstring documents); the run additionally writes
outputs/runs/<stamp>/run.json listing the commands issued and the
artifact paths, so a run can be replayed or its failing step named.

What is and is not automatic here, stated once:

  * Call #1 decides the stage list, which marks matter, and the part
    names; call #2 decides the frame on each part. Both are live.
  * Call #1's stages are routed onto the planner's three stage roles by
    the task contract (tasks/<task>.json); a plan that does not bind
    stops at `plan` with the failing check named. Stages 2-3 constraints
    come from call #3 (compile_stage) by default; stage 1's grasp TSR is
    still handle_grasp_tsr because the gripper nominal is not in the
    emission vocabulary. A fourth call-#1 stage binds nowhere.
  * The default mask provider is the ORACLE: geometric bands computed
    from the authored sidecar, rasterized into per-view masks. It
    exercises the full lift/fit/symbol path with perfect segmentation;
    it is the segmentation-oracle arm, not grounding from names. Pass a
    GroundedSAM/SAM3 dump with --ground-provider masks for that.
  * plan_pour_tea.py itself still names teapot/mug; a different object
    pair needs its two symbol lookups generalized.
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

STEPS = ("pool", "mark", "plan", "ground", "render", "select", "preview",
         "emit", "path", "video", "exec")
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
    ap.add_argument("--no-ground", action="store_true",
                    help="skip runtime part grounding (authored sidecars)")
    ap.add_argument("--ground-provider", choices=("oracle", "masks"), default="oracle")
    ap.add_argument("--masks-root", default=None, metavar="DIR")
    ap.add_argument("--no-emit", action="store_true",
                    help="skip call #3; stages 2-3 from pour_stages.* hand compilers")
    ap.add_argument("--plan-args", default="",
                    help="extra args for plan_pour_tea.py, e.g. "
                         "'--seed 3 --tilt-deg 95'")
    args = ap.parse_args()
    if bool(args.image) != bool(args.masks):
        ap.error("--image and --masks go together")
    scene = load_scene(args.scene, getattr(args, "grounding", None))
    task = args.task or scene.task
    mode = "text" if args.text_only else "marks"
    marks_dir = Path(f"outputs/marks/{scene.name}")
    stage_plan = Path(f"outputs/stage_plan/{scene.name}.{mode}.json")
    selections = Path(f"outputs/selections/{scene.name}.json")
    emissions = Path(f"outputs/emissions/{scene.name}.json")
    grounding = Path(f"outputs/grounding/{scene.name}")
    G = [] if args.no_ground else ["--grounding", str(grounding)]
    do = lambda step: STEPS.index(args.start) <= STEPS.index(step) <= STEPS.index(args.stop)
    RUN.update(dir=f"outputs/runs/{time.strftime('%Y%m%d-%H%M%S')}",
               scene=args.scene, task=task, mode=mode,
               artifacts={"marks": str(marks_dir), "stage_plan": str(stage_plan),
                          "grounding": str(grounding) if not args.no_ground else "",
                          "selections": str(selections),
                          "emissions": str(emissions) if not args.no_emit else "",
                          "preview": str(selections.with_suffix(".preview.png")),
                          "plan": "outputs/plans/vlm/pour_tea_full_vlm.npz",
                          "video_plan": "outputs/videos/vlm/pour_tea_full_vlm.mp4",
                          "video_exec": "outputs/videos/vlm/pour_tea_exec_vlm.mp4",
                          "metrics": "outputs/metrics/vlm/pour_tea_exec_vlm.json"})
    if (do("plan") or do("select") or do("emit")) and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[pipeline] ANTHROPIC_API_KEY unset (calls #1-#3 are live)")

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

    if do("ground") and not args.no_ground:
        cmd = ["scripts/ground_parts.py", "--scene", args.scene, "--stage-plan",
               str(stage_plan), "--provider", args.ground_provider, "--out",
               str(grounding), "--write"]
        if args.masks_root:
            cmd += ["--masks-root", args.masks_root]
        sh(cmd, gl=args.gl)
        # the plan must still verify against the pools it will now be
        # selected from; a part that grounded to nothing fails here
        sh(["scripts/plan_stages.py", "--scene", args.scene, "--replay",
            str(stage_plan), *(["--marks", str(marks_dir)] if not args.text_only else []),
            *G])

    if do("render"):
        parts = json.loads(stage_plan.read_text())["object_parts"]
        for obj, ps in parts.items():
            if ps:
                sh(["scripts/render_candidates.py", "--scene", args.scene, *G,
                    "--object", obj, "--vlm", "--parts", *ps], gl=args.gl)

    if do("select"):
        sh(["scripts/select_frames.py", "--scene", args.scene, *G,
            "--stage-plan", str(stage_plan), "--out", str(selections)])

    if do("preview"):
        sh(["scripts/preview_selections.py", str(selections), "--scene", args.scene, *G,
            "--render", str(selections.with_suffix(".preview.png"))], gl=args.gl)

    if do("emit") and not args.no_emit:
        sh(["scripts/emit_constraints.py", "--scene", args.scene, *G,
            "--stage-plan", str(stage_plan), "--selections", str(selections),
            "--out", str(emissions)])

    if do("path"):
        sh(["scripts/plan_pour_tea.py", "--scene", args.scene, *G,
            "--selections", str(selections),
            *([] if args.no_emit else ["--emissions", str(emissions)]),
            *shlex.split(args.plan_args)], gl=args.gl)

    if do("video"):
        sh(["scripts/render_full_plan.py", "--arm", "vlm"],
           gl=args.gl)

    if do("exec"):
        sh(["scripts/execute_pour_tea.py", "--scene", args.scene, *G, "--arm", "vlm"],
           gl=args.gl)

    _write_run(status="ok")
    print(f"\n[pipeline] task {task!r} -> {RUN['dir']}/run.json")
    for k, v in RUN["artifacts"].items():
        print(f"  {k:11s} {v}" + ("" if Path(v).exists() else "   (not produced)"))


if __name__ == "__main__":
    main()
