"""Scene picture + task query -> plan, end to end. Sequences the existing
stage scripts as subprocesses (each owns its own GL / API-key needs) and
passes artifacts between them by path; nothing is computed here.

    mark     scripts/mark_scene.py         scene RGB + numbered marks
    plan     scripts/plan_stages.py        VLM call #1 (image + query)
             -> stage plan, object parts, role bindings   [verified]
    render   scripts/render_candidates.py  --vlm menus for the objects
             and parts call #1 named
    select   scripts/select_frames.py      VLM call #2 per bound role
             -> selections artifact                        [resolved]
    path     scripts/plan_pour_tea.py      --selections -> plan npz
    exec     scripts/execute_pour_tea.py   (--execute only)

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/run_pipeline.py --task "pour tea"
    ... --from select                  # resume after a step's artifacts exist
    ... --image rgb.png --masks DIR    # marks from another provider (SAM dump)
    ... --text-only                    # call #1 without the image (ablation arm)
    ... --execute                      # also run the executor on the plan

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
from pathlib import Path

from manip_sim.scene import add_scene_arg, load_scene

STEPS = ("mark", "plan", "render", "select", "path", "exec")
PY = [sys.executable]


def sh(args: list[str], gl: str | None = None) -> None:
    env = dict(os.environ, PYTHONPATH=".")
    if gl:
        env["MUJOCO_GL"] = gl
        env["PYOPENGL_PLATFORM"] = gl
    print(f"\n$ {' '.join(shlex.quote(a) for a in args)}", flush=True)
    r = subprocess.run(PY + args, env=env)
    if r.returncode:
        sys.exit(f"[pipeline] step failed ({args[0]}, exit {r.returncode}); "
                 f"fix and resume with --from <step>")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_arg(ap)
    ap.add_argument("--task", default=None,
                    help="language query; default: the scene manifest's")
    ap.add_argument("--from", dest="start", choices=STEPS, default="mark")
    ap.add_argument("--image", default=None, metavar="RGB",
                    help="with --masks: ingest external marks instead of "
                         "rendering the sim camera")
    ap.add_argument("--masks", default=None, metavar="DIR")
    ap.add_argument("--text-only", action="store_true",
                    help="call #1 on object names + symbols, no image")
    ap.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "osmesa"))
    ap.add_argument("--execute", action="store_true")
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
    do = lambda step: STEPS.index(step) >= STEPS.index(args.start)
    if (do("plan") or do("select")) and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[pipeline] ANTHROPIC_API_KEY unset (calls #1 and #2 are live)")

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

    parts = json.loads(stage_plan.read_text())["object_parts"]
    if do("render"):
        for obj, ps in parts.items():
            if ps:
                sh(["scripts/render_candidates.py", "--scene", args.scene,
                    "--object", obj, "--vlm", "--parts", *ps], gl=args.gl)

    if do("select"):
        sh(["scripts/select_frames.py", "--scene", args.scene,
            "--stage-plan", str(stage_plan), "--out", str(selections)])

    if do("path"):
        sh(["scripts/plan_pour_tea.py", "--scene", args.scene,
            "--selections", str(selections), *shlex.split(args.plan_args)],
           gl=args.gl)

    if do("exec") and args.execute:
        sh(["scripts/execute_pour_tea.py", "--scene", args.scene, "--arm", "vlm"],
           gl=args.gl)

    print(f"\n[pipeline] task {task!r}\n  stage plan  {stage_plan}\n"
          f"  selections  {selections}\n  plan        outputs/plans/vlm/pour_tea_full_vlm.npz")


if __name__ == "__main__":
    main()
