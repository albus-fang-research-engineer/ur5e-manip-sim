"""Plan a scene with open-loop OmniManip (manip_sim/omnimanip): a program of
interaction primitives -> one joint-space plan, in the format plan_task.py
writes, so execute_task_plan.py and eval_plan.py treat every planner the same.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/plan_omnimanip.py \\
        --scene scenes/insert_marker.json

Writes outputs/plans/omnimanip/<scene>_omnimanip.npz and a .report.json beside
it. On failure only the report is written, with status "failed".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from manip_sim.omnimanip.planner import Config, OmniManipPlanner, default_program
from manip_sim.omnimanip.target_pose import Weights
from manip_sim.scene import add_scene_arg, load_scene, make_env
from manip_sim.task_plan import TaskPlanError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", default=None, help="default: manip_sim/omnimanip/programs/<scene>.json")
    ap.add_argument("--out", default=None, help="default: outputs/plans/omnimanip/<scene>_omnimanip.npz")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--seed", type=int, default=0, help="jitters the solver's start; 0 starts at the pose")
    ap.add_argument("--lambda-trans", type=float, default=Weights.lambda_trans)
    ap.add_argument("--lambda-rot", type=float, default=Weights.lambda_rot)
    add_scene_arg(ap)
    args = ap.parse_args()

    scene = load_scene(args.scene, args.grounding)
    program = args.program or default_program(scene.name)
    out = Path(args.out or f"outputs/plans/omnimanip/{scene.name}_omnimanip.npz")
    report_path = out.with_suffix(".report.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    env, _ = make_env(scene, robot=args.robot, has_renderer=False)
    cfg = Config(seed=args.seed, weights=Weights(lambda_trans=args.lambda_trans, lambda_rot=args.lambda_rot))
    planner = OmniManipPlanner(env, scene, program, cfg)
    t0 = time.time()
    try:
        plan = planner.plan()
    except TaskPlanError as e:
        planner.report.update(status="failed", failure=e.as_dict(),
                              seconds=round(time.time() - t0, 1))
        report_path.write_text(json.dumps(planner.report, indent=2) + "\n")
        print(f"[plan-omnimanip] FAILED: {e}\n  report -> {report_path}")
        env.close()
        sys.exit(1)
    planner.report["seconds"] = round(time.time() - t0, 1)
    np.savez(out, path=plan.path, stage_ids=plan.stage_ids, stage_labels=plan.stage_labels,
             q_home=plan.q_home, q_grasp=plan.q_grasp, T_ee_body=plan.T_ee_body,
             grasped=np.array(plan.grasped), planner=np.array("omnimanip"),
             program=np.array(str(program)), **(plan.target or {}),
             **{f"T0_{n}": T for n, T in plan.T0.items()})
    report_path.write_text(json.dumps(planner.report, indent=2) + "\n")
    print(f"[plan-omnimanip] {len(plan.path)} waypoints in {planner.report['seconds']} s -> {out}")
    env.close()


if __name__ == "__main__":
    main()
