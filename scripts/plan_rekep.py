"""Plan a scene with open-loop ReKep (manip_sim/rekep): a program of keypoints
and constraints -> one joint-space plan, in the format plan_task.py writes, so
execute_task_plan.py runs both planners the same way.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/plan_rekep.py \\
        --scene scenes/insert_marker.json
    MUJOCO_GL=egl PYTHONPATH=. python scripts/execute_task_plan.py \\
        --scene scenes/insert_marker.json \\
        --plan outputs/plans/rekep/insert_marker_rekep.npz

Writes outputs/plans/rekep/<scene>_rekep.npz and a .report.json beside it. On
failure only the report is written, with status "failed".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from manip_sim.rekep.planner import Config, ReKepPlanner, default_program_dir
from manip_sim.scene import add_scene_arg, load_scene, make_env
from manip_sim.task_plan import TaskPlanError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", default=None, help="default: manip_sim/rekep/programs/<scene>")
    ap.add_argument("--out", default=None, help="default: outputs/plans/rekep/<scene>_rekep.npz")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=Config.attempts)
    ap.add_argument("--spec", default=None, help="task spec whose q_grasp the plan starts from")
    ap.add_argument("--soft-path", action="store_true",
                    help="keep a path that breaks its constraints, as upstream would execute it")
    ap.add_argument("--soft-subgoal", action="store_true",
                    help="keep a subgoal that breaks its constraints, as upstream would execute it")
    ap.add_argument("--soft-ik", action="store_true",
                    help="keep a solve IK flags and a pose IK misses, as upstream would execute them")
    ap.add_argument("--upstream-loop", action="store_true",
                    help="main.py's loop: re-solve every few poses from where the arm is, reject nothing")
    ap.add_argument("--opt-rot-step", type=float, default=Config.opt_rot_step_size,
                    help="rad of rotation per path control point")
    ap.add_argument("--collision-scale", type=float, default=Config.collision_scale,
                    help="multiplies upstream's collision weights")
    ap.add_argument("--langfold-constraints", action="store_true",
                    help="stages 2-3 under LangFold's pour TSRs (manip_sim/rekep/langfold.py); needs --upstream-loop")
    add_scene_arg(ap)
    args = ap.parse_args()
    if args.langfold_constraints and not args.upstream_loop:
        ap.error("--langfold-constraints needs --upstream-loop")

    scene = load_scene(args.scene, args.grounding)
    program = args.program or default_program_dir(scene.name)
    out = Path(args.out or f"outputs/plans/rekep/{scene.name}_rekep.npz")
    report_path = out.with_suffix(".report.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    env, _ = make_env(scene, robot=args.robot, has_renderer=False)
    planner = ReKepPlanner(env, scene, program, Config(seed=args.seed, attempts=args.attempts, soft_path=args.soft_path,
                                                     soft_subgoal=args.soft_subgoal, soft_ik=args.soft_ik or args.upstream_loop,
                                                     upstream_loop=args.upstream_loop,
                                                     constraint_tolerance=1e-4 if args.upstream_loop else Config.constraint_tolerance,
                                                     opt_rot_step_size=args.opt_rot_step, collision_scale=args.collision_scale))
    if args.langfold_constraints:
        from manip_sim.rekep.langfold import PourConstraints
        held = planner.program.grasped_object
        idx = [i for i, k in enumerate(planner.program.keypoints) if k.object == held]
        planner.stage_constraints = PourConstraints(planner.symbols["teapot"], planner.symbols["mug"],
                                                    planner.T0["teapot"], planner.T0["mug"],
                                                    idx, [planner.keypoints_body[i] for i in idx])
        planner.report["constraints"] = "langfold"
    t0 = time.time()
    try:
        plan = planner.plan(np.asarray(json.loads(Path(args.spec).read_text())["q_grasp"]) if args.spec else None)
    except TaskPlanError as e:
        planner.report.update(status="failed", failure=e.as_dict(),
                              seconds=round(time.time() - t0, 1))
        report_path.write_text(json.dumps(planner.report, indent=2) + "\n")
        print(f"[plan-rekep] FAILED: {e}\n  report -> {report_path}")
        env.close()
        sys.exit(1)
    planner.report["seconds"] = round(time.time() - t0, 1)
    np.savez(out, path=plan.path, stage_ids=plan.stage_ids, stage_labels=plan.stage_labels,
             q_home=plan.q_home, q_grasp=plan.q_grasp, T_ee_body=plan.T_ee_body,
             grasped=np.array(plan.grasped), planner=np.array("rekep"),
             program=np.array(str(program)), **(plan.target or {}),
             **({} if plan.targets is None else {"targets": plan.targets}),
             **{f"T0_{n}": T for n, T in plan.T0.items()})
    report_path.write_text(json.dumps(planner.report, indent=2) + "\n")
    print(f"[plan-rekep] {len(plan.path)} waypoints in {planner.report['seconds']} s -> {out}")
    env.close()


if __name__ == "__main__":
    main()
