"""A pour-tea scene at another separation, optionally with an obstacle near the line
between the pot and the mug, and a spec holding the study's grip there: the same
in-hand transform as tasks/specs/pour_tea.json, with q_grasp solved for the new pot pose.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/make_pour_scene.py --separation 0.8 \\
        --obstacle 0.03 0.0 --out outputs/plans/pour_obstacle/sep80_obstacle

Writes <out>.scene.json and <out>.spec.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manip_sim.planning import ArmKinematics, MinkIK
from manip_sim.scene import load_scene, make_env
from manip_sim.state import PoseReader
from manip_sim.tsr import pose_from_pos_quat_wxyz

BASE_SCENE, BASE_SPEC = "scenes/pour_tea.json", "tasks/specs/pour_tea.json"
GRIP_ALLOWED = (("gripper0", "gripper0"), ("gripper0", "teapot"))


def scene_doc(base: dict, separation: float, obstacle_xy=None) -> dict:
    doc = json.loads(json.dumps(base))
    doc["objects"]["teapot"]["placement"]["xy"] = [0.0, -separation / 2]
    doc["objects"]["mug"]["placement"]["xy"] = [0.0, separation / 2]
    if obstacle_xy is not None:
        doc["objects"]["obstacle"] = {"asset": "assets/objects/obstacle",
                                      "placement": {"xy": [float(v) for v in obstacle_xy], "yaw": 0.0}}
    return doc


def teapot_pose(env, objs) -> np.ndarray:
    return pose_from_pos_quat_wxyz(*PoseReader(env, list(objs)).pose("teapot"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--separation", type=float, required=True, help="m between pot and mug, along world y")
    ap.add_argument("--obstacle", type=float, nargs=2, default=None, metavar=("X", "Y"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(Path(BASE_SPEC).read_text())

    base_scene = load_scene(BASE_SCENE, None)
    env, objs = make_env(base_scene, robot="UR5e", has_renderer=False)
    kin = ArmKinematics(env)
    T_ee_body = np.linalg.inv(kin.fk(np.asarray(spec["q_grasp"]))) @ teapot_pose(env, objs)
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene_path = args.out.parent / f"{args.out.name}.scene.json"
    scene_path.write_text(json.dumps(scene_doc(json.loads(Path(BASE_SCENE).read_text()), args.separation,
                                               args.obstacle), indent=1) + "\n")
    scene = load_scene(scene_path, None)
    env, objs = make_env(scene, robot="UR5e", has_renderer=False)
    kin, q_home = ArmKinematics(env), None
    ik = MinkIK(kin)
    q_home = env.sim.data.qpos[kin.qpos_ids].copy()
    T_grip = teapot_pose(env, objs) @ np.linalg.inv(T_ee_body)
    seeds = [q_home] + [q_home + np.r_[d, np.zeros(5)] for d in (-1.0, -0.5, 0.5, 1.0)]
    rng = np.random.default_rng(0)
    seeds += [rng.uniform(-np.pi, np.pi, 6) for _ in range(40)]
    solutions = []
    for seed in seeds:
        q, ok = ik.solve_multiseed(T_grip, [seed])
        if not ok:
            continue
        q = q + 2 * np.pi * np.round((q_home - q) / (2 * np.pi))    # IK can answer a full turn away
        if not kin.in_collision(q, allowed_prefix_pairs=GRIP_ALLOWED):
            solutions.append(q)
    if not solutions:
        raise SystemExit(f"no collision-free IK for the study's grip at separation {args.separation}")
    q = min(solutions, key=lambda s: float(np.linalg.norm(s - np.asarray(spec["q_grasp"]))))
    spec["q_grasp"] = q.tolist()
    (args.out.parent / f"{args.out.name}.spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[pour-scene] {scene_path}: {len(solutions)} grip solutions, q_grasp {np.round(q, 3).tolist()}")
    env.close()


if __name__ == "__main__":
    main()
