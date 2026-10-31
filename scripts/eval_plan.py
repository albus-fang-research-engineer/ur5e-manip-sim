"""Grade joint-space plans against one written task spec, kinematically. It
reads only the plan .npz, so every planner is graded by the same code. Motion is
checked along the interpolated path the tracker commands, not just waypoints.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_plan.py \\
        --scene scenes/insert_marker.json --spec tasks/specs/insert_marker.json \\
        --plan outputs/plans/rekep/insert_marker_rekep.npz

Writes <plan>.eval.json beside each plan:

    goal        tilt from vertical, the lower end, its distance off the opening's
                axis and depth below the rim, pass/fail per spec tolerance
    inserted    worst tilt and off-axis distance while the end is inside the rim
                (below the rim plane AND within its radius, so an object lying on
                the table does not count)
    transport   the carried object's attitude on the way: world tilt of its resting
                up axis over the travel, and how early its rotation is done
    collisions  per segment, named pairs with the deepest penetration, grouped as
                self / table / passive / other; reported, not part of goal_pass
    path        end-effector travel and rotation, joint travel, largest joint step

Tilt is measured on the axis between the two ends and does not depend on which
one is called the tip. Which end is lower is reported and checked separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from manip_sim.execution import densify
from manip_sim.frames import load_symbols
from manip_sim.planning import ArmKinematics, AttachedArmKinematics, AttachedObject
from manip_sim.scene import add_scene_arg, load_scene, make_env

ROBOT = ("robot0", "gripper0", "fixed_mount0", "mount0")
TRACK_STEP_RAD = 0.02             # Tracker.track's max_joint_step


def end_state(end_a: np.ndarray, end_b: np.ndarray, opening: np.ndarray) -> dict:
    """World positions of an object's two ends against an opening's centre."""
    a_lower = end_a[2] <= end_b[2]
    low, high = (end_a, end_b) if a_lower else (end_b, end_a)
    axis = (high - low) / np.linalg.norm(high - low)
    return {"lower": "a" if a_lower else "b",
            "tilt_deg": float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0)))),
            "off_axis_m": float(np.linalg.norm((low - opening)[:2])),
            "depth_m": float(opening[2] - low[2])}


def is_inside(state: dict, rim_radius: float) -> bool:
    return state["depth_m"] > 0.0 and state["off_axis_m"] < rim_radius


def goal_checks(state: dict, spec: dict) -> dict:
    return {"first_end_lower": state["lower"] == "a",
            "tilt": state["tilt_deg"] <= spec["max_tilt_deg"],
            "depth": state["depth_m"] >= spec["min_depth_m"],
            "inside_rim": state["off_axis_m"] < spec["rim_radius_m"]}


def transport_profile(travel_m, rest_up_tilt_deg, rotation_deg) -> dict:
    """How an object tilts on its way, from the first carried point to the last:
    the world tilt of its resting up axis at quarters of the end-effector travel,
    and how much of the total rotation is done by half the travel (near 100 means
    it turns early and carries the attitude, near 50 that it turns all the way)."""
    travel, tilt, rotation = map(np.asarray, (travel_m, rest_up_tilt_deg, rotation_deg))
    at = lambda f, y: float(np.interp(f * travel[-1], travel, y))
    total = float(np.max(rotation))
    return {"travel_m": round(float(travel[-1]), 3),
            "rest_up_tilt_deg_at_travel": {f"{int(f * 100)}%": round(at(f, tilt), 1)
                                           for f in (0.0, 0.25, 0.5, 0.75, 1.0)},
            "max_rest_up_tilt_deg": round(float(np.max(tilt)), 1),
            "total_rotation_deg": round(total, 1),
            "rotation_done_at_half_travel_pct": round(100.0 * at(0.5, rotation) / total, 1) if total > 1e-9 else None}


def collision_group(a: str, b: str, passive: str) -> str:
    names = (a, b)
    if all(n.startswith(ROBOT) for n in names):
        return "self"
    if any(n == "table" or n.startswith("table") for n in names):
        return "table"
    if any(n.startswith(passive) for n in names):
        return "passive"
    return "other"


def contacts(kin, active: str, depth_tol: float = 1e-4) -> list[tuple]:
    """(body a, body b, depth m) of penetrations involving the robot or the
    active object, without the gripper's own contacts or the grasp itself."""
    m, d, watched, out = kin.model, kin.data, ROBOT + (active,), []
    for i in range(d.ncon):
        con = d.contact[i]
        if con.dist > -depth_tol:
            continue
        a, b = sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
                      for g in (con.geom1, con.geom2))
        if not (a.startswith(watched) or b.startswith(watched)):
            continue
        if a.startswith("gripper0") and b.startswith("gripper0"):
            continue
        if {a.split("_")[0], b.split("_")[0]} == {"gripper0", active}:
            continue
        out.append((a, b, float(-con.dist)))
    return out


def evaluate(plan_path, env, scene, spec) -> dict:
    plan = np.load(plan_path)
    path, ids = plan["path"], plan["stage_ids"]
    labels = [str(s) for s in plan["stage_labels"]]
    active, passive = spec["active"], spec["passive"]
    T_ee_body = plan["T_ee_body"]
    points = load_symbols(scene.asset_dirs[active]).points
    end_a, end_b = points[spec["first_end"]], points[spec["other_end"]]
    T_p = plan[f"T0_{passive}"]
    opening = T_p[:3, :3] @ load_symbols(scene.asset_dirs[passive]).points[spec["opening"]] + T_p[:3, 3]
    free = ArmKinematics(env)
    held = AttachedArmKinematics(env, AttachedObject(T_ee_body),
                                 env.objects[active].joints[0], active)

    rest_up = load_symbols(scene.asset_dirs[active]).axes[spec["rest_up_axis"]]
    rest_up = rest_up / np.linalg.norm(rest_up)
    carried_travel, carried_tilt, carried_rotation, R_first = [], [], [], None
    segments, totals, state, inserted = [], {"self": 0.0, "table": 0.0, "passive": 0.0, "other": 0.0}, None, []
    for i, label in enumerate(labels):
        kin = held if i else free
        dense = densify(path[ids == i], TRACK_STEP_RAD)
        hits, pairs, T_prev, travel, rotation = 0, {}, None, 0.0, 0.0
        for q in dense:
            T = kin.fk(q)
            found = contacts(kin, active)
            hits += bool(found)
            for a, b, depth in found:
                key = f"{a} <-> {b}"
                pairs[key] = max(pairs.get(key, 0.0), depth)
                group = collision_group(a, b, passive)
                totals[group] = max(totals[group], depth)
            if T_prev is not None:
                travel += float(np.linalg.norm(T[:3, 3] - T_prev[:3, 3]))
                rotation += float(np.linalg.norm(
                    R.from_matrix(T_prev[:3, :3].T @ T[:3, :3]).as_rotvec()))
            T_prev = T
            if i:
                T_b = T @ T_ee_body
                R_first = T_b[:3, :3] if R_first is None else R_first
                up = T_b[:3, :3] @ rest_up
                carried_travel.append((carried_travel[-1] if carried_travel else 0.0)
                                      + (float(np.linalg.norm(T[:3, 3] - T_prev_carried[:3, 3]))
                                         if carried_travel else 0.0))
                T_prev_carried = T
                carried_tilt.append(float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))))
                carried_rotation.append(float(np.degrees(np.linalg.norm(
                    R.from_matrix(R_first.T @ T_b[:3, :3]).as_rotvec()))))
                state = end_state(T_b[:3, :3] @ end_a + T_b[:3, 3],
                                  T_b[:3, :3] @ end_b + T_b[:3, 3], opening)
                if is_inside(state, spec["rim_radius_m"]):
                    inserted.append(state)
        steps = np.abs(np.diff(path[ids == i], axis=0))
        segments.append({"segment": label, "n_waypoints": int(np.sum(ids == i)), "n_tracked": len(dense),
                         "colliding_points": hits,
                         "pairs_max_depth_mm": {k: round(v * 1000, 1) for k, v in pairs.items()},
                         "ee_travel_m": round(travel, 3), "ee_rotation_deg": round(np.degrees(rotation), 1),
                         "joint_travel_rad": round(float(steps.sum()), 2),
                         "max_joint_step_rad": round(float(steps.max()), 3) if steps.size else 0.0})
    free.set_q(path[0])
    checks = goal_checks(state, spec)
    out = {"plan": str(plan_path), "spec": spec["task"],
           "goal": {"lower_end": spec["first_end"] if state["lower"] == "a" else spec["other_end"],
                    "tilt_deg": round(state["tilt_deg"], 2),
                    "off_axis_mm": round(state["off_axis_m"] * 1000, 1),
                    "depth_mm": round(state["depth_m"] * 1000, 1)},
           "checks": checks, "goal_pass": all(checks.values()),
           "inserted": {"n_points": len(inserted),
                        "max_tilt_deg": round(max((s["tilt_deg"] for s in inserted), default=float("nan")), 2),
                        "max_off_axis_mm": round(max((s["off_axis_m"] for s in inserted), default=float("nan")) * 1000, 1)},
           "transport": transport_profile(carried_travel, carried_tilt, carried_rotation),
           "max_penetration_mm": {k: round(v * 1000, 1) for k, v in totals.items()},
           "colliding_points": sum(s["colliding_points"] for s in segments),
           "n_tracked": sum(s["n_tracked"] for s in segments),
           "ee_travel_m": round(sum(s["ee_travel_m"] for s in segments), 3),
           "ee_rotation_deg": round(sum(s["ee_rotation_deg"] for s in segments), 1),
           "joint_travel_rad": round(sum(s["joint_travel_rad"] for s in segments), 2),
           "max_joint_step_rad": max(s["max_joint_step_rad"] for s in segments),
           "segments": segments}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, metavar="JSON")
    ap.add_argument("--plan", required=True, nargs="+")
    ap.add_argument("--robot", default="UR5e")
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene, args.grounding)
    spec = json.loads(Path(args.spec).read_text())
    env, _ = make_env(scene, robot=args.robot, has_renderer=False)
    for plan_path in args.plan:
        result = evaluate(plan_path, env, scene, spec)
        Path(plan_path).with_suffix(".eval.json").write_text(json.dumps(result, indent=2) + "\n")
        g = result["goal"]
        print(f"[eval] {Path(plan_path).stem}: pass={result['goal_pass']} lower={g['lower_end']} "
              f"tilt={g['tilt_deg']} deg off_axis={g['off_axis_mm']} mm depth={g['depth_mm']} mm "
              f"| inserted max tilt={result['inserted']['max_tilt_deg']} off_axis={result['inserted']['max_off_axis_mm']} mm "
              f"| penetration mm {result['max_penetration_mm']} "
              f"({result['colliding_points']}/{result['n_tracked']} pts) "
              f"| ee {result['ee_travel_m']} m {result['ee_rotation_deg']} deg joint {result['joint_travel_rad']} rad")
    env.close()


if __name__ == "__main__":
    main()
