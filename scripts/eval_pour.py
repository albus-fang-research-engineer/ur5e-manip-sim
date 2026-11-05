"""Grade pour-tea plans against tasks/specs/pour_tea.json, kinematically. It reads
only the plan .npz, so every planner is graded by the same code. Motion is checked
along the interpolated path the tracker commands, and the object is taken as rigid
in the hand: no physics, so the pot's mass and the grip's friction play no part.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_pour.py --scene scenes/pour_tea.json \\
        --spec tasks/specs/pour_tea.json --plan outputs/plans/pour_compare/*.npz
    PYTHONPATH=. python scripts/eval_pour.py --table outputs/plans/pour_compare \\
        --spec tasks/specs/pour_tea.json --out outputs/comparison/pour_tea

Planners split this task differently (OmniManip has one pour stage, the others a
transport and a pour), so nothing here reads their stages. Every limit is on the motion
after the grip, with the same tilt limit deciding when the pot counts as carried or pouring:

    carrying    until the spout tip first enters the rim's column: the worst tilt
    pouring     from then on, once tilted past the limit: the spout tip stays inside the rim's
                column. So a tilted spout outside the rim breaks exactly one of the two. How
                far it wanders is reported only
    at the end  the spout tip over the rim, the spout's axis near straight down. Its height,
                and where a straight stream from it crosses the rim's plane, are reported only
    collisions  a plan deeper than COLLISION_TOL_MM into anything is not collision-free, and so
                not a success, however well it meets the limits above
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manip_sim.execution import densify
from manip_sim.frames import load_symbols
from manip_sim.planning import AttachedArmKinematics, AttachedObject
from manip_sim.scene import add_scene_arg, load_scene, make_env
from manip_sim.viz import stream_landing
from scripts.compare_plans import markdown_table, span
from scripts.eval_plan import TRACK_STEP_RAD, collision_group, contacts

METHODS = {"ours": "Ours, 15° per axis", "ours10": "Ours, 10.5° per axis", "ours5": "Ours, 5° per axis",
           "ours10g": "Ours, 10.5° per axis, goal region", "ours5g": "Ours, 5° per axis, goal region",
           "rekepdef": "ReKep, codebase weights", "rekepnogate": "ReKep, codebase weights, no rejection",
           "rekeploop": "ReKep, upstream's loop",
           "rekeploopcol3": "ReKep, upstream's loop, collision x3", "rekeploopcol10": "ReKep, upstream's loop, collision x10",
           "rekeploopcol20": "ReKep, upstream's loop, collision x20",
           "rekeploopcol30": "ReKep, upstream's loop, collision x30",
           "rekep": "ReKep, strict","rekepfine": "ReKep, strict, finer path", "rekepsoft": "ReKep as upstream", "rekephigh": "ReKep, spout 10 cm above rim",
           "rekepcol10": "ReKep, collision x10", "rekepcol30": "ReKep, collision x30", "rekepcol100": "ReKep, collision x100",
           "rekepcol10up": "ReKep, collision x10, upright x0.25",
           "omni": "OmniManip", "omnirrt": "OmniManip + RRT-Connect"}


COLLISION_TOL_MM = 1.0
BOX = "obstacle"                  # make_pour_scene.py's box


def split_motion(tilts: np.ndarray, off_axis: np.ndarray, rim: float, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """(carrying, pouring) masks over the motion's samples."""
    inside = off_axis < rim
    first = int(np.argmax(inside)) if inside.any() else len(off_axis)
    carrying = np.arange(len(off_axis)) < first
    return carrying, ~carrying & (tilts > limit)


def checks(row: dict, spec: dict) -> dict:
    rim = spec["rim_radius_m"] * 1000
    return {"end_over_rim": row["end"]["lateral_mm"] < rim,
            "spout_down": row["end"]["spout_off_down_deg"] <= spec["max_spout_off_down_deg"],
            "carry_tilt": row["carrying"]["max_tilt_deg"] <= spec["max_carry_tilt_deg"],
            "spout_in_rim": row["pouring"]["max_off_axis_mm"] < rim}


GOAL_CHECKS = ("end_over_rim", "spout_down")
PATH_CHECKS = ("carry_tilt", "spout_in_rim")


def collision_free(e: dict) -> bool:
    return max(e["max_penetration_mm"].values()) <= COLLISION_TOL_MM


def success(e: dict) -> bool:
    return e["goal_pass"] and e["path_pass"] and collision_free(e)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))


def evaluate(plan_path, env, scene, spec) -> dict:
    plan = np.load(plan_path)
    path, ids = plan["path"], plan["stage_ids"]
    active, passive = spec["active"], spec["passive"]
    sym = load_symbols(scene.asset_dirs[active])
    sym_p = load_symbols(scene.asset_dirs[passive])
    T_p = plan[f"T0_{passive}"]
    opening = T_p[:3, :3] @ sym_p.points[spec["opening"]] + T_p[:3, 3]
    up_world = T_p[:3, :3] @ sym_p.axes[spec["up_axis"]]
    tip_body, up_body = sym.points[spec["spout"]], sym.axes[spec["up_axis"]]
    pour_body, tilt_axis_body = sym.axes[spec["pour_axis"]], sym.axes[spec["tilt_axis"]]
    attached = AttachedObject(plan["T_ee_body"])
    kin = AttachedArmKinematics(env, attached, env.objects[active].joints[0], active)
    rim, limit = spec["rim_radius_m"], spec["max_carry_tilt_deg"]

    pairs, totals, hits = {}, {"self": 0.0, "table": 0.0, "passive": 0.0, "other": 0.0}, 0
    tilts, off_axis, tips, levels = [], [], [], []
    carried = path[ids != min(ids.tolist())]
    motion = densify(np.vstack([path[ids == min(ids.tolist())][-1], carried]), TRACK_STEP_RAD)
    for q in motion:
        T = attached.body_pose(kin.fk(q))
        tip = T[:3, :3] @ tip_body + T[:3, 3]
        d = tip - opening
        tips.append(tip)
        off_axis.append(float(np.linalg.norm(d - (d @ up_world) * up_world)))
        tilts.append(angle_deg(T[:3, :3] @ up_body, up_world))
        levels.append(abs(90.0 - angle_deg(T[:3, :3] @ tilt_axis_body, up_world)))
        found = contacts(kin, active)
        hits += bool(found)
        for a, b, depth in found:
            pairs[f"{a} <-> {b}"] = max(pairs.get(f"{a} <-> {b}", 0.0), depth)
            group = collision_group(a, b, passive)
            totals[group] = max(totals[group], depth)
    tilts, off_axis, tips = np.array(tilts), np.array(off_axis), np.array(tips)
    carrying, pouring = split_motion(tilts, off_axis, rim, limit)
    T_end = attached.body_pose(kin.fk(motion[-1]))
    pour_dir = T_end[:3, :3] @ pour_body
    land = stream_landing(tips[-1], pour_dir, opening, up_world)
    row = {"plan": str(plan_path),
           "carrying": {"max_tilt_deg": round(float(tilts[carrying].max()) if carrying.any() else 0.0, 1),
                        "tilt_at_quarters_deg": [round(float(tilts[int(f * (len(tilts) - 1))]), 1) for f in (0.25, 0.5, 0.75)]},
           "pouring": {"max_off_axis_mm": round(float(off_axis[pouring].max()) * 1000, 1) if pouring.any() else 0.0,
                       "spout_wander_mm": round(float(np.max(np.ptp(tips[pouring], axis=0))) / 2 * 1000, 1) if pouring.any() else 0.0,
                       "tilt_axis_off_level_deg": round(float(np.max(np.array(levels)[pouring])), 1) if pouring.any() else 0.0},
           "end": {"lateral_mm": round(float(off_axis[-1]) * 1000, 1),
                   "height_mm": round(float((tips[-1] - opening) @ up_world) * 1000, 1),
                   "tilt_deg": round(float(tilts[-1]), 1),
                   "spout_off_down_deg": round(angle_deg(pour_dir, -up_world), 1),
                   "stream_off_mm": None if land is None else round(float(np.linalg.norm(land - opening)) * 1000, 1)},
           "max_penetration_mm": {k: round(v * 1000, 1) for k, v in totals.items()},
           "pairs_max_depth_mm": {k: round(v * 1000, 1) for k, v in pairs.items()},
           "colliding_points": hits}
    row["checks"] = {k: bool(v) for k, v in checks(row, spec).items()}
    row["goal_pass"] = all(row["checks"][k] for k in GOAL_CHECKS)
    row["path_pass"] = all(row["checks"][k] for k in PATH_CHECKS)
    return row


def table(directory: Path, spec: dict, methods=None) -> tuple[list[tuple], dict]:
    """One column per method, over the seeds that ran: <method>_s<seed>.run.json
    says each ran, and .eval.json is there if it found a plan. Pass counts are out
    of the runs, box hits out of the plans found."""
    runs: dict = {}
    for m in methods or METHODS:
        runs[m] = []
        for f in sorted(directory.glob(f"{m}_s*.run.json")):
            ev = f.with_name(f.name.replace(".run.json", ".eval.json"))
            runs[m].append({"run": f.name.split(".")[0], **json.loads(f.read_text()),
                            "eval": json.loads(ev.read_text()) if ev.exists() else None})

    runs = {m: rs for m, rs in runs.items() if rs}

    def count(fn):
        return [f"{sum(fn(r['eval']) for r in rs if r['eval'])}/{len(rs)}" for rs in runs.values()]

    def values(fn, digits=1, unit=""):
        out = []
        for rs in runs.values():
            vs = [fn(r["eval"]) for r in rs if r["eval"]]
            out.append(span(vs, digits, unit) if vs else "–")
        return out

    def found(fn):
        return [f"{sum(fn(r['eval']) for r in rs if r['eval'])}/{sum(bool(r['eval']) for r in rs)}" for rs in runs.values()]

    def box_depth(e, on_pot):
        return max((v for k, v in e["pairs_max_depth_mm"].items() if BOX in k and (spec["active"] in k) == on_pot), default=0.0)

    def arm_on_pot(e):
        """The arm into the held pot: grouped "other" with the box, so itemized apart from it."""
        return max((v for k, v in e["pairs_max_depth_mm"].items()
                    if collision_group(*k.split(" <-> "), spec["passive"]) == "other" and BOX not in k), default=0.0)

    rim_mm, limit = spec["rim_radius_m"] * 1000, spec["max_carry_tilt_deg"]
    rows = [("", *(METHODS[m] for m in runs)),
            ("Success: goal, path limits and no collision", *count(success)),
            ("Plan returned, with or without collisions", *count(lambda e: True)),
            (f"Collision-free (at most {COLLISION_TOL_MM:.0f} mm)", *count(collision_free)),
            ("Goal reached", *count(lambda e: e["goal_pass"])),
            ("Path limits held", *count(lambda e: e["path_pass"])),
            (f"Tilt while carrying (limit {limit:.0f}°)", *values(lambda e: e["carrying"]["max_tilt_deg"], 1, "°")),
            ("Tilt at 25 / 50 / 75% of the motion, first run",
             *[" / ".join(f"{v:.0f}°" for v in next(r["eval"]["carrying"]["tilt_at_quarters_deg"] for r in rs if r["eval"]))
               if any(r["eval"] for r in rs) else "–" for rs in runs.values()]),
            (f"Spout off the mug's axis while pouring (rim {rim_mm:.1f} mm)",
             *values(lambda e: e["pouring"]["max_off_axis_mm"], 1, " mm")),
            ("Spout wander while pouring", *values(lambda e: e["pouring"]["spout_wander_mm"], 1, " mm")),
            ("Final tilt", *values(lambda e: e["end"]["tilt_deg"], 1, "°")),
            ("Final spout height above the rim", *values(lambda e: e["end"]["height_mm"], 1, " mm")),
            (f"Final spout angle from straight down (limit {spec['max_spout_off_down_deg']:.0f}°)",
             *values(lambda e: e["end"]["spout_off_down_deg"], 1, "°")),
            (f"Stream off the mug's centre (rim {rim_mm:.1f} mm)", *values(lambda e: e["end"]["stream_off_mm"], 1, " mm")),
            ("Collision: table", *values(lambda e: e["max_penetration_mm"]["table"], 1, " mm")),
            ("Collision: mug", *values(lambda e: e["max_penetration_mm"]["passive"], 1, " mm")),
            ("Collision: robot with itself", *values(lambda e: e["max_penetration_mm"].get("self", 0.0), 1, " mm")),
            ("Collision: arm with the teapot", *values(arm_on_pot, 1, " mm"))]
    if (directory / "scene.scene.json").exists():    # make_pour_scene.py wrote this scene, with its box
        rows += [("Collision: box, plans that hit it", *found(lambda e: box_depth(e, True) > 0 or box_depth(e, False) > 0)),
                 ("Collision: box, teapot depth", *values(lambda e: box_depth(e, True), 1, " mm")),
                 ("Collision: box, gripper depth", *values(lambda e: box_depth(e, False), 1, " mm"))]
    rows.append(("Planning time", *[span([r["seconds"] for r in rs], 0, " s") if rs else "–" for rs in runs.values()]))
    return rows, runs


def violations(runs: dict, spec: dict) -> list[tuple]:
    """Only the task's constraints, at the planners' own settings: what the comparison
    reports. Anything else, such as accuracy or time, stays in the full table."""
    def found(fn):
        return [f"{sum(fn(r['eval']) for r in rs if r['eval'])}/{sum(bool(r['eval']) for r in rs)}" for rs in runs.values()]

    def worst(fn, unit):
        return [span([fn(r["eval"]) for r in rs if r["eval"]], 1, unit) if any(r["eval"] for r in rs) else "–" for rs in runs.values()]

    limit, rim_mm = spec["max_carry_tilt_deg"], spec["rim_radius_m"] * 1000
    return [("", *(METHODS[m] for m in runs)),
            ("Plans returned", *[f"{sum(bool(r['eval']) for r in rs)}/{len(rs)}" for rs in runs.values()]),
            (f"Tilt while carrying, worst (limit {limit:.0f}°)", *worst(lambda e: e["carrying"]["max_tilt_deg"], "°")),
            ("Tilt while carrying, plans over the limit", *found(lambda e: not e["checks"]["carry_tilt"])),
            (f"Spout off the mug's axis while pouring, worst (rim {rim_mm:.1f} mm)", *worst(lambda e: e["pouring"]["max_off_axis_mm"], " mm")),
            ("Spout while pouring, plans outside the rim", *found(lambda e: not e["checks"]["spout_in_rim"])),
            ("Collision, deepest", *worst(lambda e: max(e["max_penetration_mm"].values()), " mm")),
            (f"Collision, plans deeper than {COLLISION_TOL_MM:.0f} mm", *found(lambda e: not collision_free(e)))]


def summary(runs: dict, spec: dict) -> list[tuple]:
    """A rate and a median / max per limit: the short table. Success is out of the runs,
    since a run with no plan did not succeed, and the rest are over the plans returned.
    Medians and maxima are across plans, of each plan's worst value."""
    def found(fn):
        return [f"{sum(fn(r['eval']) for r in rs if r['eval'])}/{sum(bool(r['eval']) for r in rs)}" for rs in runs.values()]

    def spread(fn, unit):
        out = []
        for rs in runs.values():
            vs = [fn(r["eval"]) for r in rs if r["eval"]]
            out.append(f"{np.median(vs):.1f}{unit} / {max(vs):.1f}{unit}" if vs else "–")
        return out

    limit, rim_mm = spec["max_carry_tilt_deg"], spec["rim_radius_m"] * 1000
    return [("", *(METHODS[m] for m in runs)),
            ("Success rate", *[f"{sum(success(r['eval']) for r in rs if r['eval'])}/{len(rs)}" for rs in runs.values()]),
            (f"Tilt while carrying, median / max (limit {limit:.0f}°)", *spread(lambda e: e["carrying"]["max_tilt_deg"], "°")),
            ("Path constraint violation rate", *found(lambda e: not e["path_pass"])),
            (f"Spout off the mug's axis while pouring, median / max (rim {rim_mm:.1f} mm)",
             *spread(lambda e: e["pouring"]["max_off_axis_mm"], " mm")),
            (f"Collision rate, deeper than {COLLISION_TOL_MM:.0f} mm", *found(lambda e: not collision_free(e))),
            ("Deepest collision, median / max", *spread(lambda e: max(e["max_penetration_mm"].values()), " mm"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, metavar="JSON")
    ap.add_argument("--plan", nargs="+")
    ap.add_argument("--table", type=Path, help="directory of graded runs: write the comparison")
    ap.add_argument("--out", type=Path, help="with --table: <out>.md, .full.md, .summary.md and .json")
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), help="with --table: these columns, default all that ran")
    ap.add_argument("--robot", default="UR5e")
    add_scene_arg(ap)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    if args.table:
        rows, runs = table(args.table, spec, args.methods)
        text = markdown_table(violations(runs, spec))
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.with_suffix(".md").write_text(
                "# Pour-tea constraint violations, kinematics only\n\nGenerated by scripts/eval_pour.py.\n\n" + text + "\n")
            args.out.with_suffix(".full.md").write_text(
                "# Pour-tea plan comparison, kinematics only\n\nGenerated by scripts/eval_pour.py.\n\n"
                + markdown_table(rows) + "\n")
            args.out.with_suffix(".summary.md").write_text(
                "# Pour-tea summary, kinematics only\n\nGenerated by scripts/eval_pour.py.\n\n"
                + markdown_table(summary(runs, spec)) + "\n\n"
                f"A success reaches the goal, holds the path limits and has no collision deeper than {COLLISION_TOL_MM:.0f} mm, "
                "out of all runs. The other rows are over the plans returned. A path constraint violation is "
                "tilt over the limit while carrying or the spout outside the rim while pouring. Medians and maxima "
                "are across plans, of each plan's worst value.\n")
            args.out.with_suffix(".json").write_text(json.dumps(runs, indent=2) + "\n")
        return
    scene = load_scene(args.scene, args.grounding)
    env, _ = make_env(scene, robot=args.robot, has_renderer=False)
    for plan_path in args.plan:
        row = evaluate(plan_path, env, scene, spec)
        Path(plan_path).with_suffix(".eval.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"[eval-pour] {Path(plan_path).stem}: goal={row['goal_pass']} path={row['path_pass']} "
              f"{row['carrying']} {row['pouring']} {row['end']} {row['max_penetration_mm']}")
    env.close()


if __name__ == "__main__":
    main()
