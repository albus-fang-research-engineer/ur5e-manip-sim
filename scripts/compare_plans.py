"""Build the representation study's tables from the per-plan results, so nothing
is retyped: <plan>.report.json, <plan>.eval.json (scripts/eval_plan.py) and
<plan>_exec.json (scripts/execute_task_plan.py), for runs named <method>_s<seed>.

    python3 scripts/compare_plans.py --spec tasks/specs/insert_marker.json \\
        --plans outputs/plans/compare --metrics outputs/metrics/vlm \\
        --seconds outputs/plans/compare/plan_task_seconds.json \\
        --weight-sweep outputs/plans/omnimanip_weight_sweep \\
        --separation-sweep outputs/plans/omnimanip_separation_sweep \\
        --out outputs/comparison/insert_marker --readme README.md

Writes <out>.json (every run's row) and <out>.md (the tables). With --readme, the
summary also replaces the block between the markers in that file.

Violations are measured against the one written spec, never a method's own
constraints: "over limit" is the amount past the spec, and the goal deviations
are reported as well so a pass can be told from a near miss.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

METHODS = {"tsr5": "Ours, 5°", "tsr1": "Ours, 1°", "tsr10": "Ours, 10°",
           "tsr5more": "Ours, 5°, more samples", "rekep": "ReKep", "omni": "OmniManip"}
LOST_SLIP_MM = 100.0              # past this the object is not in the hand any more
README_START, README_END = "<!-- comparison:start -->", "<!-- comparison:end -->"


def split_run(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_s(\d+)", name)
    if match is None:
        raise ValueError(f"run '{name}' is not named <method>_s<seed>")
    return match.group(1), int(match.group(2))


def physics_outcome(metrics: dict | None, rim_radius_m: float) -> str | None:
    """in_cup, beside or lost, from where the object ended after release."""
    if metrics is None:
        return None
    end = metrics.get("released_end")
    slip = max((s.get("slip_max_mm") or 0.0 for s in metrics["segments"].values()), default=0.0)
    if end is None or slip > LOST_SLIP_MM:
        return "lost"
    inside = end["lowest_point_below_target_plane_m"] > 0 \
        and end["lowest_point_from_target_axis_m"] < rim_radius_m
    return "in_cup" if inside else "beside"


def violations(ev: dict, spec: dict) -> dict:
    """How far an evaluated plan is past the shared spec, at the goal and inside
    the rim. Collisions stay separate: the spec has no allowance for them."""
    goal, inserted = ev["goal"], ev["inserted"]
    return {"goal_tilt_over_deg": round(max(0.0, goal["tilt_deg"] - spec["max_tilt_deg"]), 2),
            "goal_depth_short_mm": round(max(0.0, spec["min_depth_m"] * 1000 - goal["depth_mm"]), 1),
            "goal_outside_rim": not ev["checks"]["inside_rim"],
            "goal_wrong_end": not ev["checks"]["first_end_lower"],
            "inside_tilt_over_deg": round(max(0.0, (inserted["max_tilt_deg"] or 0.0) - spec["max_tilt_deg"]), 2)}


def own_residual(method: str, report: dict) -> str | None:
    """What the method reports on its own constraints, in its own units."""
    stages = report.get("stages") or []
    if not stages or report.get("status") != "ok":
        return None
    if method == "omni":
        solve = next(s["solve"] for s in stages if "solve" in s)
        return f"{solve['position_error_mm']} mm, {solve['angle_error_deg']}°"
    if method == "rekep":
        worst = max(max(s["subgoal_violations"], default=0.0) for s in stages)
        return f"cost {worst:.4f}"
    return "path excess " + f"{max(s['planning']['max_path_excess'] for s in stages):.4f}"


def load_runs(plans: Path, metrics: Path, seconds: dict, spec: dict) -> list[dict]:
    runs = []
    for report_path in sorted(plans.glob("*.report.json")):
        name = report_path.name[:-len(".report.json")]
        method, seed = split_run(name)
        report = json.loads(report_path.read_text())
        eval_path, exec_path = plans / f"{name}.eval.json", metrics / f"{name}_exec.json"
        ev = json.loads(eval_path.read_text()) if eval_path.exists() else None
        ex = json.loads(exec_path.read_text()) if exec_path.exists() else None
        failure = report.get("failure") or {}
        runs.append({"run": name, "method": method, "seed": seed, "status": report.get("status"),
                     "failure": f"{failure['stage']}: {failure['kind']}" if failure else None,
                     "seconds": report.get("seconds", seconds.get(name)),
                     "own_residual": own_residual(method, report), "eval": ev,
                     "violations": violations(ev, spec) if ev else None,
                     "physics": physics_outcome(ex, spec["rim_radius_m"]),
                     "released": ex.get("released_end") if ex else None,
                     "grasp_contacts": ex.get("grasp_contacts") if ex else None,
                     "slip_max_mm": max((s.get("slip_max_mm") or 0.0 for s in ex["segments"].values()),
                                        default=0.0) if ex else None})
    return runs


def span(values, digits: int = 1, unit: str = "") -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    lo, hi = min(values), max(values)
    text = f"{lo:.{digits}f}" if f"{lo:.{digits}f}" == f"{hi:.{digits}f}" else f"{lo:.{digits}f}–{hi:.{digits}f}"
    return text + unit


def count(hits: int, total: int) -> str:
    return f"{hits}/{total}" if total else "–"


def summary_rows(runs: list[dict]) -> list[tuple[str, ...]]:
    groups = {m: [r for r in runs if r["method"] == m] for m in METHODS if any(r["method"] == m for r in runs)}
    columns = list(groups.values())

    def row(label, fn):
        return (label, *(fn(g, [r for r in g if r["eval"]]) for g in columns))

    def at(f):
        return lambda _g, e: span(r["eval"]["transport"]["rest_up_tilt_deg_at_travel"][f] for r in e) + "°"

    def in_cup(g):
        return [r for r in g if r["physics"] == "in_cup"]

    def collisions(_g, e):
        hit = [r for r in e if any(r["eval"]["max_penetration_mm"].values())]
        depth = max((max(r["eval"]["max_penetration_mm"].values()) for r in hit), default=0.0)
        return count(len(hit), len(e)) + (f" ({depth:.1f} mm)" if hit else "")

    return [
        ("", *(METHODS[m] for m in groups)),
        row("Plans found", lambda g, e: count(len(e), len(g))),
        row("Meets the goal spec", lambda g, e: count(sum(r["eval"]["goal_pass"] for r in e), len(e))),
        row("**Goal deviation:** final tilt", lambda g, e: span((r["eval"]["goal"]["tilt_deg"] for r in e), 1, "°")),
        row("Lower end off the cup's axis", lambda g, e: span((r["eval"]["goal"]["off_axis_mm"] for r in e), 1, " mm")),
        row("Depth below the rim", lambda g, e: span((r["eval"]["goal"]["depth_mm"] for r in e), 1, " mm")),
        row("**Constraint violation:** goal tilt over limit",
            lambda g, e: count(sum(r["violations"]["goal_tilt_over_deg"] > 0 for r in e), len(e))),
        row("Worst tilt inside the rim", lambda g, e: span((r["eval"]["inserted"]["max_tilt_deg"] for r in e), 1, "°")),
        row("Tilt over limit inside the rim",
            lambda g, e: count(sum(r["violations"]["inside_tilt_over_deg"] > 0 for r in e), len(e))),
        row("Plans with a collision (deepest)", collisions),
        row("**Transport:** tilt at 25% of travel", at("25%")),
        row("Tilt at 50%", at("50%")),
        row("Tilt at 75%", at("75%")),
        row("Rotation done by half travel",
            lambda g, e: span((r["eval"]["transport"]["rotation_done_at_half_travel_pct"] for r in e), 0, "%")),
        row("**Path:** end-effector travel", lambda g, e: span((r["eval"]["ee_travel_m"] for r in e), 2, " m")),
        row("Joint travel", lambda g, e: span((r["eval"]["joint_travel_rad"] for r in e), 1, " rad")),
        row("Planning time", lambda g, e: span((r["seconds"] for r in e), 1, " s")),
        row("**Physics:** in cup", lambda g, e: count(len(in_cup(e)), len(e))),
        row("Beside the cup", lambda g, e: count(sum(r["physics"] == "beside" for r in e), len(e))),
        row("Object lost from the hand", lambda g, e: count(sum(r["physics"] == "lost" for r in e), len(e))),
        row("Tilt after release, in cup", lambda g, e: span(
            (r["released"]["long_axis_tilt_from_passive_up_deg"] for r in in_cup(e)), 1, "°")),
    ]


def markdown_table(rows) -> str:
    rows = [tuple(str(c) for c in r) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def per_run_rows(runs) -> list[tuple]:
    rows = [("Run", "Goal spec", "Tilt", "Off axis mm", "Depth mm", "Worst tilt inside", "Collision mm",
             "Own residual", "Seconds", "Physics", "Contacts", "Slip mm", "Failure")]
    for r in sorted(runs, key=lambda r: (list(METHODS).index(r["method"]) if r["method"] in METHODS else 99, r["seed"])):
        e = r["eval"]
        rows.append((r["run"], "–" if not e else ("pass" if e["goal_pass"] else "**fail**"),
                     *(("–",) * 5 if not e else (e["goal"]["tilt_deg"], e["goal"]["off_axis_mm"], e["goal"]["depth_mm"],
                                                 e["inserted"]["max_tilt_deg"], max(e["max_penetration_mm"].values()))),
                     r["own_residual"] or "–", r["seconds"] if r["seconds"] is not None else "–",
                     r["physics"] or "–", r["grasp_contacts"] if r["grasp_contacts"] is not None else "–",
                     r["slip_max_mm"] if r["slip_max_mm"] is not None else "–", r["failure"] or "–"))
    return rows


def sweep_rows(directory: Path, key_header: tuple, key) -> list[tuple]:
    """One row per plan in a sweep, from each report and its grade if it has one."""
    rows = [key_header + ("Result", "Final tilt", "Off axis mm", "Depth mm", "Target error")]
    entries = [(key(json.loads(p.read_text()), p), p) for p in directory.glob("*.report.json")]
    for k, path in sorted(entries, key=lambda e: e[0]):
        report = json.loads(path.read_text())
        name = path.name[:-len(".report.json")]
        ev_path = directory / f"{name}.eval.json"
        ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
        solve = next((s["solve"] for s in report.get("stages", []) if "solve" in s), None)
        failure = report.get("failure") or {}
        rows.append((*k, "ok" if report["status"] == "ok" else f"failed, {failure['stage']}: {failure['kind']}",
                     *(("–",) * 3 if not ev else (f"{ev['goal']['tilt_deg']}°", ev["goal"]["off_axis_mm"], ev["goal"]["depth_mm"])),
                     f"{solve['position_error_mm']} mm, {solve['angle_error_deg']}°" if solve else "–"))
    return rows


def replace_block(text: str, block: str) -> str:
    start, end = text.index(README_START) + len(README_START), text.index(README_END)
    return text[:start] + "\n" + block + "\n" + text[end:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--plans", required=True, type=Path)
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--seconds", type=Path, help="planning seconds for planners whose report has none")
    ap.add_argument("--weight-sweep", type=Path)
    ap.add_argument("--separation-sweep", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--readme", type=Path)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    seconds = json.loads(args.seconds.read_text()) if args.seconds else {}
    runs = load_runs(args.plans, args.metrics, seconds, spec)
    summary = markdown_table(summary_rows(runs))
    sections = [f"# Representation study: {spec['task']}", "Generated by scripts/compare_plans.py.",
                "## Summary", summary, "## Every run", markdown_table(per_run_rows(runs))]
    if args.weight_sweep:
        def weights(report, _path):
            w = report["config"]["weights"]
            return (w["lambda_rot"], w["lambda_trans"])
        sections += ["## OmniManip weight sweep", markdown_table(sweep_rows(args.weight_sweep, ("λ_rot", "λ_trans"), weights))]
    if args.separation_sweep:
        def separation(_report, path):
            return (int(re.search(r"(\d+)", path.name).group(1)) / 100,)
        sections += ["## OmniManip separation sweep (kinematics only)",
                     markdown_table(sweep_rows(args.separation_sweep, ("Separation m",), separation))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n\n".join(sections) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"spec": spec, "n_by_method": Counter(r["method"] for r in runs), "runs": runs}, indent=2) + "\n")
    if args.readme:
        args.readme.write_text(replace_block(args.readme.read_text(), summary))
    print(f"[compare] {len(runs)} runs -> {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
