"""The pour-tea grade on small hand-built rows."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_pour import GOAL_CHECKS, PATH_CHECKS, angle_deg, checks, split_motion, summary, table, violations  # noqa: E402

SPEC = json.loads((ROOT / "tasks/specs/pour_tea.json").read_text())


def row(lateral=10.0, height=70.0, down=5.0, stream=20.0, carry=5.0, off=30.0, wander=2.0):
    return {"end": {"lateral_mm": lateral, "height_mm": height, "spout_off_down_deg": down, "stream_off_mm": stream},
            "carrying": {"max_tilt_deg": carry},
            "pouring": {"max_off_axis_mm": off, "spout_wander_mm": wander}}


def test_a_plan_inside_every_limit_passes_all():
    assert all(checks(row(), SPEC).values())


@pytest.mark.parametrize("kw, failed", [
    (dict(lateral=50.0), "end_over_rim"),
    (dict(down=16.0), "spout_down"),
    (dict(carry=15.5), "carry_tilt"), (dict(off=45.0), "spout_in_rim")])
def test_each_limit_fails_alone(kw, failed):
    c = checks(row(**kw), SPEC)
    assert [k for k, v in c.items() if not v] == [failed]
    assert failed in GOAL_CHECKS + PATH_CHECKS


@pytest.mark.parametrize("stream", [60.0, None])
def test_where_the_stream_lands_does_not_decide_the_goal(stream):
    assert all(checks(row(stream=stream), SPEC).values())


def test_a_tilted_spout_outside_the_rim_counts_once():
    import numpy as np
    tilts = np.array([5.0, 40.0, 5.0, 50.0, 60.0, 90.0])
    off = np.array([300.0, 200.0, 40.0, 45.0, 30.0, 10.0])
    carrying, pouring = split_motion(tilts, off, 44.1, 15.0)
    assert carrying.tolist() == [True, True, False, False, False, False]
    assert pouring.tolist() == [False, False, False, True, True, True]


def test_a_spout_that_never_reaches_the_rim_is_all_carrying():
    import numpy as np
    carrying, pouring = split_motion(np.array([5.0, 40.0]), np.array([300.0, 200.0]), 44.1, 15.0)
    assert carrying.all() and not pouring.any()


def test_only_the_rim_limits_the_spout():
    assert all(checks(row(height=200.0, wander=40.0), SPEC).values())


def test_angle():
    import numpy as np
    assert angle_deg(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])) == pytest.approx(45.0)


def graded(pairs, other):
    return {"goal_pass": True, "path_pass": True, "carrying": {"max_tilt_deg": 5.0, "tilt_at_quarters_deg": [1.0, 2.0, 3.0]},
            "pouring": {"max_off_axis_mm": 20.0, "spout_wander_mm": 2.0},
            "end": {"tilt_deg": 95.0, "height_mm": 70.0, "spout_off_down_deg": 5.0, "stream_off_mm": 20.0},
            "max_penetration_mm": {"table": 0.0, "passive": 0.0, "other": other}, "pairs_max_depth_mm": pairs}


def write_runs(directory):
    (directory / "omni_s0.run.json").write_text(json.dumps({"seconds": 10, "exit": 0, "failure": None}))
    (directory / "omni_s0.eval.json").write_text(json.dumps(graded(
        {"obstacle_main <-> teapot_main": 3.0, "gripper0_right_finger <-> obstacle_main": 9.0, "mug_main <-> teapot_main": 7.0,
         "robot0_forearm_link <-> teapot_main": 18.0}, 18.0)))
    (directory / "omni_s1.run.json").write_text(json.dumps({"seconds": 10, "exit": 1, "failure": "no plan"}))


def test_box_rows_count_found_plans_and_split_teapot_from_gripper(tmp_path):
    write_runs(tmp_path)
    (tmp_path / "scene.scene.json").write_text("{}")
    rows = {r[0]: r[1] for r in table(tmp_path, SPEC)[0]}
    assert rows["Plan returned, with or without collisions"] == "1/2"
    assert rows["Collision-free (at most 1 mm)"] == "0/2"
    assert rows["Success: goal, path limits and no collision"] == "0/2"
    assert rows["Collision: box, plans that hit it"] == "1/1"
    assert rows["Collision: box, teapot depth"] == "3.0 mm"
    assert rows["Collision: box, gripper depth"] == "9.0 mm"
    assert rows["Collision: arm with the teapot"] == "18.0 mm"


def test_a_scene_without_a_box_has_no_box_rows(tmp_path):
    write_runs(tmp_path)
    assert not [r[0] for r in table(tmp_path, SPEC)[0] if "box" in r[0]]


def test_the_violation_table_has_only_the_constraints(tmp_path):
    write_runs(tmp_path)
    e = json.loads((tmp_path / "omni_s0.eval.json").read_text())
    e["checks"] = {"carry_tilt": True, "spout_in_rim": True}
    (tmp_path / "omni_s0.eval.json").write_text(json.dumps(e))
    rows = {r[0]: r[1] for r in violations(table(tmp_path, SPEC)[1], SPEC)}
    assert set(rows) == {"", "Plans returned", "Tilt while carrying, worst (limit 15°)", "Tilt while carrying, plans over the limit",
                         "Spout off the mug's axis while pouring, worst (rim 44.1 mm)", "Spout while pouring, plans outside the rim",
                         "Collision, deepest", "Collision, plans deeper than 1 mm"}
    assert rows["Plans returned"] == "1/2"
    assert rows["Collision, plans deeper than 1 mm"] == "1/1"


def test_the_summary_counts_no_plan_as_a_failure_and_takes_medians_over_plans():
    def plan(tilt, off, depth, path_pass=True):
        return {"carrying": {"max_tilt_deg": tilt}, "pouring": {"max_off_axis_mm": off}, "goal_pass": True, "path_pass": path_pass,
                "max_penetration_mm": {"self": 0.0, "table": 0.0, "passive": depth, "other": 0.0}}
    runs = {"omni": [{"eval": plan(1.0, 10.0, 0.0)}, {"eval": plan(3.0, 20.0, 5.0)},
                     {"eval": plan(20.0, 60.0, 0.5, path_pass=False)}, {"eval": None}]}
    rows = {r[0]: r[1] for r in summary(runs, SPEC)}
    assert rows["Success rate"] == "1/4"
    assert rows["Tilt while carrying, median / max (limit 15°)"] == "3.0° / 20.0°"
    assert rows["Path constraint violation rate"] == "1/3"
    assert rows["Spout off the mug's axis while pouring, median / max (rim 44.1 mm)"] == "20.0 mm / 60.0 mm"
    assert rows["Collision rate, deeper than 1 mm"] == "1/3"
    assert rows["Deepest collision, median / max"] == "0.5 mm / 5.0 mm"
