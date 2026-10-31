"""compare_plans' classification and violations on small hand-built rows."""

import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_plans import (README_END, README_START, load_runs, physics_outcome,  # noqa: E402
                                   replace_block, span, split_run, violations)

SPEC = {"max_tilt_deg": 10.0, "max_path_tilt_deg": 10.0, "min_depth_m": 0.04, "rim_radius_m": 0.0423}


def metrics(depth, off, slip=20.0):
    return {"released_end": {"lowest_point_below_target_plane_m": depth, "lowest_point_from_target_axis_m": off},
            "segments": {"a": {"slip_max_mm": slip}, "reach": {}}}


def evaluation(tilt=2.0, depth=50.0, inside_tilt=3.0, inside_rim=True, right_end=True):
    return {"goal": {"tilt_deg": tilt, "depth_mm": depth}, "inserted": {"max_tilt_deg": inside_tilt},
            "checks": {"inside_rim": inside_rim, "first_end_lower": right_end}}


def test_split_run():
    assert split_run("tsr5more_s3") == ("tsr5more", 3)
    with pytest.raises(ValueError):
        split_run("omni")


def test_load_runs_keeps_only_the_named_methods(tmp_path):
    for name in ("omni_s0", "same5_s0", "tsr5_s0"):
        (tmp_path / f"{name}.report.json").write_text(json.dumps({"status": "failed"}))
    assert [r["run"] for r in load_runs(tmp_path, tmp_path, {}, SPEC, ["omni", "same5"])] == ["omni_s0", "same5_s0"]
    assert len(load_runs(tmp_path, tmp_path, {}, SPEC)) == 3


def test_physics_outcomes():
    assert physics_outcome(metrics(0.0656, 0.03), 0.0423) == "in_cup"
    assert physics_outcome(metrics(0.0745, 0.0689), 0.0423) == "beside"
    assert physics_outcome(metrics(0.0748, 0.44, slip=610.0), 0.0423) == "lost"
    assert physics_outcome({"segments": {"a": {"slip_max_mm": 5.0}}}, 0.0423) == "lost"
    assert physics_outcome(None, 0.0423) is None


def test_violations_are_the_amount_past_the_spec():
    clean = violations(evaluation(), SPEC)
    assert clean["goal_tilt_over_deg"] == 0.0 and clean["goal_depth_short_mm"] == 0.0
    assert clean["inside_tilt_over_deg"] == 0.0 and not clean["goal_wrong_end"]
    v = violations(evaluation(tilt=11.96, depth=35.0, inside_tilt=63.69, inside_rim=False, right_end=False), SPEC)
    assert v["goal_tilt_over_deg"] == pytest.approx(1.96) and v["goal_depth_short_mm"] == 5.0
    assert v["inside_tilt_over_deg"] == pytest.approx(53.69) and v["goal_outside_rim"] and v["goal_wrong_end"]


def test_no_point_inside_the_rim_is_not_a_violation():
    assert violations(evaluation(inside_tilt=None), SPEC)["inside_tilt_over_deg"] == 0.0


def test_span():
    assert span([1.0, 2.5], 1, "°") == "1.0–2.5°"
    assert span([45.04, 45.01], 1, " mm") == "45.0 mm"
    assert span([None]) == "–"


def test_replace_block_keeps_everything_around_it():
    text = f"before\n{README_START}\nold\n{README_END}\nafter\n"
    assert replace_block(text, "new") == f"before\n{README_START}\nnew\n{README_END}\nafter\n"
