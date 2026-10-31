"""eval_plan's grading on synthetic geometry: tilt does not depend on which end
is named first, an object lying on the table is not "inside", and collisions
land in the right group."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_plan import (collision_group, end_state, goal_checks, is_inside,  # noqa: E402
                               transport_profile)

OPENING = np.array([0.0, 0.25, 0.9])
SPEC = {"max_tilt_deg": 10.0, "min_depth_m": 0.04, "rim_radius_m": 0.0423}


def rod(low, tilt_deg=0.0, length=0.12):
    t = np.radians(tilt_deg)
    return np.asarray(low, float), np.asarray(low, float) + length * np.array([np.sin(t), 0, np.cos(t)])


def test_tilt_does_not_depend_on_which_end_is_first():
    low, high = rod([0, 0.25, 0.85], tilt_deg=6.0)
    a = end_state(low, high, OPENING)
    b = end_state(high, low, OPENING)
    assert np.isclose(a["tilt_deg"], 6.0) and np.isclose(b["tilt_deg"], 6.0)
    assert a["lower"] == "a" and b["lower"] == "b"


def test_depth_and_off_axis_use_the_lower_end():
    low, high = rod([0.01, 0.25, 0.85])
    s = end_state(high, low, OPENING)
    assert np.isclose(s["depth_m"], 0.05) and np.isclose(s["off_axis_m"], 0.01)


def test_object_on_the_table_is_below_the_rim_but_not_inside():
    low, high = rod([0.0, -0.25, 0.81], tilt_deg=90.0)
    s = end_state(low, high, OPENING)
    assert s["depth_m"] > 0 and not is_inside(s, SPEC["rim_radius_m"])


def test_goal_checks_fail_each_tolerance_alone():
    ok = {"lower": "a", "tilt_deg": 6.0, "off_axis_m": 0.01, "depth_m": 0.05}
    assert all(goal_checks(ok, SPEC).values())
    assert not goal_checks({**ok, "lower": "b"}, SPEC)["first_end_lower"]
    assert not goal_checks({**ok, "tilt_deg": 11.0}, SPEC)["tilt"]
    assert not goal_checks({**ok, "depth_m": 0.03}, SPEC)["depth"]
    assert not goal_checks({**ok, "off_axis_m": 0.05}, SPEC)["inside_rim"]


def test_transport_profile_tells_turning_all_the_way_from_turning_late():
    travel = np.linspace(0.0, 0.5, 101)
    steady = transport_profile(travel, np.linspace(0, 90, 101), np.linspace(0, 90, 101))
    assert steady["rest_up_tilt_deg_at_travel"]["50%"] == 45.0
    assert steady["rotation_done_at_half_travel_pct"] == 50.0
    late = np.r_[np.zeros(81), np.linspace(0, 90, 20)]
    carried_level = transport_profile(travel, late, late)
    assert carried_level["rest_up_tilt_deg_at_travel"]["75%"] == 0.0
    assert carried_level["rotation_done_at_half_travel_pct"] == 0.0
    assert carried_level["max_rest_up_tilt_deg"] == 90.0


def test_collision_groups():
    assert collision_group("gripper0_right_left_outer_knuckle", "robot0_forearm_link", "cup") == "self"
    assert collision_group("robot0_wrist_1_link", "table", "cup") == "table"
    assert collision_group("cup_main", "marker_main", "cup") == "passive"
    assert collision_group("marker_main", "mug_main", "cup") == "other"
