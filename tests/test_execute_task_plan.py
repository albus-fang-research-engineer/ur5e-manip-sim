"""execute_task_plan's end-state numbers on synthetic geometry: a rod
standing in a cup, lying across the rim, and tilted with one end dipping
beside the cup (deep, but off the axis)."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.execute_task_plan import end_state  # noqa: E402
from manip_sim.tsr import make_pose  # noqa: E402

ROD = np.c_[np.zeros(21), np.zeros(21), np.linspace(-0.07, 0.07, 21)]   # 14 cm, body z
T_CUP = make_pose([0.3, 0.1, 0.8], np.eye(3))
OPENING = np.array([0.0, 0.0, 0.1])                                      # rim plane, on the axis
UP = np.array([0.0, 0.0, 1.0])


def test_rod_standing_inside():
    T_rod = make_pose([0.31, 0.1, 0.8 + 0.08], np.eye(3))
    s = end_state(T_rod, ROD, T_CUP, OPENING, UP)
    assert s["long_axis_tilt_from_passive_up_deg"] < 1.0
    assert abs(s["lowest_point_below_target_plane_m"] - 0.09) < 1e-6
    assert abs(s["lowest_point_from_target_axis_m"] - 0.01) < 1e-6


def test_rod_lying_across_rim():
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)      # body z -> world x
    T_rod = make_pose([0.3, 0.1, 0.8 + 0.11], Ry)
    s = end_state(T_rod, ROD, T_CUP, OPENING, UP)
    assert s["long_axis_tilt_from_passive_up_deg"] > 89.0
    assert s["lowest_point_below_target_plane_m"] < 0.0           # above the rim


def test_end_dipping_beside_the_cup_is_deep_but_off_axis():
    """Centre near the opening, tilted 70 deg: one end below the rim plane
    but well outside a 4 cm rim radius."""
    T_rod = make_pose([0.3, 0.1, 0.8 + 0.11], R.from_euler("y", 70, degrees=True).as_matrix())
    s = end_state(T_rod, ROD, T_CUP, OPENING, UP)
    assert s["lowest_point_below_target_plane_m"] > 0.01
    assert s["lowest_point_from_target_axis_m"] > 0.06


def test_passive_up_follows_passive_pose():
    """A tipped-over cup: depth is measured along the cup's up."""
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # body z -> world -y
    T_tipped = make_pose([0.3, 0.1, 0.8], Rx)
    T_rod = T_tipped @ make_pose([0.0, 0.0, 0.08], np.eye(3))
    s = end_state(T_rod, ROD, T_tipped, OPENING, UP)
    assert s["long_axis_tilt_from_passive_up_deg"] < 1.0
    assert abs(s["lowest_point_below_target_plane_m"] - 0.09) < 1e-6
    assert s["lowest_point_from_target_axis_m"] < 1e-6
