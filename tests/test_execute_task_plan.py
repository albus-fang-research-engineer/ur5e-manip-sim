"""execute_task_plan's end-state numbers on synthetic geometry: a rod
standing in a cup vs the same rod lying across the rim."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.execute_task_plan import end_state  # noqa: E402
from manip_sim.tsr import make_pose  # noqa: E402

th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
CUP = np.vstack([np.c_[0.04 * np.cos(th), 0.04 * np.sin(th), np.full(32, h)]
                 for h in (0.0, 0.1)])                       # 10 cm tall, r 4 cm
ROD = np.c_[np.zeros(21), np.zeros(21), np.linspace(-0.07, 0.07, 21)]   # 14 cm, body z
T_CUP = make_pose([0.3, 0.1, 0.8], np.eye(3))


def test_rod_standing_inside():
    T_rod = make_pose([0.31, 0.1, 0.8 + 0.08], np.eye(3))
    s = end_state(T_rod, ROD, T_CUP, CUP, [0, 0, 1])
    assert s["long_axis_tilt_from_passive_up_deg"] < 1.0
    assert abs(s["lowest_point_below_passive_top_m"] - 0.09) < 1e-6
    assert abs(s["centroid_offset_from_passive_axis_m"] - 0.01) < 1e-6


def test_rod_lying_across_rim():
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)      # body z -> world x
    T_rod = make_pose([0.3, 0.1, 0.8 + 0.11], Ry)
    s = end_state(T_rod, ROD, T_CUP, CUP, [0, 0, 1])
    assert s["long_axis_tilt_from_passive_up_deg"] > 89.0
    assert s["lowest_point_below_passive_top_m"] < 0.0              # above the rim


def test_passive_up_follows_passive_pose():
    """A tipped-over cup: 'below the top' is measured along the cup's up."""
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # body z -> world -y
    T_tipped = make_pose([0.3, 0.1, 0.8], Rx)
    T_rod = T_tipped @ make_pose([0.0, 0.0, 0.08], np.eye(3))
    s = end_state(T_rod, ROD, T_tipped, CUP, [0, 0, 1])
    assert s["long_axis_tilt_from_passive_up_deg"] < 1.0
    assert abs(s["lowest_point_below_passive_top_m"] - 0.09) < 1e-6
