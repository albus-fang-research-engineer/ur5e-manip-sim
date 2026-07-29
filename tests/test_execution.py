"""Execution-layer tests for the pieces that need no simulator: path
densification, the slip metric, and the controller part-dict contract
(README pin #4: whole-dict replacement, gripper subdict preserved). The
live tracking loop is exercised by scripts/execute_pour_tea.py."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from manip_sim.execution import (
    SlipMonitor,
    densify,
    joint_position_arm_part,
)
from manip_sim.tsr import make_pose


def test_densify_bounds_per_joint_step_and_keeps_endpoints():
    path = np.array([[0.0, 0.0], [0.10, -0.30], [0.10, 0.10]])
    d = densify(path, max_joint_step=0.05)
    assert np.allclose(d[0], path[0]) and np.allclose(d[-1], path[-1])
    for w in path:                          # planner configs all survive
        assert np.min(np.linalg.norm(d - w, axis=1)) < 1e-12
    steps = np.abs(np.diff(d, axis=0))
    assert np.max(steps) <= 0.05 + 1e-12


def test_densify_single_segment_short():
    path = np.array([[0.0], [0.01]])
    d = densify(path, max_joint_step=0.05)
    assert len(d) == 2                      # no needless insertion


def test_slip_monitor_zero_for_rigid_ride():
    T_ee_body = make_pose([0.0, 0.0, -0.12],
                          R.from_euler("y", 0.3).as_matrix())
    mon = SlipMonitor(T_ee_body)
    rng = np.random.default_rng(0)
    for _ in range(10):
        T_ee = make_pose(rng.normal(0, 0.3, 3),
                         R.random(random_state=rng).as_matrix())
        dpos, drot = mon.update(T_ee, T_ee @ T_ee_body)
        assert dpos < 1e-12 and drot < 1e-9
    assert mon.max_dpos < 1e-12


def test_slip_monitor_measures_injected_slip():
    T_ee_body = make_pose([0.0, 0.0, -0.12])
    mon = SlipMonitor(T_ee_body)
    T_ee = make_pose([0.1, 0.2, 0.9])
    slipped = T_ee @ T_ee_body
    slipped[:3, 3] += [0.0, 0.0, -0.02]                 # 2 cm sag
    slipped[:3, :3] = R.from_euler("x", 0.1).as_matrix() @ slipped[:3, :3]
    dpos, drot = mon.update(T_ee, slipped)
    assert dpos == pytest.approx(0.02, abs=1e-12)
    assert drot == pytest.approx(0.1, abs=1e-9)
    assert mon.max_dpos == pytest.approx(0.02, abs=1e-12)


def test_arm_part_dict_contract():
    part = joint_position_arm_part(output_max=0.07)
    assert part["type"] == "JOINT_POSITION"
    assert part["output_max"] == 0.07 and part["output_min"] == -0.07
    # the gripper controller rides inside the arm part; losing it in the
    # replacement silently drops the gripper action dimension
    assert part["gripper"] == {"type": "GRIP"}
    # no leftover OSC keys (the collision README pin #4 warns about)
    for osc_key in ("position_limits", "orientation_limits",
                    "uncouple_pos_ori", "input_ref_frame"):
        assert osc_key not in part