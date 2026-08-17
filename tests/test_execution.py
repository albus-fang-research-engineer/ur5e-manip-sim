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

# ------------------------------------------------- pour success readout
# The regression these lock down: tip_to_opening_mm folds the transport
# subgoal's DELIBERATE 40-100 mm standoff into the same scalar as the
# lateral miss, so it cannot distinguish a perfect pour from a 40 mm
# miss. lateral / standoff / stream-landing are separated here.

def _dbg(rim_radius=0.044):
    from manip_sim.frames import Frame
    from manip_sim.viz import DebugOverlay
    spout = Frame("spout", np.array([0.1, 0.0, 0.0]),
                  np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    tilt = Frame("tilt", np.array([0.1, 0.0, 0.0]),
                 np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    opening = Frame("opening", np.zeros(3), np.array([0.0, 0.0, 1.0]),
                    np.array([1.0, 0.0, 0.0]))
    return DebugOverlay(spout, tilt, opening, rim_radius=rim_radius)


def test_report_separates_standoff_from_lateral_miss():
    dbg = _dbg()
    # teapot placed so the tip sits 70 mm above the opening, dead centre
    T_teapot = make_pose([-0.1, 0.0, 0.07])
    r = dbg.report(T_teapot, make_pose([0.0, 0.0, 0.0]))
    assert r["tip_standoff_mm"] == pytest.approx(70.0, abs=1e-6)
    assert r["tip_lateral_mm"] == pytest.approx(0.0, abs=1e-9)
    assert r["tip_over_rim"]
    # the legacy scalar reports 70 mm for this PERFECT placement
    assert r["tip_to_opening_mm"] == pytest.approx(70.0, abs=1e-6)


def test_report_flags_a_lateral_miss_the_legacy_scalar_hides():
    dbg = _dbg()
    good = dbg.report(make_pose([-0.1, 0.0, 0.075]), make_pose([0, 0, 0]))
    miss = dbg.report(make_pose([-0.1, 0.06, 0.045]), make_pose([0, 0, 0]))
    # nearly identical 3-D distances ...
    assert abs(good["tip_to_opening_mm"] - miss["tip_to_opening_mm"]) < 1.0
    # ... opposite verdicts
    assert good["tip_over_rim"] and not miss["tip_over_rim"]


def test_stream_landing_requires_the_spout_to_point_down():
    from scipy.spatial.transform import Rotation as R_
    dbg = _dbg()
    T_mug = make_pose([0.0, 0.0, 0.0])
    upright = make_pose([-0.1, 0.0, 0.07])              # spout axis = +x
    assert dbg.report(upright, T_mug)["stream_lands_in_mug"] is False
    # tilt +90 deg about +y takes the spout axis from +x to -z
    T = make_pose([0.0, 0.0, 0.0], R_.from_euler("y", np.pi / 2).as_matrix())
    T[:3, 3] = np.array([0.0, 0.0, 0.07]) - T[:3, :3] @ np.array([0.1, 0, 0])
    r = dbg.report(T, T_mug)
    assert r["spout_declination_deg"] == pytest.approx(0.0, abs=1e-6)
    assert r["stream_lands_in_mug"]
    assert r["stream_lateral_mm"] == pytest.approx(0.0, abs=1e-6)
