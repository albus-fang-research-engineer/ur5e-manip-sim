"""OmniManip's Eqs. 2-5 on synthetic geometry and its straight-move helpers. The
stage loop itself is exercised by scripts/plan_omnimanip.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manip_sim.omnimanip.planner import default_program, grasp_pose, line_poses, load_program  # noqa: E402
from manip_sim.omnimanip.target_pose import (Constraint, Obstacle, Weights, collision_loss,  # noqa: E402
                                             errors, pose_from_x, rho, solve_target_pose)
from manip_sim.rekep.adapter import hull_planes  # noqa: E402

# a held rod: tail at body -y, axis toward the tip along body +y
C = Constraint(point_active_body=np.array([0.0, -0.06, 0.0]), axis_active_body=np.array([0.0, 1.0, 0.0]),
               point_passive=np.array([0.0, 0.25, 0.9]), axis_passive=np.array([0.0, 0.0, 1.0]), d=-0.045)
CUBE = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T * 0.5


def posed(rot=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, t
    return T


def satisfying(spin=0.0):
    """An object pose meeting C: body +y to world +z, spun about it."""
    Rm = R.from_euler("z", spin).as_matrix() @ R.from_euler("x", 90, degrees=True).as_matrix()
    return posed(Rm, C.point_passive + C.d * C.axis_passive - Rm @ C.point_active_body)


def test_rho_is_zero_on_the_constraint_for_any_spin_about_the_axis():
    for spin in (0.0, 0.7, 2.5):
        assert rho(satisfying(spin), np.eye(4), C, 0.1) == pytest.approx(0.0, abs=1e-12)
        pos, ang = errors(satisfying(spin), np.eye(4), C)
        assert pos == pytest.approx(0.0, abs=1e-9) and ang == pytest.approx(0.0, abs=1e-7)


def test_d_is_signed_along_the_passive_axis():
    above = satisfying().copy()
    above[2, 3] += 0.045                                   # the point at the opening, not 4.5 cm below
    assert errors(above, np.eye(4), C)[0] == pytest.approx(0.045)


def test_upside_down_is_not_aligned():
    Rm = R.from_euler("x", -90, degrees=True).as_matrix()
    T = posed(Rm, C.point_passive + C.d * C.axis_passive - Rm @ C.point_active_body)
    assert errors(T, np.eye(4), C)[1] == pytest.approx(np.pi)
    assert rho(T, np.eye(4), C, 0.1) == pytest.approx(2 * 0.1 ** 2 * 2.0)


def test_collision_loss_is_a_squared_hinge():
    box = Obstacle("box", [hull_planes(CUBE)])
    point = np.zeros((1, 3))
    assert collision_loss(posed(t=(0.6, 0, 0)), point, [box], 0.2) == pytest.approx(0.1 ** 2)
    assert collision_loss(posed(t=(0.8, 0, 0)), point, [box], 0.2) == 0.0
    assert collision_loss(posed(t=(0.0, 0, 0)), point, [box], 0.2) == pytest.approx(0.7 ** 2)


def test_collision_uses_the_nearest_of_an_objects_hulls_once():
    near, far = hull_planes(CUBE), hull_planes(CUBE + [5.0, 0, 0])
    assert collision_loss(posed(t=(0.6, 0, 0)), np.zeros((1, 3)), [Obstacle("o", [near, far])], 0.2) \
        == pytest.approx(0.1 ** 2)


def test_solve_reaches_the_constraint_with_the_small_stated_bias():
    start = posed(t=(0.0, -0.25, 0.85))
    T, report = solve_target_pose(start, np.eye(4), C, np.zeros((1, 3)), [])
    assert report["position_error_mm"] < 1.0 and report["angle_error_deg"] < 1.0
    assert report["rotation_deg"] == pytest.approx(90.0, abs=1.0)      # L_path: no spin it does not need
    assert T[:3, :3] @ C.axis_active_body == pytest.approx(C.axis_passive, abs=0.02)


def test_large_lambdas_trade_alignment_away():
    start = posed(t=(0.0, -0.25, 0.85))
    _, tight = solve_target_pose(start, np.eye(4), C, np.zeros((1, 3)), [])
    _, soft = solve_target_pose(start, np.eye(4), C, np.zeros((1, 3)), [],
                                Weights(lambda_trans=1e-3, lambda_rot=1e-2))
    assert soft["angle_error_deg"] > 10 * tight["angle_error_deg"]


def test_collision_moves_the_target_when_a_point_would_be_inside():
    box = Obstacle("box", [hull_planes(CUBE * 0.1 + C.point_passive + C.d * C.axis_passive)])
    ee_point = np.array([[0.0, -0.06, 0.0]])                           # at the active point
    _, free = solve_target_pose(posed(t=(0.0, -0.25, 0.85)), np.eye(4), C, ee_point, [])
    _, blocked = solve_target_pose(posed(t=(0.0, -0.25, 0.85)), np.eye(4), C, ee_point, [box])
    assert blocked["position_error_mm"] > free["position_error_mm"] + 5.0


def test_pose_from_x_applies_rotation_on_the_world_side():
    T = pose_from_x(np.r_[0.1, 0, 0, 0, 0, np.pi / 2], posed(t=(1.0, 0, 0)))
    assert T[:3, 3] == pytest.approx([1.1, 0, 0])
    assert T[:3, :3] @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-9)


def test_line_poses_hit_both_ends_within_step():
    a = posed(t=(0, 0, 0))
    b = posed(R.from_euler("x", 30, degrees=True).as_matrix(), (0.1, 0, 0))
    poses = line_poses(a, b, 0.01, np.radians(2.0))
    assert np.allclose(poses[0], a) and np.allclose(poses[-1], b)
    assert len(poses) == 16                                            # 15 steps for 30 deg at 2 deg
    assert np.allclose(np.diff([p[0, 3] for p in poses]), 0.1 / 15)


def test_grasp_pose_is_a_right_handed_frame_nearest_the_reference():
    T_object = posed(t=(0, -0.25, 0.81))
    T = grasp_pose(T_object, [0, 0.022, 0.0], [1, 0, 0], [0, 0, -1], posed())
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    assert T[:3, 2] == pytest.approx([0, 0, -1]) and T[:3, 0] == pytest.approx([1, 0, 0])
    assert T[:3, 3] == pytest.approx([0, -0.228, 0.81])
    flipped = grasp_pose(T_object, [0, 0.022, 0.0], [1, 0, 0], [0, 0, -1],
                         posed(R.from_euler("z", np.pi).as_matrix()))
    assert flipped[:3, 0] == pytest.approx([-1, 0, 0], abs=1e-9)


def test_insert_marker_program_loads_and_only_place_is_accepted(tmp_path):
    p = load_program(default_program("insert_marker"))
    assert p["stages"][0]["d_m"] < 0 and p["grasp"]["standoff_m"] == 0.08
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"stages": [{"action": "pour"}]}))
    with pytest.raises(ValueError, match="only grasp then one place"):
        load_program(bad)
