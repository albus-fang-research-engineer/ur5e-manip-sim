"""Grasping-library tests: numpy/scipy only, no simulator required
(mirrors test_tsr.py's scope). The IK/probe layers are exercised by the
end-to-end scripts, not here."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from manip_sim.frames import Frame
from manip_sim.grasping import (
    GraspProposal,
    classify_grasps,
    free_tsr,
    handle_grasp_tsr,
    nominal_grip_in_handle,
    propose_handle_grasps,
    wrist_flip,
)
from manip_sim.tsr import make_pose


@pytest.fixture
def handle():
    return Frame("handle", point=np.array([0.09, -0.043, 0.07]),
                 axis=np.array([0.0, 0.0, 1.0]),
                 secondary=np.array([1.0, 0.0, 0.0]))


def test_nominal_frame_is_right_handed_and_pitched():
    T = nominal_grip_in_handle([1.0, 0.0, 0.3], np.deg2rad(60.0))
    Rm = T[:3, :3]
    assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(Rm), 1.0)
    # approach (+z col) pitched 60 deg below horizontal, azimuth preserved
    z = Rm[:, 2]
    assert np.isclose(z[2], -np.sin(np.deg2rad(60.0)), atol=1e-9)
    assert z[0] > 0 and np.isclose(z[1], 0.0, atol=1e-9)
    # closing axis (+x, per the measured Robotiq85 convention) stays
    # horizontal so the pads straddle a vertical bar
    assert np.isclose(Rm[2, 0], 0.0, atol=1e-9)


def test_nominal_frame_rejects_vertical_approach():
    with pytest.raises(ValueError):
        nominal_grip_in_handle([0.0, 0.0, 1.0], 0.0)


def test_grasp_tsr_zero_and_slide(handle):
    T0_body = make_pose([0.1, -0.2, 0.85],
                        R.from_euler("z", 0.7).as_matrix())
    tsr = handle_grasp_tsr(T0_body, handle, approach_h=[-1.0, 0.5, 0.0])
    assert tsr.contains(tsr.zero(), tol=1e-9)
    # slide along the bar within bounds stays contained; beyond does not
    T0_w = T0_body @ handle.T()
    up = T0_w[:3, 2]
    T_in = tsr.zero().copy()
    T_in[:3, 3] += 0.015 * up
    assert tsr.contains(T_in, tol=1e-6)
    T_out = tsr.zero().copy()
    T_out[:3, 3] += 0.05 * up
    assert not tsr.contains(T_out, tol=1e-6)


def test_classifier_rejects_junk_keeps_nominal(handle):
    T0_body = make_pose([0.0, -0.25, 0.86])
    tsr = handle_grasp_tsr(T0_body, handle, approach_h=[-1.0, 0.5, 0.0])
    rng = np.random.default_rng(0)
    props = propose_handle_grasps(
        tsr, rng, n=60, junk_points=[np.array([-0.2, 0.0, 1.0])])
    kept, tally = classify_grasps(tsr, props)
    assert tally["junk_kept"] == 0
    assert 0 < len(kept) < len(props)          # a real filter, both ways
    for p in kept:
        assert p.provenance == "handle"
    # the classifier never consults provenance: a junk-tagged proposal at
    # the nominal pose is (correctly) kept
    trojan = GraspProposal(tsr.zero(), "junk")
    kept2, _ = classify_grasps(tsr, [trojan])
    assert len(kept2) == 1


def test_wrist_flip_is_same_grasp_involution():
    T = make_pose([0.1, 0.2, 0.3], R.random(random_state=1).as_matrix())
    F = wrist_flip(T)
    assert np.allclose(wrist_flip(F), T)               # involution
    assert np.allclose(F[:3, 3], T[:3, 3])             # same grip point
    assert np.allclose(F[:3, 2], T[:3, 2])             # same approach axis
    assert np.allclose(F[:3, 0], -T[:3, 0])            # fingers swapped
    assert np.isclose(np.linalg.det(F[:3, :3]), 1.0)   # still a rotation


def test_free_tsr_contains_everything_and_projects_identity():
    t = free_tsr()
    rng = np.random.default_rng(2)
    for _ in range(20):
        T = make_pose(rng.normal(0, 2.0, 3),
                      R.random(random_state=rng).as_matrix())
        assert t.contains(T)
        assert t.distance(T) == 0.0