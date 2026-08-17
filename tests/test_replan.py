"""Contracts of the slip-triggered re-anchor layer (manip_sim.replan):
the trigger derives its threshold from the subgoal's OWN B^w (no fresh
magic number), free rows express no preference, failures are typed."""

import numpy as np
import pytest

pytest.importorskip("mujoco")            # replan -> planning -> mujoco

from manip_sim.replan import (ReplanError, ReplanResult,      # noqa: E402
                              slip_exceeds, slip_tolerance)
from manip_sim.tsr import FREE_ROT, FREE_TRANS, TSR           # noqa: E402


def _pour_like_subgoal() -> TSR:
    # pivot +-5 mm, roll effectively free (the tilt DoF), pitch/yaw +-3 deg
    Bw = np.array([[-0.005, 0.005], [-0.005, 0.005], [-0.005, 0.005],
                   [np.deg2rad(-3.0), np.deg2rad(110.0)],
                   [np.deg2rad(-3.0), np.deg2rad(3.0)],
                   [np.deg2rad(-3.0), np.deg2rad(3.0)]])
    return TSR(T0_w=np.eye(4), Bw=Bw)


def test_slip_tolerance_tightest_finite_rows():
    tol_pos, tol_rot = slip_tolerance(_pour_like_subgoal())
    assert tol_pos == pytest.approx(0.005)
    assert tol_rot == pytest.approx(np.deg2rad(3.0))
    # the wide tilt row (113 deg span) must not set the rotation
    # threshold — the tight off-axis rows do
    assert tol_rot < np.deg2rad(4.0)


def test_slip_tolerance_free_rows_express_no_preference():
    Bw = np.array([list(FREE_TRANS)] * 3 + [list(FREE_ROT)] * 3)
    tol_pos, tol_rot = slip_tolerance(TSR(T0_w=np.eye(4), Bw=Bw))
    assert np.isinf(tol_pos) and np.isinf(tol_rot)
    fire, *_ = slip_exceeds((1.0, np.pi / 2), TSR(T0_w=np.eye(4), Bw=Bw))
    assert not fire                       # fully free subgoal never fires


def test_slip_exceeds_fires_on_either_block():
    sg = _pour_like_subgoal()
    assert not slip_exceeds((0.004, np.deg2rad(2.0)), sg)[0]
    assert slip_exceeds((0.006, 0.0), sg)[0]          # translation alone
    assert slip_exceeds((0.0, np.deg2rad(31.0)), sg)[0]   # the observed slip


def test_replan_result_stacks_stage_ids_in_order():
    rr = ReplanResult()
    rr.paths.append((2, np.zeros((4, 6))))
    rr.paths.append((3, np.ones((3, 6))))
    assert rr.path.shape == (7, 6)
    assert rr.stage_ids.tolist() == [2, 2, 2, 2, 3, 3, 3]


def test_replan_error_is_typed_and_routable():
    e = ReplanError(3, "no feasible pour configs")
    assert e.stage == 3 and "pour" in e.reason
    assert isinstance(e, RuntimeError)    # never SystemExit
