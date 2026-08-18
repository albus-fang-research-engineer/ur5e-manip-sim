"""Contracts of the vertical retreat primitive (_retreat_up): it returns
the FIRST level whose upright projection clears collision, tolerates
grazing contact only for the first two steps (mirroring the offline
post-grasp lift), and returns None — never a partial path — when dz_max
is exhausted. Fakes only: the primitive's contract is geometric, not a
simulator behavior."""

import numpy as np
import pytest

pytest.importorskip("mujoco")            # replan -> planning -> mujoco

from manip_sim.replan import _retreat_up               # noqa: E402


class _FakeKin:
    """q[0] encodes eef height; collision-free at/above clear_z."""

    def __init__(self, clear_z: float):
        self.clear_z = clear_z

    def fk(self, q):
        T = np.eye(4)
        T[2, 3] = float(q[0])
        return T

    def in_collision(self, q):
        return float(q[0]) < self.clear_z


class _FakeIK:
    def solve(self, T0_ee_target, q_seed, **kw):
        return np.array([T0_ee_target[2, 3]] + [0.0] * 5), True


class _FakeProj:
    """Stands in for the path TSR list; project is monkeypatched below."""


def _patch_projection(monkeypatch, move=0.0):
    # projection succeeds and (optionally) shifts the config; clearance
    # is then judged on the projected config through kin.in_collision
    import manip_sim.replan as rp

    def fake_project(kin, attached, tsrs, q, tol=2e-3):
        qp = np.asarray(q, float).copy()
        qp[0] += move
        return qp, True

    monkeypatch.setattr(rp, "project_config", fake_project)


def test_retreat_stops_at_first_clear_level(monkeypatch):
    _patch_projection(monkeypatch)
    kin = _FakeKin(clear_z=0.049)        # clears on the third 2 cm step
    r = _retreat_up(_FakeIK(), kin, None, np.zeros(6), [None])
    assert r is not None
    path, q_clear = r
    # steps 0.02 / 0.04 are in grazing contact (tolerated, k <= 2);
    # 0.06 is the first clear level and the retreat stops THERE
    assert np.allclose(path[:, 0], [0.02, 0.04, 0.06])
    assert q_clear[0] == pytest.approx(0.06)


def test_retreat_none_when_exhausted(monkeypatch):
    _patch_projection(monkeypatch)
    kin = _FakeKin(clear_z=10.0)         # never clears within dz_max
    assert _retreat_up(_FakeIK(), kin, None, np.zeros(6), [None]) is None


def test_retreat_none_when_grazing_persists(monkeypatch):
    """Collision past step 2 aborts the retreat (no dragging the pot up
    THROUGH an obstruction) — the caller gets the typed error instead."""
    _patch_projection(monkeypatch, move=-10.0)   # projection never clears
    kin = _FakeKin(clear_z=0.09)                 # step 3 (0.06) collides
    # steps 1-2 are grazing-tolerated, projections stay in collision, and
    # the raw climb collides at k=3 -> abort, not a partial path
    assert _retreat_up(_FakeIK(), kin, None, np.zeros(6), [None]) is None
