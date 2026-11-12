"""plan_constrained is repeatable: the same seed and the same iteration
budget give a bitwise-identical path, whatever the CPU load. Same scene
fixture as test_planning.py (object-free TableTop, kinematic attached teapot)."""

import numpy as np
import pytest

robosuite = pytest.importorskip("robosuite")
mujoco = pytest.importorskip("mujoco")

from robosuite.environments.base import register_env  # noqa: E402

from manip_sim.envs.tabletop import TableTop  # noqa: E402
from manip_sim.grasping import free_tsr  # noqa: E402
from manip_sim.planning import (  # noqa: E402
    ArmKinematics,
    AttachedObject,
    plan_constrained,
)
from manip_sim.tsr import make_pose  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    try:
        register_env(TableTop)
    except AssertionError:      # already registered
        pass
    env = robosuite.make(
        "TableTop", robots="UR5e", object_xmls={},
        table_full_size=(1.2, 1.2, 0.05), table_offset=(0, 0, 0.8),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20, ignore_done=True,
    )
    env.reset()
    kin = ArmKinematics(env)
    q_home = env.sim.data.qpos[kin.qpos_ids].copy()
    T0_ee = kin.fk(q_home)
    att = AttachedObject(np.linalg.inv(T0_ee) @ make_pose(T0_ee[:3, 3] - [0.0, 0.0, 0.16]))
    q_goal = np.clip(q_home + np.array([1.0, 0.2, 0.2, 0.0, 0.0, 0.0]),
                     *kin.joint_range.T)
    yield kin, att, q_home, q_goal
    env.close()


def _plan(scene, seed, **kw):
    kin, att, q_home, q_goal = scene
    return plan_constrained(kin, att, [free_tsr()], q_home, q_goal,
                            timeout=120.0, max_iters=500,
                            rng=np.random.default_rng(seed), **kw)


def test_same_seed_gives_identical_paths(scene):
    a, b = _plan(scene, 3), _plan(scene, 3)
    assert a.ok and b.ok, (a.reason, b.reason)
    assert np.array_equal(a.path, b.path)
    assert a.stats["n_iters"] == b.stats["n_iters"]


def test_different_seeds_may_differ(scene):
    paths = [_plan(scene, s) for s in (0, 1, 2)]
    assert all(p.ok for p in paths)
    # only the endpoints are pinned; the interior comes from the random tree
    assert any(not np.array_equal(paths[0].path, p.path) for p in paths[1:])


def test_iteration_budget_is_reported_in_the_reason(scene):
    kin, att, q_home, q_goal = scene
    r = plan_constrained(kin, att, [free_tsr()], q_home, q_goal, timeout=120.0,
                         max_iters=0, rng=np.random.default_rng(0))
    assert not r.ok
    assert "iteration budget exhausted (0)" in r.reason
    r = plan_constrained(kin, att, [free_tsr()], q_home, q_goal, timeout=0.0,
                         rng=np.random.default_rng(0))
    assert not r.ok and "timeout" in r.reason


class _SlowClock:
    """time.time() that advances one second per call: a search that ends well
    inside its timeout, on a machine so loaded that everything after it runs
    past the wall clock."""

    def __init__(self):
        self.t = 1e9

    def __call__(self):
        self.t += 1.0
        return self.t


def test_shortcut_smoothing_does_not_stop_on_the_wall_clock(scene, monkeypatch):
    """Only the search itself may stop on the clock (its safety cap). Shortcut
    smoothing runs its fixed count of shortcuts, so a search that connected
    gives the same path however long the smoothing took."""
    import time
    ref = _plan(scene, 3)
    assert ref.ok, ref.reason
    kin, att, q_home, q_goal = scene
    # timeout just above the clock calls the search needs: the search still
    # connects as before, and every shortcut runs past the timeout
    monkeypatch.setattr(time, "time", _SlowClock())
    slow = plan_constrained(kin, att, [free_tsr()], q_home, q_goal,
                            timeout=ref.stats["n_iters"] + 3.0, max_iters=500,
                            rng=np.random.default_rng(3))
    monkeypatch.undo()
    assert slow.ok, slow.reason
    assert slow.stats["n_iters"] == ref.stats["n_iters"]
    assert np.array_equal(slow.path, ref.path)
