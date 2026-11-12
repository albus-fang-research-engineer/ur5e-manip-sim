"""Planning tests. Require robosuite + mujoco (skipped when absent), no
renderer, no converted meshes -- the attached teapot is kinematic and the
scene is the object-free TableTop, mirroring scripts/plan_transport.py."""

import numpy as np
import pytest

robosuite = pytest.importorskip("robosuite")
mujoco = pytest.importorskip("mujoco")

from robosuite.environments.base import register_env  # noqa: E402

from manip_sim.envs.tabletop import TableTop  # noqa: E402
from manip_sim.frames import load_symbols  # noqa: E402
from manip_sim.planning import (  # noqa: E402
    ArmKinematics,
    AttachedObject,
    TRACK_STEP_RAD,
    MinkIK,
    chord_clear,
    densify,
    plan_constrained,
    project_config,
)
from manip_sim.grasping import free_tsr  # noqa: E402
from manip_sim.pour_stages import transport_pair  # noqa: E402
from manip_sim.tsr import make_pose, sample_intersection  # noqa: E402


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
    T0_body = make_pose(T0_ee[:3, 3] - [0.0, 0.0, 0.16])
    att = AttachedObject(np.linalg.inv(T0_ee) @ T0_body)
    tp = load_symbols("assets/objects/teapot")
    mg = load_symbols("assets/objects/mug")
    pair = transport_pair(
        make_pose([0.0, 0.25, 0.86]),
        mg.frame("opening_center", "up_axis"),
        tp.frame("spout_tip", "pour_axis"),
        T0_body[:3, 3],
        z_corridor=(-0.02, 0.45),
    )
    yield env, kin, q_home, att, pair, tp
    env.close()


def test_projection_pulls_random_configs_onto_manifold(scene):
    _, kin, _, att, pair, _ = scene
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(15):
        q, converged = project_config(kin, att, [pair.path],
                                      rng.uniform(-np.pi, np.pi, 6))
        if converged:
            ok += 1
            assert pair.path.distance(att.body_pose(kin.fk(q))) <= 2e-3 + 1e-6
    assert ok >= 12    # near-total convergence expected


def test_a_goal_region_ends_the_plan_without_reaching_the_goal_configuration(scene):
    _, kin, q_home, att, pair, _ = scene
    rng = np.random.default_rng(0)
    start = att.body_pose(kin.fk(q_home))[:3, 3]
    moved = lambda q: np.linalg.norm(att.body_pose(kin.fk(q))[:3, 3] - start) > 0.03
    # at this step one extension covers at most 1 rad, so the trees cannot join on the first pass and the region ends it
    q_goal = np.clip(q_home + np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), *kin.joint_range.T)
    q_goal = project_config(kin, att, [pair.path], q_goal)[0]
    res = plan_constrained(kin, att, [pair.path], q_home, q_goal, timeout=30, eps=0.005, rng=rng, goal_test=moved)
    assert res.ok, res.reason
    assert res.stats["ended_in_goal_region"]
    assert moved(res.path[-1])
    assert np.linalg.norm(res.path[0] - q_home) < 0.05


def test_full_stage2_plan(scene):
    _, kin, q_home, att, pair, tp = scene
    rng = np.random.default_rng(0)
    rep = sample_intersection(pair.subgoal, [pair.path], n=8, rng=rng)
    ik = MinkIK(kin)
    q_goal = None
    for Tb in rep.accepted:
        q, ok = ik.solve_multiseed(Tb @ np.linalg.inv(att.T_ee_body), [q_home])
        if ok and not kin.in_collision(q):
            q_goal = q
            break
    assert q_goal is not None, "no IK-feasible goal in 8 samples"

    res = plan_constrained(kin, att, [pair.path], q_home, q_goal,
                           timeout=30, rng=rng)
    assert res.ok, res.reason
    # manifold held along the densified path
    assert res.max_excess <= 5e-3
    # endpoint puts the spout tip in the rim band above the opening —
    # asserted RELATIVE TO THE SIDECAR (hardcoding the opening height
    # breaks the moment frames.json is calibrated)
    spout = tp.frame("spout_tip", "pour_axis")
    mg = load_symbols("assets/objects/mug")
    opening_w = (make_pose([0.0, 0.25, 0.86])
                 @ mg.frame("opening_center", "up_axis").T())[:3, 3]
    tip = (att.body_pose(kin.fk(res.path[-1])) @ spout.T())[:3, 3]
    assert np.linalg.norm(tip[:2] - opening_w[:2]) < 0.035
    assert opening_w[2] + 0.02 < tip[2] < opening_w[2] + 0.09
    # start and end match the requested configs
    assert np.linalg.norm(res.path[0] - q_home) < 0.05
    assert np.linalg.norm(res.path[-1] - q_goal) < 0.05

class _ThinPost:
    """The arm's kinematics with a stand-in obstacle: a post 0.03 rad thick in
    joint 0, thinner than the planner's 0.12 rad node spacing, so a search that
    checks only its nodes can step over it."""

    def __init__(self, kin, q_ref):
        self.kin, self.q_ref = kin, np.asarray(q_ref, float)

    def __getattr__(self, name):
        return getattr(self.kin, name)

    def in_collision(self, q, **_):
        d = np.asarray(q) - self.q_ref
        return bool(0.04 < d[0] < 0.07 and abs(d[1]) < 0.3)


def test_chord_clear_sees_an_obstacle_between_clear_nodes(scene):
    _, kin, q_home, _, _, _ = scene
    post = _ThinPost(kin, q_home)
    qb = q_home + np.array([0.11, 0, 0, 0, 0, 0])
    assert not post.in_collision(q_home) and not post.in_collision(qb)
    assert not chord_clear(post, q_home, qb)
    assert chord_clear(post, q_home, q_home + np.array([0.03, 0, 0, 0, 0, 0]))


def test_planned_path_is_clear_at_the_trackers_step(scene):
    """Every config the tracker commands (the path densified at
    TRACK_STEP_RAD, as eval_plan.py grades it) is collision-free, not only the
    planner's nodes."""
    _, kin, q_home, att, _, _ = scene
    post = _ThinPost(kin, q_home)
    q_goal = np.clip(q_home + np.array([0.6, 0, 0, 0, 0, 0]), *kin.joint_range.T)
    res = plan_constrained(post, att, [free_tsr()], q_home, q_goal, timeout=60.0,
                           max_iters=500, rng=np.random.default_rng(0))
    assert res.ok, res.reason
    hits = [q for q in densify(res.path, TRACK_STEP_RAD) if post.in_collision(q)]
    assert not hits, f"{len(hits)} tracked configs inside the post"
