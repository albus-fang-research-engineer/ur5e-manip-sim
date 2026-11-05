"""ReKep adapter on synthetic geometry and the real scene: the frame, the IK's
upstream-shaped result, the SDF's convention, and the program contract. The
stage loop itself is exercised by scripts/plan_rekep.py."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import manip_sim.rekep.vendor  # noqa: F401
except ImportError as e:
    pytest.skip(str(e), allow_module_level=True)
from manip_sim.planning import ArmKinematics  # noqa: E402
from manip_sim.rekep.adapter import (T_RK_SITE, T_SITE_RK, IKResult, MinkTracker, ReKepIK, body_pose,  # noqa: E402
                                     farthest_point_sampling, hull_distance, hull_planes,
                                     rekep_to_site, scene_sdf, sdf_shape, site_to_rekep)
from manip_sim.rekep.planner import Config, ReKepPlanner, _ChainFailure, default_program_dir  # noqa: E402
from manip_sim.rekep.program import load_program  # noqa: E402
from manip_sim.scene import load_scene, make_env  # noqa: E402

CUBE = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T * 0.5


@pytest.fixture(scope="module")
def sim():
    scene = load_scene(ROOT / "scenes/insert_marker.json")
    env, _ = make_env(scene, has_renderer=False)
    yield scene, env, ArmKinematics(env)
    env.close()


def test_frame_is_a_proper_rotation_mapping_rekep_x_to_approach():
    assert np.isclose(np.linalg.det(T_SITE_RK[:3, :3]), 1.0)
    assert np.allclose(T_SITE_RK[:3, :3] @ [1, 0, 0], [0, 0, 1])      # rekep x = site z
    assert np.allclose(T_SITE_RK @ T_RK_SITE, np.eye(4))


def test_pose_round_trip():
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix()
    T[:3, 3] = [0.1, -0.2, 0.9]
    assert np.allclose(rekep_to_site(site_to_rekep(T)), T)


def test_hull_distance_on_a_unit_cube():
    A, b = hull_planes(CUBE)
    d = hull_distance(np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]], float), A, b)
    assert np.allclose(d, [-0.5, 0.0, 0.5])


def test_farthest_point_sampling_keeps_the_spread():
    pts = np.vstack([np.zeros((50, 3)), [[1.0, 0, 0]], [[0, 1.0, 0]]])
    out = farthest_point_sampling(pts, 3)
    assert len(out) == 3 and {tuple(p) for p in out} >= {(1.0, 0, 0), (0, 1.0, 0)}


def test_ik_reaches_a_pose_and_counts_descents(sim):
    _, _, kin = sim
    q0 = kin.data.qpos[kin.qpos_ids].copy()
    T = kin.fk(q0)
    ik = ReKepIK(kin, q0)
    assert ik.solve_site(T, initial_joint_pos=q0).num_descents == 0
    T_target = T.copy()
    T_target[:3, 3] += [0.1, 0.05, -0.05]
    r = ik.solve_site(T_target, 1e-3, 5e-3, 100, q0)
    assert r.success and r.num_descents > 0
    assert np.linalg.norm(kin.fk(r.cspace_position)[:3, 3] - T_target[:3, 3]) <= 1e-3
    kin.set_q(q0)


def test_ik_reports_failure_on_an_unreachable_pose(sim):
    _, _, kin = sim
    q0 = kin.data.qpos[kin.qpos_ids].copy()
    T = kin.fk(q0)
    T[:3, 3] += [5.0, 0, 0]
    r = ReKepIK(kin, q0).solve_site(T, max_iterations=20, initial_joint_pos=q0)
    assert not r.success and r.num_descents == 20
    kin.set_q(q0)


def test_soft_ik_keeps_a_pose_the_chain_misses(sim):
    _, _, kin = sim
    q0 = kin.data.qpos[kin.qpos_ids].copy()
    T = kin.fk(q0)
    T_far = T.copy()
    T_far[:3, 3] += [5.0, 0, 0]
    planner = ReKepPlanner.__new__(ReKepPlanner)
    planner.track_ik, planner.tracking, planner.targets = ReKepIK(kin, q0), [], []
    planner.cfg = Config()
    with pytest.raises(_ChainFailure):
        planner._ik_step(T, T_far, q0, 0)
    planner.cfg = Config(soft_ik=True)
    assert len(planner._ik_step(T, T_far, q0, 0)) == 1
    assert planner.tracking[-1][0] > 1.0
    kin.set_q(q0)


def test_backtrack_goes_to_the_nearest_stage_that_still_holds():
    planner = ReKepPlanner.__new__(ReKepPlanner)
    planner.cfg = Config()
    high, low = (lambda ee, kp: 0.5), (lambda ee, kp: 0.0)
    def stages(*paths):
        return SimpleNamespace(stages=[SimpleNamespace(path=list(p)) for p in paths])
    points = np.zeros((3, 3))
    planner.program = stages([], [low], [low])
    assert planner._backtrack_to(2, points) is None
    planner.program = stages([], [low], [high])
    assert planner._backtrack_to(2, points) == 1
    planner.program = stages([], [high], [high])
    assert planner._backtrack_to(2, points) == 0
    planner.program = stages([], [high])
    assert planner._backtrack_to(0, points) is None


def test_mink_tracker_reaches_a_pose_and_reports_a_miss(sim):
    _, _, kin = sim
    q0 = kin.data.qpos[kin.qpos_ids].copy()
    T = kin.fk(q0)
    T[:3, 3] += [0.05, 0.02, -0.02]
    tracker = MinkTracker(kin)
    r = tracker.solve_site(T, 2e-3, 8e-3, 200, q0)
    assert r.success and r.position_error <= 2e-3
    T[:3, 3] += [5.0, 0, 0]
    assert not tracker.solve_site(T, 2e-3, 8e-3, 50, q0).success
    kin.set_q(q0)


def test_mink_tracker_stops_short_of_a_pose_out_of_reach(sim):
    _, _, kin = sim
    q0 = kin.data.qpos[kin.qpos_ids].copy()
    T0 = kin.fk(q0)
    tracker = MinkTracker(kin)
    for dx in ([0.6, 0, 0], [0, 0.8, 0.3], [5.0, 0, 0]):
        T = T0.copy()
        T[:3, 3] += dx
        r = tracker.solve_site(T, 1e-3, 5e-3, 200, q0)
        start = np.linalg.norm(T0[:3, 3] - T[:3, 3])
        assert not r.success and r.position_error < start
        r = tracker.solve_near(T, 1e-3, 5e-3, 200, q0, 0.35)
        assert np.max(np.abs(r.cspace_position - q0)) <= 0.35 and r.position_error < start
    kin.set_q(q0)


def test_soft_ik_does_not_switch_elbow_branches_in_one_step():
    q_a = np.array([0.0, -1.0, 0.0, -4.2, -0.2, 2.0])
    flipped = np.array([0.0, -6.9, -1.0, -0.8, 0.1, 5.4])         # a hit on the other branch
    wrist = np.array([0.0, -1.0, 0.1, -2.8, -0.2, 0.7])           # a big step on the same branch
    class Tracker:
        def __init__(self, q):
            self.q, self.near = q, []
        def solve_site(self, T, pos_tol, rot_tol, iters, q0):
            return IKResult(True, 0.0, iters, self.q, 0.0)
        def solve_near(self, T, pos_tol, rot_tol, iters, q0, max_step):
            self.near.append(max_step)
            return IKResult(False, 0.05, iters, q0 + max_step, 0.0)
    planner = ReKepPlanner.__new__(ReKepPlanner)
    planner.cfg, planner.tracking, planner.targets = Config(soft_ik=True), [], []
    planner.track_ik = Tracker(flipped)
    assert np.allclose(planner._ik_step(np.eye(4), np.eye(4), q_a, 0)[-1], q_a + 0.35)
    assert planner.track_ik.near == [0.35] and planner.tracking[-1][0] == 0.05
    q_a[2] = 0.1
    planner.track_ik = Tracker(wrist)
    assert np.allclose(planner._ik_step(np.eye(4), np.eye(4), q_a, 0)[-1], wrist)
    assert planner.track_ik.near == []


def test_ik_solve_takes_the_solvers_frame(sim):
    _, _, kin = sim
    q0 = kin.data.qpos[kin.qpos_ids].copy()
    r = ReKepIK(kin, q0).solve(kin.fk(q0) @ T_SITE_RK, initial_joint_pos=q0)
    assert r.success and r.num_descents == 0
    kin.set_q(q0)


def test_sdf_sees_table_and_cup_and_can_exclude_the_marker(sim):
    scene, _, kin = sim
    lo = np.array([-0.2, -0.4, scene.table_top_z - 0.02])
    hi = np.array([0.2, 0.4, scene.table_top_z + 0.2])
    sdf = scene_sdf(kin, lo, hi, 0.01)
    assert sdf.shape == sdf_shape(lo, hi, 0.01)
    axes = [np.linspace(a, b, n) for a, b, n in zip(lo, hi, sdf.shape)]

    def at(p):
        return sdf[tuple(int(np.argmin(np.abs(a - x))) for a, x in zip(axes, p))]

    assert at([0.15, 0.0, scene.table_top_z - 0.01]) > 0            # in the table
    assert at([0.15, 0.0, scene.table_top_z + 0.1]) == pytest.approx(-0.1, abs=0.011)

    marker = body_pose(kin, "marker")[:3, 3]
    assert at(marker) > 0
    without = scene_sdf(kin, lo, hi, 0.01, exclude=("marker",))
    ix = tuple(int(np.argmin(np.abs(a - x))) for a, x in zip(axes, marker))
    assert without[ix] < 0

    cup = body_pose(kin, "cup")[:3, 3]
    assert at(cup + [0.15, 0, 0]) < at(cup + [0.05, 0, 0])          # nearer the cup, higher


def test_insert_marker_program_loads_and_grasping_cost_is_wired(tmp_path):
    held = {"held": False}
    p = load_program(default_program_dir("insert_marker"),
                     lambda i: 0.0 if held["held"] else 1.0)
    assert p.grasped_object == "marker" and len(p.stages) == 3
    assert p.stages[0].is_grasp and p.stages[-1].release_keypoint == 0
    assert p.stages[1].path[0](np.zeros(3), np.zeros((5, 3))) == 1.0
    held["held"] = True
    assert p.stages[1].path[0](np.zeros(3), np.zeros((5, 3))) == 0.0


def test_program_must_grasp_first(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "keypoints": [{"object": "marker", "point": "tip"}], "num_stages": 2,
        "grasp_keypoints": [-1, 0], "release_keypoints": [-1, -1]}))
    with pytest.raises(ValueError, match="first stage"):
        load_program(tmp_path, lambda i: 0.0)
