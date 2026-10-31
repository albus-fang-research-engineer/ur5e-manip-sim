"""ReKep adapter on synthetic geometry and the real scene: the frame, the IK's
upstream-shaped result, the SDF's convention, and the program contract. The
stage loop itself is exercised by scripts/plan_rekep.py."""

import json
import sys
from pathlib import Path

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
from manip_sim.rekep.adapter import (T_RK_SITE, T_SITE_RK, ReKepIK, body_pose,  # noqa: E402
                                     farthest_point_sampling, hull_distance, hull_planes,
                                     rekep_to_site, scene_sdf, sdf_shape, site_to_rekep)
from manip_sim.rekep.planner import default_program_dir  # noqa: E402
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
