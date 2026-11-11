"""manip_sim.antipodal on synthetic meshes: geometry the answers can be checked against."""
import numpy as np
import pytest
import trimesh

from manip_sim import antipodal as ap
from manip_sim.tsr import make_pose

TABLE = 0.8


def _on_table(mesh):
    """Pose that rests the mesh on the table, centered over the origin."""
    return make_pose([0.3, 0.0, TABLE - mesh.bounds[0, 2]], np.eye(3))


def _box(ext=(0.03, 0.10, 0.03)):
    return trimesh.creation.box(extents=ext)


def test_first_hits_matches_the_analytic_crossing():
    m = _box((0.02, 0.04, 0.06))
    o = np.array([[-0.05, 0.0, 0.0], [0.0, -0.05, 0.01], [0.0, 0.0, 0.2]])
    d = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    t, f = ap.first_hits(m.vertices, m.faces, o, d, max_t=0.1)
    assert np.allclose(t[:2], [0.04, 0.03], atol=1e-9) and np.all(f[:2] >= 0)
    assert np.isinf(t[2]) and f[2] == -1                    # pointing away: no hit


def test_box_grasps_are_antipodal_supported_and_clear_of_the_table():
    m = _box()
    T0 = _on_table(m)
    gs, funnel = ap.synthesize(m.vertices, m.faces, T0, TABLE, seed=0)
    assert gs and funnel["returned"] == len(gs)
    for g in gs:
        assert abs(g.width - 0.03) < 1e-6                    # across the 30 mm faces
        assert g.antipodality > 0.99 and g.pad_support > 0.5
        Tw = T0 @ g.T_body_site
        corners = ap._box_corners(ap.SIM_2F85) @ Tw[:3, :3].T + Tw[:3, 3]
        assert corners[:, 2].min() >= TABLE + ap.TABLE_CLEAR_M - 1e-9
        assert (Tw[:3, :3] @ [0, 0, 1])[2] <= ap.MAX_UP_SIN + 1e-9   # never from below
    mids = [0.5 * (g.contacts[0] + g.contacts[1]) for g in gs]
    for i in range(len(mids)):
        for j in range(i):
            assert np.linalg.norm(mids[i] - mids[j]) >= ap.SPREAD_M - 1e-9


def test_grasps_spread_along_a_long_object_and_the_best_is_near_its_center():
    m = _box((0.03, 0.12, 0.03))
    gs, _ = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=0)
    ys = [0.5 * (g.contacts[0][1] + g.contacts[1][1]) for g in gs]
    assert min(ys) < -0.03 and max(ys) > 0.03                 # both ends are offered
    assert abs(ys[0]) < 0.015 and gs[0].com > gs[-1].com      # best first: near the COM


def test_a_cylinder_supports_the_pads_less_than_a_flat_box():
    cyl = trimesh.creation.cylinder(radius=0.012, height=0.12, sections=48)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    gc, _ = ap.synthesize(cyl.vertices, cyl.faces, _on_table(cyl), TABLE, seed=0)
    gb, _ = ap.synthesize(_box().vertices, _box().faces, _on_table(_box()), TABLE, seed=0)
    assert gc and gb
    assert max(g.pad_support for g in gc) < min(g.pad_support for g in gb)
    assert all(abs(g.width - 0.024) < 2e-3 for g in gc)


def test_an_object_wider_than_the_gripper_gives_no_grasp():
    m = _box((0.10, 0.10, 0.10))
    gs, funnel = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=0)
    assert gs == [] and funnel["opposite_hit"] == 0


@pytest.mark.parametrize("seed", [0, 1])
def test_synthesis_is_deterministic_for_a_seed(seed):
    m = _box()
    a, _ = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=seed)
    b, _ = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=seed)
    assert [g.score for g in a] == [g.score for g in b]
    assert all(np.array_equal(x.T_body_site, y.T_body_site) for x, y in zip(a, b))
