"""manip_sim.antipodal on synthetic meshes: geometry the answers can be checked against."""
import json

import numpy as np
import pytest
import trimesh

from manip_sim import antipodal as ap
from manip_sim.tsr import make_pose

TABLE = 0.8
PARALLEL = ap.Gripper(parallel_pads=True)       # the closed pads forced to zero tilt


def _on_table(mesh):
    """Pose that rests the mesh on the table, centered over the origin."""
    return make_pose([0.3, 0.0, TABLE - mesh.bounds[0, 2]], np.eye(3))


def _box(ext=(0.03, 0.10, 0.03)):
    return trimesh.creation.box(extents=ext)


def _pad_faces_in_body(g, gripper=ap.SIM_2F85):
    """Each landed pad's inner-face corners and tilt, object body frame."""
    S = ap.closed_state_of(g, gripper)
    T = g.T_body_site
    return {s: (S.pad(s).corners() @ T[:3, :3].T + T[:3, 3], S.pad(s)) for s in ap.SIDES}


def test_first_hits_matches_the_analytic_crossing():
    m = _box((0.02, 0.04, 0.06))
    o = np.array([[-0.05, 0.0, 0.0], [0.0, -0.05, 0.01], [0.0, 0.0, 0.2]])
    d = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    t, f = ap.first_hits(m.vertices, m.faces, o, d, max_t=0.1)
    assert np.allclose(t[:2], [0.04, 0.03], atol=1e-9) and np.all(f[:2] >= 0)
    assert np.isinf(t[2]) and f[2] == -1                    # pointing away: no hit


def test_the_measured_default_state_reproduces_the_planner_boxes():
    """The table's default state is the planner's partly open state: its boxes
    match SIM_2F85_BOXES (measured independently, as AABBs). The free-air
    stroke at the same gap is the physically closed-to state; it differs by up
    to ~2 mm at the inner knuckles, since the planner's state holds them at
    their zero (off the physical stroke)."""
    ref = dict(zip(ap.SIM_2F85_BOX_NAMES, ap.SIM_2F85_BOXES))
    t = ap.load_table(str(ap.CLOSING_TABLE))
    assert abs(t.default.gap - 0.064) < 0.001
    for state, tol_pad, tol_all in ((t.default, 0.001, 0.001),
                                    (ap.SIM_2F85.free_air(t.default.gap), 0.0012, 0.0025)):
        boxes = dict(zip(state.names, state.aabbs()))
        for k, (lo, hi) in ref.items():
            err = max(np.abs(boxes[k][0] - lo).max(), np.abs(boxes[k][1] - hi).max())
            assert err < (tol_pad if ("fingerpad" in k or "fingertip" in k) else tol_all), (k, err)


def test_the_approach_state_is_a_setting():
    t = ap.load_table(str(ap.CLOSING_TABLE))
    assert ap.SIM_2F85.approach() is t.open                  # fully open by default
    assert ap.Gripper(approach_state="default").approach() is t.default
    assert abs(t.open.gap - max(g for g, _ in t.free_air)) < 1e-5 and t.open.gap > 0.095
    assert abs(ap.Gripper(approach_state="default").max_width - 0.056) < 0.001
    assert ap.SIM_2F85.max_width > 0.06
    with pytest.raises(ValueError):
        ap.Gripper(approach_state="half").approach()


def test_the_closed_pads_sit_where_the_measured_closing_puts_them():
    """Closed on the marker's 19 mm barrel the pads are toed in, touch the
    cylinder they were measured on, and their fingertip ends pull back toward
    the palm several mm from where the open pads end."""
    rows = json.loads(ap.CLOSING_TABLE.read_text())["on_object"]
    row = next(r for r in rows if abs(r["width_m"] - 0.019) < 1e-6)
    S = ap.SIM_2F85.closed(0.019)
    assert S.tilt_deg() > 10.0
    axis = np.array([0.0, 0.0, row["object_z_m"]])
    for s in ap.SIDES:
        pf = S.pad(s)
        dist = pf.normal @ (axis - pf.center)                 # face plane to the cylinder axis
        assert 0.0095 - 0.0015 < dist < 0.0095 + 0.0005
    assert ap._contact_z(S)[1] < ap._contact_z(ap.SIM_2F85.approach())[1] - 0.004
    assert abs(ap.SIM_2F85.approach().tilt_deg()) > 10.0     # wide open, the pads splay


def test_box_grasps_are_antipodal_and_clear_of_the_table():
    m = _box()
    T0 = _on_table(m)
    gs, funnel = ap.synthesize(m.vertices, m.faces, T0, TABLE, seed=0)
    assert gs and funnel["returned"] == len(gs)
    for g in gs:
        assert abs(g.width - 0.03) < 1e-6                    # across the 30 mm faces
        assert g.antipodality > 0.99 and g.pad_support > 0.0
        Tw = T0 @ g.T_body_site
        for state in (ap.SIM_2F85.approach(), ap.SIM_2F85.closed(g.width)):   # approaching, closed
            corners = state.corners() @ Tw[:3, :3].T + Tw[:3, 3]
            assert corners[:, 2].min() >= TABLE + ap.TABLE_CLEAR_M - 1e-9
        z_lo, z_hi = ap._contact_z(ap.SIM_2F85.closed(g.width))   # the contact is on the closed pad
        assert z_lo < g.depth < z_hi
        assert (Tw[:3, :3] @ [0, 0, 1])[2] <= ap.MAX_UP_SIN + 1e-9   # never from below
    kept = [(0.5 * (g.contacts[0] + g.contacts[1]), g.detail["approach_bin"]) for g in gs]
    for i in range(len(kept)):
        for j in range(i):
            if kept[i][1] == kept[j][1]:                       # spread within an approach bin
                assert np.linalg.norm(kept[i][0] - kept[j][0]) >= ap.SPREAD_M - 1e-9


@pytest.mark.parametrize("gripper", [ap.SIM_2F85, PARALLEL], ids=["measured-tilt", "parallel"])
def test_a_pad_fully_on_a_flat_face_gets_the_support_its_tilt_allows(gripper):
    """A flat face under a pad tilted by t (relative to the face) is within
    PAD_SUPPORT_M of the pad over the rows up to PAD_SUPPORT_M / sin t from
    its touching edge: with the sim's toed-in pads that is a part of the pad;
    with parallel pads, all of it. Checked on grasps whose pads lie fully on
    the 30 mm faces (support may drop where a pad overhangs an edge)."""
    m = _box((0.03, 0.10, 0.06))
    gs, _ = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, gripper=gripper, seed=0)
    on_face = []
    for g in gs:
        if abs(abs(g.contacts[0][0]) - 0.015) > 1e-6:          # across the 30 mm faces only
            continue
        faces = _pad_faces_in_body(g, gripper)
        if all(np.all(np.abs(c[:, 1]) < 0.05 - 5e-4) and np.all(np.abs(c[:, 2]) < 0.03 - 5e-4)
               for c, _ in faces.values()):
            on_face.append((g, faces))
    assert on_face
    for g, faces in on_face:
        expect = []
        for c, pf in faces.values():
            step = 2 * pf.half_len / (ap.PAD_GRID[1] - 1)
            rise = step * np.sin(np.radians(abs(pf.tilt_deg)))   # per grid row along the pad
            expect.append([sum(1 for k in range(ap.PAD_GRID[1]) if k * rise <= ap.PAD_SUPPORT_M + e)
                           for e in (-2e-4, 2e-4)])
        lo = min(e[0] for e in expect) / ap.PAD_GRID[1]
        hi = min(e[1] for e in expect) / ap.PAD_GRID[1]
        assert lo - 1e-9 <= g.pad_support <= hi + 1e-9, (g.pad_support, lo, hi)
        if gripper.parallel_pads:
            assert g.pad_support > 0.99                        # flat face, flat pad: all of it
        else:
            assert g.pad_support < 0.7                         # tilted pad: its touching edge rows


def test_a_cylinder_supports_parallel_pads_less_than_a_flat_box():
    """Like for like: both standing, side grasps, pads parallel. A round barrel
    meets a flat pad along a line, a flat face over its area. (With the sim's
    toed-in pads the order can flip: a tilted pad meets a flat face only at its
    edge rows.)"""
    cyl = trimesh.creation.cylinder(radius=0.012, height=0.12, sections=48)
    box = _box((0.024, 0.024, 0.12))
    gc, _ = ap.synthesize(cyl.vertices, cyl.faces, _on_table(cyl), TABLE, gripper=PARALLEL, seed=0)
    gb, _ = ap.synthesize(box.vertices, box.faces, _on_table(box), TABLE, gripper=PARALLEL, seed=0)
    sc = [g.pad_support for g in gc if g.detail["approach_bin"] == "side"]
    sb = [g.pad_support for g in gb if g.detail["approach_bin"] == "side"]
    assert sc and sb
    assert np.mean(sc) < np.mean(sb)
    assert all(abs(g.width - 0.024) < 2e-3 for g in gc)


def test_grasps_spread_along_a_long_object_and_the_best_is_near_its_center():
    m = _box((0.03, 0.12, 0.03))
    gs, _ = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=0)
    ys = [0.5 * (g.contacts[0][1] + g.contacts[1][1]) for g in gs]
    assert min(ys) < -0.03 and max(ys) > 0.03                 # both ends are offered
    assert abs(ys[0]) < 0.015 and gs[0].com > gs[-1].com      # best first: near the COM


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


def test_a_pad_over_a_step_is_dropped():
    """A bar with a collar 5 mm proud: grasps on the bar whose pad area overlaps
    the collar would touch the collar first, so none of the kept ones do."""
    bar = trimesh.creation.box(extents=(0.02, 0.12, 0.02))
    collar = trimesh.creation.box(extents=(0.03, 0.01, 0.02))
    collar.apply_translation([0, 0.02, 0])
    m = trimesh.util.concatenate([bar, collar])
    gs, funnel = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=0)
    assert funnel["pad_lands_off_contact"] > 0
    on_bar = [g for g in gs if abs(g.width - 0.02) < 1e-4]
    assert on_bar
    clo, chi = collar.bounds
    for g in on_bar:                 # no kept bar grasp's landed pad lies over the collar
        for c, _ in _pad_faces_in_body(g).values():
            assert c[:, 1].max() <= clo[1] + 1e-4 or c[:, 1].min() >= chi[1] - 1e-4


def test_a_taper_within_the_friction_cone_is_not_a_step():
    """A standing prism whose closing faces lean 2 deg each (4 deg apart, well
    inside the friction cone): the faces are smooth, so no pad is dropped as
    landing on a step, and grasps across the tapered faces are offered."""
    h, w0 = 0.06, 0.024
    w1 = w0 - 2 * h * np.tan(np.radians(2.0))
    pts = [[sx * w / 2, sy * 0.05, z] for z, w in ((0.0, w0), (h, w1)) for sx in (-1, 1) for sy in (-1, 1)]
    m = trimesh.convex.convex_hull(np.array(pts))
    gs, funnel = ap.synthesize(m.vertices, m.faces, _on_table(m), TABLE, seed=0)
    assert funnel["pad_lands_off_contact"] == 0
    across = [g for g in gs if abs(g.T_body_site[0, 0]) > 0.95]   # closing across the taper
    assert across and all(g.pad_support > 0 for g in across)


def test_the_closed_fingers_do_not_pass_through_the_object():
    """The marker's 19 mm barrel, standing: every returned grasp's closed
    fingers (toed in, each moved to where its pad lands) stay out of the
    object, and the check does catch fingers pushed into it. (Lying on the
    table, the barrel leaves almost no depth window: see the report.)"""
    cyl = trimesh.creation.cylinder(radius=0.0095, height=0.14, sections=48)
    gs, funnel = ap.synthesize(cyl.vertices, cyl.faces, _on_table(cyl), TABLE, seed=0)
    assert gs
    cloud = trimesh.sample.sample_surface(cyl, 8000, seed=7)[0]
    for g in gs:
        T = g.T_body_site
        local = (cloud - T[:3, 3]) @ T[:3, :3]
        S = ap.closed_state_of(g)
        assert ap.intrusions(S, local) == []
        assert ap.intrusions(ap.SIM_2F85.approach(), local, contact_tol=None) == []
        pushed = S.moved({s: S.pad(s).normal * 0.004 for s in ap.SIDES})   # 4 mm further in
        assert ap.intrusions(pushed, local)


def test_the_planners_grasp_phase_gets_the_fully_open_fingers():
    """set_gripper_joints opens the planner's grasp-phase fingers to the measured
    open state: the pads end ~98.7 mm apart, wider than the spawn state's ~64 mm."""
    import mujoco
    from manip_sim.planning import ArmKinematics
    from manip_sim.scene import load_scene, make_env
    env, _ = make_env(load_scene("scenes/insert_marker_upright.json"), robot="UR5e",
                      has_renderer=False)
    kin = ArmKinematics(env)

    def pad_gap():
        ids = [mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"gripper0_right_{s}_fingerpad_collision") for s in ("left", "right")]
        return float(np.linalg.norm(kin.data.geom_xpos[ids[0]] - kin.data.geom_xpos[ids[1]]))
    before = pad_gap()
    ap.set_gripper_joints(kin, ap.approach_joints())
    assert pad_gap() > before + 0.02
