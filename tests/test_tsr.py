"""Unit tests for the TSR core and the hand-authored pour stages.

Pure numpy/scipy — runs without MuJoCo. The pour-stage tests are the
important ones: they assert the *geometric meaning* of the frames (spout tip
lands over the opening; positive tilt sends the spout DOWN and pivots about
the tip), which is exactly the class of sign/convention bug that is
miserable to diagnose inside the simulator.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from manip_sim.frames import Frame
from manip_sim.pour_stages import pour_pair, transport_pair
from manip_sim.tsr import (
    FREE_ROT,
    FREE_TRANS,
    TSR,
    bounds,
    make_pose,
    sample_intersection,
)

RNG = np.random.default_rng(0)


def random_pose(rng) -> np.ndarray:
    return make_pose(rng.normal(size=3), R.random(rng=rng).as_matrix())


def random_bounded_tsr(rng) -> TSR:
    lo_t = rng.uniform(-0.2, 0.0, 3)
    hi_t = lo_t + rng.uniform(0.0, 0.3, 3)
    lo_r = rng.uniform(-1.0, 0.0, 3)
    hi_r = lo_r + rng.uniform(0.0, 1.5, 3)
    Bw = np.column_stack([np.concatenate([lo_t, lo_r]),
                          np.concatenate([hi_t, hi_r])])
    return TSR(T0_w=random_pose(rng), Tw_e=random_pose(rng), Bw=Bw)


# ------------------------------------------------------------------- core


def test_zero_displacement_at_zero_pose():
    for _ in range(20):
        t = random_bounded_tsr(RNG)
        # zero() only lies inside the TSR if 0 is inside Bw; test displacement
        d = t.displacement(t.T0_w @ t.Tw_e)
        assert np.allclose(d, 0.0, atol=1e-9)


def test_sample_is_contained():
    for _ in range(20):
        t = random_bounded_tsr(RNG)
        for _ in range(20):
            assert t.contains(t.sample(RNG), tol=1e-7)


def test_sample_displacement_roundtrip():
    for _ in range(20):
        t = random_bounded_tsr(RNG)
        d = t.sample_displacement(RNG)
        from manip_sim.tsr import displacement_to_pose
        T = t.T0_w @ displacement_to_pose(d) @ t.Tw_e
        d2 = t.displacement(T)
        assert np.allclose(d, d2, atol=1e-7), (d, d2)


def test_project_contains_and_idempotent():
    for _ in range(30):
        t = random_bounded_tsr(RNG)
        T = random_pose(RNG)
        P = t.project(T)
        assert t.contains(P, tol=1e-7)
        P2 = t.project(P)
        assert np.allclose(P, P2, atol=1e-7)


def test_project_noop_inside():
    t = random_bounded_tsr(RNG)
    T = t.sample(RNG)
    assert np.allclose(t.project(T), T, atol=1e-7)


def test_distance_positive_outside():
    t = TSR(T0_w=np.eye(4), Bw=bounds(z=(0.0, 0.1), yaw=FREE_ROT))
    far = make_pose([0.0, 0.0, 0.5])
    assert t.distance(far) == pytest.approx(0.4, abs=1e-9)
    assert not t.contains(far)


def test_angle_wrap_across_pi():
    """Bounds authored past pi: yaw in [170deg, 190deg] must admit a pose at
    yaw = -175deg (== +185deg)."""
    t = TSR(
        T0_w=np.eye(4),
        Bw=bounds(yaw=(np.deg2rad(170), np.deg2rad(190))),
    )
    T = make_pose(rot=R.from_euler("z", np.deg2rad(-175)).as_matrix())
    assert t.contains(T, tol=1e-7)
    # and the representative reported is the shifted one
    d = t.displacement(T)
    assert np.deg2rad(170) - 1e-9 <= d[5] <= np.deg2rad(190) + 1e-9


def test_dual_rpy_solution():
    """Ry(pi) has primary scipy decomposition (pi, 0, pi); a TSR pinned to
    roll=yaw=0 with pitch near pi must accept it via the dual solution."""
    t = TSR(
        T0_w=np.eye(4),
        Bw=bounds(roll=(-0.05, 0.05), pitch=(3.0, 3.3), yaw=(-0.05, 0.05)),
    )
    T = make_pose(rot=R.from_euler("y", np.pi).as_matrix())
    assert t.contains(T, tol=1e-7)


def test_sampling_unbounded_raises():
    t = TSR(T0_w=np.eye(4), Bw=bounds(x=FREE_TRANS))
    with pytest.raises(ValueError):
        t.sample(RNG)


def test_intersection_report():
    sampler = TSR(T0_w=np.eye(4), Bw=bounds(z=(0.0, 1.0)), name="sub")
    gate = TSR(T0_w=np.eye(4), Bw=bounds(z=(0.0, 0.5)), name="path")
    rep = sample_intersection(sampler, [gate], n=200, rng=np.random.default_rng(1))
    assert 0.35 < rep.acceptance_rate < 0.65      # ~half the corridor
    assert all(gate.contains(T, tol=1e-7) for T in rep.accepted)
    assert "path" in rep.per_constraint_rejections


# ------------------------------------------------- pour stages, synthetic scene


def _scene():
    """Synthetic world mirroring view_pour_tea.py: teapot at (0,-0.25),
    mug at (0,+0.25), table top z=0.8, teapot yawed so the spout (body
    direction at yaw +2.7 from +x) faces the mug."""
    spout_yaw = 2.7
    bearing = np.arctan2(0.5, 0.0)                     # +y toward the mug
    teapot_yaw = bearing - spout_yaw
    T0_teapot = make_pose([0.0, -0.25, 0.86],
                          R.from_euler("z", teapot_yaw).as_matrix())
    T0_mug = make_pose([0.0, 0.25, 0.86])
    c, s = np.cos(spout_yaw), np.sin(spout_yaw)
    spout_tip = Frame("spout_tip", np.array([-0.09, 0.043, 0.08]),
                      np.array([c, s, 0.0]), np.array([0.0, 0.0, -1.0]))
    tilt = Frame("tilt_axis", spout_tip.point.copy(),
                 np.array([-s, c, 0.0]), np.array([c, s, 0.0]))
    opening = Frame("opening", np.array([0.0, 0.0, 0.10]),
                    np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    return T0_teapot, T0_mug, spout_tip, tilt, opening


def test_transport_subgoal_places_spout_over_opening():
    T0_teapot, T0_mug, spout_tip, _, opening = _scene()
    pair = transport_pair(
        T0_mug_body=T0_mug,
        mug_opening=opening,
        spout_tip=spout_tip,
        teapot_body_pos_now=T0_teapot[:3, 3],
    )
    rng = np.random.default_rng(2)
    rep = sample_intersection(pair.subgoal, [pair.path], n=100, rng=rng)
    assert rep.acceptance_rate > 0.5, rep.summary()   # consistent hand pair
    opening_world = (T0_mug @ opening.T())[:3, 3]
    for T_body in rep.accepted:
        tip_world = (T_body @ spout_tip.T())[:3, 3]
        # tip inside the rim square, in the height band
        assert abs(tip_world[0] - opening_world[0]) <= 0.02 + 1e-6
        assert abs(tip_world[1] - opening_world[1]) <= 0.02 + 1e-6
        assert 0.03 - 1e-6 <= tip_world[2] - opening_world[2] <= 0.08 + 1e-6
        # teapot near-upright: body z-axis within 15deg-ish of world z
        # (per-axis roll/pitch bounds admit slightly more on the diagonal)
        cos_tilt = (T_body[:3, :3] @ np.array([0, 0, 1.0]))[2]
        assert cos_tilt >= np.cos(np.deg2rad(22.0))


def test_pour_positive_roll_tips_spout_down_about_tip_pivot():
    T0_teapot, _, spout_tip, tilt, _ = _scene()
    pair = pour_pair(T0_body_at_entry=T0_teapot, tilt_frame=tilt,
                     tilt_target=np.deg2rad(95), tilt_tol=np.deg2rad(2),
                     pivot_tol=1e-6, off_axis_tol=1e-6)
    rng = np.random.default_rng(3)
    tip0 = (T0_teapot @ spout_tip.T())[:3, 3]
    body0 = T0_teapot[:3, 3]
    spout_dir0 = (T0_teapot[:3, :3] @ spout_tip.axis)
    for _ in range(10):
        T_body = pair.subgoal.sample(rng)
        # pivot: the spout tip stays put
        tip = (T_body @ spout_tip.T())[:3, 3]
        assert np.allclose(tip, tip0, atol=1e-4), (tip, tip0)
        # the body center swings (it is NOT the pivot)
        assert np.linalg.norm(T_body[:3, 3] - body0) > 0.05
        # spout direction rotated ~95deg toward -z: strongly downward
        spout_dir = T_body[:3, :3] @ spout_tip.axis
        assert spout_dir[2] < -0.9, spout_dir
        # entry pose is on the PATH tsr (roll ~ 0 inside tilt_range)
        assert pair.path.contains(T0_teapot, tol=1e-6)
        # and the sampled pour attitude is too (subgoal within path)
        assert pair.path.contains(T_body, tol=1e-6)


def test_pour_path_rejects_wrong_direction_tilt():
    """Tilting the spout UP (negative roll beyond the settle allowance) must
    leave the pour path TSR."""
    T0_teapot, _, spout_tip, tilt, _ = _scene()
    pair = pour_pair(T0_body_at_entry=T0_teapot, tilt_frame=tilt)
    # rotate the body by -30deg about the tilt axis anchored at the tip
    axis_w = T0_teapot[:3, :3] @ tilt.axis
    tip_w = (T0_teapot @ spout_tip.T())[:3, 3]
    Rot = R.from_rotvec(-np.deg2rad(30) * axis_w).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rot
    T[:3, 3] = tip_w - Rot @ tip_w
    T_body = T @ T0_teapot
    assert not pair.path.contains(T_body, tol=1e-6)
    # +30deg the same way IS on the path
    Rot = R.from_rotvec(+np.deg2rad(30) * axis_w).as_matrix()
    T[:3, :3] = Rot
    T[:3, 3] = tip_w - Rot @ tip_w
    assert pair.path.contains(T @ T0_teapot, tol=1e-6)


# ------------------------------------------------------------- symbol sidecars


def test_load_symbols_and_compose(tmp_path):
    import json
    from manip_sim.frames import load_symbols
    spec = {
        "object": "teapot",
        "points": {"spout_tip": {"xyz": [-0.09, 0.043, 0.08],
                                 "status": "placeholder"}},
        "axes": {"pour_axis": {"xyz": [-0.9041, 0.4274, 0.0]},
                 "up_axis": {"xyz": [0, 0, 1]},
                 "tilt_axis": {"xyz": [-0.4274, -0.9041, 0.0]}},
        "quantities": {"spout_len": {"value": 0.05}},
    }
    (tmp_path / "frames.json").write_text(json.dumps(spec))
    sym = load_symbols(tmp_path)
    f = sym.frame("spout_tip", "pour_axis")
    T = f.T()
    # +z of the frame is the pour axis
    assert np.allclose(T[:3, 2], spec["axes"]["pour_axis"]["xyz"], atol=1e-4)
    assert np.allclose(T[:3, 3], spec["points"]["spout_tip"]["xyz"])
    # orthonormal, right-handed
    Rm = T[:3, :3]
    assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(Rm) == pytest.approx(1.0)
    # placeholder status propagates through composition
    assert f.status == "placeholder"
    assert sym.quantities["spout_len"] == 0.05
    # tilt frame with pour_axis as secondary: +x = pour direction
    ft = sym.frame("spout_tip", "tilt_axis", secondary="pour_axis")
    assert np.allclose(ft.T()[:3, 0], spec["axes"]["pour_axis"]["xyz"], atol=1e-4)


def test_shipped_sidecars_load():
    from manip_sim.frames import load_symbols
    teapot = load_symbols("assets/objects/teapot")
    mug = load_symbols("assets/objects/mug")
    assert {"spout_tip", "handle_center"} <= set(teapot.points)
    assert {"pour_axis", "up_axis", "tilt_axis"} <= set(teapot.axes)
    assert "opening_center" in mug.points
    # tilt axis really is up x pour (the positive-roll-pours convention)
    assert np.allclose(np.cross(teapot.axes["up_axis"], teapot.axes["pour_axis"]),
                       teapot.axes["tilt_axis"], atol=1e-3)