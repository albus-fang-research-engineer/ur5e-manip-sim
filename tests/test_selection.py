"""Selection -> w-frame wiring contract: shared ID space between menu
and resolver, the axis resolution ladder (refined column / coarse-kept /
frames.json fallback), the anchoring guarantee (directions are position-
free; the candidate xyz anchors the triad), sign application, typed
resolution failures, and the generalized per-row Bw coupling in frames
whose z is NOT the refined up (the pour frame's z = tilt case), with
refine_frame's rejection asymmetry preserved per component."""

import json

import numpy as np
import pytest

from manip_sim.frames import Symbols
from manip_sim.refine import RefineResult
from manip_sim.refine_frame import (assemble_frame, azimuth_from_semantic,
                                    azimuth_from_vector)
from manip_sim.selection import (ResolutionError, couple_resolved,
                                 couple_rot_bounds_in_frame, load_pool,
                                 load_selections, menu_from_pool,
                                 refine_body_basis, resolve_selection,
                                 selection_from_json, selection_to_json)
from manip_sim.tsr import FREE_ROT, bounds
from manip_sim.vlm import PointAxisSelection

from test_refine import make_revolution_cloud


UP = np.array([0.0, 0.0, 1.0])
FRONT = np.array([1.0, 0.0, 0.0])


def _up_result(sigma_deg=0.5, accepted=True) -> RefineResult:
    return RefineResult(UP.copy(), UP.copy(), "revolution", accepted,
                        5.0, 0.001, sigma_deg, 4000)


def _basis(up_sigma=0.5, az_sigma=2.0, up_ok=True, az_ok=True):
    up = _up_result(up_sigma, up_ok)
    if az_ok:
        cands = [azimuth_from_vector(FRONT, up.direction, az_sigma)]
    else:
        cands = [azimuth_from_vector(UP, up.direction, az_sigma)]  # rejected
    return assemble_frame(up, cands)


def _symbols():
    return Symbols(
        object="teapot",
        points={"spout_tip": np.array([-0.0861, 0.0848, 0.0382]),
                "handle_center": np.array([0.0656, -0.0661, 0.0019])},
        axes={"pour_axis": np.array([-0.7243, 0.6895, 0.0]),
              "up_axis": np.array([0.0, 0.0, 1.0]),
              "handle_axis": np.array([0.0109, 0.023, 0.9997]),
              "tilt_axis": np.array([-0.6895, -0.7243, 0.0])},
    )


def _pool():
    return {0: {"id": 0, "xyz": np.array([-0.0861, 0.0848, 0.0382]),
                "source": "constructed", "symbol": "spout_tip",
                "part": "spout", "on_surface": False},
            1: {"id": 1, "xyz": np.array([0.0656, -0.0661, 0.0019]),
                "source": "constructed", "symbol": "handle_center",
                "part": "handle", "on_surface": False},
            2: {"id": 2, "xyz": np.array([0.05, 0.05, 0.09]),
                "source": "part", "part": "lid", "on_surface": True},
            3: {"id": 3, "xyz": np.array([0.0, -0.08, 0.02]),
                "source": "fps", "on_surface": True}}


def _sel(**kw):
    d = dict(candidate_id=0, axis="teapot.pour_axis", sign="+",
             secondary=None, rationale="")
    d.update(kw)
    return PointAxisSelection(**d)


# ------------------------------------------------------- pool / menu glue

def test_load_pool_and_menu_share_ids(tmp_path):
    cands = [{"id": i, **{k: (list(v) if isinstance(v, np.ndarray) else v)
                          for k, v in c.items() if k != "id"}}
             for i, c in _pool().items()]
    (tmp_path / "candidates.json").write_text(
        json.dumps({"object": "teapot", "candidates": cands}))
    pool = load_pool(tmp_path)
    menu = menu_from_pool(pool)
    assert set(menu) == set(pool) == {0, 1, 2, 3}
    assert "spout_tip" in menu[0] and "lid" in menu[2]
    assert isinstance(pool[0]["xyz"], np.ndarray)


def test_load_pool_missing_is_typed(tmp_path):
    with pytest.raises(ResolutionError, match="propose_interaction_points"):
        load_pool(tmp_path)


# --------------------------------------------------- resolution: no basis

def test_frames_json_arm_matches_symbols_frame():
    """Without a basis the resolved frame is byte-for-byte the hand-
    authored composition path, anchored at the candidate."""
    sym, pool = _symbols(), _pool()
    rf = resolve_selection(_sel(candidate_id=1, axis="teapot.handle_axis"),
                           pool, sym)
    ref = sym.frame("handle_center", "handle_axis")
    assert np.allclose(rf.frame.T(), ref.T())
    assert rf.axis_source == "frames.json"
    assert rf.secondary_source == "default"
    assert rf.basis is None


def test_sign_token_flips_resolved_axis():
    sym, pool = _symbols(), _pool()
    plus = resolve_selection(_sel(sign="+"), pool, sym)
    minus = resolve_selection(_sel(sign="-"), pool, sym)
    assert np.allclose(plus.frame.T()[:3, 2], -minus.frame.T()[:3, 2])


def test_named_secondary_resolves():
    sym, pool = _symbols(), _pool()
    rf = resolve_selection(_sel(axis="teapot.tilt_axis",
                                secondary="teapot.pour_axis"), pool, sym)
    ref = sym.frame("spout_tip", "tilt_axis", secondary="pour_axis")
    assert np.allclose(rf.frame.T(), ref.T())
    assert rf.secondary_source == "frames.json"


# ------------------------------------------------- resolution: with basis

def test_refined_columns_resolve_and_anchor_at_candidate():
    """The anchoring guarantee: refined directions are position-free;
    the SAME basis anchored at two different candidates gives identical
    orientation and the candidates' own origins."""
    sym, pool, basis = _symbols(), _pool(), _basis()
    a = resolve_selection(_sel(candidate_id=0, axis="teapot.up_axis"),
                          pool, sym, basis=basis)
    b = resolve_selection(_sel(candidate_id=2, axis="teapot.up_axis"),
                          pool, sym, basis=basis)
    assert a.axis_source == b.axis_source == "refined"
    assert np.allclose(a.frame.T()[:3, :3], b.frame.T()[:3, :3])
    assert np.allclose(a.frame.T()[:3, 3], pool[0]["xyz"])
    assert np.allclose(b.frame.T()[:3, 3], pool[2]["xyz"])
    # secondary defaulted to the refined front -> frame IS the basis
    assert np.allclose(a.frame.T()[:3, :3], basis.R)
    assert a.secondary_source == "refined"


def test_refined_frame_is_gs_fixed_point():
    """Columns of one orthonormal basis pass through Frame.T()'s
    Gram-Schmidt unchanged — GS as consistency check, not construction."""
    sym, pool, basis = _symbols(), _pool(), _basis()
    rf = resolve_selection(_sel(axis="teapot.tilt_axis",
                                secondary="teapot.pour_axis"),
                           pool, sym, basis=basis)
    R = rf.frame.T()[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(R[:, 2], basis.R[:, 1])   # z = tilt (left column)
    assert np.allclose(R[:, 0], basis.R[:, 0])   # x = pour, untouched by GS


def test_unmapped_axis_falls_back_to_frames_json_with_basis():
    sym, pool, basis = _symbols(), _pool(), _basis()
    rf = resolve_selection(_sel(candidate_id=1, axis="teapot.handle_axis"),
                           pool, sym, basis=basis)
    assert rf.axis_source == "frames.json"
    assert np.allclose(rf.frame.axis, sym.axes["handle_axis"])


def test_in_plane_columns_refuse_arbitrary_x():
    """Azimuth rejected -> basis x/y are arbitrary; pour/tilt must NOT
    resolve from the basis (frames.json fallback, recorded)."""
    sym, pool = _symbols(), _pool()
    basis = _basis(az_ok=False)
    assert not basis.accepted
    rf = resolve_selection(_sel(axis="teapot.pour_axis"), pool, sym,
                           basis=basis)
    assert rf.axis_source == "frames.json"
    assert np.allclose(rf.frame.axis, sym.axes["pour_axis"])
    # and the omitted secondary must not silently use the arbitrary front
    assert rf.secondary_source == "default"


def test_coarse_kept_up_is_placeholder():
    sym, pool = _symbols(), _pool()
    basis = _basis(up_ok=False)
    rf = resolve_selection(_sel(axis="teapot.up_axis"), pool, sym,
                           basis=basis)
    assert rf.axis_source == "coarse-kept"
    assert rf.frame.status == "placeholder"


# ------------------------------------------------------- typed failures

def test_unknown_candidate_id_is_typed():
    with pytest.raises(ResolutionError, match="candidate_id 99"):
        resolve_selection(_sel(candidate_id=99), _pool(), _symbols())


def test_world_axis_rejected():
    with pytest.raises(ResolutionError, match="world axis"):
        resolve_selection(_sel(axis="world.z"), _pool(), _symbols())


def test_cross_object_axis_rejected():
    with pytest.raises(ResolutionError, match="cross-object"):
        resolve_selection(_sel(axis="mug.up_axis"), _pool(), _symbols())


def test_unknown_axis_name_rejected():
    with pytest.raises(ResolutionError, match="grounded axes"):
        resolve_selection(_sel(axis="teapot.spout_axis"), _pool(),
                          _symbols())


# ------------------------------------------------------- coupling, general

def test_coupling_reduces_to_frame_result_rule_on_basis_frame():
    basis = _basis(up_sigma=0.5, az_sigma=4.0)
    authored = bounds(roll=(-np.deg2rad(1), np.deg2rad(1)),
                      pitch=(-np.deg2rad(1), np.deg2rad(1)),
                      yaw=(-np.deg2rad(1), np.deg2rad(1)))
    assert np.allclose(
        couple_rot_bounds_in_frame(basis, basis.R, authored),
        basis.couple_rot_bounds(authored))


def test_coupling_permutes_with_tilt_frame():
    """The pour frame: z = tilt (left), x = pour (front) -> y = -up.
    Pitch (about -up) is the azimuth-governed row; roll and yaw are
    tilt-governed. This is the row-sigma permutation the wiring exists
    to get right."""
    basis = _basis(up_sigma=0.5, az_sigma=4.0)
    R_pour = np.column_stack([basis.R[:, 0],            # x = pour
                              np.cross(basis.R[:, 1], basis.R[:, 0]),
                              basis.R[:, 1]])           # z = tilt
    assert np.allclose(R_pour[:, 1], -basis.R[:, 2])    # y = -up
    tiny = bounds(roll=(-1e-4, 1e-4), pitch=(-1e-4, 1e-4),
                  yaw=(-1e-4, 1e-4))
    Bw = couple_rot_bounds_in_frame(basis, R_pour, tiny, k=3.0)
    half = np.degrees(0.5 * (Bw[3:, 1] - Bw[3:, 0]))
    az = basis.azimuth.sigma_deg
    assert half[0] == pytest.approx(3 * 0.5, rel=1e-6)   # roll <- up sigma
    assert half[1] == pytest.approx(3 * az, rel=1e-6)    # pitch <- azimuth
    assert half[2] == pytest.approx(3 * 0.5, rel=1e-6)   # yaw <- up sigma


def test_coupling_mixes_variances_at_oblique_axis():
    basis = _basis(up_sigma=1.0, az_sigma=5.0)
    c = np.cos(np.pi / 4)
    z = np.array([c, 0.0, c])                    # 45 deg between front, up
    x = np.array([c, 0.0, -c])
    R = np.column_stack([x, np.cross(z, x), z])
    tiny = bounds(yaw=(-1e-4, 1e-4))
    Bw = couple_rot_bounds_in_frame(basis, R, tiny, k=3.0)
    expect = 3 * np.sqrt(0.5 * 1.0 ** 2 + 0.5 * 5.0 ** 2)
    got = np.degrees(0.5 * (Bw[5, 1] - Bw[5, 0]))
    assert got == pytest.approx(expect, rel=1e-6)


def test_coupling_az_rejected_frees_up_component_rows_only():
    basis = _basis(az_ok=False)
    R_pour = np.column_stack([basis.R[:, 0],
                              np.cross(basis.R[:, 1], basis.R[:, 0]),
                              basis.R[:, 1]])           # y = -up
    authored = bounds(roll=(-np.deg2rad(2), np.deg2rad(2)),
                      pitch=(-np.deg2rad(2), np.deg2rad(2)),
                      yaw=(-np.deg2rad(2), np.deg2rad(2)))
    Bw = couple_rot_bounds_in_frame(basis, R_pour, authored)
    assert tuple(Bw[4]) == FREE_ROT                     # pitch about -up
    # roll/yaw are pure-tilt rows: floored by the up sigma, not freed
    for row in (3, 5):
        half = 0.5 * (Bw[row, 1] - Bw[row, 0])
        assert half == pytest.approx(
            max(np.deg2rad(2), 3 * np.deg2rad(basis.up.sigma_deg)))


def test_coupling_up_rejected_keeps_authored_tilt_component():
    basis = _basis(up_ok=False, az_sigma=4.0)
    authored = bounds(roll=(-np.deg2rad(2), np.deg2rad(2)),
                      pitch=(-np.deg2rad(2), np.deg2rad(2)),
                      yaw=(-np.deg2rad(2), np.deg2rad(2)))
    Bw = couple_rot_bounds_in_frame(basis, basis.R, authored)
    for row in (3, 4):                                  # authored kept
        assert np.allclose(Bw[row], authored[row])
    half = 0.5 * (Bw[5, 1] - Bw[5, 0])                  # az still floors
    assert half == pytest.approx(3 * np.deg2rad(4.0) /
                                 basis.azimuth.conditioning)


def test_coupling_preserves_translation_free_rows_and_midpoints():
    basis = _basis()
    Bw0 = bounds(x=(-0.01, 0.01), z=(0.02, 0.05),
                 roll=FREE_ROT, pitch=(0.0, np.deg2rad(2)),
                 yaw=(-np.deg2rad(5), np.deg2rad(5)))
    Bw = couple_rot_bounds_in_frame(basis, basis.R, Bw0)
    assert np.allclose(Bw[:3], Bw0[:3])                 # translations
    assert tuple(Bw[3]) == FREE_ROT                     # free stays free
    mid0 = 0.5 * (Bw0[4, 0] + Bw0[4, 1])                # off-center pitch
    assert 0.5 * (Bw[4, 0] + Bw[4, 1]) == pytest.approx(mid0)


def test_couple_resolved_ground_truth_arm_passthrough():
    rf = resolve_selection(_sel(), _pool(), _symbols())
    Bw0 = bounds(yaw=(-0.01, 0.01))
    assert np.allclose(couple_resolved(rf, Bw0), Bw0)


# ------------------------------------------------ basis builder round trip

def test_refine_body_basis_on_synthetic_revolution():
    from test_refine import _tilt, TRUE
    P = make_revolution_cloud(TRUE, n=4000, noise=0.0005)
    coarse_up = _tilt(TRUE, 25.0)
    front = np.cross(TRUE, [0.0, 0.0, 1.0])
    front = front / np.linalg.norm(front)          # true in-plane front
    fr = refine_body_basis(P, coarse_up, _tilt(front, 15.0, seed=1),
                           front_pair=(np.zeros(3), 0.12 * front))
    assert fr.up.accepted and fr.accepted
    assert fr.azimuth.route == "constructed"
    assert np.degrees(np.arccos(abs(fr.R[:, 2] @ TRUE))) < 1.0
    R = fr.R
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


# ----------------------------------------------------------- serialization

def test_selection_json_round_trip(tmp_path):
    sel = _sel(candidate_id=2, axis="teapot.tilt_axis", sign="-",
               secondary="teapot.pour_axis", rationale="pivot at spout")
    assert selection_from_json(selection_to_json(sel)) == sel
    path = tmp_path / "selections.json"
    path.write_text(json.dumps({"pour": selection_to_json(sel),
                                "grasp": selection_to_json(_sel())}))
    roles = load_selections(path)
    assert roles["pour"] == sel and roles["grasp"] == _sel()