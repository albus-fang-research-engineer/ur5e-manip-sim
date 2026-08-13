"""Refinement contract on synthetic geometry (no meshes needed) plus a
mesh-gated check mirroring test_proposal's skip pattern. The synthetic
cases replicate the validated demo conditions: noisy, partial, and
cluttered clouds with coarse seeds well off the true axis must snap to
sub-degree accuracy; seeds outside the basin must be rejected, not
trusted; the sign must always follow the coarse input, never the fit."""

import json
from pathlib import Path

import numpy as np
import pytest

from manip_sim.refine import (MAX_SNAP_DEG, RefineResult, refine_axis,
                              snap_to_candidates)


def _rot_from_z(axis: np.ndarray) -> np.ndarray:
    """Rotation taking +z to `axis`."""
    from scipy.spatial.transform import Rotation as R
    axis = axis / np.linalg.norm(axis)
    v = np.cross([0.0, 0.0, 1.0], axis)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if axis[2] > 0 else R.from_euler(
            "x", np.pi).as_matrix()
    ang = np.arctan2(s, axis[2])
    return R.from_rotvec(v / s * ang).as_matrix()


def _tilt(axis: np.ndarray, deg: float, seed: int = 0) -> np.ndarray:
    """A unit vector `deg` degrees away from `axis`."""
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(seed)
    perp = np.cross(axis, rng.normal(size=3))
    perp /= np.linalg.norm(perp)
    return R.from_rotvec(perp * np.deg2rad(deg)).as_matrix() @ axis


def make_revolution_cloud(axis, n=4000, noise=0.001, arc_deg=360.0,
                          handle=False, seed=0):
    """Surface of revolution with a teapot-like radius profile r(h),
    optional partial azimuthal coverage (single-view shell) and an
    optional off-surface blob standing in for a handle."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-0.06, 0.06, n)
    r = 0.045 + 0.015 * np.cos(h / 0.06 * np.pi / 2)     # belly profile
    th = rng.uniform(0.0, np.deg2rad(arc_deg), n)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th), h])
    pts += rng.normal(scale=noise, size=pts.shape)
    if handle:
        m = n // 8
        blob = (np.array([0.085, 0.0, 0.0])
                + rng.normal(scale=0.008, size=(m, 3)))
        pts = np.vstack([pts, blob])
    return pts @ _rot_from_z(np.asarray(axis, float)).T


TRUE = np.array([0.36, -0.48, 0.8])
TRUE = TRUE / np.linalg.norm(TRUE)


def test_revolution_snaps_from_coarse_seed():
    P = make_revolution_cloud(TRUE)
    res = refine_axis(P, _tilt(TRUE, 25.0), "revolution")
    assert res.accepted
    err = np.degrees(np.arccos(np.clip(res.direction @ TRUE, -1, 1)))
    assert err < 1.0
    assert res.residual_rms < 0.003
    assert np.isfinite(res.sigma_deg) and res.sigma_deg < 2.0


def test_revolution_basin_edge_converges():
    P = make_revolution_cloud(TRUE)
    res = refine_axis(P, _tilt(TRUE, 50.0), "revolution")
    assert res.accepted
    err = np.degrees(np.arccos(np.clip(res.direction @ TRUE, -1, 1)))
    assert err < 2.0


def test_partial_arc_and_handle_clutter():
    # single-view half shell WITH the handle blob — the realistic worst
    # case a segmented RGB-D crop hands the fitter
    P = make_revolution_cloud(TRUE, arc_deg=180.0, noise=0.002, handle=True)
    res = refine_axis(P, _tilt(TRUE, 25.0), "revolution")
    assert res.accepted
    err = np.degrees(np.arccos(np.clip(res.direction @ TRUE, -1, 1)))
    assert err < 2.5


def test_sign_follows_coarse_not_fit():
    P = make_revolution_cloud(TRUE)
    res = refine_axis(P, _tilt(-TRUE, 20.0, seed=3), "revolution")
    # coarse points the WRONG way; refinement must faithfully keep that
    # sign (sign repair is the semantic layer's one-token edit, not ours)
    assert res.direction @ TRUE < 0


def test_out_of_basin_rejected_typed():
    P = make_revolution_cloud(TRUE, noise=0.004)
    coarse = _tilt(TRUE, 85.0)
    res = refine_axis(P, coarse, "revolution")
    if not res.accepted:                      # the contract path
        assert np.allclose(res.direction, coarse / np.linalg.norm(coarse))
        assert res.note
    else:                                     # converged anyway: fine,
        assert res.snap_deg <= MAX_SNAP_DEG   # but only within the basin


def test_too_few_points_rejected():
    res = refine_axis(np.random.default_rng(0).normal(size=(10, 3)),
                      [0, 0, 1], "revolution")
    assert not res.accepted and "points" in res.note


def test_sigma_tracks_noise():
    lo = refine_axis(make_revolution_cloud(TRUE, noise=0.0005),
                     _tilt(TRUE, 15.0), "revolution")
    hi = refine_axis(make_revolution_cloud(TRUE, noise=0.004),
                     _tilt(TRUE, 15.0), "revolution")
    assert lo.accepted and hi.accepted
    assert lo.sigma_deg < hi.sigma_deg


def test_couple_rot_bound_floor():
    P = make_revolution_cloud(TRUE, noise=0.004)
    res = refine_axis(P, _tilt(TRUE, 15.0), "revolution")
    tight = np.deg2rad(0.01)                  # authored absurdly tight
    coupled = res.couple_rot_bound(tight, k=3.0)
    assert coupled >= 3.0 * np.deg2rad(res.sigma_deg) - 1e-12
    wide = np.deg2rad(30.0)                   # authored wider than 3-sigma
    assert res.couple_rot_bound(wide) == wide
    rejected = RefineResult(TRUE, TRUE, "revolution", False, 0.0,
                            np.nan, np.nan, 0)
    assert rejected.couple_rot_bound(tight) == tight   # no trusted sigma


def test_plane_and_pca():
    rng = np.random.default_rng(1)
    n = np.array([0.2, 0.3, 0.933])
    n /= np.linalg.norm(n)
    e1, e2 = np.linalg.svd(np.outer(n, n))[0][:, 1:].T
    flat = (rng.uniform(-0.05, 0.05, (2000, 1)) * e1
            + rng.uniform(-0.05, 0.05, (2000, 1)) * e2
            + rng.normal(scale=0.0005, size=(2000, 3)))
    res = refine_axis(flat, _tilt(n, 20.0), "plane")
    assert res.accepted
    assert np.degrees(np.arccos(np.clip(res.direction @ n, -1, 1))) < 1.0

    bar_axis = np.array([0.7, 0.7, 0.14])
    bar_axis /= np.linalg.norm(bar_axis)
    bar = (rng.uniform(-0.05, 0.05, (1500, 1)) * bar_axis
           + rng.normal(scale=0.003, size=(1500, 3)))
    res = refine_axis(bar, _tilt(bar_axis, 20.0), "pca")
    assert res.accepted
    assert np.degrees(np.arccos(
        np.clip(res.direction @ bar_axis, -1, 1))) < 2.0


def test_snap_to_candidates_selector():
    cands = np.eye(3)                          # a fitted frame's axes
    i, axis, margin = snap_to_candidates([0.1, -0.15, -0.95], cands)
    assert i == 2
    assert np.allclose(axis, [0, 0, -1])       # sign resolved toward coarse
    assert margin > 0.5                        # 90-deg separation: decisive


MESH_CASES = [(n, Path(f"assets/objects/{n}"))
              for n in ("teapot", "mug")]


@pytest.mark.parametrize("name,obj_dir", MESH_CASES)
def test_refine_on_converted_meshes(name, obj_dir):
    """End-to-end on the real assets: sample the visual mesh, perturb the
    calibrated up_axis by 25 deg, refine, land back within 2 deg."""
    trimesh = pytest.importorskip("trimesh")
    mesh_path = obj_dir / "meshes" / f"{name}_visual.obj"
    if not mesh_path.exists():
        pytest.skip(f"{mesh_path} not converted")
    spec = json.loads((obj_dir / "frames.json").read_text())
    up = np.asarray(spec["axes"]["up_axis"]["xyz"], float)
    up /= np.linalg.norm(up)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, 6000, seed=0)
    res = refine_axis(np.asarray(pts), _tilt(up, 25.0), "revolution")
    assert res.accepted
    assert np.degrees(np.arccos(np.clip(res.direction @ up, -1, 1))) < 2.0
