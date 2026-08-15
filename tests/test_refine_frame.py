"""Frame assembly contract on synthetic geometry: per-route azimuth
sigma propagation, conditioning/elongation gates, explicit fallback
order, the yaw-free-on-rejection asymmetry, and row-wise Bw coupling
(roll/pitch from the up fit, yaw from the azimuth route — never
cross-contaminated)."""

import numpy as np
import pytest

from manip_sim.refine import RefineResult, refine_axis
from manip_sim.refine_frame import (MIN_CONDITIONING, SEMANTIC_SIGMA_DEG,
                                    assemble_frame, azimuth_from_part_cloud,
                                    azimuth_from_points,
                                    azimuth_from_semantic,
                                    azimuth_from_vector)
from manip_sim.tsr import FREE_ROT, bounds

from test_refine import TRUE, _tilt, make_revolution_cloud


UP = np.array([0.0, 0.0, 1.0])


def _up_result(sigma_deg=0.5, accepted=True) -> RefineResult:
    """A stand-in refined up without running a fit."""
    return RefineResult(UP.copy(), UP.copy(), "revolution", accepted,
                        5.0, 0.001, sigma_deg, 4000)


def _ang(u, v):
    return np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1)))


# ------------------------------------------------------------ route: vector

def test_vector_projection_and_sigma_amplification():
    # source 45 deg out of plane: conditioning cos(45), sigma amplified
    src = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    res = azimuth_from_vector(src, UP, sigma_deg=4.0)
    assert res.accepted
    assert np.allclose(res.direction, [1.0, 0.0, 0.0], atol=1e-12)
    assert abs(res.direction @ UP) < 1e-12
    assert res.conditioning == pytest.approx(np.cos(np.pi / 4), abs=1e-9)
    assert res.sigma_deg == pytest.approx(4.0 / np.cos(np.pi / 4), rel=1e-6)


def test_vector_near_parallel_rejected():
    src = _tilt(UP, 10.0)                      # 10 deg off up: sin10 < gate
    res = azimuth_from_vector(src, UP, sigma_deg=4.0)
    assert not res.accepted
    assert res.conditioning < MIN_CONDITIONING
    assert "not informative" in res.note


# ------------------------------------------------------- route: constructed

def test_constructed_sigma_from_lever_arm():
    # 86 mm horizontal lever, 2 mm point sigma -> ~1.9 deg yaw sigma
    res = azimuth_from_points(np.zeros(3), np.array([0.086, 0.0, 0.01]),
                              UP, point_sigma_m=0.002)
    assert res.accepted and res.route == "constructed"
    expect = np.degrees(np.arctan2(np.sqrt(2) * 0.002, 0.086))
    assert res.sigma_deg == pytest.approx(expect, rel=1e-6)
    assert _ang(res.direction, [1, 0, 0]) < 1e-6


def test_constructed_vertical_pair_rejected():
    res = azimuth_from_points(np.zeros(3), np.array([0.005, 0.0, 0.09]),
                              UP, point_sigma_m=0.002)
    assert not res.accepted
    assert "near-vertical" in res.note


# --------------------------------------------------------- route: part_pca

def _tube(direction, n=800, L=0.08, r=0.008, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    perp = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(d, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(d, perp)
    t = rng.uniform(-L / 2, L / 2, n)
    th = rng.uniform(0, 2 * np.pi, n)
    return (np.outer(t, d) + r * np.outer(np.cos(th), perp)
            + r * np.outer(np.sin(th), perp2)
            + rng.normal(scale=0.0005, size=(n, 3)))


def test_part_pca_spout_tube():
    # spout tilted 30 deg above horizontal; coarse front only sign-grade
    true_dir = np.array([np.cos(np.pi / 6), 0.0, np.sin(np.pi / 6)])
    res = azimuth_from_part_cloud(_tube(true_dir), UP,
                                  coarse_front=[0.7, 0.3, 0.0])
    assert res.accepted and res.route == "part_pca"
    assert _ang(res.direction, [1, 0, 0]) < 2.0
    assert res.sigma_deg < 3.0


def test_part_pca_sign_follows_coarse():
    true_dir = np.array([1.0, 0.0, 0.3])
    res = azimuth_from_part_cloud(_tube(true_dir), UP,
                                  coarse_front=[-1.0, 0.1, 0.0])
    assert res.accepted
    assert res.direction @ np.array([1.0, 0.0, 0.0]) < 0  # flipped w/ coarse


def test_part_pca_blob_rejected_on_elongation():
    rng = np.random.default_rng(3)
    blob = rng.normal(scale=0.01, size=(600, 3))
    res = azimuth_from_part_cloud(blob, UP, coarse_front=[1, 0, 0])
    assert not res.accepted
    assert "not elongated" in res.note


def test_part_pca_vertical_tube_rejected_on_conditioning():
    # the real-teapot handle case: elongated but ~parallel to up
    res = azimuth_from_part_cloud(_tube(_tilt(UP, 5.0)), UP,
                                  coarse_front=[1, 0, 0])
    assert not res.accepted
    assert res.conditioning < MIN_CONDITIONING


# ---------------------------------------------------------- route: semantic

def test_semantic_passes_declared_sigma_through():
    res = azimuth_from_semantic([1.0, 0.2, 0.1], UP)
    assert res.accepted and res.route == "semantic"
    assert res.sigma_deg >= SEMANTIC_SIGMA_DEG          # / conditioning


# ----------------------------------------------------------------- assembly

def test_assembly_orthonormal_right_handed():
    up = _up_result()
    fr = assemble_frame(up, [azimuth_from_semantic([1, 0, 0], UP)])
    R = fr.R
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(R[:, 2], UP)
    assert np.allclose(np.cross(R[:, 2], R[:, 0]), R[:, 1])


def test_assembly_fallback_order_and_note():
    up = _up_result()
    bad = azimuth_from_points(np.zeros(3), np.array([0.0, 0.0, 0.09]),
                              UP, 0.002)                 # rejected
    good = azimuth_from_semantic([0, 1, 0], UP)
    fr = assemble_frame(up, [bad, good])
    assert fr.accepted
    assert fr.azimuth.route == "semantic"
    assert "constructed rejected" in fr.note


def test_assembly_all_rejected_yields_unaccepted_frame():
    up = _up_result()
    bad = azimuth_from_vector(_tilt(UP, 5.0), UP, 4.0)
    fr = assemble_frame(up, [bad])
    assert not fr.accepted
    R = fr.R
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)   # still a frame
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_assembly_rejects_mismatched_up():
    up = _up_result()
    cand = azimuth_from_semantic([1, 0, 0], _tilt(UP, 15.0))
    with pytest.raises(ValueError, match="different up"):
        assemble_frame(up, [cand])


# ----------------------------------------------------------------- coupling

def test_coupling_rows_are_independent():
    up = _up_result(sigma_deg=0.4)                       # tight fit
    az = azimuth_from_points(np.zeros(3), np.array([0.086, 0, 0.01]),
                             UP, 0.002)                  # ~1.9 deg
    fr = assemble_frame(up, [az])
    authored = bounds(z=(0.0, 0.05),
                      roll=(-np.deg2rad(0.1), np.deg2rad(0.1)),
                      pitch=(-np.deg2rad(0.1), np.deg2rad(0.1)),
                      yaw=(-np.deg2rad(1.0), np.deg2rad(1.0)))
    Bw = fr.couple_rot_bounds(authored, k=3.0)
    assert np.allclose(Bw[:3], authored[:3])             # translations pass
    # roll/pitch floored by 3 * up sigma, NOT by azimuth sigma
    half_rp = 0.5 * (Bw[3, 1] - Bw[3, 0])
    assert half_rp == pytest.approx(3 * np.deg2rad(0.4), rel=1e-6)
    assert np.allclose(Bw[3], Bw[4])
    # yaw floored by 3 * azimuth sigma, NOT by up sigma
    half_y = 0.5 * (Bw[5, 1] - Bw[5, 0])
    assert half_y == pytest.approx(3 * np.deg2rad(az.sigma_deg), rel=1e-6)


def test_coupling_authored_wider_than_floor_untouched():
    up = _up_result(sigma_deg=0.4)
    fr = assemble_frame(up, [azimuth_from_semantic([1, 0, 0], UP)])
    authored = bounds(roll=(-0.5, 0.5), pitch=(-0.5, 0.5),
                      yaw=FREE_ROT)
    Bw = fr.couple_rot_bounds(authored)
    assert np.allclose(Bw[3], (-0.5, 0.5))               # 0.5 rad >> 3 sigma
    assert np.allclose(Bw[5], FREE_ROT)                  # free stays free


def test_coupling_preserves_midpoint():
    up = _up_result(sigma_deg=2.0)
    az = azimuth_from_semantic([1, 0, 0], UP)
    fr = assemble_frame(up, [az])
    authored = bounds(roll=(np.deg2rad(29.9), np.deg2rad(30.1)),
                      yaw=(np.deg2rad(179), np.deg2rad(181)))
    Bw = fr.couple_rot_bounds(authored, k=3.0)
    assert 0.5 * (Bw[3, 0] + Bw[3, 1]) == pytest.approx(np.deg2rad(30))
    assert 0.5 * (Bw[5, 0] + Bw[5, 1]) == pytest.approx(np.deg2rad(180))


def test_coupling_rejected_azimuth_frees_yaw_keeps_roll_pitch():
    # the documented asymmetry: rejected up -> authored kept;
    # rejected azimuth -> yaw FREE (x is arbitrary)
    up = _up_result(accepted=False)
    bad = azimuth_from_vector(_tilt(UP, 5.0), UP, 4.0)
    fr = assemble_frame(up, [bad])
    authored = bounds(roll=(-0.01, 0.01), pitch=(-0.01, 0.01),
                      yaw=(-0.01, 0.01))
    Bw = fr.couple_rot_bounds(authored)
    assert np.allclose(Bw[3], (-0.01, 0.01))             # authored kept
    assert np.allclose(Bw[5], FREE_ROT)                  # yaw freed


def test_coupling_huge_sigma_caps_to_free():
    up = _up_result(sigma_deg=0.4)
    az = azimuth_from_vector([1, 0, 0.1], UP, sigma_deg=70.0)  # 3s > pi
    fr = assemble_frame(up, [az])
    Bw = fr.couple_rot_bounds(bounds(yaw=(-0.1, 0.1)))
    assert np.allclose(Bw[5], FREE_ROT)


# -------------------------------------------------------------- integration

def test_end_to_end_with_real_up_fit_and_to_frame():
    """Refined up from the synthetic revolution cloud + constructed
    azimuth -> FrameResult -> frames.Frame, whose T() rotation must
    reproduce R exactly (x already perpendicular to z, so the Frame's
    Gram-Schmidt is a fixed point)."""
    P = make_revolution_cloud(TRUE)
    up = refine_axis(P, _tilt(TRUE, 25.0), "revolution")
    assert up.accepted
    # a constructed pair whose in-plane lever is well conditioned
    e1 = np.cross(up.direction, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    az = azimuth_from_points(np.zeros(3), 0.09 * e1 + 0.01 * up.direction,
                             up.direction, point_sigma_m=0.002)
    fr = assemble_frame(up, [az])
    assert fr.accepted
    f = fr.to_frame("teapot.pour_test", origin=[0.01, 0.02, 0.03])
    assert f.status == "calibrated"
    T = f.T()
    assert np.allclose(T[:3, :3], fr.R, atol=1e-9)
    assert np.allclose(T[:3, 3], [0.01, 0.02, 0.03])