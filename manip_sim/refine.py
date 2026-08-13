"""Geometric axis refinement — snap a coarse semantic direction onto a
primitive fitted to the actual geometry.

The division of labor (the architecture invariant this module enforces):
the semantic layer (PointSO, Orient Anything, a VLM axis token) decides
WHICH feature and WHICH sign; the metric direction comes from a
least-squares primitive fit on the observed points. A network-regressed
direction that is 15-45 deg off becomes the fit's initialization and its
sign reference, never the answer. The frame error is then the fit
residual on real geometry — a quantifiable number — instead of "whatever
the network regressed", which is what lets rotational Bw widths be
lower-bounded by estimator uncertainty (couple_rot_bound below).

Fitters (chosen by feature type, not by object):

  revolution   axis-of-revolution: jointly fit the axis and a low-order
               polynomial radius profile r(h), so bodies whose radius
               varies with height (teapot belly, tapered mug) are modeled
               exactly rather than forced through a constant-radius
               cylinder. Constant cylinder is the profile_degree=0
               special case. Robust loss + residual trimming tolerate
               off-revolution attachments (handles, spouts) without
               segmentation.
  plane        SVD plane fit -> normal. Flat faces: lids, drawer fronts.
  pca          leading principal component. Elongated features: handle
               bars, spouts.

All fitters return a LINE, not a direction — the residual is invariant
under flipping — so the returned direction is signed toward the coarse
input. If the coarse vector has the wrong sign, refinement faithfully
preserves that error: sign repair is a discrete, one-token semantic
edit and deliberately not this module's job.

Convergence: the revolution fit converges from coarse seeds up to ~60 deg
off the true axis (validated on noisy, partial, cluttered synthetic
clouds); beyond the basin it can lock onto a wrong feature, so snaps
larger than max_snap_deg are REJECTED and the coarse direction returned
with accepted=False — a typed outcome the caller can route, not a
SystemExit.

numpy/scipy only, no simulator dependency — unit-testable in isolation,
and equally applicable to exact mesh samples (sim) and segmented sensor
clouds (deployment).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# Convergence basin of the revolution fit, measured empirically: seeds
# past ~60 deg start capturing wrong features. Kept a constant, not a
# flag — it is part of the module's contract, not a tuning knob.
MAX_SNAP_DEG = 60.0
MIN_POINTS = 50


# ------------------------------------------------------------------ result

@dataclass(frozen=True)
class RefineResult:
    direction: np.ndarray   # unit vector, signed toward `coarse`
    coarse: np.ndarray      # the input, normalized
    method: str             # "revolution" | "plane" | "pca"
    accepted: bool          # False -> direction is just `coarse` passed back
    snap_deg: float         # angle between coarse and fitted axis
    residual_rms: float     # RMS point-to-primitive distance, meters
    sigma_deg: float        # 1-sigma angular uncertainty of the fitted axis
    inliers: int            # points surviving the robust trim
    note: str = ""

    def couple_rot_bound(self, authored_halfwidth_rad: float,
                         k: float = 3.0) -> float:
        """Rotational-bound / estimator-uncertainty coupling: the
        half-width actually used for a Bw rotation row is the authored
        value lower-bounded by k-sigma of the axis estimate. Authored
        bounds tighter than what the frame is known to can never take
        effect; as tracking or refinement sharpens the estimate the
        effective bound tightens back toward the authored one. Returns
        radians (tsr.py's language). A rejected refinement has no trusted
        sigma, so the bound is left untouched."""
        if not self.accepted:
            return authored_halfwidth_rad
        return max(authored_halfwidth_rad, k * np.deg2rad(self.sigma_deg))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero-length direction")
    return v / n


def _basis_perp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing a right-handed frame with a."""
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, h)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


# ------------------------------------------------------------------ fitters
#
# Each fitter returns (axis_unit_unsigned, residual_rms_m, sigma_deg,
# n_inliers). Sign resolution and basin guarding happen in refine_axis.

def fit_revolution_axis(P: np.ndarray, seed: np.ndarray,
                        profile_degree: int = 4,
                        trim_rounds: int = 2) -> tuple[np.ndarray, float,
                                                       float, int]:
    """Axis-of-revolution fit: find the axis about which the points are
    rotationally symmetric, with the radius free to vary along the axis.

    Parametrization (all smooth, one least_squares call per trim round):
      axis   a(u,v) = normalize(a0 + u e1 + v e2)   tangent perturbation
      point  c(s,t) = centroid + s e1 + t e2         in the plane perp a0
      radius r(h)   = Chebyshev polynomial of degree `profile_degree` in
                      the height coordinate h = (P - c) . a, normalized
                      by the seed-frame extent for conditioning
      residual_i = dist(P_i, axis) - r(h_i)

    soft_l1 loss plus MAD trimming between rounds absorbs off-revolution
    geometry (a mug handle is ~10-15% of surface points and lands far
    outside the radial band). sigma_deg comes from the (u, v) block of
    the Gauss-Newton covariance at the solution — for near-unit a0 the
    tangent coefficients ARE small angles, so their standard deviations
    read directly as axis angular uncertainty.
    """
    P = np.asarray(P, dtype=float)
    a0 = _unit(seed)
    e1, e2 = _basis_perp(a0)
    centroid = P.mean(axis=0)
    h_scale = max(float(np.abs((P - centroid) @ a0).max()), 1e-6)

    def unpack(p):
        a = a0 + p[0] * e1 + p[1] * e2
        a = a / np.linalg.norm(a)
        c = centroid + p[2] * e1 + p[3] * e2
        return a, c, p[4:]

    def residuals(p, pts):
        a, c, coef = unpack(p)
        d = pts - c
        h = (d @ a) / h_scale
        radial = d - np.outer(d @ a, a)
        r = np.linalg.norm(radial, axis=1)
        return r - np.polynomial.chebyshev.chebval(np.clip(h, -1.5, 1.5),
                                                   coef)

    pts = P
    r0 = float(np.median(np.linalg.norm(
        (pts - centroid) - np.outer((pts - centroid) @ a0, a0), axis=1)))

    # Round 0 is deliberately RIGID: constant radius (degree 0) under a
    # cauchy loss. With the full profile free from the start, a partial
    # (single-view) shell plus an off-revolution blob gives the
    # polynomial enough freedom to absorb the blob into r(h) and drag
    # the axis; the rigid model cannot bend, so the blob shows up as
    # gross residuals, the axis lands in the true basin, and the trim
    # removes the blob before the profile is released.
    p_rigid = np.zeros(5)
    p_rigid[4] = r0
    sol = least_squares(lambda q, x: residuals(np.r_[q[:4], q[4],
                                                     np.zeros(profile_degree)],
                                               x),
                        p_rigid, args=(pts,), loss="cauchy",
                        f_scale=max(0.05 * r0, 1e-4), max_nfev=200)
    res = residuals(np.r_[sol.x[:4], sol.x[4], np.zeros(profile_degree)],
                    pts)
    mad = float(np.median(np.abs(res - np.median(res)))) + 1e-9
    keep = np.abs(res - np.median(res)) <= 4.0 * 1.4826 * mad
    if keep.sum() >= MIN_POINTS:
        pts = pts[keep]

    p = np.zeros(4 + profile_degree + 1)
    p[:5] = sol.x
    for _ in range(trim_rounds + 1):
        sol = least_squares(residuals, p, args=(pts,), loss="soft_l1",
                            f_scale=max(0.05 * r0, 1e-4), max_nfev=200)
        p = sol.x
        res = residuals(p, pts)
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-9
        keep = np.abs(res) <= 4.0 * 1.4826 * mad
        if keep.sum() < MIN_POINTS or keep.all():
            break
        pts = pts[keep]

    a, _, _ = unpack(sol.x)
    res = residuals(sol.x, pts)
    rms = float(np.sqrt(np.mean(res ** 2)))

    # covariance of the tangent parameters from the final Jacobian
    J = sol.jac
    dof = max(len(res) - len(sol.x), 1)
    s2 = 2.0 * sol.cost / dof
    try:
        cov = s2 * np.linalg.pinv(J.T @ J)
        sigma = float(np.degrees(np.sqrt(max(cov[0, 0] + cov[1, 1], 0.0))))
    except np.linalg.LinAlgError:
        sigma = float("nan")
    return a, rms, sigma, len(pts)


def fit_plane_normal(P: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """SVD plane fit; axis = normal (smallest singular vector). sigma is
    the standard small-noise propagation: normal tilt about an in-plane
    direction has std ~= rms / (sqrt(N) * spread along that direction);
    the reported value is the worse of the two in-plane directions."""
    P = np.asarray(P, dtype=float)
    C = P - P.mean(axis=0)
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    n = Vt[2]
    rms = float(np.sqrt(np.mean((C @ n) ** 2)))
    spread = S[:2] / np.sqrt(len(P))            # in-plane std deviations
    sigma = float(np.degrees(rms / (np.sqrt(len(P)) *
                                    max(spread.min(), 1e-9))))
    return n, rms, sigma, len(P)


def fit_pca_axis(P: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Leading principal component — elongated features (handle bar,
    spout tube). Residual is the RMS off-axis distance."""
    P = np.asarray(P, dtype=float)
    C = P - P.mean(axis=0)
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    a = Vt[0]
    off = C - np.outer(C @ a, a)
    rms = float(np.sqrt(np.mean(np.sum(off ** 2, axis=1))))
    along = S[0] / np.sqrt(len(P))
    sigma = float(np.degrees(rms / (np.sqrt(len(P)) * max(along, 1e-9))))
    return a, rms, sigma, len(P)


FITTERS = {
    "revolution": fit_revolution_axis,
    "plane": lambda P, seed: fit_plane_normal(P),
    "pca": lambda P, seed: fit_pca_axis(P),
}


# --------------------------------------------------------------- public API

def refine_axis(points: np.ndarray, coarse: np.ndarray, feature: str,
                max_snap_deg: float = MAX_SNAP_DEG) -> RefineResult:
    """Snap `coarse` onto the `feature`-type primitive fitted to `points`.

    points   (N,3) — mesh surface samples or a segmented sensor cloud, in
             whatever frame `coarse` is expressed in
    coarse   the semantic layer's direction: initialization + sign, only
    feature  "revolution" | "plane" | "pca"

    The fitted line is signed toward `coarse`. Snaps beyond max_snap_deg
    (outside the validated basin) and degenerate inputs are rejected:
    the coarse direction comes back unchanged with accepted=False so the
    caller can route the failure instead of silently trusting a fit that
    likely locked onto the wrong feature.
    """
    if feature not in FITTERS:
        raise ValueError(f"unknown feature type {feature!r}; "
                         f"choose from {sorted(FITTERS)}")
    c = _unit(coarse)
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(P) < MIN_POINTS:
        return RefineResult(c, c, feature, False, 0.0, float("nan"),
                            float("nan"), len(P),
                            f"only {len(P)} points (< {MIN_POINTS})")

    axis, rms, sigma, n_in = FITTERS[feature](P, c)
    if np.dot(axis, c) < 0.0:                   # sign from semantics
        axis = -axis
    snap = _angle_deg(axis, c)
    if snap > max_snap_deg:
        return RefineResult(c, c, feature, False, snap, rms, sigma, n_in,
                            f"snap {snap:.1f} deg exceeds the "
                            f"{max_snap_deg:.0f} deg basin — coarse "
                            "direction kept")
    return RefineResult(axis, c, feature, True, snap, rms, sigma, n_in)


def snap_to_candidates(coarse: np.ndarray,
                       candidates: np.ndarray) -> tuple[int, np.ndarray,
                                                        float]:
    """Discrete selection: pick the candidate axis (sign-resolved) most
    aligned with `coarse`. This is the selector factorization — a noisy
    scorer choosing among well-separated fitted candidates (e.g. the
    +-3 axes of a fitted frame, 90 deg apart) is robust where the same
    scorer regressing a metric direction is not.

    Returns (index into candidates, signed unit axis, margin) where
    margin is the |cos| gap between the best and second-best candidate —
    small margins flag genuinely ambiguous selections worth escalating.
    """
    c = _unit(coarse)
    A = np.asarray(candidates, dtype=float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    scores = np.abs(A @ c)
    order = np.argsort(scores)[::-1]
    i = int(order[0])
    margin = float(scores[order[0]] - scores[order[1]]) if len(A) > 1 else 1.0
    axis = A[i] if A[i] @ c >= 0 else -A[i]
    return i, axis, margin
