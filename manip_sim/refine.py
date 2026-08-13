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

Convergence and identifiability: for well-conditioned revolution bodies
(cylindrical or profiled, attachments a minority of surface) the fit
converges from coarse seeds up to ~60 deg off the true axis. For
DEGENERATE bodies — near-spherical bellies that are symmetric about
every axis through their center, so the data alone does not determine
the axis — the multistart + seed-prior machinery keeps the module
honest through ~40 deg of coarse error: outcomes are correct
acceptances or typed rejections (competing self-consistent modes
surfaced as sigma), never confident coin-flips. Snaps larger than
max_snap_deg and weakly identifiable axes are REJECTED and the coarse
direction returned with accepted=False — typed outcomes the caller can
route, not a SystemExit. When a revolution fit rejects on
identifiability, the discriminating geometry is usually the opening/rim
circle: fit that (Kasa, as calibrate_frames_from_mesh already does for
the mug) on the segmented opening region instead.

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
# Data-only axis sigma beyond which a fit is considered degenerate: the
# geometry did not determine the axis (near-spherical body, tiny
# support) and the returned direction owes too much to the seed prior
# to be trusted as a refinement.
MAX_SIGMA_DEG = 10.0
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

def _band_gate(P: np.ndarray, axis: np.ndarray, center: np.ndarray,
               nbins: int = 12, k: float = 3.5) -> np.ndarray:
    """Revolution-consistency gate: keep points whose radius is typical
    for their height band about `axis`. A spout or handle is a gross
    radial outlier WITHIN its band even under a basin-level (~25-45 deg)
    axis error, while a legitimate radius profile (belly, shoulder) is
    the band structure itself — so this removes attachments that global
    MAD trimming cannot see (their residuals inflate the global MAD that
    is supposed to catch them)."""
    d = P - center
    h = d @ axis
    radial = np.linalg.norm(d - np.outer(h, axis), axis=1)
    span = float(h.max() - h.min()) + 1e-9
    bins = np.clip(((h - h.min()) / span * nbins).astype(int), 0, nbins - 1)
    keep = np.zeros(len(P), bool)
    for b in range(nbins):
        m = bins == b
        if m.sum() < 10:
            keep[m] = True
            continue
        med = float(np.median(radial[m]))
        mad = float(np.median(np.abs(radial[m] - med))) + 1e-9
        keep[m] = np.abs(radial[m] - med) <= k * 1.4826 * mad
    return keep


def fit_revolution_axis(P: np.ndarray, seed: np.ndarray,
                        profile_degree: int = 4,
                        trim_rounds: int = 2) -> tuple[np.ndarray, float,
                                                       float, int]:
    """Multistart wrapper around _fit_revolution_once. Cluttered bodies
    with weak identifiability (near-spherical belly + coplanar
    attachments) have MULTIPLE local minima, and which one captures the
    solver depends on where inside the basin the seed happens to sit —
    a 25-deg coarse can land true while a 40-deg coarse from the same
    distribution locks onto the attachment-elected axis with a small,
    confidently-wrong sigma. So: run the fit from the seed and from
    eight tangent-jittered starts, cluster the solutions, and prefer
    the cluster angularly closest to the seed — the mode-level analogue
    of the in-fit prior (the semantic direction arbitrates exactly what
    the data leaves ambiguous). When the data decisively prefers a
    DIFFERENT mode than the prior, the disagreement is surfaced as
    sigma and rejected upstream rather than silently resolved either
    way. Unimodal problems pay only the constant factor: all starts
    land in one cluster and the answer is unchanged."""
    a0 = _unit(seed)
    e1, e2 = _basis_perp(a0)
    # eight jitter azimuths at 20 deg: enough that from a basin-level
    # coarse some start lands inside the true attractor for any azimuth
    # of the true direction. A second, wider ring was tried and removed:
    # past ~45 deg on degenerate bodies an INTERMEDIATE self-consistent
    # mode can sit nearer the coarse than the truth, and under prior
    # arbitration that mode wins no matter how many starts find the
    # true one — wider coverage only added cost and conservative
    # rejections inside the trusted regime.
    j = np.deg2rad(20.0)
    starts = [a0]
    for az in np.arange(8) * (np.pi / 4):
        d = np.cos(az) * e1 + np.sin(az) * e2
        s = a0 * np.cos(j) + d * np.sin(j)
        starts.append(s / np.linalg.norm(s))
    sols = [_fit_revolution_once(P, s, profile_degree, trim_rounds)
            for s in starts]
    # sign everything toward the seed, then greedy 10-deg clustering
    axes = [(a if a @ a0 >= 0 else -a, rms, sig, nin)
            for a, rms, sig, nin in sols]
    clusters: list[list[tuple]] = []
    for s in axes:
        for cl in clusters:
            if _angle_deg(s[0], cl[0][0]) < 10.0:
                cl.append(s)
                break
        else:
            clusters.append([s])
    # Mode arbitration. A hard fact discovered on the degenerate class:
    # competing modes can each be SELF-CONSISTENT (the gate carves the
    # cloud into agreement with whichever axis is being fit — a
    # near-spherical body supports any axis at noise-level rms, and the
    # attachment-elected mode even explains MORE geometry), so no
    # data-side score can arbitrate between well-separated modes. The
    # arbitration information is exactly the semantic prior. Policy:
    # drop junk clusters (rms > 2x the best), pick the nearest
    # survivor to the seed, and if the runner-up survivor is nearly as
    # close (< 15 deg gap) the prior cannot arbitrate either — report
    # the inter-mode angle as sigma so refine_axis's identifiability
    # gate turns it into a typed rejection instead of a coin-flip.
    def cl_axis(cl):
        return _unit(np.mean([m[0] for m in cl], axis=0))

    def cl_rms(cl):
        return min(m[1] for m in cl)

    best_rms = min(cl_rms(cl) for cl in clusters)
    survivors = [cl for cl in clusters if cl_rms(cl) <= 2.0 * best_rms]
    survivors.sort(key=lambda cl: _angle_deg(cl_axis(cl), a0))
    best_cl = survivors[0]
    a, rms, sig, nin = min(best_cl, key=lambda m: m[1])
    if len(survivors) > 1:
        d0 = _angle_deg(cl_axis(survivors[0]), a0)
        d1 = _angle_deg(cl_axis(survivors[1]), a0)
        if d1 - d0 < 15.0:
            sig = max(sig, _angle_deg(a, cl_axis(survivors[1])))
    return a, rms, sig, nin


def _fit_revolution_once(P: np.ndarray, seed: np.ndarray,
                         profile_degree: int = 4,
                         trim_rounds: int = 2) -> tuple[np.ndarray, float,
                                                        float, int]:
    """Axis-of-revolution fit: find the axis about which the points are
    rotationally symmetric, with the radius free to vary along the axis.

    Parametrization (all smooth, one least_squares call per round):
      axis   a(u,v) = normalize(a0 + u e1 + v e2)   tangent perturbation
      point  c(s,t) = centroid + s e1 + t e2         in the plane perp a0
      radius r(h)   = Chebyshev polynomial of degree `profile_degree` in
                      the height coordinate h = (P - c) . a, normalized
                      by the seed-frame extent for conditioning
      residual_i = dist(P_i, axis) - r(h_i)

    Three defenses against the failure modes real tableware exhibits:

      band gate    attachments (spout, handle) covering 30%+ of the
                   surface make their own residuals invisible to global
                   MAD trimming; the height-banded gate (seed frame,
                   re-run for re-admission in the fitted frame after the
                   rigid round) removes them by within-band radial
                   consistency instead.
      rigid round  the first solve pins the profile to a constant radius
                   under a cauchy loss, so a partial shell cannot bend
                   r(h) around clutter before the axis lands in the
                   true basin.
      seed prior   a NEAR-SPHERICAL body is symmetric about every axis
                   through its center: the data then does not determine
                   the axis at all and the residual attachments elect a
                   wrong one. Two weak Tikhonov rows on (u, v) resolve
                   exactly the directions the data leaves free toward
                   the semantic seed; in well-conditioned fits their
                   pull is equivalent to sub-mm residual and vanishes in
                   the data gradient.

    The reported sigma comes from the DATA-ONLY block of the final
    Gauss-Newton covariance (prior rows excluded) — degeneracy shows up
    as a large sigma instead of being masked by the prior, and
    refine_axis rejects on it.
    """
    P = np.asarray(P, dtype=float)
    a0 = _unit(seed)
    e1, e2 = _basis_perp(a0)
    centroid = P.mean(axis=0)
    h_scale = max(float(np.abs((P - centroid) @ a0).max()), 1e-6)
    # prior weight: tilting a full basin (~1 rad) off the seed costs as
    # much as ~0.7 mm rms over all points — decisive in a flat valley,
    # noise-level against any real curvature gradient. The prior is
    # split across many rows so each stays within the robust loss's
    # quadratic regime: a single row of magnitude ~20x f_scale would be
    # downweighted as an "outlier" by the very machinery meant to
    # suppress clutter, silently disabling the prior exactly when the
    # solver drifts (total quadratic stiffness is row-count invariant:
    # k rows of (w/sqrt(k)) u cost w^2 u^2).
    n_prior = 64
    w_prior = 0.7e-3 * np.sqrt(len(P)) / np.sqrt(n_prior)

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
        data = r - np.polynomial.chebyshev.chebval(np.clip(h, -1.5, 1.5),
                                                   coef)
        prior = np.tile([w_prior * p[0], w_prior * p[1]], n_prior)
        return np.concatenate([data, prior])

    def data_stats(p, pts):
        res = residuals(p, pts)[:-2 * n_prior]
        return res, float(np.sqrt(np.mean(res ** 2)))

    # seed-frame gate, then the rigid constant-radius round
    keep = _band_gate(P, a0, centroid)
    pts = P[keep] if keep.sum() >= MIN_POINTS else P
    r0 = float(np.median(np.linalg.norm(
        (pts - centroid) - np.outer((pts - centroid) @ a0, a0), axis=1)))
    p_rigid = np.zeros(5)
    p_rigid[4] = r0
    zpad = np.zeros(profile_degree)
    sol = least_squares(lambda q, x: residuals(np.r_[q, zpad], x),
                        p_rigid, args=(pts,), loss="cauchy",
                        f_scale=max(0.05 * r0, 1e-4), max_nfev=200)
    a_rigid, c_rigid, _ = unpack(np.r_[sol.x, zpad])

    # re-admission: re-gate ALL points in the fitted frame, so body
    # points mis-gated under the tilted seed come back before the
    # profile is released
    keep = _band_gate(P, a_rigid, c_rigid)
    pts = P[keep] if keep.sum() >= MIN_POINTS else P

    p = np.zeros(4 + profile_degree + 1)
    p[:5] = sol.x
    for _ in range(trim_rounds + 1):
        sol = least_squares(residuals, p, args=(pts,), loss="soft_l1",
                            f_scale=max(0.05 * r0, 1e-4), max_nfev=200)
        p = sol.x
        res, _ = data_stats(p, pts)
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-9
        keep = np.abs(res) <= 4.0 * 1.4826 * mad
        if keep.sum() < MIN_POINTS or keep.all():
            break
        pts = pts[keep]

    a, _, _ = unpack(sol.x)
    res, rms = data_stats(sol.x, pts)

    # covariance of the tangent parameters from the DATA rows of the
    # final Jacobian — the prior rows would clamp exactly the
    # degeneracy the sigma is supposed to expose
    J = sol.jac[:-2 * n_prior]
    dof = max(len(res) - len(sol.x), 1)
    s2 = float(np.sum(res ** 2)) / dof
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
    if np.isfinite(sigma) and sigma > MAX_SIGMA_DEG:
        return RefineResult(c, c, feature, False, snap, rms, sigma, n_in,
                            f"axis weakly identifiable (sigma "
                            f"{sigma:.1f} deg > {MAX_SIGMA_DEG:.0f}) — "
                            "the geometry does not determine this axis; "
                            "coarse direction kept")
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
