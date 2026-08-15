"""Azimuth refinement and frame assembly — compose a refined UP axis and
one azimuth source into a full SO(3) frame with PER-ROW uncertainty.

refine.py refines one axis at a time; a w frame needs three, but they are
not three independent problems. The refined up consumes two rotational DOF
(the roll/pitch tilts), leaving exactly one — the azimuth about up — and
"front" and "left" are both spent the moment that azimuth is chosen:

    z = up_refined
    x = normalize(project(front_source, perp z))     the azimuth
    y = z cross x                                    handedness, free

so there is no such thing as independently refining "left" — fitting it
would over-determine the frame and manufacture an inconsistency to
reconcile. This module therefore refines the AZIMUTH, from exactly one
source, and assembles the frame in the frames.py convention (axis -> +z,
front -> +x, +y = z cross x), so a FrameResult drops into the same
composition path as a hand-authored frames.json frame.

Azimuth routes (explicit caller choice with declared fallback order —
automatic switching would hide which uncertainty regime a frame is in,
and the winning route belongs in the preview-loop evidence):

  constructed   metric vector between grounded points (spout_tip -
                opening_center); sigma from point uncertainty over the
                in-plane lever arm. The teapot's pouring frame lives
                here.
  part_pca      leading component of a segmented part cloud (spout
                tube); sigma from the fit, propagated through the
                projection; gated on elongation so a round blob cannot
                elect an arbitrary direction.
  vector        a pre-calibrated axis symbol (frames.json pour_axis)
                with caller-supplied sigma.
  semantic      Orient Anything / PointSO front, projection only. The
                coarse error passes through essentially intact —
                projection removes the out-of-plane component, not the
                in-plane one — so this route carries semantic-grade
                sigma and exists as the honest fallback, not a
                refinement.

Sign stays with the semantic layer throughout, as in refine.py: pca
lines are signed toward the coarse front; constructed vectors carry
their own sign semantics (tip minus center points AT the spout).

Per-row uncertainty is the point of the exercise. In a w frame whose +z
is the refined up (the only frame this module builds), the Bw rotation
rows split cleanly: roll and pitch tilt z and inherit the UP fit's
sigma; yaw spins about z and inherits the AZIMUTH route's sigma —
which may be three orders of magnitude apart (sub-degree revolution fit
vs 20-deg semantic front). couple_rot_bounds applies
RefineResult.couple_rot_bound's floor rule row-wise instead of smearing
one scalar across the rotation block.

One deliberate asymmetry against refine.py's coupling: a REJECTED up
refinement leaves the roll/pitch bounds authored-as-is (the coarse
direction still means something; there is just no trusted sigma —
refine.py's rule). A rejected/absent azimuth instead forces the yaw row
FREE: with no accepted route the frame's x is an arbitrary
perpendicular, and a tight yaw bound about an arbitrary reference is
not conservative, it is meaningless.

numpy only (scipy enters via refine.py); no simulator dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .refine import (MIN_POINTS, RefineResult, _angle_deg, _basis_perp,
                     _unit)
from .tsr import FREE_ROT

# Projection conditioning gate: the in-plane fraction |proj(source)|
# below which an azimuth source is rejected as near-parallel to up.
# At sin(20 deg) ~= 0.34, source-direction error is already amplified
# ~3x into azimuth error; past that the route is not informative.
# (Load-bearing on real assets: the teapot's handle_axis is ~1 deg off
# vertical — a handle-cloud PCA MUST reject here, not return a garbage
# azimuth with a confident sigma.)
MIN_CONDITIONING = float(np.sin(np.deg2rad(20.0)))
# Elongation gate for the part_pca route: ratio of the first two
# singular values below which the "leading" component is an arbitrary
# direction of a blob rather than the axis of a tube.
MIN_ELONGATION = 2.0
# Default sigma for the semantic route: the coarse-front error regime
# observed from PointSO / Orient Anything (15-45 deg coarse inputs;
# the front is usually the better-behaved of the two labeled axes).
# A constant, not a flag — it is the declared trust level of the
# semantic layer, and moving it per-call would let the coupling be
# quietly defeated.
SEMANTIC_SIGMA_DEG = 20.0


# ---------------------------------------------------------------- results

@dataclass(frozen=True)
class AzimuthResult:
    direction: np.ndarray   # unit, exactly perpendicular to `up`
    up: np.ndarray          # the up this azimuth was projected against
    route: str              # "constructed" | "part_pca" | "vector" | "semantic"
    accepted: bool
    sigma_deg: float        # azimuth (yaw) 1-sigma AFTER projection
    conditioning: float     # in-plane fraction of the source direction
    note: str = ""


@dataclass(frozen=True)
class FrameResult:
    R: np.ndarray           # 3x3, columns [x=front, y=left, z=up]
    up: RefineResult        # carries roll/pitch sigma (or the rejection)
    azimuth: AzimuthResult  # the winning route (or the last rejection)
    accepted: bool          # an azimuth route was accepted; if False, x
                            # is an arbitrary perpendicular and yaw must
                            # be treated as free
    note: str = ""

    def couple_rot_bounds(self, Bw: np.ndarray, k: float = 3.0) -> np.ndarray:
        """Row-wise rotational coupling for a Bw authored in THIS frame
        (w +z = this up; anything else and the row split below is
        wrong). Translation rows pass through untouched. Roll/pitch
        rows are floored by k-sigma of the up fit via
        RefineResult.couple_rot_bound (rejected up -> authored kept,
        its rule). The yaw row is floored by k-sigma of the azimuth
        route; no accepted route, or a floor reaching pi, makes it
        FREE_ROT — see the module docstring for why rejection is FREE
        here but authored-kept for up. Rows already free (span >= 2pi)
        are never touched. Midpoints are preserved: coupling widens a
        bound about the authored center, it never re-centers."""
        Bw = np.array(Bw, dtype=float, copy=True)
        if Bw.shape != (6, 2):
            raise ValueError("Bw must be 6x2")
        two_pi = 2.0 * np.pi

        def widen(row: int, half: float) -> None:
            lo, hi = Bw[row]
            if hi - lo >= two_pi - 1e-9:        # already free
                return
            mid = 0.5 * (lo + hi)
            if not np.isfinite(half) or half >= np.pi:
                Bw[row] = FREE_ROT
                return
            Bw[row] = (mid - half, mid + half)

        for row in (3, 4):                      # roll, pitch: tilt z
            lo, hi = Bw[row]
            if hi - lo >= two_pi - 1e-9:
                continue
            half = self.up.couple_rot_bound(0.5 * (hi - lo), k=k)
            widen(row, half)
        lo, hi = Bw[5]                          # yaw: spin about z
        if hi - lo < two_pi - 1e-9:
            if not self.azimuth.accepted:
                Bw[5] = FREE_ROT
            else:
                half = max(0.5 * (hi - lo),
                           k * np.deg2rad(self.azimuth.sigma_deg))
                widen(5, half)
        return Bw

    def to_frame(self, name: str, origin: np.ndarray):
        """Package as a frames.Frame (origin in the same body coords as
        the fitted directions), so downstream composition — Frame.T(),
        world_T(), the emission DSL's frame() references — is byte-for-
        byte the path a hand-authored frames.json frame takes. Frame
        re-runs Gram-Schmidt; x is already perpendicular to z, so it is
        a fixed point of that construction."""
        from .frames import Frame
        return Frame(name=name, point=np.asarray(origin, float).reshape(3),
                     axis=self.R[:, 2].copy(), secondary=self.R[:, 0].copy(),
                     status="calibrated" if (self.accepted
                                             and self.up.accepted)
                     else "placeholder",
                     comment=f"refine_frame: up={self.up.method}"
                             f"/{'ok' if self.up.accepted else 'rej'}, "
                             f"azimuth={self.azimuth.route}"
                             f"/{'ok' if self.accepted else 'rej'}")


# ----------------------------------------------------------- azimuth routes
#
# Every route does the same three things — project the source direction
# onto the plane perpendicular to up, gate on conditioning, propagate the
# source sigma through the projection (first order: azimuth error =
# source direction error / in-plane fraction) — and differs only in
# where the source and its sigma come from.

def _project(source: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    """In-plane unit direction and conditioning (in-plane fraction) of a
    unit source against a unit up."""
    v = source - float(source @ up) * up
    c = float(np.linalg.norm(v))
    return (v / c if c > 1e-12 else np.zeros(3)), c


def _rejected(up: np.ndarray, route: str, cond: float,
              note: str) -> AzimuthResult:
    return AzimuthResult(np.zeros(3), up, route, False, float("inf"),
                         cond, note)


def azimuth_from_vector(v: np.ndarray, up: np.ndarray, sigma_deg: float,
                        route: str = "vector") -> AzimuthResult:
    """A metric direction that carries its own sign — a calibrated axis
    symbol (pour_axis) or any caller-computed vector — with the
    caller's declared sigma for it. The declared sigma is the trust
    level of wherever the vector came from; this route does not audit
    it, only propagates it through the projection."""
    u = _unit(up)
    d, c = _project(_unit(v), u)
    if c < MIN_CONDITIONING:
        return _rejected(u, route, c,
                         f"source within {np.degrees(np.arcsin(max(c, 0.0))):.0f} deg "
                         f"of up (conditioning {c:.2f} < "
                         f"{MIN_CONDITIONING:.2f}) — azimuth not "
                         "informative")
    return AzimuthResult(d, u, route, True, float(sigma_deg) / c, c)


def azimuth_from_points(p_from: np.ndarray, p_to: np.ndarray,
                        up: np.ndarray,
                        point_sigma_m: float) -> AzimuthResult:
    """The constructed route: azimuth of (p_to - p_from), e.g.
    spout_tip - opening_center. Sigma is real error propagation, which
    is what earns this route the top of the fallback order: each
    endpoint wobbles by point_sigma_m, the difference by sqrt(2) of
    that, and only wobble across the in-plane LEVER ARM turns into
    azimuth — so a 2 mm point error over an 86 mm spout lever is
    ~1.9 deg of yaw, and the yaw row gets a floor commensurate with
    the up fit's instead of the semantic layer's."""
    u = _unit(up)
    v = np.asarray(p_to, float).reshape(3) - np.asarray(p_from,
                                                        float).reshape(3)
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return _rejected(u, "constructed", 0.0, "coincident points")
    d, c = _project(v / L, u)
    lever = c * L
    if c < MIN_CONDITIONING:
        return _rejected(u, "constructed", c,
                         f"point pair near-vertical (in-plane lever "
                         f"{lever * 1000:.0f} mm of {L * 1000:.0f} mm "
                         f"total) — azimuth not informative")
    sigma = float(np.degrees(np.arctan2(np.sqrt(2.0) * point_sigma_m,
                                        lever)))
    return AzimuthResult(d, u, "constructed", True, sigma, c)


def azimuth_from_part_cloud(points: np.ndarray, up: np.ndarray,
                            coarse_front: np.ndarray) -> AzimuthResult:
    """Leading principal component of a segmented part cloud (spout
    tube, horizontal handle bar), signed toward the coarse front,
    projected. Inlines the PCA instead of calling fit_pca_axis because
    the gate needs the singular values: a non-elongated blob has a
    perfectly computable but MEANINGLESS leading component, and its
    small-noise sigma formula would not flag it. Elongation and
    conditioning both gate — a vertical tube (the teapot's handle) is
    elongated AND uninformative about azimuth."""
    u = _unit(up)
    cf = _unit(coarse_front)
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(P) < MIN_POINTS:
        return _rejected(u, "part_pca", 0.0,
                         f"only {len(P)} points (< {MIN_POINTS})")
    C = P - P.mean(axis=0)
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    elong = float(S[0] / max(S[1], 1e-12))
    if elong < MIN_ELONGATION:
        return _rejected(u, "part_pca", float("nan"),
                         f"part not elongated (s0/s1 {elong:.2f} < "
                         f"{MIN_ELONGATION:.1f}) — leading component is "
                         "arbitrary")
    a = Vt[0] if Vt[0] @ cf >= 0 else -Vt[0]   # sign from semantics
    off = C - np.outer(C @ a, a)
    rms = float(np.sqrt(np.mean(np.sum(off ** 2, axis=1))))
    along = S[0] / np.sqrt(len(P))
    sigma_axis = float(np.degrees(rms / (np.sqrt(len(P))
                                         * max(along, 1e-9))))
    d, c = _project(a, u)
    if c < MIN_CONDITIONING:
        return _rejected(u, "part_pca", c,
                         f"part axis within "
                         f"{np.degrees(np.arcsin(max(c, 0.0))):.0f} deg of "
                         f"up (conditioning {c:.2f}) — azimuth not "
                         "informative")
    return AzimuthResult(d, u, "part_pca", True, sigma_axis / c, c,
                         f"elongation {elong:.1f}, {len(P)} pts")


def azimuth_from_semantic(front: np.ndarray, up: np.ndarray,
                          sigma_deg: float = SEMANTIC_SIGMA_DEG
                          ) -> AzimuthResult:
    """Orient Anything / PointSO front, projection only — the honest
    fallback. Projection removes the out-of-plane component of the
    coarse error and nothing else; the in-plane component IS the
    azimuth error, so the semantic sigma passes through (divided by
    conditioning, same first-order geometry as every route). Use this
    when no metric azimuth source exists for the object, and author
    the yaw bound loose — the looseness is the truth."""
    return azimuth_from_vector(front, up, sigma_deg, route="semantic")


# ---------------------------------------------------------------- assembly

def assemble_frame(up: RefineResult,
                   candidates: list[AzimuthResult]) -> FrameResult:
    """First-accepted-wins over an explicit, caller-ordered candidate
    list (constructed -> part_pca -> semantic is the expected order for
    the tea task). Rejected candidates ahead of the winner are recorded
    in the note so the preview loop sees WHY the frame is in the
    uncertainty regime it is in, not just which. If nothing is
    accepted, the frame is still returned — x an arbitrary (but
    deterministic) perpendicular, accepted=False — because downstream
    still needs SOME frame to express a yaw-free Bw in; the FREE yaw
    from couple_rot_bounds is what makes that frame safe to use.

    Every candidate must have been projected against THIS up (the
    routes take `up` precisely so this cannot silently drift); a
    mismatch is a programming error and raises."""
    z = _unit(up.direction)
    skipped: list[str] = []
    for cand in candidates:
        if _angle_deg(cand.up, z) > 1e-3:
            raise ValueError(
                f"azimuth candidate ({cand.route}) was projected against "
                "a different up than the frame is being assembled with")
        if cand.accepted:
            x = cand.direction
            R = np.column_stack([x, np.cross(z, x), z])
            note = ("; ".join(f"{s}" for s in skipped) + "; " if skipped
                    else "") + f"azimuth via {cand.route}"
            return FrameResult(R, up, cand, True, note)
        skipped.append(f"{cand.route} rejected ({cand.note})")
    e1, _ = _basis_perp(z)                      # deterministic fallback x
    R = np.column_stack([e1, np.cross(z, e1), z])
    last = candidates[-1] if candidates else _rejected(
        z, "none", 0.0, "no azimuth candidates supplied")
    return FrameResult(R, up, last, False,
                       "no azimuth route accepted: "
                       + ("; ".join(skipped) if skipped
                          else "empty candidate list")
                       + " — x is arbitrary, yaw must be free")