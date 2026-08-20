"""Runtime part grounding: VLM call #1's free-text part names -> 3D part
point sets -> fitted primitives -> the frames.json symbol table every
downstream consumer already reads.

This is the producer frames.py's docstring promised ("the VLM grounding
pipeline later becomes a producer of this same artifact and nothing
downstream changes"), and the one code path sim and hardware share:

    part name ─► 2D part mask per view ─► lift to surface samples ─►
    primitive fit ─► symbols {<part>_center, <part>_axis, ...}

Where the MASKS come from is the provider's business (scripts/
ground_parts.py): an oracle raster of geometric part bands in sim, a
GroundedSAM/SAM3 dump on the same rendered views, or SAM3 on the
registration frame on hardware. Everything after the mask is here, and
it is numpy-only.

Lift. Each dense surface sample is projected into each view with the
SAME analytic pinhole the renders used and depth-tested against the
view's depth buffer (the occlusion test render_candidates.py applies to
marks). A visible sample votes for every part whose mask contains its
pixel; a sample is assigned to the part with the most votes across the
views it is visible in, provided that part won a majority of them. The
lift is exact up to the sample density and mask quality — no depth
back-projection, no single-sample noise: the 3D points were never
estimated from the image, only LABELLED by it.

Fit. One primitive per part, chosen by geometry, never by name:

    ring    points lie near a planar circle (plane rms and circle rms
            both small relative to the radius) — rims, openings, lids.
            axis = plane normal (signed toward up), center = circle
            center, radius.
    line    elongated cloud (leading/second PCA spread ratio above
            ELONGATION) — handle bars, spouts, stems. axis = leading
            component signed AWAY from the body centroid, center = the
            point on the axis at the cloud's median extent, tip = the
            far end, length.
    blob    anything else — lids, bases, knobs. axis = outward
            direction from the body centroid, center = centroid.

Symbol names are `<part>_<kind>` (handle_center, handle_axis, rim_axis,
spout_tip, ...), so they are whatever call #1 said the part was; the
task-specific names the hand-authored sidecars use (pour_axis,
opening_center) do not reappear. Two universal symbols are always
emitted: `up_axis` (the object's rest up, supplied by the caller — body
+z in sim, the measured pose's world-up on hardware) and, for every
line part, `<part>_lateral_axis` = up x axis, the horizontal direction
perpendicular to the part (the tilt axis of a spout, the swing axis of
a handle), because the rotation a task applies about an elongated part
is almost always about that line and it is not otherwise nameable.

Every symbol carries its fit residual and sigma in `comment` and the
primitive in `status`-adjacent metadata so the uncertainty-coupled
bound widening (refine.py's couple_rot_bound) has a per-symbol sigma to
work from instead of a scene-wide guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------- constants
# Properties of the method, not knobs: perturbing them is a sensitivity
# ablation run under a --tag, never a runtime flag.
MIN_PART_POINTS = 30        # below this a part is reported ungrounded
MASK_DILATE_PX = 2          # oracle raster dilation (provider side uses it too)
RING_REL_RMS = 0.08         # circle rms / radius gate for "ring"
RING_PLANE_REL = 0.08       # plane rms / radius gate for "ring"
RING_MIN_ARC = 0.6          # fraction of circle angles covered (>= 60%)
ELONGATION = 2.5            # S0 / S1 spread ratio for "line"
UP_PARALLEL_COS = 0.985     # |up . axis| above this -> no lateral axis


# ------------------------------------------------------------ results

@dataclass
class GroundedPart:
    name: str
    primitive: str                      # "ring" | "line" | "blob" | "ungrounded"
    n_points: int
    center: np.ndarray | None = None
    axis: np.ndarray | None = None      # unit, signed (see module doc)
    tip: np.ndarray | None = None       # line only
    radius: float | None = None         # ring only
    length: float | None = None         # line only
    rms: float = float("nan")           # primitive residual, meters
    sigma_deg: float = float("nan")     # axis direction uncertainty
    note: str = ""

    def summary(self) -> str:
        if self.primitive == "ungrounded":
            return f"{self.name}: UNGROUNDED ({self.note})"
        extra = ""
        if self.radius is not None:
            extra += f" r={self.radius * 1000:.1f}mm"
        if self.length is not None:
            extra += f" L={self.length * 1000:.0f}mm"
        return (f"{self.name}: {self.primitive} n={self.n_points}{extra} "
                f"rms={self.rms * 1000:.1f}mm sigma={self.sigma_deg:.1f}deg")


# --------------------------------------------------------------- lift

def lift_masks(uv_by_view: dict[str, np.ndarray],
               visible_by_view: dict[str, np.ndarray],
               masks_by_view: dict[str, dict[str, np.ndarray]],
               n_samples: int) -> dict[str, np.ndarray]:
    """Label dense surface samples by per-view part masks.

    uv_by_view[v]       (N,2) pixel coords of the samples in view v
    visible_by_view[v]  (N,) depth-test result in view v
    masks_by_view[v]    {part: HxW bool} for view v (parts may be absent)

    Returns {part: (N,) bool} — the sample belongs to the part that won
    a strict majority of its visible views' votes. Samples visible
    nowhere, or with no winning part, belong to no part.
    """
    parts = sorted({p for m in masks_by_view.values() for p in m})
    votes = {p: np.zeros(n_samples, int) for p in parts}
    seen = np.zeros(n_samples, int)
    for v, uv in uv_by_view.items():
        vis = visible_by_view[v]
        if not vis.any() or v not in masks_by_view:
            continue
        pix = uv.round().astype(int)
        seen += vis
        for p, m in masks_by_view[v].items():
            h, w = m.shape
            inb = vis & (pix[:, 0] >= 0) & (pix[:, 0] < w) & \
                (pix[:, 1] >= 0) & (pix[:, 1] < h)
            hit = np.zeros(n_samples, bool)
            hit[inb] = m[pix[inb, 1], pix[inb, 0]]
            votes[p] += hit
    if not parts:
        return {}
    V = np.stack([votes[p] for p in parts], axis=1)        # (N, P)
    best = V.argmax(axis=1)
    top = V[np.arange(n_samples), best]
    win = (top * 2 > seen) & (seen > 0)
    return {p: win & (best == i) for i, p in enumerate(parts)}


def raster_masks(uv: np.ndarray, visible: np.ndarray,
                 labels: dict[str, np.ndarray], h: int, w: int,
                 dilate_px: int = MASK_DILATE_PX) -> dict[str, np.ndarray]:
    """Oracle provider: rasterize labelled visible samples into per-part
    masks (what a segmenter would have produced on this view)."""
    out = {}
    pix = uv.round().astype(int)
    for p, lab in labels.items():
        m = np.zeros((h, w), bool)
        sel = lab & visible & (pix[:, 0] >= 0) & (pix[:, 0] < w) & \
            (pix[:, 1] >= 0) & (pix[:, 1] < h)
        m[pix[sel, 1], pix[sel, 0]] = True
        out[p] = _dilate(m, dilate_px) if dilate_px else m
    return out


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


# ---------------------------------------------------------------- fits

def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def _perp_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(n, a))
    return e1, np.cross(n, e1)


def _plane(P):
    c = P.mean(0)
    C = P - c
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    n = Vt[2]
    rms = float(np.sqrt(np.mean((C @ n) ** 2)))
    spread = S[:2] / np.sqrt(len(P))
    sigma = float(np.degrees(rms / (np.sqrt(len(P)) * max(spread.min(), 1e-9))))
    return c, n, S, rms, sigma


def _circle(P, n, c0):
    e1, e2 = _perp_basis(n)
    xy = np.column_stack([(P - c0) @ e1, (P - c0) @ e2])
    A = np.column_stack([2.0 * xy, np.ones(len(xy))])
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
    cx, cy = sol[0], sol[1]
    R = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12)))
    center = c0 + cx * e1 + cy * e2
    d = P - center
    z = d @ n
    r = np.sqrt(np.maximum((d ** 2).sum(1) - z ** 2, 0.0))
    rms = float(np.sqrt(np.mean((r - R) ** 2 + z ** 2)))
    ang = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    arc = len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 36))) / 36.0
    return center, R, rms, arc


def fit_part(name: str, P: np.ndarray, up: np.ndarray,
             body_center: np.ndarray) -> GroundedPart:
    """One primitive for one part's point set (body coords)."""
    P = np.asarray(P, float)
    up = _unit(up)
    if len(P) < MIN_PART_POINTS:
        return GroundedPart(name, "ungrounded", len(P),
                            note=f"{len(P)} points < {MIN_PART_POINTS}")
    c, n, S, prms, psig = _plane(P)

    # ring: planar + circular + most of the arc present
    center, R, crms, arc = _circle(P, n, c)
    if (prms <= RING_PLANE_REL * R and crms <= RING_REL_RMS * R
            and arc >= RING_MIN_ARC):
        axis = n if n @ up >= 0 else -n
        return GroundedPart(name, "ring", len(P), center=center, axis=axis,
                            radius=R, rms=crms, sigma_deg=psig,
                            note=f"arc {arc:.0%}")

    # line: elongated
    C = P - c
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    if S[0] / max(S[1], 1e-12) >= ELONGATION:
        a = Vt[0]
        if a @ (c - body_center) < 0:
            a = -a
        t = C @ a
        off = C - np.outer(t, a)
        rms = float(np.sqrt(np.mean((off ** 2).sum(1))))
        along = S[0] / np.sqrt(len(P))
        sigma = float(np.degrees(rms / (np.sqrt(len(P)) * max(along, 1e-9))))
        tip = c + a * np.quantile(t, 0.995)
        center_on_axis = c + a * np.median(t)
        return GroundedPart(name, "line", len(P), center=center_on_axis,
                            axis=a, tip=tip, length=float(t.max() - t.min()),
                            rms=rms, sigma_deg=sigma)

    # blob
    a = _unit(c - body_center) if np.linalg.norm(c - body_center) > 1e-6 else up
    rms = float(np.sqrt(np.mean((C ** 2).sum(1))))
    return GroundedPart(name, "blob", len(P), center=c, axis=a, rms=rms,
                        sigma_deg=float("nan"), note="outward axis only")


# ------------------------------------------------------------ symbols

def symbols_from_parts(obj: str, parts: dict[str, GroundedPart],
                       up: np.ndarray, provider: str) -> dict:
    """frames.json document (the same schema load_symbols reads)."""
    up = _unit(up)
    doc = {
        "object": obj, "units": "meters",
        "coordinates": "body frame of mjcf body 'object' (what PoseReader returns)",
        "schema": "grounding symbol table: independent interaction POINTS and "
                  "semantic AXES; frames are composed downstream as "
                  "frame(origin=<point>, axis=<axis>). Axis -> frame +z.",
        "provenance": {"producer": "manip_sim.part_grounding",
                       "mask_provider": provider,
                       "parts": {n: {"primitive": g.primitive, "n_points": g.n_points,
                                     "rms_m": None if np.isnan(g.rms) else round(g.rms, 5),
                                     "sigma_deg": None if np.isnan(g.sigma_deg) else round(g.sigma_deg, 2),
                                     "note": g.note}
                                 for n, g in parts.items()}},
        "points": {}, "axes": {}, "quantities": {},
    }

    def pt(name, xyz, note):
        doc["points"][name] = {"xyz": [round(float(x), 4) for x in xyz],
                               "status": "grounded", "comment": note}

    def ax(name, xyz, note):
        doc["axes"][name] = {"xyz": [round(float(x), 4) for x in _unit(xyz)],
                             "status": "grounded", "comment": note}

    ax("up_axis", up, "object rest up, supplied by the pose source")
    for n, g in parts.items():
        if g.primitive == "ungrounded":
            continue
        note = f"{g.primitive} fit, rms {g.rms * 1000:.1f} mm"
        pt(f"{n}_center", g.center, note)
        ax(f"{n}_axis", g.axis, note + (f", sigma {g.sigma_deg:.1f} deg"
                                        if not np.isnan(g.sigma_deg) else ""))
        if g.primitive == "ring":
            doc["quantities"][f"{n}_radius"] = {"value": round(g.radius, 4),
                                                "status": "grounded",
                                                "comment": note}
        if g.primitive == "line":
            pt(f"{n}_tip", g.tip, "far end of the " + note)
            doc["quantities"][f"{n}_length"] = {"value": round(g.length, 4),
                                                "status": "grounded",
                                                "comment": note}
            if abs(g.axis @ up) < UP_PARALLEL_COS:
                ax(f"{n}_lateral_axis", np.cross(up, g.axis),
                   f"up x {n}_axis: horizontal, perpendicular to the {n}")
    return doc
