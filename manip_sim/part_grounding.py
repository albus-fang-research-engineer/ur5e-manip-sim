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
opening_center) do not reappear. Alongside the part fits every object
carries its CANONICAL FRAME (vlm.FRAME_AXES), caller-supplied, never
fitted: `up_axis` (rest up — body +z in sim, the measured pose's
world-up on hardware), and when the caller has a confident front
(scene facing in sim, Orient Anything on hardware) `front_axis` and
`lateral_axis` = up x front. The grounding compiler roots every stage's
w frame on the passive object's canonical frame, so these are the axes
a task's rotations are expressed about; the part axes remain for call
#2's frame selection and as relation-row subjects.

Every axis symbol carries its estimator sigma as a structured
`sigma_deg` field (fit sigma for line/ring parts, the caller's declared
sigma for the canonical frame, absent for blobs) so the compiler's
uncertainty coupling has a per-symbol sigma to work from instead of a
scene-wide guess.

Depth semantics. Every grounded POINT symbol also carries `depth_check`
("interior" | "skip") and the raw `winding` value it was decided from
(annotate_depth_checks, called by the driver that holds the mesh): is the
point inside the object's material? Decided by geometry -- the generalized
winding number of the point w.r.t. the mesh -- never by primitive or name,
because the primitive does not determine it: a ring's center is free space
(rim opening, handle hole), a blob's centroid may sit in an opening, and a
LINE fit's center lies on the PCA axis through the cloud's centroid, which
for a curved strap is off the material (a 180-deg C-handle's center sits
~8 mm inside the curve, outside a 6 mm tube). A straight spout's center is
interior. "surface" is never emitted: grounded points are centers and tips,
not surface samples; it stays the default for hand-authored points.
Consumers (hardware check_symbols) gate `interior` on depth and `skip` on
image placement only.
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
UP_PARALLEL_COS = 0.985     # |up . front| above this -> no usable front
WINDING_INSIDE = 0.5        # |generalized winding number| >= this -> inside material


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


# ------------------------------------------------------------ depth semantics

def winding_number(points: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Generalized winding number of each point w.r.t. the triangle mesh
    (Jacobson et al. 2013): the signed solid angle all faces subtend,
    over 4 pi, each face's solid angle by Van Oosterom & Strackee. ~1
    inside a closed outward-oriented mesh, ~0 outside, saturating within a
    few mm of the surface; small holes and cracks (reconstructed meshes)
    move it continuously instead of flipping it, which is why it is used
    rather than ray parity. Exactly ON a face the value is undefined: that
    face subtends +-2 pi, the sign set by a floating-point zero, so it
    reads 0 or 1. Sign follows face orientation (inward-wound meshes read
    -1 inside); callers compare |w|. O(len(points) x len(F)), numpy only."""
    P = np.atleast_2d(np.asarray(points, float))
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    out = np.empty(len(P))
    for i, p in enumerate(P):
        a, b, c = V[F[:, 0]] - p, V[F[:, 1]] - p, V[F[:, 2]] - p
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
               + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb)
        out[i] = float(np.sum(2.0 * np.arctan2(num, den)) / (4.0 * np.pi))
    return out


def mesh_orientation(V: np.ndarray, F: np.ndarray) -> dict:
    """Face-orientation health of a triangle mesh, from its directed edges,
    computed on the surface's GEOMETRY: vertices are first merged by exact
    position. UV-atlased meshes (TRELLIS -> Any6D) split vertices along
    texture seams, so by index the surface is cut into disk-shaped charts:
    every seam edge counts as a boundary, and a whole chart wound the wrong
    way would be invisible to an index-based count. Measured on the mug's
    final_mesh (20260928_190618): by index 24117 boundary edges and an
    Euler characteristic of ~236 (one disk per chart); merged, 94198 of
    106500 vertices, 0 inconsistent, 3 boundary, 0 non-manifold edges. The
    seam duplicates are bit-identical copies (same source vertex, same
    transforms, same text), so exact matching needs no tolerance.

    The winding number is only meaningful when faces are consistently
    oriented: then every directed edge occurs at most once (a neighbour
    traverses it the other way). `inconsistent_edges` counts directed edges
    occurring more than once. That count does not separate two causes:
    `flipped_edges` counts edges with exactly two faces traversing them the
    SAME way -- a face wound against its neighbour, which is what makes depth
    labels unreliable near it; `pinch_edges` counts edges with more than two
    faces whose traversals balance (as many each way) -- consistently wound
    sheets touching along an edge, which TRELLIS meshes do where thin parts
    meet (measured on the 20261023_032004 mug: 272 four-face pinch edges, all
    balanced, = all 544 inconsistent directed edges; 0 flipped). The winding
    number sums solid angles over consistently oriented faces, so a pinch
    leaves it exact; consumers gate on `flipped_edges`. `boundary_edges` counts undirected edges with a single
    face (holes, open rims): the winding number degrades gracefully near
    them, so they are recorded, not judged. `non_manifold_edges` (three or
    more faces on one edge) is recorded likewise. Faces that collapse
    under the merge are dropped from the counts. winding_number itself
    needs no merge: it sums solid angles and never reads connectivity."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    _, inv = np.unique(V, axis=0, return_inverse=True)
    Fw = np.asarray(inv).reshape(-1)[F]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    _, directed = np.unique(E, axis=0, return_counts=True)
    _, ui, undirected = np.unique(np.sort(E, axis=1), axis=0, return_inverse=True,
                                  return_counts=True)
    fwd = np.bincount(np.asarray(ui).reshape(-1), weights=(E[:, 0] < E[:, 1]),
                      minlength=len(undirected))        # traversals low -> high index
    return {"inconsistent_edges": int((directed > 1).sum()),
            "flipped_edges": int(((undirected == 2) & (fwd != 1)).sum()),
            "pinch_edges": int(((undirected > 2) & (2 * fwd == undirected)).sum()),
            "boundary_edges": int((undirected == 1).sum()),
            "non_manifold_edges": int((undirected > 2).sum()),
            "merged_vertices": int(np.asarray(inv).max() + 1) if len(V) else 0,
            "vertices": int(len(V))}


def annotate_depth_checks(frames: dict, V: np.ndarray, F: np.ndarray) -> dict:
    """Add `depth_check` + `winding` to every point of a frames.json
    document (in place; returned for chaining), from the mesh the points
    are expressed in. inside (|w| >= WINDING_INSIDE) -> "interior",
    outside -> "skip".

    The label is only arbitrary for points AT the surface (the winding
    value is undefined there, see winding_number), and there either label
    is benign by construction: such a point passes a depth gate that
    accepts a few mm either side of the surface (hardware check_symbols:
    interior band [-5, +30] mm in front-surface depth) and passes a
    placement-only `skip` trivially. Off the surface the value saturates
    to ~1 / ~0 within a few mm, so the label is only ever arbitrary where
    it does not matter.

    Applied after symbols_from_parts rather than inside it: the point
    names and positions are created there, symbols_from_parts stays
    mesh-free, and nothing in the document is derived from depth_check.

    The mesh's orientation health goes into provenance.depth_semantics:
    with inconsistent_edges > 0 the labels are unreliable, and a consumer
    that gates on them (the hardware seam) should refuse rather than trust
    them. Labels are still written, so a sim run is never blocked by a
    diagnostic."""
    pts = frames.get("points", {})
    names = list(pts)
    frames.setdefault("provenance", {})["depth_semantics"] = {
        "method": "generalized winding number w.r.t. the grounding mesh",
        "inside_threshold": WINDING_INSIDE, **mesh_orientation(V, F)}
    if not names:
        return frames
    w = winding_number(np.array([pts[n]["xyz"] for n in names], float), V, F)
    for n, wi in zip(names, w):
        pts[n]["depth_check"] = "interior" if abs(wi) >= WINDING_INSIDE else "skip"
        pts[n]["winding"] = round(float(wi), 3)
    return frames


# ------------------------------------------------------------ symbols

def symbols_from_parts(obj: str, parts: dict[str, GroundedPart],
                       up: np.ndarray, provider: str,
                       front: np.ndarray | None = None,
                       up_sigma_deg: float = 0.0,
                       front_sigma_deg: float = 0.0) -> dict:
    """frames.json document (the same schema load_symbols reads). `up` /
    `front` are the caller's canonical frame (body coords; front is
    orthogonalized against up and dropped if near-parallel) with the
    caller's declared sigmas."""
    up = _unit(up)
    doc = {
        "object": obj, "units": "meters",
        "coordinates": "body frame of mjcf body 'object' (what PoseReader returns)",
        "schema": "grounding symbol table: independent interaction POINTS and "
                  "semantic AXES; frames are composed downstream as "
                  "frame(origin=<point>, axis=<axis>). Axis -> frame +z. "
                  "up_axis/front_axis/lateral_axis are the canonical frame.",
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

    def ax(name, xyz, note, sigma=None):
        doc["axes"][name] = {"xyz": [round(float(x), 4) for x in _unit(xyz)],
                             "status": "grounded", "comment": note,
                             "sigma_deg": None if sigma is None or np.isnan(sigma)
                             else round(float(sigma), 2)}

    ax("up_axis", up, "canonical frame: rest up, supplied by the pose source",
       up_sigma_deg)
    if front is not None:
        f = _unit(front)
        if abs(f @ up) < UP_PARALLEL_COS:
            f = _unit(f - (f @ up) * up)
            ax("front_axis", f, "canonical frame: front, supplied by the caller "
               "(scene facing / Orient Anything)", front_sigma_deg)
            ax("lateral_axis", np.cross(up, f), "canonical frame: up x front",
               float(np.hypot(up_sigma_deg, front_sigma_deg)))
    for n, g in parts.items():
        if g.primitive == "ungrounded":
            continue
        note = f"{g.primitive} fit, rms {g.rms * 1000:.1f} mm"
        pt(f"{n}_center", g.center, note)
        ax(f"{n}_axis", g.axis, note + (f", sigma {g.sigma_deg:.1f} deg"
                                        if not np.isnan(g.sigma_deg) else ""),
           g.sigma_deg)
        if g.primitive == "ring":
            doc["quantities"][f"{n}_radius"] = {"value": round(g.radius, 4),
                                                "status": "grounded",
                                                "comment": note}
        if g.primitive == "line":
            pt(f"{n}_tip", g.tip, "far end of the " + note)
            doc["quantities"][f"{n}_length"] = {"value": round(g.length, 4),
                                                "status": "grounded",
                                                "comment": note}
    return doc
