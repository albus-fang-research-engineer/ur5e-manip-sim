"""Interaction point candidate proposal — the 3D-propose half of the
3D-propose / 2D-select grounding pipeline.

Generates the candidate interaction point POOL on an object mesh, in body
coordinates, as the union of four candidate classes (per the goal-pose
architecture plan):

    fps          farthest point sampling over the surface — the COVERAGE
                 generator: guarantees no large surface patch is
                 unrepresented, task-agnostic by design
    curvature    saliency-driven samples concentrated on high-curvature
                 geometry (rims, spout lips, handle bars, edges) — where
                 contact-relevant features live; saliency is the absolute
                 angle defect (discrete Gaussian curvature) of the mesh
    part         per-part quota sampling inside geometrically derived part
                 regions (spout / handle / lid / rim / base bands) — the
                 stand-in for GroundedSAM part lifting in the sim-native
                 arm; guarantees every semantically nameable part is
                 represented regardless of its area or curvature
    constructed  points DERIVED from fitted primitives rather than sampled
                 from the surface — opening centers, cavity-axis points,
                 spout tip, handle mid-grip. Surface sampling is
                 structurally incapable of producing these (the center of
                 a mug opening is not ON the mug); cf. OmniManip's
                 tangible/intangible taxonomy and MOKA's mask-center
                 augmentation. Sourced from the calibrated frames.json
                 symbols where they exist (single source of truth; do not
                 refit what calibrate_frames_from_mesh.py already fit).

The four classes are UNIONED (never intersected — FPS and curvature
optimize contradictory objectives, their intersection is near-empty by
construction), then deduplicated by greedy 3D NMS with a class-priority
ordering (constructed > part > curvature > fps) so that when a sampled
point and a semantically grounded point are behaviorally equivalent, the
grounded one survives and carries its provenance. The NMS radius is tied
to the downstream subgoal-TSR tolerance: candidates the planner cannot
distinguish should be merged before the VLM is ever asked to distinguish
them.

A per-part coverage guarantee runs after NMS: every discovered part region
must retain at least one candidate, else its best candidate is re-admitted
(logged). The per-VIEW legibility invariant (occlusion test, screen-space
NMS, >=1-view coverage) is the marked-render generator's job, not this
module's — this module owns the 3D pool only.

Coordinates: identical convention to frames.json / PoseReader — the body
frame of mjcf body 'object'. Mesh vertices ARE body coordinates.

No dependencies beyond numpy: OBJ faces are parsed directly (the converter
writes plain triangle/polygon OBJ) and angle defects are computed inline,
keeping this importable in both docker wings without trimesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# tunables — constants, not flags: these are properties of the method under
# ablation, and silently varying them between runs would unblind comparisons.
# The pool budget targets the 30-50 range from the plan; NMS radius matches
# the ~1 cm translational tolerance of the subgoal TSRs.
# ---------------------------------------------------------------------------
DENSE_SAMPLES = 20000        # area-weighted surface samples backing all classes
FPS_QUOTA = 16               # coverage class size
CURVATURE_QUOTA = 12         # saliency class size
PART_QUOTA_EACH = 8          # per part region — dense enough that the menu's
                             # per-part FPS spread (selection.vlm_subset tier 1)
                             # has material to spread; 4 over a full handle band
                             # was sparse before NMS
CURVATURE_QUANTILE = 0.90    # samples above this saliency quantile are eligible
POOL_BUDGET = 48             # hard cap on the union pool (30-50 per plan)
NMS_RADIUS_M = 0.010         # merge radius, tied to subgoal B^w tolerance
SEED = 0                     # deterministic pool: same mesh -> same candidates


# ---------------------------------------------------------------------------
# mesh loading + surface sampling
# ---------------------------------------------------------------------------

def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices AND triangle faces of a wavefront OBJ. Polygonal faces are
    fan-triangulated; v/vt/vn index forms are handled; indices may be
    negative (OBJ allows relative indexing)."""
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(vs) + i)
                for k in range(1, len(idx) - 1):        # fan triangulation
                    fs.append([idx[0], idx[k], idx[k + 1]])
    if not vs:
        raise SystemExit(f"[proposal] no vertices in {path}")
    if not fs:
        raise SystemExit(f"[proposal] no faces in {path} — curvature and "
                         "surface sampling need connectivity")
    return np.asarray(vs, dtype=float), np.asarray(fs, dtype=int)


def surface_samples(V: np.ndarray, F: np.ndarray, n: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n area-weighted random points on the surface; returns (points,
    face_index_per_point). Area weighting keeps sampling density uniform in
    surface measure, so FPS/curvature classes see the mesh, not its
    tessellation."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        raise SystemExit("[proposal] degenerate mesh: zero total face area")
    fi = rng.choice(len(F), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    P = a[fi] + u[:, None] * (b[fi] - a[fi]) + v[:, None] * (c[fi] - a[fi])
    return P, fi


# ---------------------------------------------------------------------------
# curvature saliency: absolute angle defect (discrete Gaussian curvature)
# ---------------------------------------------------------------------------

def vertex_defects(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """|2*pi - sum of incident face angles| per vertex, normalized by the
    vertex's barycentric area (one third of incident face area) — i.e. a
    discrete GAUSSIAN CURVATURE DENSITY, so saliency is a property of the
    shape rather than the tessellation (raw defect scales with vertex
    spacing and would rank dense patches over sharp ones). High on rims,
    creases, tips, handle bars; ~0 on flats and clean cylinders. Open
    boundaries (mug rim edge) read as high defect — desirable here, since
    boundary loops ARE contact-salient features. Clipped at the 99th
    percentile so degenerate slivers cannot own the quantile."""
    angsum = np.zeros(len(V))
    area = np.zeros(len(V))
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    for k in range(3):
        p = V[F[:, k]]
        q = V[F[:, (k + 1) % 3]] - p
        r = V[F[:, (k + 2) % 3]] - p
        qn = np.linalg.norm(q, axis=1) + 1e-12
        rn = np.linalg.norm(r, axis=1) + 1e-12
        ang = np.arccos(np.clip((q * r).sum(1) / (qn * rn), -1.0, 1.0))
        np.add.at(angsum, F[:, k], ang)
        np.add.at(area, F[:, k], fa / 3.0)
    d = np.abs(2.0 * np.pi - angsum) / np.maximum(area, 1e-10)
    return np.minimum(d, np.quantile(d, 0.99))


# ---------------------------------------------------------------------------
# farthest point sampling
# ---------------------------------------------------------------------------

def fps(P: np.ndarray, k: int, start: int = 0) -> np.ndarray:
    """Indices of k farthest-point samples from P (deterministic given
    start). Standard O(n*k) incremental min-distance update."""
    k = min(k, len(P))
    if k == 0:
        return np.empty(0, dtype=int)
    chosen = [start]
    dmin = np.linalg.norm(P - P[start], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(P - P[i], axis=1))
    return np.asarray(chosen, dtype=int)


# ---------------------------------------------------------------------------
# part regions — geometric band logic mirroring calibrate_frames_from_mesh.py
# (the sim-native stand-in for GroundedSAM face-set lifting; same trusted
# symbols anchor the same bands)
# ---------------------------------------------------------------------------

def _q(x: np.ndarray, quantile: float) -> np.ndarray:
    return x >= np.quantile(x, quantile)


def part_masks(name: str, P: np.ndarray, spec: dict) -> dict[str, np.ndarray]:
    """Boolean sample masks per named part region, derived from the SAME
    calibrated frames.json symbols the compiler consumes. Unknown objects
    fall back to top/bottom bands only (still gives part coverage of the
    two universally meaningful contact regions)."""
    z = P[:, 2]
    masks: dict[str, np.ndarray] = {}
    if name == "teapot":
        a = np.asarray(spec["axes"]["pour_axis"]["xyz"], dtype=float)
        a = a / np.linalg.norm(a)
        proj = P @ a
        masks["spout"] = _q(proj, 0.97)
        masks["handle"] = _q(-proj, 0.97)
        masks["lid"] = _q(z, 0.985)
        masks["base"] = _q(-z, 0.98)
    elif name == "mug":
        oc = np.asarray(spec["points"]["opening_center"]["xyz"], dtype=float)
        rr = float(spec["quantities"]["rim_radius"]["value"])
        r_xy = np.linalg.norm(P[:, :2] - oc[:2], axis=1)
        masks["rim"] = _q(z, 0.98)
        masks["handle"] = r_xy > 1.12 * rr
        masks["base"] = _q(-z, 0.98)
    else:
        masks["top"] = _q(z, 0.98)
        masks["base"] = _q(-z, 0.98)
    return {k: m for k, m in masks.items() if m.sum() >= 3}


# ---------------------------------------------------------------------------
# constructed candidates — primitive-derived points surface sampling cannot
# produce. frames.json is the single source of truth for anything already
# calibrated there; only derived quantities (cavity midpoint, band centroids)
# are computed here, from the same band logic.
# ---------------------------------------------------------------------------

def constructed_candidates(name: str, V: np.ndarray,
                           spec: dict) -> list[dict]:
    out = []
    part_of = {"spout_tip": "spout", "handle_center": "handle",
               "opening_center": "rim"}
    for sym, entry in spec.get("points", {}).items():
        out.append({"xyz": np.asarray(entry["xyz"], dtype=float),
                    "source": "constructed", "symbol": sym,
                    "part": part_of.get(sym),
                    "on_surface": False})
    z = V[:, 2]
    if name == "mug":
        oc = np.asarray(spec["points"]["opening_center"]["xyz"], dtype=float)
        up = np.asarray(spec["axes"]["up_axis"]["xyz"], dtype=float)
        up = up / np.linalg.norm(up)
        depth = float(oc @ up - (V @ up).min())     # opening to base along up
        out.append({"xyz": oc - 0.5 * depth * up, "source": "constructed",
                    "symbol": "mid_cavity", "part": None,
                    "on_surface": False})
        base = V[_q(-z, 0.98)]
        out.append({"xyz": np.array([oc[0], oc[1], float(base[:, 2].mean())]),
                    "source": "constructed", "symbol": "base_center",
                    "part": "base", "on_surface": False})
        rr = float(spec["quantities"]["rim_radius"]["value"])
        r_xy = np.linalg.norm(V[:, :2] - oc[:2], axis=1)
        hb = V[r_xy > 1.12 * rr]
        if len(hb) >= 10:
            out.append({"xyz": hb.mean(axis=0), "source": "constructed",
                        "symbol": "handle_center", "part": "handle",
                        "on_surface": False})
    elif name == "teapot":
        lid = V[_q(z, 0.985)]
        out.append({"xyz": np.array([float(lid[:, 0].mean()),
                                     float(lid[:, 1].mean()),
                                     float(lid[:, 2].mean())]),
                    "source": "constructed", "symbol": "lid_center",
                    "part": "lid", "on_surface": False})
        base = V[_q(-z, 0.98)]
        out.append({"xyz": np.array([float(base[:, 0].mean()),
                                     float(base[:, 1].mean()),
                                     float(base[:, 2].mean())]),
                    "source": "constructed", "symbol": "base_center",
                    "part": "base", "on_surface": False})
    return out


# ---------------------------------------------------------------------------
# pool assembly: union -> priority NMS -> budget -> part coverage guarantee
# ---------------------------------------------------------------------------

@dataclass
class Pool:
    candidates: list[dict]
    class_counts_raw: dict[str, int]
    class_counts_kept: dict[str, int]
    readmitted: list[str] = field(default_factory=list)


def propose(name: str, V: np.ndarray, F: np.ndarray, spec: dict,
            seed: int = SEED) -> Pool:
    rng = np.random.default_rng(seed)
    P, fi = surface_samples(V, F, DENSE_SAMPLES, rng)
    defect = vertex_defects(V, F)
    sal = defect[F[fi]].mean(axis=1)        # per-sample: mean defect of face

    raw: list[dict] = []

    # constructed (priority 0)
    raw += constructed_candidates(name, V, spec)

    # part-aware (priority 1): FPS quota inside each geometric part band
    masks = part_masks(name, P, spec)
    for part, m in sorted(masks.items()):
        Pp = P[m]
        for i in fps(Pp, PART_QUOTA_EACH):
            raw.append({"xyz": Pp[i], "source": "part", "part": part,
                        "on_surface": True})

    # curvature-aware (priority 2): FPS spread over the salient decile
    m = sal >= np.quantile(sal, CURVATURE_QUANTILE)
    Pc, sc = P[m], sal[m]
    for i in fps(Pc, CURVATURE_QUOTA):
        raw.append({"xyz": Pc[i], "source": "curvature",
                    "saliency": float(sc[i]), "part": None,
                    "on_surface": True})

    # fps coverage (priority 3)
    for i in fps(P, FPS_QUOTA):
        raw.append({"xyz": P[i], "source": "fps", "part": None,
                    "on_surface": True})

    # greedy priority NMS + budget cap
    prio = {"constructed": 0, "part": 1, "curvature": 2, "fps": 3}
    order = sorted(range(len(raw)),
                   key=lambda i: (prio[raw[i]["source"]],
                                  -raw[i].get("saliency", 0.0)))
    kept: list[dict] = []
    for i in order:
        c = raw[i]
        if len(kept) >= POOL_BUDGET:
            break
        if all(np.linalg.norm(c["xyz"] - k["xyz"]) >= NMS_RADIUS_M
               for k in kept):
            kept.append(c)

    # part coverage guarantee: every discovered part keeps >=1 candidate
    readmitted = []
    covered = {c.get("part") for c in kept}
    for part in masks:
        if part not in covered:
            best = next((raw[i] for i in order
                         if raw[i].get("part") == part), None)
            if best is not None:
                kept.append(best)
                readmitted.append(part)

    # stable final ordering + ids
    kept.sort(key=lambda c: (prio[c["source"]], c.get("part") or "~",
                             c.get("symbol") or ""))
    for j, c in enumerate(kept):
        c["id"] = j

    def count(cands):
        d: dict[str, int] = {}
        for c in cands:
            d[c["source"]] = d.get(c["source"], 0) + 1
        return d

    return Pool(kept, count(raw), count(kept), readmitted)


def pool_to_json(name: str, pool: Pool, seed: int = SEED) -> dict:
    return {
        "object": name,
        "units": "meters",
        "coordinates": "body frame of mjcf body 'object' "
                       "(same convention as frames.json / PoseReader)",
        "schema": "interaction point candidate POOL for the 3D-propose / "
                  "2D-select grounding stage; union of fps + curvature + "
                  "part + constructed classes, priority-NMS deduplicated. "
                  "Marked-render generation and per-view legibility live "
                  "downstream.",
        "params": {"dense_samples": DENSE_SAMPLES, "fps_quota": FPS_QUOTA,
                   "curvature_quota": CURVATURE_QUOTA,
                   "part_quota_each": PART_QUOTA_EACH,
                   "curvature_quantile": CURVATURE_QUANTILE,
                   "pool_budget": POOL_BUDGET,
                   "nms_radius_m": NMS_RADIUS_M, "seed": seed},
        "class_counts": {"raw": pool.class_counts_raw,
                         "kept": pool.class_counts_kept},
        "coverage_readmitted_parts": pool.readmitted,
        "candidates": [
            {k: (round(float(x), 4) if k == "saliency"
                 else [round(float(v), 4) for v in x] if k == "xyz" else x)
             for k, x in c.items() if x is not None}
            for c in pool.candidates
        ],
    }