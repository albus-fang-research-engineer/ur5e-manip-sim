"""Selection -> w-frame wiring — the bridge from a validated VLM
touchpoint-#2 output (candidate mark ID + grounded axis name + sign) to
an anchored frames.Frame, closing the gap between the propose/render
island (proposal.py, render_candidates.py, vlm.py) and the planning
island (pour_stages.py, plan_pour_tea.py's hardcoded frames.json reads).

The resolution contract, per selection:

    origin     candidate_id -> xyz from the candidate pool (body coords;
               proposal.py's deterministic IDs are the shared address
               space — menu_from_pool below builds the Vocabulary menu
               from the SAME pool, so the parser's accept set and this
               resolver's lookup table cannot drift apart)
    axis       grounded name -> direction, through the resolution ladder
               below; the VLM's sign token (its one geometric
               contribution) is applied to the resolved direction
    secondary  grounded name -> direction through the same ladder, or
               the declared default (see below)

Axis resolution ladder — refined basis over per-selection Gram-Schmidt:

The original wiring sketch composed frames by Gram-Schmidt of raw
frames.json vectors per selection. That is superseded, not discarded:
perception now refines the WHOLE body orientation once per object
(Orient Anything coarse up/front in the convergence basin ->
refine.py fit -> refine_frame.py azimuth + assembly -> one orthonormal
basis R = [front, left, up] with per-row sigma), and the named semantic
axes are signed COLUMNS of that basis:

    up_axis -> R[:,2]     pour_axis -> R[:,0]     tilt_axis -> R[:,1]

(exactly the frames.json relations: tilt = up x pour). Frame.T() still
runs Gram-Schmidt, but columns of one orthonormal basis are a fixed
point of it — GS survives as a consistency guarantee instead of a
construction step. Names OUTSIDE the basis map (handle_axis, a
part-level fit) fall back to the frames.json calibrated vector, with
the source recorded. No basis supplied -> frames.json throughout (the
ground-truth arm of the emission ablation, unchanged).

Anchoring: RefineResult/FrameResult store DIRECTIONS only — position-
free vectors. Anchoring happens exactly once, when the resolved
directions and the candidate xyz meet in Frame(point=..., axis=...);
the refined up needs no re-fit or translation to "move" it to the
interaction point, and the demo render drawing up through the cloud
centroid is a plotting choice, not a property of the estimate. The one
anchor-adjacent subtlety is UNCERTAINTY, handled below.

Per-row coupling in an arbitrary frame built from the basis:
FrameResult.couple_rot_bounds is valid only when the frame's +z IS the
refined up. A selected frame may put any signed column (or any GS
combination) at +z — the pour frame's z is the tilt axis — which
permutes/mixes which Bw row inherits which estimator's sigma.
couple_rot_bounds_in_frame generalizes it: to first order the basis
orientation error is a rotation-vector with covariance

    Sigma = sigma_up^2 (I - u u^T) + sigma_az^2 (u u^T),   u = refined up

(tilts of up in any in-plane direction carry the up fit's sigma; spin
about up carries the azimuth route's), so a Bw row bounding rotation
about frame axis f gets sigma_row = sqrt(f^T Sigma f) — reducing
exactly to couple_rot_bounds when the frame equals the basis. Rejection
semantics follow refine_frame.py's asymmetry, per component: a rejected
UP contributes no floor (authored kept — the coarse direction still
means something, there is just no trusted sigma); a rejected AZIMUTH
makes any row with a nonzero up-component of f FREE (that component
bounds rotation against an ARBITRARY x reference — a tight bound there
is not conservative, it is meaningless).

Typed failure: every resolution failure raises ResolutionError with a
routable message — the hook the typed-failure router will map onto
reselect_point / reselect_axis repair actions.

numpy + manip_sim.frames / refine / refine_frame / vlm only; no
simulator dependency, importable by scripts, the preview loop, and the
hardware orchestrator alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import Frame, Symbols
from .refine import RefineResult, refine_axis, extract_ring_from_mesh
from .refine_frame import (AzimuthResult, FrameResult, assemble_frame,
                           azimuth_from_part_cloud, azimuth_from_points,
                           azimuth_from_semantic)
from .tsr import FREE_ROT
from .vlm import PointAxisSelection, WORLD_AXES

# ------------------------------------------------------------- constants
# Constants, not flags — properties of the method under ablation.

# Named semantic axes that ARE columns of the refined body basis
# R = [front, left, up], as (column, sign). Names absent here resolve
# through frames.json even when a basis is supplied (they are part-level
# fits, not body-basis columns).
REFINED_COLUMNS: dict[str, tuple[int, float]] = {
    "pour_axis": (0, +1.0),     # front — the azimuth the routes refined
    "tilt_axis": (1, +1.0),     # left  = up x pour (frames.json relation)
    "up_axis": (2, +1.0),       # the refined (or coarse-kept) up
}
# In-plane basis columns (front/left) are ARBITRARY when no azimuth
# route was accepted; resolving them from such a basis would launder an
# arbitrary direction into a calibrated-looking frame. They fall back to
# frames.json instead (recorded in provenance).
_IN_PLANE_COLS = (0, 1)
# Up-component squared above which a Bw rotation row is forced FREE when
# the azimuth was rejected (the row then bounds rotation against an
# arbitrary reference). c = |f . u| > ~0.001 — essentially any
# non-perpendicular axis.
AZ_FREE_EPS2 = 1e-6
# Calibrated-point trust for the constructed azimuth route (the residual
# regime of calibrate_frames_from_mesh's fits on converted meshes;
# deployment-side this becomes the grounding pipeline's mask-lift +
# primitive-fit residual). Same constant, same rationale as the demo.
POINT_SIGMA_M = 0.002
# Extremal-band fraction along the coarse front for carving a part
# sliver when no segmented part cloud exists (sim stand-in for the
# segmented spout mask; selection along the COARSE front on purpose —
# selection is the semantic layer's job and must survive its error).
SLIVER_FRAC = 0.22
# VLM-facing menu subset: at most this many marks reach the touchpoint-#2
# render and menu (5-8 per plan; crowding at the full 48-mark pool is a
# bigger legibility threat than font size). Surface samples of a
# stage-matching part are capped so one part cannot flood the menu that
# the constructed points and coverage tiers must share.
SUBSET_BUDGET = 8
SUBSET_PART_CAP = 2


class ResolutionError(ValueError):
    """A licensed-but-unresolvable selection: valid vocabulary, wrong
    geometry/tables. Routable by the typed-failure layer."""


# ------------------------------------------------------ pool <-> menu glue

def load_pool(asset_dir) -> dict[int, dict]:
    """candidates.json (propose_interaction_points.py --write) as an
    id-indexed table. IDs are deterministic per mesh+frames.json, so the
    same table the menu was built from resolves the returned ID."""
    path = Path(asset_dir) / "candidates.json"
    if not path.exists():
        raise ResolutionError(
            f"no candidate pool at {path} — run "
            "scripts/propose_interaction_points.py --write first")
    spec = json.loads(path.read_text())
    pool: dict[int, dict] = {}
    for c in spec.get("candidates", []):
        c = dict(c)
        c["xyz"] = np.asarray(c["xyz"], dtype=float).reshape(3)
        pool[int(c["id"])] = c
    if not pool:
        raise ResolutionError(f"empty candidate pool in {path}")
    return pool


def menu_tag(c: dict) -> str:
    """Short human tag for one candidate — symbol if grounded, else
    part, else class. The VLM sees these next to the mark IDs."""
    if c.get("symbol"):
        return f"{c['source']} {c['symbol']}" + (
            f" ({c['part']})" if c.get("part") else "")
    if c.get("part"):
        return f"{c['source']}:{c['part']}"
    return c["source"]


def menu_from_pool(pool: dict[int, dict]) -> dict[int, str]:
    """The Vocabulary menu for touchpoint #2, from the SAME pool this
    module resolves against — single ID space, no drift."""
    return {i: menu_tag(c) for i, c in sorted(pool.items())}


def _part_match(token: str, cand: dict) -> bool:
    """Free-text stage part name vs. a candidate's part tag / symbol.
    Stage parts are free text by design (vlm.StageSpec); pool tags are
    the fixed band names — bidirectional substring on normalized
    lowercase covers 'spout' vs 'spout_tip' and 'cavity' vs
    'mid_cavity' without a synonym table."""
    t = token.strip().lower()
    if not t:
        return False
    for tag in (cand.get("part"), cand.get("symbol")):
        if not tag:
            continue
        g = str(tag).strip().lower()
        if t == g or t in g or g in t:
            return True
    return False


def vlm_subset(pool: dict[int, dict],
               parts: list[str] | tuple[str, ...] | None = None,
               budget: int = SUBSET_BUDGET) -> dict[int, dict]:
    """The <= budget candidates offered to VLM touchpoint #2, chosen from
    the full pool by semantic priority. IDs are the POOL ids, never
    renumbered — this subset must feed BOTH menu_from_pool (the parser's
    accept set) and the marked render (what the VLM sees), so drawn
    marks and licensed IDs cannot drift apart. Filtering in the render
    script alone would leave the menu offering unmarked IDs.

    Stopgap ranking, replaced (not re-plumbed) when step-5 TSR
    pre-scoring lands: the pre-scorer will prune zero-admittance
    candidates and reorder the survivors; the contract — one subset,
    consumed by menu and renderer alike — is the durable part, and this
    heuristic remains the semantic prior when no scores exist.

    Tiers, each in ascending-ID order, filled until the budget:

      0  constructed candidates matching a stage part (free text)
      1  surface part-class candidates of matching parts, capped at
         SUBSET_PART_CAP per part
      2  remaining constructed (primitive-derived points — cavity and
         base centers — that surface sampling cannot produce)
      3  one part-class candidate per part not yet represented (the
         pool's coverage guarantee, echoed at menu scale)
      4  curvature, then 5 fps, as saliency/coverage filler

    With no stage parts, tiers 0-1 are empty and the ordering degrades
    to constructed -> per-part coverage -> curvature -> fps.
    """
    tokens = [p for p in (parts or []) if p and p.strip()]

    def match(c: dict) -> bool:
        return any(_part_match(t, c) for t in tokens)

    ordered = [pool[i] for i in sorted(pool)]
    chosen: dict[int, dict] = {}
    per_part: dict[str, int] = {}

    def take(c: dict) -> None:
        if len(chosen) < budget and c["id"] not in chosen:
            chosen[c["id"]] = c

    for c in ordered:                                    # tier 0
        if c["source"] == "constructed" and match(c):
            take(c)
    for c in ordered:                                    # tier 1
        if c["source"] == "part" and match(c):
            part = c.get("part") or ""
            if per_part.get(part, 0) < SUBSET_PART_CAP:
                before = len(chosen)
                take(c)
                if len(chosen) > before:
                    per_part[part] = per_part.get(part, 0) + 1
    for c in ordered:                                    # tier 2
        if c["source"] == "constructed":
            take(c)
    covered = {c.get("part") for c in chosen.values() if c.get("part")}
    for c in ordered:                                    # tier 3
        part = c.get("part")
        if c["source"] == "part" and part and part not in covered:
            before = len(chosen)
            take(c)
            if len(chosen) > before:
                covered.add(part)
    for source in ("curvature", "fps"):                  # tiers 4, 5
        for c in ordered:
            if c["source"] == source:
                take(c)

    return {i: chosen[i] for i in sorted(chosen)}


# ------------------------------------------------------- basis construction

def extremal_band(P: np.ndarray, up: np.ndarray, front: np.ndarray,
                  frac: float = SLIVER_FRAC) -> np.ndarray:
    """Extremal band of the cloud along the in-plane coarse front — the
    sim/calibration stand-in for a segmented part mask (spout sliver)."""
    u = up / np.linalg.norm(up)
    f = front - float(front @ u) * u
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    h = (P - P.mean(axis=0)) @ f
    return P[h >= h.max() - frac * (h.max() - h.min())]


def refine_body_basis(P: np.ndarray, coarse_up: np.ndarray,
                      coarse_front: np.ndarray, *, mesh=None,
                      front_pair: tuple[np.ndarray, np.ndarray] | None = None,
                      part_cloud: np.ndarray | None = None,
                      point_sigma_m: float = POINT_SIGMA_M) -> FrameResult:
    """The Orient-Anything -> refined-basis ladder, once per object:
    coarse up/front (in the validated convergence basin) in, one
    FrameResult out. Lifts demo_refine_frame's stage logic into the
    library so the orchestrator and the demo share one path.

      UP       revolution fit; on typed rejection and with a mesh, the
               terminal ring routes (top then bottom).
      AZIMUTH  explicit declared order: constructed (front_pair, e.g.
               handle_center -> spout_tip) -> part_pca (part_cloud,
               caller-carved; use extremal_band when no segmentation
               exists) -> semantic (always present, the honest
               fallback). Routes the caller does not supply are simply
               absent — route choice stays explicit.
    """
    up = refine_axis(P, coarse_up, "revolution")
    if not up.accepted and mesh is not None:
        for side in ("top", "bottom"):
            rpts = extract_ring_from_mesh(mesh, coarse_up, side)
            if len(rpts) < 50:
                continue
            rres = refine_axis(rpts, coarse_up, "rim")
            if rres.accepted:
                up = rres
                break
    cands: list[AzimuthResult] = []
    if front_pair is not None:
        p_from, p_to = front_pair
        cands.append(azimuth_from_points(p_from, p_to, up.direction,
                                         point_sigma_m=point_sigma_m))
    if part_cloud is not None:
        cands.append(azimuth_from_part_cloud(part_cloud, up.direction,
                                             coarse_front))
    cands.append(azimuth_from_semantic(coarse_front, up.direction))
    return assemble_frame(up, cands)


# ----------------------------------------------------------- resolution

@dataclass(frozen=True)
class ResolvedFrame:
    """A selection made metric: the anchored Frame plus everything the
    preview loop needs to say WHERE each ingredient came from."""
    frame: Frame                     # drops into the existing composition path
    selection: PointAxisSelection    # provenance: what the VLM said
    candidate: dict                  # the pool entry the ID resolved to
    axis_source: str                 # "refined" | "coarse-kept" | "frames.json"
    secondary_source: str            # same, or "default"
    basis: FrameResult | None        # for Bw coupling downstream (None ->
                                     # frames.json arm, no coupling data)


def _split_qualified(name: str, symbols: Symbols) -> str:
    if name in WORLD_AXES:
        raise ResolutionError(
            f"'{name}' is a world axis: world-anchored w frames are built "
            "in world coordinates (pour_stages), not resolved in a body "
            "frame — select a grounded object axis here")
    obj, _, local = name.partition(".")
    if not local:
        raise ResolutionError(f"axis '{name}' is not qualified "
                              "(expected '<object>.<axis>')")
    if obj != symbols.object:
        raise ResolutionError(
            f"axis '{name}' names object '{obj}' but the selection is "
            f"being resolved against '{symbols.object}' — cross-object "
            "axes are a stage-structure error, not a frame ingredient")
    return local


def _resolve_axis(name: str, symbols: Symbols,
                  basis: FrameResult | None) -> tuple[np.ndarray, str]:
    """(unit direction in body coords, source tag) for a qualified axis
    name, through the ladder: refined column when a basis is supplied
    and the name maps to a trustworthy column; frames.json otherwise."""
    local = _split_qualified(name, symbols)
    if basis is not None and local in REFINED_COLUMNS:
        col, csign = REFINED_COLUMNS[local]
        if col in _IN_PLANE_COLS and not basis.accepted:
            pass          # arbitrary x/y — fall through to frames.json
        else:
            v = csign * basis.R[:, col].copy()
            if col == 2 and not basis.up.accepted:
                return v, "coarse-kept"     # coarse passed back by refine
            return v, "refined"
    if local not in symbols.axes:
        raise ResolutionError(
            f"axis '{name}' is not in {symbols.object}'s grounded axes "
            f"{sorted(symbols.axes)} and does not map to a refined basis "
            "column")
    return symbols.axes[local].copy(), "frames.json"


def resolve_selection(selection: PointAxisSelection, pool: dict[int, dict],
                      symbols: Symbols,
                      basis: FrameResult | None = None) -> ResolvedFrame:
    """(candidate_id, axis, sign, secondary) -> anchored Frame.

    Origin: the candidate's xyz — anchoring is exactly this line; the
    resolved directions are position-free and need no recomputation.
    Axis: resolution ladder, VLM sign applied. Secondary: same ladder;
    None defaults to the refined front when a trustworthy basis exists
    (so `axis=up_axis, secondary=None` reproduces the refined basis
    anchored at the candidate), else frames.py's body -z default.
    """
    cand = pool.get(int(selection.candidate_id))
    if cand is None:
        raise ResolutionError(
            f"candidate_id {selection.candidate_id} is not in the pool "
            f"(ids {sorted(pool)}) — menu and pool built from different "
            "artifacts?")
    axis_dir, axis_src = _resolve_axis(selection.axis, symbols, basis)
    if selection.sign == "-":
        axis_dir = -axis_dir

    if selection.secondary is not None:
        sec_dir, sec_src = _resolve_axis(selection.secondary, symbols, basis)
    elif basis is not None and basis.accepted:
        sec_dir, sec_src = basis.R[:, 0].copy(), "refined"
    else:
        sec_dir, sec_src = np.array([0.0, 0.0, -1.0]), "default"

    status = "placeholder" if axis_src == "coarse-kept" else "calibrated"
    frame = Frame(
        name=f"{symbols.object}.selected({selection.candidate_id},"
             f"{selection.sign}{selection.axis.partition('.')[2]})",
        point=cand["xyz"].copy(), axis=axis_dir, secondary=sec_dir,
        status=status,
        comment=f"selection: mark {selection.candidate_id} "
                f"[{menu_tag(cand)}], axis {selection.sign}"
                f"{selection.axis} ({axis_src}), secondary {sec_src}")
    return ResolvedFrame(frame=frame, selection=selection, candidate=cand,
                         axis_source=axis_src, secondary_source=sec_src,
                         basis=basis)


# ------------------------------------------------- coupling, general frame

def couple_rot_bounds_in_frame(basis: FrameResult, R_frame: np.ndarray,
                               Bw: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Row-wise rotational coupling for a Bw authored in ANY frame whose
    orientation R_frame (columns = frame x,y,z in body coords) was built
    from this basis — the generalization of FrameResult.couple_rot_bounds
    beyond z == refined up. See the module docstring for the variance-
    projection rule and the per-component rejection semantics. Reduces
    exactly to couple_rot_bounds when R_frame == basis.R. Translation
    rows pass through; already-free rows are never touched; midpoints
    are preserved (coupling widens, never re-centers)."""
    Bw = np.array(Bw, dtype=float, copy=True)
    if Bw.shape != (6, 2):
        raise ValueError("Bw must be 6x2")
    R_frame = np.asarray(R_frame, dtype=float)
    if R_frame.shape != (3, 3):
        raise ValueError("R_frame must be 3x3")
    u = basis.R[:, 2]
    two_pi = 2.0 * np.pi
    var_up = (k * np.deg2rad(basis.up.sigma_deg)) ** 2 \
        if basis.up.accepted else 0.0
    var_az = (k * np.deg2rad(basis.azimuth.sigma_deg)) ** 2 \
        if basis.azimuth.accepted else 0.0

    for row in (3, 4, 5):
        lo, hi = Bw[row]
        if hi - lo >= two_pi - 1e-9:            # already free
            continue
        f = R_frame[:, row - 3]
        c2 = float(f @ u) ** 2                  # up component (spin weight)
        s2 = max(0.0, 1.0 - c2)                 # in-plane (tilt weight)
        if not basis.azimuth.accepted and c2 > AZ_FREE_EPS2:
            Bw[row] = FREE_ROT                  # bound vs arbitrary x
            continue
        floor = np.sqrt(var_up * s2 + var_az * c2)
        half = max(0.5 * (hi - lo), floor)
        mid = 0.5 * (lo + hi)
        if half >= np.pi:
            Bw[row] = FREE_ROT
        else:
            Bw[row] = (mid - half, mid + half)
    return Bw


def couple_resolved(rf: ResolvedFrame, Bw: np.ndarray,
                    k: float = 3.0) -> np.ndarray:
    """Coupling for a Bw authored in a ResolvedFrame's frame. No basis
    (frames.json arm) -> authored bounds pass through unchanged, the
    ground-truth-arm behavior."""
    if rf.basis is None:
        return np.array(Bw, dtype=float, copy=True)
    return couple_rot_bounds_in_frame(rf.basis, rf.frame.T()[:3, :3], Bw, k)


# ----------------------------------------------------------- serialization

def selection_to_json(sel: PointAxisSelection) -> dict:
    return {"candidate_id": sel.candidate_id, "axis": sel.axis,
            "sign": sel.sign, "secondary": sel.secondary,
            "rationale": sel.rationale}


def selection_from_json(d: dict) -> PointAxisSelection:
    return PointAxisSelection(
        candidate_id=int(d["candidate_id"]), axis=str(d["axis"]),
        sign=str(d["sign"]), secondary=d.get("secondary"),
        rationale=str(d.get("rationale", "")))


def load_selections(path) -> dict[str, PointAxisSelection]:
    """A role-keyed selections artifact ({"grasp": {...}, ...}) — what
    the orchestrator will write per stage from Client.select_point_axis
    outputs, and what plan_pour_tea consumes via --selections."""
    spec = json.loads(Path(path).read_text())
    return {role: selection_from_json(d) for role, d in spec.items()}