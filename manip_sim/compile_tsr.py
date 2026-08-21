"""The grounding compiler (touchpoint-#3 consumer): resolves a validated
StageEmission — pure symbols — to numeric TSR pairs (T0_w, Tw_e, B^w).

This is the module vlm.py's docstring promises: the enum -> numeric
tables live HERE and nowhere else. The VLM layer owns membership of the
token alphabets; this layer owns their meaning. The tables below are the
frozen constants of the paper's generalizability argument — they are
deliberately module-level constants, not flags (perturbing them is the
sensitivity-analysis ablation, run by editing this file under a --tag,
never by a runtime knob that would let per-scenario tuning leak in).

Compilation model
-----------------
w frame: composed exactly as the hand-authored path does —
Symbols.frame(w_origin, w_axis) rooted on the pose of the object that
owns w_origin (body_poses[owner] @ frame.T()). The caller decides WHICH
pose roots it; freezing (the pour's "w at stage entry") is therefore the
caller's move, same as pour_pair's contract. w_axis maps to w's +z.

Tw_e (e_feature given -> e is the ACTIVE BODY frame):
inv(make_pose(e_feature.point, R_body0^T @ R0_w)). Two independent
components, each carrying one lesson:

  * translation = the feature POINT, never the feature frame — the
    pour_stages.transport_pair trap: a full feature frame makes
    rotation bounds bind the feature attitude (spout at the ceiling)
    and empties the intersection. Since trans(A @ B) ignores B's
    rotation, translation rows still read exactly "feature point in w".
  * rotation = R_body0^T @ R0_w zeroes the NOMINAL rotational
    displacement (the generalization of pour_pair's Tw_e =
    inv(T_w_body_at_entry)); without it, a w frame whose attitude
    differs from the body's (the tilt frame) puts the nominal at a
    large RPY value and every row rule below — all derived as
    displacements about the nominal — mis-centers. The v1 cost,
    bounded by ALIGN_TOL via the Case-A nominal gate: relation rows
    are centered on the entry attitude, not the gravity-true one, so
    an entry tilted within the gate shifts the compiled center by
    that tilt.

e_feature=None gives Tw_e = I: e is whatever frame the caller tests
(the grasp classifier over gripper poses), whose nominal is not the
active body's and must not be corrected against it.

B^w rows — the rule table (v1, restrictions are CompileErrors so they
route to touchpoint #5 rather than silently mis-compiling):

  rot rows start UNSET; addressed rows are set (intersecting on double
  address); unaddressed rows default to +-ROT_TOL_RAD["tight"]. trans
  rows start UNSET and default to FREE. The asymmetry is forced by the
  vocabulary and is fail-safe per domain: per-row rotational freedom is
  expressible ({"relation": "free", "row": ...}) so undeclared rotation
  of a held object stays tight; per-row translational freedom is not
  expressible, so undeclared translation stays free and explicit terms
  narrow it (the planner explores translation inside the other terms;
  an undeclared free spin of a full teapot is the hazardous default).

  RotRow(axis A, relation, reference R, tol t): the constraint is
  "angle(displaced A, static R) in [center(relation) +- t]" with center
  parallel=0, perpendicular=pi/2, antiparallel=pi. A must belong to the
  ACTIVE object (it displaces); R must be a world axis or an axis of the
  w-owning/passive object (static in w). Box conversion requires the
  directions to diagonalize in w at zero displacement (ALIGN_TOL gate):

    A ~ +-z, R ~ +-z, nominal angle matches center  -> roll,pitch +- t
    A, R both in w's xy-plane                       -> yaw in
        signed_center +- t, where signed_center is the branch of
        (beta - alpha -+ center) chosen POSITIVE — resolvable only
        because frames.json axes are sign-calibrated ("tilt_axis:
        positive tips spout down"); an ambiguous branch (both positive
        or both negative after dedupe) is a CompileError, as is any
        alignment case outside these two rows of the table.

  TransTerm -> rows (offsets: anchors must live on the w-owning object,
  so their w coordinates are constants; `off` below):

    above    z in [off_z + clearance, off_z + clearance + band(slack)]
    below    z in [off_z - clearance - band(slack), off_z - clearance]
    centered x,y in off_xy +- CENTERED_TOL_M[tol]   (lateral only:
             "centered on the w axis" — it deliberately does NOT touch
             z so it composes with above/below without emptying them)
    inside   x,y,z in off +- INSIDE_TOL_M[slack]    (containment cube;
             this is the pour pivot pin at slack="snug")
    along    half-line [0, inf) / (-inf, 0] on the w row the named axis
             aligns with, sign composed from axis alignment and the term
             sign token
    expr     row in [eval(lo), eval(hi)] over the object quantity
             tables (grammar already validated by vlm.validate_expr;
             evaluated here against frames.json values)
    free     no-op inside a list (rows already default free)

Known expressivity gaps, deliberately not papered over (they are
ablation content, not bugs): the grasp slide band (+-2 cm along the
handle with +-5 mm lateral) has no anisotropic containment term, so the
symbolic grasp classifier is a tighter cube; the pour path's bounded
tilt corridor (-3..110 deg) is only expressible as a fully free tilt
row. Where the hand-authored and compiled arms differ, the experiment —
not this module — adjudicates.

numpy-only (imports manip_sim.tsr / .frames); no simulator, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import Frame, Symbols
from .tsr import FREE_ROT, FREE_TRANS, TSR, make_pose
from .vlm import RotRow, StageEmission, TSRSpec, TransTerm

# ------------------------------------------------- the frozen enum tables
# Anchored, where a hand-authored constant exists, to pour_stages.py —
# the ground-truth arm — so the "schema" arm reproduces it by table
# lookup, not by tuning. Each line names its anchor.

ROT_TOL_RAD = {                       # rotational half-widths
    "tight": np.deg2rad(5.0),         # pour tilt_tol
    "moderate": np.deg2rad(15.0),     # transport upright_tol
    "loose": np.deg2rad(30.0),
}
CLEARANCE_M = {                       # standoff from the anchor plane
    "contact": 0.0,
    "small": 0.01,
    "medium": 0.03,                   # transport height lo
    "large": 0.08,
}
SLACK_BAND_M = {                      # band height for above/below
    "snug": 0.02,
    "moderate": 0.05,                 # transport height span (0.08-0.03)
    "loose": 0.12,
}
CENTERED_TOL_M = {                    # lateral half-width for centered
    "snug": 0.005,                    # grasp lateral_tol
    "moderate": 0.02,                 # transport rim_margin
    "loose": 0.05,
}
INSIDE_TOL_M = {                      # containment-cube half-width
    "snug": 0.005,                    # pour pivot_tol
    "moderate": 0.015,
    "loose": 0.04,
}

ALIGN_TOL_RAD = np.deg2rad(10.0)      # diagonalization gate

_ROT_CENTER = {"parallel": 0.0, "perpendicular": np.pi / 2,
               "antiparallel": np.pi}
_ROW_IDX = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}


class CompileError(ValueError):
    """A licit emission the v1 rule table cannot ground. `slot` names
    the offending schema slot (the authored_in pointer the critic /
    repair loop edits); `reason` is instructive on purpose — it is the
    text a touchpoint-#5 repair prompt will eventually carry."""

    def __init__(self, slot: str, reason: str):
        super().__init__(f"{slot}: {reason}")
        self.slot = slot
        self.reason = reason


@dataclass
class CompiledStage:
    stage: int
    name: str
    path: TSR
    subgoal: TSR
    w_frame: Frame                    # the composed w (body coords)
    notes: tuple[str, ...] = ()       # per-slot compilation provenance


# ------------------------------------------------------------ small math

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _classify_axis(v_w: np.ndarray, slot: str):
    """('z', sign) if within ALIGN_TOL of +-e_z; ('plane', alpha) if
    within ALIGN_TOL of w's xy-plane (alpha = signed in-plane angle);
    CompileError otherwise."""
    v = _unit(v_w)
    cz = float(np.clip(v[2], -1.0, 1.0))
    if np.arccos(abs(cz)) <= ALIGN_TOL_RAD:
        return ("z", 1.0 if cz > 0 else -1.0)
    if abs(np.arcsin(cz)) <= ALIGN_TOL_RAD:
        return ("plane", float(np.arctan2(v[1], v[0])))
    raise CompileError(slot, (
        "axis neither aligned with w's z nor in its xy-plane "
        f"(components in w: {np.round(v, 3).tolist()}); re-anchor "
        "w_axis so the constrained direction diagonalizes, or free "
        "the row"))


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _eval_expr(s: str, quantities: dict[str, float], slot: str) -> float:
    """Evaluate a bound expression already validated by
    vlm.validate_expr (same grammar: + - * / ( ) unary-minus, idents
    are qualified quantity symbols, numeric literals legal)."""
    from .vlm import _tokenize_expr  # same tokenizer, single grammar
    toks = _tokenize_expr(s)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None)

    def eat():
        nonlocal pos
        pos += 1
        return toks[pos - 1][1]

    def factor() -> float:
        k, v = peek()
        if k == "num":
            return float(eat())
        if k == "ident":
            name = eat()
            if name not in quantities:
                raise CompileError(slot, f"unknown quantity {name!r}")
            return quantities[name]
        if (k, v) == ("op", "-"):
            eat(); return -factor()
        if (k, v) == ("op", "("):
            eat(); r = expr()
            if peek() != ("op", ")"):
                raise CompileError(slot, f"malformed expression {s!r}")
            eat(); return r
        raise CompileError(slot, f"malformed expression {s!r}")

    def term() -> float:
        r = factor()
        while peek() in (("op", "*"), ("op", "/")):
            op = eat(); f = factor()
            r = r * f if op == "*" else r / f
        return r

    def expr() -> float:
        r = term()
        while peek() in (("op", "+"), ("op", "-")):
            op = eat(); t = term()
            r = r + t if op == "+" else r - t
        return r

    out = expr()
    if pos != len(toks):
        raise CompileError(slot, f"trailing tokens in expression {s!r}")
    return out


# ------------------------------------------------------------- row boxes

class _Box:
    """6x2 bound accumulator with UNSET rows and slot-named
    intersection."""

    def __init__(self):
        self.rows: list[tuple[float, float] | None] = [None] * 6

    def narrow(self, idx: int, lo: float, hi: float, slot: str):
        if hi < lo:
            raise CompileError(slot, f"bounds inverted: [{lo}, {hi}]")
        cur = self.rows[idx]
        if cur is None:
            self.rows[idx] = (lo, hi)
            return
        nlo, nhi = max(cur[0], lo), min(cur[1], hi)
        if nhi < nlo:
            raise CompileError(slot, (
                f"row {idx} intersection empty: existing "
                f"[{cur[0]:.4f}, {cur[1]:.4f}] vs [{lo:.4f}, {hi:.4f}]"))
        self.rows[idx] = (nlo, nhi)

    def finish(self, trans_default, rot_default) -> np.ndarray:
        out = []
        for i, r in enumerate(self.rows):
            if r is None:
                r = trans_default if i < 3 else rot_default
            out.append(r)
        return np.array(out, dtype=float)


# ------------------------------------------------------------ resolution

def _split(qualified: str, slot: str) -> tuple[str, str]:
    if "." not in qualified:
        raise CompileError(slot, f"unqualified symbol {qualified!r}")
    obj, name = qualified.split(".", 1)
    return obj, name


def _axis_dir_world(qualified: str, symbols: dict[str, Symbols],
                    body_poses: dict[str, np.ndarray],
                    slot: str) -> np.ndarray:
    obj, name = _split(qualified, slot)
    if obj == "world":
        return {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
                "z": np.array([0, 0, 1.0])}[name]
    if obj not in symbols:
        raise CompileError(slot, f"unknown object {obj!r}")
    if name not in symbols[obj].axes:
        raise CompileError(slot, f"unknown axis {qualified!r}")
    return body_poses[obj][:3, :3] @ _unit(symbols[obj].axes[name])


# ------------------------------------------------------------- compiler

def compile_stage(emission: StageEmission,
                  symbols: dict[str, Symbols],
                  body_poses: dict[str, np.ndarray],
                  e_feature: Frame | None = None) -> CompiledStage:
    """Ground one StageEmission. `body_poses` are the world body poses
    to root frames on (pass the frozen entry pose to get pour-at-entry
    semantics). `e_feature` is the active-object feature whose POINT
    Tw_e pins at w (translation-only, per the transport_pair lesson);
    None -> Tw_e = I (the TSR constrains the body/gripper frame
    directly)."""
    notes: list[str] = []

    # -- w frame ---------------------------------------------------------
    w_obj, w_point = _split(emission.w_origin, "w_origin")
    a_obj, w_axis = _split(emission.w_axis, "w_axis")
    if w_obj not in symbols:
        raise CompileError("w_origin", f"unknown object {w_obj!r}")
    if a_obj != w_obj:
        raise CompileError("w_axis", (
            f"w_axis {emission.w_axis!r} must belong to the object that "
            f"owns w_origin ({w_obj!r}); cross-object w frames are not "
            "compilable in v1"))
    if w_point not in symbols[w_obj].points:
        raise CompileError("w_origin", f"unknown point {emission.w_origin!r}")
    if w_axis not in symbols[w_obj].axes:
        raise CompileError("w_axis", f"unknown axis {emission.w_axis!r}")
    if w_obj not in body_poses:
        raise CompileError("w_origin", f"no body pose supplied for {w_obj!r}")
    w_frame = symbols[w_obj].frame(w_point, w_axis)
    T0_w = body_poses[w_obj] @ w_frame.T()
    R0_w = T0_w[:3, :3]
    Tf_inv = np.linalg.inv(w_frame.T())
    notes.append(f"w = {w_obj}.frame({w_point},{w_axis}) rooted on "
                 f"body_poses[{w_obj!r}]")

    # -- Tw_e ------------------------------------------------------------
    if e_feature is not None:
        if emission.active not in body_poses:
            raise CompileError("active", (
                f"e_feature given but no body pose for the active object "
                f"{emission.active!r} — the nominal-displacement "
                "correction needs it"))
        R_corr = body_poses[emission.active][:3, :3].T @ R0_w
        Tw_e = np.linalg.inv(make_pose(e_feature.point, R_corr))
        notes.append(f"Tw_e = inv(pose({e_feature.name} point, "
                     "R_body0^T R_w)) — feature point pinned, nominal "
                     "rotational displacement zeroed")
    else:
        Tw_e = np.eye(4)
        notes.append("Tw_e = I")

    def _anchor_off(qualified: str, slot: str) -> np.ndarray:
        obj, name = _split(qualified, slot)
        if obj != w_obj:
            raise CompileError(slot, (
                f"anchor {qualified!r} is not on the w-owning object "
                f"{w_obj!r}; it would not be static in w — set w_origin "
                "to the anchor instead"))
        if name not in symbols[obj].points:
            raise CompileError(slot, f"unknown point {qualified!r}")
        p = symbols[obj].points[name]
        return (Tf_inv @ np.append(p, 1.0))[:3]

    quantities = {f"{o}.{q}": v for o, s in symbols.items()
                  for q, v in s.quantities.items()}

    # every row rule below is a displacement about a zero nominal;
    # e_feature's Tw_e guarantees that exactly. Without it, relation
    # rows are only valid when the w attitude already matches the
    # active body's — gate it (best-effort; e may be a gripper frame
    # the compiler cannot see, in which case the body check is the
    # closest available proxy).
    if e_feature is None and emission.active in body_poses:
        R_mis = R0_w.T @ body_poses[emission.active][:3, :3]
        _nominal_ok = np.arccos(np.clip((np.trace(R_mis) - 1) / 2,
                                        -1.0, 1.0)) <= ALIGN_TOL_RAD
    else:
        _nominal_ok = True

    def _compile_spec(spec: TSRSpec, ctx: str) -> np.ndarray:
        box = _Box()

        # ---- rotation rows
        for i, r in enumerate(spec.rot):
            slot = f"{emission.name}.{ctx}.rot[{i}]"
            if r.relation == "free":
                box.narrow(_ROW_IDX[r.row], *FREE_ROT, slot=slot)
                continue
            if not _nominal_ok:
                raise CompileError(slot, (
                    "relation row with Tw_e = I but w attitude differs "
                    "from the active body's beyond ALIGN_TOL — the "
                    "nominal displacement is not zero and the row would "
                    "mis-center; supply e_feature, re-anchor w_axis, or "
                    "(single-object stage, e = gripper) emit the row as "
                    "{\"relation\": \"free\", \"row\": ...}"))
            a_own, _ = _split(r.axis, slot)
            if a_own != emission.active:
                raise CompileError(slot, (
                    f"constrained axis {r.axis!r} must belong to the "
                    f"active object {emission.active!r} (only the active "
                    "object displaces)"))
            ref_own, _ = _split(r.reference, slot)
            if ref_own == emission.active:
                raise CompileError(slot, (
                    f"reference {r.reference!r} is on the active object; "
                    "it displaces with the constrained axis and the "
                    "relation is degenerate"))
            a_w = R0_w.T @ _axis_dir_world(r.axis, symbols, body_poses, slot)
            ref_w = R0_w.T @ _axis_dir_world(r.reference, symbols,
                                             body_poses, slot)
            ka, va = _classify_axis(a_w, slot)
            kr, vr = _classify_axis(ref_w, slot)
            center = _ROT_CENTER[r.relation]
            tol = ROT_TOL_RAD[r.tol]
            if ka == "z" and kr == "z":
                nominal = 0.0 if va * vr > 0 else np.pi
                if abs(nominal - center) > ALIGN_TOL_RAD:
                    raise CompileError(slot, (
                        f"relation {r.relation!r} with both axes on w's z "
                        f"has nominal angle {np.degrees(nominal):.0f} deg; "
                        "a z/z cone can only bound roll,pitch about its "
                        "nominal — the tip direction is otherwise "
                        "ambiguous. Anchor w_axis on the rotation axis "
                        "and use an in-plane reference instead"))
                box.narrow(3, -tol, tol, slot)
                box.narrow(4, -tol, tol, slot)
                notes.append(f"{slot}: z/z cone -> roll,pitch +-"
                             f"{np.degrees(tol):.0f} deg")
            elif ka == "plane" and kr == "plane":
                branches = sorted({_wrap(vr - va - center),
                                   _wrap(vr - va + center)})
                if len(branches) == 2 and np.isclose(*branches):
                    branches = branches[:1]
                pos = [b for b in branches if b > 1e-9]
                if len(branches) == 1:
                    c = branches[0]
                elif len(pos) == 1:
                    c = pos[0]
                else:
                    raise CompileError(slot, (
                        f"yaw branch ambiguous ({[round(np.degrees(b), 1) for b in branches]} deg): the axis sign "
                        "convention does not disambiguate the rotation "
                        "direction; re-sign the axis in frames.json or "
                        "pick a signed reference"))
                box.narrow(5, c - tol, c + tol, slot)
                notes.append(f"{slot}: in-plane -> yaw "
                             f"{np.degrees(c):.0f} +- "
                             f"{np.degrees(tol):.0f} deg")
            else:
                raise CompileError(slot, (
                    f"mixed alignment (axis {ka!r}, reference {kr!r}) is "
                    "outside the v1 rule table"))

        # ---- translation terms
        for i, t in enumerate(spec.trans):
            slot = f"{emission.name}.{ctx}.trans[{i}]"
            if t.term == "free":
                continue
            if t.term in ("above", "below"):
                off = _anchor_off(t.anchor, slot)
                clr = CLEARANCE_M[t.clearance]
                band = SLACK_BAND_M[t.slack]
                if t.term == "above":
                    box.narrow(2, off[2] + clr, off[2] + clr + band, slot)
                else:
                    box.narrow(2, off[2] - clr - band, off[2] - clr, slot)
            elif t.term == "centered":
                off = _anchor_off(t.anchor, slot)
                tol = CENTERED_TOL_M[t.tol]
                box.narrow(0, off[0] - tol, off[0] + tol, slot)
                box.narrow(1, off[1] - tol, off[1] + tol, slot)
            elif t.term == "inside":
                off = _anchor_off(t.anchor, slot)
                tol = INSIDE_TOL_M[t.slack]
                for idx in range(3):
                    box.narrow(idx, off[idx] - tol, off[idx] + tol, slot)
            elif t.term == "along":
                d_w = R0_w.T @ _axis_dir_world(t.axis, symbols,
                                               body_poses, slot)
                v = _unit(d_w)
                idx = int(np.argmax(np.abs(v)))
                if np.arccos(abs(np.clip(v[idx], -1, 1))) > ALIGN_TOL_RAD:
                    raise CompileError(slot, (
                        f"along-axis {t.axis!r} does not align with a w "
                        f"basis vector (in w: {np.round(v, 3).tolist()})"))
                s = np.sign(v[idx]) * (1.0 if t.sign == "+" else -1.0)
                if s > 0:
                    box.narrow(idx, 0.0, np.inf, slot)
                else:
                    box.narrow(idx, -np.inf, 0.0, slot)
            elif t.term == "expr":
                lo = _eval_expr(t.expr_lo, quantities, slot)
                hi = _eval_expr(t.expr_hi, quantities, slot)
                box.narrow(_ROW_IDX[t.row], lo, hi, slot)

        tight = ROT_TOL_RAD["tight"]
        return box.finish(trans_default=FREE_TRANS,
                          rot_default=(-tight, tight))

    path_Bw = _compile_spec(emission.path_tsr, "path")
    goal_Bw = _compile_spec(emission.subgoal_tsr, "subgoal")

    def _tsr(Bw, kind):
        return TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                   name=f"{emission.name}/{kind}(emitted)")

    return CompiledStage(stage=emission.stage, name=emission.name,
                         path=_tsr(path_Bw, "path"),
                         subgoal=_tsr(goal_Bw, "subgoal"),
                         w_frame=w_frame, notes=tuple(notes))
