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
w frame: ONE canonical frame per object, used as w in every stage that
constrains against it — never chosen per stage, never emitted. It is the
object's canonical frame (OmniManip's canonical object space: up/front
from the pose source / Orient Anything, carried in frames.json as the
up_axis / front_axis / lateral_axis symbols) anchored at the interaction
point call #2 selected on it:

    z = passive up_axis,  x = passive front_axis,  origin = w_point

rooted on the passive body pose and frozen at stage entry (the caller
passes the entry poses, same as before). A passive with no confident
front_axis gets its x from the MOVER's lateral (up x front) projected
into w's horizontal plane at entry, then from world x/y: a tip of the
mover's front toward the vertical — the pour family — then rotates
about w.x, i.e. the ROLL row, which is pour_stages.pour_pair's
permutation arising from the fallback rather than a special case. A
stage with no passive object (the grasp) is the same rule with the
gripper as the mover: the grasped object owns w, Tw_e = I, and relation
rows cannot be grounded (the gripper has no grounded axes).

Tw_e (mover given): inv(make_pose(e_point, R_goal^T @ R0_w)). The
translation pins the mover's feature POINT at w (the transport_pair
lesson: a full feature frame makes rotation bounds bind the feature
attitude and empties the intersection); the rotation makes the GOAL
attitude the zero displacement, where the goal attitude is the one
that satisfies the relation rows, reached by the smallest rotation from
the entry attitude (below). Every row rule is a displacement about that
zero, so relation rows are centered where they are satisfied, not at
the entry — the v1 "Case-A nominal gate" is gone with its cost.

B^w rows — the rule table (v2; restrictions are CompileErrors so they
route to touchpoint #5 rather than silently mis-compiling):

  Every row starts UNSET; trans rows default FREE, and — new in v2 —
  rot rows default FREE too: a relation row fixes only the DOFs its axis
  pair determines, and a DOF no row mentions is free (the pour's
  "front_axis antiparallel world.z" fixes the tilt and leaves the
  heading about world.z free). The explicit {"relation": "free", "row":
  ...} form stays licit as a no-op.

  RotRow(axis A, relation, reference R, tol t): A belongs to the mover
  (it displaces); R is world.z or an axis of the w-owning object (static
  in w). With a_e = A's world direction at entry:

    parallel / antiparallel   goal: the minimal rotation taking a_e onto
                              +-R (about a_e x R; a near half-turn is a
                              CompileError — the tip direction is
                              ambiguous). Fixed DOFs: the two tilts of A
                              off the reference, i.e. the two RPY rows
                              NOT about the w basis vector R aligns with
                              (R ~ +-w.z -> roll, pitch; ~ +-w.x ->
                              pitch, yaw; ~ +-w.y -> roll, yaw); rotation
                              about R stays free.
    perpendicular             goal: the minimal rotation taking a_e into
                              the plane perpendicular to R (a_e parallel
                              to R is ambiguous -> CompileError). Fixed
                              DOF: the one tilt of A toward R, the row
                              about the w basis vector R x A_goal aligns
                              with.
    two parallel/antiparallel rows with independent axes fully determine
    the goal (least-squares fit, pairs must be mutually consistent);
    other combinations and >2 relation rows are outside the table.

  Each fixed row is +-t widened by the referenced symbols' sigmas:
  half = max(t, SIGMA_K * sqrt(sigma_A^2 + sigma_R^2 [+ sigma_up_w^2 on
  roll/pitch, the w frame's own tilt])) — refine.couple_rot_bound's
  floor rule, row-wise, from the per-symbol sigmas frames.json carries
  (none -> no floor, the authored sidecars' regime). The box requires the
  fixed rotation axes to land on w basis vectors within ALIGN_TOL; with
  canonical references that is automatic (R is w's z or x or y), a part
  axis or a tilted passive can fail it, typed.

  PATH rows additionally carry the entry-to-goal sweep: when the
  rotation from entry to the path rows' goal is about an axis within
  ALIGN_TOL of w.x (roll) or w.z (yaw), that row becomes the corridor
  [entry - t, goal + t]; otherwise — including w.y, because pitch is the
  middle Euler angle and a wide pitch corridor crosses its gimbal lock —
  the path's rotation rows are all FREE and translation carries the
  path. Known limitation, deliberately kept: wide path corridors are
  expressible only about the passive frame's basis directions; other
  reorientations need a stage split.

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
symbolic grasp classifier is a tighter cube; the pour path's hand-
authored tilt corridor (-3..110 deg, with overshoot past the goal)
compiles as the symmetric sweep [entry - t, goal + t] (-5..95 deg at
"tight") — overshoot is not a token; and the pour pivot is the PASSIVE
anchor (above + centered on the opening, "snug") rather than the hand
arm's +-5 mm pin of the spout tip at its entry position, so the
compiled pour may translate the tip within those tolerances while it
tilts. Where the hand-authored and compiled arms differ, the experiment
— not this module — adjudicates.

numpy-only (imports manip_sim.tsr / .frames); no simulator, no network.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from .frames import Frame, Symbols
from .tsr import FREE_ROT, FREE_TRANS, TSR, make_pose
from .vlm import FRAME_AXES, StageEmission, TSRSpec

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

ALIGN_TOL_RAD = np.deg2rad(10.0)      # "lands on a w basis vector" gate
SIGMA_K = 3.0                         # k-sigma floor, refine.couple_rot_bound's k

_UP, _FRONT, _LATERAL = FRAME_AXES
_ROW_IDX = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}


class CompileError(ValueError):
    """A licit emission the rule table cannot ground. `slot` names
    the offending schema slot (the authored_in pointer the critic /
    repair loop edits); `reason` is instructive on purpose — it is the
    text a touchpoint-#5 repair prompt will eventually carry."""

    def __init__(self, slot: str, reason: str,
                 others: tuple["CompileError", ...] = ()):
        self.slot = slot
        self.reason = reason
        self.others = tuple(others)   # further slots rejected in the same
                                      # emission (one per TSR), so a repair
                                      # turn sees every failure at once
        super().__init__("; ".join(f"{e.slot}: {e.reason}"
                                   for e in self.all()))

    def all(self) -> tuple["CompileError", ...]:
        return (self,) + self.others

    def text(self) -> str:
        """Every rejected slot, for the repair turn / gate log."""
        return str(self)


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


def _basis_row(v_w: np.ndarray, slot: str, what: str) -> tuple[int, float]:
    """(row k, sign) of the w basis vector unit v_w lies along, within
    ALIGN_TOL; CompileError otherwise."""
    v = _unit(v_w)
    k = int(np.argmax(np.abs(v)))
    if np.arccos(min(abs(float(v[k])), 1.0)) > ALIGN_TOL_RAD:
        raise CompileError(slot, (
            f"{what} does not lie along a w basis vector (components in "
            f"w: {np.round(v, 3).tolist()}); the bound would not "
            "diagonalize — relate a canonical axis (up/front/lateral) or "
            "free the row"))
    return k, (1.0 if v[k] > 0 else -1.0)


def _rot_onto(a: np.ndarray, t: np.ndarray, slot: str) -> np.ndarray:
    """Minimal rotation (3x3) taking unit a onto unit t."""
    c = float(np.clip(a @ t, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.eye(3)
    if th > np.pi - ALIGN_TOL_RAD:
        raise CompileError(slot, (
            f"axis is within {np.degrees(np.pi - th):.0f} deg of a half-"
            "turn from the reference at entry: the rotation axis is "
            "ambiguous (which way to flip). Relate an axis that is not "
            "already (anti)parallel to the reference, or split the stage"))
    n = _unit(np.cross(a, t))
    return R.from_rotvec(n * th).as_matrix()


def _kabsch(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Least-squares rotation taking each a_i onto t_i (two independent
    pairs; the cross-product pair fixes handedness)."""
    (a1, t1), (a2, t2) = pairs
    M = (np.outer(t1, a1) + np.outer(t2, a2)
         + np.outer(np.cross(t1, t2), np.cross(a1, a2)))
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)) or 1.0)])
    return U @ D @ Vt


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

def _canonical_w(w_obj: str, mover: str | None, symbols: dict[str, Symbols],
                 body_poses: dict[str, np.ndarray], w_point: np.ndarray,
                 slot: str) -> tuple[Frame, str]:
    """The w-owning object's canonical frame anchored at w_point (body
    coords): z = up_axis, x = front_axis, else the fallback ladder in
    the module docstring. Returns (Frame, route note)."""
    sym = symbols[w_obj]
    if _UP not in sym.axes:
        raise CompileError(slot, f"{w_obj!r} has no {_UP}: no canonical "
                                 "frame to root w on")
    up_b = _unit(sym.axes[_UP])
    R_o = body_poses[w_obj][:3, :3]
    z_w = R_o @ up_b
    if _FRONT in sym.axes:
        return (Frame(name=f"{w_obj}.canonical",
                      point=np.asarray(w_point, float).reshape(3).copy(),
                      axis=up_b.copy(), secondary=_unit(sym.axes[_FRONT])),
                f"{w_obj}.{_FRONT}")
    # no confident front: borrow the azimuth from the mover's lateral so a
    # tip of its front toward the vertical rotates about w.x (roll), then
    # from world x/y
    x_world, route = None, ""
    if mover is not None and _FRONT in symbols[mover].axes:
        f = body_poses[mover][:3, :3] @ _unit(symbols[mover].axes[_FRONT])
        f_h = f - float(f @ z_w) * z_w
        if np.linalg.norm(f_h) > np.sin(ALIGN_TOL_RAD):
            x_world = _unit(np.cross(z_w, _unit(f_h)))
            route = f"{mover} lateral (up x front) at entry"
    if x_world is None:
        for e, nm in ((np.array([1.0, 0, 0]), "world.x"),
                      (np.array([0, 1.0, 0]), "world.y")):
            e_h = e - float(e @ z_w) * z_w
            if np.linalg.norm(e_h) > np.sin(ALIGN_TOL_RAD):
                x_world, route = _unit(e_h), nm
                break
    return (Frame(name=f"{w_obj}.canonical(fallback)",
                  point=np.asarray(w_point, float).reshape(3).copy(),
                  axis=up_b.copy(), secondary=R_o.T @ x_world), route)


def _solve_rows(rel: list, R0_w: np.ndarray) -> tuple[np.ndarray, list]:
    """rel: [(slot, RotRow, a_entry_world, r_world)] relation rows. Returns
    (R_delta world rotation entry -> goal, fixed rows [(row, slot, tol)])
    per the v2 rule table."""
    if not rel:
        return np.eye(3), []
    kinds = [r.relation for _, r, _, _ in rel]
    if len(rel) == 1:
        slot, r, a, d = rel[0]
        if r.relation == "perpendicular":
            n = np.cross(a, d)
            if np.linalg.norm(n) < np.sin(ALIGN_TOL_RAD):
                raise CompileError(slot, (
                    "axis is (anti)parallel to the reference at entry; the "
                    "direction to tip it into the perpendicular plane is "
                    "ambiguous — relate an axis that is already off the "
                    "reference, or split the stage"))
            Rd = R.from_rotvec(_unit(n) * (np.arccos(np.clip(a @ d, -1, 1))
                                           - np.pi / 2)).as_matrix()
            n_fix = R0_w.T @ np.cross(d, Rd @ a)
            k, _ = _basis_row(n_fix, slot, "the constrained tilt axis")
            return Rd, [(k, slot, r.tol)]
        t = d if r.relation == "parallel" else -d
        Rd = _rot_onto(a, t, slot)
        k, _ = _basis_row(R0_w.T @ t, slot, f"reference {r.reference!r}")
        return Rd, [(i, slot, r.tol) for i in range(3) if i != k]
    if len(rel) == 2 and all(k != "perpendicular" for k in kinds):
        (s1, r1, a1, d1), (s2, r2, a2, d2) = rel
        t1 = d1 if r1.relation == "parallel" else -d1
        t2 = d2 if r2.relation == "parallel" else -d2
        if np.linalg.norm(np.cross(a1, a2)) < np.sin(ALIGN_TOL_RAD):
            raise CompileError(s2, "second relation row constrains an axis "
                                   "parallel to the first's; the pair is not "
                                   "independent — drop one row")
        ang_a = np.arccos(np.clip(a1 @ a2, -1, 1))
        ang_t = np.arccos(np.clip(t1 @ t2, -1, 1))
        if abs(ang_a - ang_t) > ALIGN_TOL_RAD:
            raise CompileError(s2, (
                f"rows are mutually inconsistent: the axes are "
                f"{np.degrees(ang_a):.0f} deg apart but the references "
                f"{np.degrees(ang_t):.0f} deg apart; no attitude satisfies "
                "both — drop or restate one"))
        Rd = _kabsch([(a1, t1), (a2, t2)])
        fixed = []
        for s, r, t in ((s1, r1, t1), (s2, r2, t2)):
            k, _ = _basis_row(R0_w.T @ t, s, f"reference {r.reference!r}")
            fixed += [(i, s, r.tol) for i in range(3) if i != k]
        return Rd, fixed
    raise CompileError(rel[-1][0], (
        f"{len(rel)} relation rows ({kinds}) are outside the rule table: "
        "one row of any relation, or two parallel/antiparallel rows with "
        "independent axes (which fully determine the attitude). Drop the "
        "extra row or split the stage"))


def compile_stage(emission: StageEmission,
                  symbols: dict[str, Symbols],
                  body_poses: dict[str, np.ndarray],
                  w_point: np.ndarray,
                  e_point: np.ndarray | None = None) -> CompiledStage:
    """Ground one StageEmission. `body_poses` are the world body poses
    to root frames on — pass the ENTRY poses; the w frame (including a
    fallback azimuth) and the goal attitude are frozen on them.
    `w_point`: the interaction point call #2 selected on the w-owning
    object (the passive, or the grasped object when there is no
    passive), in its body coords — w's origin. `e_point`: the mover's
    selected feature point (its body coords) that Tw_e pins at w; None
    -> the mover's body origin. Must be None when the mover is the
    gripper (no passive): Tw_e = I then."""
    notes: list[str] = []
    w_obj = emission.passive or emission.active
    mover = emission.active if emission.passive else None   # None: gripper
    w_slot = "passive" if emission.passive else "active"
    if w_obj not in symbols:
        raise CompileError(w_slot, f"unknown object {w_obj!r}")
    if w_obj not in body_poses:
        raise CompileError(w_slot, f"no body pose supplied for {w_obj!r}")
    if mover is not None and mover not in symbols:
        raise CompileError("active", f"unknown object {mover!r}")
    if mover is not None and mover not in body_poses:
        raise CompileError("active", (
            f"no body pose for the active object {mover!r} — the entry "
            "attitude roots the goal attitude and Tw_e"))
    if mover is None and e_point is not None:
        raise ValueError("e_point given but the moving frame is the gripper")

    # -- w frame ---------------------------------------------------------
    w_frame, route = _canonical_w(w_obj, mover, symbols, body_poses,
                                  np.asarray(w_point, float), w_slot)
    T0_w = body_poses[w_obj] @ w_frame.T()
    R0_w = T0_w[:3, :3]
    Tf_inv = np.linalg.inv(w_frame.T())
    notes.append(f"w = {w_obj} canonical frame at "
                 f"{np.round(w_frame.point, 4).tolist()}: z = {_UP}, "
                 f"x = {route}; rooted on body_poses[{w_obj!r}]")
    R_entry = body_poses[mover][:3, :3] if mover is not None else None
    e_xyz = (np.zeros(3) if e_point is None
             else np.asarray(e_point, float).reshape(3))

    def _anchor_off(qualified: str, slot: str) -> np.ndarray:
        obj, name = _split(qualified, slot)
        if obj != w_obj:
            raise CompileError(slot, (
                f"anchor {qualified!r} is not on the w-owning object "
                f"{w_obj!r}; it would not be static in w — anchor on a "
                f"{w_obj} point instead"))
        if name not in symbols[obj].points:
            raise CompileError(slot, f"unknown point {qualified!r}")
        p = symbols[obj].points[name]
        return (Tf_inv @ np.append(p, 1.0))[:3]

    def _sigma(qualified: str) -> float:
        obj, name = qualified.split(".", 1)
        if obj == "world":
            return 0.0
        return float(symbols[obj].sigmas.get(f"axes.{name}", 0.0))

    sigma_up_w = _sigma(f"{w_obj}.{_UP}")      # w's own tilt uncertainty
    quantities = {f"{o}.{q}": v for o, s in symbols.items()
                  for q, v in s.quantities.items()}

    def _compile_spec(spec: TSRSpec, ctx: str) -> tuple[np.ndarray, np.ndarray]:
        box = _Box()

        # ---- rotation rows: collect relation rows, solve the goal attitude
        rel: list = []
        for i, r in enumerate(spec.rot):
            slot = f"{emission.name}.{ctx}.rot[{i}]"
            if r.relation == "free":
                continue                      # rows default free in v2
            if mover is None:
                raise CompileError(slot, (
                    "the moving frame is the gripper (no passive object), "
                    "which has no grounded axes, so relation rows cannot "
                    "be grounded; write rot as \"free\" or per-row "
                    "{\"relation\": \"free\", \"row\": ...}"))
            a_own, a_name = _split(r.axis, slot)
            if a_own != mover:
                raise CompileError(slot, (
                    f"constrained axis {r.axis!r} must belong to the "
                    f"active object {mover!r} (only the active object "
                    "displaces)"))
            if a_name not in symbols[mover].axes:
                raise CompileError(slot, f"unknown axis {r.axis!r}")
            ref_own, _ = _split(r.reference, slot)
            if ref_own == mover:
                raise CompileError(slot, (
                    f"reference {r.reference!r} is on the active object; "
                    "it displaces with the constrained axis and the "
                    "relation is degenerate"))
            if ref_own not in ("world", w_obj):
                raise CompileError(slot, (
                    f"reference {r.reference!r} is neither world.z nor a "
                    f"{w_obj} axis; it is not static in w"))
            a = R_entry @ _unit(symbols[mover].axes[a_name])
            d = _unit(_axis_dir_world(r.reference, symbols, body_poses, slot))
            rel.append((slot, r, a, d))
        Rd, fixed = _solve_rows(rel, R0_w)
        R_goal = Rd @ R_entry if mover is not None else None

        half: dict[int, tuple[float, str]] = {}
        for k, slot, tol in fixed:
            r = next(rr for s, rr, _, _ in rel if s == slot)
            sig = np.sqrt(_sigma(r.axis) ** 2 + _sigma(r.reference) ** 2
                          + (sigma_up_w ** 2 if k < 2 else 0.0))
            h = max(ROT_TOL_RAD[tol], SIGMA_K * np.deg2rad(sig))
            if k not in half or h < half[k][0]:
                half[k] = (h, slot)
        rows: dict[int, tuple[float, float, str]] = {
            k: (-h, h, s) for k, (h, s) in half.items()}
        if rel:
            notes.append(f"{emission.name}.{ctx}: goal attitude from "
                         f"{len(rel)} relation row(s); fixed "
                         f"{[('roll', 'pitch', 'yaw')[k] for k in sorted(half)]}"
                         f", rest free")

        # ---- path corridor: the entry -> goal sweep. The entry's
        # displacement about the goal zero, in w, is Rd^T; its RPY rows
        # must fit the box the path will be checked against.
        if ctx == "path" and rel:
            Rd_w = R0_w.T @ Rd.T @ R0_w
            rv = R.from_matrix(Rd_w).as_rotvec()
            th = float(np.linalg.norm(rv))
            with warnings.catch_warnings():       # gimbal lock at pitch 90:
                warnings.simplefilter("ignore")   # never a corridor anyway
                rpy = (R.from_matrix(Rd_w).as_euler("xyz") if th > 1e-6
                       else np.zeros(3))
            fits = {k: abs(float(rpy[k])) <= h for k, (h, _) in half.items()}
            if th > 1e-6 and not all(fits.values()):
                n_w = rv / th
                k = int(np.argmax(np.abs(n_w)))
                aligned = np.arccos(min(abs(float(n_w[k])), 1.0)) <= ALIGN_TOL_RAD
                if (aligned and k in (0, 2)
                        and all(ok for j, ok in fits.items() if j != k)):
                    h = half[k][0] if k in half else max(
                        ROT_TOL_RAD[r.tol] for _, r, _, _ in rel)
                    ent = float(rpy[k])                # entry displacement
                    rows[k] = (min(0.0, ent) - h, max(0.0, ent) + h,
                               rel[0][0])
                    notes.append(f"{emission.name}.path: corridor on "
                                 f"{('roll', 'pitch', 'yaw')[k]} from entry "
                                 f"({np.degrees(ent):.0f} deg) to goal")
                else:
                    rows = {}
                    notes.append(f"{emission.name}.path: entry->goal "
                                 f"rotation (axis in w {np.round(n_w, 2).tolist()}) "
                                 "is not a single corridor about w.x or w.z "
                                 "— rotation rows free, translation carries "
                                 "the path")
        for k, (lo, hi, slot) in rows.items():
            box.narrow(3 + k, lo, hi, slot)

        if mover is None:
            Tw_e = np.eye(4)
        else:
            Tw_e = np.linalg.inv(make_pose(e_xyz, R_goal.T @ R0_w))

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

        return box.finish(trans_default=FREE_TRANS, rot_default=FREE_ROT), Tw_e

    # compile both TSRs before raising so one rejection carries every
    # failed slot: a retry budget spent one slot per attempt on the same
    # mistake in path and subgoal is a budget wasted
    errs: list[CompileError] = []
    tsrs: dict[str, TSR] = {}
    for ctx, spec in (("path", emission.path_tsr),
                      ("subgoal", emission.subgoal_tsr)):
        try:
            Bw, Tw_e = _compile_spec(spec, ctx)
        except CompileError as e:
            errs.append(e)
            continue
        tsrs[ctx] = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                        name=f"{emission.name}/{ctx}(emitted)")
    if errs:
        raise CompileError(errs[0].slot, errs[0].reason, tuple(errs[1:]))

    return CompiledStage(stage=emission.stage, name=emission.name,
                         path=tsrs["path"], subgoal=tsrs["subgoal"],
                         w_frame=w_frame, notes=tuple(notes))
