"""VLM client and typed I/O layer — the shared substrate for the five
discrete VLM touchpoints:

    #1  plan_stages         stage plan / part naming
    #2  select_point_axis   interaction point + axis (MOKA-style marks)
    #3  emit_constraints    per-stage TSR schema filling
    #4  critique_preview    render-and-check critic verdict
    #5  repair              typed failure -> symbolic repair action

Architectural invariant (the one the whole pipeline is built around): the
VLM never emits a metric quantity. Every response is parsed against a
closed, licensed vocabulary — mark IDs, grounded point/axis names,
relation tokens, enum tolerances, symbolic bound expressions — and
anything outside it is REJECTED, not coerced. Concretely:

  * rotational structure admits only relation tokens + tolerance enums;
    a numeric literal anywhere in a rotation slot is a hard parse
    rejection (the IKER/CoPa/SoFar SO(3) failure mode, blocked at the
    API level, not by prompt hope);
  * translational bounds admit symbolic terms (above/centered/along/...)
    with clearance + slack enums, plus arithmetic EXPRESSIONS over
    grounded quantity symbols (rim_radius, ...); bare numeric literals
    inside expressions are permitted but FLAGGED and logged — the
    documented soft boundary of the bound vocabulary;
  * signs are the VLM's only geometric contribution: a binary +/- token.

Single-source vocabulary: one Vocabulary object is built from the same
artifacts the geometric pipeline uses (frames.json Symbols tables, the
candidates.json pool / its menu subset) and serves BOTH as the menu text
rendered into the prompt and as the accept set the validator enforces.
The prompt and the parser therefore cannot drift apart.

Symbolic-only boundary: enum -> numeric tables (tight -> +-5 deg,
clearance small -> mm, ...) deliberately do NOT live in this module.
They belong to the grounding compiler (compile_tsr), which resolves a
validated StageEmission to numeric (T0_w, Tw_e, B^w). This file's typed
outputs are the compiler's input contract; the CriticVerdict / Repair
schemas below are likewise the contract preview_stage_goals.py and the
typed-failure router must produce/consume.

Retry contract: parse rejections are fed back to the model verbatim as a
follow-up turn (the rejection names exactly which token was outside the
license) with a bounded budget; transport errors (429/5xx) back off
independently. Sampling is temperature 0 — selection variance is an
ablation axis, not a default.

Transport: stdlib urllib POST to the Anthropic Messages API; the
transport is an injectable callable so tests run offline and the
official SDK is a drop-in swap. API key from ANTHROPIC_API_KEY.

numpy-free, simulator-free: pure stdlib on purpose — this module must be
importable by scripts, the preview loop, and the hardware orchestrator
alike.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------- constants

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000
TEMPERATURE = 0.0        # deterministic-as-available; variance is an ablation
MAX_PARSE_RETRIES = 2    # rejections fed back; attempts = 1 + retries
HTTP_RETRIES = 3         # 429/5xx, exponential backoff
HTTP_BACKOFF_S = 2.0

# ------------------------------------------------- closed token alphabets
# These are the LICENSED discrete alphabets. The compiler owns their
# numeric meaning; this layer owns only membership.

ROT_RELATIONS = ("parallel", "antiparallel", "perpendicular")
ROT_TOLS = ("tight", "moderate", "loose")
CLEARANCES = ("contact", "small", "medium", "large")
SLACKS = ("snug", "moderate", "loose")
SIGNS = ("+", "-")
ROT_ROWS = ("roll", "pitch", "yaw")
TRANS_ROWS = ("x", "y", "z")
WORLD_AXES = ("world.x", "world.y", "world.z")
TRANS_TERMS = ("free", "above", "below", "centered", "along", "inside",
               "expr")

# Typed failures the planner/preview loop may raise to touchpoint #5
# (superset of the three current SystemExit dead-ends; the router maps
# onto these names).
FAILURE_TYPES = ("empty_goal_intersection", "no_feasible_ik",
                 "projection_divergence", "planner_timeout",
                 "lookahead_starved", "preview_rejected",
                 "grasp_filter_empty")

# Symbolic repair vocabulary (one-token diffs; auditable, cheap to
# re-compile).
REPAIR_ACTIONS = ("relax_tolerance", "free_dof", "reselect_point",
                  "reselect_axis", "insert_stage", "reorder_stages",
                  "widen_clearance", "widen_slack")

CRITIC_VERDICTS = ("accept", "reject")


# ------------------------------------------------------------ exceptions

class VLMError(RuntimeError):
    """Transport-level failure or retry budget exhausted."""


class ParseRejection(ValueError):
    """Response fell outside the licensed vocabulary. `reason` is sent
    back to the model verbatim on retry, so make it name the offending
    token and the accept set."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------ vocabulary

@dataclass(frozen=True)
class Vocabulary:
    """The single source of truth for what the VLM may say.

    Built from the same artifacts the geometric pipeline consumes:
    frames.json Symbols tables (points/axes/quantities per object) and
    the candidate menu (pool IDs offered for touchpoint #2). Rendered
    into prompts by describe_*(); enforced by the validators below.
    """

    # object -> table of names, e.g. {"teapot": {"points": (...),
    # "axes": (...), "quantities": (...)}}
    objects: dict[str, dict[str, tuple[str, ...]]]
    # candidate menu for touchpoint #2: id -> short human tag
    menu: dict[int, str] = field(default_factory=dict)
    # object marks for touchpoint #1 (mark-addressed mode): id -> one-line
    # description of the mark (bbox/area; NEVER the object name)
    marks: dict[int, str] = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @staticmethod
    def from_symbols(symbols: dict[str, "object"],
                     menu: dict[int, str] | None = None) -> "Vocabulary":
        """Build from manip_sim.frames.Symbols instances (duck-typed:
        needs .points/.axes/.quantities dicts) keyed by object name."""
        objects = {
            name: {
                "points": tuple(sorted(sym.points)),
                "axes": tuple(sorted(sym.axes)),
                "quantities": tuple(sorted(sym.quantities)),
            }
            for name, sym in symbols.items()
        }
        return Vocabulary(objects=objects, menu=dict(menu or {}))

    @staticmethod
    def from_asset_dirs(asset_dirs: dict[str, Path],
                        menu: dict[int, str] | None = None) -> "Vocabulary":
        """Build straight from frames.json files (no numpy import)."""
        objects: dict[str, dict[str, tuple[str, ...]]] = {}
        for name, d in asset_dirs.items():
            spec = json.loads((Path(d) / "frames.json").read_text())
            objects[name] = {
                "points": tuple(sorted(spec.get("points", {}))),
                "axes": tuple(sorted(spec.get("axes", {}))),
                "quantities": tuple(sorted(spec.get("quantities", {}))),
            }
        return Vocabulary(objects=objects, menu=dict(menu or {}))

    @staticmethod
    def from_marks(markset) -> "Vocabulary":
        """Touchpoint #1, mark-addressed: the accept set is the mark IDs
        on the scene image; no object names, no symbols (nothing is
        registered before the model says what matters)."""
        return Vocabulary(objects={}, marks={
            i: f"bbox {list(m.bbox)}, area {m.area} px"
            for i, m in sorted(markset.marks.items())})

    # -- membership ------------------------------------------------------

    def _qualified(self, table: str) -> set[str]:
        return {f"{o}.{n}" for o, t in self.objects.items()
                for n in t[table]}

    def point_names(self) -> set[str]:
        return self._qualified("points")

    def axis_names(self) -> set[str]:
        return self._qualified("axes") | set(WORLD_AXES)

    def quantity_names(self) -> set[str]:
        return self._qualified("quantities")

    # -- prompt rendering --------------------------------------------------

    def describe_symbols(self) -> str:
        lines = []
        for o, t in sorted(self.objects.items()):
            lines.append(f"object `{o}`:")
            lines.append(f"  points:     {', '.join(t['points']) or '(none)'}")
            lines.append(f"  axes:       {', '.join(t['axes']) or '(none)'}")
            lines.append(
                f"  quantities: {', '.join(t['quantities']) or '(none)'}")
        lines.append(f"world axes: {', '.join(WORLD_AXES)}")
        return "\n".join(lines)

    def describe_marks(self) -> str:
        return "\n".join(f"  mark {i}: {d}" for i, d in sorted(self.marks.items()))

    def describe_menu(self) -> str:
        if not self.menu:
            return "(no candidate menu)"
        return "\n".join(f"  [{i}] {tag}"
                         for i, tag in sorted(self.menu.items()))


# --------------------------------------------------------- typed outputs

@dataclass(frozen=True)
class StageSpec:
    """#1 output element. `active`/`passive` are object HANDLES: in
    mark-addressed mode the model emits mark IDs and the parser derives
    a handle from its free-text label (StagePlan.objects keeps the
    id<->handle map); in text-only mode handles are the given names.
    Part names are FREE TEXT by design — semantic identity seeds for
    GroundedSAM, the one licensed use of unconstrained strings (2D
    touches identity, never coordinates) — keyed by the handle whose
    crop gets the seed."""
    index: int
    name: str
    active: str
    passive: str | None           # None for single-object stages
    parts: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def objects(self) -> tuple[str, ...]:
        return (self.active,) + ((self.passive,) if self.passive else ())

    def all_parts(self) -> tuple[str, ...]:
        return tuple(p for ps in self.parts.values() for p in ps)


@dataclass(frozen=True)
class ObjectRef:
    """What call #1 said about one scene object. `mark` is None in
    text-only mode. `label` is the model's free-text name — the naming
    seed for mesh generation/registration on hardware."""
    handle: str
    label: str
    mark: int | None = None


@dataclass(frozen=True)
class StagePlan:
    task: str
    stages: tuple[StageSpec, ...]
    objects: dict[str, ObjectRef] = field(default_factory=dict)

    def handle_of_mark(self) -> dict[int, str]:
        return {o.mark: h for h, o in self.objects.items() if o.mark is not None}

    def relabel(self, names: dict[str, str]) -> "StagePlan":
        """Rewrite handles (e.g. to sim ground-truth names, or to
        registered asset names). Unmapped handles are kept."""
        f = lambda h: names.get(h, h) if h is not None else None
        stages = tuple(StageSpec(
            index=s.index, name=s.name, active=f(s.active), passive=f(s.passive),
            parts={f(h): ps for h, ps in s.parts.items()}) for s in self.stages)
        objects = {f(h): ObjectRef(handle=f(h), label=o.label, mark=o.mark)
                   for h, o in self.objects.items()}
        return StagePlan(task=self.task, stages=stages, objects=objects)


@dataclass(frozen=True)
class PointAxisSelection:
    """#2 output: mark ID from the offered menu, axis by grounded name,
    sign as the +-1 disambiguation, optional secondary axis reference
    for the Gram-Schmidt x."""
    candidate_id: int
    axis: str              # qualified: "teapot.pour_axis"
    sign: str              # "+" | "-"
    secondary: str | None  # qualified axis name or None (body -z default)
    rationale: str         # free text, logged only — never parsed


@dataclass(frozen=True)
class RotRow:
    """One rotational structure element: relation between a grounded
    axis and a reference axis, with an enum tolerance — OR an explicit
    per-row free declaration."""
    axis: str | None       # qualified axis, None when row-free form
    relation: str          # ROT_RELATIONS or "free"
    reference: str | None  # qualified axis / world axis, None when free
    tol: str | None        # ROT_TOLS, None when free
    row: str | None = None # ROT_ROWS for the row-free form


@dataclass(frozen=True)
class TransTerm:
    """One translational bound term from the closed term vocabulary.
    `expr_lo`/`expr_hi` (term == "expr") are validated arithmetic over
    grounded quantity symbols; numeric literals inside them are legal
    but reported in `flags`."""
    term: str                     # TRANS_TERMS
    anchor: str | None = None     # qualified point (above/below/centered/inside)
    axis: str | None = None       # qualified axis (along)
    sign: str | None = None       # SIGNS (along)
    clearance: str | None = None  # CLEARANCES
    slack: str | None = None      # SLACKS
    tol: str | None = None        # SLACKS reused for lateral tolerance
    row: str | None = None        # TRANS_ROWS (expr)
    expr_lo: str | None = None
    expr_hi: str | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TSRSpec:
    rot: tuple[RotRow, ...]
    trans: tuple[TransTerm, ...]   # (TransTerm("free"),) for free


@dataclass(frozen=True)
class StageEmission:
    """#3 output: the symbolic schema for one stage — compile_tsr's
    input contract. `w_*` compose Symbols.frame(origin, axis)."""
    stage: int
    name: str
    active: str
    passive: str | None
    w_origin: str          # qualified point on the PASSIVE object
    w_axis: str            # qualified axis anchoring w's +z
    path_tsr: TSRSpec
    subgoal_tsr: TSRSpec
    verify: str            # free-text predicate for the render check


def emission_from_json(d: dict) -> StageEmission:
    """Inverse of dataclasses.asdict on a StageEmission (the emissions
    artifact emit_constraints.py writes)."""
    def spec(t):
        return TSRSpec(rot=tuple(RotRow(**r) for r in t["rot"]),
                       trans=tuple(TransTerm(**{**x, "flags": tuple(x.get("flags", ()))})
                                   for x in t["trans"]))
    return StageEmission(stage=int(d["stage"]), name=d["name"], active=d["active"],
                         passive=d.get("passive"), w_origin=d["w_origin"],
                         w_axis=d["w_axis"], path_tsr=spec(d["path_tsr"]),
                         subgoal_tsr=spec(d["subgoal_tsr"]), verify=d.get("verify", ""))


def load_emissions(path) -> dict[str, StageEmission]:
    """Role-keyed emissions from an emit_constraints.py artifact. Refuses
    artifacts whose compile gate failed — the planner must never receive
    an emission the compiler already rejected."""
    doc = json.loads(Path(path).read_text())
    bad = [c["stage"] for c in doc.get("compiled", []) if not c["grounded"]]
    if bad:
        raise SystemExit(f"[vlm] {path}: compile gate failed for {bad}; "
                         "refusing to plan on it")
    roles = doc.get("roles")
    if not roles:
        raise SystemExit(f"[vlm] {path} has no role keys; re-run emit_constraints.py")
    return {r: emission_from_json(e) for r, e in zip(roles, doc["emissions"])}


@dataclass(frozen=True)
class CriticEdit:
    """A one-token diff against a named authored slot. `target` is the
    report.json `authored_in` pointer (e.g. 'stage2.subgoal.trans[1]');
    membership of the pointer set is checked by the CALLER against the
    live report — this layer checks token legality only."""
    target: str
    action: str            # REPAIR_ACTIONS subset applicable to edits
    token: str | None      # replacement enum token where applicable


@dataclass(frozen=True)
class CriticVerdict:
    verdict: str                     # accept | reject
    edits: tuple[CriticEdit, ...]    # non-empty iff reject
    diagnosis: str                   # free text, logged


@dataclass(frozen=True)
class RepairAction:
    action: str            # REPAIR_ACTIONS
    target: str | None     # named slot / stage the action applies to
    token: str | None      # new enum token where applicable
    rationale: str


# --------------------------------------------- expression micro-grammar
# expr := term (('+'|'-') term)*
# term := factor (('*'|'/') factor)*
# factor := NUMBER | IDENT | '-' factor | '(' expr ')'
# IDENT must be a licensed quantity symbol; NUMBER is legal but flagged.

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*|\.\d+)|(?P<ident>[A-Za-z_][\w.]*)"
    r"|(?P<op>[+\-*/()]))")


def _tokenize_expr(s: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(s):
        m = _TOKEN_RE.match(s, i)
        if not m or m.end() == i:
            raise ParseRejection(
                f"illegal character in bound expression at {s[i:]!r}")
        i = m.end()
        for kind in ("num", "ident", "op"):
            v = m.group(kind)
            if v is not None:
                out.append((kind, v))
                break
    return out


def validate_expr(s: str, quantities: set[str]) -> tuple[str, ...]:
    """Validate an arithmetic bound expression. Returns flags (numeric
    literals encountered); raises ParseRejection on illegal identifiers
    or malformed structure."""
    toks = _tokenize_expr(s)
    if not toks:
        raise ParseRejection("empty bound expression")
    flags: list[str] = []
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None)

    def eat(kind=None, val=None):
        nonlocal pos
        k, v = peek()
        if k is None or (kind and k != kind) or (val and v != val):
            raise ParseRejection(
                f"malformed bound expression {s!r} near token {pos}")
        pos += 1
        return v

    def factor():
        k, v = peek()
        if k == "num":
            eat("num")
            flags.append(f"numeric literal {v!r} in expression {s!r}")
        elif k == "ident":
            eat("ident")
            if v not in quantities:
                raise ParseRejection(
                    f"unknown quantity symbol {v!r}; licensed quantities: "
                    f"{sorted(quantities) or '(none)'}")
        elif k == "op" and v == "-":
            eat("op", "-"); factor()
        elif k == "op" and v == "(":
            eat("op", "("); expr(); eat("op", ")")
        else:
            raise ParseRejection(
                f"malformed bound expression {s!r} near token {pos}")

    def term():
        factor()
        while peek() == ("op", "*") or peek() == ("op", "/"):
            eat("op"); factor()

    def expr():
        term()
        while peek() == ("op", "+") or peek() == ("op", "-"):
            eat("op"); term()

    expr()
    if pos != len(toks):
        raise ParseRejection(f"trailing tokens in bound expression {s!r}")
    return tuple(flags)


# ------------------------------------------------------- parse utilities

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n?```$")


def _load_json(text: str):
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseRejection(
            f"response is not valid JSON ({e}); respond with a single "
            "JSON object and nothing else") from None


def _need(obj: dict, key: str, typ, ctx: str):
    if key not in obj:
        raise ParseRejection(f"missing key {key!r} in {ctx}")
    v = obj[key]
    if not isinstance(v, typ):
        raise ParseRejection(
            f"key {key!r} in {ctx} must be {getattr(typ, '__name__', typ)}")
    return v


def _enum(v, allowed, ctx: str):
    if v not in allowed:
        raise ParseRejection(
            f"{v!r} is not a licensed token for {ctx}; allowed: "
            f"{list(allowed)}")
    return v


def _no_numbers(obj, ctx: str):
    """Hard invariant for rotation slots: reject numerics anywhere."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        raise ParseRejection(
            f"numeric value {obj!r} in {ctx}: rotational structure admits "
            "only relation tokens and tolerance enums, never numbers")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _no_numbers(v, f"{ctx}.{k}")
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_numbers(v, f"{ctx}[{i}]")


# ------------------------------------------------- per-touchpoint parsing

def _slug(label: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return out or "object"


def _parse_parts(raw, allowed: tuple[str, ...], ctx: str,
                 to_handle=lambda k: k) -> dict[str, tuple[str, ...]]:
    """`parts` is keyed by the stage's objects (mark id or name as the
    model addressed them); values are non-empty free-text lists."""
    if not isinstance(raw, dict):
        raise ParseRejection(f"{ctx}.parts must be an object keyed by the "
                             f"stage's active/passive")
    out = {}
    for k, v in raw.items():
        h = to_handle(k)
        if h not in allowed:
            raise ParseRejection(f"{ctx}.parts key {k!r} is not this stage's "
                                 f"active/passive ({list(allowed)})")
        if not isinstance(v, list) or not all(isinstance(p, str) and p.strip() for p in v):
            raise ParseRejection(f"{ctx}.parts[{k!r}] must be a list of non-empty strings")
        out[h] = tuple(v)
    return out


def parse_stage_plan(raw: str, vocab: Vocabulary, task: str) -> StagePlan:
    """Mark-addressed when `vocab.marks` is set (active/passive/parts keys
    are mark IDs, plus an `objects` map id -> label); text-only otherwise
    (names from `vocab.objects`)."""
    doc = _load_json(raw)
    stages_raw = _need(doc, "stages", list, "stage plan")
    if not stages_raw:
        raise ParseRejection("stage plan has no stages")

    if vocab.marks:
        objs_raw = _need(doc, "objects", dict, "stage plan")
        refs: dict[str, ObjectRef] = {}
        handle_of: dict[int, str] = {}
        for k, label in objs_raw.items():
            try:
                mid = int(k)
            except (TypeError, ValueError):
                raise ParseRejection(f"objects key {k!r} is not a mark id")
            if mid not in vocab.marks:
                raise ParseRejection(f"mark {mid} is not on the image; offered "
                                     f"marks: {sorted(vocab.marks)}")
            if not isinstance(label, str) or not label.strip():
                raise ParseRejection(f"objects[{k}] must be a non-empty label")
            h = base = _slug(label)
            n = 2
            while h in refs:
                h, n = f"{base}_{n}", n + 1
            refs[h] = ObjectRef(handle=h, label=label.strip(), mark=mid)
            handle_of[mid] = h

        def obj(v, ctx):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ParseRejection(f"{ctx} must be a mark id (int)")
            if v not in handle_of:
                raise ParseRejection(f"{ctx}: mark {v} is not declared in "
                                     f"`objects` ({sorted(handle_of)})")
            return handle_of[v]

        def key(k):
            try:
                return handle_of.get(int(k), k)
            except (TypeError, ValueError):
                return k
    else:
        names = set(vocab.objects)
        refs = {n: ObjectRef(handle=n, label=n, mark=None) for n in names}

        def obj(v, ctx):
            return _enum(v, names, ctx)

        def key(k):
            return k

    stages = []
    for i, st in enumerate(stages_raw):
        if not isinstance(st, dict):
            raise ParseRejection(f"stage[{i}] must be an object")
        name = _need(st, "name", str, f"stage[{i}]")
        active = obj(st.get("active"), f"stage[{i}].active")
        passive = st.get("passive")
        if passive is not None:
            passive = obj(passive, f"stage[{i}].passive")
        if passive == active:
            raise ParseRejection(f"stage[{i}]: passive equals active")
        allowed = (active,) + ((passive,) if passive else ())
        parts = _parse_parts(_need(st, "parts", dict, f"stage[{i}]"),
                             allowed, f"stage[{i}]", key)
        stages.append(StageSpec(index=i, name=name, active=active,
                                passive=passive, parts=parts))
    used = {h for s in stages for h in s.objects()}
    return StagePlan(task=task, stages=tuple(stages),
                     objects={h: r for h, r in refs.items() if h in used})


def parse_point_axis(raw: str, vocab: Vocabulary) -> PointAxisSelection:
    doc = _load_json(raw)
    cid = _need(doc, "candidate_id", int, "selection")
    if cid not in vocab.menu:
        raise ParseRejection(
            f"candidate_id {cid} is not on the offered menu; offered IDs: "
            f"{sorted(vocab.menu)}")
    axis = _enum(_need(doc, "axis", str, "selection"), vocab.axis_names(),
                 "selection.axis")
    sign = _enum(_need(doc, "sign", str, "selection"), SIGNS,
                 "selection.sign")
    secondary = doc.get("secondary")
    if secondary is not None:
        secondary = _enum(secondary, vocab.axis_names(),
                          "selection.secondary")
    rationale = doc.get("rationale", "")
    if not isinstance(rationale, str):
        raise ParseRejection("selection.rationale must be a string")
    return PointAxisSelection(candidate_id=cid, axis=axis, sign=sign,
                              secondary=secondary, rationale=rationale)


def _parse_rot_rows(rows, vocab: Vocabulary, ctx: str) -> tuple[RotRow, ...]:
    if rows == "free":
        return tuple(RotRow(axis=None, relation="free", reference=None,
                            tol=None, row=r) for r in ROT_ROWS)
    if not isinstance(rows, list):
        raise ParseRejection(f"{ctx}.rot must be 'free' or a list")
    _no_numbers(rows, f"{ctx}.rot")
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ParseRejection(f"{ctx}.rot[{i}] must be an object")
        c = f"{ctx}.rot[{i}]"
        if r.get("relation") == "free":
            row = _enum(_need(r, "row", str, c), ROT_ROWS, f"{c}.row")
            out.append(RotRow(axis=None, relation="free", reference=None,
                              tol=None, row=row))
            continue
        axis = _enum(_need(r, "axis", str, c), vocab.axis_names(),
                     f"{c}.axis")
        rel = _enum(_need(r, "relation", str, c), ROT_RELATIONS,
                    f"{c}.relation")
        ref = _enum(_need(r, "reference", str, c), vocab.axis_names(),
                    f"{c}.reference")
        tol = _enum(_need(r, "tol", str, c), ROT_TOLS, f"{c}.tol")
        out.append(RotRow(axis=axis, relation=rel, reference=ref, tol=tol))
    return tuple(out)


def _parse_trans_terms(terms, vocab: Vocabulary,
                       ctx: str) -> tuple[TransTerm, ...]:
    if terms == "free":
        return (TransTerm(term="free"),)
    if not isinstance(terms, list):
        raise ParseRejection(f"{ctx}.trans must be 'free' or a list")
    points, axes, quants = (vocab.point_names(), vocab.axis_names(),
                            vocab.quantity_names())
    out = []
    for i, t in enumerate(terms):
        if not isinstance(t, dict):
            raise ParseRejection(f"{ctx}.trans[{i}] must be an object")
        c = f"{ctx}.trans[{i}]"
        term = _enum(_need(t, "term", str, c), TRANS_TERMS, f"{c}.term")
        if term == "free":
            out.append(TransTerm(term="free"))
        elif term in ("above", "below"):
            out.append(TransTerm(
                term=term,
                anchor=_enum(_need(t, "anchor", str, c), points,
                             f"{c}.anchor"),
                clearance=_enum(_need(t, "clearance", str, c), CLEARANCES,
                                f"{c}.clearance"),
                slack=_enum(t.get("slack", "moderate"), SLACKS,
                            f"{c}.slack")))
        elif term == "centered":
            out.append(TransTerm(
                term=term,
                anchor=_enum(_need(t, "anchor", str, c), points,
                             f"{c}.anchor"),
                tol=_enum(_need(t, "tol", str, c), SLACKS, f"{c}.tol")))
        elif term == "along":
            out.append(TransTerm(
                term=term,
                axis=_enum(_need(t, "axis", str, c), axes, f"{c}.axis"),
                sign=_enum(_need(t, "sign", str, c), SIGNS, f"{c}.sign")))
        elif term == "inside":
            out.append(TransTerm(
                term=term,
                anchor=_enum(_need(t, "anchor", str, c), points,
                             f"{c}.anchor"),
                slack=_enum(t.get("slack", "snug"), SLACKS, f"{c}.slack")))
        elif term == "expr":
            row = _enum(_need(t, "row", str, c), TRANS_ROWS, f"{c}.row")
            lo = _need(t, "lo", str, c)
            hi = _need(t, "hi", str, c)
            flags = validate_expr(lo, quants) + validate_expr(hi, quants)
            out.append(TransTerm(term=term, row=row, expr_lo=lo,
                                 expr_hi=hi, flags=flags))
    if not out:
        raise ParseRejection(f"{ctx}.trans has no terms")
    return tuple(out)


def parse_emission(raw: str, vocab: Vocabulary) -> StageEmission:
    doc = _load_json(raw)
    ctx = "emission"
    stage = _need(doc, "stage", int, ctx)
    name = _need(doc, "name", str, ctx)
    objs = set(vocab.objects)
    active = _enum(_need(doc, "active", str, ctx), objs, f"{ctx}.active")
    passive = doc.get("passive")
    if passive is not None:
        passive = _enum(passive, objs, f"{ctx}.passive")
    w_origin = _enum(_need(doc, "w_origin", str, ctx), vocab.point_names(),
                     f"{ctx}.w_origin")
    w_axis = _enum(_need(doc, "w_axis", str, ctx), vocab.axis_names(),
                   f"{ctx}.w_axis")
    p = _need(doc, "path_tsr", dict, ctx)
    g = _need(doc, "subgoal_tsr", dict, ctx)
    path = TSRSpec(rot=_parse_rot_rows(p.get("rot", "free"), vocab, "path"),
                   trans=_parse_trans_terms(p.get("trans", "free"), vocab,
                                            "path"))
    goal = TSRSpec(rot=_parse_rot_rows(g.get("rot", "free"), vocab,
                                       "subgoal"),
                   trans=_parse_trans_terms(g.get("trans", "free"), vocab,
                                            "subgoal"))
    verify = doc.get("verify", "")
    if not isinstance(verify, str):
        raise ParseRejection("emission.verify must be a string")
    return StageEmission(stage=stage, name=name, active=active,
                         passive=passive, w_origin=w_origin, w_axis=w_axis,
                         path_tsr=path, subgoal_tsr=goal, verify=verify)


def parse_critic(raw: str, vocab: Vocabulary) -> CriticVerdict:
    doc = _load_json(raw)
    verdict = _enum(_need(doc, "verdict", str, "critic"), CRITIC_VERDICTS,
                    "critic.verdict")
    edits_raw = doc.get("edits", [])
    if not isinstance(edits_raw, list):
        raise ParseRejection("critic.edits must be a list")
    edits = []
    for i, e in enumerate(edits_raw):
        c = f"critic.edits[{i}]"
        if not isinstance(e, dict):
            raise ParseRejection(f"{c} must be an object")
        action = _enum(_need(e, "action", str, c), REPAIR_ACTIONS,
                       f"{c}.action")
        token = e.get("token")
        if token is not None:
            _enum(token, ROT_TOLS + CLEARANCES + SLACKS, f"{c}.token")
        edits.append(CriticEdit(target=_need(e, "target", str, c),
                                action=action, token=token))
    if verdict == "reject" and not edits:
        raise ParseRejection(
            "a reject verdict must carry at least one typed edit")
    diagnosis = doc.get("diagnosis", "")
    if not isinstance(diagnosis, str):
        raise ParseRejection("critic.diagnosis must be a string")
    return CriticVerdict(verdict=verdict, edits=tuple(edits),
                         diagnosis=diagnosis)


def parse_repair(raw: str, vocab: Vocabulary) -> RepairAction:
    doc = _load_json(raw)
    action = _enum(_need(doc, "action", str, "repair"), REPAIR_ACTIONS,
                   "repair.action")
    token = doc.get("token")
    if token is not None:
        _enum(token, ROT_TOLS + CLEARANCES + SLACKS, "repair.token")
    target = doc.get("target")
    if target is not None and not isinstance(target, str):
        raise ParseRejection("repair.target must be a string")
    rationale = doc.get("rationale", "")
    if not isinstance(rationale, str):
        raise ParseRejection("repair.rationale must be a string")
    return RepairAction(action=action, target=target, token=token,
                        rationale=rationale)


# ---------------------------------------------------------- prompt build

_JSON_ONLY = ("Respond with a single JSON object and nothing else — no "
              "prose, no markdown fences.")


def image_block(path: str | Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    suffix = Path(path).suffix.lower().lstrip(".")
    media = {"jpg": "jpeg"}.get(suffix, suffix)
    return {"type": "image",
            "source": {"type": "base64", "media_type": f"image/{media}",
                       "data": data}}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def build_stage_plan_prompt(task: str, vocab: Vocabulary,
                            view_paths: list[Path] | None = None
                            ) -> tuple[str, list]:
    """Mark-addressed (vocab.marks + scene image) or text-only
    (vocab.objects, the ablation baseline)."""
    head = ("You are the task-planning module of a robotic manipulation "
            "pipeline. You decompose a task into an ordered list of stages. "
            "You make only discrete symbolic choices; all geometry is "
            "computed downstream.\n\n")
    if vocab.marks:
        if not view_paths:
            raise ValueError("mark-addressed stage planning needs the marked "
                             "scene image")
        system = (head +
            "The image shows the scene with every foreground object "
            "labelled by a numbered mark. Decide which marks the task "
            "involves and name them; ignore the rest. Each stage has one "
            "active object (the one that moves) and at most one passive "
            "object it is constrained against.\n\n"
            f"Marks on the image:\n{vocab.describe_marks()}\n\n"
            "Output schema: {\"objects\": {\"<mark id>\": \"<short name>\", ...}, "
            "\"stages\": [{\"name\": str, \"active\": <mark id>, "
            "\"passive\": <mark id|null>, \"parts\": {\"<mark id>\": [str, ...]}}]}. "
            "`parts` lists, per object in the stage, the part names (free "
            "text) that must be segmented and grounded for this stage "
            "(e.g. \"handle\", \"spout\", \"rim\"). Use only mark ids "
            "that appear on the image. " + _JSON_ONLY)
    else:
        system = (head +
            f"Scene objects and grounded symbols:\n{vocab.describe_symbols()}\n\n"
            "Output schema: {\"stages\": [{\"name\": str, \"active\": <object>, "
            "\"passive\": <object|null>, \"parts\": {\"<object>\": [str, ...]}}]}. "
            "`parts` lists, per object in the stage, the part names (free "
            "text) that must be segmented and grounded for this stage "
            "(e.g. \"handle\", \"spout\", \"rim\"). " + _JSON_ONLY)
    content = [_text(f"Task: {task}")] + [image_block(p) for p in (view_paths or [])]
    return system, [{"role": "user", "content": content}]


def build_point_axis_prompt(stage: StageSpec, vocab: Vocabulary,
                            view_paths: list[Path]) -> tuple[str, list]:
    system = (
        "You are the interaction-point selection module. The images are "
        "rendered canonical views of the object with numbered candidate "
        "marks (filled = visible in that view, hollow = occluded; reason "
        "across views — constructed points such as cavity centers are "
        "off-surface by design). Choose the single best candidate for "
        f"the stage, the grounded axis anchoring the task frame, and the "
        "axis sign. You may ONLY use the offered IDs and names.\n\n"
        f"Candidate menu:\n{vocab.describe_menu()}\n\n"
        f"Grounded symbols:\n{vocab.describe_symbols()}\n\n"
        "Output schema: {\"candidate_id\": int, \"axis\": \"object.axis\", "
        "\"sign\": \"+\"|\"-\", \"secondary\": \"object.axis\"|null, "
        "\"rationale\": str}. " + _JSON_ONLY)
    content: list[dict] = [_text(
        f"Stage {stage.index} ({stage.name}): active={stage.active}, "
        f"passive={stage.passive}, parts={ {h: list(p) for h, p in stage.parts.items()} }. Select the "
        "interaction point and axis.")]
    content += [image_block(p) for p in view_paths]
    return system, [{"role": "user", "content": content}]


def build_emission_prompt(stage: StageSpec, vocab: Vocabulary,
                          selection: PointAxisSelection | None = None,
                          view_paths: list[Path] | None = None
                          ) -> tuple[str, list]:
    system = (
        "You are the constraint-emission module. Fill the TSR schema for "
        "one stage: a path TSR (holds along the whole motion) and a "
        "subgoal TSR (goal region). Every slot is a discrete token or a "
        "symbolic reference; a deterministic compiler produces all "
        "numerics. You must NEVER write a numeric rotation. Translational "
        "'expr' bounds may use arithmetic over the grounded quantity "
        "symbols only.\n\n"
        f"Grounded symbols:\n{vocab.describe_symbols()}\n\n"
        "Vocabulary: rot relations " + str(list(ROT_RELATIONS)) +
        " with tol in " + str(list(ROT_TOLS)) +
        "; trans terms 'free' | above/below(anchor, clearance in "
        + str(list(CLEARANCES)) + ", slack in " + str(list(SLACKS)) +
        ") | centered(anchor, tol in " + str(list(SLACKS)) +
        ") | along(axis, sign) | inside(anchor, slack) | expr(row in "
        + str(list(TRANS_ROWS)) + ", lo, hi as quantity arithmetic).\n\n"
        "Output schema: {\"stage\": int, \"name\": str, \"active\": obj, "
        "\"passive\": obj|null, \"w_origin\": \"obj.point\", \"w_axis\": "
        "\"obj.axis\", \"path_tsr\": {\"rot\": \"free\"|[rot row, ...], "
        "\"trans\": \"free\"|[trans term, ...]}, \"subgoal_tsr\": "
        "{...same...}, \"verify\": str}.\n"
        "rot row: {\"axis\": \"obj.axis\", \"relation\": str, \"reference\": "
        "\"obj.axis\"|\"world.z\", \"tol\": str} or {\"relation\": \"free\", "
        "\"row\": \"roll\"|\"pitch\"|\"yaw\"}.\n"
        "trans term: {\"term\": \"free\"} | {\"term\": \"above\"|\"below\", "
        "\"anchor\": \"obj.point\", \"clearance\": str, \"slack\": str} | "
        "{\"term\": \"centered\", \"anchor\": \"obj.point\", \"tol\": str} | "
        "{\"term\": \"along\", \"axis\": \"obj.axis\", \"sign\": \"+\"|\"-\"} | "
        "{\"term\": \"inside\", \"anchor\": \"obj.point\", \"slack\": str} | "
        "{\"term\": \"expr\", \"row\": \"x\"|\"y\"|\"z\", \"lo\": str, "
        "\"hi\": str}. Use exactly these key names.\n"
        "Compilability rule: a rot row is grounded only if its axis and "
        "its reference are BOTH aligned with w_axis (parallel or "
        "antiparallel to it) or BOTH perpendicular to w_axis; a mix is "
        "rejected. Choose w_axis so every rot row you write satisfies "
        "this. "
        + _JSON_ONLY)
    parts: list[dict] = [_text(
        f"Emit the TSR pair for stage {stage.index} ({stage.name}): "
        f"active={stage.active}, passive={stage.passive}."
        + ((f" The constraint frame w is owned by the passive object: "
            f"w_origin and w_axis must both be {stage.passive}.* symbols, "
            f"every trans anchor must be a {stage.passive}.* point, and a "
            f"rot row's axis must be a {stage.active}.* axis related to "
            f"world.z or a {stage.passive}.* axis. A z/z row already "
            "bounds roll and pitch and a plane/plane row already bounds "
            "yaw; do not add a second row relating the active object to "
            "the same reference.")
           if stage.passive else
           " This stage has no passive object: the constrained frame is "
           "the GRIPPER acting on the active object, so there is no "
           "static reference and relation rot rows cannot be grounded. "
           "Write every rot row as {\"relation\": \"free\", "
           "\"row\": ...}; unaddressed rows stay tight.")
        + (f" Selected interaction point candidate "
           f"{selection.candidate_id}, axis {selection.axis}, sign "
           f"{selection.sign}." if selection else ""))]
    for p in (view_paths or []):     # two-pass ablation arm: selected-axis render
        parts.append(image_block(p))
    return system, [{"role": "user", "content": parts}]


def build_critic_prompt(report: dict, view_paths: list[Path],
                        vocab: Vocabulary) -> tuple[str, list]:
    system = (
        "You are the render-and-check critic. You receive the preview "
        "report (metric detections with `authored_in` pointers naming "
        "the schema slot each bound came from) and rendered evidence of "
        "sampled goal configurations. Accept, or reject with typed "
        "one-token edits against named slots. Edits must target the "
        "constraint-authoring layer only; geometric/planning issues are "
        "not yours to fix.\n\n"
        "Edit actions: " + str(list(REPAIR_ACTIONS)) + "; tokens from "
        + str(list(ROT_TOLS + CLEARANCES + SLACKS)) + ".\n\n"
        "Output schema: {\"verdict\": \"accept\"|\"reject\", \"edits\": "
        "[{\"target\": <authored_in pointer>, \"action\": str, \"token\": "
        "str|null}], \"diagnosis\": str}. " + _JSON_ONLY)
    content: list[dict] = [_text("Preview report:\n"
                                 + json.dumps(report, indent=2))]
    content += [image_block(p) for p in view_paths]
    return system, [{"role": "user", "content": content}]


def build_repair_prompt(failure_type: str, context: dict,
                        vocab: Vocabulary) -> tuple[str, list]:
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"unknown failure type {failure_type!r}; "
                         f"known: {FAILURE_TYPES}")
    system = (
        "You are the typed-failure repair module. The planner reports a "
        "typed failure with context; choose ONE symbolic repair action. "
        "Repairs are one-token diffs — auditable and cheap to "
        "re-compile.\n\nActions: " + str(list(REPAIR_ACTIONS)) +
        "; tokens from " + str(list(ROT_TOLS + CLEARANCES + SLACKS)) +
        ".\n\nOutput schema: {\"action\": str, \"target\": str|null, "
        "\"token\": str|null, \"rationale\": str}. " + _JSON_ONLY)
    return system, [{"role": "user", "content": [_text(
        f"Failure: {failure_type}\nContext:\n"
        + json.dumps(context, indent=2))]}]


# ---------------------------------------------------------------- client

def _urllib_transport(payload: dict) -> str:
    """Default transport: POST the Messages API, return concatenated text
    blocks. Retries 429/5xx with exponential backoff."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise VLMError("ANTHROPIC_API_KEY is not set")
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(HTTP_RETRIES + 1):
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"content-type": "application/json",
                     "x-api-key": key,
                     "anthropic-version": API_VERSION})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                doc = json.loads(resp.read().decode())
            return "".join(b.get("text", "") for b in doc.get("content", [])
                           if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529) and attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF_S * (2 ** attempt))
                continue
            raise VLMError(f"API error {e.code}: "
                           f"{e.read().decode(errors='replace')[:500]}")
        except urllib.error.URLError as e:
            last = e
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF_S * (2 ** attempt))
                continue
    raise VLMError(f"transport failed after {HTTP_RETRIES + 1} attempts: "
                   f"{last}")


@dataclass
class CallLog:
    """One touchpoint invocation: attempts, rejections, flags — the audit
    trail the emission ablation and typed-repair loop reference."""
    touchpoint: str
    attempts: int = 0
    rejections: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    raw: str = ""          # accepted (parsed) response text, for replay


class Client:
    """The five touchpoints, each = prompt build -> transport -> strict
    parse, with bounded parse-rejection retry (the rejection reason is
    appended verbatim as a follow-up user turn)."""

    def __init__(self, transport=None, model: str = MODEL,
                 max_tokens: int = MAX_TOKENS):
        self._transport = transport or _urllib_transport
        self.model = model
        self.max_tokens = max_tokens
        self.logs: list[CallLog] = []

    # -- core loop -------------------------------------------------------

    def _ask(self, touchpoint: str, system: str, messages: list,
             parse) -> object:
        log = CallLog(touchpoint)
        self.logs.append(log)
        msgs = list(messages)
        for _ in range(1 + MAX_PARSE_RETRIES):
            log.attempts += 1
            raw = self._transport({"model": self.model,
                                   "max_tokens": self.max_tokens,
                                   "temperature": TEMPERATURE,
                                   "system": system,
                                   "messages": msgs})
            try:
                result = parse(raw)
            except ParseRejection as rej:
                log.rejections.append(rej.reason)
                msgs = msgs + [
                    {"role": "assistant", "content": [_text(raw)]},
                    {"role": "user", "content": [_text(
                        f"Your response was rejected: {rej.reason}. "
                        "Re-emit the corrected JSON object only.")]},
                ]
                continue
            for f in getattr(result, "flags_all", lambda: [])():
                log.flags.append(f)
            log.raw = raw
            return result
        raise VLMError(
            f"{touchpoint}: parse retry budget exhausted after "
            f"{log.attempts} attempts; rejections: {log.rejections}")

    # -- touchpoints -----------------------------------------------------

    def plan_stages(self, task: str, vocab: Vocabulary,
                    view_paths: list[Path] | None = None) -> StagePlan:
        system, messages = build_stage_plan_prompt(task, vocab, view_paths)
        return self._ask("plan_stages", system, messages,
                         lambda raw: parse_stage_plan(raw, vocab, task))

    def select_point_axis(self, stage: StageSpec, vocab: Vocabulary,
                          view_paths: list[Path]) -> PointAxisSelection:
        if not vocab.menu:
            raise ValueError("select_point_axis requires a candidate menu "
                             "in the vocabulary")
        system, messages = build_point_axis_prompt(stage, vocab, view_paths)
        return self._ask("select_point_axis", system, messages,
                         lambda raw: parse_point_axis(raw, vocab))

    def emit_constraints(self, stage: StageSpec, vocab: Vocabulary,
                         selection: PointAxisSelection | None = None,
                         view_paths: list[Path] | None = None,
                         rejections: list[tuple[str, str]] | None = None
                         ) -> StageEmission:
        """`rejections`: (raw_emission, slot-named CompileError text) pairs
        from earlier attempts at this stage, replayed as assistant/user
        turn pairs so the model repairs ITS OWN emission rather than
        re-rolling from the prompt (same shape as the parse-rejection
        retry in _ask). The minimal form of touchpoint #5 (repair) until
        it exists: the compiler's typed failure is what the model sees,
        nothing else."""
        system, messages = build_emission_prompt(stage, vocab, selection,
                                                 view_paths)
        for raw, r in rejections or []:
            messages = messages + [
                {"role": "assistant", "content": [_text(raw)]},
                {"role": "user", "content": [_text(
                    f"The compiler rejected that emission: {r}. Re-emit "
                    "the corrected JSON object only, changing as little "
                    "as needed.")]}]
        emission = self._ask("emit_constraints", system, messages,
                             lambda raw: parse_emission(raw, vocab))
        # surface literal flags into the call log
        log = self.logs[-1]
        for spec in (emission.path_tsr, emission.subgoal_tsr):
            for t in spec.trans:
                log.flags.extend(t.flags)
        if log.flags:
            print(f"[vlm] emit_constraints: {len(log.flags)} flagged "
                  f"numeric literal(s): {log.flags}")
        return emission

    def critique_preview(self, report: dict, view_paths: list[Path],
                         vocab: Vocabulary) -> CriticVerdict:
        system, messages = build_critic_prompt(report, view_paths, vocab)
        return self._ask("critique_preview", system, messages,
                         lambda raw: parse_critic(raw, vocab))

    def repair(self, failure_type: str, context: dict,
               vocab: Vocabulary) -> RepairAction:
        system, messages = build_repair_prompt(failure_type, context, vocab)
        return self._ask("repair", system, messages,
                         lambda raw: parse_repair(raw, vocab))