"""Critic / repair core for VLM calls #4 (critique_preview) and #5 (repair):
the shared edit grammar, the vocabulary-checked parser, the edit applier,
the loop guards and the preview report. Pure library: no rendering, no
network, no VLM call. The prompts live in vlm.py (build_critic_prompt_v2);
docs/deferred.md is the design record.

Edits are typed and small. The VLM names a target and an action from a
closed table, plus a token whose type the action fixes:

    level      one step along an enum ladder (tight < moderate < loose,
               snug < moderate < loose, contact < small < medium < large);
               the VLM names the next level, code checks it is exactly one
               step in the action's direction. Code never picks magnitudes.
    point      a grounded point of the stage's active object
               ("teapot.spout_center"), mapped to its call #2 candidate
               mark; the swap goes back through selection.resolve_selection
               like any call #2 pick (one selection path).
    direction  a licensed signed canonical direction of the active object
               ("teapot.+up"), replacing a rotation row's constrained axis.
    none       free_dof; reemit_stage (which needs a diagnosis instead).

Action sets. The critic only sees members of the goal and path sets, so
its evidence can justify removing poses but never adding them: tighten,
re-pick, or ask call #3 to re-emit a stage. widen_clearance is repair-only:
in compile_tsr the clearance shifts the above/below band without narrowing
it (z in [off + clr, off + clr + band]), so it adds poses as well as
removing them. Repair answers planner failures, where the specification
is too tight, so it may also loosen or shift.

Targets (pointers):
    <stage>.<path|subgoal>.<rot|trans>[i]   a row of the stage's emission;
                                            the same slot names
                                            compile_tsr.CompileError uses
    <stage>.point.<object>                  the stage's call #2 point choice
    <stage>                                 the whole stage (reemit_stage)
where <stage> is the emission's stage name. Stages whose role is context
(the grasp: hand-written in sim, taught on hardware) cannot be edited.

Guards (LoopLedger): one round budget shared by calls 4 and 5 (default 2);
neither call may reverse the other's edit on the same target (checked at
parse time when the parser is given the ledger, so a reversal goes back to
the model as a parse retry); a reject once the budget is spent returns the
typed failure preview_rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .vlm import (_FLIP, CLEARANCES, CRITIC_VERDICTS, ROT_ROWS, ROT_TOLS,
                  SLACKS, TRANS_ROWS, ParseRejection, PointAxisSelection,
                  RotRow, StageEmission, TransTerm, TSRSpec, Vocabulary,
                  _enum, _load_json, _need, _norm_dir)

# ------------------------------------------------------------------ actions

CRITIC_ACTIONS = ("tighten_tolerance", "tighten_slack", "reselect_point",
                  "reselect_axis", "reemit_stage")
REPAIR_ACTIONS_V2 = ("relax_tolerance", "free_dof", "widen_slack",
                     "widen_clearance", "reselect_point", "reselect_axis",
                     "reemit_stage")
ACTION_SETS = {"critic": CRITIC_ACTIONS, "repair": REPAIR_ACTIONS_V2}

# action -> (token type, target kind, meaning). Single source for the
# parser, the applier and the prompt's action table.
ACTION_TABLE: dict[str, tuple[str | None, str, str]] = {
    "tighten_tolerance": (
        "level", "row",
        "one step tighter on a rotation row's tol (loose -> moderate -> "
        "tight) or a centered term's tol (loose -> moderate -> snug)"),
    "relax_tolerance": (
        "level", "row",
        "one step looser on a rotation row's tol or a centered term's tol"),
    "tighten_slack": (
        "level", "row",
        "one step narrower band on an above/below/inside term's slack "
        "(loose -> moderate -> snug)"),
    "widen_slack": (
        "level", "row",
        "one step wider band on an above/below/inside term's slack"),
    "widen_clearance": (
        "level", "row",
        "one step more standoff on an above/below term's clearance "
        "(contact -> small -> medium -> large); the band keeps its height "
        "and moves away from the anchor, so it drops the nearest poses "
        "and adds as many farther away"),
    "free_dof": (
        None, "row", "free a row (rotation or translation) entirely"),
    "reselect_point": (
        "point", "point",
        "replace the stage's call #2 point on the active object with "
        "another grounded point of that object"),
    "reselect_axis": (
        "direction", "row",
        "replace a rotation row's constrained direction with another "
        "signed canonical direction of the active object"),
    "reemit_stage": (
        None, "stage",
        "ask call #3 to write the stage again, with your diagnosis "
        "appended (e.g. a path row the motion needs is missing)"),
}

# ladders, ordered from most restrictive to least
_ROT_TOL_LADDER = ROT_TOLS                 # tight, moderate, loose
_SLACK_LADDER = SLACKS                     # snug, moderate, loose
_CLEARANCE_LADDER = tuple(reversed(CLEARANCES))   # large ... contact

# action -> (field kind, step): step +1 = toward the restrictive end
_LEVEL_ACTIONS = {
    "tighten_tolerance": ("tol", +1), "relax_tolerance": ("tol", -1),
    "tighten_slack": ("slack", +1), "widen_slack": ("slack", -1),
    "widen_clearance": ("clearance", +1),
}
# direction of each action on its target: +1 restricts, -1 loosens, 0
# replaces (re-pick, or a clearance shift); used by the no-reverse rule
_DIRECTION = {"tighten_tolerance": +1, "tighten_slack": +1,
              "widen_clearance": 0, "relax_tolerance": -1,
              "widen_slack": -1, "free_dof": -1, "reselect_point": 0,
              "reselect_axis": 0, "reemit_stage": 0}


# ------------------------------------------------------------------ refusals

class EditRefusal(ParseRejection):
    """A licit-looking edit the specification cannot take. `code` types it
    for logs and tests; `reason` goes back to the model verbatim on a parse
    retry (vlm.Client._ask), so it names the offending target or token and
    what is allowed."""

    CODES = ("unknown_action", "unknown_target", "not_editable",
             "wrong_target", "bad_token", "not_one_step", "no_change",
             "point_not_in_pool", "duplicate_target", "conflict",
             "missing_diagnosis", "reversal")

    def __init__(self, code: str, reason: str):
        assert code in self.CODES, code
        super().__init__(reason)
        self.code = code


# ------------------------------------------------------------------ state

@dataclass(frozen=True)
class SpecState:
    """The editable specification: call #2 selections plus call #3
    emissions, role-keyed.

    emissions   role -> StageEmission (roles that call #3 authored)
    selections  role -> PointAxisSelection (call #2; may include roles
                with no emission, e.g. the passive's point)
    objects     role -> object the selection is on (selection_objects)
    pools       object -> candidate pool (selection.load_pool)
    offered     role -> candidate ids call #2 was shown for that role; when
                given, a re-pick must stay inside it
    context     roles shown to the critic but not editable; None = every
                role whose emission has no passive object (the grasp)
    """
    emissions: dict[str, StageEmission]
    selections: dict[str, PointAxisSelection]
    objects: dict[str, str]
    pools: dict[str, dict[int, dict]]
    offered: dict[str, frozenset[int]] | None = None
    context: frozenset[str] | None = None

    def context_roles(self) -> frozenset[str]:
        if self.context is not None:
            return frozenset(self.context)
        return frozenset(r for r, e in self.emissions.items()
                         if e.passive is None)

    def role_of_stage(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for r, e in self.emissions.items():
            if e.name in out:
                raise ValueError(f"two roles share the stage name {e.name!r} "
                                 f"({out[e.name]}, {r}); pointers would be "
                                 "ambiguous")
            out[e.name] = r
        return out

    def point_of(self, role: str) -> str | None:
        """'obj.symbol' of the role's selected candidate, or None when the
        candidate is not a named point."""
        obj = self.objects[role]
        c = self.pools.get(obj, {}).get(int(self.selections[role].candidate_id))
        return f"{obj}.{c['symbol']}" if c and c.get("symbol") else None


def authored_slots(em: StageEmission) -> list[str]:
    """Every row pointer of one emission, in compile_tsr's slot naming."""
    out = []
    for ctx, spec in (("path", em.path_tsr), ("subgoal", em.subgoal_tsr)):
        out += [f"{em.name}.{ctx}.rot[{i}]" for i in range(len(spec.rot))]
        out += [f"{em.name}.{ctx}.trans[{i}]" for i in range(len(spec.trans))]
    return out


def candidate_for_point(pool: dict[int, dict], name: str,
                        offered: frozenset[int] | None = None) -> int | None:
    """The candidate id whose grounded symbol is `name` (unqualified), or
    None. With `offered`, only ids call #2 was shown count."""
    ids = sorted(i for i, c in pool.items() if c.get("symbol") == name
                 and (offered is None or i in offered))
    return ids[0] if ids else None


# ------------------------------------------------------------------ edits

@dataclass(frozen=True)
class Edit:
    """One validated edit. `before`/`after` are the target's value as text
    (a level, a point token, a row rendering) for logs and the no-reverse
    rule; `payload` carries what the applier needs."""
    target: str
    action: str
    token: str | None
    role: str
    kind: str                      # "row" | "point" | "stage"
    before: str
    after: str
    direction: int
    payload: tuple = ()


@dataclass(frozen=True)
class StageCheck:
    stage: str
    expected: str
    observed: str


@dataclass(frozen=True)
class Verdict:
    """Parsed call #4 output (source "critic") or call #5 output (source
    "repair"; stages empty, verdict "reject" by construction)."""
    source: str
    verdict: str
    stages: tuple[StageCheck, ...]
    edits: tuple[Edit, ...]
    diagnosis: str


def _row_ref(state: SpecState, target: str):
    """(role, ctx, table, index) for a row pointer, else None."""
    for name, role in state.role_of_stage().items():
        if not target.startswith(name + "."):
            continue
        rest = target[len(name) + 1:]
        for ctx in ("path", "subgoal"):
            for table in ("rot", "trans"):
                head = f"{ctx}.{table}["
                if rest.startswith(head) and rest.endswith("]"):
                    idx = rest[len(head):-1]
                    if idx.isdigit():
                        return role, ctx, table, int(idx)
    return None


def _spec(em: StageEmission, ctx: str) -> TSRSpec:
    return em.path_tsr if ctx == "path" else em.subgoal_tsr


def row_text(row) -> str:
    """One emitted row as the critic reads it."""
    if isinstance(row, RotRow):
        if row.relation == "free":
            return f"free{' (' + row.row + ')' if row.row else ''}"
        return f"{row.axis} {row.relation} {row.reference}, tol {row.tol}"
    t: TransTerm = row
    if t.term == "free":
        return "free"
    if t.term in ("above", "below"):
        return (f"{t.term} {t.anchor}, clearance {t.clearance}, "
                f"slack {t.slack}")
    if t.term == "centered":
        return f"centered on {t.anchor}, tol {t.tol}"
    if t.term == "inside":
        return f"inside {t.anchor}, slack {t.slack}"
    if t.term == "along":
        return f"along {t.axis} ({t.sign})"
    if t.term == "expr":
        return f"{t.row} in [{t.expr_lo}, {t.expr_hi}]"
    return t.term


def _level_field(row, kind: str) -> tuple[str, tuple[str, ...]] | None:
    """(field name, ladder most-restrictive-first) the action kind edits on
    this row, or None if the row has no such field."""
    if isinstance(row, RotRow):
        return ("tol", _ROT_TOL_LADDER) if (kind == "tol" and
                                            row.relation != "free") else None
    if kind == "tol" and row.term == "centered":
        return "tol", _SLACK_LADDER
    if kind == "slack" and row.term in ("above", "below", "inside"):
        return "slack", _SLACK_LADDER
    if kind == "clearance" and row.term in ("above", "below"):
        return "clearance", _CLEARANCE_LADDER
    return None


def validate_edit(raw: dict, ctx: str, source: str, state: SpecState,
                  vocab: Vocabulary) -> Edit:
    """One raw edit object -> a validated Edit, or EditRefusal."""
    actions = ACTION_SETS[source]
    action = _need(raw, "action", str, ctx)
    if action not in actions:
        raise EditRefusal("unknown_action", (
            f"{ctx}.action {action!r} is not a {source} action; allowed: "
            f"{list(actions)}"))
    target = _need(raw, "target", str, ctx)
    token = raw.get("token")
    if token is not None and not isinstance(token, str):
        raise EditRefusal("bad_token", f"{ctx}.token must be a string or null")
    tok_type, kind, _ = ACTION_TABLE[action]
    by_stage = state.role_of_stage()
    context = state.context_roles()

    def _editable(role: str, name: str) -> None:
        if role in context:
            raise EditRefusal("not_editable", (
                f"{ctx}: stage {name!r} is context (its grasp is not "
                "authored here) and cannot be edited"))

    if kind == "stage":
        role = by_stage.get(target)
        if role is None:
            raise EditRefusal("unknown_target", (
                f"{ctx}.target {target!r}: {action} needs a stage name; "
                f"stages: {sorted(by_stage)}"))
        _editable(role, target)
        if token is not None:
            raise EditRefusal("bad_token", f"{ctx}: {action} takes no token")
        return Edit(target, action, None, role, "stage", "emitted",
                    "re-emit", _DIRECTION[action])

    if kind == "point":
        role = None
        for name, r in by_stage.items():
            if (r in state.objects and r in state.selections
                    and target == f"{name}.point.{state.objects[r]}"):
                role, sname = r, name
        if role is None:
            ok = [f"{n}.point.{state.objects[r]}" for n, r in by_stage.items()
                  if r in state.objects and r in state.selections
                  and r not in context
                  and state.objects[r] == state.emissions[r].active]
            raise EditRefusal("unknown_target", (
                f"{ctx}.target {target!r}: {action} needs a point pointer "
                f"<stage>.point.<object>; allowed: {sorted(ok)}"))
        _editable(role, sname)
        obj = state.objects[role]
        em = state.emissions[role]
        if obj != em.active:
            raise EditRefusal("wrong_target", (
                f"{ctx}: only the active object's point can be re-picked "
                f"({em.active!r}), not {obj!r}"))
        names = vocab.objects.get(obj, {}).get("points", ())
        allowed = [f"{obj}.{n}" for n in names]
        if token is None or token not in allowed:
            raise EditRefusal("bad_token", (
                f"{ctx}.token {token!r} is not a grounded point of {obj!r}; "
                f"allowed: {allowed}"))
        offered = (state.offered or {}).get(role)
        cid = candidate_for_point(state.pools.get(obj, {}),
                                  token.split(".", 1)[1], offered)
        if cid is None:
            raise EditRefusal("point_not_in_pool", (
                f"{ctx}.token {token!r} has no candidate mark in the pool "
                f"call #2 chose from for {sname!r}; pick a point that was "
                "offered"))
        before = state.point_of(role) or f"mark {state.selections[role].candidate_id}"
        if cid == int(state.selections[role].candidate_id):
            raise EditRefusal("no_change", f"{ctx}: {token} is already selected")
        return Edit(target, action, token, role, "point", before, token,
                    _DIRECTION[action], (cid,))

    # row targets
    ref = _row_ref(state, target)
    if ref is None:
        ok = [s for r, e in state.emissions.items() if r not in context
              for s in authored_slots(e)]
        raise EditRefusal("unknown_target", (
            f"{ctx}.target {target!r} is not an authored row; allowed: {ok}"))
    role, tsr_ctx, table, idx = ref
    em = state.emissions[role]
    _editable(role, em.name)
    rows = getattr(_spec(em, tsr_ctx), table)
    if idx >= len(rows):
        raise EditRefusal("unknown_target", (
            f"{ctx}.target {target!r}: {em.name}.{tsr_ctx}.{table} has "
            f"{len(rows)} rows"))
    row = rows[idx]
    before = row_text(row)

    if action in _LEVEL_ACTIONS:
        fkind, step = _LEVEL_ACTIONS[action]
        lf = _level_field(row, fkind)
        if lf is None:
            raise EditRefusal("wrong_target", (
                f"{ctx}: {action} needs a row with a {fkind}; {target} is "
                f"'{before}'"))
        fname, ladder = lf
        cur = getattr(row, fname)
        if cur not in ladder:
            raise EditRefusal("wrong_target", (
                f"{ctx}: {target} has {fname} {cur!r}, not one of "
                f"{list(ladder)}; {action} cannot step it"))
        i = ladder.index(cur)
        # ladders run most restrictive first: restricting moves toward 0
        j = i - step
        if not 0 <= j < len(ladder):
            raise EditRefusal("not_one_step", (
                f"{ctx}: {target} {fname} is already {cur!r}; {action} has "
                "no further step"))
        if token is None or token not in ladder:
            raise EditRefusal("bad_token", (
                f"{ctx}.token {token!r}: {action} on {fname} takes one of "
                f"{list(ladder)}; the next step from {cur!r} is "
                f"{ladder[j]!r}"))
        if token != ladder[j]:
            raise EditRefusal("not_one_step", (
                f"{ctx}.token {token!r}: one step only; from {cur!r} "
                f"{action} goes to {ladder[j]!r}"))
        new_row = replace(row, **{fname: token})
        return Edit(target, action, token, role, "row", before,
                    row_text(new_row), _DIRECTION[action],
                    (tsr_ctx, table, idx, new_row))

    if action == "free_dof":
        if token is not None:
            raise EditRefusal("bad_token", f"{ctx}: free_dof takes no token")
        if (row.relation == "free" if isinstance(row, RotRow)
                else row.term == "free"):
            raise EditRefusal("no_change", f"{ctx}: {target} is already free")
        new_row = (RotRow(axis=None, relation="free", reference=None,
                          tol=None, row=None) if isinstance(row, RotRow)
                   else TransTerm(term="free"))
        return Edit(target, action, None, role, "row", before, "free",
                    _DIRECTION[action], (tsr_ctx, table, idx, new_row))

    if action == "reselect_axis":
        if not isinstance(row, RotRow) or row.relation == "free":
            raise EditRefusal("wrong_target", (
                f"{ctx}: reselect_axis needs a non-free rotation row; "
                f"{target} is '{before}'"))
        allowed = [f"{em.active}.{d}"
                   for d in vocab.canonical_directions(em.active)]
        if token not in allowed:
            raise EditRefusal("bad_token", (
                f"{ctx}.token {token!r} is not a direction of {em.active!r}; "
                f"allowed: {allowed}"))
        axis, sign = _norm_dir(token, vocab, f"{ctx}.token")
        rel = row.relation
        if sign == "-":
            rel = _FLIP[rel]
        new_row = replace(row, axis=axis, relation=rel)
        if new_row == row:
            raise EditRefusal("no_change", f"{ctx}: {target} already uses {token}")
        return Edit(target, action, token, role, "row", before,
                    row_text(new_row), _DIRECTION[action],
                    (tsr_ctx, table, idx, new_row))

    raise EditRefusal("wrong_target", f"{ctx}: {action} cannot target {target}")


def _parse_edits(edits_raw, source: str, state: SpecState,
                 vocab: Vocabulary,
                 ledger: "LoopLedger | None" = None) -> tuple[Edit, ...]:
    if not isinstance(edits_raw, list):
        raise ParseRejection(f"{source}.edits must be a list")
    out: list[Edit] = []
    for i, e in enumerate(edits_raw):
        c = f"{source}.edits[{i}]"
        if not isinstance(e, dict):
            raise ParseRejection(f"{c} must be an object")
        out.append(validate_edit(e, c, source, state, vocab))
    seen: dict[str, int] = {}
    for i, e in enumerate(out):
        if e.target in seen:
            raise EditRefusal("duplicate_target", (
                f"{source}.edits[{i}] targets {e.target!r} again (edits "
                f"[{seen[e.target]}] and [{i}]); at most one edit per target"))
        seen[e.target] = i
    reemit = {e.role for e in out if e.action == "reemit_stage"}
    clash = [e.target for e in out
             if e.role in reemit and e.action != "reemit_stage"]
    if clash:
        raise EditRefusal("conflict", (
            f"{source}: reemit_stage rewrites the whole stage, so it cannot "
            f"be combined with other edits on that stage ({clash})"))
    if ledger is not None:
        for e in out:
            ledger.check_reverse(source, e)
    return tuple(out)


def _need_text(doc: dict, key: str, ctx: str) -> str:
    v = _need(doc, key, str, ctx)
    if not v.strip():
        raise ParseRejection(f"{ctx}.{key} must be a non-empty string")
    return v


def parse_verdict(raw: str, vocab: Vocabulary, state: SpecState,
                  ledger: "LoopLedger | None" = None) -> Verdict:
    """Call #4 output -> Verdict. Schema:
        {"stages": [{"stage": <name>, "expected": str, "observed": str}, ...],
         "verdict": "accept"|"reject",
         "edits": [{"target": str, "action": str, "token": str|null}, ...],
         "diagnosis": str}
    One stages entry per stage, expected before observed (the model writes
    what the task needs before describing the render). A reject needs at
    least one edit and a non-empty diagnosis; an accept carries no edits.
    With `ledger`, an edit that reverses the other call's earlier edit is
    refused here, so the model gets a parse retry."""
    doc = _load_json(raw)
    if not isinstance(doc, dict):
        raise ParseRejection("critic output must be a JSON object")
    names = sorted(state.role_of_stage())
    stages_raw = _need(doc, "stages", list, "critic")
    checks: list[StageCheck] = []
    for i, s in enumerate(stages_raw):
        c = f"critic.stages[{i}]"
        if not isinstance(s, dict):
            raise ParseRejection(f"{c} must be an object")
        name = _enum(_need(s, "stage", str, c), names, f"{c}.stage")
        checks.append(StageCheck(name, _need_text(s, "expected", c),
                                 _need_text(s, "observed", c)))
    got = [c.stage for c in checks]
    if sorted(got) != names:
        raise ParseRejection(
            f"critic.stages must list every stage exactly once: expected "
            f"{names}, got {got}")
    verdict = _enum(_need(doc, "verdict", str, "critic"), CRITIC_VERDICTS,
                    "critic.verdict")
    edits = _parse_edits(doc.get("edits", []), "critic", state, vocab, ledger)
    diagnosis = doc.get("diagnosis", "")
    if not isinstance(diagnosis, str):
        raise ParseRejection("critic.diagnosis must be a string")
    if verdict == "reject":
        if not edits:
            raise ParseRejection(
                "a reject verdict must carry at least one typed edit")
        if not diagnosis.strip():
            raise EditRefusal("missing_diagnosis", (
                "a reject verdict needs a diagnosis: what is wrong, in terms "
                "of the stages' expected vs. observed"))
    elif edits:
        raise ParseRejection("an accept verdict carries no edits")
    return Verdict("critic", verdict, tuple(checks), edits, diagnosis)


def parse_repair_v2(raw: str, vocab: Vocabulary, state: SpecState,
                    ledger: "LoopLedger | None" = None) -> Verdict:
    """Call #5 output -> Verdict (source "repair"). Schema:
        {"edits": [{"target", "action", "token"}, ...], "diagnosis": str}
    At least one edit and a non-empty diagnosis."""
    doc = _load_json(raw)
    if not isinstance(doc, dict):
        raise ParseRejection("repair output must be a JSON object")
    edits = _parse_edits(_need(doc, "edits", list, "repair"), "repair",
                         state, vocab, ledger)
    if not edits:
        raise ParseRejection("repair must carry at least one typed edit")
    diagnosis = _need_text(doc, "diagnosis", "repair")
    return Verdict("repair", "reject", (), edits, diagnosis)


# ------------------------------------------------------------------ applier

@dataclass(frozen=True)
class ApplyResult:
    """The new specification, the stages call #3 must re-emit
    (role -> diagnosis to append), and the roles whose compile must be
    redone (every edited role)."""
    state: SpecState
    reemit: dict[str, str]
    changed_roles: tuple[str, ...]


def apply_edits(state: SpecState, edits: tuple[Edit, ...],
                diagnosis: str) -> ApplyResult:
    """Apply validated edits all-or-none. Pure: `state` is not modified;
    a failure leaves the caller with the old state. At most one edit per
    target is enforced again here (the parser already did), since callers
    may build Edits directly."""
    targets = [e.target for e in edits]
    if len(set(targets)) != len(targets):
        raise EditRefusal("duplicate_target",
                          f"at most one edit per target: {targets}")
    emissions = dict(state.emissions)
    selections = dict(state.selections)
    reemit: dict[str, str] = {}
    for e in edits:
        if e.action == "reemit_stage":
            reemit[e.role] = diagnosis
        elif e.kind == "point":
            (cid,) = e.payload
            old = selections[e.role]
            selections[e.role] = replace(
                old, candidate_id=int(cid),
                rationale=f"re-picked {e.token} by edit on {e.target}")
        else:
            tsr_ctx, table, idx, new_row = e.payload
            em = emissions[e.role]
            spec = _spec(em, tsr_ctx)
            rows = list(getattr(spec, table))
            rows[idx] = new_row
            spec = replace(spec, **{table: tuple(rows)})
            emissions[e.role] = replace(em, **{
                "path_tsr" if tsr_ctx == "path" else "subgoal_tsr": spec})
    new = replace(state, emissions=emissions, selections=selections)
    return ApplyResult(new, reemit,
                       tuple(sorted({e.role for e in edits})))


# ------------------------------------------------------------------ guards

@dataclass
class LoopLedger:
    """Round budget and no-reverse rule shared by calls 4 and 5.

    Usage per verdict:
        verdict = parse_verdict(raw, vocab, state, ledger)  # checks reversals
        outcome = ledger.admit(verdict)   # "accept" | "apply" | typed failure dict
        if outcome == "apply":
            res = apply_edits(state, verdict.edits, verdict.diagnosis)
            ledger.record(verdict)

    reemit_stage regenerates the stage, so recording it forgets that
    stage's row history (the old row pointers name rows that no longer
    exist); its point-choice history is kept. A reemit by one call is not
    treated as reversing the other's row edits: the new emission is
    compiled and previewed again like any edit.
    """
    budget: int = 2
    rounds_used: int = 0
    history: list[dict] = field(default_factory=list)

    def admit(self, verdict: Verdict):
        """'accept', 'apply', or the preview_rejected failure record. The
        reversal check runs again here for callers that parsed without the
        ledger; it raises EditRefusal (no retry at this point)."""
        if verdict.verdict == "accept":
            return "accept"
        if self.rounds_used >= self.budget:
            return {"type": "preview_rejected", "routes_to": None,
                    "source": verdict.source, "rounds_used": self.rounds_used,
                    "budget": self.budget, "diagnosis": verdict.diagnosis,
                    "edits": [_edit_record(e) for e in verdict.edits],
                    "history": list(self.history)}
        for e in verdict.edits:
            self.check_reverse(verdict.source, e)
        return "apply"

    def check_reverse(self, source: str, e: Edit) -> None:
        other = [h for h in self.history
                 if h["target"] == e.target and h["source"] != source]
        if not other:
            return
        last = other[-1]
        reverses = (e.direction != 0 and e.direction == -last["direction"]) \
            or (e.direction == 0 and last["direction"] == 0
                and e.after == last["before"])
        if reverses:
            raise EditRefusal("reversal", (
                f"{e.target}: {e.action} would undo the {last['source']}'s "
                f"earlier {last['action']} ({last['before']} -> "
                f"{last['after']}); choose a different edit"))

    def record(self, verdict: Verdict) -> None:
        self.rounds_used += 1
        for e in verdict.edits:
            if e.action == "reemit_stage":
                rows = (f"{e.target}.path.", f"{e.target}.subgoal.")
                self.history = [h for h in self.history
                                if not h["target"].startswith(rows)]
        for e in verdict.edits:
            self.history.append({"round": self.rounds_used,
                                 "source": verdict.source,
                                 **_edit_record(e)})


def _edit_record(e: Edit) -> dict:
    return {"target": e.target, "action": e.action, "token": e.token,
            "role": e.role, "before": e.before, "after": e.after,
            "direction": e.direction}


# ------------------------------------------------------------------ preview report

_BOX_ROWS = TRANS_ROWS + ROT_ROWS          # x y z roll pitch yaw (in w)


@dataclass(frozen=True)
class StagePreviewInput:
    """Everything the preview report (and the preview renderer) needs for
    one stage. Poses are world (base) poses of each object's canonical
    frame -- the frame symbols are expressed in.

    role, emission   the stage's role and call #3 emission
    compiled         compile_tsr.CompiledStage, compiled at this stage's
                     entry (stage k+1 against stage k's subgoal nominal)
    poses            object -> 4x4 pose at the stage's entry (the passive
                     does not move; the active is at its entry)
    symbols          object -> frames.Symbols
    w_point          w's origin in the w owner's body coords
                     (selection.selected_points)
    e_point          the active object's selected point (body coords), or
                     None for a gripper stage
    goal_samples     optional world poses of the active object at
                     IK-feasible goals (best first); reported as a spread
    editable         False for context stages (the grasp)
    """
    role: str
    emission: StageEmission
    compiled: Any
    poses: dict[str, np.ndarray]
    symbols: dict[str, Any]
    w_point: np.ndarray
    e_point: np.ndarray | None = None
    goal_samples: tuple[np.ndarray, ...] = ()
    editable: bool = True


def _bounds(Bw: np.ndarray) -> tuple[dict, list[str]]:
    out, free = {}, []
    for k, name in enumerate(_BOX_ROWS):
        lo, hi = float(Bw[k, 0]), float(Bw[k, 1])
        if k >= 3 and hi - lo >= 2 * np.pi - 1e-9:
            out[name] = "free"
            free.append(name)
        elif not (np.isfinite(lo) or np.isfinite(hi)):
            out[name] = "free"
            free.append(name)
        elif not (np.isfinite(lo) and np.isfinite(hi)):     # one-sided
            out[name] = [round(lo, 4) if np.isfinite(lo) else None,
                         round(hi, 4) if np.isfinite(hi) else None]
        else:
            out[name] = [round(lo, 4), round(hi, 4)]
    return out, free


def _point_offsets(T_active: np.ndarray, w_world: np.ndarray, up: np.ndarray,
                   points: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for name, p in points.items():
        x = (T_active @ np.r_[np.asarray(p, float), 1.0])[:3] - w_world
        h = float(x @ up)
        out[name] = (float(np.linalg.norm(x - h * up)), h)
    return out


def stage_report(si: StagePreviewInput, point_choice: str | None = None) -> dict:
    """One stage of the preview report: rows as text with their pointers,
    the compiled bounds in w (free rows listed), and the active object's
    labeled points relative to w at the subgoal nominal (horizontal
    distance and height along the passive's up). Generic: no task terms."""
    em, cs = si.emission, si.compiled
    rec: dict[str, Any] = {
        "stage": em.name, "role": si.role, "active": em.active,
        "passive": em.passive, "editable": bool(si.editable),
        "goal": em.goal}
    if point_choice is not None:
        rec["point_choice"] = {"pointer": f"{em.name}.point.{em.active}",
                               "current": point_choice}
    for ctx, spec, tsr in (("path", em.path_tsr, cs.path),
                           ("subgoal", em.subgoal_tsr, cs.subgoal)):
        rows = [{"pointer": f"{em.name}.{ctx}.rot[{i}]", "text": row_text(r)}
                for i, r in enumerate(spec.rot)]
        rows += [{"pointer": f"{em.name}.{ctx}.trans[{i}]",
                  "text": row_text(t)} for i, t in enumerate(spec.trans)]
        bounds, free = _bounds(np.asarray(tsr.Bw, float))
        rec[ctx] = {"rows": rows, "bounds_in_w": bounds, "free_rows": free}
    if em.passive is not None:
        T_p = np.asarray(si.poses[em.passive], float)
        w = (T_p @ np.r_[np.asarray(si.w_point, float), 1.0])[:3]
        up = T_p[:3, :3] @ np.asarray(si.symbols[em.passive].axes["up_axis"], float)
        up = up / np.linalg.norm(up)
        pts = si.symbols[em.active].points
        nom = _point_offsets(np.asarray(cs.subgoal.nominal(), float), w, up, pts)
        rec["points_at_goal"] = {
            "relative_to": f"w origin ({em.passive}'s selected point), "
                           f"height along {em.passive} up",
            "at": "subgoal nominal",
            "points": {n: {"horizontal_m": round(h, 4), "height_m": round(z, 4)}
                       for n, (h, z) in nom.items()}}
        if si.goal_samples:
            per = [_point_offsets(np.asarray(T, float), w, up, pts)
                   for T in si.goal_samples]
            rec["goal_spread"] = {
                "n_goals": len(per),
                "points": {n: {"horizontal_m": [round(min(p[n][0] for p in per), 4),
                                                round(max(p[n][0] for p in per), 4)],
                               "height_m": [round(min(p[n][1] for p in per), 4),
                                            round(max(p[n][1] for p in per), 4)]}
                           for n in pts}}
    return rec


def build_preview_report(stages: list[StagePreviewInput],
                         state: SpecState | None = None) -> dict:
    """The report call #4 reads: one record per stage, in stage order,
    plus the editable pointers (so the prompt and the parser agree)."""
    recs = []
    for si in stages:
        choice = (state.point_of(si.role)
                  if state is not None and si.role in state.selections
                  and state.objects.get(si.role) == si.emission.active
                  else None)
        recs.append(stage_report(si, choice))
    out: dict[str, Any] = {"stages": recs}
    if state is not None:
        ctx = state.context_roles()
        out["editable_pointers"] = sorted(
            [s for r, e in state.emissions.items() if r not in ctx
             for s in authored_slots(e)]
            + [f"{e.name}.point.{state.objects[r]}"
               for r, e in state.emissions.items()
               if r not in ctx and state.objects.get(r) == e.active]
            + [e.name for r, e in state.emissions.items() if r not in ctx])
    return out


def report_json(report: dict) -> str:
    return json.dumps(report, indent=2, default=float)
