"""Fault injection for the repair study (docs/deferred.md, D9): one edit to a
specification that plans and passes the grader, recorded with the ground
truth the scorer needs. Pure library: no planning, no rendering, no VLM.

A fault is data (tasks/repair_study/faults.json), keyed by role, not by
stage name, so one catalog serves every starting specification of a task:

    {"id": "pour_wrong_anchor", "task": "pour_tea", "category": "wrong_anchor",
     "role": "transport_active", "op": "set_point", "point": "handle_center",
     "expect": "...", "unfixable_by": ["render_check"]}

Ops (all on one role's call #2 selection or call #3 emission):

    set_field   one field of one row        ctx, table, index | match, field, value
    set_point   the role's call #2 point    point (a grounded symbol of the
                                            role's object, mapped to its
                                            candidate id in the pool)
    add_row     append a row                ctx, table, row (RotRow / TransTerm
                                            fields)

`match` picks the row by field equality ({"term": "above"}) and must match
exactly one row, so a catalog entry survives row reordering between
starting specifications. A fault that does not apply to the given
specification (no such role, no matching row, value already set, point not
in the pool) raises FaultNotApplicable; it is never silently skipped.

The record names the target with the critic's pointer grammar
(`<stage>.<ctx>.<table>[i]`, `<stage>.point.<object>`), which is the ground
truth for localization: did the first edit or resample hit this stage, this
row? Whether a fault really makes the specification infeasible is not
decided here: qualification plans the faulty specification and keeps the
fault only if planning fails.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .critic import candidate_for_point, row_text
from .selection import selection_from_json, selection_to_json
from .vlm import (CLEARANCES, ROT_TOLS, SLACKS, PointAxisSelection, RotRow,
                  StageEmission, TransTerm, emission_from_json)

CATEGORIES = ("clearance", "wrong_anchor", "handoff_rejected",
              "contradictory_rows", "tolerance_too_tight")
CONDITIONS = ("typed", "render_check", "untyped")
OPS = ("set_field", "set_point", "add_row")

# fields a set_field may change, and the values each may take
_FIELD_VALUES = {
    "clearance": CLEARANCES, "slack": SLACKS, "tol": ROT_TOLS + SLACKS,
    "axis": None, "reference": None, "relation": None, "anchor": None,
    "term": None,
}


class FaultNotApplicable(ValueError):
    """The fault cannot be injected into this specification as written."""


@dataclass(frozen=True)
class Fault:
    id: str
    task: str
    category: str
    role: str
    op: str
    ctx: str | None = None             # "path" | "subgoal"
    table: str | None = None           # "rot" | "trans"
    index: int | None = None
    match: dict | None = None
    field: str | None = None
    value: str | None = None
    point: str | None = None
    row: dict | None = None
    expect: str = ""                   # the typed failure we expect; logged, never sent
    unfixable_by: tuple[str, ...] = ()  # conditions that cannot fix it by construction
    note: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"fault {self.id}: category {self.category!r} not in {CATEGORIES}")
        if self.op not in OPS:
            raise ValueError(f"fault {self.id}: op {self.op!r} not in {OPS}")
        bad = [c for c in self.unfixable_by if c not in CONDITIONS]
        if bad:
            raise ValueError(f"fault {self.id}: unknown conditions {bad}")


@dataclass(frozen=True)
class FaultRecord:
    """What was injected, for the run record and the scorer."""
    fault_id: str
    task: str
    category: str
    role: str
    stage: str                 # the emission's stage name
    pointer: str               # critic pointer of the target
    before: str
    after: str
    expect: str
    unfixable_by: tuple[str, ...]


@dataclass(frozen=True)
class Spec:
    """A specification as the planners read it: role-keyed call #2
    selections (with each role's object) and call #3 emissions, plus the
    emissions artifact's other keys, kept so a written spec stays a valid
    artifact."""
    selections: dict[str, PointAxisSelection]
    objects: dict[str, str]
    emissions: dict[str, StageEmission]
    roles: tuple[str, ...]                      # emission order
    extra: dict = field(default_factory=dict)   # e.g. task, arm, compiled


def load_spec(selections_path, emissions_path) -> Spec:
    sel_doc = json.loads(Path(selections_path).read_text())
    em_doc = json.loads(Path(emissions_path).read_text())
    roles = tuple(em_doc["roles"])
    emissions = {r: emission_from_json(e) for r, e in zip(roles, em_doc["emissions"])}
    sels = {r: selection_from_json(d) for r, d in sel_doc.items()}
    objects = {r: d.get("object") or (sels[r].axis or "").partition(".")[0]
               for r, d in sel_doc.items()}
    extra = {k: v for k, v in em_doc.items() if k not in ("roles", "emissions")}
    return Spec(sels, objects, emissions, roles, extra)


def write_spec(spec: Spec, out_dir) -> tuple[Path, Path]:
    """selections.json and emissions.json in out_dir. The source
    artifact's compile-gate record ("compiled") is dropped: it described
    the specification before the fault, and load_emissions trusts it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = {r: {"object": spec.objects[r], **selection_to_json(s)}
           for r, s in spec.selections.items()}
    extra = {k: v for k, v in spec.extra.items() if k != "compiled"}
    em = {**extra, "roles": list(spec.roles),
          "emissions": [asdict(spec.emissions[r]) for r in spec.roles]}
    ps, pe = out / "selections.json", out / "emissions.json"
    ps.write_text(json.dumps(sel, indent=1) + "\n")
    pe.write_text(json.dumps(em, indent=1) + "\n")
    return ps, pe


def load_catalog(path, task: str | None = None) -> list[Fault]:
    doc = json.loads(Path(path).read_text())
    faults = [Fault(**{**f, "unfixable_by": tuple(f.get("unfixable_by", ()))})
              for f in doc["faults"]]
    ids = [f.id for f in faults]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate fault ids")
    return [f for f in faults if task is None or f.task == task]


def _rows(em: StageEmission, ctx: str, table: str):
    spec = em.path_tsr if ctx == "path" else em.subgoal_tsr
    return spec, list(getattr(spec, table))


def _with_rows(em: StageEmission, ctx: str, table: str, rows) -> StageEmission:
    spec, _ = _rows(em, ctx, table)
    spec = replace(spec, **{table: tuple(rows)})
    return replace(em, **{"path_tsr" if ctx == "path" else "subgoal_tsr": spec})


def _find_row(f: Fault, rows) -> int:
    if (f.index is None) == (f.match is None):
        raise ValueError(f"fault {f.id}: give exactly one of index or match")
    if f.index is not None:
        if not 0 <= f.index < len(rows):
            raise FaultNotApplicable(f"{f.id}: {f.ctx}.{f.table} has {len(rows)} rows, "
                                     f"no index {f.index}")
        return f.index
    hits = [i for i, r in enumerate(rows)
            if all(getattr(r, k, None) == v for k, v in f.match.items())]
    if len(hits) != 1:
        raise FaultNotApplicable(f"{f.id}: {f.ctx}.{f.table} rows matching {f.match}: "
                                 f"{len(hits)} (need exactly 1)")
    return hits[0]


def _new_row(table: str, d: dict):
    if table == "rot":
        return RotRow(**{"row": None, **d})
    return TransTerm(**{**d, "flags": tuple(d.get("flags", ()))})


def inject(spec: Spec, f: Fault, pools: dict[str, dict[int, dict]] | None = None
           ) -> tuple[Spec, FaultRecord]:
    """The faulty specification and its record. `pools`: object ->
    candidate pool (selection.load_pool), needed for set_point only."""
    if f.role not in spec.emissions:
        raise FaultNotApplicable(f"{f.id}: no emission for role {f.role!r} "
                                 f"(roles: {list(spec.roles)})")
    em = spec.emissions[f.role]

    if f.op == "set_point":
        if f.role not in spec.selections:
            raise FaultNotApplicable(f"{f.id}: no call #2 selection for role {f.role!r}")
        obj = spec.objects[f.role]
        if obj != em.active:
            raise FaultNotApplicable(f"{f.id}: role {f.role!r} selects on {obj!r}, "
                                     f"not the active object {em.active!r}")
        pool = (pools or {}).get(obj)
        if pool is None:
            raise ValueError(f"fault {f.id}: set_point needs the pool of {obj!r}")
        cid = candidate_for_point(pool, f.point)
        if cid is None:
            raise FaultNotApplicable(f"{f.id}: {obj}.{f.point} has no candidate in the pool")
        old = spec.selections[f.role]
        if int(old.candidate_id) == cid:
            raise FaultNotApplicable(f"{f.id}: {obj}.{f.point} is already selected")
        c_old = pool.get(int(old.candidate_id), {})
        before = (f"{obj}.{c_old['symbol']}" if c_old.get("symbol")
                  else f"mark {old.candidate_id}")
        sels = {**spec.selections, f.role: replace(
            old, candidate_id=cid, rationale=f"fault {f.id}: {obj}.{f.point}")}
        rec = FaultRecord(f.id, f.task, f.category, f.role, em.name,
                          f"{em.name}.point.{obj}", before, f"{obj}.{f.point}",
                          f.expect, f.unfixable_by)
        return replace(spec, selections=sels), rec

    if f.ctx not in ("path", "subgoal") or f.table not in ("rot", "trans"):
        raise ValueError(f"fault {f.id}: {f.op} needs ctx path|subgoal and table rot|trans")
    _, rows = _rows(em, f.ctx, f.table)

    if f.op == "add_row":
        new = _new_row(f.table, f.row or {})
        if f.table == "trans" and len(rows) == 1 and rows[0].term == "free":
            raise FaultNotApplicable(f"{f.id}: {f.ctx}.trans is a single free term; "
                                     "adding a row beside it would contradict 'free'")
        if f.table == "rot" and rows and all(r.relation == "free" for r in rows):
            raise FaultNotApplicable(f"{f.id}: {f.ctx}.rot is declared free; a constrained "
                                     "row beside it would be malformed, not the intended fault")
        idx = len(rows)
        rows.append(new)
        before, after = "(no row)", row_text(new)
    else:  # set_field
        if f.field not in _FIELD_VALUES:
            raise ValueError(f"fault {f.id}: field {f.field!r} not editable "
                             f"({sorted(_FIELD_VALUES)})")
        allowed = _FIELD_VALUES[f.field]
        if allowed is not None and f.value not in allowed:
            raise ValueError(f"fault {f.id}: {f.field} value {f.value!r} not in {allowed}")
        idx = _find_row(f, rows)
        old = rows[idx]
        if not hasattr(old, f.field):
            raise FaultNotApplicable(f"{f.id}: row '{row_text(old)}' has no {f.field}")
        if getattr(old, f.field) == f.value:
            raise FaultNotApplicable(f"{f.id}: {f.field} is already {f.value!r}")
        if getattr(old, f.field) is None:
            raise FaultNotApplicable(f"{f.id}: row '{row_text(old)}' does not use {f.field}")
        rows[idx] = replace(old, **{f.field: f.value})
        before, after = row_text(old), row_text(rows[idx])

    em2 = _with_rows(em, f.ctx, f.table, rows)
    rec = FaultRecord(f.id, f.task, f.category, f.role, em.name,
                      f"{em.name}.{f.ctx}.{f.table}[{idx}]", before, after,
                      f.expect, f.unfixable_by)
    return replace(spec, emissions={**spec.emissions, f.role: em2}), rec


def spec_diff(a: Spec, b: Spec) -> list[dict]:
    """Every row and point choice that differs between two specifications
    of the same task, by critic pointer (stage names from `a`; a stage
    renamed by a re-emit counts once, as the whole stage). The scorer's
    collateral damage is this diff minus the fault's own pointer."""
    out = []
    for role in a.roles:
        ea, eb = a.emissions.get(role), b.emissions.get(role)
        if eb is None:
            out.append({"pointer": ea.name, "before": "stage", "after": "(missing)"})
            continue
        if role in a.selections and role in b.selections and \
                a.selections[role].candidate_id != b.selections[role].candidate_id:
            out.append({"pointer": f"{ea.name}.point.{a.objects[role]}",
                        "before": f"mark {a.selections[role].candidate_id}",
                        "after": f"mark {b.selections[role].candidate_id}"})
        for ctx in ("path", "subgoal"):
            for table in ("rot", "trans"):
                ra, rb = _rows(ea, ctx, table)[1], _rows(eb, ctx, table)[1]
                for i in range(max(len(ra), len(rb))):
                    x = row_text(ra[i]) if i < len(ra) else "(no row)"
                    y = row_text(rb[i]) if i < len(rb) else "(no row)"
                    if x != y:
                        out.append({"pointer": f"{ea.name}.{ctx}.{table}[{i}]",
                                    "before": x, "after": y})
    return out
