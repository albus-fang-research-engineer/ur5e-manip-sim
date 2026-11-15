"""The planner failure record VLM call #5 (repair v2) reads, built from a
planner's .report.json. Pure library: no planning, no VLM call.

Two report shapes:

    scripts/plan_task.py      {"status": "ok" | "failed", ...,
                               "failure": TaskPlanError.as_dict()}:
                              {"kind", "stage" (a role), "message"
                              ("[role] kind: text"), "detail"}
    scripts/plan_pour_tea.py  writes no .report.json: it stops with a
                              SystemExit line ("[transport] FAILED
                              no_admitted_handoff: ...", "[pour] planning
                              failed: ...", "[grasp] no reachable grasp — add
                              IK seeds ..."). A driver records that line as
                              {"status": "failed", "failure": "<line>"}
                              (score_authoring_ablation.last_failure extracts
                              it); the line is typed here with the same kinds
                              plan_task uses.

The record keeps what the repair model can act on: the stage by its
emission name (the name edit pointers use, not the role), the failure type,
the message, and the evidence (goal-funnel counts, collision offenders, the
handoff filter's counts and the next stage's compile reasons, the
CompileError slot). Paths, timings, seeds and per-candidate tables are
dropped.
"""

from __future__ import annotations

import math
import re

from .critic import _num

# detail keys kept as evidence, in this order; anything else is dropped
EVIDENCE_KEYS = (
    "slot",                                          # CompileError
    "n_sampled", "accept_rate", "n_ik", "n_collision_free", "n_contained",
    "n_rounds", "offenders",                         # goal funnel
    "draws", "checked", "admitted", "rejected",      # handoff filter
    "reasons",                                       # planning (walk_goals)
    "excess", "ik", "offender",                      # synthesized start
    "proposed", "classified", "ik_tried", "ik_feasible", "collision_free",
    "reachable", "funnel", "skips", "lookahead",     # grasp
)
_LOOKAHEAD_KEYS = ("score", "fractions")
_EXCESS_ROWS = (("x", "cm"), ("y", "cm"), ("z", "cm"),
                ("roll", "deg"), ("pitch", "deg"), ("yaw", "deg"))

# plan_pour_tea exit lines without a "FAILED <kind>:" tag -> plan_task kind
_POUR_TEA_KINDS = (
    ("no feasible goal configs", "empty_goal_set"),
    ("no feasible pour configs", "empty_goal_set"),
    ("planning failed", "planning"),
    ("no entry led to a pour plan", "planning"),
    ("post-grasp lift infeasible", "lift"),
    ("lacks role", "compile"),
)
_LINE_RE = re.compile(r"^\s*\[(?P<label>[^\]]+)\]\s*(?P<rest>.*)$", re.S)
_TAG_RE = re.compile(r"^FAILED\s+(?P<kind>[a-z_]+):\s*(?P<msg>.*)$", re.S)
_COMPILE_RE = re.compile(r"^(?:[\w.]+\.)?CompileError:\s*(?P<rest>.*)$", re.S)


def _stage_name(label: str, stage_names: dict[str, str]) -> str:
    """The stage name for a failure's role label. plan_pour_tea labels a
    stage by a prefix of its role ("transport" for transport_active); a
    label that names no role ("setup", or "grasp" when the grasp role is
    called otherwise) is kept as it is."""
    if label in stage_names:
        return stage_names[label]
    hits = [r for r in stage_names if r.startswith(label + "_")]
    return stage_names[hits[0]] if len(hits) == 1 else label


def _name_roles(text: str, stage_names: dict[str, str]) -> str:
    """Role identifiers inside a message (e.g. the next stage in
    no_admitted_handoff) -> their stage names, quoted. Only snake_case
    roles: a plain-word role ("pour") is also an English word."""
    for role, name in sorted(stage_names.items(), key=lambda kv: -len(kv[0])):
        if "_" in role:
            text = re.sub(rf"(?<![\w.]){re.escape(role)}(?![\w.])",
                          f"'{name}'", text)
    return text


def _excess(v) -> dict | None:
    """TSR.excess (x y z in m, roll pitch yaw in rad) -> the rows outside
    the path, readable like the report's bounds."""
    if not isinstance(v, (list, tuple)) or len(v) != 6:
        return None
    out = {}
    for (row, unit), x in zip(_EXCESS_ROWS, v):
        if x:
            val = 100 * x if unit == "cm" else math.degrees(x)
            out[row] = f"{_num(val, unit)} {unit}"
    return out


def _evidence(detail) -> dict:
    if not isinstance(detail, dict):
        return {}
    out = {}
    for k in EVIDENCE_KEYS:
        if k not in detail or detail[k] is None:
            continue
        v = detail[k]
        if k == "excess":
            v = _excess(v)
            if v is None:
                continue
        elif k == "lookahead" and isinstance(v, list):
            v = [{kk: row[kk] for kk in _LOOKAHEAD_KEYS if kk in row}
                 if isinstance(row, dict) else row for row in v]
        out[k] = v
    return out


def _from_line(line: str, stage_names: dict[str, str],
               exit_line: bool = True) -> dict:
    """A "[label] ..." failure line (plan_pour_tea's exit, or
    str(TaskPlanError) with exit_line False) -> {stage, type, message}.
    An exit line's operator advice after " — " (CLI flags to try) is
    dropped."""
    # plan_pour_tea does not catch a CompileError: its last line is the exception,
    # "...CompileError: <stage name>.<path|subgoal>.<rot|trans>[i]: <reason>"
    ce = _COMPILE_RE.match(line.strip())
    if ce:
        names = sorted(stage_names.values(), key=len, reverse=True)
        slot = ce["rest"].split(": ", 1)[0]
        stage = next((n for n in names if slot.startswith(n + ".")), None)
        return {"stage": stage, "type": "compile",
                "message": _name_roles(ce["rest"].strip(), stage_names),
                "evidence": {"slot": slot}}
    m = _LINE_RE.match(line)
    label, rest = (m["label"], m["rest"]) if m else (None, line)
    t = _TAG_RE.match(rest)
    if t:
        kind, msg = t["kind"], t["msg"]
    else:
        msg = rest
        kind = next((k for key, k in _POUR_TEA_KINDS if key in rest),
                    "grasp" if label == "grasp" else "unknown")
    msg = (msg.split(" — ")[0] if exit_line else msg).strip()
    return {"stage": _stage_name(label, stage_names) if label else None,
            "type": kind, "message": _name_roles(msg, stage_names)}


def _label_lookahead(rec: dict, mover_roles, stage_names: dict[str, str]) -> None:
    """The grasp lookahead's per-candidate fractions, labeled: for each mover
    stage (in plan order, as plan_task probes them), the best fraction of its
    probed goals reachable from any grasp candidate; a stage at 0 is named as
    the one no grasp can reach. Replaces the unlabeled list."""
    ev = rec.get("evidence") or {}
    rows = ev.get("lookahead")
    if not mover_roles or not isinstance(rows, list) or not rows:
        return
    fr = [r.get("fractions") for r in rows if isinstance(r, dict)]
    fr = [f for f in fr if isinstance(f, list) and len(f) == len(mover_roles)]
    if not fr:
        return
    names = [stage_names.get(r, r) for r in mover_roles]
    best = {n: round(max(f[i] for f in fr), 3) for i, n in enumerate(names)}
    ev.pop("lookahead")
    ev["best_reachable_fraction_from_any_grasp"] = best
    ev["grasp_candidates_probed"] = len(fr)
    zero = [n for n, b in best.items() if b == 0.0]
    if zero:
        ev["unreachable_stages"] = zero
        rec["message"] += ("; no grasp candidate reaches any probed goal of "
                           + ", ".join(f"'{n}'" for n in zero))


def failure_from_report(report_json: dict,
                        stage_names: dict[str, str],
                        mover_roles: list[str] | None = None,
                        log_text: str | None = None) -> dict | None:
    """A planner .report.json -> the failure record call #5 is shown:

        {"stage": <stage name>, "type": <failure kind>, "message": str,
         "evidence": {...}}             (evidence omitted when empty)

    None when the report says the plan succeeded. `stage_names` maps
    role -> the emission's stage name. `mover_roles` (the stages with a
    passive object, in plan order) labels a grasp failure's lookahead
    fractions by stage (_label_lookahead). `log_text` (the planner's log) adds
    the failing stage's goal funnel and collision offenders to a one-line
    failure (plan_pour_tea prints them on their own lines). The run record (not
    this record, which the model sees) carries signature(). A report with no outcome (status
    "running": the planner crashed) raises ValueError."""
    status = report_json.get("status")
    fail = report_json.get("failure")
    if status == "ok":
        return None
    if fail is None:
        raise ValueError(f"planner report has no failure record "
                         f"(status {status!r})")
    if isinstance(fail, str):
        rec = _from_line(fail, stage_names)
        if log_text and rec.get("stage"):
            ev = _funnel_from_log(log_text, rec["stage"], stage_names)
            if ev:
                rec["evidence"] = {**rec.get("evidence", {}), **ev}
        return rec
    if not isinstance(fail, dict):
        raise ValueError(f"planner report failure must be a dict or a "
                         f"line, not {type(fail).__name__}")
    rec = _from_line(str(fail.get("message", "")), stage_names,
                     exit_line=False)
    kind, role = fail.get("kind"), fail.get("stage")
    if kind:
        rec["type"] = kind
        # str(TaskPlanError) is "[role] kind: text": keep the text only
        if rec["message"].startswith(f"{kind}: "):
            rec["message"] = rec["message"][len(kind) + 2:]
    if role:
        rec["stage"] = _stage_name(role, stage_names)
    ev = _evidence(fail.get("detail"))
    if ev:
        rec["evidence"] = ev
        _label_lookahead(rec, mover_roles, stage_names)
    return rec


# plan_pour_tea prints a stage's goal funnel on its own lines and exits with a
# one-line failure; the funnel is the evidence plan_task puts in its report
_FUNNEL_RE = re.compile(r"^\[(?P<label>[^\]]+)\] goal funnel: (?P<n>\d+) sampled -> (?P<ik>\d+) "
                        r"IK-feasible -> (?P<free>\d+) collision-free -> (?P<cont>\d+) contained")
_OFFENDERS_RE = re.compile(r"^\[(?P<label>[^\]]+)\] collision offenders: (?P<list>.*)$")
_PAIR_RE = re.compile(r"(?P<pair>[\w.]+ <-> [\w.]+) x(?P<n>\d+)")


def _funnel_from_log(log_text: str, stage: str, stage_names: dict[str, str]) -> dict:
    """The failing stage's last goal funnel and collision offenders from a
    planner log (plan_pour_tea), in plan_task's evidence keys."""
    out: dict = {}
    for line in log_text.splitlines():
        m = _FUNNEL_RE.match(line.strip())
        if m and _stage_name(m["label"], stage_names) == stage:
            out.update(n_sampled=int(m["n"]), n_ik=int(m["ik"]),
                       n_collision_free=int(m["free"]), n_contained=int(m["cont"]))
            out.pop("offenders", None)           # offenders belong to this funnel
        m = _OFFENDERS_RE.match(line.strip())
        if m and _stage_name(m["label"], stage_names) == stage:
            out["offenders"] = {p["pair"]: int(p["n"]) for p in _PAIR_RE.finditer(m["list"])}
    return out


# which step a failure collapsed at (pilot-3 discussion, 2026-11-14): the study keeps
# collision / handoff / compile; unreachable and no_path are reported, not repaired
SIGNATURES = ("compile", "collision", "unreachable", "handoff", "no_path", "grasp", "other")


def signature(rec: dict) -> str:
    """The failure's signature from its type and goal-funnel evidence:
    collision (IK-feasible goals exist, none collision-free), unreachable
    (no IK-feasible goal), handoff (goals exist, none a start the next stage
    accepts), compile, no_path (goals found, the search found no path), grasp
    (no grasp survived; see the labeled lookahead), other."""
    t = rec.get("type")
    ev = rec.get("evidence") or {}
    if t == "compile":
        return "compile"
    if t == "no_admitted_handoff":
        return "handoff"
    if t in ("empty_goal_set", "synthesized_start_unreachable"):
        if ev.get("n_ik") == 0:
            return "unreachable"
        if ev.get("n_ik") and not ev.get("n_collision_free"):
            return "collision"
        return "collision" if ev.get("offenders") else "other"
    if t == "planning":
        return "no_path"
    if t == "grasp":
        return "grasp"
    return "other"
