"""VLM touchpoint-#1 driver: plan_stages call(s) written as artifacts and
VERIFIED against what the pipeline downstream can actually consume.

Three layers of verification, offline-testable (tests/test_plan_stages.py)
and run live here:

  parser      already in manip_sim.vlm (closed object vocabulary, typed
              stages) — not re-checked here
  grounding   every emitted free-text part must ground to >= 1 candidate
              in the active/passive object's candidates.json, through the
              SAME matcher vlm_subset uses (selection._part_match). This
              is the check that matters: vlm_subset degrades silently — an
              ungroundable token ("grip", "lip") still yields a full menu
              with zero part-biased marks, and call #2 never learns the
              stage plan failed to ground.
  contract    the structure select_frames.py hand-authors today
              (OBJECT_PARTS + ROLES), restated over POOL TAGS and loaded
              from tasks/<task>.json (--task-spec): coverage, one emitted
              stage per planner role, and role-level ordering (grasp
              before every interaction role). Passing means the emitted
              plan could replace the fixed ROLES.

Scene objects come from scenes/<scene>.json (--scene); the manifest is
ground truth for verification only and is not shown to the model beyond
the symbol vocabulary (until call #1 becomes mark-addressed).

Two input modes, one verifier:

  --marks DIR   mark-addressed (default pipeline, OmniManip order): the
                model sees marked.png and emits mark ids + labels; the
                verifier maps ids -> manifest names through marks.gt.json
                (sim only) and adds an identity[] check per role. Build
                the directory with scripts/mark_scene.py.
  (no --marks)  text-only ablation arm: object names + symbols in the
                prompt, no image. This is the arm the first 5/5 was
                measured on; it isolates decomposition from perception
                and must not be compared to image-conditioned baselines.

Artifacts (per run k): <out>.json (plan + per-object parts handoff +
checks + ROLE BINDINGS), <out>.log.json (Client.logs + raw transport text
per attempt).

The artifact is the #1 -> #2/#3 handoff, not just a verification record:
`plan_grounded` is the plan over manifest names (marks resolved through
marks.gt.json in sim; raw labels on hardware), `object_parts` the
per-object part union, and `roles` maps each planner role of the task
contract to (object, emitted stage index). select_frames.py and
emit_constraints.py read these via load_bindings() instead of their
hand-authored ROLES / STAGES tables:

    PYTHONPATH=. python scripts/select_frames.py --stage-plan outputs/stage_plan/pour_tea.marks.json

    PYTHONPATH=. python scripts/plan_stages.py --marks outputs/marks/pour_tea
    PYTHONPATH=. python scripts/plan_stages.py                 # text-only arm
    PYTHONPATH=. python scripts/plan_stages.py --repeat 5      # pass-rate table
    PYTHONPATH=. python scripts/plan_stages.py --replay outputs/stage_plan/pour_tea.0.json
    PYTHONPATH=. python scripts/plan_stages.py --scene scenes/X.json --task-spec tasks/X.json

--replay re-verifies saved artifacts offline (no API key) and REWRITES
the artifact (checks, object_parts, roles) against the pools it is run
with, so the handoff downstream reads always reflects the pools it will
be selected from; --repeat is the repeatability measurement at the
module's fixed TEMPERATURE.

Gate policy. All checks are always computed and reported; which of them
BLOCK (nonzero exit, load_bindings refusal) depends on where in the
pipeline the verifier runs:

  contract checks   identity[], coverage[], role[], ordering[] — always
                    block. They are the planner-side requirement.
  grounding checks  grounds[], menu[] — block by default, and in the
                    --no-ground (authored-pool) arm; advisory under
                    --defer-grounding, which run_pipeline passes at the
                    `plan` step of the runtime-grounding arm: there the
                    pool is PRODUCED from call #1's part names by
                    ground_parts.py, so checking the names against the
                    authored pool first would reject every part name
                    outside the authored band vocabulary ("body",
                    "interior") before the step that grounds them runs.

After ground_parts.py, parts the provider could not ground (listed in
<grounding>/grounding.json) are pruned from the plan at --replay
--grounding and recorded as `parts_dropped`; whether the drop is fatal
is then decided by the contract checks (a dropped "handle" fails
coverage[teapot]/role[grasp]; a dropped "body" fails nothing). This is
the one place the pipeline narrows a plan, and it is visible in the
artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from manip_sim.selection import _part_match, load_pool, vlm_subset
from manip_sim.perception.marks import load_gt, load_marks
from manip_sim.vlm import (Client, ObjectRef, StagePlan, StageSpec,
                           Vocabulary, _urllib_transport)
from manip_sim.scene import add_scene_arg, load_scene

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASK_SPEC = REPO / "tasks/pour_tea.json"
OUT = Path("outputs/stage_plan/pour_tea.json")


# --------------------------------------------------------- task contract

@dataclass(frozen=True)
class TaskSpec:
    """The planner-side contract a stage plan must satisfy, as data
    (tasks/<name>.json) so a scene variant can swap it (a lid-on variant
    adds a `remove_lid` role and an ordering edge).

      required_tags  object -> pool tags some attributed part must reach
      roles          role -> (active, passive, {object: tags}); "*" = any
      ordering       [{before: role, after: [role, ...]}] over role hits
      role_after     role -> role it must bind STRICTLY LATER than. Needed
                     when two roles share one predicate (transport_active
                     and pour both = teapot->mug on spout): without it
                     both bind the first matching stage and the later
                     emitted stage is never used.
    """
    name: str
    required_tags: dict[str, tuple[str, ...]]
    roles: dict[str, tuple[str, str | None, dict[str, tuple[str, ...]]]]
    ordering: tuple[tuple[str, tuple[str, ...]], ...]
    role_after: dict[str, str] = field(default_factory=dict)


def load_task_spec(path: str | Path = DEFAULT_TASK_SPEC) -> TaskSpec:
    doc = json.loads(Path(path).read_text())
    return TaskSpec(
        name=doc["name"],
        required_tags={o: tuple(t) for o, t in doc["required_tags"].items()},
        roles={r: (v["active"], v.get("passive"),
                   {o: tuple(t) for o, t in v["tags"].items()})
               for r, v in doc["roles"].items()},
        ordering=tuple((e["before"], tuple(e["after"]))
                       for e in doc.get("ordering", [])),
        role_after={r: v["after"] for r, v in doc["roles"].items()
                    if v.get("after")})


TASK_SPEC = load_task_spec()
# scene default task string; --task overrides
TASK = json.loads((REPO / "scenes/pour_tea.json").read_text())["task"]

Check = tuple[str, bool, str]
GROUNDING_CHECKS = ("grounds", "menu")   # advisory under --defer-grounding


def blocking(checks: list[Check], defer_grounding: bool = False) -> list[str]:
    """Failed check names that block the handoff under the gate policy."""
    return [c for c, ok, _ in checks if not ok
            and not (defer_grounding and c.split("[")[0] in GROUNDING_CHECKS)]


def read_dropped(grounding: Path | None) -> dict[str, tuple[str, ...]]:
    """object -> parts ground_parts.py could not ground (its
    grounding.json), or {} when not grounding / nothing dropped."""
    if grounding is None:
        return {}
    f = Path(grounding) / "grounding.json"
    if not f.exists():
        return {}
    doc = json.loads(f.read_text())
    return {o: tuple(ps) for o, ps in doc.get("ungrounded", {}).items() if ps}


def prune_parts(plan: StagePlan, drop: dict[str, tuple[str, ...]]) -> StagePlan:
    """Remove per-object parts from every stage (over manifest names)."""
    if not drop:
        return plan
    stages = tuple(StageSpec(
        index=s.index, name=s.name, active=s.active, passive=s.passive,
        parts={o: tuple(p for p in ps if p not in drop.get(o, ()))
               for o, ps in s.parts.items()}) for s in plan.stages)
    return StagePlan(task=plan.task, stages=stages, objects=plan.objects)


# ------------------------------------------------------------ grounding

def _grounds_in(part: str, pool: dict[int, dict]) -> bool:
    return any(_part_match(part, c) for c in pool.values())


def attribute(stage: StageSpec, pools: dict[str, dict[int, dict]]
              ) -> tuple[dict[str, tuple[str, ...]], list[str], list[str]]:
    """(parts_by_object, ungrounded, unknown_object). Parts are already
    object-keyed by the schema; this checks each grounds in ITS object's
    pool. Objects with no pool (handle did not map to a registered
    asset) are reported as unknown rather than raising."""
    ungrounded, unknown = [], []
    by_obj = {o: stage.parts.get(o, ()) for o in stage.objects()}
    for o, ps in by_obj.items():
        if o not in pools:
            unknown.append(o)
            continue
        ungrounded += [f"{o}.{p}" for p in ps if not _grounds_in(p, pools[o])]
    return by_obj, ungrounded, unknown


def object_parts(plan: StagePlan, pools: dict[str, dict[int, dict]]
                 ) -> dict[str, tuple[str, ...]]:
    """The #1 -> #2 handoff: per-object parts union (what select_frames'
    OBJECT_PARTS hand-authors), order-preserving, deduplicated."""
    acc: dict[str, list[str]] = {o: [] for o in pools}
    for s in plan.stages:
        for o, ps in s.parts.items():
            for p in ps:
                if p not in acc.setdefault(o, []):
                    acc[o].append(p)
    return {o: tuple(v) for o, v in acc.items()}


def _tag_covered(tag: str, parts: tuple[str, ...], pool: dict[int, dict]
                 ) -> bool:
    """Some attributed part grounds to a candidate carrying `tag`. A tag
    may list alternatives ("rim|opening"): under runtime grounding the
    pool's part labels are whatever call #1 named the part, so the
    contract names the acceptable spellings instead of one band name."""
    alts = [t.strip().lower() for t in tag.split("|")]
    tagged = [c for c in pool.values()
              if str(c.get("part") or "").lower() in alts
              or str(c.get("symbol") or "").lower() in alts
              or any(str(c.get("part") or "").lower().startswith(a) for a in alts)]
    return any(_part_match(p, c) for p in parts for c in tagged)


# --------------------------------------------------------------- verify

def match_roles(plan: StagePlan, pools: dict[str, dict[int, dict]],
                spec: TaskSpec = TASK_SPEC) -> dict[str, list[int]]:
    """role -> emitted stage indices satisfying (active, passive, tags).
    The single matching rule behind BOTH the role[] check and the
    planner binding, so a plan that verifies is by construction one
    that binds."""
    per_stage = {s.index: attribute(s, pools)[0] for s in plan.stages}
    out: dict[str, list[int]] = {}
    for role, (act, pas, need) in spec.roles.items():
        hits = []
        for s in plan.stages:
            if act != "*" and s.active != act:
                continue
            if pas != "*" and s.passive != pas:
                continue
            by_obj = per_stage[s.index]
            if all(obj in by_obj and obj in pools and all(
                    _tag_covered(t, by_obj[obj], pools[obj]) for t in tags)
                   for obj, tags in need.items()):
                hits.append(s.index)
        out[role] = hits
    return out


def bind_roles(plan: StagePlan, pools: dict[str, dict[int, dict]],
               spec: TaskSpec = TASK_SPEC) -> dict[str, dict]:
    """The planner handoff: role -> {"object", "stage"}. `object` is the
    object whose tags the role demands (the frame call #2 must place on
    it); `stage` is the FIRST emitted stage matching the role — or, for
    a role with `after`, the first match strictly later than the stage
    that role bound (None when unmatched / none later). Roles whose
    contract names more than one object are ambiguous as frame roles
    and rejected here rather than guessed."""
    idx = match_roles(plan, pools, spec)
    out: dict[str, dict] = {}
    pending = list(spec.roles.items())
    while pending:                      # resolve `after` dependencies first
        role, (_act, _pas, need) = pending.pop(0)
        if len(need) != 1:
            raise ValueError(f"role {role!r} names {len(need)} objects; a "
                             "frame role binds exactly one")
        dep = spec.role_after.get(role)
        if dep and dep not in out:
            if dep not in spec.roles:
                raise ValueError(f"role {role!r} is after unknown role {dep!r}")
            pending.append((role, (_act, _pas, need)))
            continue
        hits = idx[role]
        if dep is not None:
            floor = out[dep]["stage"]
            hits = [i for i in hits if floor is not None and i > floor]
        out[role] = {"object": next(iter(need)),
                     "stage": hits[0] if hits else None}
    return out


def verify(plan: StagePlan, pools: dict[str, dict[int, dict]],
           spec: TaskSpec = TASK_SPEC) -> list[Check]:
    """All checks always run; (name, passed, detail)."""
    out: list[Check] = []
    per_stage = {s.index: attribute(s, pools) for s in plan.stages}

    # -- identity (mark mode): every used object resolved to a manifest
    #    name, i.e. the model picked a mark that IS a scene object
    for h, ref in plan.objects.items():
        if ref.mark is not None:
            out.append((f"identity[{h}]", h in pools,
                        f"mark {ref.mark} labelled {ref.label!r} -> {h}"
                        + ("" if h in pools else " (no registered asset)")))

    # -- grounding: every part grounds in its own object's pool
    for s in plan.stages:
        by_obj, ungrounded, unknown = per_stage[s.index]
        detail = "parts " + json.dumps({o: list(p) for o, p in by_obj.items()})
        if unknown:
            detail += f" — UNKNOWN OBJECT {unknown}"
        if ungrounded:
            detail += f" — UNGROUNDED {ungrounded}"
        out.append((f"grounds[stage{s.index}]", not (ungrounded or unknown),
                    f"{s.name}: " + detail))

    # -- menu: the subset call #2 would see carries a mark for each part
    parts = object_parts(plan, pools)
    for obj, ps in parts.items():
        if not ps or obj not in pools:
            continue
        sub = vlm_subset(pools[obj], list(ps))
        blind = [p for p in ps
                 if not any(_part_match(p, c) for c in sub.values())]
        out.append((f"menu[{obj}]", not blind,
                    f"parts {list(ps)} -> {len(sub)}-mark menu"
                    + (f" — NO MARK for {blind}" if blind else "")))

    # -- coverage: required pool tags reached per object
    for obj, tags in spec.required_tags.items():
        missing = [t for t in tags
                   if obj not in pools
                   or not _tag_covered(t, parts.get(obj, ()), pools[obj])]
        out.append((f"coverage[{obj}]", not missing,
                    f"needs tags {list(tags)}, attributed {list(parts.get(obj, ()))}"
                    + (f" — MISSING {missing}" if missing else "")))

    # -- roles: one emitted stage per planner role
    role_idx = match_roles(plan, pools, spec)
    for role, (act, pas, need) in spec.roles.items():
        hits = role_idx[role]
        names = [f"{i}:{next(x.name for x in plan.stages if x.index == i)}"
                 for i in hits]
        out.append((f"role[{role}]", bool(hits),
                    f"matched {names}" if hits
                    else f"no stage with active={act} passive={pas} {need}"))

    # -- ordering: role-level precedence (first hit of `before` precedes
    #    the first hit of every `after` role)
    for before, afters in spec.ordering:
        b = role_idx.get(before, [])
        a = {r: role_idx.get(r, []) for r in afters}
        ok = bool(b) and all(v and min(b) < min(v) for v in a.values())
        out.append((f"ordering[{before}<{','.join(afters)}]", ok,
                    f"{before} at {b}, " + ", ".join(f"{r} at {v}" for r, v in a.items())))
    return out


# ------------------------------------------------------------- plumbing

class RecordingTransport:
    """Wraps a transport so raw model text survives into the log —
    Client.logs keeps rejections only, and a verification artifact needs
    what the model said on the attempt that PASSED the parser too."""

    def __init__(self, inner=None):
        self.inner = inner or _urllib_transport
        self.raw: list[str] = []

    def __call__(self, payload: dict) -> str:
        text = self.inner(payload)
        self.raw.append(text)
        return text


def plan_from_artifact(path: Path) -> StagePlan:
    doc = json.loads(Path(path).read_text())
    return plan_from_artifact_doc(doc.get("plan", doc))


@dataclass(frozen=True)
class Bindings:
    """What downstream consumes from a call-#1 artifact: the grounded
    plan, the per-object part union, and role -> (object, StageSpec).
    Roles are the task contract's; the StageSpec is the emitted stage
    the role matched, RESTRICTED to the role's object's parts so call
    #2's prompt names the parts on the object whose menu it sees."""
    plan: StagePlan
    object_parts: dict[str, tuple[str, ...]]
    roles: dict[str, tuple[str, StageSpec]]
    path: Path


def load_bindings(path: str | Path) -> Bindings:
    """Read a plan_stages.py artifact as the #1 -> #2/#3 handoff. Refuses
    artifacts whose checks failed or whose roles did not bind — the
    consumer must never plan on a plan the contract rejected."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if "roles" not in doc or "plan_grounded" not in doc:
        raise SystemExit(f"[plan-stages] {path} predates role bindings; "
                         "re-run plan_stages.py (or --replay it) to regenerate")
    gate = doc.get("gate")
    bad = (gate["blocking"] if gate is not None       # recorded policy
           else [c["check"] for c in doc["checks"] if not c["pass"]])
    if bad:
        raise SystemExit(f"[plan-stages] {path} failed checks {bad}; "
                         "refusing to hand it downstream")
    gplan = plan_from_artifact_doc(doc["plan_grounded"])
    by_idx = {s.index: s for s in gplan.stages}
    roles: dict[str, tuple[str, StageSpec]] = {}
    for role, b in doc["roles"].items():
        if b["stage"] is None:
            raise SystemExit(f"[plan-stages] {path}: role {role!r} unbound")
        obj, s = b["object"], by_idx[b["stage"]]
        roles[role] = (obj, StageSpec(
            index=s.index, name=s.name, active=s.active, passive=s.passive,
            parts={obj: tuple(s.parts.get(obj, ()))}))
    return Bindings(plan=gplan,
                    object_parts={o: tuple(v) for o, v in doc["object_parts"].items()},
                    roles=roles, path=path)


def plan_from_artifact_doc(doc: dict) -> StagePlan:
    stages = tuple(StageSpec(
        index=s["index"], name=s["name"], active=s["active"], passive=s["passive"],
        parts={o: tuple(p) for o, p in s["parts"].items()}) for s in doc["stages"])
    objects = {h: ObjectRef(**o) for h, o in doc.get("objects", {}).items()}
    return StagePlan(task=doc["task"], stages=stages, objects=objects)


def to_ground_truth(plan: StagePlan, gt: dict[int, str] | None) -> StagePlan:
    """Mark mode: rewrite handles to manifest names so the contract
    (written over names) applies. Marks without ground truth keep their
    handle and fail identity[]."""
    if not gt or not plan.objects:
        return plan
    return plan.relabel({h: gt[o.mark] for h, o in plan.objects.items()
                         if o.mark in gt})


def run_one(plan: StagePlan, pools: dict[str, dict[int, dict]],
            out: Path, logs: list | None = None, raw: list[str] | None = None,
            spec: TaskSpec = TASK_SPEC, gt: dict[int, str] | None = None,
            mode: str = "text", views: list[Path] | None = None,
            defer_grounding: bool = False,
            dropped: dict[str, tuple[str, ...]] | None = None) -> list[Check]:
    gplan = prune_parts(to_ground_truth(plan, gt), dropped or {})
    checks = verify(gplan, pools, spec)
    parts = object_parts(gplan, pools)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "plan": asdict(plan),                      # as emitted (handles)
        "plan_grounded": asdict(gplan),            # over manifest names, pruned
        "parts_dropped": {o: list(v) for o, v in (dropped or {}).items()},
        "mode": mode,
        "views": [str(v) for v in (views or [])],
        "task_spec": spec.name,
        "object_parts": {o: list(v) for o, v in parts.items()},
        "roles": bind_roles(gplan, pools, spec),
        "checks": [{"check": c, "pass": ok, "detail": d} for c, ok, d in checks],
        "gate": {"defer_grounding": defer_grounding,
                 "blocking": blocking(checks, defer_grounding)},
    }, indent=2) + "\n")
    if logs is not None:
        out.with_suffix(".log.json").write_text(json.dumps({
            "client_logs": [asdict(l) for l in logs],
            "raw_responses": raw or [],
        }, indent=2, default=str) + "\n")
    return checks


def print_report(plan: StagePlan, checks: list[Check], parts: dict,
                 roles: dict | None = None, defer_grounding: bool = False,
                 dropped: dict | None = None) -> None:
    print(f"[plan-stages] task: {plan.task}")
    for h, o in plan.objects.items():
        if o.mark is not None:
            print(f"  mark {o.mark} -> {h!r} ({o.label})")
    for s in plan.stages:
        arrow = f" -> {s.passive}" if s.passive else ""
        print(f"  [{s.index}] {s.name}: {s.active}{arrow}, parts "
              f"{ {o: list(p) for o, p in s.parts.items()} }")
    if dropped:
        print(f"  parts dropped (not grounded): "
              f"{ {o: list(p) for o, p in dropped.items()} }")
    print("[plan-stages] verification:")
    block = set(blocking(checks, defer_grounding))
    for check, ok, detail in checks:
        tag = "PASS" if ok else ("FAIL" if check in block else "WARN")
        print(f"  {tag}  {check}: {detail}")
    if defer_grounding:
        print("  (grounds[]/menu[] advisory: pool not yet grounded — "
              "re-verified after ground_parts.py)")
    print("[plan-stages] #1 -> #2 handoff (select_frames.OBJECT_PARTS equivalent):")
    for obj, ps in parts.items():
        if ps:
            print(f"  MUJOCO_GL=osmesa PYTHONPATH=. python "
                  f"scripts/render_candidates.py --object {obj} --vlm "
                  f"--parts {' '.join(ps)}")
    if roles:
        print("[plan-stages] role bindings (select_frames.ROLES equivalent):")
        for role, b in roles.items():
            print(f"  {role:18s} -> {b['object']} @ stage {b['stage']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    ap.add_argument("--repeat", type=int, default=1,
                    help="live calls to make; artifacts get a .k suffix")
    ap.add_argument("--replay", nargs="+", metavar="JSON",
                    help="verify saved artifacts offline; no API call")
    ap.add_argument("--task-spec", default=str(DEFAULT_TASK_SPEC), metavar="JSON",
                    help="planner contract the plan is verified against")
    ap.add_argument("--defer-grounding", action="store_true",
                    help="grounds[]/menu[] advisory (pool not produced yet); "
                         "contract checks still block")
    ap.add_argument("--marks", default=None, metavar="DIR",
                    help="mark set from scripts/mark_scene.py -> mark-addressed "
                         "mode with the scene image; absent -> text-only arm")
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene, getattr(args, "grounding", None))
    spec = load_task_spec(args.task_spec)
    if args.task == TASK:
        args.task = scene.task

    pools = {n: load_pool(d) for n, d in scene.asset_dirs.items()}
    mode = "marks" if args.marks else "text"
    gt = None
    if args.marks:
        markset = load_marks(args.marks)
        gt_path = Path(args.marks) / "marks.gt.json"
        gt = load_gt(gt_path) if gt_path.exists() else None
        if gt is None:
            print("[plan-stages] no marks.gt.json: identity/contract checks "
                  "run over raw handles (hardware mode)")
    base = Path(args.out)
    if args.out == str(OUT):                       # default -> tag by mode
        base = base.with_name(f"{base.stem}.{mode}{base.suffix}")

    dropped = read_dropped(scene.grounding)
    runs: list[list[Check]] = []
    if args.replay:
        for p in args.replay:
            doc = json.loads(Path(p).read_text())
            plan = plan_from_artifact_doc(doc.get("plan", doc))
            checks = run_one(plan, pools, Path(p), None, None, spec, gt,
                             doc.get("mode", mode),
                             [Path(v) for v in doc.get("views", [])],
                             args.defer_grounding, dropped)
            gplan = prune_parts(to_ground_truth(plan, gt), dropped)
            print(f"\n=== replay {p} (rewritten)")
            print_report(plan, checks, object_parts(gplan, pools),
                         bind_roles(gplan, pools, spec),
                         args.defer_grounding, dropped)
            runs.append(checks)
    else:
        if args.marks:
            vocab = Vocabulary.from_marks(markset)
            views = [markset.dir / markset.marked]
        else:
            vocab = Vocabulary.from_asset_dirs(scene.asset_dirs)
            views = None
        for k in range(args.repeat):
            transport = RecordingTransport()
            client = Client(transport=transport)
            plan = client.plan_stages(args.task, vocab, views)
            out = base if args.repeat == 1 else base.with_name(
                f"{base.stem}.{k}{base.suffix}")
            checks = run_one(plan, pools, out, client.logs, transport.raw,
                             spec, gt, mode, views, args.defer_grounding, dropped)
            gplan = prune_parts(to_ground_truth(plan, gt), dropped)
            print(f"\n=== run {k} -> {out}")
            print_report(plan, checks, object_parts(gplan, pools),
                         bind_roles(gplan, pools, spec),
                         args.defer_grounding, dropped)
            runs.append(checks)

    if len(runs) > 1:
        print("\n[plan-stages] pass rate per check:")
        names = [c for c, _, _ in runs[0]]
        for name in names:
            n = sum(1 for r in runs for c, ok, _ in r if c == name and ok)
            print(f"  {n}/{len(runs)}  {name}")
        allpass = sum(1 for r in runs if all(ok for _, ok, _ in r))
        print(f"  {allpass}/{len(runs)}  ALL")
        if args.defer_grounding:
            gated = sum(1 for r in runs if not blocking(r, True))
            print(f"  {gated}/{len(runs)}  GATE (contract checks only)")

    failed = [c for r in runs for c in blocking(r, args.defer_grounding)]
    if failed:
        sys.exit(f"[plan-stages] FAILED checks: {sorted(set(failed))} — "
                 "the emitted plan cannot replace select_frames.ROLES")
    print("\n[plan-stages] contract holds: emitted plan grounds and reproduces "
          "the hand-authored structure")


if __name__ == "__main__":
    main()
