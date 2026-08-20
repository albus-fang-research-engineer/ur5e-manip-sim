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
  contract    the pour-tea structure select_frames.py hand-authors today
              (OBJECT_PARTS + ROLES), restated over POOL TAGS: coverage,
              grasp-before-interaction ordering, and one emitted stage per
              planner role. Passing means the emitted plan could replace
              the fixed ROLES.

Part attribution: StageSpec.parts is a flat list with no object key (the
schema is shared with the #2 prompt), so each part is attributed to the
stage's active or passive object by which pool it grounds in. A part that
grounds in both (e.g. "handle" — both objects carry the tag) is attributed
to the active object and FLAGGED; the durable fix is an object-keyed
`parts` in the schema, which hardware GroundedSAM seeding needs anyway.

Artifacts (per run k): <out>.json (plan + per-object parts handoff +
checks), <out>.log.json (Client.logs + raw transport text per attempt).

    PYTHONPATH=. python scripts/plan_stages.py                 # 1 live call
    PYTHONPATH=. python scripts/plan_stages.py --repeat 5      # pass-rate table
    PYTHONPATH=. python scripts/plan_stages.py --replay outputs/stage_plan/pour_tea.0.json

--replay re-verifies saved artifacts offline (no API key); --repeat is
the repeatability measurement at the module's fixed TEMPERATURE.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from manip_sim.selection import _part_match, load_pool, vlm_subset
from manip_sim.vlm import (Client, StagePlan, StageSpec, Vocabulary,
                           _urllib_transport)

OBJECTS = {
    "teapot": Path("assets/objects/teapot"),
    "mug": Path("assets/objects/mug"),
}
TASK = "pour tea from the teapot into the mug"
OUT = Path("outputs/stage_plan/pour_tea.json")

# Pour-tea contract over POOL TAGS (candidates.json `part` / frames.json
# symbol names), mirroring select_frames.OBJECT_PARTS / ROLES.
REQUIRED_TAGS: dict[str, tuple[str, ...]] = {
    "teapot": ("handle", "spout"),
    "mug": ("rim",),
}
# role -> (active, passive, {object: required tags}); passive "*" = any
ROLE_SIGNATURES: dict[str, tuple[str, str | None, dict[str, tuple[str, ...]]]] = {
    "grasp": ("teapot", None, {"teapot": ("handle",)}),
    "transport_active": ("teapot", "mug", {"teapot": ("spout",)}),
    "pour": ("teapot", "mug", {"teapot": ("spout",)}),
    "transport_passive": ("*", "*", {"mug": ("rim",)}),
}

Check = tuple[str, bool, str]


# ---------------------------------------------------------- attribution

def _grounds_in(part: str, pool: dict[int, dict]) -> bool:
    return any(_part_match(part, c) for c in pool.values())


def attribute(stage: StageSpec, pools: dict[str, dict[int, dict]]
              ) -> tuple[dict[str, tuple[str, ...]], list[str], list[str]]:
    """(parts_by_object, ungrounded, ambiguous) for one stage. Parts are
    attributed to the stage's objects by which pool they ground in."""
    objs = [stage.active] + ([stage.passive] if stage.passive else [])
    by_obj: dict[str, list[str]] = {o: [] for o in objs}
    ungrounded, ambiguous = [], []
    for p in stage.parts:
        hits = [o for o in objs if _grounds_in(p, pools[o])]
        if not hits:
            ungrounded.append(p)
        elif len(hits) > 1:
            ambiguous.append(p)
            by_obj[stage.active].append(p)
        else:
            by_obj[hits[0]].append(p)
    return {o: tuple(v) for o, v in by_obj.items()}, ungrounded, ambiguous


def object_parts(plan: StagePlan, pools: dict[str, dict[int, dict]]
                 ) -> dict[str, tuple[str, ...]]:
    """The #1 -> #2 handoff: per-object parts union (what select_frames'
    OBJECT_PARTS hand-authors), order-preserving, deduplicated."""
    acc: dict[str, list[str]] = {o: [] for o in pools}
    for s in plan.stages:
        by_obj, _, _ = attribute(s, pools)
        for o, ps in by_obj.items():
            for p in ps:
                if p not in acc[o]:
                    acc[o].append(p)
    return {o: tuple(v) for o, v in acc.items()}


def _tag_covered(tag: str, parts: tuple[str, ...], pool: dict[int, dict]
                 ) -> bool:
    """Some attributed part grounds to a candidate carrying `tag`."""
    tagged = [c for c in pool.values()
              if str(c.get("part") or "").lower() == tag
              or str(c.get("symbol") or "").lower() == tag]
    return any(_part_match(p, c) for p in parts for c in tagged)


# --------------------------------------------------------------- verify

def verify(plan: StagePlan, pools: dict[str, dict[int, dict]]) -> list[Check]:
    """All checks always run; (name, passed, detail)."""
    out: list[Check] = []
    per_stage = {s.index: attribute(s, pools) for s in plan.stages}

    # -- grounding: every part grounds in its stage's objects
    for s in plan.stages:
        by_obj, ungrounded, ambiguous = per_stage[s.index]
        detail = "attributed " + json.dumps(by_obj)
        if ambiguous:
            detail += f" — AMBIGUOUS (both pools) {ambiguous}, took active"
        if ungrounded:
            detail += f" — UNGROUNDED {ungrounded}"
        out.append((f"grounds[stage{s.index}]", not ungrounded,
                    f"{s.name}: " + detail))

    # -- menu: the subset call #2 would see carries a mark for each part
    parts = object_parts(plan, pools)
    for obj, ps in parts.items():
        if not ps:
            continue
        sub = vlm_subset(pools[obj], list(ps))
        blind = [p for p in ps
                 if not any(_part_match(p, c) for c in sub.values())]
        out.append((f"menu[{obj}]", not blind,
                    f"parts {list(ps)} -> {len(sub)}-mark menu"
                    + (f" — NO MARK for {blind}" if blind else "")))

    # -- coverage: required pool tags reached per object
    for obj, tags in REQUIRED_TAGS.items():
        missing = [t for t in tags
                   if not _tag_covered(t, parts.get(obj, ()), pools[obj])]
        out.append((f"coverage[{obj}]", not missing,
                    f"needs tags {list(tags)}, attributed {list(parts.get(obj, ()))}"
                    + (f" — MISSING {missing}" if missing else "")))

    # -- ordering: teapot-only (grasp) precedes teapot->mug (interaction)
    grasp_idx = [s.index for s in plan.stages
                 if s.active == "teapot" and s.passive is None]
    inter_idx = [s.index for s in plan.stages
                 if s.active == "teapot" and s.passive == "mug"]
    ok = bool(grasp_idx) and bool(inter_idx) and min(grasp_idx) < min(inter_idx)
    out.append(("ordering[grasp<interaction]", ok,
                f"teapot-only at {grasp_idx}, teapot->mug at {inter_idx}"))

    # -- roles: one emitted stage per planner role
    for role, (act, pas, need) in ROLE_SIGNATURES.items():
        hits = []
        for s in plan.stages:
            if act != "*" and s.active != act:
                continue
            if pas != "*" and s.passive != pas:
                continue
            by_obj = per_stage[s.index][0]
            if all(obj in by_obj and all(
                    _tag_covered(t, by_obj[obj], pools[obj]) for t in tags)
                   for obj, tags in need.items()):
                hits.append(f"{s.index}:{s.name}")
        out.append((f"role[{role}]", bool(hits),
                    f"matched {hits}" if hits
                    else f"no stage with active={act} passive={pas} {need}"))
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
    doc = doc.get("plan", doc)
    return StagePlan(task=doc["task"], stages=tuple(
        StageSpec(**{**s, "parts": tuple(s["parts"])}) for s in doc["stages"]))


def run_one(plan: StagePlan, pools: dict[str, dict[int, dict]],
            out: Path, logs: list | None = None, raw: list[str] | None = None
            ) -> list[Check]:
    checks = verify(plan, pools)
    parts = object_parts(plan, pools)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "plan": asdict(plan),
        "object_parts": {o: list(v) for o, v in parts.items()},
        "checks": [{"check": c, "pass": ok, "detail": d} for c, ok, d in checks],
    }, indent=2) + "\n")
    if logs is not None:
        out.with_suffix(".log.json").write_text(json.dumps({
            "client_logs": [asdict(l) for l in logs],
            "raw_responses": raw or [],
        }, indent=2, default=str) + "\n")
    return checks


def print_report(plan: StagePlan, checks: list[Check], parts: dict) -> None:
    print(f"[plan-stages] task: {plan.task}")
    for s in plan.stages:
        arrow = f" -> {s.passive}" if s.passive else ""
        print(f"  [{s.index}] {s.name}: {s.active}{arrow}, parts {list(s.parts)}")
    print("[plan-stages] verification:")
    for check, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")
    print("[plan-stages] #1 -> #2 handoff (select_frames.OBJECT_PARTS equivalent):")
    for obj, ps in parts.items():
        if ps:
            print(f"  MUJOCO_GL=osmesa PYTHONPATH=. python "
                  f"scripts/render_candidates.py --object {obj} --vlm "
                  f"--parts {' '.join(ps)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    ap.add_argument("--repeat", type=int, default=1,
                    help="live calls to make; artifacts get a .k suffix")
    ap.add_argument("--replay", nargs="+", metavar="JSON",
                    help="verify saved artifacts offline; no API call")
    args = ap.parse_args()

    pools = {n: load_pool(d) for n, d in OBJECTS.items()}
    base = Path(args.out)

    runs: list[list[Check]] = []
    if args.replay:
        for p in args.replay:
            plan = plan_from_artifact(Path(p))
            checks = verify(plan, pools)
            print(f"\n=== replay {p}")
            print_report(plan, checks, object_parts(plan, pools))
            runs.append(checks)
    else:
        vocab = Vocabulary.from_asset_dirs(OBJECTS)   # no menu: #1 is symbols-only
        for k in range(args.repeat):
            transport = RecordingTransport()
            client = Client(transport=transport)
            plan = client.plan_stages(args.task, vocab)
            out = base if args.repeat == 1 else base.with_name(
                f"{base.stem}.{k}{base.suffix}")
            checks = run_one(plan, pools, out, client.logs, transport.raw)
            print(f"\n=== run {k} -> {out}")
            print_report(plan, checks, object_parts(plan, pools))
            runs.append(checks)

    if len(runs) > 1:
        print("\n[plan-stages] pass rate per check:")
        names = [c for c, _, _ in runs[0]]
        for name in names:
            n = sum(1 for r in runs for c, ok, _ in r if c == name and ok)
            print(f"  {n}/{len(runs)}  {name}")
        allpass = sum(1 for r in runs if all(ok for _, ok, _ in r))
        print(f"  {allpass}/{len(runs)}  ALL")

    failed = [c for r in runs for c, ok, _ in r if not ok]
    if failed:
        sys.exit(f"[plan-stages] FAILED checks: {sorted(set(failed))} — "
                 "the emitted plan cannot replace select_frames.ROLES")
    print("\n[plan-stages] contract holds: emitted plan grounds and reproduces "
          "the hand-authored structure")


if __name__ == "__main__":
    main()
