"""VLM touchpoint-#1 driver: one live plan_stages call, written as a
demo artifact and STRUCTURALLY VERIFIED against the hand-authored
contract in select_frames.ROLES — the check the live loop needs before
call #2 is allowed to consume an emitted plan instead of the fixed one.

What this is (and is not): the wrapper (Client.plan_stages), prompt
(build_stage_plan_prompt) and parser (parse_stage_plan) already exist
and are tested; what has never existed is a driver that makes the call
and confronts the emission with the structure the pour-tea planner
actually requires. This script closes exactly that gap without making
plan_pour_tea.py stage-generic (that waits for the call-#3 emission
compiler): the emitted plan is compared, logged, and its per-object
part filters are printed as the render commands they would drive —
the real #1 -> #2 dependency, human-inspectable before it goes live.

Verification is a CONTRACT check, not string equality — stage names
and parts are free text by design (GroundedSAM seeds):

  coverage   every object the task touches appears as some stage's
             active; the union of its parts (substring-matched) covers
             the parts select_frames hands to vlm_subset
             (teapot: handle+spout, mug: rim)
  ordering   a teapot-active stage with no passive (the grasp) precedes
             every teapot-active stage with passive=mug (transport/pour)
  roles      each of the four planner roles finds at least one emitted
             stage whose (active, passive, parts) signature matches

Each check prints PASS/FAIL; any FAIL exits nonzero after the artifact
is written (the failing plan is evidence, not garbage).

Requires ANTHROPIC_API_KEY.

    PYTHONPATH=. python scripts/plan_stages.py
    PYTHONPATH=. python scripts/plan_stages.py --task "pour tea" \
        --out outputs/stage_plan/other.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from manip_sim.vlm import Client, StagePlan, Vocabulary

OBJECTS = {
    "teapot": Path("assets/objects/teapot"),
    "mug": Path("assets/objects/mug"),
}
TASK = "pour tea from the teapot into the mug"
OUT = Path("outputs/stage_plan/pour_tea.json")

# the contract select_frames.py currently hand-authors (its OBJECT_PARTS
# and ROLES) restated as checkable structure. Part terms are matched by
# substring in either direction ("handle" ~ "handle loop"), lowercased.
REQUIRED_PARTS: dict[str, tuple[str, ...]] = {
    "teapot": ("handle", "spout"),
    "mug": ("rim",),
}
ROLE_SIGNATURES: dict[str, dict] = {
    "grasp": {"active": "teapot", "passive": None, "parts": ("handle",)},
    "transport_active": {"active": "teapot", "passive": "mug",
                         "parts": ("spout",)},
    "pour": {"active": "teapot", "passive": "mug", "parts": ("spout",)},
    "transport_passive": {"active": "mug", "parts": ("rim",)},
}


def _part_match(want: str, emitted: tuple[str, ...]) -> bool:
    w = want.lower()
    return any(w in p.lower() or p.lower() in w for p in emitted)


def verify(plan: StagePlan) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) triples — all checks always run."""
    out: list[tuple[str, bool, str]] = []

    # -- coverage
    emitted_parts: dict[str, tuple[str, ...]] = {}
    for s in plan.stages:
        emitted_parts[s.active] = emitted_parts.get(s.active, ()) + s.parts
    for obj, req in REQUIRED_PARTS.items():
        have = emitted_parts.get(obj, ())
        missing = [p for p in req if not _part_match(p, have)]
        out.append((
            f"coverage[{obj}]", not missing,
            f"needs {list(req)}, emitted {sorted(set(have))}"
            + (f" — MISSING {missing}" if missing else "")))

    # -- ordering: grasp-like precedes interaction-like on the teapot
    grasp_idx = [s.index for s in plan.stages
                 if s.active == "teapot" and s.passive is None]
    inter_idx = [s.index for s in plan.stages
                 if s.active == "teapot" and s.passive == "mug"]
    ok = bool(grasp_idx) and bool(inter_idx) and min(grasp_idx) < min(
        inter_idx)
    out.append(("ordering[grasp<interaction]", ok,
                f"teapot-only stages at {grasp_idx}, "
                f"teapot->mug stages at {inter_idx}"))

    # -- role signatures
    for role, sig in ROLE_SIGNATURES.items():
        hits = [s for s in plan.stages
                if s.active == sig["active"]
                and ("passive" not in sig or s.passive == sig["passive"])
                and all(_part_match(p, s.parts) for p in sig["parts"])]
        out.append((
            f"role[{role}]", bool(hits),
            "matched stage(s) " + str([f"{s.index}:{s.name}" for s in hits])
            if hits else f"no stage with signature {sig}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    args = ap.parse_args()

    vocab = Vocabulary.from_asset_dirs(OBJECTS)   # no menu: call #1 is
    client = Client()                             # symbols-only by design
    plan = client.plan_stages(args.task, vocab)

    print(f"[plan-stages] task: {args.task}")
    for s in plan.stages:
        arrow = f" -> {s.passive}" if s.passive else ""
        print(f"  [{s.index}] {s.name}: {s.active}{arrow}, "
              f"parts {list(s.parts)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(plan), indent=2) + "\n")
    log = out.with_suffix(".log.json")
    log.write_text(json.dumps([asdict(l) for l in client.logs], indent=2,
                              default=str) + "\n")
    print(f"[plan-stages] wrote {out} (+ {log})")

    checks = verify(plan)
    failed = [c for c, ok, _ in checks if not ok]
    print("\n[plan-stages] contract verification vs select_frames.ROLES:")
    for check, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")

    # the #1 -> #2 dependency, printed as the commands it would drive
    print("\n[plan-stages] render filters this plan implies:")
    for obj in OBJECTS:
        parts = sorted({p for s in plan.stages if s.active == obj
                        for p in s.parts})
        if parts:
            print(f"  MUJOCO_GL=osmesa PYTHONPATH=. python "
                  f"scripts/render_candidates.py --object {obj} --vlm "
                  f"--parts {' '.join(parts)}")

    if failed:
        sys.exit(f"[plan-stages] contract FAILED: {failed} — the emitted "
                 "plan cannot replace the hand-authored ROLES yet")
    print("\n[plan-stages] contract holds — the emitted plan reproduces "
          "the hand-authored structure; next: wire select_frames.py to "
          "consume it (--stage-plan) once call #3 exists to close the loop")


if __name__ == "__main__":
    main()