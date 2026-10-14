"""Role binding of the insert-marker contract (tasks/insert_marker.json)
on the stage decompositions call #1 produces or plausibly could: two
stages (the observed Sonnet 4.6 plan, `body` on the insertion) and three
stages with `tip`/`body` mixed across the carry and the insertion.

transport_active and insert share ONE marker->cup predicate, the way
pour-tea's transport_active and pour share `spout`: the `after` floor and
the optional-yield rule then decide by stage ORDER which marker->cup
stage is the carry and which the insertion. Tagging the carry `tip` and
the insertion `tip|body` instead binds the insertion to the carry stage
whenever the carry names `body` (the real insertion then binds nowhere
and every check still passes) — the cases below pin that shut."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plan_stages as ps  # noqa: E402

from manip_sim.selection import load_pool  # noqa: E402

ASSETS = {n: ROOT / f"assets/objects/{n}" for n in ("marker", "cup")}
pytestmark = pytest.mark.skipif(
    not all((d / "candidates.json").exists() for d in ASSETS.values()),
    reason="needs committed candidates.json pools")

SPEC = ps.load_task_spec(ROOT / "tasks/insert_marker.json")
GRASP = {"name": "grasp marker", "active": "marker", "passive": None,
         "parts": {"marker": ["body", "tip"]}}


def _plan(*marker_parts_per_mover_stage):
    stages = [{**GRASP, "index": 0}]
    for k, mp in enumerate(marker_parts_per_mover_stage, start=1):
        stages.append({"index": k, "name": f"stage {k}", "active": "marker",
                       "passive": "cup",
                       "parts": {"marker": mp, "cup": ["opening"]}})
    return ps.plan_from_artifact_doc(
        {"task": "insert the marker into the cup", "stages": stages})


@pytest.fixture(scope="module")
def pools():
    return {n: load_pool(d) for n, d in ASSETS.items()}


@pytest.mark.parametrize("movers, carry, insert", [
    ((["body"],), None, 1),                    # observed: Sonnet 4.6, 5/5
    ((["tip"],), None, 1),
    ((["tip"], ["tip"]), 1, 2),
    ((["body"], ["body"]), 1, 2),              # insert-only relaxation bound insert @ 1
    ((["body"], ["tip"]), 1, 2),               # ... and regressed this one from @ 2
])
def test_carry_and_insert_bind_by_order(pools, movers, carry, insert):
    b = ps.bind_roles(_plan(*movers), pools, SPEC, by_name=True)
    assert b["grasp"]["stage"] == 0
    assert b["transport_active"]["stage"] == carry
    assert b["insert"]["stage"] == insert


def test_contract_still_requires_tip_somewhere():
    """`tip` left the role predicates but stays a required tag, so call
    #2's object-level menu always carries tip candidates."""
    assert "tip" in SPEC.required_tags["marker"]


# ------------------------------------- handoff to calls #2/#3 (load_bindings)

def _artifact(pools, tmp_path, *movers):
    out = tmp_path / "plan.json"
    ps.run_one(_plan(*movers), pools, out, spec=SPEC, defer_grounding=True)
    return out


@pytest.mark.parametrize("movers, emitted, synth", [
    ((["body"],), ("grasp", "insert"), {"insert"}),
    ((["tip"],), ("grasp", "insert"), {"insert"}),
    ((["body"], ["tip"]), ("grasp", "transport_active", "insert"), set()),
])
def test_bindings_skip_unbound_optional_and_emit_order(pools, tmp_path, movers,
                                                       emitted, synth):
    from scripts.emit_constraints import bound_stages
    # no spec passed: the default is pour-tea's, so this also proves the
    # contract comes from the artifact's recorded task_spec
    b = ps.load_bindings(_artifact(pools, tmp_path, *movers))
    assert b.spec.name == "insert_marker"
    assert set(b.roles) == set(emitted) | {"transport_passive"}
    assert b.entry_synthesized == frozenset(synth)
    by_idx = {s.index: s for s in b.plan.stages}
    assert tuple(r for _, r in bound_stages(b, by_idx)) == emitted


def test_unbound_required_role_still_refused(pools, tmp_path):
    out = _artifact(pools, tmp_path, ["body"])
    doc = json.loads(out.read_text())
    doc["roles"]["insert"]["stage"] = None
    out.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="'insert' unbound"):
        ps.load_bindings(out)


def test_artifact_without_task_spec_uses_the_parameter(pools, tmp_path):
    out = _artifact(pools, tmp_path, ["body"])
    doc = json.loads(out.read_text())
    del doc["task_spec"]
    out.write_text(json.dumps(doc))
    assert ps.load_bindings(out, SPEC).entry_synthesized == {"insert"}
    with pytest.raises(SystemExit, match="unbound"):     # pour-tea default:
        ps.load_bindings(out)                             # nothing optional
