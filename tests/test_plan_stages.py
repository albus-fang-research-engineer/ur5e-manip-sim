"""Offline verification of VLM touchpoint #1 (scripts/plan_stages.py):
the grounding + contract checks against the REAL candidate pools, with
canned plans standing in for the model. Exercises exactly the cases the
live run must discriminate: a natural plan with the mug passive
throughout (must PASS), ungroundable free-text parts (vlm_subset would
silently hand call #2 a menu with no part-biased marks), wrong grasp
part, wrong ordering, and the end-to-end Client -> verify -> artifact
path through a fake transport."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plan_stages as ps  # noqa: E402

from manip_sim.selection import load_pool  # noqa: E402
from manip_sim.vlm import Client, Vocabulary, parse_stage_plan  # noqa: E402

ASSETS = {n: ROOT / f"assets/objects/{n}" for n in ("teapot", "mug")}
pytestmark = pytest.mark.skipif(
    not all((d / "candidates.json").exists() for d in ASSETS.values()),
    reason="needs committed candidates.json pools")


@pytest.fixture(scope="module")
def pools():
    return {n: load_pool(d) for n, d in ASSETS.items()}


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary.from_asset_dirs(ASSETS)


def plan(vocab, *stages):
    return parse_stage_plan(json.dumps({"stages": list(stages)}), vocab,
                            ps.TASK)


def st(name, active, passive, parts):
    """parts: dict object -> list, or a flat list routed by the test's own
    rule (rim -> mug, everything else -> active) for brevity."""
    if not isinstance(parts, dict):
        d = {}
        for part in parts:
            o = passive if (part == "rim" and passive) else active
            d.setdefault(o, []).append(part)
        parts = d
    return {"name": name, "active": active, "passive": passive,
            "parts": {o: list(v) for o, v in parts.items()}}


def failed(checks):
    return sorted(c for c, ok, _ in checks if not ok)


NATURAL = (st("grasp the teapot by its handle", "teapot", None, ["handle"]),
           st("carry spout over the mug", "teapot", "mug", ["spout", "rim"]),
           st("tilt to pour", "teapot", "mug", ["spout", "rim"]))


# ---------------------------------------------------------------- PASS

def test_natural_plan_passes(vocab, pools):
    """Mug is passive in every stage; 'rim' is keyed to the mug — the
    plan the model is most likely to emit."""
    checks = ps.verify(plan(vocab, *NATURAL), pools)
    assert failed(checks) == [], checks


def test_handoff_is_per_object(vocab, pools):
    parts = ps.object_parts(plan(vocab, *NATURAL), pools)
    assert parts == {"teapot": ("handle", "spout"), "mug": ("rim",)}


def test_free_text_variants_ground(vocab, pools):
    """Substring matching is the licensed tolerance: 'spout tip' ~ spout,
    'handle loop' ~ handle. Not a synonym table."""
    p = plan(vocab,
             st("grasp", "teapot", None, ["handle loop"]),
             st("move", "teapot", "mug", ["spout tip", "rim"]),
             st("pour", "teapot", "mug", ["spout tip", "rim"]))
    assert failed(ps.verify(p, pools)) == []


# ---------------------------------------------------------------- FAIL

def test_ungroundable_part_fails_grounding(vocab, pools):
    """'grip' grounds nowhere; vlm_subset would still return a full menu
    for it (silent degradation) — this is the check that catches it."""
    p = plan(vocab,
             st("grasp", "teapot", None, ["grip"]),
             st("move", "teapot", "mug", ["spout", "rim"]),
             st("pour", "teapot", "mug", ["spout", "rim"]))
    f = failed(ps.verify(p, pools))
    assert "grounds[stage0]" in f
    assert "coverage[teapot]" in f and "role[grasp]" in f


def test_wrong_grasp_part_fails_role(vocab, pools):
    p = plan(vocab,
             st("grasp", "teapot", None, ["lid"]),   # grounds, wrong role
             st("move", "teapot", "mug", ["spout", "rim"]),
             st("pour", "teapot", "mug", ["spout", "rim"]))
    f = failed(ps.verify(p, pools))
    assert "grounds[stage0]" not in f           # lid IS a pool tag
    assert "role[grasp]" in f and "coverage[teapot]" in f


def test_missing_mug_part_fails_passive_role(vocab, pools):
    p = plan(vocab,
             st("grasp", "teapot", None, ["handle"]),
             st("pour", "teapot", "mug", ["spout"]))
    f = failed(ps.verify(p, pools))
    assert "coverage[mug]" in f and "role[transport_passive]" in f


def test_ordering_fails_when_interaction_precedes_grasp(vocab, pools):
    p = plan(vocab,
             st("pour", "teapot", "mug", ["spout", "rim"]),
             st("grasp", "teapot", None, ["handle"]))
    assert any(c.startswith("ordering[grasp<") for c in failed(ps.verify(p, pools)))


def test_part_grounds_only_in_its_own_object(vocab, pools):
    """'handle' is a pool tag on both objects; keyed parts make the
    attribution explicit, so a rim keyed to the teapot is UNGROUNDED
    even though the mug has one."""
    s = plan(vocab, st("pour", "teapot", "mug",
                       {"teapot": ["handle", "rim"], "mug": ["handle"]})).stages[0]
    by_obj, ungrounded, unknown = ps.attribute(s, pools)
    assert by_obj == {"teapot": ("handle", "rim"), "mug": ("handle",)}
    assert ungrounded == ["teapot.rim"] and unknown == []


# ------------------------------------------------------- mark-addressed

MARK_VOCAB = Vocabulary(objects={}, marks={1: "a", 2: "b", 3: "c"})
GT = {1: "mug", 2: "teapot", 3: "bowl"}


def mplan(*stages, objects=None):
    objects = objects or {"2": "teapot", "1": "mug"}
    return parse_stage_plan(json.dumps({"objects": objects,
                                        "stages": list(stages)}),
                            MARK_VOCAB, ps.TASK)


def test_mark_plan_passes_through_ground_truth(pools):
    p = mplan(st("grasp", 2, None, {"2": ["handle"]}),
              st("carry", 2, 1, {"2": ["spout"], "1": ["rim"]}),
              st("pour", 2, 1, {"2": ["spout"], "1": ["rim"]}))
    g = ps.to_ground_truth(p, GT)
    assert g.stages[1].parts == {"teapot": ("spout",), "mug": ("rim",)}
    checks = ps.verify(g, pools)
    assert failed(checks) == [], checks
    assert "identity[teapot]" in [c for c, _, _ in checks]


def test_mark_plan_wrong_object_fails_identity_and_roles(pools):
    # model picks the bowl (mark 3) as the thing to pour from
    p = mplan(st("grasp", 3, None, {"3": ["handle"]}),
              st("pour", 3, 1, {"3": ["spout"], "1": ["rim"]}),
              objects={"3": "teapot", "1": "mug"})
    f = failed(ps.verify(ps.to_ground_truth(p, GT), pools))
    assert "identity[bowl]" in f and "role[grasp]" in f and "role[pour]" in f


def test_mark_plan_hardware_mode_keeps_handles(pools):
    p = mplan(st("grasp", 2, None, {"2": ["handle"]}))
    assert ps.to_ground_truth(p, None) is p
    f = failed(ps.verify(p, pools))
    assert "identity[teapot]" not in f          # label happened to match


# --------------------------------------------------------- end-to-end

def test_client_to_artifact_roundtrip(vocab, pools, tmp_path):
    raw = json.dumps({"stages": list(NATURAL)})
    transport = ps.RecordingTransport(inner=lambda payload: raw)
    client = Client(transport=transport)
    p = client.plan_stages(ps.TASK, vocab)
    out = tmp_path / "pour_tea.json"
    checks = ps.run_one(p, pools, out, client.logs, transport.raw)
    assert failed(checks) == []
    doc = json.loads(out.read_text())
    assert doc["mode"] == "text"
    assert doc["object_parts"] == {"teapot": ["handle", "spout"],
                                   "mug": ["rim"]}
    log = json.loads(out.with_suffix(".log.json").read_text())
    assert log["raw_responses"] == [raw]
    # artifact replays to the same verdict
    assert failed(ps.verify(ps.plan_from_artifact(out), pools)) == []


# ---------------------------------------------------------- task spec

def test_task_spec_round_trips_the_hardcoded_contract():
    spec = ps.load_task_spec()
    assert spec.required_tags == {"teapot": ("handle", "spout"), "mug": ("rim|opening|lip",)}
    assert set(spec.roles) == {"grasp", "transport_active", "pour",
                               "transport_passive"}
    assert spec.roles["grasp"] == ("teapot", None, {"teapot": ("handle",)})
    assert spec.roles["transport_passive"][:2] == ("*", "*")
    assert spec.ordering == (("grasp", ("transport_active", "pour")),)


def test_verify_accepts_alternate_spec(vocab, pools):
    # same plan, a spec that demands a role the plan cannot satisfy
    spec = ps.TaskSpec(name="x", required_tags={},
                       roles={"remove_lid": ("teapot", None, {"teapot": ("lid",)})},
                       ordering=(("remove_lid", ()),))
    f = failed(ps.verify(plan(vocab, *NATURAL), pools, spec))
    assert "role[remove_lid]" in f


# ------------------------------------------------------- role bindings

def test_bind_roles_matches_authored_table(vocab, pools):
    b = ps.bind_roles(plan(vocab, *NATURAL), pools)
    assert b == {"grasp": {"object": "teapot", "stage": 0},
                 "transport_active": {"object": "teapot", "stage": 1},
                 "pour": {"object": "teapot", "stage": 2},
                 "transport_passive": {"object": "mug", "stage": 1}}


def test_bind_roles_unmatched_is_none(vocab, pools):
    # no stage carries a grounded grasp part -> role unbound, not raised
    b = ps.bind_roles(plan(vocab, st("grasp", "teapot", None, ["spout"])), pools)
    assert b["grasp"]["stage"] is None


def test_artifact_carries_bindings_and_loads(vocab, pools, tmp_path):
    import select_frames as sf
    raw = json.dumps({"stages": list(NATURAL)})
    client = Client(transport=lambda payload: raw)
    out = tmp_path / "pour_tea.json"
    ps.run_one(client.plan_stages(ps.TASK, vocab), pools, out)
    b = ps.load_bindings(out)
    assert b.object_parts == sf.OBJECT_PARTS
    # same (object, parts-on-that-object) per role as the authored table;
    # stage names/indices are the model's, not the table's
    for role, (obj, stage) in b.roles.items():
        a_obj, a_stage = sf.ROLES[role]
        assert obj == a_obj
        assert stage.parts == a_stage.parts
        assert set(stage.parts) == {obj}


def test_artifact_with_failed_checks_is_refused(vocab, pools, tmp_path):
    raw = json.dumps({"stages": [st("grasp", "teapot", None, ["spout"])]})
    client = Client(transport=lambda payload: raw)
    out = tmp_path / "bad.json"
    ps.run_one(client.plan_stages(ps.TASK, vocab), pools, out)
    with pytest.raises(SystemExit, match="failed checks"):
        ps.load_bindings(out)


def test_mark_artifact_binds_over_ground_truth(pools, tmp_path):
    p = mplan(st("grasp", 2, None, {"2": ["handle"]}),
              st("carry", 2, 1, {"2": ["spout"], "1": ["rim"]}),
              st("pour", 2, 1, {"2": ["spout"], "1": ["rim"]}))
    out = tmp_path / "marks.json"
    ps.run_one(p, pools, out, gt=GT, mode="marks")
    doc = json.loads(out.read_text())
    assert doc["plan"]["stages"][0]["active"] == "teapot"   # handle from label
    assert doc["plan_grounded"]["stages"][0]["active"] == "teapot"
    b = ps.load_bindings(out)
    assert b.roles["transport_passive"][0] == "mug"


# ------------------------------------------------- gate policy (deferral)

def _body_interior_plan(vocab):
    """The live failure of 2026-08-20: extra parts outside the authored
    band vocabulary ('body', 'interior'); every contract check passes."""
    return plan(vocab,
                st("grasp teapot", "teapot", None, ["handle", "body"]),
                st("move teapot over cup", "teapot", "mug",
                   {"teapot": ["spout", "body"], "mug": ["rim", "opening"]}),
                st("tilt teapot to pour", "teapot", "mug",
                   {"teapot": ["spout", "body"], "mug": ["rim", "interior"]}))


def test_extra_parts_block_only_when_grounding_is_strict(vocab, pools):
    checks = ps.verify(_body_interior_plan(vocab), pools)
    f = failed(checks)
    assert {"grounds[stage0]", "menu[teapot]", "menu[mug]"} <= set(f)
    assert not any(c.startswith(("coverage", "role", "ordering")) for c in f)
    assert set(ps.blocking(checks)) == set(f)             # strict: all block
    assert ps.blocking(checks, defer_grounding=True) == []  # deferred: none


def test_deferred_artifact_loads_and_strict_one_refuses(vocab, pools, tmp_path):
    p = _body_interior_plan(vocab)
    strict, deferred = tmp_path / "s.json", tmp_path / "d.json"
    ps.run_one(p, pools, strict)
    ps.run_one(p, pools, deferred, defer_grounding=True)
    with pytest.raises(SystemExit, match="failed checks"):
        ps.load_bindings(strict)
    b = ps.load_bindings(deferred)
    assert b.object_parts["teapot"] == ("handle", "body", "spout")
    doc = json.loads(deferred.read_text())
    assert doc["gate"] == {"defer_grounding": True, "blocking": []}


def test_dropped_extra_parts_pass_strict_gate(vocab, pools, tmp_path):
    """After ground_parts.py records body/interior as ungrounded the
    strict replay prunes them and every check passes."""
    g = tmp_path / "grounding"
    g.mkdir()
    (g / "grounding.json").write_text(json.dumps(
        {"ungrounded": {"teapot": ["body"], "mug": ["interior"]}}))
    dropped = ps.read_dropped(g)
    out = tmp_path / "a.json"
    checks = ps.run_one(_body_interior_plan(vocab), pools, out, dropped=dropped)
    assert failed(checks) == []
    doc = json.loads(out.read_text())
    assert doc["parts_dropped"] == {"teapot": ["body"], "mug": ["interior"]}
    assert doc["object_parts"]["teapot"] == ["handle", "spout"]
    assert "body" not in json.dumps(doc["plan_grounded"])
    assert "body" in json.dumps(doc["plan"])              # emitted kept verbatim
    assert ps.load_bindings(out).roles["grasp"][1].parts == {"teapot": ("handle",)}


def test_dropped_required_part_still_fails_contract(vocab, pools, tmp_path):
    dropped = {"teapot": ("handle",)}
    checks = ps.run_one(_body_interior_plan(vocab), pools, tmp_path / "a.json",
                        defer_grounding=True, dropped=dropped)
    assert {"coverage[teapot]", "role[grasp]"} <= set(ps.blocking(checks, True))


def test_pour_binds_strictly_after_transport(vocab, pools):
    """transport_active and pour share one predicate; `after` makes pour
    bind the LATER matching stage instead of the same one."""
    p = plan(vocab,
             st("grasp", "teapot", None, ["handle"]),
             st("move over", "teapot", "mug", ["spout", "rim"]),
             st("tilt", "teapot", "mug", ["spout", "rim"]))
    b = ps.bind_roles(p, pools)
    assert b["transport_active"]["stage"] == 1 and b["pour"]["stage"] == 2


def test_pour_unbound_when_no_later_stage(vocab, pools):
    p = plan(vocab,
             st("grasp", "teapot", None, ["handle"]),
             st("move and pour", "teapot", "mug", ["spout", "rim"]))
    assert ps.bind_roles(p, pools)["pour"]["stage"] is None


def test_stale_bindings_violating_after_are_refused(vocab, pools, tmp_path):
    """An artifact written before `after` existed binds pour to the
    transport stage; load_bindings must refuse it (emit would otherwise
    emit the transport stage twice and never the tilt stage)."""
    raw = json.dumps({"stages": list(NATURAL)})
    client = Client(transport=lambda payload: raw)
    out = tmp_path / "stale.json"
    ps.run_one(client.plan_stages(ps.TASK, vocab), pools, out)
    doc = json.loads(out.read_text())
    doc["roles"]["pour"]["stage"] = doc["roles"]["transport_active"]["stage"]
    out.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="must be after 'transport_active'"):
        ps.load_bindings(out)


def test_artifact_keeps_unpruned_request_for_regrounding(vocab, pools, tmp_path):
    """ground -> replay must be idempotent: the artifact records the
    parts the plan asked for BEFORE pruning, so ground_parts.py
    re-attempts (and re-reports as ungrounded) parts dropped last time."""
    raw = json.dumps({"stages": list(NATURAL)})
    client = Client(transport=lambda payload: raw)
    out = tmp_path / "pour_tea.json"
    plan = client.plan_stages(ps.TASK, vocab)
    full = ps.object_parts(ps.to_ground_truth(plan, None), pools)
    drop_obj, drop_part = next((o, v[0]) for o, v in full.items() if v)
    ps.run_one(plan, pools, out, dropped={drop_obj: (drop_part,)})
    doc = json.loads(out.read_text())
    assert drop_part not in doc["object_parts"][drop_obj]
    assert doc["object_parts_requested"] == {o: list(v) for o, v in full.items()}
