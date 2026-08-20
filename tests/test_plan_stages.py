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
    assert spec.required_tags == {"teapot": ("handle", "spout"), "mug": ("rim",)}
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
                 "pour": {"object": "teapot", "stage": 1},
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
