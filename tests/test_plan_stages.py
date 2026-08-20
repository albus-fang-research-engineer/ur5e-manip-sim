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
    return {"name": name, "active": active, "passive": passive,
            "parts": list(parts)}


def failed(checks):
    return sorted(c for c, ok, _ in checks if not ok)


NATURAL = (st("grasp the teapot by its handle", "teapot", None, ["handle"]),
           st("carry spout over the mug", "teapot", "mug", ["spout", "rim"]),
           st("tilt to pour", "teapot", "mug", ["spout", "rim"]))


# ---------------------------------------------------------------- PASS

def test_natural_plan_passes(vocab, pools):
    """Mug is passive in every stage and 'rim' rides in the pour stage's
    flat parts list — the plan the model is most likely to emit. Pool
    attribution must route 'rim' to the mug."""
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
    assert "ordering[grasp<interaction]" in failed(ps.verify(p, pools))


def test_ambiguous_part_goes_to_active_and_is_flagged(vocab, pools):
    """'handle' is a pool tag on both objects."""
    s = plan(vocab, st("pour", "teapot", "mug", ["handle"])).stages[0]
    by_obj, ungrounded, ambiguous = ps.attribute(s, pools)
    assert by_obj["teapot"] == ("handle",) and by_obj["mug"] == ()
    assert ambiguous == ["handle"] and ungrounded == []


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
    assert doc["object_parts"] == {"teapot": ["handle", "spout"],
                                   "mug": ["rim"]}
    log = json.loads(out.with_suffix(".log.json").read_text())
    assert log["raw_responses"] == [raw]
    # artifact replays to the same verdict
    assert failed(ps.verify(ps.plan_from_artifact(out), pools)) == []
