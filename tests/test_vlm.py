"""Offline tests for the VLM typed I/O layer (manip_sim/vlm.py): the
licensed-vocabulary invariants, per-touchpoint parsing, and the bounded
retry loop — all against an injected fake transport, no network, no API
key. The vocabulary is built from the REAL frames.json sidecars when
present (so prompt-menu == accept-set is exercised on the actual
symbols), with a synthetic fallback otherwise."""

import json
from pathlib import Path

import pytest

from manip_sim.vlm import (CLEARANCES, MAX_PARSE_RETRIES, ROT_TOLS,
                           Client, ParseRejection, PointAxisSelection,
                           StageSpec, VLMError, Vocabulary, parse_critic,
                           parse_emission, parse_point_axis,
                           parse_repair, parse_stage_plan, validate_expr)

ASSETS = {n: Path(f"assets/objects/{n}") for n in ("teapot", "mug")}
MENU = {3: "constructed opening_center", 7: "part handle",
        12: "curvature spout ridge"}


@pytest.fixture
def vocab():
    if all((d / "frames.json").exists() for d in ASSETS.values()):
        return Vocabulary.from_asset_dirs(ASSETS, menu=MENU)
    return Vocabulary(objects={
        "teapot": {"points": ("spout_tip", "handle_center"),
                   "axes": ("pour_axis", "up_axis", "handle_axis",
                            "tilt_axis"),
                   "quantities": ()},
        "mug": {"points": ("opening_center",), "axes": ("up_axis",),
                "quantities": ("rim_radius",)},
    }, menu=MENU)


def canned(*texts):
    """Transport returning successive canned responses."""
    it = iter(texts)

    def transport(payload):
        return next(it)
    return transport


STAGE = StageSpec(index=1, name="transport", active="teapot",
                  passive="mug", parts={"teapot": ("spout",), "mug": ("rim",)})


# ------------------------------------------------------------ vocabulary

def test_vocabulary_from_real_assets(vocab):
    assert "teapot.pour_axis" in vocab.axis_names()
    assert "mug.opening_center" in vocab.point_names()
    assert "world.z" in vocab.axis_names()
    # prompt menu and accept set are the same object
    assert set(MENU) == set(vocab.menu)
    desc = vocab.describe_symbols()
    assert "pour_axis" in desc and "opening_center" in desc


# ------------------------------------------------------- expression grammar

def test_expr_accepts_quantity_arithmetic():
    # symbol-only arithmetic is flag-free; the literal flag is syntactic,
    # so a scale factor like /2 IS flagged (logged, never rejected)
    assert validate_expr("rim_radius + rim_radius", {"rim_radius"}) == ()
    flags = validate_expr("rim_radius/2", {"rim_radius"})
    assert len(flags) == 1 and "'2'" in flags[0]


def test_expr_flags_bare_literals():
    flags = validate_expr("rim_radius - 0.01", {"rim_radius"})
    assert len(flags) == 1 and "0.01" in flags[0]


def test_expr_rejects_unknown_symbol():
    with pytest.raises(ParseRejection):
        validate_expr("cavity_depth", {"rim_radius"})


def test_expr_rejects_malformed():
    for bad in ("rim_radius +", "(rim_radius", "rim_radius ** 2", ""):
        with pytest.raises(ParseRejection):
            validate_expr(bad, {"rim_radius"})


# ----------------------------------------------------------- touchpoint #1

def test_stage_plan_parses(vocab):
    raw = json.dumps({"stages": [
        {"name": "grasp", "active": "teapot", "passive": None,
         "parts": {"teapot": ["handle"]}},
        {"name": "pour", "active": "teapot", "passive": "mug",
         "parts": {"teapot": ["spout"], "mug": ["rim"]}}]})
    plan = parse_stage_plan(raw, vocab, "pour tea")
    assert plan.stages[1].passive == "mug"
    assert plan.stages[0].parts == {"teapot": ("handle",)}
    assert plan.objects["mug"].mark is None


def test_stage_plan_rejects_part_key_outside_stage(vocab):
    raw = json.dumps({"stages": [
        {"name": "grasp", "active": "teapot", "passive": None,
         "parts": {"mug": ["rim"]}}]})
    with pytest.raises(ParseRejection):
        parse_stage_plan(raw, vocab, "pour tea")


def test_stage_plan_mark_addressed():
    vocab = Vocabulary(objects={}, marks={1: "bbox", 2: "bbox", 3: "bbox"})
    raw = json.dumps({
        "objects": {"2": "tea pot", "3": "mug"},
        "stages": [
            {"name": "grasp", "active": 2, "passive": None,
             "parts": {"2": ["handle"]}},
            {"name": "pour", "active": 2, "passive": 3,
             "parts": {"2": ["spout"], "3": ["rim"]}}]})
    plan = parse_stage_plan(raw, vocab, "pour tea")
    assert plan.stages[0].active == "tea_pot" and plan.stages[1].passive == "mug"
    assert plan.stages[1].parts == {"tea_pot": ("spout",), "mug": ("rim",)}
    assert plan.objects["tea_pot"].mark == 2 and plan.objects["mug"].label == "mug"
    assert plan.handle_of_mark() == {2: "tea_pot", 3: "mug"}
    gt = plan.relabel({"tea_pot": "teapot"})
    assert gt.stages[1].parts == {"teapot": ("spout",), "mug": ("rim",)}
    # undeclared / off-image marks are hard rejections
    for bad in ({"objects": {"2": "a"}, "stages": [{"name": "x", "active": 9,
                                                   "passive": None, "parts": {}}]},
                {"objects": {"7": "a"}, "stages": [{"name": "x", "active": 7,
                                                   "passive": None, "parts": {}}]},
                {"objects": {"2": "a"}, "stages": [{"name": "x", "active": "2",
                                                   "passive": None, "parts": {}}]}):
        with pytest.raises(ParseRejection):
            parse_stage_plan(json.dumps(bad), vocab, "t")


def test_stage_plan_rejects_unknown_object(vocab):
    raw = json.dumps({"stages": [{"name": "grasp", "active": "kettle",
                                  "passive": None, "parts": {"kettle": ["handle"]}}]})
    with pytest.raises(ParseRejection):
        parse_stage_plan(raw, vocab, "pour tea")


# ----------------------------------------------------------- touchpoint #2

def test_selection_parses_and_strips_fences(vocab):
    raw = ("```json\n" + json.dumps(
        {"candidate_id": 3, "axis": "teapot.pour_axis", "sign": "+",
         "secondary": "teapot.up_axis", "rationale": "spout side"})
        + "\n```")
    sel = parse_point_axis(raw, vocab)
    assert sel.candidate_id == 3 and sel.axis == "teapot.pour_axis"


def test_selection_rejects_off_menu_id(vocab):
    raw = json.dumps({"candidate_id": 99, "axis": "teapot.pour_axis",
                      "sign": "+"})
    with pytest.raises(ParseRejection) as e:
        parse_point_axis(raw, vocab)
    assert "menu" in str(e.value)


def test_selection_rejects_unlicensed_axis(vocab):
    raw = json.dumps({"candidate_id": 3, "axis": "teapot.magic_axis",
                      "sign": "+"})
    with pytest.raises(ParseRejection):
        parse_point_axis(raw, vocab)


# ----------------------------------------------------------- touchpoint #3

def good_emission():
    return {
        "stage": 2, "name": "pour", "active": "teapot", "passive": "mug",
        "w_origin": "mug.opening_center", "w_axis": "mug.up_axis",
        "path_tsr": {
            "rot": [{"axis": "teapot.up_axis", "relation": "parallel",
                     "reference": "world.z", "tol": "moderate"}],
            "trans": "free"},
        "subgoal_tsr": {
            "rot": [{"axis": "teapot.pour_axis",
                     "relation": "antiparallel",
                     "reference": "mug.up_axis", "tol": "loose"},
                    {"relation": "free", "row": "yaw"}],
            "trans": [
                {"term": "above", "anchor": "mug.opening_center",
                 "clearance": "small", "slack": "moderate"},
                {"term": "centered", "anchor": "mug.opening_center",
                 "tol": "snug"},
                {"term": "expr", "row": "x", "lo": "-mug.rim_radius",
                 "hi": "mug.rim_radius"}]},
        "verify": "liquid would fall inside the mug rim"}


def test_emission_parses(vocab):
    em = parse_emission(json.dumps(good_emission()), vocab)
    assert em.path_tsr.rot[0].tol == "moderate"
    assert em.subgoal_tsr.rot[1].relation == "free"
    assert em.subgoal_tsr.trans[0].clearance == "small"
    # expr term validated, no bare literals -> no flags
    assert em.subgoal_tsr.trans[2].flags == ()


def test_emission_hard_rejects_numeric_rotation(vocab):
    doc = good_emission()
    doc["path_tsr"]["rot"][0]["tol"] = 15          # enum -> number
    with pytest.raises(ParseRejection) as e:
        parse_emission(json.dumps(doc), vocab)
    assert "never numbers" in str(e.value)


def test_emission_rejects_unlicensed_relation(vocab):
    doc = good_emission()
    doc["subgoal_tsr"]["rot"][0]["relation"] = "roughly_facing"
    with pytest.raises(ParseRejection):
        parse_emission(json.dumps(doc), vocab)


def test_emission_flags_translational_literal(vocab):
    doc = good_emission()
    doc["subgoal_tsr"]["trans"][2]["hi"] = "mug.rim_radius + 0.02"
    em = parse_emission(json.dumps(doc), vocab)
    assert any("0.02" in f for f in em.subgoal_tsr.trans[2].flags)


def test_emission_rejects_unknown_anchor(vocab):
    doc = good_emission()
    doc["subgoal_tsr"]["trans"][0]["anchor"] = "mug.spout_tip"
    with pytest.raises(ParseRejection):
        parse_emission(json.dumps(doc), vocab)


# -------------------------------------------------------- touchpoints #4/#5

def test_critic_reject_requires_edits(vocab):
    raw = json.dumps({"verdict": "reject", "edits": [],
                      "diagnosis": "tilt insufficient"})
    with pytest.raises(ParseRejection):
        parse_critic(raw, vocab)


def test_critic_parses_typed_edit(vocab):
    raw = json.dumps({"verdict": "reject", "edits": [
        {"target": "stage2.subgoal.rot[0]", "action": "relax_tolerance",
         "token": "loose"}], "diagnosis": "goal region too tight"})
    v = parse_critic(raw, vocab)
    assert v.edits[0].action == "relax_tolerance"
    assert v.edits[0].token in ROT_TOLS


def test_repair_parses_and_rejects_unknown_action(vocab):
    ok = json.dumps({"action": "widen_clearance",
                     "target": "stage2.subgoal.trans[0]",
                     "token": "medium", "rationale": "collisions at rim"})
    r = parse_repair(ok, vocab)
    assert r.token in CLEARANCES
    bad = json.dumps({"action": "try_harder", "rationale": ""})
    with pytest.raises(ParseRejection):
        parse_repair(bad, vocab)


# ------------------------------------------------------------- retry loop

def test_retry_feeds_rejection_back_then_succeeds(vocab):
    seen = []

    def transport(payload):
        seen.append(payload)
        if len(seen) == 1:
            return json.dumps({"candidate_id": 99,
                               "axis": "teapot.pour_axis", "sign": "+"})
        return json.dumps({"candidate_id": 7,
                           "axis": "teapot.pour_axis", "sign": "-"})

    c = Client(transport=transport)
    sel = c.select_point_axis(STAGE, vocab, view_paths=[])
    assert isinstance(sel, PointAxisSelection) and sel.candidate_id == 7
    log = c.logs[-1]
    assert log.attempts == 2 and len(log.rejections) == 1
    # the second request carries the rejection as a follow-up user turn
    turns = seen[1]["messages"]
    assert turns[-1]["role"] == "user"
    assert "rejected" in turns[-1]["content"][0]["text"]


def test_retry_budget_exhausts(vocab):
    c = Client(transport=lambda p: "not json at all")
    with pytest.raises(VLMError) as e:
        c.plan_stages("pour tea", vocab)
    assert c.logs[-1].attempts == 1 + MAX_PARSE_RETRIES
    assert "retry budget" in str(e.value)


def test_menu_required_for_selection(vocab):
    c = Client(transport=canned("{}"))
    with pytest.raises(ValueError):
        c.select_point_axis(STAGE, Vocabulary(objects=vocab.objects),
                            view_paths=[])