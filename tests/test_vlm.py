"""Offline tests for the VLM typed I/O layer (manip_sim/vlm.py): the
licensed-vocabulary invariants, per-touchpoint parsing, and the bounded
retry loop — all against an injected fake transport, no network, no API
key. The vocabulary is built from the REAL frames.json sidecars when
present (so prompt-menu == accept-set is exercised on the actual
symbols), with a synthetic fallback otherwise."""

import json
from pathlib import Path

import pytest

from manip_sim.vlm import (CANONICAL_DIRS, CLEARANCES, MAX_PARSE_RETRIES,
                           ROT_TOLS, Client, ParseRejection,
                           PointAxisSelection, StageSpec, VLMError,
                           Vocabulary, parse_critic, parse_emission,
                           parse_point_axis,
                           parse_point_axis_directions,
                           parse_point_axis_named,
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
                            "tilt_axis", "front_axis", "lateral_axis"),
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

def test_selection_point_only_parses_and_strips_fences(vocab):
    raw = ("```json\n" + json.dumps(
        {"candidate_id": 3, "rationale": "spout side"}) + "\n```")
    sel = parse_point_axis(raw, vocab)
    assert sel.candidate_id == 3
    assert sel.axis is None and sel.secondary is None


def test_selection_point_only_rejects_off_menu_id(vocab):
    raw = json.dumps({"candidate_id": 99, "rationale": "x"})
    with pytest.raises(ParseRejection) as e:
        parse_point_axis(raw, vocab)
    assert "menu" in str(e.value)


def test_selection_directions_arm_normalizes_all_tokens(vocab):
    expect = {"front": "front_axis", "left": "lateral_axis",
              "up": "up_axis"}
    for d in CANONICAL_DIRS:
        sel = parse_point_axis_directions(
            json.dumps({"candidate_id": 3, "direction": d}), vocab,
            "teapot")
        assert sel.axis == f"teapot.{expect[d[1:]]}" and sel.sign == d[0]


def test_selection_directions_arm_world_axis_unrepresentable(vocab):
    raw = json.dumps({"candidate_id": 3, "direction": "world.z"})
    with pytest.raises(ParseRejection) as e:
        parse_point_axis_directions(raw, vocab, "mug")
    assert "direction" in str(e.value)


def test_selection_directions_arm_licensing_follows_object_axes(vocab):
    assert set(vocab.canonical_directions("mug")) == {"+up", "-up"}
    assert set(vocab.canonical_directions("teapot")) == set(CANONICAL_DIRS)
    with pytest.raises(ParseRejection) as e:
        parse_point_axis_directions(json.dumps(
            {"candidate_id": 3, "direction": "+front"}), vocab, "mug")
    assert "+up" in str(e.value)


def test_selection_named_arm_still_parses(vocab):
    raw = json.dumps({"candidate_id": 3, "axis": "teapot.pour_axis",
                      "sign": "+", "secondary": "teapot.up_axis"})
    sel = parse_point_axis_named(raw, vocab)
    assert sel.axis == "teapot.pour_axis" and sel.secondary == "teapot.up_axis"
    with pytest.raises(ParseRejection):
        parse_point_axis_named(json.dumps(
            {"candidate_id": 3, "axis": "teapot.magic_axis", "sign": "+"}),
            vocab)


# ----------------------------------------------------------- touchpoint #3

def good_emission():
    return {
        "stage": 2, "name": "pour", "active": "teapot", "passive": "mug",
        "path_tsr": {
            "rot": [{"axis": "teapot.+up", "relation": "parallel",
                     "reference": "world.z", "tol": "moderate"}],
            "trans": "free"},
        "subgoal_tsr": {
            "rot": [{"axis": "teapot.+front",
                     "relation": "antiparallel",
                     "reference": "mug.+up", "tol": "loose"},
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


# ---------------------------------------- #3 six-direction axis alphabet

def _rot(axis, rel, ref, tol="moderate"):
    return {"axis": axis, "relation": rel, "reference": ref, "tol": tol}


def _emission_with(rows):
    doc = good_emission()
    doc["subgoal_tsr"]["rot"] = rows
    return doc


def test_direction_names_is_the_drawn_triad_and_nothing_else(vocab):
    """The #3 accept set is exactly the licensed signed canonical
    directions per object: teapot all six, the sim mug only +-up (no
    front_axis column). Fitted axes and world.* are NOT in it — the
    drawn triad and the token alphabet are the same set."""
    d = vocab.direction_names()
    assert {f"teapot.{t}" for t in CANONICAL_DIRS} <= d
    assert {"mug.+up", "mug.-up"} <= d
    assert "mug.+front" not in d and "mug.-left" not in d
    assert not any(n.endswith("_axis") for n in d)
    assert not any(n.startswith("world.") for n in d)
    # the fitted table is untouched for its other customers
    assert "teapot.pour_axis" in vocab.axis_names()


@pytest.mark.parametrize("a,rel,ref,want_rel", [
    ("teapot.+up", "parallel", "mug.+up", "parallel"),
    ("teapot.-up", "antiparallel", "mug.+up", "parallel"),
    ("teapot.+up", "antiparallel", "mug.-up", "parallel"),
    ("teapot.-up", "parallel", "mug.-up", "parallel"),
    ("teapot.-up", "parallel", "mug.+up", "antiparallel"),
    ("teapot.-front", "antiparallel", "world.z", "parallel"),
    ("teapot.+front", "antiparallel", "world.z", "antiparallel"),
    ("teapot.-left", "perpendicular", "mug.+up", "perpendicular"),
    ("teapot.+left", "perpendicular", "mug.-up", "perpendicular"),
])
def test_rot_row_sign_normalization(vocab, a, rel, ref, want_rel):
    """Signs are stripped at parse time; a negative sign product flips
    parallel<->antiparallel; perpendicular is sign-invariant; world.z
    carries '+'. The compiler sees only unsigned canonical axis names."""
    em = parse_emission(json.dumps(_emission_with([_rot(a, rel, ref)])), vocab)
    row = em.subgoal_tsr.rot[0]
    assert row.axis == f"teapot.{a.split('.')[1][1:].replace('left', 'lateral')}_axis"
    assert row.reference == ("world.z" if ref == "world.z"
                             else "mug.up_axis")
    assert row.relation == want_rel
    assert row.tol == "moderate"


def test_rot_row_equivalent_forms_yield_identical_rows(vocab):
    a = parse_emission(json.dumps(_emission_with(
        [_rot("teapot.-up", "antiparallel", "mug.+up")])), vocab)
    b = parse_emission(json.dumps(_emission_with(
        [_rot("teapot.+up", "parallel", "mug.+up")])), vocab)
    assert a.subgoal_tsr.rot == b.subgoal_tsr.rot


@pytest.mark.parametrize("slot,bad", [
    ("axis", "teapot.pour_axis"),      # fitted axis name: not a token any more
    ("axis", "teapot.up_axis"),        # unsigned canonical column name
    ("axis", "+up"),                   # unqualified
    ("axis", "mug.+front"),            # unlicensed for this object (no front)
    ("axis", "world.z"),               # world is a reference, never an axis
    ("reference", "world.x"),          # scene layout, not gravity
    ("reference", "world.y"),
    ("reference", "mug.up_axis"),
    ("reference", "mug.+front"),
    ("reference", "teapot.pour_axis"),
])
def test_rot_row_rejects_tokens_outside_the_direction_alphabet(vocab, slot, bad):
    row = _rot("teapot.+up", "parallel", "mug.+up")
    row[slot] = bad
    with pytest.raises(ParseRejection) as e:
        parse_emission(json.dumps(_emission_with([row])), vocab)
    assert repr(bad) in str(e.value) and slot in str(e.value)


def test_rot_row_reference_rejection_offers_world_z(vocab):
    """The rejection text is the model's repair input: it must list
    world.z among the allowed references, and the list is sorted so the
    text is deterministic across runs."""
    row = _rot("teapot.+up", "parallel", "world.x")
    with pytest.raises(ParseRejection) as e:
        parse_emission(json.dumps(_emission_with([row])), vocab)
    msg = str(e.value)
    assert "'world.z'" in msg and "'world.x'" not in msg.split("allowed")[1]
    allowed = msg.split("allowed: ")[1]
    assert allowed == str(sorted(eval(allowed)))


def test_along_takes_a_fused_direction_token(vocab):
    doc = good_emission()
    doc["subgoal_tsr"]["trans"] = [{"term": "along", "axis": "mug.-up"}]
    em = parse_emission(json.dumps(doc), vocab)
    t = em.subgoal_tsr.trans[0]
    assert (t.term, t.axis, t.sign) == ("along", "mug.up_axis", "-")


@pytest.mark.parametrize("term", [
    {"term": "along", "axis": "mug.up_axis", "sign": "-"},   # old form
    {"term": "along", "axis": "mug.-up", "sign": "-"},       # redundant sign
    {"term": "along", "axis": "mug.+front"},                 # unlicensed
    {"term": "along", "axis": "world.z"},
])
def test_along_rejects_separate_sign_and_non_direction_tokens(vocab, term):
    doc = good_emission()
    doc["subgoal_tsr"]["trans"] = [term]
    with pytest.raises(ParseRejection):
        parse_emission(json.dumps(doc), vocab)


def test_emission_prompt_speaks_the_direction_alphabet(vocab):
    """The #3 prompt lists licensed directions per object and no fitted
    axis names; world.z stays as the one world reference; the point-only
    #2 selection contributes only its candidate id (no axis/sign
    injection). The schema-only and image arms share this text."""
    from manip_sim.vlm import build_emission_prompt
    system, msgs = build_emission_prompt(STAGE, vocab)
    assert vocab.describe_directions() in system
    assert "pour_axis" not in system and "handle_axis" not in system
    assert "up_axis" not in system and "front_axis" not in system
    assert "world.z" in system and "world.x" not in system
    for d in CANONICAL_DIRS:
        assert f"obj.{d}" in system
    sel = PointAxisSelection(candidate_id=12, axis=None, sign="+",
                             secondary=None, rationale="")
    _, msgs = build_emission_prompt(STAGE, vocab, selection=sel)
    text = msgs[0]["content"][0]["text"]
    assert "candidate 12" in text and "None" not in text and "sign" not in text


def test_describe_directions_lists_licensed_set_per_object(vocab):
    desc = vocab.describe_directions()
    assert "object `mug`:" in desc and "+up, -up" in desc
    assert "+front, -front, +left, -left, +up, -up" in desc
    assert "opening_center" in desc and "rim_radius" in desc
    assert "pour_axis" not in desc


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
            return json.dumps({"candidate_id": 99, "rationale": "r"})
        return json.dumps({"candidate_id": 7, "rationale": "r"})

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

def test_emission_prompt_single_object_stage_forbids_relation_rows(vocab):
    """Grasp (no passive) constrains the gripper frame with Tw_e = I; the
    compiler's nominal gate rejects relation rows there, so the prompt
    must steer the model to free rows. Two-object stages get no such
    instruction."""
    from manip_sim.vlm import build_emission_prompt
    grasp = StageSpec(index=0, name="grasp", active="teapot", passive=None,
                      parts={"teapot": ("handle",)})
    _, msgs = build_emission_prompt(grasp, vocab)
    text = msgs[0]["content"][0]["text"]
    assert "no passive object" in text and '"relation": "free"' in text
    _, msgs = build_emission_prompt(STAGE, vocab)
    assert "no passive object" not in msgs[0]["content"][0]["text"]


def test_emission_retry_appends_rejections_as_user_turns(vocab):
    """Compile-gate rejections ride into the next emission as follow-up
    user turns (emit_constraints.py's stopgap for touchpoint #5)."""
    seen = []
    def transport(payload):
        seen.append(payload)
        return json.dumps({
            "stage": 1, "name": "transport", "active": "teapot",
            "passive": "mug",
            "path_tsr": {"rot": "free", "trans": "free"},
            "subgoal_tsr": {"rot": "free", "trans": "free"},
            "verify": "x"})
    c = Client(transport=transport)
    c.emit_constraints(STAGE, vocab)
    prior = c.logs[-1].raw
    assert prior and json.loads(prior)["passive"] == "mug"
    c.emit_constraints(STAGE, vocab, rejections=[(prior, "s.rot[0]: bad")])
    turns = seen[1]["messages"]
    # prior emission replayed as the assistant turn, rejection as the user turn
    assert [t["role"] for t in turns[-2:]] == ["assistant", "user"]
    assert turns[-2]["content"][0]["text"] == prior
    assert "s.rot[0]: bad" in turns[-1]["content"][0]["text"]


def test_emission_prompt_states_ownership_and_fixed_w(vocab):
    """The compiler roots w by rule (never emitted), requires trans
    anchors on the passive object and rot axes on the active one; the
    prompt must say so up front rather than spend the retry budget
    discovering it."""
    from manip_sim.vlm import build_emission_prompt
    system, msgs = build_emission_prompt(STAGE, vocab)
    text = msgs[0]["content"][0]["text"]
    assert "fixed by rule, not emitted" in system
    assert "w_origin" not in system and "w_axis" not in system
    assert "no row mentions are FREE" in system
    assert f"axis must be a {STAGE.active}." in text
    assert f"anchors must be {STAGE.passive}." in text
    assert "GRIPPER" not in text
    grasp = StageSpec(index=0, name="grasp", active="teapot", passive=None,
                      parts={"teapot": ("handle",)})
    _, msgs = build_emission_prompt(grasp, vocab)
    text = msgs[0]["content"][0]["text"]
    assert "GRIPPER" in text and "cannot be grounded" in text
    assert f"anchor trans terms on {grasp.active} points" in text
