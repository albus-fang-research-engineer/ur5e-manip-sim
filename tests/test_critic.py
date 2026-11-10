"""manip_sim.critic: edit grammar, parser, applier, loop guards, preview
report. Offline; no VLM call. Geometry mirrors the 11/05 hardware carry:
the carry anchored on the teapot handle, the spout far from the mug."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from manip_sim import critic as cr
from manip_sim.frames import Symbols
from manip_sim.selection import resolve_selection
from manip_sim.vlm import (Client, ParseRejection, PointAxisSelection, RotRow,
                           StageEmission, TransTerm, TSRSpec, Vocabulary,
                           build_critic_prompt_v2)

# ------------------------------------------------------------------ fixtures

TEAPOT_POINTS = {"base_center": [0.0, 0.0, 0.0], "body_center": [0.0, 0.0, 0.06],
                 "handle_center": [-0.09, 0.0, 0.07], "lid_center": [0.0, 0.0, 0.12],
                 "spout_center": [0.10, 0.0, 0.08]}
AXES = {"up_axis": [0, 0, 1.0], "front_axis": [1.0, 0, 0], "lateral_axis": [0, 1.0, 0]}


def _symbols():
    tp = Symbols(object="teapot",
                 points={k: np.array(v) for k, v in TEAPOT_POINTS.items()},
                 axes={k: np.array(v) for k, v in AXES.items()})
    mug = Symbols(object="mug",
                  points={"rim_center": np.array([0, 0, 0.10]),
                          "opening_center": np.array([0, 0, 0.09])},
                  axes={"up_axis": np.array([0, 0, 1.0])},
                  quantities={"rim_radius": 0.0445})
    return {"teapot": tp, "mug": mug}


VOCAB = Vocabulary.from_symbols(_symbols())


def _pool(points: dict) -> dict:
    pool = {i: {"id": i, "source": "constructed", "symbol": n,
                "xyz": np.asarray(p, float)}
            for i, (n, p) in enumerate(points.items())}
    pool[len(pool)] = {"id": len(pool), "source": "part", "part": "body",
                       "xyz": np.array([0.05, 0.05, 0.05])}   # unnamed
    return pool


POOLS = {"teapot": _pool(TEAPOT_POINTS),
         "mug": _pool({"rim_center": [0, 0, 0.10], "opening_center": [0, 0, 0.09]})}
# teapot pool ids: 0 base, 1 body, 2 handle, 3 lid, 4 spout, 5 unnamed


def _emissions():
    grasp = StageEmission(
        stage=0, name="grasp teapot by handle", active="teapot", passive=None,
        path_tsr=TSRSpec(rot=(), trans=(TransTerm("free"),)),
        subgoal_tsr=TSRSpec(rot=(), trans=(TransTerm("free"),)), verify="")
    carry = StageEmission(
        stage=1, name="carry teapot above mug", active="teapot", passive="mug",
        path_tsr=TSRSpec(
            rot=(RotRow("teapot.up_axis", "parallel", "world.z", "moderate"),
                 RotRow(None, "free", None, None, "yaw")),
            trans=(TransTerm("free"),)),
        subgoal_tsr=TSRSpec(
            rot=(RotRow("teapot.up_axis", "parallel", "world.z", "loose"),),
            trans=(TransTerm("above", anchor="mug.rim_center", clearance="small",
                             slack="moderate"),
                   TransTerm("centered", anchor="mug.rim_center", tol="moderate"))),
        verify="", goal="spout over the mug")
    pour = StageEmission(
        stage=2, name="tilt teapot to pour", active="teapot", passive="mug",
        path_tsr=TSRSpec(rot=(RotRow("teapot.up_axis", "perpendicular",
                                     "world.z", "tight"),),
                         trans=(TransTerm("inside", anchor="mug.opening_center",
                                          slack="snug"),)),
        subgoal_tsr=TSRSpec(rot=(RotRow("teapot.front_axis", "antiparallel",
                                        "world.z", "tight"),),
                            trans=(TransTerm("free"),)), verify="")
    return {"grasp": grasp, "transport_active": carry, "pour": pour}


def _state(**kw) -> cr.SpecState:
    sels = {"grasp": PointAxisSelection(2, None, "+", None, ""),
            "transport_active": PointAxisSelection(2, None, "+", None, ""),
            "pour": PointAxisSelection(4, None, "+", None, ""),
            "transport_passive": PointAxisSelection(0, None, "+", None, "")}
    objs = {"grasp": "teapot", "transport_active": "teapot", "pour": "teapot",
            "transport_passive": "mug"}
    return cr.SpecState(emissions=_emissions(), selections=sels, objects=objs,
                        pools=POOLS, **kw)


CARRY = "carry teapot above mug"
POUR = "tilt teapot to pour"
GRASP = "grasp teapot by handle"


def _stages(verdict="reject"):
    return [{"stage": s, "expected": "e", "observed": "o"}
            for s in (GRASP, CARRY, POUR)]


def _raw(edits, verdict="reject", diagnosis="handle over the mug, spout 14 cm off",
         stages=None):
    return json.dumps({"stages": _stages() if stages is None else stages,
                       "verdict": verdict, "edits": edits,
                       "diagnosis": diagnosis})


def _parse(edits, **kw):
    return cr.parse_verdict(_raw(edits, **kw), VOCAB, _state())


def _repair(edits, st=None, ledger=None):
    return cr.parse_repair_v2(json.dumps({"edits": edits, "diagnosis": "x"}),
                              VOCAB, st or _state(), ledger)


# ------------------------------------------------------------------ action sets

def test_action_sets_split_by_what_the_evidence_can_justify():
    assert set(cr.CRITIC_ACTIONS) == {"tighten_tolerance", "tighten_slack",
                                      "reselect_point", "reselect_axis",
                                      "reemit_stage"}
    # widen_clearance shifts the above/below band (adds poses), so repair-only
    assert not {"relax_tolerance", "free_dof", "widen_slack",
                "widen_clearance"} & set(cr.CRITIC_ACTIONS)
    assert {"relax_tolerance", "free_dof", "widen_slack",
            "widen_clearance"} <= set(cr.REPAIR_ACTIONS_V2)
    for a in cr.CRITIC_ACTIONS + cr.REPAIR_ACTIONS_V2:
        assert a in cr.ACTION_TABLE


def test_v1_repair_actions_untouched():
    from manip_sim.vlm import REPAIR_ACTIONS
    assert REPAIR_ACTIONS == ("relax_tolerance", "free_dof", "reselect_point",
                              "reselect_axis", "insert_stage", "reorder_stages",
                              "widen_clearance", "widen_slack")


# ------------------------------------------------------------------ parsing

def test_accept_parses_with_every_stage_checked():
    v = _parse([], verdict="accept", diagnosis="")
    assert v.verdict == "accept" and v.edits == ()
    assert [c.stage for c in v.stages] == [GRASP, CARRY, POUR]


@pytest.mark.parametrize("stages, msg", [
    ([{"stage": CARRY, "expected": "e", "observed": "o"}], "every stage"),
    (_stages() + [{"stage": CARRY, "expected": "e", "observed": "o"}], "every stage"),
    ([{"stage": GRASP, "expected": "", "observed": "o"}], "non-empty"),
    ([{"stage": "nope", "expected": "e", "observed": "o"}], "licensed"),
])
def test_stages_must_cover_each_stage_once_with_text(stages, msg):
    with pytest.raises(ParseRejection, match=msg):
        _parse([], verdict="accept", diagnosis="", stages=stages)


def test_reject_needs_edits_and_diagnosis_accept_needs_none():
    with pytest.raises(ParseRejection, match="at least one"):
        _parse([])
    with pytest.raises(cr.EditRefusal) as e:
        _parse([{"target": CARRY, "action": "reemit_stage", "token": None}],
               diagnosis="  ")
    assert e.value.code == "missing_diagnosis"
    with pytest.raises(ParseRejection, match="no edits"):
        _parse([{"target": CARRY, "action": "reemit_stage", "token": None}],
               verdict="accept")


def test_reselect_point_maps_to_the_candidate_mark():
    v = _parse([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    (e,) = v.edits
    assert e.payload == (4,) and e.before == "teapot.handle_center"
    assert e.after == "teapot.spout_center" and e.role == "transport_active"


@pytest.mark.parametrize("edit, code", [
    # point not grounded on the active object
    ({"target": f"{CARRY}.point.teapot", "action": "reselect_point",
      "token": "mug.rim_center"}, "bad_token"),
    ({"target": f"{CARRY}.point.teapot", "action": "reselect_point",
      "token": "tight"}, "bad_token"),
    # the current pick
    ({"target": f"{CARRY}.point.teapot", "action": "reselect_point",
      "token": "teapot.handle_center"}, "no_change"),
    # unknown pointer / the passive's point / the context stage
    ({"target": f"{CARRY}.point.mug", "action": "reselect_point",
      "token": "mug.opening_center"}, "unknown_target"),
    ({"target": f"{GRASP}.point.teapot", "action": "reselect_point",
      "token": "teapot.body_center"}, "not_editable"),
    ({"target": f"{CARRY}.subgoal.rot[7]", "action": "tighten_tolerance",
      "token": "moderate"}, "unknown_target"),
    ({"target": "carry.subgoal.rot[0]", "action": "tighten_tolerance",
      "token": "moderate"}, "unknown_target"),
    # actions outside the critic set
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "relax_tolerance",
      "token": "loose"}, "unknown_action"),
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "insert_stage",
      "token": None}, "unknown_action"),
    # level tokens: wrong ladder, more than one step, wrong field
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
      "token": "snug"}, "bad_token"),
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
      "token": "tight"}, "not_one_step"),
    ({"target": f"{CARRY}.subgoal.trans[1]", "action": "tighten_slack",
      "token": "snug"}, "wrong_target"),
    ({"target": f"{POUR}.path.rot[0]", "action": "tighten_tolerance",
      "token": "tight"}, "not_one_step"),
    ({"target": f"{CARRY}.path.rot[1]", "action": "tighten_tolerance",
      "token": "tight"}, "wrong_target"),
    # directions
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "reselect_axis",
      "token": "mug.+up"}, "bad_token"),
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "reselect_axis",
      "token": "teapot.+up"}, "no_change"),
    ({"target": f"{CARRY}.subgoal.trans[0]", "action": "reselect_axis",
      "token": "teapot.+front"}, "wrong_target"),
    # reemit_stage takes a stage name and no token
    ({"target": f"{CARRY}.subgoal", "action": "reemit_stage", "token": None},
     "unknown_target"),
    ({"target": CARRY, "action": "reemit_stage", "token": "tight"}, "bad_token"),
])
def test_typed_refusals(edit, code):
    with pytest.raises(cr.EditRefusal) as e:
        _parse([edit])
    assert e.value.code == code
    assert isinstance(e.value, ParseRejection) and e.value.reason


def test_off_ladder_level_and_missing_object_are_typed_refusals():
    st = _state()
    ems = dict(st.emissions)
    carry = ems["transport_active"]
    ems["transport_active"] = carry.__class__(**{
        **carry.__dict__, "subgoal_tsr": TSRSpec(
            rot=carry.subgoal_tsr.rot,
            trans=(TransTerm("above", anchor="mug.rim_center", clearance="small"),))})
    st2 = cr.SpecState(ems, st.selections, st.objects, st.pools)
    with pytest.raises(cr.EditRefusal) as e:
        cr.parse_verdict(_raw([{"target": f"{CARRY}.subgoal.trans[0]",
                                "action": "tighten_slack", "token": "snug"}]),
                         VOCAB, st2)
    assert e.value.code == "wrong_target"
    objs = {k: v for k, v in st.objects.items() if k != "transport_active"}
    st3 = cr.SpecState(st.emissions, st.selections, objs, st.pools)
    for target in (f"{CARRY}.point.", f"{CARRY}.point.teapot"):
        with pytest.raises(cr.EditRefusal) as e:
            cr.parse_verdict(_raw([{"target": target, "action": "reselect_point",
                                    "token": "teapot.spout_center"}]), VOCAB, st3)
        assert e.value.code == "unknown_target"


def test_point_not_offered_to_call2_is_refused():
    st = _state(offered={"transport_active": frozenset({1, 2, 3})})
    raw = _raw([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    with pytest.raises(cr.EditRefusal) as e:
        cr.parse_verdict(raw, VOCAB, st)
    assert e.value.code == "point_not_in_pool"


def test_point_without_a_named_candidate_is_refused():
    pools = dict(POOLS)
    pools["teapot"] = {i: c for i, c in POOLS["teapot"].items()
                       if c.get("symbol") != "lid_center"}
    st = _state()
    st = cr.SpecState(st.emissions, st.selections, st.objects, pools)
    raw = _raw([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.lid_center"}])
    with pytest.raises(cr.EditRefusal) as e:
        cr.parse_verdict(raw, VOCAB, st)
    assert e.value.code == "point_not_in_pool"


def test_one_step_levels_in_the_right_direction():
    v = _parse([
        {"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
         "token": "moderate"},
        {"target": f"{CARRY}.subgoal.trans[0]", "action": "tighten_slack",
         "token": "snug"},
        {"target": f"{CARRY}.subgoal.trans[1]", "action": "tighten_tolerance",
         "token": "snug"},
    ])
    assert [(e.before, e.after) for e in v.edits] == [
        ("teapot.up_axis parallel world.z, tol loose",
         "teapot.up_axis parallel world.z, tol moderate"),
        ("above mug.rim_center, clearance small, slack moderate",
         "above mug.rim_center, clearance small, slack snug"),
        ("centered on mug.rim_center, tol moderate",
         "centered on mug.rim_center, tol snug")]
    r = _repair([{"target": f"{CARRY}.subgoal.trans[0]", "action": "widen_clearance",
                  "token": "medium"}])
    assert r.edits[0].after == "above mug.rim_center, clearance medium, slack moderate"
    assert r.edits[0].direction == 0
    with pytest.raises(cr.EditRefusal) as e:
        _repair([{"target": f"{CARRY}.subgoal.trans[0]", "action": "widen_clearance",
                  "token": "small"}])
    assert e.value.code == "not_one_step"
    with pytest.raises(cr.EditRefusal) as e:
        _parse([{"target": f"{CARRY}.subgoal.trans[0]", "action": "widen_clearance",
                 "token": "medium"}])
    assert e.value.code == "unknown_action"


def test_duplicate_target_and_reemit_conflict():
    with pytest.raises(cr.EditRefusal) as e:
        _parse([{"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
                 "token": "moderate"},
                {"target": f"{CARRY}.subgoal.rot[0]", "action": "reselect_axis",
                 "token": "teapot.+front"}])
    assert e.value.code == "duplicate_target"
    with pytest.raises(cr.EditRefusal) as e:
        _parse([{"target": CARRY, "action": "reemit_stage", "token": None},
                {"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    assert e.value.code == "conflict"


def test_repair_may_loosen_and_free():
    raw = json.dumps({"edits": [
        {"target": f"{CARRY}.subgoal.rot[0]", "action": "free_dof", "token": None},
        {"target": f"{CARRY}.subgoal.trans[0]", "action": "widen_slack",
         "token": "loose"}], "diagnosis": "no feasible IK at the goal"})
    v = cr.parse_repair_v2(raw, VOCAB, _state())
    assert v.source == "repair" and [e.direction for e in v.edits] == [-1, -1]
    with pytest.raises(cr.EditRefusal) as e:
        cr.parse_repair_v2(json.dumps({"edits": [
            {"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
             "token": "moderate"}], "diagnosis": "x"}), VOCAB, _state())
    assert e.value.code == "unknown_action"


# ------------------------------------------------------------------ applier

def test_reselect_swaps_the_selection_through_the_candidate_pool():
    st = _state()
    v = _parse([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    res = cr.apply_edits(st, v.edits, v.diagnosis)
    new = res.state.selections["transport_active"]
    assert new.candidate_id == 4 and new.axis is None and new.sign == "+"
    assert st.selections["transport_active"].candidate_id == 2      # pure
    # the one selection path: call #2's resolver anchors on the spout now
    rf = resolve_selection(new, POOLS["teapot"], _symbols()["teapot"])
    assert np.allclose(rf.frame.point, TEAPOT_POINTS["spout_center"])
    assert res.changed_roles == ("transport_active",) and res.reemit == {}


def test_level_and_axis_edits_change_one_token():
    st = _state()
    v = _parse([
        {"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
         "token": "moderate"},
        {"target": f"{CARRY}.path.rot[0]", "action": "reselect_axis",
         "token": "teapot.-front"}])
    res = cr.apply_edits(st, v.edits, v.diagnosis)
    em = res.state.emissions["transport_active"]
    assert em.subgoal_tsr.rot[0].tol == "moderate"
    assert em.subgoal_tsr.trans == st.emissions["transport_active"].subgoal_tsr.trans
    # signed direction folded into the relation, as the call #3 parser does
    assert em.path_tsr.rot[0] == RotRow("teapot.front_axis", "antiparallel",
                                        "world.z", "moderate")
    assert res.state.emissions["pour"] == st.emissions["pour"]


def test_reemit_marks_the_stage_with_the_diagnosis():
    v = _parse([{"target": CARRY, "action": "reemit_stage", "token": None}],
               diagnosis="no path row keeps the teapot upright")
    res = cr.apply_edits(_state(), v.edits, v.diagnosis)
    assert res.reemit == {"transport_active": "no path row keeps the teapot upright"}
    assert res.state.emissions == _state().emissions


def test_apply_is_all_or_none():
    st = _state()
    v = _parse([{"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
                 "token": "moderate"}])
    dup = v.edits + v.edits
    with pytest.raises(cr.EditRefusal):
        cr.apply_edits(st, dup, "x")
    assert st.emissions["transport_active"].subgoal_tsr.rot[0].tol == "loose"


def test_free_dof_keeps_row_indices():
    raw = json.dumps({"edits": [{"target": f"{CARRY}.path.rot[0]",
                                 "action": "free_dof", "token": None}],
                      "diagnosis": "x"})
    v = cr.parse_repair_v2(raw, VOCAB, _state())
    em = cr.apply_edits(_state(), v.edits, "x").state.emissions["transport_active"]
    assert em.path_tsr.rot[0].relation == "free" and len(em.path_tsr.rot) == 2


# ------------------------------------------------------------------ guards

def test_budget_spans_both_calls_and_ends_in_preview_rejected():
    led = cr.LoopLedger(budget=2)
    v1 = _parse([{"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
                  "token": "moderate"}])
    assert led.admit(v1) == "apply"
    led.record(v1)
    v2 = _repair([{"target": f"{POUR}.path.rot[0]", "action": "relax_tolerance",
                   "token": "moderate"}])
    assert led.admit(v2) == "apply"
    led.record(v2)
    out = led.admit(_parse([{"target": CARRY, "action": "reemit_stage",
                             "token": None}]))
    assert out["type"] == "preview_rejected" and out["rounds_used"] == 2
    assert len(out["history"]) == 2
    assert led.admit(_parse([], verdict="accept", diagnosis="")) == "accept"


def test_repair_may_not_undo_the_critics_repick():
    st = _state()
    led = cr.LoopLedger()
    v = _parse([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    led.admit(v)
    st = cr.apply_edits(st, v.edits, v.diagnosis).state
    led.record(v)
    back = _repair([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                     "token": "teapot.handle_center"}], st)
    with pytest.raises(cr.EditRefusal) as e:
        led.admit(back)
    assert e.value.code == "reversal"
    # a different point is not a reversal
    other = _repair([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                      "token": "teapot.lid_center"}], st)
    assert led.admit(other) == "apply"


def test_reversal_is_refused_inside_the_parse_retry_loop():
    st = _state()
    led = cr.LoopLedger()
    v = _parse([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    led.admit(v)
    st = cr.apply_edits(st, v.edits, v.diagnosis).state
    led.record(v)
    with pytest.raises(cr.EditRefusal) as e:
        _repair([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                  "token": "teapot.handle_center"}], st, led)
    assert e.value.code == "reversal" and isinstance(e.value, ParseRejection)


def test_reemit_forgets_that_stages_row_history_only():
    st = _state()
    led = cr.LoopLedger(budget=3)
    v = _parse([{"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
                 "token": "moderate"},
                {"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])
    led.admit(v)
    st = cr.apply_edits(st, v.edits, v.diagnosis).state
    led.record(v)
    r = _repair([{"target": CARRY, "action": "reemit_stage", "token": None}], st, led)
    led.admit(r)
    led.record(r)
    targets = [h["target"] for h in led.history]
    assert f"{CARRY}.subgoal.rot[0]" not in targets
    assert f"{CARRY}.point.teapot" in targets and CARRY in targets
    # the new emission's row may be relaxed; the point still may not go back
    assert led.admit(_repair([{"target": f"{CARRY}.subgoal.rot[0]",
                               "action": "relax_tolerance", "token": "loose"}],
                             st)) == "apply"
    with pytest.raises(cr.EditRefusal):
        _repair([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                  "token": "teapot.handle_center"}], st, led)


def test_repair_may_not_loosen_a_row_the_critic_tightened():
    st = _state()
    led = cr.LoopLedger()
    v = _parse([{"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
                 "token": "moderate"}])
    led.admit(v)
    st = cr.apply_edits(st, v.edits, v.diagnosis).state
    led.record(v)
    for edit in ({"target": f"{CARRY}.subgoal.rot[0]", "action": "relax_tolerance",
                  "token": "loose"},
                 {"target": f"{CARRY}.subgoal.rot[0]", "action": "free_dof",
                  "token": None}):
        with pytest.raises(cr.EditRefusal) as e:
            led.admit(_repair([edit], st))
        assert e.value.code == "reversal"
    # the same source may continue in its own direction
    more = cr.parse_verdict(_raw([{"target": f"{CARRY}.subgoal.rot[0]",
                                   "action": "tighten_tolerance", "token": "tight"}]),
                            VOCAB, st)
    assert led.admit(more) == "apply"


# ------------------------------------------------------------------ preview report

def _compiled():
    T_sub = np.eye(4)
    T_sub[:3, 3] = [0.09, 0.0, 0.07]     # handle over the rim center: handle at w
    Bw_goal = np.array([[-0.02, 0.02], [-0.02, 0.02], [0.01, 0.06],
                        [-0.26, 0.26], [-0.26, 0.26], [-np.pi, np.pi]])
    Bw_path = np.array([[-np.inf, np.inf]] * 3 + [[-0.26, 0.26], [-0.26, 0.26],
                                                  [-np.pi, np.pi]])
    return SimpleNamespace(path=SimpleNamespace(Bw=Bw_path),
                           subgoal=SimpleNamespace(Bw=Bw_goal, nominal=lambda: T_sub))


def _preview_input(samples=()):
    st = _state()
    return cr.StagePreviewInput(
        role="transport_active", emission=st.emissions["transport_active"],
        compiled=_compiled(), poses={"mug": np.eye(4)}, symbols=_symbols(),
        w_point=np.array([0, 0, 0.10]), e_point=np.array(TEAPOT_POINTS["handle_center"]),
        goal_samples=samples)


def test_stage_report_points_rows_and_readable_bounds():
    rep = cr.build_preview_report([_preview_input()], _state())
    (s,) = rep["stages"]
    assert s["stage"] == CARRY and s["editable"] is True
    assert s["point_choice"] == {"pointer": f"{CARRY}.point.teapot",
                                 "current": "teapot.handle_center"}
    pts = s["points_at_goal"]["points"]
    assert pts["handle_center"] == {"horizontal_cm": 0.0, "height_cm": 4.0}
    assert pts["spout_center"]["horizontal_cm"] == pytest.approx(19.0)
    assert s["path"]["bounds_in_w"] == {"x": "free", "y": "free", "z": "free",
                                        "roll": "-14.9 to +14.9 deg",       # 0.26 rad
                                        "pitch": "-14.9 to +14.9 deg",
                                        "yaw": "free"}
    assert s["subgoal"]["bounds_in_w"]["z"] == "+1.0 to +6.0 cm"
    assert s["subgoal"]["bounds_in_w"]["x"] == "-2.0 to +2.0 cm"
    assert "free_rows" not in s["path"] and "free_rows" not in s["subgoal"]
    assert {"pointer": f"{CARRY}.path.rot[1]", "text": "free (yaw)"} in s["path"]["rows"]
    ptrs = set(rep["editable_pointers"])
    assert f"{CARRY}.point.teapot" in ptrs and CARRY in ptrs
    assert not any(p.startswith(GRASP) for p in ptrs)
    json.dumps(rep)                                   # serializable


def test_point_choice_shown_only_for_the_active_objects_selection():
    st = _state()
    objs = dict(st.objects, transport_active="mug")
    sels = dict(st.selections, transport_active=PointAxisSelection(0, None, "+", None, ""))
    st2 = cr.SpecState(st.emissions, sels, objs, st.pools)
    rep = cr.build_preview_report([_preview_input()], st2)
    assert "point_choice" not in rep["stages"][0]
    assert f"{CARRY}.point.teapot" not in rep["editable_pointers"]
    assert f"{CARRY}.point.mug" not in rep["editable_pointers"]


def test_goal_spread_from_samples():
    a, b = np.eye(4), np.eye(4)
    a[:3, 3] = [0.09, 0.0, 0.07]
    b[:3, 3] = [0.09, 0.02, 0.09]
    s = cr.stage_report(_preview_input((a, b)))
    sp = s["goal_spread"]
    assert sp["n_goals"] == 2
    assert sp["points"]["handle_center"]["height_cm"] == [4.0, 6.0]
    assert sp["points"]["handle_center"]["horizontal_cm"] == [0.0, 2.0]


# ------------------------------------------------------------------ client

def test_client_critique_v2_retries_a_typed_refusal(tmp_path):
    img = tmp_path / "v.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    bad = _raw([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.handle_center"}])
    good = _raw([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                  "token": "teapot.spout_center"}])
    seen = []

    def transport(payload):
        seen.append(payload)
        return [bad, good][len(seen) - 1]

    cl = Client(transport=transport)
    rep = cr.build_preview_report([_preview_input()], _state())
    v = cl.critique_preview_v2(rep, [("state 1: initial scene", img)], VOCAB, _state())
    assert v.edits[0].payload == (4,) and len(seen) == 2
    assert "already selected" in seen[1]["messages"][-1]["content"][0]["text"]


def test_prompt_v2_lists_the_critic_actions_only():
    system, msgs = build_critic_prompt_v2({"stages": []}, [], VOCAB)
    for a in cr.CRITIC_ACTIONS:
        assert f"  {a}:" in system
    for a in ("relax_tolerance", "free_dof", "widen_slack", "widen_clearance",
              "insert_stage"):
        assert a not in system
    assert "expected" in system and "editable=false" in system
    assert "teapot.spout_center" in system and "teapot.+up" in system
    assert "mug.+up" in system and "mug.+front" not in system


# ------------------------------------------------------------------ collision facts

def test_bounds_text_one_sided_fixed_and_fractional_degrees():
    Bw = np.array([[0.03, np.inf], [-np.inf, 0.02], [0.05, 0.05],
                   [-np.radians(7.5), np.radians(7.5)], [0, 0], [-np.pi, np.pi]])
    assert cr._bounds(Bw) == {"x": "from +3.0 cm", "y": "up to +2.0 cm",
                              "z": "fixed at +5.0 cm", "roll": "-7.5 to +7.5 deg",
                              "pitch": "fixed at +0 deg", "yaw": "free"}


def test_context_stage_keeps_only_what_names_it():
    st = _state()
    si = cr.StagePreviewInput(
        role="grasp", emission=st.emissions["grasp"], compiled=_compiled(),
        poses={"mug": np.eye(4)}, symbols=_symbols(), w_point=np.zeros(3),
        editable=False)
    rec = cr.stage_report(si, "teapot.handle_center")
    assert set(rec) == {"stage", "role", "active", "editable", "goal", "point_choice"}
    assert rec["editable"] is False


def test_report_has_no_collision_facts():
    """Collision and search facts live in the image captions only."""
    import inspect
    assert "manifest" not in inspect.signature(cr.build_preview_report).parameters
    rep = cr.build_preview_report([_preview_input()], _state())
    assert "collisions" not in rep["stages"][0] and "grasp_collisions" not in rep


def test_num_never_prints_negative_zero():
    assert cr._num(-0.0004, "cm") == "+0.0" and cr._num(-0.0, "cm") == "+0.0"
    assert cr._num(-0.04, "deg") == "+0" and cr._num(-0.0, "deg") == "+0"
    assert cr._num(-0.06, "cm") == "-0.1" and cr._num(-15.0, "deg") == "-15"
    Bw = np.zeros((6, 2))
    Bw[0] = [-1e-5, 1e-5]
    assert cr._bounds(Bw)["x"] == "+0.0 to +0.0 cm"


def test_report_json_is_compact_one_stage_per_line():
    rep = cr.build_preview_report([_preview_input(), _preview_input()], _state())
    text = cr.report_json(rep)
    assert json.loads(text) == json.loads(json.dumps(rep))
    lines = text.splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    assert sum(l.startswith('  {"stage"') for l in lines) == 2
    assert len(text) < len(json.dumps(rep, indent=2))


def test_prompt_v2_says_collisions_are_the_planners():
    system, _ = build_critic_prompt_v2({"stages": []}, [], VOCAB)
    assert "the planner avoids collisions with the scene" in system
    assert "facts, not requests to tighten" in system
    assert "Collision notes in the image captions are facts" in system
    assert "Only \"no collision-free pose found in the region\" is evidence" in system


def test_client_critique_v2_uses_the_critic_token_budget(tmp_path):
    """The critic v2 request carries CRITIC_MAX_TOKENS; the client default
    (calls #1-#3, pinned in _request.json) is untouched."""
    from manip_sim.vlm import CRITIC_MAX_TOKENS, MAX_TOKENS
    img = tmp_path / "v.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    seen = []

    def transport(payload):
        seen.append(payload)
        return _raw([], verdict="accept", diagnosis="fine")

    cl = Client(transport=transport)
    rep = cr.build_preview_report([_preview_input()], _state())
    cl.critique_preview_v2(rep, [("state 1: initial scene", img)], VOCAB, _state())
    assert seen[0]["max_tokens"] == CRITIC_MAX_TOKENS != MAX_TOKENS
    assert cl.max_tokens == MAX_TOKENS
