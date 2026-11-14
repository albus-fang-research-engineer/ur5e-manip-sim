"""VLM call #5, repair v2 (opt-in): the prompt (vlm.build_repair_prompt_v2),
the client touchpoint (Client.repair_v2) and the planner failure record
(repair_context.failure_from_report). Offline: every call goes through a
scripted transport. Fixtures are test_critic's (the 11/05 handle carry)."""

from __future__ import annotations

import json

import pytest

from manip_sim import critic as cr
from manip_sim.repair_context import failure_from_report
from manip_sim.vlm import (CRITIC_MAX_TOKENS, REPAIR_REASON_WITHHELD, Client,
                           VLMError, build_repair_prompt_v2)
from tests.test_critic import (CARRY, GRASP, POUR, VOCAB, _parse,
                               _preview_input, _state)

STAGE_NAMES = {r: e.name for r, e in _state().emissions.items()}


def test_grasp_lookahead_is_labeled_by_stage():
    """A grasp failure's per-candidate lookahead fractions name the mover
    stage no grasp can reach (the insert stage of a wrong-anchor fault)."""
    rep = {"status": "failed", "failure": {
        "kind": "grasp", "stage": "grasp",
        "message": "[grasp] grasp: no candidate with a positive lookahead score "
                   "survived approach + reach",
        "detail": {"skips": [], "lookahead": [
            {"score": 0.0, "fractions": [0.0, 0.0]},
            {"score": 0.0, "fractions": [0.5, 0.0]},
            {"score": 0.0, "fractions": [0.125, 0.0]}]}}}
    names = {"grasp": "grasp the marker", "transport_active": "align",
             "insert": "insert marker into cup"}
    rec = failure_from_report(rep, names, ["transport_active", "insert"])
    ev = rec["evidence"]
    assert "lookahead" not in ev
    assert ev["best_reachable_fraction_from_any_grasp"] == {
        "align": 0.5, "insert marker into cup": 0.0}
    assert ev["unreachable_stages"] == ["insert marker into cup"]
    assert "'insert marker into cup'" in rec["message"]
    # without the mover order the record is as before (unlabeled)
    assert "lookahead" in failure_from_report(rep, names)["evidence"]

# a plan_task report as scripts/plan_task.py writes it on failure
EMPTY_GOAL = {
    "scene": "scenes/pour_tea.json", "stage_plan": "outputs/x/stage_plan.json",
    "selections": "outputs/x/selections.json", "emissions": "outputs/x/em.json",
    "seed": 0, "status": "failed", "grasp": {"source": "standin"},
    "stages": [{"role": "transport_active", "name": CARRY}],
    "failure": {
        "kind": "empty_goal_set", "stage": "transport_active",
        "message": "[transport_active] empty_goal_set: no collision-free, "
                   "contained goal configuration",
        "detail": {"n_sampled": 120, "accept_rate": 0.41, "n_ik": 96,
                   "n_collision_free": 0, "n_contained": 0,
                   "offenders": {"teapot<->mug": 96}, "n_rounds": 1}},
    "seconds": 41.3}


def _report():
    return cr.build_preview_report([_preview_input()], _state())


def _failure():
    return failure_from_report(EMPTY_GOAL, STAGE_NAMES)


def _answer(edits, diagnosis="handle over the mug: no collision-free goal"):
    return json.dumps({"edits": edits, "diagnosis": diagnosis})


GOOD = _answer([{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                 "token": "teapot.spout_center"}])


def _scripted(*answers):
    seen = []

    def transport(payload):
        seen.append(payload)
        return answers[len(seen) - 1]
    return transport, seen


def _texts(messages):
    return [b["text"] for b in messages[0]["content"] if b["type"] == "text"]


# ------------------------------------------------------------------ prompt

def test_prompt_lists_the_repair_actions_only():
    system, _ = build_repair_prompt_v2(_report(), _failure(), VOCAB)
    for a in cr.REPAIR_ACTIONS_V2:
        assert f"  {a}: {cr.ACTION_TABLE[a][2]}." in system
    for a in set(cr.CRITIC_ACTIONS) - set(cr.REPAIR_ACTIONS_V2):
        assert a not in system                      # tighten_tolerance, tighten_slack
    assert "editable_pointers" in system and "editable=false" in system
    assert "Never give a number" in system and "one step only" in system
    assert "contact / small / medium / large" in system
    assert "teapot.spout_center" in system and "teapot.+up" in system
    assert "\"edits\"" in system and "\"stages\"" not in system


def test_prompt_carries_the_report_and_the_typed_failure():
    _, msgs = build_repair_prompt_v2(_report(), _failure(), VOCAB)
    report_text, failure_text = _texts(msgs)
    assert report_text == "Specification report:\n" + cr.report_json(_report())
    assert failure_text.startswith("Planner failure:\n")
    rec = json.loads(failure_text.split("\n", 1)[1])
    assert rec["stage"] == CARRY and rec["type"] == "empty_goal_set"
    assert rec["evidence"]["offenders"] == {"teapot<->mug": 96}


def test_reason_withheld_hides_the_failure_and_keeps_everything_else():
    typed_sys, typed = build_repair_prompt_v2(_report(), _failure(), VOCAB)
    sys_w, withheld = build_repair_prompt_v2(_report(), _failure(), VOCAB,
                                             reason_withheld=True)
    assert sys_w == typed_sys
    t_report, _ = _texts(typed)
    w_report, w_failure = _texts(withheld)
    assert w_report == t_report                     # the rows are still there
    assert f"{CARRY}.subgoal.rot[0]" in w_report
    assert w_failure == REPAIR_REASON_WITHHELD
    assert w_failure.count(".") == 1                # one sentence
    for leak in ("empty_goal_set", CARRY, "teapot<->mug", "96", "120"):
        assert leak not in w_failure
    # the record itself is optional when withheld
    assert build_repair_prompt_v2(_report(), None, VOCAB,
                                  reason_withheld=True) == (sys_w, withheld)
    with pytest.raises(ValueError):
        build_repair_prompt_v2(_report(), None, VOCAB)


def test_views_follow_the_failure_with_their_captions(tmp_path):
    img = tmp_path / "v.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    _, msgs = build_repair_prompt_v2(_report(), _failure(), VOCAB,
                                     views=[("state 1: carry goal", img)])
    kinds = [b["type"] for b in msgs[0]["content"]]
    assert kinds == ["text", "text", "text", "image"]
    assert msgs[0]["content"][2]["text"] == "state 1: carry goal"


# ------------------------------------------------------------------ client

def test_client_repair_v2_parses_a_valid_answer():
    transport, seen = _scripted(GOOD)
    cl = Client(transport=transport)
    v = cl.repair_v2(_report(), _failure(), VOCAB, _state())
    assert isinstance(v, cr.Verdict) and v.source == "repair"
    assert v.edits[0].payload == (4,) and v.diagnosis.startswith("handle")
    assert len(seen) == 1 and seen[0]["max_tokens"] == CRITIC_MAX_TOKENS
    assert cl.logs[-1].touchpoint == "repair_v2"


@pytest.mark.parametrize("bad, reason", [
    # a critic-only action
    ({"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
      "token": "moderate"}, "is not a repair action"),
    # two steps: tight -> loose
    ({"target": f"{POUR}.path.rot[0]", "action": "relax_tolerance",
      "token": "loose"}, "one step only"),
    # the grasp is context: not among editable_pointers
    ({"target": f"{GRASP}.path.trans[0]", "action": "free_dof", "token": None},
     "is context"),
    # not a pointer of this specification
    ({"target": f"{CARRY}.subgoal.trans[7]", "action": "widen_slack",
      "token": "loose"}, "has 2 rows"),
])
def test_client_repair_v2_retries_a_refused_edit(bad, reason):
    transport, seen = _scripted(_answer([bad]), GOOD)
    cl = Client(transport=transport)
    v = cl.repair_v2(_report(), _failure(), VOCAB, _state())
    assert v.edits[0].action == "reselect_point" and len(seen) == 2
    retry = seen[1]["messages"][-1]["content"][0]["text"]
    assert retry.startswith("Your response was rejected:") and reason in retry
    assert cl.logs[-1].attempts == 2 and len(cl.logs[-1].rejections) == 1


def test_client_repair_v2_respects_the_no_reverse_rule():
    st, led = _state(), cr.LoopLedger()
    v = _parse([{"target": f"{CARRY}.subgoal.rot[0]", "action": "tighten_tolerance",
                 "token": "moderate"}])
    assert led.admit(v) == "apply"
    st = cr.apply_edits(st, v.edits, v.diagnosis).state
    led.record(v)
    undo = _answer([{"target": f"{CARRY}.subgoal.rot[0]",
                     "action": "relax_tolerance", "token": "loose"}])
    transport, seen = _scripted(undo, GOOD)
    out = Client(transport=transport).repair_v2(_report(), _failure(), VOCAB,
                                                st, ledger=led)
    assert "would undo the critic's" in seen[1]["messages"][-1]["content"][0]["text"]
    assert led.admit(out) == "apply"
    # without the ledger the same undo parses: the rule lives in the ledger
    transport, _ = _scripted(undo)
    assert Client(transport=transport).repair_v2(
        _report(), _failure(), VOCAB, st).edits[0].after.endswith("tol loose")


def test_client_repair_v2_gives_up_after_the_parse_retry_budget():
    bad = _answer([{"target": f"{CARRY}.subgoal.rot[0]",
                    "action": "tighten_tolerance", "token": "tight"}])
    transport, seen = _scripted(bad, bad, bad)
    with pytest.raises(VLMError):
        Client(transport=transport).repair_v2(_report(), None, VOCAB, _state(),
                                              reason_withheld=True)
    assert len(seen) == 3


# ------------------------------------------------------------------ failure record

def test_failure_from_plan_task_report_keeps_evidence_drops_paths_and_timings():
    rec = _failure()
    assert rec == {
        "stage": CARRY, "type": "empty_goal_set",
        "message": "no collision-free, contained goal configuration",
        "evidence": {"n_sampled": 120, "accept_rate": 0.41, "n_ik": 96,
                     "n_collision_free": 0, "n_contained": 0, "n_rounds": 1,
                     "offenders": {"teapot<->mug": 96}}}
    text = json.dumps(rec)
    for gone in ("outputs/", "seconds", "41.3", "transport_active", "scenes/"):
        assert gone not in text


def test_failure_from_report_compile_slot_and_handoff_names_stages():
    compile_rep = {"status": "failed", "failure": {
        "kind": "compile", "stage": "pour",
        "message": f"[pour] compile: {POUR}.path.trans[0]: entry outside the "
                   "z band",
        "detail": {"slot": f"{POUR}.path.trans[0]"}}}
    rec = failure_from_report(compile_rep, STAGE_NAMES)
    assert rec == {"stage": POUR, "type": "compile",
                   "message": f"{POUR}.path.trans[0]: entry outside the z band",
                   "evidence": {"slot": f"{POUR}.path.trans[0]"}}
    handoff = {"status": "failed", "failure": {
        "kind": "no_admitted_handoff", "stage": "transport_active",
        "message": "[transport_active] no_admitted_handoff: none of this "
                   "stage's goals is a start pour accepts",
        "detail": {"draws": 3, "checked": 30, "admitted": 0,
                   "rejected": {f"{POUR}.path.trans[0]: entry above the band": 30}}}}
    rec = failure_from_report(handoff, STAGE_NAMES)
    assert rec["stage"] == CARRY and rec["type"] == "no_admitted_handoff"
    assert rec["evidence"]["admitted"] == 0 and rec["evidence"]["checked"] == 30
    start = {"status": "failed", "failure": {
        "kind": "synthesized_start_outside_path", "stage": "pour",
        "message": "[pour] synthesized_start_outside_path: the synthesized "
                   "start is not inside the stage's path TSR",
        "detail": {"excess": [0.0, 0.0, 0.031, 0.0, 0.2618, 0.0]}}}
    assert failure_from_report(start, STAGE_NAMES)["evidence"] == {
        "excess": {"z": "+3.1 cm", "pitch": "+15 deg"}}


def test_failure_from_report_success_is_none_and_running_is_an_error():
    assert failure_from_report({"status": "ok", "stages": []}, STAGE_NAMES) is None
    with pytest.raises(ValueError):
        failure_from_report({"status": "running", "stages": []}, STAGE_NAMES)


def test_failure_from_report_grasp_keeps_counts_not_tables():
    rep = {"status": "failed", "failure": {
        "kind": "grasp", "stage": "grasp",
        "message": "[grasp] grasp: no reachable, collision-free grasp",
        "detail": {"source": "standin", "part": "handle", "proposed": 80,
                   "classified": 31, "ik_tried": 12, "ik_feasible": 4,
                   "collision_free": 0, "offenders": {"gripper0<->table": 4},
                   "grip_site_height_above_table_m": [0.1, 0.2],
                   "lookahead": [{"score": 0.0, "fractions": [0.0, 0.5],
                                  "slide_offset_mm": 3.0}]}}}
    rec = failure_from_report(rep, STAGE_NAMES)
    assert rec["stage"] == GRASP and rec["type"] == "grasp"
    assert rec["evidence"] == {"proposed": 80, "classified": 31, "ik_tried": 12,
                               "ik_feasible": 4, "collision_free": 0,
                               "offenders": {"gripper0<->table": 4},
                               "lookahead": [{"score": 0.0, "fractions": [0.0, 0.5]}]}


@pytest.mark.parametrize("line, stage, kind, message", [
    ("[transport] FAILED no_admitted_handoff: no transport goal is a start "
     "the pour accepts", CARRY, "no_admitted_handoff",
     "no transport goal is a start the pour accepts"),
    ("[pour] no feasible pour configs — try a smaller --tilt-deg or a "
     "different grasp seed.", POUR, "empty_goal_set", "no feasible pour configs"),
    ("[pour] planning failed: timeout", POUR, "planning", "planning failed: timeout"),
    ("[grasp] no reachable grasp — add IK seeds or revisit the approach "
     "elevation.", GRASP, "grasp", "no reachable grasp"),
])
def test_failure_from_plan_pour_tea_exit_line(line, stage, kind, message):
    """plan_pour_tea writes no report.json; its SystemExit line, recorded as
    the failure, is typed with plan_task's kinds and its stage label mapped
    to the stage name (transport -> transport_active)."""
    rec = failure_from_report({"status": "failed", "failure": line}, STAGE_NAMES)
    assert rec == {"stage": stage, "type": kind, "message": message}
    assert "--" not in json.dumps(rec)
