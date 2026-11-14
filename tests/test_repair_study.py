"""manip_sim.repair_study: outcomes, cost, localization, collateral,
oscillation and the per-fault table, on hand-written records."""

from __future__ import annotations

from manip_sim import repair_study as rs

CARRY = "carry teapot above mug"
POUR = "tilt teapot to pour"
FAULT = {"fault_id": "pour_wrong_anchor", "task": "pour_tea", "category": "wrong_anchor",
         "role": "transport_active", "stage": CARRY, "pointer": f"{CARRY}.point.teapot",
         "before": "teapot.spout_tip", "after": "teapot.handle_center",
         "expect": "", "unfixable_by": ["render_check"]}


def _call(attempts=1, i=100, o=10):
    return {"touchpoint": "x", "attempts": attempts, "input_tokens": i, "output_tokens": o}


def _rec(cond, rounds, final, diff=()):
    return {"fault": FAULT, "condition": cond, "repeat": 0, "rounds": rounds,
            "final": final, "diff": list(diff)}


TYPED_FIXED = _rec("typed", [
    {"kind": "critic", "calls": [_call(2)], "stage_sha": {CARRY: "a", POUR: "p"},
     "edits": [{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                "token": "teapot.spout_tip", "before": "teapot.handle_center",
                "after": "teapot.spout_tip", "direction": 0}]},
    {"kind": "critic", "calls": [_call()], "stage_sha": {CARRY: "b", POUR: "p"}, "edits": []},
], {"accepted": True, "grades": [{"seed": s, "success": True} for s in range(3)]})

CHECK_FAILED = _rec("render_check", [
    {"kind": "check", "stage": CARRY, "calls": [_call()], "labels": [{"stage": CARRY, "label": "failure"}]},
    {"kind": "regenerate", "stage": CARRY, "calls": [_call()], "stage_sha": {CARRY: "c"}},
    {"kind": "check", "stage": CARRY, "calls": [_call()]},
    {"kind": "regenerate", "stage": CARRY, "calls": [_call()], "stage_sha": {CARRY: "d"}},
    {"kind": "regenerate", "stage": CARRY, "calls": [_call()], "stage_sha": {CARRY: "c"}},
], {"accepted": False, "grades": []},
    diff=[{"pointer": f"{CARRY}.point.teapot"}, {"pointer": f"{CARRY}.subgoal.trans[0]"},
          {"pointer": f"{CARRY}.path.rot[0]"}])


def test_outcomes():
    assert rs.outcome(TYPED_FIXED) == "fixed"
    assert rs.outcome(CHECK_FAILED) == "budget_spent"
    assert rs.outcome(_rec("typed", [], {"accepted": True, "grades": []})) == "accepted_plan_failed"
    two_of_three = [{"seed": 0, "success": True}, {"seed": 1, "success": False},
                    {"seed": 2, "success": True}]
    assert rs.outcome(_rec("typed", [], {"grades": two_of_three})) == "fixed"
    assert rs.outcome(_rec("typed", [], {"grades": two_of_three[1:2] * 2 + two_of_three[:1]})) \
        == "planned_grader_failed"
    assert rs.outcome(_rec("typed", [], {"error": "API 500"})) == "error"


def test_calls_count_parse_retries():
    assert rs.n_calls(TYPED_FIXED) == 3
    assert rs.n_tokens(TYPED_FIXED) == (200, 20)


def test_localization_stage_and_pointer():
    assert rs.localized(TYPED_FIXED) == (True, True)
    assert rs.localized(CHECK_FAILED) == (True, None)
    assert rs.localized(_rec("typed", [], {})) == (None, None)


def test_stage_names_with_dots_resolve():
    rec = _rec("typed", [{"kind": "critic", "stage_sha": {"put 0.5 in. above": "x"},
                          "edits": [{"target": "put 0.5 in. above.subgoal.rot[0]",
                                     "before": "a", "after": "b", "direction": 1}]}], {})
    assert rs.first_step(rec)["stage"] == "put 0.5 in. above"


def test_collateral_excludes_the_fault_pointer():
    assert rs.collateral(CHECK_FAILED) == 2
    assert rs.collateral(TYPED_FIXED) == 0


def test_oscillation_counts_returns_and_reversals():
    assert rs.oscillations(CHECK_FAILED) == 1            # back to emission "c"
    rec = _rec("typed", [
        {"kind": "critic", "edits": [{"target": "t", "before": "moderate", "after": "tight",
                                      "direction": 1}]},
        {"kind": "repair", "edits": [{"target": "t", "before": "tight", "after": "moderate",
                                      "direction": -1}]}], {})
    assert rs.oscillations(rec) == 1
    assert rs.oscillations(TYPED_FIXED) == 0


def test_table_marks_structural_cells_and_calls_per_fix():
    s = rs.summarize([TYPED_FIXED, CHECK_FAILED], call_cap=8)
    cells = {c["condition"]: c for c in s["cells"]}
    assert cells["typed"]["calls_per_fix"] == 3.0 and cells["typed"]["fixed"] == 1
    assert cells["render_check"]["structural"] and cells["render_check"]["calls_per_fix"] is None
    assert cells["typed"]["fixed_within"][:4] == [0, 0, 1, 1]
    md = rs.table_md(s)
    assert "cannot fix by construction" in md and "1/1, 3.0 calls/fix" in md
