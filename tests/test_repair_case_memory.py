"""scripts/run_repair_case.py --critic-memory: the critic's history in the sim
loop (rounds from the shared ledger, stages tracked by role across a call #3
rename, nothing of a planner failure or the repair call's reasoning shown)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from manip_sim import critic as cr
from manip_sim.vlm import critic_review_text
from scripts.run_repair_case import Case


def _edit(action, target, before=None, after=None):
    return NS(action=action, target=target, token=None, before=before, after=after,
              direction=None)


def _case(names):
    case = NS(a=NS(critic_memory=True), earlier=[], _edit=Case._edit)
    case.stage_names = lambda: dict(names)
    case.review = lambda ledger: Case.review(case, ledger)
    case.remember = lambda *a, **k: Case.remember(case, *a, **k)
    return case


def test_memory_off_sends_no_review():
    case = _case({})
    case.a.critic_memory = False
    assert Case.review(case, cr.LoopLedger(budget=2)) is None


def test_history_tracks_roles_and_hides_planner_failures():
    names = {"transport_active": "carry the teapot", "pour": "tilt the teapot to pour"}
    case = _case(names)
    ledger = cr.LoopLedger(budget=2)
    r1 = case.review(ledger)
    assert (r1["round"], r1["rounds_max"], r1["earlier"]) == (1, 3, [])

    # round 1: the critic rejects; the carry row is tightened, the pour re-emitted,
    # and call #3 renames the pour stage
    stages = [NS(stage="carry the teapot", observed="upright row moderate"),
              NS(stage="tilt the teapot to pour", observed="front axis contradicts up")]
    verdict = NS(diagnosis="pour rows contradict each other",
                 edits=[_edit("tighten_tolerance", "carry the teapot.path.rot[0]",
                              "tol moderate", "tol tight"),
                        _edit("reemit_stage", "tilt the teapot to pour")])
    case.remember(1, dict(names), stages, verdict, ["pour"], after_accept=False)
    names["pour"] = "pour"
    ledger.rounds_used = 1
    r2 = case.review(ledger)
    assert (r2["round"], r2["rounds_max"]) == (2, 3)
    e = r2["earlier"][0]
    assert e["reemitted"] == ["pour"]
    assert [s["stage"] for s in e["stages"]] == ["carry the teapot", "pour"]
    _, text = critic_review_text(r2)
    assert "Round 1: rejected. Diagnosis: pour rows contradict each other" in text
    assert "observed, pour: (rows since replaced by a re-emission)" in text
    assert "front axis contradicts up" not in text
    assert "observed, carry the teapot: upright row moderate" in text
    assert "applied: tighten_tolerance on carry the teapot.path.rot[0]" in text
    assert "reemit_stage" not in text

    # round 2: the critic accepts, the planner fails, call #5 edits the pour and
    # has the carry written again
    stages2 = [NS(stage="carry the teapot", observed="tight upright row"),
               NS(stage="pour", observed="sound")]
    fix = NS(diagnosis="empty_goal_set: 0 of 120 goals collision-free (teapot<->mug)",
             edits=[_edit("tighten_slack", "pour.subgoal.trans[1]",
                          "slack moderate", "slack snug"),
                    _edit("reemit_stage", "carry the teapot")])
    case.remember(2, dict(names), stages2, fix, ["transport_active"], after_accept=True)
    ledger.rounds_used = 2
    r3 = case.review(ledger)
    system, text = critic_review_text(r3)
    assert "this is round 3" in system and "This is the last round" in system
    assert "Round 2: you accepted; the specification then failed a later check" in text
    assert "empty_goal_set" not in text and "collision-free (teapot" not in text
    assert "applied: tighten_slack on pour.subgoal.trans[1]" in text
    # what it accepted up to the failure is not kept by rule (2)
    assert "Rule (2) covers only rounds after round 2" in text
    assert "upright row moderate" not in text and "tight upright row" not in text
    # round 1's edit on the carry names a row the re-emission replaced
    assert "carry the teapot.path.rot[0]" not in text
    assert "re-emitted since: carry the teapot (its rows are new)" in text
