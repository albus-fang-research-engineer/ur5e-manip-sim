"""Offline checks of score_authoring_ablation.py's labeling (no planning)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_authoring_ablation as sc  # noqa: E402


def test_gate_failures_name_the_stage_and_label_the_unused_grasp():
    ok = {"stages_reached": 2, "stages_total": 2}
    assert sc.gate_failure(ok) is None
    assert "no stage counts" in sc.gate_failure({})                 # not None == None
    assert "no stage counts" in sc.gate_failure({"stages_reached": 2})
    a0 = {"stages_reached": 0, "stages_total": 3,
          "compiled": [{"stage": "grasp the marker", "grounded": False}],
          "emissions": [{"name": "grasp the marker", "passive": None}]}
    assert "grasp stage (unused" in sc.gate_failure(a0)
    a4 = {"stages_reached": 1, "stages_total": 2,
          "compiled": [{"stage": "align", "grounded": True},
                       {"stage": "insert marker into cup", "grounded": False}],
          "emissions": []}
    assert sc.gate_failure(a4) == "gate failed on 'insert marker into cup' (1/2 stages)"
    last = {"stages_reached": 3, "stages_total": 3,          # the failed stage is counted
            "compiled": [{"stage": "align", "grounded": True},
                         {"stage": "insert", "grounded": False}], "emissions": []}
    assert sc.gate_failure(last) == "gate failed on 'insert' (3/3 stages)"
    named = {**a0, "emissions": []}              # by name alone, no structural evidence
    assert "unused" not in sc.gate_failure(named)


def test_plan_failures_are_categorized_from_the_planners_own_words():
    assert sc.plan_category("[plan-task] FAILED [insert] compile: insert marker into cup.path: "
                            "the entry attitude lies outside the path's roll bound by 4 deg"
                            ) == "start outside the next path"
    assert sc.plan_category("[plan-task] FAILED [insert] empty_goal_set: no collision-free, "
                            "contained goal configuration") == "no reachable goal"
    assert sc.plan_category("[transport] no feasible goal configs.") == "no reachable goal"
    assert sc.plan_category("Traceback ... KeyError") == "other planning failure"


def test_success_uses_each_tasks_existing_definition():
    clean = {"self": 0.0, "table": 0.0, "passive": 0.0, "other": 0.0}
    ev = {"goal_pass": True, "path_pass": True, "max_penetration_mm": clean}
    assert sc.success("insert_marker", ev) == (True, True)
    assert sc.success("insert_marker", {**ev, "max_penetration_mm": {**clean, "passive": 0.4}}
                      ) == (False, False)                          # any penetration fails
    # pour tea tolerates up to eval_pour.COLLISION_TOL_MM
    assert sc.success("pour_tea", {**ev, "max_penetration_mm": {**clean, "passive": 0.4}}
                      ) == (True, True)
    assert sc.success("pour_tea", {**ev, "path_pass": False}) == (True, False)
