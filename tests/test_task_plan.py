"""manip_sim.task_plan (the generic planner's blocks) and the plan_task /
run_pipeline wiring, without MuJoCo scenes: synthetic geometry and
duck-typed kinematics."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manip_sim.grasping import handle_grasp_tsr, propose_handle_grasps
from manip_sim.frames import Frame
from manip_sim.task_plan import (GRASP_ELEVATION_RAD, SYNTH_START_CLEARANCE_M,
                                 FunnelReport, TaskPlanError, decomposition,
                                 goal_funnel, grasp_part, lowest_end,
                                 mover_stages, probe_fractions,
                                 rotation_angle_deg, sample_goals,
                                 synthesized_start, topdown_approach_h,
                                 walk_goals)
from manip_sim.tsr import make_pose

ROOT = Path(__file__).resolve().parents[1]


def _rod(n=200, half=0.06):
    y = np.linspace(-half, half, n)
    return np.column_stack([np.zeros(n), y, np.zeros(n)])


def _cup_vertices():
    th = np.linspace(0, 2 * np.pi, 64)
    rim = np.column_stack([0.042 * np.cos(th), 0.042 * np.sin(th), np.full(64, 0.0375)])
    base = np.column_stack([0.035 * np.cos(th), 0.035 * np.sin(th), np.full(64, -0.0375)])
    return np.vstack([rim, base])


# ------------------------------------------------------- synthesized start

def test_synthesized_start_clears_the_passive_by_the_margin():
    T_cup = make_pose([0.0, 0.25, 0.84], np.eye(3))
    R_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # rod along world z
    T_nom = make_pose([0.0, 0.25, 0.85], R_up)                   # deep inside the cup
    T, shift = synthesized_start(T_nom, _rod(), T_cup, _cup_vertices(), [0, 0, 1])
    low = (T[:3, :3] @ _rod().T).T[:, 2].min() + T[2, 3]
    high = 0.84 + 0.0375
    assert shift > 0
    assert low == pytest.approx(high + SYNTH_START_CLEARANCE_M, abs=1e-9)
    assert np.allclose(T[:3, :3], R_up) and np.allclose(T[:2, 3], T_nom[:2, 3])


def test_synthesized_start_never_moves_down():
    T_cup = make_pose([0.0, 0.0, 0.84], np.eye(3))
    T_nom = make_pose([0.0, 0.0, 1.20], np.eye(3))
    T, shift = synthesized_start(T_nom, _rod(), T_cup, _cup_vertices(), [0, 0, 1])
    assert shift == 0.0 and np.allclose(T, T_nom)


def test_synthesized_start_follows_the_passive_up_not_world_z():
    R_tilt = make_pose([0, 0, 0], np.eye(3))
    c, s = np.cos(0.3), np.sin(0.3)
    R_tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    T_cup = make_pose([0, 0, 0], R_tilt)
    T, shift = synthesized_start(make_pose([0, 0, 0], np.eye(3)), _rod(), T_cup,
                                 _cup_vertices(), [0, 0, 1])
    d = T[:3, 3] / np.linalg.norm(T[:3, 3])
    assert np.allclose(d, R_tilt[:, 2])


# ------------------------------------------------------------------ grasp

def _frames_doc(prim="line"):
    return {"points": {"body_center": {"xyz": [0, 0, 0]}},
            "axes": {"body_axis": {"xyz": [0, -1, 0]}},
            "quantities": {"body_length": {"value": 0.074}},
            "provenance": {"parts": {"body": {"primitive": prim}}}}


def test_grasp_part_takes_the_slide_from_the_fitted_length():
    gp = grasp_part(_frames_doc(), "body")
    assert gp.primitive == "line" and gp.half_length == pytest.approx(0.037)
    assert np.allclose(gp.axis, [0, -1, 0])


@pytest.mark.parametrize("doc, part", [(_frames_doc("blob"), "body"),
                                       (_frames_doc(), None),
                                       (_frames_doc(), "cap")])
def test_grasp_part_refuses_parts_without_an_axis(doc, part):
    with pytest.raises(TaskPlanError) as e:
        grasp_part(doc, part)
    assert e.value.kind == "grasp_part"


def test_topdown_grasp_points_down_and_closes_across_the_bar():
    gp = grasp_part(_frames_doc(), "body")
    T0 = make_pose([0.0, -0.25, 0.86], np.eye(3))
    fr = Frame("marker.body", gp.center, gp.axis, np.array([0.0, 0.0, 1.0]))
    T0_w = T0 @ fr.T()
    tsr = handle_grasp_tsr(T0, fr, topdown_approach_h(T0_w),
                           elevation=GRASP_ELEVATION_RAD, slide=gp.half_length)
    T_nom = tsr.zero()
    assert T_nom[:3, 2] @ np.array([0, 0, -1.0]) > 0.999        # approach straight down
    assert abs(T_nom[:3, 0] @ (T0[:3, :3] @ gp.axis)) < 1e-6    # closing across the bar


def test_proposals_slide_over_the_part_length_not_the_point():
    gp = grasp_part(_frames_doc(), "body")
    T0 = make_pose([0.0, -0.25, 0.86], np.eye(3))
    fr = Frame("marker.body", gp.center, gp.axis, np.array([0.0, 0.0, 1.0]))
    tsr = handle_grasp_tsr(T0, fr, topdown_approach_h(T0 @ fr.T()),
                           elevation=GRASP_ELEVATION_RAD, slide=gp.half_length)
    props = propose_handle_grasps(tsr, np.random.default_rng(0), n=200,
                                  slide_span=gp.half_length)
    along = np.array([(p.T0_ee[:3, 3] - T0[:3, 3]) @ (T0[:3, :3] @ gp.axis) for p in props])
    assert along.min() < -0.8 * gp.half_length and along.max() > 0.8 * gp.half_length


def test_vertical_grasp_axis_is_a_typed_failure():
    with pytest.raises(TaskPlanError):
        topdown_approach_h(np.eye(4))           # w.z = world z


# ---------------------------------------------------------------- funnels

class FakeIK:
    def __init__(self, ok=lambda T: True):
        self.ok = ok

    def solve_multiseed(self, T, seeds, **kw):
        return (T[:3, 3].copy(), True) if self.ok(T) else (None, False)


class FakeKin:
    def __init__(self, collide=lambda q: False):
        self.collide = collide

    def in_collision(self, q):
        return self.collide(q)

    def fk(self, q):
        return make_pose(q, np.eye(3))


class Box:
    """TSR-like containment on x only."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def contains(self, T, tol=0.0):
        return self.lo - tol <= T[0, 3] <= self.hi + tol


def test_goal_funnel_counts_and_offenders():
    att = SimpleNamespace(T_ee_body=np.eye(4), body_pose=lambda T: T)
    samples = [make_pose([x, 0, 0], np.eye(3)) for x in (0.0, 0.1, 0.2, 0.3)]
    rep = FunnelReport()
    goals = goal_funnel(samples, FakeIK(lambda T: T[0, 3] < 0.25),
                        FakeKin(lambda q: q[0] == 0.1), att, [], [Box(-1, 0.15)],
                        lambda kin: "gripper0 <-> cup", rep)
    assert (rep.n_ik, rep.n_collision_free, rep.n_contained) == (3, 2, 1)
    assert rep.offenders == {"gripper0 <-> cup": 1} and len(goals) == 1


def test_sample_goals_does_not_escalate_a_thin_intersection():
    calls = []

    def sampler(sub, cons, n, rng):
        calls.append(n)
        return SimpleNamespace(accepted=[], acceptance_rate=0.0)
    att = SimpleNamespace(T_ee_body=np.eye(4), body_pose=lambda T: T)
    goals, rep = sample_goals(None, None, 120, 480, 0.02, None, FakeIK(), FakeKin(),
                              att, [], lambda k: "?", np.zeros(3), sampler=sampler)
    assert goals == [] and calls == [120] and rep.n_rounds == 1


def test_probe_fractions_score_is_the_product_over_stages():
    T_g = np.eye(4)
    s1 = [make_pose([x, 0, 0], np.eye(3)) for x in (0.0, 0.1)]
    s2 = [make_pose([x, 0, 0], np.eye(3)) for x in (0.0, 0.5, 0.6, 0.7)]
    fr, sc = probe_fractions(FakeIK(lambda T: T[0, 3] < 0.55), FakeKin(), T_g, T_g,
                             [s1, s2], [])
    assert fr == [1.0, 0.5] and sc == pytest.approx(0.5)


def test_walk_goals_failure_carries_the_reason_histogram():
    def plan(kin, att, tsrs, q0, g, timeout):
        return SimpleNamespace(ok=False, reason="timeout")
    with pytest.raises(TaskPlanError) as e:
        walk_goals(plan, None, None, [], None, [1, 2, 3], 2, 1.0, "insert")
    assert e.value.kind == "planning" and e.value.detail["reasons"] == {"timeout": 2}


# ------------------------------------------------------------------ stages

def _stage(name, active, passive):
    return SimpleNamespace(name=name, active=active, passive=passive)


def test_mover_stages_skip_the_grasp_and_flag_synthesized():
    st = [(_stage("grasp", "marker", None), "grasp"),
          (_stage("insert", "marker", "cup"), "insert")]
    ms = mover_stages(st, {"grasp": "g", "insert": "i"}, frozenset({"insert"}))
    assert [(m.role, m.synthesized, m.emission) for m in ms] == [("insert", True, "i")]


def test_mover_stage_without_emission_is_typed():
    st = [(_stage("carry", "marker", "cup"), "transport_active")]
    with pytest.raises(TaskPlanError) as e:
        mover_stages(st, {}, frozenset())
    assert e.value.kind == "compile"


def test_lowest_end_names_the_nearest_symbol():
    R_down = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)    # body +y -> world +z
    rec = lowest_end(make_pose([0, 0, 1], R_down), _rod(),
                     {"tip_center": np.array([0, 0.0486, 0]),
                      "body_tip": np.array([0, -0.0365, 0])})
    assert rec["nearest_symbol"] == "body_tip"
    assert rec["lowest_point_body"][1] == pytest.approx(-0.06)


def test_decomposition_line_names_where_the_rotation_happened():
    line = decomposition([{"role": "insert", "reorient_free_reach_deg": 90.0,
                           "reorient_in_stage_deg": 2.0},
                          {"role": "transport_active", "reorient_in_stage_deg": 88.0}])
    assert "90 deg in the free-space reach" in line and "88 deg inside the stage" in line
    assert rotation_angle_deg(np.eye(3), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])) \
        == pytest.approx(90.0)


# ------------------------------------------------------ run_pipeline wiring

def test_plan_task_imports():
    import scripts.plan_task as pt  # noqa: F401
    assert callable(pt.main)


def _run_pipeline(monkeypatch, tmp_path, extra):
    import scripts.run_pipeline as rp
    calls = []

    class Done:
        returncode = 0
    monkeypatch.setattr(rp.subprocess, "run", lambda args, env=None: calls.append(args) or Done())
    monkeypatch.setattr(rp, "ISSUED", [])
    monkeypatch.setattr(rp, "RUN", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--scene", str(ROOT / "scenes/insert_marker.json"),
        "--task-spec", str(ROOT / "tasks/insert_marker.json"), *extra])
    rp.main()
    return calls


def test_planner_task_runs_plan_task_and_is_recorded(monkeypatch, tmp_path):
    calls = _run_pipeline(monkeypatch, tmp_path,
                          ["--from", "path", "--until", "path", "--planner", "task"])
    (cmd,) = calls
    assert any(a.endswith("plan_task.py") for a in cmd)
    assert {"--stage-plan", "--selections", "--emissions"} <= set(cmd)
    (run_json,) = tmp_path.glob("outputs/runs/*/run.json")
    doc = json.loads(run_json.read_text())
    assert doc["planner"] == "task"
    assert doc["artifacts"]["plan"].endswith("insert_marker_task_vlm.npz")


def test_planner_default_keeps_plan_pour_tea(monkeypatch, tmp_path):
    calls = _run_pipeline(monkeypatch, tmp_path, ["--from", "path", "--until", "path"])
    assert any(a.endswith("plan_pour_tea.py") for a in calls[0])


def test_planner_task_refuses_video_and_exec(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_pipeline(monkeypatch, tmp_path, ["--from", "path", "--planner", "task"])


def test_topdown_grasp_on_a_slightly_tilted_bar_still_closes_across_it():
    c, s = np.cos(0.17), np.sin(0.17)
    axis = np.array([0.0, c, s])                       # bar 10 deg off horizontal
    T0 = make_pose([0.0, -0.25, 0.86], np.eye(3))
    fr = Frame("marker.body", np.zeros(3), axis, np.array([0.0, 0.0, 1.0]))
    tsr = handle_grasp_tsr(T0, fr, topdown_approach_h(T0 @ fr.T()),
                           elevation=GRASP_ELEVATION_RAD, slide=0.03)
    T_nom = tsr.zero()
    assert T_nom[:3, 2] @ np.array([0, 0, -1.0]) > 0.999
    assert abs(T_nom[:3, 0] @ axis) < 1e-6
