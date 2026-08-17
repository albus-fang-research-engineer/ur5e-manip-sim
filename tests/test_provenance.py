"""Unit tests for the emission-ablation arm stamp.

Pure numpy — runs without MuJoCo. The tests that matter are the refusals:
the whole point of the module is that a hand-authored run can never be
filed as a VLM run, so every path where the arm is ambiguous must raise
rather than pick a default.
"""

import numpy as np
import pytest

from manip_sim.provenance import (
    AUTO,
    LEGACY_PLAN,
    arm_of_run,
    metrics_path,
    plan_path,
    read_arm,
    read_selections,
    resolve_arm,
    resolve_plan_path,
    stamp,
    video_path,
)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    """All provenance paths are relative to the repo root."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_plan(arm, selections=None):
    p = plan_path(arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, path=np.zeros((3, 6)), **stamp(arm, selections))
    return p


def test_arm_of_run():
    assert arm_of_run(None) == "hand"
    assert arm_of_run("outputs/selections/pour_tea.json") == "vlm"


def test_paths_carry_the_arm_twice():
    for arm in ("hand", "vlm"):
        for p in (plan_path(arm), video_path("pour_tea_full", arm),
                  video_path("pour_tea_exec", arm),
                  metrics_path("pour_tea_exec", arm)):
            assert p.parent.name == arm        # folder
            assert p.stem.endswith(f"_{arm}")  # filename suffix


def test_the_two_arms_never_collide():
    assert plan_path("hand") != plan_path("vlm")
    assert video_path("pour_tea_full", "hand") != \
        video_path("pour_tea_full", "vlm")


def test_stamp_round_trips_without_pickle(cwd):
    write_plan("vlm", "outputs/selections/pour_tea.json")
    z = np.load(plan_path("vlm"))               # allow_pickle=False default
    assert read_arm(z) == "vlm"
    assert read_selections(z) == "outputs/selections/pour_tea.json"


def test_hand_arm_has_no_selections(cwd):
    write_plan("hand")
    assert read_selections(np.load(plan_path("hand"))) == ""


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        stamp("groundtruth")
    with pytest.raises(ValueError):
        plan_path("gt")


def test_auto_picks_the_only_plan_on_disk(cwd):
    write_plan("vlm")
    assert resolve_plan_path(None, AUTO) == plan_path("vlm")


def test_auto_refuses_when_both_arms_present(cwd):
    write_plan("hand")
    write_plan("vlm")
    with pytest.raises(SystemExit):
        resolve_plan_path(None, AUTO)
    # ... but an explicit arm disambiguates
    assert resolve_plan_path(None, "hand") == plan_path("hand")


def test_auto_falls_back_to_legacy_then_errors(cwd):
    with pytest.raises(SystemExit):
        resolve_plan_path(None, AUTO)
    LEGACY_PLAN.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LEGACY_PLAN, path=np.zeros((3, 6)))
    assert resolve_plan_path(None, AUTO) == LEGACY_PLAN


def test_explicit_plan_wins(cwd):
    assert resolve_plan_path("somewhere/else.npz", "vlm").name == "else.npz"


def test_stamp_beats_a_wrong_flag(cwd):
    p = write_plan("hand")
    z = np.load(p)
    assert resolve_arm(AUTO, z, p) == "hand"
    assert resolve_arm("hand", z, p) == "hand"
    with pytest.raises(SystemExit):        # the mislabeling this prevents
        resolve_arm("vlm", z, p)


def test_unstamped_plan_needs_an_explicit_arm(cwd):
    LEGACY_PLAN.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LEGACY_PLAN, path=np.zeros((3, 6)))
    z = np.load(LEGACY_PLAN)
    assert read_arm(z) is None
    with pytest.raises(SystemExit):
        resolve_arm(AUTO, z, LEGACY_PLAN)
    assert resolve_arm("vlm", z, LEGACY_PLAN) == "vlm"
