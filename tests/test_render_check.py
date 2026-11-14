"""manip_sim.render_check: the OmniManip-style condition's prompt, parser and
label rule. Offline; the client runs on a scripted transport."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from manip_sim import render_check as rc
from manip_sim.vlm import Client, ParseRejection, VLMError


def _view(tmp_path, n_images=2):
    imgs = []
    for i in range(n_images):
        p = tmp_path / f"v{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        imgs.append(p)
    return rc.StageView(name="carry teapot above mug", active="teapot", passive="mug",
                        rows=("carry teapot above mug.subgoal.trans[0]: above "
                              "mug.opening_center, clearance small, slack moderate",),
                        images=tuple(imgs))


def test_prompt_carries_alg1_inputs_and_only_images(tmp_path):
    system, messages = rc.build_check_prompt("pour tea into the mug", _view(tmp_path))
    blocks = messages[0]["content"]
    text = blocks[0]["text"]
    assert "Task: pour tea into the mug" in text
    assert "Stage: carry teapot above mug" in text
    assert "clearance small" in text
    assert [b["type"] for b in blocks[1:]] == ["image", "image"]
    # label-only: nothing typed about failures reaches the checker
    for word in ("collision-free pose", "CompileError", "no feasible", "entry"):
        assert word not in text
    assert "interaction_value" not in system
    for label in rc.LABELS:
        assert f'"{label}"' in system


def test_parse_accepts_paper_schema_and_ignores_extra_keys():
    raw = json.dumps({"Think Step By Step": "looks fine", "Reasonable": "refine",
                      "interaction_value": 0})
    c = rc.parse_check(raw, "s")
    assert (c.stage, c.label, c.action) == ("s", "refine", "regenerate_near")


@pytest.mark.parametrize("doc", [
    {"Reasonable": "success"},
    {"Think Step By Step": "x", "Reasonable": "maybe"},
    {"Think Step By Step": "  ", "Reasonable": "failure"},
    ["success"],
])
def test_parse_rejects(doc):
    with pytest.raises(ParseRejection):
        rc.parse_check(json.dumps(doc), "s")


def test_check_stage_retries_through_the_client(tmp_path):
    replies = iter(['{"Reasonable": "success"}',
                    '{"Think Step By Step": "spout over the rim", "Reasonable": "failure"}'])
    cl = Client(transport=lambda payload: next(replies))
    c = rc.check_stage(cl, "pour tea", _view(tmp_path))
    assert c.label == "failure" and c.action == "regenerate"
    log = cl.logs[-1]
    assert log.touchpoint == "render_check" and log.attempts == 2


def test_check_stage_gives_up_after_the_parse_budget(tmp_path):
    cl = Client(transport=lambda payload: "not json")
    with pytest.raises(VLMError):
        rc.check_stage(cl, "pour tea", _view(tmp_path))


def test_refine_enters_refinement_once():
    assert rc.action_for("refine", refined_before=False) == "regenerate_near"
    assert rc.action_for("refine", refined_before=True) == "regenerate"
    assert rc.action_for("failure", refined_before=False) == "regenerate"
    assert rc.action_for("success", refined_before=True) == "accept"


def test_first_rejected_follows_stage_order():
    cs = [rc.Check("a", "success", "."), rc.Check("b", "refine", "."),
          rc.Check("c", "failure", ".")]
    assert rc.first_rejected(cs).stage == "b"
    assert rc.first_rejected(cs[:1]) is None
