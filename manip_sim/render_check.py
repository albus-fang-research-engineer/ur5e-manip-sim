"""OmniManip-style render-and-check for the repair study (docs/deferred.md
D9): the "render_check" condition, a port of OmniManip's self-correction
via Resampling, Rendering and Checking (RRC; Pan et al., CVPR 2025, Sec. 3.3,
Algorithm 1, supplement A.2-A.3) onto Langfold's pipeline. Pure library:
the prompt, the parser, and the label -> action rule. The loop driver
renders, calls the model and regenerates stages.

What the paper gives, and what this port does with it:

    paper                                          port
    ---------------------------------------------  -------------------------------
    Check: state <- VLM(T, S_i, I_i, C_i^(k),      inputs: task, stage, the stage's
      refine)  (Alg. 1)                            images (same pixels as the
                                                   critic, no captions), the stage's
                                                   rows as text
    "returns one of three outcomes: success,       the same three labels
      failure, or refinement" (Sec. 3.3)
    "If success, the constraint is accepted ...    success: next stage
      If failure, the next constraint is
      evaluated" (Sec. 3.3)                        failure: call #3 writes the stage
                                                   again from its own prompt (a
                                                   fresh candidate)
    "fine-grained resampling around the            refine: call #3 writes the stage
      predicted interaction direction" (Sec. 3.3)  again with its previous answer
                                                   shown and no reason given
    output schema, supp. Figs. 12-14:              {"Think Step By Step", "Reasonable"};
      {"Think Step By Step", "Reasonable",         interaction_value dropped: our
       "interaction_value"}                        specification has no distance
                                                   slot and the model never writes
                                                   numbers
    "a small set of interaction examples as        the supplement's Fig. 14 (success)
      prompts" (Sec. 4.1)                          and Fig. 12 (refine) outputs as
                                                   examples, interaction_value removed
    grasp "without using RRC" (supp. A.3)          context stages are not checked
    prompt text: "available for download here"    unavailable (the links 404, and
      (supp. A.2)                                  were already 404 when archived on
                                                   2025-07-17): REBUILT. Lines marked
                                                   [paper] restate the paper; lines
                                                   marked [ours] are ours, several
                                                   adapted from the CoRAL
                                                   reimplementation's rrc.txt

DRAFT PROMPT TEXT: for review before any paid call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .vlm import (_JSON_ONLY, ParseRejection, _enum, _load_json, _need, _text,
                  image_block)

LABELS = ("success", "refine", "failure")
ACTIONS = {"success": "accept", "refine": "regenerate_near", "failure": "regenerate"}

# The paper's own example outputs (supp. Figs. 14 and 12), interaction_value removed.
_EXAMPLES = (
    '{"Think Step By Step": "The image shows a cup positioned above a dish. The red '
    'arrow indicates a downward direction, suggesting the cup will be placed onto the '
    'dish. The base center of the cup aligns with the center of the dish, which is '
    'appropriate for the task of placing the cup on the dish. The interaction seems '
    'reasonable as the cup is positioned correctly to be placed on the dish.", '
    '"Reasonable": "success"}',
    '{"Think Step By Step": "The image shows a laptop with its lid partially open. The '
    'red arrow is pointing horizontally towards the top edge of the laptop lid, '
    'indicating the direction to push the lid closed. The interaction part is the '
    'finger of the gripper, which should push the top edge of the laptop lid. The '
    'arrow is correctly aligned with the task of closing the lid, as it points '
    'directly towards the edge that needs to be pushed. However, the arrow is not '
    'perpendicular to the surface of the lid, which would be more effective for '
    'pushing it closed.", "Reasonable": "refine"}',
)

# Follow-up turn for "refine" (call #3 sees its previous answer, then this).
# NEW PROMPT TEXT, flagged: carries the label only, never a reason.
REFINE_TURN = ("A visual check of this stage returned: refine (roughly right, "
               "somewhat off). Write the stage again, close to your previous "
               "answer. Re-emit the JSON object only.")


@dataclass(frozen=True)
class StageView:
    """What the checker sees of one stage."""
    name: str
    active: str
    passive: str | None
    rows: tuple[str, ...]                 # "<pointer>: <row text>", path then subgoal
    images: tuple[Path, ...]              # the critic's images of this stage, no captions


@dataclass(frozen=True)
class Check:
    stage: str
    label: str
    reasoning: str

    @property
    def action(self) -> str:
        return ACTIONS[self.label]


def build_check_prompt(task: str, view: StageView) -> tuple[str, list]:
    passive = view.passive or "scene"
    system = "\n".join([
        # [ours] role line; the paper names the check, not its wording
        "You are checking a planned robot interaction before it is executed.",
        # [paper] supp. A.2: "Takes rendered images of the interaction results as
        # inputs. Determines whether the interaction can be successful or if
        # refinement is needed"
        "You receive rendered images of the planned interaction and decide "
        "whether it can be successful or needs refinement.",
        "",
        # [ours] how our renders differ from the paper's single target pose + arrow
        f"The images show the {view.active} rendered where the plan allows it to "
        f"end this stage, and how it may move on the way, relative to the "
        f"{passive}. Judge the rendered {view.active} itself: its position, its "
        "orientation, and which of its parts faces the "
        f"{passive}.",
        "",
        # [paper] the three outcomes (Sec. 3.3); definitions [ours], after CoRAL's rrc.txt
        "Answer with one of three outcomes:",
        "- \"success\": the planned interaction is right for the task and "
        "feasible without collision.",
        "- \"refine\": roughly right, but somewhat off; a nearby version of this "
        "stage would be better. The stage will be written again close to this one.",
        "- \"failure\": wrong for the task, and no nearby version would fix it. "
        "The stage will be written again from scratch.",
        "",
        # [paper] example outputs, supp. Figs. 14 and 12 (interaction_value removed)
        "Examples of answers for other tasks:",
        *_EXAMPLES,
        "",
        # [paper] output schema, supp. Figs. 12-14, minus interaction_value
        "Output schema: {\"Think Step By Step\": str, \"Reasonable\": "
        "\"success\"|\"refine\"|\"failure\"}. " + _JSON_ONLY,
    ])
    # [paper] Alg. 1 inputs: task T, stage S_i, image I_i, constraint C_i^(k)
    text = "\n".join([
        f"Task: {task}",
        f"Stage: {view.name}, with the {view.active} as the active object"
        + (f" and the {view.passive} as the passive object." if view.passive else "."),
        "Planned constraint:",
        *(f"  {r}" for r in view.rows),
    ])
    content = [_text(text)] + [image_block(p) for p in view.images]
    return system, [{"role": "user", "content": content}]


def parse_check(raw: str, stage: str) -> Check:
    doc = _load_json(raw)
    if not isinstance(doc, dict):
        raise ParseRejection("the check must be a JSON object")
    reasoning = _need(doc, "Think Step By Step", str, "check")
    label = _enum(_need(doc, "Reasonable", str, "check"), LABELS, "check.Reasonable")
    if not reasoning.strip():
        raise ParseRejection("check.\"Think Step By Step\" must be non-empty")
    return Check(stage, label, reasoning)


def check_stage(client, task: str, view: StageView) -> Check:
    """One check call through the client's parse-retry loop (touchpoint
    "render_check", logged like every other call)."""
    system, messages = build_check_prompt(task, view)
    return client._ask("render_check", system, messages,
                       lambda raw: parse_check(raw, view.name))


def action_for(label: str, refined_before: bool) -> str:
    """What a label triggers for one stage. Alg. 1 enters the refinement
    phase once ("if state is Refine and refine is False"); after that a
    refine label no longer resamples near the candidate, it moves on like a
    failure. The CoRAL reimplementation (omnimanip/rrc.py, plan) does the
    same."""
    if label == "refine" and refined_before:
        return ACTIONS["failure"]
    return ACTIONS[label]


def first_rejected(checks: list[Check]) -> Check | None:
    """Alg. 1 runs stage by stage: the first stage that is not a success is
    the one regenerated this round; later stages are checked again after
    it, since they start where it ends."""
    return next((c for c in checks if c.label != "success"), None)
