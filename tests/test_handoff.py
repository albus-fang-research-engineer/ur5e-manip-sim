"""task_plan's handoff check and consistency score on a synthetic two-stage case:
stage 1's goal allows +-15 deg of tilt; stage 2's path allows a narrower band."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_code_authoring import OPENING, POSES, SYMBOLS, TIP  # noqa: E402
from manip_sim import code_authoring as ca  # noqa: E402
from manip_sim.task_plan import (MoverStage, chained, handoff_consistency,  # noqa: E402
                                 handoff_rejection)
from manip_sim.vlm import StageSpec  # noqa: E402

FREE_T = [[None, None]] * 3


def _stage(i, name, path_rot, sub_rot, sub_z=(0.03, 0.08)):
    em = ca.CodeEmission(stage=i, name=name, active="teapot", passive="mug",
                         medium="bounds-json", goal="g", source=json.dumps({
                             "goal": "g", "path": FREE_T + path_rot,
                             "subgoal": [[-0.02, 0.02], [-0.02, 0.02], list(sub_z)] + sub_rot}))
    spec = StageSpec(index=i, name=name, active="teapot", passive="mug",
                     parts={"teapot": ("spout",), "mug": ("rim",)})
    return MoverStage(role=f"r{i}", stage=spec, emission=em, synthesized=False)


WIDE = [[-15, 15], [-15, 15], [-180, 180]]


def _goal_region(s1):
    return ca.compile_any(s1.emission, SYMBOLS, POSES, w_point=OPENING, e_point=TIP).subgoal


def test_a_narrower_next_path_rejects_part_of_the_goal_region():
    s1 = _stage(1, "carry", [[-180, 180]] * 3, WIDE)
    s2 = _stage(2, "lower", [[-5, 5], [-5, 5], [-180, 180]], WIDE, sub_z=(0.01, 0.02))
    sub = _goal_region(s1)
    hc = handoff_consistency(ca.compile_any, sub, s2, SYMBOLS, {"mug": POSES["mug"]},
                             OPENING, TIP, n=40, seed=0)
    assert 0 < hc["rejected"] < hc["tested"] == 40 and hc["to"] == "r2"
    # the check is the compile itself: a level start passes, a 12 deg tilt does not
    level = sub.T0_w @ sub.Tw_e
    assert handoff_rejection(ca.compile_any, s2, SYMBOLS, {"mug": POSES["mug"], "teapot": level},
                             OPENING, TIP) is None
    from manip_sim.tsr import make_pose
    from scipy.spatial.transform import Rotation
    tilt = sub.T0_w @ make_pose([0, 0, 0.05], Rotation.from_euler("x", 12, degrees=True)
                                .as_matrix()) @ sub.Tw_e
    assert "outside" in handoff_rejection(ca.compile_any, s2, SYMBOLS,
                                          {"mug": POSES["mug"], "teapot": tilt}, OPENING, TIP)


def test_a_next_path_as_wide_as_the_goal_rejects_nothing():
    s1 = _stage(1, "carry", [[-180, 180]] * 3, WIDE)
    s2 = _stage(2, "lower", [[-20, 20], [-20, 20], [-180, 180]], WIDE, sub_z=(0.01, 0.02))
    hc = handoff_consistency(ca.compile_any, _goal_region(s1), s2, SYMBOLS,
                             {"mug": POSES["mug"]}, OPENING, TIP, n=40, seed=0)
    assert hc["rejected"] == 0


def test_only_a_same_mover_unsynthesized_next_stage_is_chained():
    s1, s2 = _stage(1, "a", WIDE, WIDE), _stage(2, "b", WIDE, WIDE)
    assert chained([s1, s2], 0) is s2 and chained([s1, s2], 1) is None
    synth = MoverStage(role="r2", stage=s2.stage, emission=s2.emission, synthesized=True)
    assert chained([s1, synth], 0) is None
    other = MoverStage(role="r2", stage=StageSpec(index=2, name="b", active="mug",
                                                  passive="teapot", parts={}),
                       emission=s2.emission, synthesized=False)
    assert chained([s1, other], 0) is None
