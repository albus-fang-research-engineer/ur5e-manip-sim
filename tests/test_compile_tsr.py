"""Offline tests for the grounding compiler (manip_sim/compile_tsr.py):
enum-table grounding, the v1 rot/trans rule table, its typed
CompileErrors, and — the check that matters — near-equivalence of the
compiled transport/pour pairs against the hand-authored ground-truth arm
in pour_stages.py where the enum anchors coincide. Real frames.json
sidecars, no network, no simulator."""

import json

import numpy as np
import pytest

from manip_sim.compile_tsr import (ALIGN_TOL_RAD, CENTERED_TOL_M,
                                   CLEARANCE_M, INSIDE_TOL_M, ROT_TOL_RAD,
                                   SLACK_BAND_M, CompileError, compile_stage)
from manip_sim.frames import load_symbols
from manip_sim.pour_stages import pour_pair, transport_pair
from manip_sim.tsr import make_pose
from manip_sim.vlm import Vocabulary, parse_emission

SYMBOLS = {"teapot": load_symbols("assets/objects/teapot"),
           "mug": load_symbols("assets/objects/mug")}
VOCAB = Vocabulary.from_asset_dirs({"teapot": "assets/objects/teapot",
                                    "mug": "assets/objects/mug"})
# canonical scene attitude: both objects upright; positions arbitrary
POSES = {"teapot": make_pose((0.3, 0.1, 0.05)),
         "mug": make_pose((0.55, -0.2, 0.02))}


def emission(doc: dict):
    return parse_emission(json.dumps(doc), VOCAB)


TRANSPORT = {
    "stage": 2, "name": "transport", "active": "teapot", "passive": "mug",
    "w_origin": "mug.opening_center", "w_axis": "mug.up_axis",
    "path_tsr": {"rot": [
        {"axis": "teapot.up_axis", "relation": "parallel",
         "reference": "world.z", "tol": "moderate"},
        {"relation": "free", "row": "yaw"}],
        "trans": "free"},
    "subgoal_tsr": {"rot": [
        {"axis": "teapot.up_axis", "relation": "parallel",
         "reference": "world.z", "tol": "moderate"},
        {"relation": "free", "row": "yaw"}],
        "trans": [
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "medium", "slack": "moderate"},
        {"term": "centered", "anchor": "mug.opening_center",
         "tol": "moderate"}]},
    "verify": "spout tip hovers over the mug opening",
}

POUR = {
    "stage": 3, "name": "pour", "active": "teapot", "passive": "mug",
    "w_origin": "teapot.spout_tip", "w_axis": "teapot.tilt_axis",
    "path_tsr": {"rot": [{"relation": "free", "row": "yaw"}],
                 "trans": [{"term": "inside",
                            "anchor": "teapot.spout_tip",
                            "slack": "snug"}]},
    "subgoal_tsr": {"rot": [
        {"axis": "teapot.up_axis", "relation": "perpendicular",
         "reference": "world.z", "tol": "tight"}],
        "trans": [{"term": "inside", "anchor": "teapot.spout_tip",
                   "slack": "snug"}]},
    "verify": "teapot tilted past horizontal about the spout tip",
}


# ------------------------------------------------ transport equivalence

def test_transport_subgoal_matches_hand_authored():
    cs = compile_stage(emission(TRANSPORT), SYMBOLS, POSES,
                       e_feature=SYMBOLS["teapot"].frame("spout_tip",
                                                         "pour_axis"))
    hand = transport_pair(
        POSES["mug"], SYMBOLS["mug"].frame("opening_center", "up_axis"),
        SYMBOLS["teapot"].frame("spout_tip", "pour_axis"),
        POSES["teapot"][:3, 3]).subgoal
    np.testing.assert_allclose(cs.subgoal.T0_w, hand.T0_w, atol=1e-12)
    np.testing.assert_allclose(cs.subgoal.Tw_e, hand.Tw_e, atol=1e-12)
    # rows: x,y +-rim_margin; z (0.03, 0.08); roll,pitch +-15deg; yaw free
    np.testing.assert_allclose(cs.subgoal.Bw[:5], hand.Bw[:5], atol=1e-9)
    lo, hi = cs.subgoal.Bw[5]
    assert lo <= -np.pi + 1e-9 and hi >= np.pi - 1e-9


def test_transport_path_rows():
    cs = compile_stage(emission(TRANSPORT), SYMBOLS, POSES)
    Bw = cs.path.Bw
    assert np.isinf(Bw[0]).any() and np.isinf(Bw[1]).any()  # x,y free
    assert np.isinf(Bw[2]).any()                            # z undeclared
    t = ROT_TOL_RAD["moderate"]
    np.testing.assert_allclose(Bw[3], (-t, t))
    np.testing.assert_allclose(Bw[4], (-t, t))


# ------------------------------------------------------ pour compilation

SPOUT = SYMBOLS["teapot"].frame("spout_tip", "pour_axis")


def test_pour_subgoal_yaw_positive_branch_and_pivot():
    cs = compile_stage(emission(POUR), SYMBOLS, POSES, e_feature=SPOUT)
    Bw = cs.subgoal.Bw
    tol = INSIDE_TOL_M["snug"]
    for r in range(3):                       # pivot pinned +-5 mm
        np.testing.assert_allclose(Bw[r], (-tol, tol), atol=1e-9)
    t = ROT_TOL_RAD["tight"]
    np.testing.assert_allclose(Bw[5], (np.pi / 2 - t, np.pi / 2 + t),
                               atol=ALIGN_TOL_RAD)   # positive branch
    tight = ROT_TOL_RAD["tight"]             # off-axis rows default tight
    np.testing.assert_allclose(Bw[3], (-tight, tight))
    np.testing.assert_allclose(Bw[4], (-tight, tight))
    # same pivot geometry as the hand-authored pair
    hand = pour_pair(POSES["teapot"],
                     SYMBOLS["teapot"].frame("spout_tip", "tilt_axis",
                                             secondary="pour_axis"))
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 3],
                               hand.subgoal.T0_w[:3, 3], atol=1e-9)


def test_pour_path_tilt_row_free():
    cs = compile_stage(emission(POUR), SYMBOLS, POSES, e_feature=SPOUT)
    lo, hi = cs.path.Bw[5]
    assert lo <= -np.pi + 1e-9 and hi >= np.pi - 1e-9


# ------------------------------------------------------------ expr terms

def test_expr_rows_evaluate_quantities():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": "free", "trans": [
        {"term": "expr", "row": "x", "lo": "0 - mug.rim_radius",
         "hi": "mug.rim_radius"}]}
    cs = compile_stage(emission(doc), SYMBOLS, POSES)
    r = SYMBOLS["mug"].quantities["rim_radius"]
    np.testing.assert_allclose(cs.subgoal.Bw[0], (-r, r), atol=1e-9)


# --------------------------------------------------------- typed errors

def _expect(doc, poses=POSES, **kw):
    with pytest.raises(CompileError) as e:
        compile_stage(emission(doc), SYMBOLS, poses, **kw)
    return e.value


def test_anchor_off_w_object_rejected():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": "free", "trans": [
        {"term": "centered", "anchor": "teapot.spout_tip",
         "tol": "moderate"}]}
    err = _expect(doc)
    assert "static in w" in err.reason and "trans[0]" in err.slot


def test_reference_on_active_object_rejected():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": [
        {"axis": "teapot.up_axis", "relation": "parallel",
         "reference": "teapot.pour_axis", "tol": "tight"}],
        "trans": "free"}
    assert "degenerate" in _expect(doc).reason


def test_empty_intersection_rejected():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": "free", "trans": [
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "medium", "slack": "moderate"},
        {"term": "inside", "anchor": "mug.opening_center",
         "slack": "snug"}]}          # z bands [0.03,0.08] vs [-5,5] mm
    assert "intersection empty" in _expect(doc).reason


def test_mixed_alignment_rejected():
    doc = dict(POUR)
    doc["subgoal_tsr"] = {"rot": [
        {"axis": "teapot.tilt_axis", "relation": "parallel",
         "reference": "world.z", "tol": "tight"}],   # plane vs z in w
        "trans": "free"}
    assert "mixed alignment" in _expect(doc, e_feature=SPOUT).reason


def test_relation_row_without_feature_on_misaligned_w_rejected():
    # the tilt-frame w differs from the body attitude; relation rows
    # with Tw_e = I would mis-center — the guard must fire
    assert "nominal displacement is not zero" in _expect(POUR).reason


def test_zero_displacement_reproduces_entry_pose():
    cs = compile_stage(emission(POUR), SYMBOLS, POSES, e_feature=SPOUT)
    np.testing.assert_allclose(cs.subgoal.T0_w @ cs.subgoal.Tw_e,
                               POSES["teapot"], atol=1e-10)


def test_cross_object_w_rejected():
    doc = dict(TRANSPORT)
    doc["w_axis"] = "teapot.up_axis"
    assert "cross-object" in _expect(doc).reason
