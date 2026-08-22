"""Offline tests for the grounding compiler (manip_sim/compile_tsr.py):
the canonical-frame rule for w, enum-table grounding, the v2 rot rule
table (goal attitude, fixed-DOF rows, path corridor), its typed
CompileErrors, and — the check that matters — equivalence of the
compiled transport pair against the hand-authored ground-truth arm in
pour_stages.py up to the yaw of T0_w, plus the pour's goal attitude
against pour_pair's. Real frames.json sidecars, no network, no
simulator."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from manip_sim.compile_tsr import (CENTERED_TOL_M,
                                   CLEARANCE_M, ROT_TOL_RAD, SIGMA_K,
                                   SLACK_BAND_M, CompileError, compile_stage)
from manip_sim.frames import Symbols, load_symbols
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
TIP = SYMBOLS["teapot"].points["spout_tip"]
OPENING = SYMBOLS["mug"].points["opening_center"]


def emission(doc: dict):
    return parse_emission(json.dumps(doc), VOCAB)


def _is_free(row):
    return row[0] <= -np.pi + 1e-9 and row[1] >= np.pi - 1e-9


def _u(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _tilted(T, deg, axis=(1.0, 0.0, 0.0)):
    T = T.copy()
    T[:3, :3] = R.from_rotvec(np.deg2rad(deg) * np.asarray(axis)).as_matrix() @ T[:3, :3]
    return T


UPRIGHT = {"axis": "teapot.up_axis", "relation": "parallel",
           "reference": "world.z", "tol": "moderate"}
TRANSPORT = {
    "stage": 2, "name": "transport", "active": "teapot", "passive": "mug",
    "path_tsr": {"rot": [UPRIGHT], "trans": "free"},
    "subgoal_tsr": {"rot": [UPRIGHT], "trans": [
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "medium", "slack": "moderate"},
        {"term": "centered", "anchor": "mug.opening_center",
         "tol": "moderate"}]},
    "verify": "spout tip hovers over the mug opening",
}

SPOUT_DOWN = {"axis": "teapot.front_axis", "relation": "antiparallel",
              "reference": "world.z", "tol": "tight"}
POUR = {
    "stage": 3, "name": "pour", "active": "teapot", "passive": "mug",
    "path_tsr": {"rot": [SPOUT_DOWN], "trans": [
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "small", "slack": "moderate"},
        {"term": "centered", "anchor": "mug.opening_center",
         "tol": "moderate"}]},
    "subgoal_tsr": {"rot": [SPOUT_DOWN], "trans": [
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "small", "slack": "snug"},
        {"term": "centered", "anchor": "mug.opening_center",
         "tol": "snug"}]},
    "verify": "spout points down into the mug opening",
}


def compile_pour(poses=POSES, doc=POUR):
    return compile_stage(emission(doc), SYMBOLS, poses, w_point=OPENING,
                         e_point=TIP)


# --------------------------------------------------------- canonical w

def test_w_is_passive_canonical_frame_at_selected_point():
    cs = compile_stage(emission(TRANSPORT), SYMBOLS, POSES, w_point=OPENING,
                       e_point=TIP)
    T0_w = cs.subgoal.T0_w
    np.testing.assert_allclose(T0_w[:3, 3], (POSES["mug"] @ np.append(OPENING, 1))[:3])
    np.testing.assert_allclose(T0_w[:3, 2], POSES["mug"][:3, :3] @ SYMBOLS["mug"].axes["up_axis"])
    # the mug has no front: x falls back to the TEAPOT's lateral at entry
    lat = _u(POSES["teapot"][:3, :3] @ SYMBOLS["teapot"].axes["lateral_axis"])
    np.testing.assert_allclose(T0_w[:3, 0], lat, atol=1e-9)
    assert any("teapot lateral" in n for n in cs.notes)
    assert np.allclose(cs.path.T0_w, T0_w)       # one w per stage


def test_w_uses_passive_front_when_present():
    # swap roles: the teapot (which has a front) is passive
    doc = {**TRANSPORT, "active": "mug", "passive": "teapot",
           "path_tsr": {"rot": "free", "trans": "free"},
           "subgoal_tsr": {"rot": [{"axis": "mug.up_axis", "relation": "parallel",
                                    "reference": "teapot.up_axis", "tol": "tight"}],
                           "trans": "free"}}
    cs = compile_stage(emission(doc), SYMBOLS, POSES, w_point=TIP)
    front = _u(POSES["teapot"][:3, :3] @ SYMBOLS["teapot"].axes["front_axis"])
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 0], front, atol=1e-9)
    assert any("teapot.front_axis" in n for n in cs.notes)


def test_w_falls_back_to_world_x_when_mover_has_no_front():
    tp = SYMBOLS["teapot"]
    nofront = Symbols("teapot", tp.points, {k: v for k, v in tp.axes.items()
                                             if k not in ("front_axis", "lateral_axis")})
    cs = compile_stage(emission(TRANSPORT), {**SYMBOLS, "teapot": nofront},
                       POSES, w_point=OPENING, e_point=TIP)
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 0], [1, 0, 0], atol=1e-9)
    assert any("world.x" in n for n in cs.notes)


# ------------------------------------------------ transport equivalence

def test_transport_subgoal_matches_hand_authored_up_to_w_yaw():
    cs = compile_stage(emission(TRANSPORT), SYMBOLS, POSES, w_point=OPENING,
                       e_point=TIP)
    hand = transport_pair(
        POSES["mug"], SYMBOLS["mug"].frame("opening_center", "up_axis"),
        SYMBOLS["teapot"].frame("spout_tip", "pour_axis"),
        POSES["teapot"][:3, 3]).subgoal
    # same origin and z; the two w differ by a yaw about z
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 3], hand.T0_w[:3, 3], atol=1e-12)
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 2], hand.T0_w[:3, 2], atol=1e-12)
    dR = hand.T0_w[:3, :3].T @ cs.subgoal.T0_w[:3, :3]
    rv = R.from_matrix(dR).as_rotvec()
    assert abs(abs(rv[2]) - np.linalg.norm(rv)) < 1e-9
    # the zero-displacement body pose is the same physical pose
    np.testing.assert_allclose(cs.subgoal.zero(), hand.zero(), atol=1e-12)
    # rows: x,y +-rim_margin; z (0.03, 0.08); roll,pitch +-15deg; yaw free
    np.testing.assert_allclose(cs.subgoal.Bw[:5], hand.Bw[:5], atol=1e-9)
    assert _is_free(cs.subgoal.Bw[5])


def test_transport_path_rows():
    cs = compile_stage(emission(TRANSPORT), SYMBOLS, POSES, w_point=OPENING)
    Bw = cs.path.Bw
    assert np.isinf(Bw[0]).any() and np.isinf(Bw[1]).any()  # x,y free
    assert np.isinf(Bw[2]).any()                            # z undeclared
    t = ROT_TOL_RAD["moderate"]
    np.testing.assert_allclose(Bw[3], (-t, t))
    np.testing.assert_allclose(Bw[4], (-t, t))
    assert _is_free(Bw[5])                                  # unmentioned DOF
    # no corridor: the entry already satisfies the row
    assert not any("corridor" in n for n in cs.notes)


# ------------------------------------------------------ pour compilation

def test_pour_goal_attitude_is_entry_tilted_about_lateral():
    cs = compile_pour()
    R_goal = cs.subgoal.zero()[:3, :3]
    # front points straight down at the goal, by the smallest rotation
    np.testing.assert_allclose(R_goal @ _u(SYMBOLS["teapot"].axes["front_axis"]),
                               [0, 0, -1], atol=1e-9)
    hand = pour_pair(POSES["teapot"],
                     SYMBOLS["teapot"].frame("spout_tip", "tilt_axis",
                                             secondary="pour_axis"),
                     tilt_target=np.pi / 2)
    np.testing.assert_allclose(R_goal, hand.subgoal.nominal()[:3, :3], atol=1e-9)
    # the pivot is the opening, not the frozen tip: w sits at the mug
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 3],
                               (POSES["mug"] @ np.append(OPENING, 1))[:3])


def test_pour_subgoal_rows_fix_tilt_and_free_heading():
    cs = compile_pour()
    Bw = cs.subgoal.Bw
    t = ROT_TOL_RAD["tight"]
    np.testing.assert_allclose(Bw[3], (-t, t))
    np.testing.assert_allclose(Bw[4], (-t, t))
    assert _is_free(Bw[5])
    ctol, clr, band = CENTERED_TOL_M["snug"], CLEARANCE_M["small"], SLACK_BAND_M["snug"]
    np.testing.assert_allclose(Bw[0], (-ctol, ctol))
    np.testing.assert_allclose(Bw[1], (-ctol, ctol))
    np.testing.assert_allclose(Bw[2], (clr, clr + band))
    # spinning the goal attitude about world z (heading) stays contained
    T_goal = cs.subgoal.nominal()
    for yaw in (-2.0, 0.7, 3.0):
        T = make_pose(T_goal[:3, 3], R.from_rotvec([0, 0, yaw]).as_matrix() @ T_goal[:3, :3])
        T[:3, 3] = T_goal[:3, 3] + (T_goal[:3, :3] - T[:3, :3]) @ TIP  # re-pin the tip
        assert cs.subgoal.contains(T, tol=1e-6)


def test_pour_path_corridor_on_roll_from_entry_to_goal():
    cs = compile_pour()
    t = ROT_TOL_RAD["tight"]
    # w.x = teapot lateral, so the tilt sweeps the ROLL row: entry at -90
    np.testing.assert_allclose(cs.path.Bw[3], (-np.pi / 2 - t, t), atol=1e-9)
    np.testing.assert_allclose(cs.path.Bw[4], (-t, t))
    assert _is_free(cs.path.Bw[5])
    assert any("corridor on roll" in n for n in cs.notes)
    # the entry attitude, with the tip carried over the opening, is on
    # the path and off the subgoal; the goal is on both
    o = cs.path.T0_w[:3, 3]
    mid_z = 0.5 * (cs.path.Bw[2, 0] + cs.path.Bw[2, 1])
    T_entry = make_pose(o + [0, 0, mid_z] - POSES["teapot"][:3, :3] @ TIP,
                        POSES["teapot"][:3, :3])
    d = cs.path.displacement(T_entry)
    assert d[3] == pytest.approx(-np.pi / 2, abs=1e-9)
    assert cs.path.contains(T_entry, tol=1e-9)
    assert not cs.subgoal.contains(T_entry, tol=1e-6)
    assert cs.path.contains(cs.subgoal.nominal(), tol=1e-9)
    # tilting the spout UP (beyond the settle allowance) leaves the path
    lat = _u(POSES["teapot"][:3, :3] @ SYMBOLS["teapot"].axes["lateral_axis"])
    T_up = _tilted(T_entry, -20.0, lat)
    T_up[:3, 3] = T_entry[:3, 3] + (T_entry[:3, :3] - T_up[:3, :3]) @ TIP
    assert not cs.path.contains(T_up, tol=1e-6)
    T_dn = _tilted(T_entry, +40.0, lat)
    T_dn[:3, 3] = T_entry[:3, 3] + (T_entry[:3, :3] - T_dn[:3, :3]) @ TIP
    assert cs.path.contains(T_dn, tol=1e-6)


def test_pour_goal_follows_entry_heading():
    # frozen at entry: a yawed entry tilts about ITS lateral
    yawed = {**POSES, "teapot": _tilted(POSES["teapot"], 50.0, (0, 0, 1))}
    cs = compile_pour(yawed)
    lat = _u(yawed["teapot"][:3, :3] @ SYMBOLS["teapot"].axes["lateral_axis"])
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 0], lat, atol=1e-9)
    dR = cs.subgoal.zero()[:3, :3] @ yawed["teapot"][:3, :3].T
    rv = R.from_matrix(dR).as_rotvec()
    assert np.linalg.norm(rv) == pytest.approx(np.pi / 2, abs=1e-9)
    np.testing.assert_allclose(rv / np.linalg.norm(rv), lat, atol=1e-9)


def test_pour_corridor_off_basis_frees_path_rotation():
    # give the mug a confident front that is NOT the teapot's lateral:
    # the tilt axis lands between w.x and w.y, the corridor is not a
    # single row, and translation carries the path
    mg = SYMBOLS["mug"]
    front = np.array([1.0, 0.0, 0.0])
    fronted = Symbols("mug", mg.points, {**mg.axes, "front_axis": front,
                                         "lateral_axis": np.cross([0, 0, 1.0], front)},
                      mg.quantities)
    cs = compile_pour(doc=POUR)  # baseline: corridor exists
    assert any("corridor" in n for n in cs.notes)
    cs = compile_stage(emission(POUR), {**SYMBOLS, "mug": fronted}, POSES,
                       w_point=OPENING, e_point=TIP)
    assert all(_is_free(cs.path.Bw[i]) for i in (3, 4, 5))
    assert any("translation carries the path" in n for n in cs.notes)
    t = ROT_TOL_RAD["tight"]                    # subgoal unaffected
    np.testing.assert_allclose(cs.subgoal.Bw[3], (-t, t))
    np.testing.assert_allclose(cs.subgoal.Bw[4], (-t, t))


def test_corridor_never_on_pitch():
    # a passive front aligned with the teapot's front puts the tilt axis
    # on w.y: pitch is the middle Euler angle, so no corridor
    mg = SYMBOLS["mug"]
    front = SYMBOLS["teapot"].axes["front_axis"]
    fronted = Symbols("mug", mg.points, {**mg.axes, "front_axis": front,
                                         "lateral_axis": np.cross([0, 0, 1.0], front)},
                      mg.quantities)
    cs = compile_stage(emission(POUR), {**SYMBOLS, "mug": fronted}, POSES,
                       w_point=OPENING, e_point=TIP)
    assert all(_is_free(cs.path.Bw[i]) for i in (3, 4, 5))


# ------------------------------------------------------- rule-table cases

def test_perpendicular_fixes_one_tilt():
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": [
        {"axis": "teapot.front_axis", "relation": "perpendicular",
         "reference": "world.z", "tol": "loose"}], "trans": "free"}}
    cs = compile_stage(emission(doc), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)
    # front is horizontal at entry: goal == entry; the tilt of front toward
    # z is rotation about w.x (the lateral) -> roll only
    t = ROT_TOL_RAD["loose"]
    np.testing.assert_allclose(cs.subgoal.Bw[3], (-t, t))
    assert _is_free(cs.subgoal.Bw[4]) and _is_free(cs.subgoal.Bw[5])
    np.testing.assert_allclose(cs.subgoal.zero()[:3, :3], POSES["teapot"][:3, :3],
                               atol=1e-12)


def test_two_rows_fully_determine_attitude():
    mg = SYMBOLS["mug"]
    fronted = Symbols("mug", mg.points, {**mg.axes, "front_axis": np.array([0, 1.0, 0]),
                                         "lateral_axis": np.array([-1.0, 0, 0])},
                      mg.quantities)
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": [
        UPRIGHT,
        {"axis": "teapot.front_axis", "relation": "parallel",
         "reference": "mug.front_axis", "tol": "loose"}], "trans": "free"}}
    syms = {**SYMBOLS, "mug": fronted}
    em = parse_emission(json.dumps(doc), Vocabulary.from_symbols(syms))
    cs = compile_stage(em, syms, POSES, w_point=OPENING, e_point=TIP)
    R_goal = cs.subgoal.zero()[:3, :3]
    np.testing.assert_allclose(R_goal @ _u(SYMBOLS["teapot"].axes["up_axis"]), [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(R_goal @ _u(SYMBOLS["teapot"].axes["front_axis"]), [0, 1, 0], atol=1e-9)
    # roll/pitch from the up row (moderate), yaw from the front row (loose)
    m, l = ROT_TOL_RAD["moderate"], ROT_TOL_RAD["loose"]
    np.testing.assert_allclose(cs.subgoal.Bw[3:], [(-m, m), (-m, m), (-l, l)])
    # the path only carries the upright row, already satisfied: no corridor
    assert not any("corridor" in n for n in cs.notes)


def test_sigma_floors_fixed_rows():
    tp = SYMBOLS["teapot"]
    noisy = Symbols("teapot", tp.points, tp.axes, tp.quantities,
                    sigmas={"axes.front_axis": 5.0})
    cs = compile_stage(emission(POUR), {**SYMBOLS, "teapot": noisy}, POSES,
                       w_point=OPENING, e_point=TIP)
    h = SIGMA_K * np.deg2rad(5.0)             # > tight (5 deg)
    np.testing.assert_allclose(cs.subgoal.Bw[3], (-h, h))
    np.testing.assert_allclose(cs.subgoal.Bw[4], (-h, h))
    np.testing.assert_allclose(cs.path.Bw[3], (-np.pi / 2 - h, h), atol=1e-9)


# ------------------------------------------------------------ expr terms

def test_expr_rows_evaluate_quantities():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": "free", "trans": [
        {"term": "expr", "row": "x", "lo": "0 - mug.rim_radius",
         "hi": "mug.rim_radius"}]}
    cs = compile_stage(emission(doc), SYMBOLS, POSES, w_point=OPENING)
    r = SYMBOLS["mug"].quantities["rim_radius"]
    np.testing.assert_allclose(cs.subgoal.Bw[0], (-r, r), atol=1e-9)


# --------------------------------------------------------- typed errors

def _expect(doc, poses=POSES, symbols=SYMBOLS, **kw):
    kw.setdefault("w_point", OPENING)
    with pytest.raises(CompileError) as e:
        compile_stage(emission(doc), symbols, poses, **kw)
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


def test_axis_on_passive_object_rejected():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": [
        {"axis": "mug.up_axis", "relation": "parallel",
         "reference": "world.z", "tol": "tight"}], "trans": "free"}
    assert "must belong to the active object" in _expect(doc).reason


def test_empty_intersection_rejected():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": "free", "trans": [
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "medium", "slack": "moderate"},
        {"term": "inside", "anchor": "mug.opening_center",
         "slack": "snug"}]}          # z bands [0.03,0.08] vs [-5,5] mm
    assert "intersection empty" in _expect(doc).reason


def test_reference_off_w_basis_rejected():
    # a passive tilted 45 deg: world.z lands between w's basis vectors
    tilted = {**POSES, "mug": _tilted(POSES["mug"], 45.0)}
    err = _expect(TRANSPORT, poses=tilted, e_point=TIP)
    assert "w basis vector" in err.reason and "rot[0]" in err.slot


def test_half_turn_flip_rejected():
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": [
        {"axis": "teapot.up_axis", "relation": "antiparallel",
         "reference": "world.z", "tol": "tight"}], "trans": "free"}}
    assert "half-turn" in _expect(doc, e_point=TIP).reason


def test_perpendicular_from_parallel_entry_rejected():
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": [
        {"axis": "teapot.up_axis", "relation": "perpendicular",
         "reference": "world.z", "tol": "tight"}], "trans": "free"}}
    assert "ambiguous" in _expect(doc, e_point=TIP).reason


def test_relation_rows_with_gripper_mover_rejected():
    doc = {"stage": 1, "name": "grasp", "active": "teapot", "passive": None,
           "path_tsr": {"rot": "free", "trans": "free"},
           "subgoal_tsr": {"rot": [UPRIGHT], "trans": [
               {"term": "centered", "anchor": "teapot.handle_center",
                "tol": "snug"}]}, "verify": ""}
    err = _expect(doc, w_point=SYMBOLS["teapot"].points["handle_center"])
    assert "gripper" in err.reason


def test_grasp_stage_roots_w_on_grasped_object():
    doc = {"stage": 1, "name": "grasp", "active": "teapot", "passive": None,
           "path_tsr": {"rot": "free", "trans": "free"},
           "subgoal_tsr": {"rot": "free", "trans": [
               {"term": "centered", "anchor": "teapot.handle_center",
                "tol": "snug"}]}, "verify": ""}
    h = SYMBOLS["teapot"].points["handle_center"]
    cs = compile_stage(emission(doc), SYMBOLS, POSES, w_point=h)
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 3], (POSES["teapot"] @ np.append(h, 1))[:3])
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 0],
                               _u(POSES["teapot"][:3, :3] @ SYMBOLS["teapot"].axes["front_axis"]))
    assert np.allclose(cs.subgoal.Tw_e, np.eye(4))
    assert all(_is_free(cs.subgoal.Bw[i]) for i in (3, 4, 5))
    with pytest.raises(ValueError):
        compile_stage(emission(doc), SYMBOLS, POSES, w_point=h, e_point=TIP)


def test_missing_canonical_up_rejected():
    mg = SYMBOLS["mug"]
    noup = Symbols("mug", mg.points, {}, mg.quantities)
    err = _expect(TRANSPORT, symbols={**SYMBOLS, "mug": noup})
    assert "up_axis" in err.reason and err.slot == "passive"


# ------------------------------------------ emissions artifact round trip

def test_emission_json_round_trip_and_taskframes_switch(tmp_path):
    from dataclasses import asdict

    from manip_sim.pour_stages import TaskFrames
    from manip_sim.vlm import emission_from_json, load_emissions

    em_t, em_p = emission(TRANSPORT), emission(POUR)
    assert emission_from_json(asdict(em_t)) == em_t
    art = tmp_path / "em.json"
    art.write_text(json.dumps({
        "roles": ["transport_active", "pour"],
        "emissions": [asdict(em_t), asdict(em_p)],
        "compiled": [{"stage": "transport", "grounded": True},
                     {"stage": "pour", "grounded": True}]}))
    ems = load_emissions(art)
    assert set(ems) == {"transport_active", "pour"}

    tp, mg = SYMBOLS["teapot"], SYMBOLS["mug"]
    common = dict(spout_tip=tp.frame("spout_tip", "pour_axis"),
                  tilt_frame=tp.frame("spout_tip", "tilt_axis", secondary="pour_axis"),
                  opening=mg.frame("opening_center", "up_axis"), symbols=SYMBOLS)
    hand = TaskFrames(**common)
    emitted = TaskFrames(**common, emissions=ems)
    h2 = hand.transport(POSES["teapot"], POSES["mug"])
    e2 = emitted.transport(POSES["teapot"], POSES["mug"])
    # same w origin and the same nominal body pose; both expose .path/.subgoal
    assert np.allclose(h2.subgoal.T0_w[:3, 3], e2.subgoal.T0_w[:3, 3])
    assert np.allclose(h2.subgoal.zero(), e2.subgoal.zero())
    h3 = hand.pour(POSES["teapot"], POSES["mug"], np.pi / 2)
    e3 = emitted.pour(POSES["teapot"], POSES["mug"], np.pi / 2)
    # same goal attitude; the emitted pivot is the opening, the hand one the tip
    assert np.allclose(h3.subgoal.nominal()[:3, :3], e3.subgoal.nominal()[:3, :3])
    assert np.allclose(e3.subgoal.T0_w[:3, 3], e2.subgoal.T0_w[:3, 3])

    # gate: a failed compile is refused
    art.write_text(json.dumps({"roles": ["pour"], "emissions": [asdict(em_p)],
                               "compiled": [{"stage": "pour", "grounded": False}]}))
    with pytest.raises(SystemExit, match="compile gate failed"):
        load_emissions(art)


def test_path_and_subgoal_errors_reported_together():
    # the same mistake in both TSRs must cost one retry, not two
    doc = dict(TRANSPORT)
    bad = {"rot": "free", "trans": [
        {"term": "centered", "anchor": "teapot.spout_tip", "tol": "moderate"}]}
    doc["path_tsr"] = bad
    doc["subgoal_tsr"] = bad
    err = _expect(doc)
    assert [e.slot.rsplit(".", 2)[1] for e in err.all()] == ["path", "subgoal"]
    assert err.text().count("static in w") == 2
    # single-TSR failure keeps the plain shape
    doc["path_tsr"] = {"rot": "free", "trans": "free"}
    assert _expect(doc).others == ()
