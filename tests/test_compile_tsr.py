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


UPRIGHT = {"axis": "teapot.+up", "relation": "parallel",
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

SPOUT_DOWN = {"axis": "teapot.+front", "relation": "antiparallel",
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


# pour's ENTRY is transport's subgoal center (the compiler is handed entry
# poses, and the path must admit the entry): chain it the way the emit
# gate does rather than compiling pour against the spawn pose
TRANSPORT_CS = compile_stage(emission(TRANSPORT), SYMBOLS, POSES,
                             w_point=OPENING, e_point=TIP)
POUR_POSES = {**POSES, "teapot": TRANSPORT_CS.subgoal.nominal()}


def compile_pour(poses=POUR_POSES, doc=POUR):
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
           "subgoal_tsr": {"rot": [{"axis": "mug.+up", "relation": "parallel",
                                    "reference": "teapot.+up", "tol": "tight"}],
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
    # yaw the entry about the (frozen) tip so the tip stays over the opening
    yawed_T = _tilted(POUR_POSES["teapot"], 50.0, (0, 0, 1))
    yawed_T[:3, 3] += (POUR_POSES["teapot"][:3, :3] - yawed_T[:3, :3]) @ TIP
    yawed = {**POUR_POSES, "teapot": yawed_T}
    cs = compile_pour(yawed)
    lat = _u(yawed["teapot"][:3, :3] @ SYMBOLS["teapot"].axes["lateral_axis"])
    np.testing.assert_allclose(cs.subgoal.T0_w[:3, 0], lat, atol=1e-9)
    dR = cs.subgoal.zero()[:3, :3] @ yawed["teapot"][:3, :3].T
    rv = R.from_matrix(dR).as_rotvec()
    assert np.linalg.norm(rv) == pytest.approx(np.pi / 2, abs=1e-9)
    np.testing.assert_allclose(rv / np.linalg.norm(rv), lat, atol=1e-9)


def test_pour_corridor_off_basis_is_a_typed_rejection():
    # give the mug a confident front that is NOT the teapot's lateral:
    # the tilt axis lands between w.x and w.y, the corridor is not a
    # single row: a typed error on the path row, never a silent free
    mg = SYMBOLS["mug"]
    front = np.array([1.0, 0.0, 0.0])
    fronted = Symbols("mug", mg.points, {**mg.axes, "front_axis": front,
                                         "lateral_axis": np.cross([0, 0, 1.0], front)},
                      mg.quantities)
    cs = compile_pour(doc=POUR)  # baseline: corridor exists
    assert any("corridor" in n for n in cs.notes)
    with pytest.raises(CompileError) as e:
        compile_stage(emission(POUR), {**SYMBOLS, "mug": fronted}, POUR_POSES,
                      w_point=OPENING, e_point=TIP)
    assert e.value.slot == "pour.path.rot[0]"
    assert "not about a single w axis" in e.value.reason
    assert "leave the path rotation free" in e.value.reason
    t = ROT_TOL_RAD["tight"]                    # subgoal unaffected
    np.testing.assert_allclose(cs.subgoal.Bw[3], (-t, t))
    np.testing.assert_allclose(cs.subgoal.Bw[4], (-t, t))


def test_corridor_never_on_pitch_and_says_so():
    # a passive front aligned with the teapot's front puts the tilt axis
    # on w.y: pitch is the middle Euler angle, so no corridor — and that
    # is a typed error on the path row, not a silently freed rotation
    mg = SYMBOLS["mug"]
    front = SYMBOLS["teapot"].axes["front_axis"]
    fronted = Symbols("mug", mg.points, {**mg.axes, "front_axis": front,
                                         "lateral_axis": np.cross([0, 0, 1.0], front)},
                      mg.quantities)
    with pytest.raises(CompileError) as e:
        compile_stage(emission(POUR), {**SYMBOLS, "mug": fronted}, POUR_POSES,
                      w_point=OPENING, e_point=TIP)
    assert e.value.slot == "pour.path.rot[0]"
    assert "about w.y (pitch)" in e.value.reason


def test_v3_schema_only_spout_roll_path_row_now_rejected():
    # ablation-modality-v3 schema_only.1/.2: path "lateral points world.z"
    # is a 90 deg roll about the spout — axis w.y. Before: silently freed,
    # PASS with a note. Now: typed rejection on the path row.
    doc = {**POUR, "path_tsr": {"rot": [
        {"axis": "teapot.+left", "relation": "parallel",
         "reference": "world.z", "tol": "moderate"}], "trans": POUR["path_tsr"]["trans"]}}
    with pytest.raises(CompileError) as e:
        compile_pour(doc=doc)
    assert e.value.slot == "pour.path.rot[0]"
    assert "about w.y (pitch)" in e.value.reason


# ------------------------------------------------------- rule-table cases

def test_perpendicular_fixes_one_tilt():
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": [
        {"axis": "teapot.+front", "relation": "perpendicular",
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
        {"axis": "teapot.+front", "relation": "parallel",
         "reference": "mug.+front", "tol": "loose"}], "trans": "free"}}
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
    cs = compile_stage(emission(POUR), {**SYMBOLS, "teapot": noisy}, POUR_POSES,
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
        {"axis": "teapot.+up", "relation": "parallel",
         "reference": "teapot.+front", "tol": "tight"}],
        "trans": "free"}
    assert "degenerate" in _expect(doc).reason


def test_axis_on_passive_object_rejected():
    doc = dict(TRANSPORT)
    doc["subgoal_tsr"] = {"rot": [
        {"axis": "mug.+up", "relation": "parallel",
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
        {"axis": "teapot.+up", "relation": "antiparallel",
         "reference": "world.z", "tol": "tight"}], "trans": "free"}}
    assert "half-turn" in _expect(doc, e_point=TIP).reason


INCONSISTENT = [                # axes 90 deg apart, references 180 deg apart
    UPRIGHT,
    {"axis": "teapot.+front", "relation": "antiparallel",
     "reference": "mug.+up", "tol": "loose"}]


def test_implied_perpendicular_row_is_dropped_with_note():
    # pour pair: spout down + lateral stays horizontal. The second row is
    # implied by the first (left _|_ front, front -> -up  =>  left _|_ up).
    doc = {**POUR, "subgoal_tsr": {"rot": [
        SPOUT_DOWN,
        {"axis": "teapot.+left", "relation": "perpendicular",
         "reference": "mug.+up", "tol": "moderate"}], "trans": "free"}}
    cs = compile_pour(doc=doc)
    single = compile_pour(doc={**POUR, "subgoal_tsr": {"rot": [SPOUT_DOWN],
                                                       "trans": "free"}})
    np.testing.assert_allclose(cs.subgoal.zero(), single.subgoal.zero(), atol=1e-12)
    np.testing.assert_allclose(cs.subgoal.Bw, single.subgoal.Bw)
    assert any("pour.subgoal.rot[1] is implied by pour.subgoal.rot[0]" in n
               for n in cs.notes)


def test_perpendicular_row_not_implied_still_rejected():
    # reference off the aligning target's line: a yaw constraint, not implied
    mg = SYMBOLS["mug"]
    fronted = Symbols("mug", mg.points, {**mg.axes, "front_axis": np.array([0, 1.0, 0]),
                                         "lateral_axis": np.array([-1.0, 0, 0])},
                      mg.quantities)
    doc = {**POUR, "subgoal_tsr": {"rot": [
        SPOUT_DOWN,
        {"axis": "teapot.+left", "relation": "perpendicular",
         "reference": "mug.+front", "tol": "moderate"}], "trans": "free"}}
    syms = {**SYMBOLS, "mug": fronted}
    em = parse_emission(json.dumps(doc), Vocabulary.from_symbols(syms))
    with pytest.raises(CompileError) as e:
        compile_stage(em, syms, POUR_POSES, w_point=OPENING, e_point=TIP)
    # the path row is also off-basis for this fronted mug; both slots carry
    assert any("outside the rule table" in x.reason for x in e.value.all())


# ------------------------------------------------- path containment (step 3)

def _path(doc, rot=None, trans=None):
    p = dict(doc["path_tsr"])
    if rot is not None:
        p["rot"] = rot
    if trans is not None:
        p["trans"] = trans
    return {**doc, "path_tsr": p}


ABOVE_LARGE = {"term": "above", "anchor": "mug.opening_center",
               "clearance": "large", "slack": "loose"}          # z in [0.08, 0.20]


def test_entry_outside_path_beyond_budget_rejected():
    # carry-high transport path; the teapot enters on the table, 0.052 m
    # below the band: more than the start projection is trusted to close
    err = _expect(_path(TRANSPORT, trans=[ABOVE_LARGE]), e_point=TIP)
    assert err.slot == "transport.path.trans[0]"
    assert "entry pose lies outside" in err.reason and "0.03" in err.reason
    assert "leave z free" in err.reason


def test_entry_outside_path_within_budget_is_a_note():
    # pour entry tip is 0.055 m above the opening; above(contact, moderate)
    # is [0.00, 0.05]: 0.005 m outside, inside the budget -> note, and the
    # band overlaps the subgoal's [0.01, 0.03] with positive measure
    cs = compile_pour(doc=_path(POUR, trans=[
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "contact", "slack": "moderate"},
        {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"}]))
    assert any("within the start-projection budget" in n for n in cs.notes)


def test_touching_translation_bands_are_empty_for_the_planner():
    # path above(medium, moderate) = [0.03, 0.08] vs subgoal above(small,
    # snug) = [0.01, 0.03]: they touch at one point. The planner samples
    # the intersection, so this never yields a goal -> typed rejection
    err = _expect(_path(POUR, trans=[
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "medium", "slack": "moderate"},
        {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"}]),
        poses=POUR_POSES, e_point=TIP)
    assert err.slot == "pour.path.trans[0]"
    assert "does not meet the subgoal" in err.reason


def test_path_translation_missing_subgoal_rejected():
    # path z [0.08, 0.20] vs subgoal z [0.01, 0.03]: the path excludes the
    # goal; slot is the path term that set the violated side
    err = _expect(_path(POUR, trans=[
        ABOVE_LARGE,
        {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"}]),
        poses=POUR_POSES, e_point=TIP)
    assert err.slot == "pour.path.trans[0]"
    assert "does not meet the subgoal" in err.reason


def test_path_rotation_excluding_goal_attitude_rejected():
    # ablation-modality-v1's framed pour: path "front perpendicular up"
    # (spout stays level), subgoal "front antiparallel up" (spout down).
    # Different Tw_e -> sampled check; no subgoal attitude is on the path.
    err = _expect(_path(POUR, rot=[
        {"axis": "teapot.+front", "relation": "perpendicular",
         "reference": "mug.+up", "tol": "moderate"}]),
        poses=POUR_POSES, e_point=TIP)
    assert err.slot == "pour.path.rot[0]"
    assert "0/64" in err.reason and "exclude the goal attitude" in err.reason


def test_path_contains_both_ends_by_construction():
    cs = compile_pour()
    assert cs.path.contains(POUR_POSES["teapot"], tol=1e-9)
    assert cs.path.contains(cs.subgoal.nominal(), tol=1e-9)
    assert not any("outside" in n for n in cs.notes)


# ------------------------------------------------------- stage seam (planner)

def test_stage_seam_rejects_pour_path_narrower_than_transport_goal():
    # ablation-modality-v2 framed.1: transport subgoal above(medium, moderate)
    # z in [0.03, 0.08]; pour path above(small, moderate) z in [0.01, 0.06].
    # The planner may end transport at z = 0.07; the pour path excludes it.
    from manip_sim.compile_tsr import check_stage_seam
    em = emission(POUR)
    cs = compile_pour()                 # POUR fixture's path is above(small)
    with pytest.raises(CompileError) as e:
        check_stage_seam(TRANSPORT_CS, cs, em)
    assert e.value.slot == "pour.path.trans[0]"        # the above() term
    assert "does not admit every transport goal" in e.value.reason
    assert "z band" in e.value.reason and "0.02" in e.value.reason


def test_stage_seam_passes_when_path_covers_the_previous_subgoal():
    from manip_sim.compile_tsr import check_stage_seam
    doc = _path(POUR, trans=[
        {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"},
        {"term": "above", "anchor": "mug.opening_center",
         "clearance": "small", "slack": "loose"}])         # z in [0.01, 0.13]
    cs = compile_pour(doc=doc)
    notes = check_stage_seam(TRANSPORT_CS, cs, emission(doc))
    assert notes and "admits every transport goal" in notes[0]


def test_inconsistent_pair_slots_both_rows():
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": INCONSISTENT, "trans": "free"}}
    err = _expect(doc, e_point=TIP)
    assert err.slot == "transport.subgoal.rot[0]"
    assert [o.slot for o in err.others] == ["transport.subgoal.rot[1]"]
    assert "Drop one" in err.reason and "restate" not in err.reason
    assert "sign" in err.reason


def test_inconsistent_pair_slots_survive_two_tsr_combine():
    doc = {**TRANSPORT, "path_tsr": {"rot": INCONSISTENT, "trans": "free"},
           "subgoal_tsr": {"rot": INCONSISTENT, "trans": "free"}}
    err = _expect(doc, e_point=TIP)
    assert [e.slot for e in err.all()] == [
        "transport.path.rot[0]", "transport.path.rot[1]",
        "transport.subgoal.rot[0]", "transport.subgoal.rot[1]"]


def test_perpendicular_from_parallel_entry_rejected():
    doc = {**TRANSPORT, "subgoal_tsr": {"rot": [
        {"axis": "teapot.+up", "relation": "perpendicular",
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
    h3 = hand.pour(POUR_POSES["teapot"], POSES["mug"], np.pi / 2)
    e3 = emitted.pour(POUR_POSES["teapot"], POSES["mug"], np.pi / 2)
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


# --------------------------- six-direction alphabet: compiler unchanged

def test_direction_tokens_compile_identically_to_unsigned_rows():
    """Patch-1 invariant: the parser's sign normalization hands the
    compiler exactly the unsigned RotRow it consumed before the alphabet
    change, so B^w, T0_w and Tw_e are bit-identical between (a) the
    six-direction emission parsed through vlm.py and (b) a hand-built
    StageEmission in the compiler's own unsigned form. The two signed
    spellings of the pour tilt are checked against the same target."""
    from dataclasses import replace
    from manip_sim.vlm import RotRow
    parsed = emission(POUR)                        # "teapot.+front antiparallel world.z"
    flipped = dict(POUR)
    flipped["path_tsr"] = {**POUR["path_tsr"], "rot": [
        {"axis": "teapot.-front", "relation": "parallel",
         "reference": "world.z", "tol": "tight"}]}
    flipped["subgoal_tsr"] = {**POUR["subgoal_tsr"], "rot": flipped["path_tsr"]["rot"]}
    parsed_flipped = emission(flipped)
    unsigned = RotRow(axis="teapot.front_axis", relation="antiparallel",
                      reference="world.z", tol="tight")
    hand = replace(parsed,
                   path_tsr=replace(parsed.path_tsr, rot=(unsigned,)),
                   subgoal_tsr=replace(parsed.subgoal_tsr, rot=(unsigned,)))
    assert parsed.path_tsr.rot == (unsigned,) == parsed_flipped.path_tsr.rot
    outs = [compile_stage(e, SYMBOLS, POUR_POSES, w_point=OPENING, e_point=TIP)
            for e in (parsed, parsed_flipped, hand)]
    for cs in outs[1:]:
        for k in ("path", "subgoal"):
            a, b = getattr(outs[0], k), getattr(cs, k)
            np.testing.assert_array_equal(a.Bw, b.Bw)
            np.testing.assert_array_equal(a.T0_w, b.T0_w)
            np.testing.assert_array_equal(a.Tw_e, b.Tw_e)


# ------------------------------- emit-gate pair consistency (not the
# compiler: the planner's sample_intersection test, run at compile time)

def _pour_doc(path_rot, sub_trans):
    above = {"term": "above", "anchor": "mug.opening_center",
             "clearance": "small", "slack": "moderate"}
    return {"stage": 3, "name": "pour", "active": "teapot", "passive": "mug",
            "path_tsr": {"rot": path_rot, "trans": [above]},
            "subgoal_tsr": {"rot": [{"axis": "teapot.+front",
                                     "points": "mug.-up", "tol": "moderate"}],
                            "trans": sub_trans},
            "verify": ""}


_SUB_TRANS = [{"term": "above", "anchor": "mug.opening_center",
               "clearance": "small", "slack": "snug"},
              {"term": "centered", "anchor": "mug.opening_center", "tol": "snug"}]


def _compiled(doc):
    from manip_sim.compile_tsr import compile_stage
    return compile_stage(emission(doc), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)


def test_pair_check_rejects_a_path_row_on_the_rotated_direction():
    """The specimen from ablation-modality-v1: every run put the path
    row on the spout direction (true at entry), which pins the tilt to
    +-30 deg while the subgoal needs 90 deg. compile_stage itself now
    rejects it (path must admit the subgoal attitude), slotted on the
    path row, before the gate's joint sampling check would run."""
    from manip_sim.compile_tsr import CompileError
    with pytest.raises(CompileError) as e:
        _compiled(_pour_doc([{"axis": "teapot.+front",
                              "perpendicular_to": "mug.+up", "tol": "loose"}],
                            _SUB_TRANS))
    assert e.value.slot == "pour.path.rot[0]"
    assert "0/64" in e.value.reason
    assert "leave the path rotation free" in e.value.reason


def test_pair_check_passes_the_invariant_path_row():
    """The correct pour path pins the axis the stage turns ABOUT (the
    lateral stays horizontal) and leaves the turn free: subgoal on the
    path manifold, nothing raised."""
    from manip_sim.compile_tsr import check_pair_consistency
    cs = _compiled(_pour_doc([{"axis": "teapot.+left",
                               "perpendicular_to": "mug.+up", "tol": "tight"}],
                             _SUB_TRANS))
    check_pair_consistency(cs)
    assert any("fixed ['pitch']" in n for n in cs.notes)


def test_pair_check_rejects_an_unsampleable_subgoal():
    from manip_sim.compile_tsr import CompileError, check_pair_consistency
    cs = _compiled(_pour_doc([{"axis": "teapot.+left",
                               "perpendicular_to": "mug.+up", "tol": "tight"}],
                             _SUB_TRANS[:1]))          # z bounded, x/y not
    with pytest.raises(CompileError) as e:
        check_pair_consistency(cs)
    assert e.value.slot == "pour.subgoal.trans"
    assert "['x', 'y'] are unbounded" in e.value.reason


def test_pair_check_is_deterministic():
    # the rotation probe is seeded: three compiles, one message
    from manip_sim.compile_tsr import CompileError
    msgs = set()
    for _ in range(3):
        with pytest.raises(CompileError) as e:
            _compiled(_pour_doc([{"axis": "teapot.+front",
                                  "perpendicular_to": "mug.+up", "tol": "loose"}],
                                _SUB_TRANS))
        msgs.add(e.value.reason)
    assert len(msgs) == 1


# ------------------------------------------------------- accounting (step 10)

def test_stage_accounting_classifies_the_pour_goal_and_sweep():
    from manip_sim.compile_tsr import stage_accounting
    cs = compile_pour()
    a = stage_accounting(cs, emission(POUR), SYMBOLS, POUR_POSES)
    assert a["goal_class"] == {"+front": "down", "+left": "level", "+up": "level"}
    assert a["sweep"]["axis"] == "+left" and a["sweep"]["angle_deg"] == pytest.approx(90.0, abs=0.1)
    assert a["x_source"] == "teapot lateral (up x front) at entry"
    assert a["free_rows"] == {"path": ["yaw"], "subgoal": ["yaw"]}
    np.testing.assert_allclose(a["goal_dirs"]["+front"], [0, 0, -1], atol=1e-3)


def test_stage_accounting_transport_has_no_sweep_and_grasp_only_x_source():
    from manip_sim.compile_tsr import stage_accounting
    cs = TRANSPORT_CS
    a = stage_accounting(cs, emission(TRANSPORT), SYMBOLS, POSES)
    assert a["goal_class"]["+up"] == "up" and a["sweep"]["axis"] == "none"
    grasp = {"stage": 1, "name": "grasp", "active": "teapot", "passive": None,
             "path_tsr": {"rot": "free", "trans": "free"},
             "subgoal_tsr": {"rot": "free", "trans": [
                 {"term": "inside", "anchor": "teapot.handle_center", "slack": "snug"}]},
             "verify": ""}
    g = compile_stage(emission(grasp), SYMBOLS, POSES,
                      w_point=SYMBOLS["teapot"].points["handle_center"])
    assert stage_accounting(g, emission(grasp), SYMBOLS, POSES) == {"x_source": g.x_source}
