"""scripts/render_preview.py: the critic's preview images. Geometry (state
chaining, the representative pose, one-row-at-a-time boundary poses, view
choice, gripper loaders, label points) runs without GL; the render test
skips without an offscreen GL context."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manip_sim.tsr import TSR, displacement_to_pose, make_pose
from scripts import render_preview as rp

HW_ROBOTIQ = Path(__file__).resolve().parents[2] / "ur5e-manip-hardware"
HW_URDF = HW_ROBOTIQ / "curobo_runtime/robot/ur5e_robotiq.urdf"
HW_MESHES = HW_ROBOTIQ / "test/curobo_incontainer/robotiq/meshes/collision"


def _pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return make_pose(np.array([x, y, z]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))


def _path():
    # x, y free; z in [0.03, 0.10]; roll/pitch +-15 deg (moderate); yaw free;
    # w offset from the world origin so frames matter
    Bw = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [0.03, 0.10],
                   [-np.radians(15), np.radians(15)], [-np.radians(15), np.radians(15)],
                   [-np.pi, np.pi]])
    return TSR(T0_w=_pose(0.5, 0.1, 0.0), Bw=Bw)


def _subgoal(z=0.06):
    Bw = np.array([[-0.01, 0.01], [-0.01, 0.01], [z, z],
                   [0, 0], [0, 0], [-0.1, 0.1]])
    return TSR(T0_w=_pose(0.5, 0.1, 0.0), Bw=Bw)


# ------------------------------------------------------------------ states

def test_chain_states_goal_is_subgoal_nominal_and_gripper_follows():
    init = {"teapot": _pose(0.2, 0.0, 0.0), "mug": _pose(0.5, 0.1, 0.0)}
    sg = _subgoal()
    st = rp.StageInput("transport_active", "carry", "teapot", "mug", _path(), sg)
    tcp = _pose(0.2, -0.1, 0.05)
    states, skipped = rp.chain_states(init, [st], rp.GraspInput("teapot", tcp))
    assert [s.key for s in states] == ["initial", "grasp", "goal:transport_active"]
    assert skipped == []
    assert states[0].gripper_pose is None
    np.testing.assert_allclose(states[1].gripper_pose, tcp)
    goal = states[2]
    np.testing.assert_allclose(goal.poses["teapot"], sg.nominal())
    np.testing.assert_allclose(goal.poses["mug"], init["mug"])          # passive unchanged
    # the gripper keeps its grasp transform relative to the held object
    T_obj_tcp = np.linalg.inv(init["teapot"]) @ tcp
    np.testing.assert_allclose(goal.gripper_pose, goal.poses["teapot"] @ T_obj_tcp)


def test_uncompiled_stage_ends_the_chain():
    init = {"teapot": _pose(), "mug": _pose(0.5)}
    a = rp.StageInput("transport_active", "carry", "teapot", "mug", None, None,
                      error="compile rejected at path.rot[0]")
    b = rp.StageInput("pour", "pour", "teapot", "mug", _path(), _subgoal())
    states, skipped = rp.chain_states(init, [a, b])
    assert [s.key for s in states] == ["initial"]
    assert [s["role"] for s in skipped] == ["transport_active", "pour"]
    assert "path.rot[0]" in skipped[0]["reason"]
    assert "transport_active" in skipped[1]["reason"]


# ------------------------------------------------------------------ boundary

def test_representative_pose_is_clamped_midpoint():
    path = _path()
    entry = path.T0_w @ displacement_to_pose(np.array([0.0, 0.0, 0.0, 0, 0, 0])) @ path.Tw_e
    goal = path.T0_w @ displacement_to_pose(np.array([0.2, 0.0, 0.08, 0, 0, 0])) @ path.Tw_e
    d, T = rp.representative_pose(path, entry, goal)
    # x free: plain midpoint; z: midpoint 0.04 is inside [0.03, 0.10]
    np.testing.assert_allclose(d[:3], [0.1, 0.0, 0.04], atol=1e-9)
    assert path.contains(T, tol=1e-9)
    entry_low = path.T0_w @ displacement_to_pose(np.array([0, 0, -0.10, 0, 0, 0]))
    d2, _ = rp.representative_pose(path, entry_low, entry_low)
    assert d2[2] == pytest.approx(0.03)                  # clamped into the box


def test_boundaries_move_one_row_at_a_time():
    path = _path()
    d_rep = np.array([0.1, 0.0, 0.05, 0.02, -0.01, 0.3])
    bounds, pinned = rp.boundary_samples(path, d_rep)
    assert pinned == []
    assert [rp.ROW_NAMES[b.row] for b in bounds] == list(rp.ROW_NAMES)
    for b in bounds:
        for v, T in zip(b.values, b.poses):
            d = path.displacement(T)
            others = [k for k in range(6) if k != b.row]
            np.testing.assert_allclose(d[others], d_rep[others], atol=1e-7)
            if b.row >= 3:      # angles compare modulo 2 pi
                assert np.cos(d[b.row] - v) == pytest.approx(1.0)
            else:
                assert d[b.row] == pytest.approx(v)


def test_boundary_kinds_tokens_and_free_samples():
    path = _path()
    d_rep = np.zeros(6)
    d_rep[2] = 0.05
    by = {rp.ROW_NAMES[b.row]: b for b in rp.boundary_samples(path, d_rep)[0]}
    assert by["z"].kind == "bounded" and by["z"].values == pytest.approx([0.03, 0.10])
    assert by["roll"].token == "moderate" and by["pitch"].token == "moderate"
    assert by["roll"].values == pytest.approx([-np.radians(15), np.radians(15)])
    assert by["yaw"].kind == "free" and by["yaw"].token is None
    assert by["yaw"].values == pytest.approx([np.radians(a) for a in rp.FREE_ROT_SAMPLES_DEG])
    assert by["x"].kind == "free"
    assert by["x"].values == pytest.approx([-rp.FREE_TRANS_SHOW_M, rp.FREE_TRANS_SHOW_M])
    assert "moderate" in by["roll"].captions[0] and "free" in by["yaw"].captions[0]


def test_pinned_rows_are_listed_not_drawn():
    bounds, pinned = rp.boundary_samples(_subgoal(), np.zeros(6))
    assert pinned == ["z", "roll", "pitch"]
    assert [rp.ROW_NAMES[b.row] for b in bounds] == ["x", "y", "yaw"]


def test_row_labels_are_appended_to_captions():
    bounds, _ = rp.boundary_samples(_path(), np.zeros(6), {3: "stage1.path.rot[0]"})
    roll = next(b for b in bounds if b.row == 3)
    assert all(c.endswith("stage1.path.rot[0]") for c in roll.captions)


def test_pick_view_rotation_along_axis_translation_across():
    cams = {"top": {"pos": np.array([0, 0, 3.0])},
            "+x": {"pos": np.array([3.0, 0, 1.0])},
            "+y": {"pos": np.array([0, 3.0, 1.0])}}
    c = np.zeros(3)
    z, x = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    assert rp.pick_view(cams, c, z, rotation=True, views=("top", "+x", "+y")) == "top"
    assert rp.pick_view(cams, c, x, rotation=True, views=("top", "+x", "+y")) == "+x"
    assert rp.pick_view(cams, c, z, rotation=False, views=("top", "+x", "+y")) == "+x"
    assert rp.pick_view(cams, c, x, rotation=False, views=("top", "+x", "+y")) == "top"


# ------------------------------------------------------------------ labels, gripper

def test_label_points_follow_the_object_pose():
    T = _pose(1.0, 2.0, 0.5, yaw=np.pi / 2)
    pts = rp.label_points({"teapot": {"spout_center": np.array([0.1, 0, 0])},
                           "ghost": {"p": np.zeros(3)}}, {"teapot": T})
    assert [t for t, _ in pts] == ["teapot.spout_center"]
    np.testing.assert_allclose(pts[0][1], [1.0, 2.1, 0.5], atol=1e-12)


def test_label_projection_lands_on_the_image_center():
    pytest.importorskip("mujoco")
    from scripts.render_candidates import canonical_cameras, project
    c = np.array([0.3, -0.2, 0.1])
    cams = canonical_cameras(c, 0.5)
    uv, depth = project(c[None], cams["iso"], px=rp.STATE_PX)
    np.testing.assert_allclose(uv[0], [rp.STATE_PX / 2] * 2, atol=1e-6)
    assert depth[0] > 0


def test_gripper_obj_roundtrip(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=[0.02, 0.04, 0.06])
    g = rp.GripperModel(np.asarray(box.vertices), np.asarray(box.faces), "box")
    m = trimesh.load(str(g.write_obj(tmp_path / "g.obj")), force="mesh", process=False)
    np.testing.assert_allclose(m.extents, [0.02, 0.04, 0.06], atol=1e-6)


@pytest.mark.skipif(not (HW_URDF.is_file() and HW_MESHES.is_dir()),
                    reason="hardware 2F-85 URDF/meshes not checked out next to the sim")
def test_hardware_gripper_from_urdf_hangs_below_tool0():
    pytest.importorskip("trimesh")
    g = rp.gripper_from_urdf(HW_URDF, "tool0", HW_MESHES)
    lo, hi = g.V.min(0), g.V.max(0)
    # the 2F-85 (adapter + body + open fingers) extends ~0.15-0.20 m along
    # tool0's +z and is ~0.15 m wide at most; nothing behind the flange
    assert 0.12 < hi[2] < 0.25 and lo[2] > -0.01
    assert np.all(hi[:2] - lo[:2] < 0.20)


def test_sim_gripper_tcp_at_the_grip_site():
    pytest.importorskip("mujoco")
    pytest.importorskip("trimesh")
    pytest.importorskip("robosuite")
    g = rp.sim_gripper()
    # the grip site sits 0.145 m out from the adapter, between the open
    # fingertips: the body lies behind it (-z) and the pads straddle it
    assert g.V[:, 2].min() == pytest.approx(-0.145, abs=0.01)
    assert g.V[:, 2].max() > -0.01
    assert g.V[:, 2].max() < 0.05


# ------------------------------------------------------------------ inputs, render

def _gl_ok() -> bool:
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><visual><global offwidth="8" offheight="8"/></visual>'
            '<worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
        r = mujoco.Renderer(m, 8, 8)
        r.update_scene(mujoco.MjData(m))
        r.render()
        r.close()
        return True
    except Exception:
        return False


def _box_asset(root: Path, name: str, half=(0.03, 0.03, 0.05)) -> Path:
    import trimesh
    d = root / name
    (d / "meshes").mkdir(parents=True)
    trimesh.creation.box(extents=2 * np.asarray(half)).export(d / "meshes" / f"{name}_visual.obj")
    (d / f"{name}.xml").write_text(
        f'<mujoco><asset><mesh name="{name}_vis" file="meshes/{name}_visual.obj"/></asset>'
        f'<worldbody><body name="object"><geom type="mesh" mesh="{name}_vis" group="1" '
        f'rgba="0.2 0.6 0.5 1"/></body></worldbody></mujoco>')
    return d


def test_load_inputs_and_render(tmp_path):
    pytest.importorskip("trimesh")
    if not _gl_ok():
        pytest.skip("no offscreen GL (run in the manip-sim image with MUJOCO_GL=osmesa)")
    dirs = {"teapot": _box_asset(tmp_path, "teapot"), "mug": _box_asset(tmp_path, "mug")}
    path, sg = _path(), _subgoal()
    tsr = lambda t: {"T0_w": t.T0_w.tolist(), "Tw_e": t.Tw_e.tolist(),       # noqa: E731
                     "Bw": [[float(a), float(b)] for a, b in t.Bw]}
    doc = {"scene": {"asset_dirs": {k: str(v) for k, v in dirs.items()},
                     "table_top_z": -0.05, "table_size": [1.5, 1.5, 0.02]},
           "initial": {"teapot": _pose(0.2).tolist(), "mug": _pose(0.5, 0.1).tolist()},
           "stages": [{"role": "transport_active", "name": "carry", "mover": "teapot",
                       "passive": "mug", "path": tsr(path), "subgoal": tsr(sg),
                       "entry": _pose(0.2, 0, 0.05).tolist()},
                      {"role": "pour", "name": "pour", "mover": "teapot", "passive": "mug",
                       "path": None, "subgoal": None, "error": "compile rejected"}],
           "grasp": {"obj": "teapot", "tcp_pose": _pose(0.2, -0.1, 0.02).tolist()},
           "labels": {"teapot": {"spout_center": [0.03, 0, 0.02]},
                      "mug": {"opening_center": [0, 0, 0.05]}},
           "gripper": None}
    f = tmp_path / "in.json"
    f.write_text(json.dumps(doc))
    kw = rp.load_inputs(f)
    kw["gripper"] = rp.GripperModel(                    # a tetrahedron stand-in
        np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, -0.05]], float),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), "tet")
    m = rp.render_preview(out_dir=tmp_path / "out", **kw)
    kinds = [(i["kind"], i.get("state") or i.get("stage")) for i in m["images"]]
    assert kinds == [("state", "initial"), ("state", "grasp"),
                     ("state", "goal:transport_active"), ("path_boundary", "transport_active"),
                     ("path_boundary", "transport_active")]
    assert [i.get("group") for i in m["images"][3:]] == ["translation", "rotation"]
    assert [s["role"] for s in m["skipped"]] == ["pour"]
    for i in m["images"]:
        assert (tmp_path / "out" / i["file"]).is_file()
    labs = m["images"][0]["labels"]
    assert set(labs) == {"iso", "top"}
    for v in ("iso", "top"):
        assert {l["text"] for l in labs[v]} == {"teapot.spout_center", "mug.opening_center"}
        assert {t["text"] for t in m["images"][0]["tags"][v]} == {"teapot", "mug"}
    # objects are drawn (visual group 1): the state image is not just table + sky
    from PIL import Image
    im = np.asarray(Image.open(tmp_path / "out" / m["images"][0]["file"]).convert("RGB"), float)
    teal = (np.abs(im - [0.2 * 255, 0.6 * 255, 0.5 * 255]).sum(-1) < 90).sum()
    assert teal > 200
    rows = {r["row"]: r for i in m["images"][3:] for r in i["rows"]}
    assert rows["roll"]["token"] == "moderate" and rows["yaw"]["kind"] == "free"
    saved = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert saved["images"][0]["file"] == m["images"][0]["file"]


def test_from_preview_inputs_skips_context_and_gripper_stages():
    from types import SimpleNamespace as NS
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    sym = {"teapot": NS(points={"spout_center": np.array([0.1, 0, 0])})}
    cs = NS(path=_path(), subgoal=_subgoal())
    grasp = NS(role="grasp", emission=NS(name="grasp", active="teapot", passive=None),
               compiled=None, poses=init, symbols=sym, editable=False)
    carry = NS(role="transport_active", emission=NS(name="carry", active="teapot", passive="mug"),
               compiled=cs, poses=init, symbols=sym, editable=True)
    pour = NS(role="pour", emission=NS(name="pour", active="teapot", passive="mug"),
              compiled=None, poses=init, symbols=sym, editable=True)
    initial, stages, labels = rp.from_preview_inputs([grasp, carry, pour])
    assert [s.role for s in stages] == ["transport_active", "pour"]
    assert stages[0].path is cs.path and stages[1].subgoal is None
    assert stages[1].error == "not compiled"
    assert set(labels) == {"teapot"} and "spout_center" in labels["teapot"]
    np.testing.assert_allclose(initial["mug"], init["mug"])


def test_one_sided_row_never_samples_past_its_bound():
    Bw = np.array([[0, 0], [0, 0], [0.03, np.inf], [0, 0], [0, 0], [0, 0]], float)
    path = TSR(T0_w=np.eye(4), Bw=Bw)
    d_rep = np.zeros(6)
    d_rep[2] = 0.05
    (b,), pinned = rp.boundary_samples(path, d_rep)
    assert pinned == ["x", "y", "roll", "pitch", "yaw"]
    assert b.kind == "one-sided" and b.values == pytest.approx([0.03, 0.15])
    assert all(path.contains(T, tol=1e-9) for T in b.poses)
    assert "lower bound" in b.captions[0] and "open above" in b.captions[1]


def test_representative_pose_wraps_free_angles():
    path = _path()
    yaw = lambda a: path.T0_w @ displacement_to_pose(np.array([0, 0, 0.05, 0, 0, a]))  # noqa: E731
    d, _ = rp.representative_pose(path, yaw(np.radians(170)), yaw(np.radians(-170)))
    assert abs(np.cos(d[5]) - np.cos(np.pi)) < 1e-9        # 180 deg, not 0


def test_boundary_base_is_each_stages_own_entry_state(monkeypatch):
    """A stage whose montage is skipped (path None) still advances the chain:
    the next stage's boundary base is that stage's goal state."""
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    a = rp.StageInput("a", "a", "teapot", "mug", None, _subgoal(0.06))
    b = rp.StageInput("b", "b", "teapot", "mug", _path(), _subgoal(0.08))
    seen = []
    monkeypatch.setattr(rp, "_extent", lambda scene, sets, g, gp: (np.zeros(3), 1.0))
    monkeypatch.setattr(rp, "_render", lambda scene, poses, extras, views, px, c, r:
                        seen.append(poses) or {v: (np.zeros((px, px, 3), np.uint8),
                                                   {"pos": np.ones(3), "quat": np.array([1.0, 0, 0, 0])})
                                               for v in views})
    monkeypatch.setattr(rp, "_draw_labels", lambda img, cam, pts, px, tags=(), **kw: ([], []))
    import scripts.render_candidates as rc
    monkeypatch.setattr(rc, "canonical_cameras",
                        lambda c, r: {v: {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([1.0, 0, 0, 0])}
                                      for v in rp.BOUNDARY_VIEWS})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = rp.render_preview(None, init, [a, b], {}, Path(d))
    assert [i.get("stage") for i in m["images"] if i["kind"] == "path_boundary"] == ["b", "b"]
    # b's representative is the midpoint of ITS entry (a's goal, z 0.06) and
    # its goal (z 0.08): 0.07. A stale entry (the initial teapot, z 0) gives 0.04.
    rep = next(i for i in m["images"] if i["kind"] == "path_boundary")["representative"]
    assert rep["displacement"][2] == pytest.approx(0.07)
    assert seen and all(np.allclose(p["mug"], init["mug"]) for p in seen)


# ------------------------------------------------------------------ collisions

def _scene(dirs, table_z=-0.05):
    return rp.PreviewScene(dirs, table_z, (1.5, 1.5, 0.02))


def test_collision_checker_boxes_tolerance_and_table(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    dirs = {"teapot": _box_asset(tmp_path, "teapot"), "mug": _box_asset(tmp_path, "mug")}
    ck = rp.CollisionChecker(_scene(dirs))           # boxes 6 x 6 x 10 cm, table at -0.05
    mug = _pose(0.5, 0.0, 0.0)
    hit = ck.check({"teapot": _pose(0.47, 0.0, 0.0), "mug": mug}, ["teapot"])
    assert [(c["a"], c["b"]) for c in hit] == [("teapot", "mug")]
    assert hit[0]["depth_mm"] > 20
    # 1 mm overlap is resting contact; 1 cm apart is clear
    assert ck.check({"teapot": _pose(0.441, 0.0, 0.0), "mug": mug}, ["teapot"]) == []
    assert ck.check({"teapot": _pose(0.43, 0.0, 0.0), "mug": mug}, ["teapot"]) == []
    # resting on the table (bottom at the table top) is not a collision;
    # 1 cm into it is
    far = _pose(0.0, 0.0, 0.0)
    assert ck.check({"teapot": far, "mug": mug}, ["teapot"]) == []
    sunk = ck.check({"teapot": _pose(0.0, 0.0, -0.01), "mug": mug}, ["teapot"])
    assert [(c["a"], c["b"]) for c in sunk] == [("teapot", "table")]
    # only movers are checked: the mug sunk into the table is not reported
    assert ck.check({"teapot": far, "mug": _pose(0.5, 0, -0.02)}, ["teapot"]) == []


def _cup(root: Path, name="cup", wall=0.006, half=(0.05, 0.05, 0.05)) -> Path:
    """An open-top box with walls `wall` thick: a concave object."""
    import trimesh
    hx, hy, hz = half
    parts = [trimesh.creation.box(extents=[2 * hx, 2 * hy, wall],
                                  transform=_pose(0, 0, -hz + wall / 2))]
    for sx in (-1, 1):
        parts.append(trimesh.creation.box(extents=[wall, 2 * hy, 2 * hz],
                                          transform=_pose(sx * (hx - wall / 2), 0, 0)))
    for sy in (-1, 1):
        parts.append(trimesh.creation.box(extents=[2 * hx - 2 * wall, wall, 2 * hz],
                                          transform=_pose(0, sy * (hy - wall / 2), 0)))
    d = root / name
    (d / "meshes").mkdir(parents=True)
    trimesh.util.concatenate(parts).export(d / "meshes" / f"{name}_visual.obj")
    (d / f"{name}.xml").write_text("<mujoco/>")
    return d


def test_collision_checker_sees_into_a_concave_object(tmp_path):
    """A small block inside a cup's cavity touches nothing (a convex hull
    would call it a collision); through the wall it collides."""
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    dirs = {"block": _box_asset(tmp_path, "block", half=(0.015, 0.015, 0.015)),
            "cup": _cup(tmp_path)}
    ck = rp.CollisionChecker(_scene(dirs, table_z=-1.0))
    cup = _pose(0, 0, 0)
    assert ck.check({"block": _pose(0, 0, 0.0), "cup": cup}, ["block"]) == []
    through = ck.check({"block": _pose(0.045, 0, 0.0), "cup": cup}, ["block"])
    assert [(c["a"], c["b"]) for c in through] == [("block", "cup")]


def test_collision_checker_gripper_and_held_object(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    import trimesh
    dirs = {"teapot": _box_asset(tmp_path, "teapot"), "mug": _box_asset(tmp_path, "mug")}
    b = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
    g = rp.GripperModel(np.asarray(b.vertices), np.asarray(b.faces), "cube")
    ck = rp.CollisionChecker(_scene(dirs), g)
    poses = {"teapot": _pose(0.0, 0, 0), "mug": _pose(0.5, 0, 0)}
    # the gripper inside the object it holds is not a collision
    assert ck.check(poses, ["teapot"], _pose(0.0, 0, 0.03), held="teapot") == []
    # the gripper in the mug is
    hit = ck.check(poses, ["teapot"], _pose(0.5, 0, 0.03), held="teapot")
    assert [(c["a"], c["b"]) for c in hit] == [("gripper", "mug")]
    assert rp.contact_text(hit + [{"a": "gripper", "b": "table", "depth_mm": 3}]) == \
        "gripper collides with mug and table"
    # a mover that is NOT the held object is checked against the held one
    near = {"teapot": _pose(0.0, 0, 0), "mug": _pose(0.03, 0, 0)}
    hit = ck.check(near, ["mug"], None, held="teapot")
    assert [(c["a"], c["b"]) for c in hit] == [("mug", "teapot")]


class _FakeChecker(rp.CollisionChecker):
    """Collides the mover with the mug whenever pred(mover pose) is true."""

    def __init__(self, pred):
        super().__init__(None)
        self.pred = pred

    def check(self, poses, movers, gripper_pose=None, held=None):
        self.calls = getattr(self, "calls", []) + [(list(movers), gripper_pose, held)]
        return ([{"a": m, "b": "mug", "depth_mm": 10.0} for m in movers if self.pred(poses[m])]
                if movers else [])


def _patched_render(monkeypatch, init, stage, checker, grasp=None, style="panels"):
    seen = []
    monkeypatch.setattr(rp, "_extent", lambda scene, sets, g, gp: (np.zeros(3), 1.0))
    monkeypatch.setattr(rp, "_render", lambda scene, poses, extras, views, px, c, r:
                        seen.append(poses) or {v: (np.zeros((px, px, 3), np.uint8),
                                                   {"pos": np.ones(3), "quat": np.array([1.0, 0, 0, 0])})
                                               for v in views})
    monkeypatch.setattr(rp, "_draw_labels", lambda img, cam, pts, px, tags=(), **kw: ([], []))
    import scripts.render_candidates as rc
    monkeypatch.setattr(rc, "canonical_cameras",
                        lambda c, r: {v: {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([1.0, 0, 0, 0])}
                                      for v in rp.BOUNDARY_VIEWS})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = rp.render_preview(None, init, [stage], {}, Path(d), grasp=grasp, style=style,
                              collisions=checker)
        from PIL import Image
        sizes = {i["file"]: Image.open(Path(d) / i["file"]).size for i in m["images"]}
    return m, seen, sizes


def _carry():
    # entry and goal both at z 0.06 over w (x 0.5): the center pose is x 0.5
    return rp.StageInput("carry", "carry", "teapot", "mug", _path(), _subgoal(0.06),
                         entry=_pose(0.5, 0.1, 0.06))


def test_center_free_drops_colliding_panels_as_text(monkeypatch):
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    ck = _FakeChecker(lambda T: T[0, 3] < 0.45)        # only x -10 cm collides
    m, seen, sizes = _patched_render(monkeypatch, init, _carry(), ck)
    tr, rot = [i for i in m["images"] if i["kind"] == "path_boundary"]
    assert (tr["group"], rot["group"]) == ("translation", "rotation")
    assert tr["center_collides"] is False
    assert tr["notes"] == ["x -10.0 cm (free): teapot collides with mug (the planner avoids this)"]
    x = next(r for r in tr["rows"] if r["row"] == "x")
    assert [p["dropped"] for p in x["panels"]] == [True, False]
    assert x["panels"][0]["collisions"][0]["b"] == "mug"
    assert rot["notes"] == []
    # x -10 is not drawn: translation shows 5 panels, rotation 6
    drawn = [p["teapot"][0, 3] for p in seen[2:]]          # after initial + goal
    assert len(drawn) == 11 and not any(v < 0.45 for v in drawn)
    for f, (w, h) in sizes.items():
        assert max(w, h) <= rp.MAX_IMAGE_EDGE, f
    goal = next(i for i in m["images"] if i.get("state") == "goal:carry")
    assert goal["collisions"] == [] and goal["notes"] == []


def test_no_free_center_keeps_and_marks_every_panel(monkeypatch):
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    m, seen, _ = _patched_render(monkeypatch, init, _carry(), _FakeChecker(lambda T: True))
    tr = next(i for i in m["images"] if i.get("group") == "translation")
    assert tr["center_collides"] is True
    srch = tr["representative"]["search"]
    n_both = rp.FREE_POSE_SAMPLES_TRANS + rp.FREE_POSE_SAMPLES_FULL
    assert srch["shown"] == "none_free" and srch["n_checked"] == n_both
    assert srch["pass"] is None
    assert tr["notes"][0] == (f"no collision-free pose found in the region "
                              f"({n_both} samples): teapot collides with mug")
    assert all(not p["dropped"] and p["collides"] for r in tr["rows"] for p in r["panels"])
    # the kept colliding panels are named in text (nothing is drawn on them)
    assert tr["notes"][1] == (
        "center pose in collision; all panels kept; panels in collision: "
        "x -10.0 cm (free) with mug; x +10.0 cm (free) with mug; y -10.0 cm (free) with mug; "
        "y +10.0 cm (free) with mug; z +3.0 cm (lower bound) with mug; "
        "z +10.0 cm (upper bound) with mug")
    goal = next(i for i in m["images"] if i.get("state") == "goal:carry")
    assert goal["search"]["shown"] == "none_free"
    assert goal["notes"][0].startswith("no collision-free pose found in the region")
    np.testing.assert_allclose(goal["poses"]["teapot"], _subgoal(0.06).nominal(), atol=1e-5)


def test_colliding_center_moves_to_the_nearest_free_pose(monkeypatch):
    """The center (x 0.5) collides for x < 0.55: the panels are taken
    around the nearest free FPS sample, and rule (b) drops the colliding
    ones around it."""
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    m, seen, _ = _patched_render(monkeypatch, init, _carry(), _FakeChecker(lambda T: T[0, 3] < 0.55))
    tr = next(i for i in m["images"] if i.get("group") == "translation")
    srch = tr["representative"]["search"]
    assert srch["nominal_collides"] and srch["shown"] == "nearest_free"
    assert srch["pass"] == "translation" and srch["delta"]["rot_deg"] == 0.0
    assert 1 <= srch["n_checked"] <= rp.FREE_POSE_SAMPLES_TRANS
    dc = tr["representative"]["shown_displacement"]
    assert dc[0] >= 0.05 and srch["delta"]["trans_cm"] >= 5.0
    # refined toward the nominal: within a few mm of the boundary (x 0.55)
    assert srch["n_refine"] == rp.FREE_POSE_BISECT and dc[0] < 0.05 + 0.005
    assert tr["center_collides"] is False
    assert tr["notes"][0].startswith("center pose collides with mug; nearest collision-free "
                                     "pose shown (\u0394 ")
    x = next(r for r in tr["rows"] if r["row"] == "x")
    assert x["values"] == pytest.approx([dc[0] - 0.1, dc[0] + 0.1], abs=1e-5)
    assert [p["dropped"] for p in x["panels"]] == [True, False]


def test_goal_draws_the_nearest_free_pose_but_the_chain_stays_nominal(monkeypatch):
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    a = rp.StageInput("a", "a", "teapot", "mug", None, _subgoal(0.06))
    b = rp.StageInput("b", "b", "teapot", "mug", _path(), _subgoal(0.08))
    nominal_a = _subgoal(0.06).nominal()
    # collides within 3 mm of the nominal's position
    ck = _FakeChecker(lambda T: np.linalg.norm(T[:3, 3] - nominal_a[:3, 3]) < 0.003)
    seen = []
    monkeypatch.setattr(rp, "_extent", lambda scene, sets, g, gp: (np.zeros(3), 1.0))
    monkeypatch.setattr(rp, "_render", lambda scene, poses, extras, views, px, c, r:
                        seen.append(poses) or {v: (np.zeros((px, px, 3), np.uint8),
                                                   {"pos": np.ones(3), "quat": np.array([1.0, 0, 0, 0])})
                                               for v in views})
    monkeypatch.setattr(rp, "_draw_labels", lambda img, cam, pts, px, tags=(), **kw: ([], []))
    import scripts.render_candidates as rc
    monkeypatch.setattr(rc, "canonical_cameras",
                        lambda c, r: {v: {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([1.0, 0, 0, 0])}
                                      for v in rp.BOUNDARY_VIEWS})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = rp.render_preview(None, init, [a, b], {}, Path(d), collisions=ck)
    ga = next(i for i in m["images"] if i.get("state") == "goal:a")
    assert ga["search"]["shown"] == "nearest_free" and ga["search"]["n_checked"] >= 1
    assert ga["search"]["pass"] == "translation"
    assert ga["notes"][0].startswith("nominal pose collides with mug; nearest collision-free")
    drawn = np.array(ga["poses"]["teapot"])
    off = np.linalg.norm(drawn[:3, 3] - nominal_a[:3, 3])
    assert 0.003 <= off < 0.005                                 # bisected to the boundary
    assert _subgoal(0.06).contains(drawn, tol=1e-4)            # inside the subgoal region
    assert ga["collisions"][0]["b"] == "mug"                    # recorded at the nominal
    # b starts from a's NOMINAL (z 0.06): its center is 0.07, not shifted
    rep = next(i for i in m["images"] if i["kind"] == "path_boundary")["representative"]
    assert rep["displacement"][2] == pytest.approx(0.07)


def test_search_box_clips_free_rows_to_the_display_range():
    lo, hi = rp.search_box(_path(), np.array([0.01, 0.0, 0.05, 0, 0, 1.0]))
    assert lo[:3] == pytest.approx([0.01 - rp.FREE_TRANS_SHOW_M, -rp.FREE_TRANS_SHOW_M, 0.03])
    assert hi[:3] == pytest.approx([0.01 + rp.FREE_TRANS_SHOW_M, rp.FREE_TRANS_SHOW_M, 0.10])
    assert (lo[5], hi[5]) == pytest.approx((1.0 - np.pi / 2, 1.0 + np.pi / 2))
    assert (lo[3], hi[3]) == pytest.approx((-np.radians(15), np.radians(15)))


def test_fps_samples_are_seeded_inside_the_box_and_nearest_first():
    lo, hi = rp.search_box(_path(), np.array([0, 0, 0.05, 0, 0, 0]))
    d0 = np.array([0, 0, 0.05, 0, 0, 0])
    a = rp.fps_displacements(lo, hi, d0)
    b = rp.fps_displacements(lo, hi, d0)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (rp.FREE_POSE_SAMPLES, 6)
    assert np.all(a >= lo - 1e-12) and np.all(a <= hi + 1e-12)
    dist = rp.pose_distance(a, d0)
    assert np.all(np.diff(dist) >= -1e-12)
    # the metric: 1 rad weighs ROT_DIST_M_PER_RAD meters
    assert rp.pose_distance(np.array([0, 0, 0, 1.0, 0, 0]), np.zeros(6))[0] == \
        pytest.approx(rp.ROT_DIST_M_PER_RAD)


def test_nearest_free_pose_pinned_region_has_none():
    Bw = np.zeros((6, 2))
    tsr = TSR(T0_w=np.eye(4), Bw=Bw)
    d, rec = rp.nearest_free_pose(tsr, np.zeros(6), lambda T: [{"a": "t", "b": "m", "depth_mm": 5}])
    # every row pinned: no pose but d0 exists, so nothing is sampled
    assert rec["shown"] == "none_free" and rec["n_checked"] == 0
    np.testing.assert_array_equal(d, np.zeros(6))
    d, rec = rp.nearest_free_pose(tsr, np.zeros(6), lambda T: [])
    assert rec == {"nominal_collides": False, "contacts": [], "shown": "nominal",
                   "pass": None, "n_checked": 0, "n_refine": 0, "delta": None}


def test_full_pass_skipped_when_rotation_is_pinned_and_duplicates_dropped():
    hit = [{"a": "t", "b": "m", "depth_mm": 5}]
    Bw = np.zeros((6, 2))
    Bw[2] = [0.0, 0.1]                                   # only z open
    tsr = TSR(T0_w=np.eye(4), Bw=Bw)
    _, rec = rp.nearest_free_pose(tsr, np.zeros(6), lambda T: hit)
    assert rec["n_checked"] == rp.FREE_POSE_SAMPLES_TRANS and rec["pass"] is None
    # a fully pinned box yields no sample (d0 itself is known); an all-pinned
    # box away from d0 yields that one point once
    z = np.zeros(6)
    assert rp.fps_displacements(z, z, z).shape == (0, 6)
    p = np.full(6, 0.01)
    np.testing.assert_allclose(rp.fps_displacements(p, p, z), [p])


def test_caller_contacts_are_used_for_the_nominal():
    """The nominal is not checked a second time: contacts0 decides it."""
    hit = [{"a": "teapot", "b": "mug", "depth_mm": 5.0}]
    calls = []
    d, rec = rp.nearest_free_pose(_path(), np.array([0.0, 0.0, 0.05, 0, 0, 0]),
                                  lambda T: calls.append(T) or [], contacts0=hit)
    assert rec["nominal_collides"] and rec["shown"] == "nearest_free"
    assert len(calls) == 1 + rp.FREE_POSE_BISECT         # one FPS sample + bisection


def test_translation_pass_first_rotation_only_when_needed():
    hit = [{"a": "teapot", "b": "mug", "depth_mm": 5.0}]
    path = _path()
    d0 = np.array([0.0, 0.0, 0.05, 0, 0, 0])
    # a lift clears it: found in pass (a), rotation untouched
    d, rec = rp.nearest_free_pose(path, d0, lambda T: hit if T[2, 3] < 0.07 else [])
    assert rec["pass"] == "translation" and rec["n_checked"] <= rp.FREE_POSE_SAMPLES_TRANS
    np.testing.assert_array_equal(d[3:], d0[3:])
    assert 0.07 <= d[2] < 0.075                           # bisected to the boundary
    assert rp.search_note(rec, "nominal")[0].endswith(f"cm)")
    # only a tilt clears it: pass (a) spends its budget, pass (b) finds it
    upright = lambda T: hit if T[2, 2] > 1 - 1e-9 else []          # noqa: E731
    d, rec = rp.nearest_free_pose(path, d0, upright)
    assert rec["pass"] == "full"
    assert rec["n_checked"] > rp.FREE_POSE_SAMPLES_TRANS
    assert rec["delta"]["rot_deg"] > 0
    assert rp.search_note(rec, "nominal")[0].endswith("; needed rotation)")


def test_leader_segment_from_dot_edge_to_label_box():
    seg = rp.leader_segment((100.0, 100.0), (130.0, 90.0, 200.0, 110.0), gap=11.0)
    assert seg[0] == pytest.approx((111.0, 100.0)) and seg[1] == pytest.approx((130.0, 100.0))
    assert rp.leader_segment((100.0, 100.0), (105.0, 90.0, 200.0, 110.0), gap=11.0) is None


def test_all_dropped_group_is_text_only(monkeypatch):
    """Every translation panel collides, the center does not: no translation
    image, its lines kept in the manifest."""
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    rep = _pose(0.5, 0.1, 0.06)

    def pred(T):
        moved = np.linalg.norm(T[:3, 3] - rep[:3, 3]) > 1e-6
        rotated = not np.allclose(T[:3, :3], rep[:3, :3])
        return moved and not rotated
    m, _, _ = _patched_render(monkeypatch, init, _carry(), _FakeChecker(pred))
    groups = [i["group"] for i in m["images"] if i["kind"] == "path_boundary"]
    assert groups == ["rotation"]
    (t,) = m["path_text_only"]
    assert t["group"] == "translation" and len(t["notes"]) == 6


def test_collisions_off_records_none(monkeypatch):
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    m, _, _ = _patched_render(monkeypatch, init, _carry(), False)
    assert m["collision_check"] is None
    tr = next(i for i in m["images"] if i.get("group") == "translation")
    assert tr["center_collides"] is None
    assert all(p["collides"] is None and not p["dropped"] for r in tr["rows"] for p in r["panels"])


# ------------------------------------------------------------------ labels, size

def test_merge_anchors_joins_close_labels_only():
    out = rp.merge_anchors([("mug.rim_center", (100.0, 100.0)),
                            ("mug.opening_center", (104.0, 103.0)),
                            ("teapot.spout_center", (130.0, 100.0))])
    assert [t for t, _ in out] == ["mug.rim_center / mug.opening_center", "teapot.spout_center"]
    assert out[0][1] == pytest.approx((102.0, 101.5))
    # different objects never merge, however close
    out = rp.merge_anchors([("teapot.handle_center", (100.0, 100.0)),
                            ("mug.rim_center", (101.0, 100.0))])
    assert [t for t, _ in out] == ["teapot.handle_center", "mug.rim_center"]


def test_tag_goes_below_when_the_spot_above_is_covered():
    # the mug's top-center spot lies inside the teapot's box: tag below
    assert rp._tag_anchor((100, 200, 160, 300), [(80, 120, 200, 210)], 448) == (130.0, 314.0)
    assert rp._tag_anchor((100, 200, 160, 300), [], 448) == (130.0, 186.0)
    # above and below both covered: beside it
    assert rp._tag_anchor((100, 200, 160, 300), [(80, 120, 200, 210), (80, 290, 200, 400)],
                          448) == (60.0, 250.0)
    # covered on every side: no tag (it would sit on another body)
    assert rp._tag_anchor((100, 200, 160, 300), [(0, 0, 448, 448)], 448) is None


def test_grid_and_fit_stay_under_the_api_edge():
    from PIL import Image
    tiles = [Image.new("RGB", (rp.PANEL_PX, rp.PANEL_PX)) for _ in range(6)]
    g = rp._grid(tiles, rp.PATH_COLS, "t")
    assert g.size[0] == rp.PATH_COLS * rp.PANEL_PX and max(g.size) <= rp.MAX_IMAGE_EDGE
    big = rp._fit(Image.new("RGB", (2400, 1200)))
    assert big.size == (rp.MAX_IMAGE_EDGE, 784)
    state = rp._grid([Image.new("RGB", (rp.STATE_PX, rp.STATE_PX))] * 2, 2, "s")
    assert state.size[0] == 2 * rp.STATE_PX                # not scaled: 1536 <= 1568


def test_goal_search_leaves_the_gripper_when_the_mover_is_not_held(monkeypatch):
    """The gripper holds the teapot; a stage moving the mug draws and checks
    the gripper where it is, never attached to the mug."""
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    tcp = _pose(0.2, -0.1, 0.05)
    nominal = _subgoal(0.06).nominal()
    ck = _FakeChecker(lambda T: np.allclose(T, nominal, atol=1e-9))
    st = rp.StageInput("move_mug", "move mug", "mug", "teapot", None, _subgoal(0.06))
    m, _, _ = _patched_render(monkeypatch, init, st, ck, grasp=rp.GraspInput("teapot", tcp))
    goal = next(i for i in m["images"] if i.get("state") == "goal:move_mug")
    assert goal["search"]["shown"] == "nearest_free"
    np.testing.assert_allclose(goal["gripper_pose"], tcp, atol=1e-5)
    mug_calls = [c for c in ck.calls if c[0] == ["mug"]]
    assert mug_calls and all(np.allclose(g, tcp) for _, g, _ in mug_calls)
    assert all(h == "teapot" for _, _, h in mug_calls)


def test_overlay_caption_lists_kept_bounds_only(monkeypatch):
    init = {"teapot": _pose(0.2), "mug": _pose(0.5, 0.1)}
    ck = _FakeChecker(lambda T: T[0, 3] < 0.45)          # only x -10 cm collides
    m, _, _ = _patched_render(monkeypatch, init, _carry(), ck, style="overlay")
    tr = next(i for i in m["images"] if i.get("group") == "translation")
    x = next(r for r in tr["rows"] if r["row"] == "x")
    assert x["captions"] == ["x free: +10.0 cm"]          # the -10 cm bound is gone
    assert [p["dropped"] for p in x["panels"]] == [True, False]


def test_place_labels_can_return_request_indices():
    from PIL import Image

    from scripts.render_stage_frames import LabelRequest, place_labels
    reqs = [LabelRequest((100.0, 100.0), "low", (0, 0, 0), 0.1),
            LabelRequest((300.0, 300.0), "high", (0, 0, 0), 5.0)]
    plain = place_labels(Image.new("RGB", (448, 448)), list(reqs), [], px=448)
    pairs = place_labels(Image.new("RGB", (448, 448)), list(reqs), [], px=448, with_index=True)
    assert [i for i, _ in pairs] == [1, 0]                # placement order: priority first
    assert [b for _, b in pairs] == plain
