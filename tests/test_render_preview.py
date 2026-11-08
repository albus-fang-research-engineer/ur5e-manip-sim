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
                     ("state", "goal:transport_active"), ("path_boundary", "transport_active")]
    assert [s["role"] for s in m["skipped"]] == ["pour"]
    for i in m["images"]:
        assert (tmp_path / "out" / i["file"]).is_file()
    labs = m["images"][0]["labels"]
    assert set(labs) == {"iso", "top"}
    for v in ("iso", "top"):
        assert {l["text"] for l in labs[v]} == {"teapot.spout_center", "mug.opening_center"}
    # objects are drawn (visual group 1): the state image is not just table + sky
    from PIL import Image
    im = np.asarray(Image.open(tmp_path / "out" / m["images"][0]["file"]).convert("RGB"), float)
    teal = (np.abs(im - [0.2 * 255, 0.6 * 255, 0.5 * 255]).sum(-1) < 90).sum()
    assert teal > 200
    rows = {r["row"]: r for r in m["images"][3]["rows"]}
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
    monkeypatch.setattr(rp, "_draw_labels", lambda img, cam, pts, px: [])
    import scripts.render_candidates as rc
    monkeypatch.setattr(rc, "canonical_cameras",
                        lambda c, r: {v: {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([1.0, 0, 0, 0])}
                                      for v in rp.BOUNDARY_VIEWS})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = rp.render_preview(None, init, [a, b], {}, Path(d))
    assert [i.get("stage") for i in m["images"] if i["kind"] == "path_boundary"] == ["b"]
    # b's representative is the midpoint of ITS entry (a's goal, z 0.06) and
    # its goal (z 0.08): 0.07. A stale entry (the initial teapot, z 0) gives 0.04.
    rep = next(i for i in m["images"] if i["kind"] == "path_boundary")["representative"]
    assert rep["displacement"][2] == pytest.approx(0.07)
    assert seen and all(np.allclose(p["mug"], init["mug"]) for p in seen)
