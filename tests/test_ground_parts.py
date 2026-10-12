"""scripts/ground_parts.py end to end on a synthetic asset, with the
MuJoCo render replaced by analytic orthographic projections (the lift
only needs uv + visibility per view)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ground_parts as gp  # noqa: E402

from manip_sim.frames import load_symbols  # noqa: E402
from manip_sim.selection import load_pool  # noqa: E402


def _write_box_asset(d: Path, name="widget", hx=0.04, hy=0.04, hz=0.06):
    (d / "meshes").mkdir(parents=True)
    V = [(sx * hx, sy * hy, sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    F = [(1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
         (2, 4, 8), (2, 8, 6), (4, 3, 7), (4, 7, 8), (3, 1, 5), (3, 5, 7)]
    lines = [f"v {x} {y} {z}" for x, y, z in V] + [f"f {a} {b} {c}" for a, b, c in F]
    (d / "meshes" / f"{name}_visual.obj").write_text("\n".join(lines) + "\n")
    (d / f"{name}.xml").write_text("<mujoco/>\n")
    (d / "frames.json").write_text(json.dumps({"points": {}, "axes": {
        "up_axis": {"xyz": [0, 0, 1]}}, "quantities": {}}))


def _fake_render(name, obj_dir, V, out_views, px=gp.VIEW_PX):
    """Six axis-aligned orthographic views, everything visible on the
    near side only (depth test by sign of the view axis)."""
    axes = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0],
            "-y": [0, -1, 0], "top": [0, 0, 1], "bottom": [0, 0, -1]}
    scale = px / (2 * 0.08)

    def project_samples(P):
        uv_by, vis_by = {}, {}
        for v, a in axes.items():
            a = np.asarray(a, float)
            e1 = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.5 else np.array([1.0, 0, 0])
            e2 = np.cross(a, e1)
            uv_by[v] = np.column_stack([px / 2 + scale * (P @ e1),
                                        px / 2 + scale * (P @ e2)])
            # visible iff on the face nearest the camera (box: extremal along a)
            d = P @ a
            vis_by[v] = d >= d.max() - 1e-6
        return uv_by, vis_by

    return {v: {} for v in axes}, project_samples, {}


def test_ground_object_oracle_writes_loadable_dir(tmp_path, monkeypatch):
    asset = tmp_path / "assets" / "widget"
    _write_box_asset(asset)
    monkeypatch.setattr(gp, "render_depth_views", _fake_render)
    out = tmp_path / "grounding"
    res = gp.ground_object("widget", asset, ("top", "base"), out, "oracle",
                           None, write=True)
    assert res == {"top": "blob", "base": "blob"}      # flat faces: blobs
    g = out / "widget"
    sym = load_symbols(g)
    assert {"top_center", "base_center"} <= set(sym.points)
    assert sym.points["top_center"][2] > 0.05 and sym.points["base_center"][2] < -0.05
    assert sym.axes["top_axis"] @ [0, 0, 1] > 0.9
    pool = load_pool(g)
    parts = {c.get("part") for c in pool.values()}
    assert {"top", "base"} <= parts
    assert (g / "widget.xml").is_symlink() and (g / "meshes").is_symlink()
    prov = json.loads((g / "frames.json").read_text())["provenance"]
    assert prov["mask_provider"] == "oracle" and prov["parts"]["top"]["n_points"] > 30


def test_scene_grounding_override(tmp_path):
    from manip_sim.scene import load_scene
    asset = tmp_path / "assets" / "widget"
    _write_box_asset(asset)
    scene_json = tmp_path / "s.json"
    scene_json.write_text(json.dumps({
        "name": "s", "task": "t", "table": {"size": [1, 1, 0.05], "top_z": 0.8},
        "objects": {"widget": {"asset": str(asset), "placement": {"xy": [0, 0]}}}}))
    g = tmp_path / "grounding"
    assert load_scene(scene_json, g).asset_dirs["widget"] == asset      # nothing grounded yet
    (g / "widget").mkdir(parents=True)
    (g / "widget" / "frames.json").write_text("{}")
    s = load_scene(scene_json, g)
    assert s.asset_dirs["widget"] == g / "widget"
    assert Path(s.object_xmls["widget"]) == asset / "widget.xml"      # spawn xml stays authored


# ------------------------------------------------------- the hardware seam's contract

def test_points_carry_depth_check_and_winding(tmp_path, monkeypatch):
    asset = tmp_path / "assets" / "widget"
    _write_box_asset(asset)
    monkeypatch.setattr(gp, "render_depth_views", _fake_render)
    gp.ground_object("widget", asset, ("top", "base"), tmp_path / "g", "oracle", None, write=True)
    pts = json.loads((tmp_path / "g" / "widget" / "frames.json").read_text())["points"]
    for name, p in pts.items():
        assert p["depth_check"] in ("interior", "skip"), name
        assert p["depth_check"] == ("interior" if abs(p["winding"]) >= 0.5 else "skip")
    # this fixture box is a sampling stand-in, not a solid: its faces are
    # inconsistently wound and it is not closed, so the winding values are
    # not meaningful -- and provenance says so, which is the point here
    ds = json.loads((tmp_path / "g" / "widget" / "frames.json").read_text())[
        "provenance"]["depth_semantics"]
    assert ds["inconsistent_edges"] == 4 and ds["boundary_edges"] == 6
    assert ds["inside_threshold"] == 0.5


def test_up_sigma_reaches_up_and_lateral(tmp_path, monkeypatch):
    """up_sigma_deg is plumbed into symbols_from_parts, so lateral_axis's
    sigma (hypot(up, front)) is computed from it, not from a 0.0 that a
    caller patched over afterwards. Default stays 0.0 (sim unchanged)."""
    asset = tmp_path / "assets" / "widget"
    _write_box_asset(asset)
    monkeypatch.setattr(gp, "render_depth_views", _fake_render)
    gp.ground_object("widget", asset, ("top",), tmp_path / "a", "oracle", None, write=True,
                     front=np.array([1.0, 0, 0]), front_sigma_deg=20.0, up_sigma_deg=2.07)
    ax = json.loads((tmp_path / "a" / "widget" / "frames.json").read_text())["axes"]
    assert ax["up_axis"]["sigma_deg"] == 2.07 and ax["front_axis"]["sigma_deg"] == 20.0
    assert ax["lateral_axis"]["sigma_deg"] == round(float(np.hypot(2.07, 20.0)), 2)
    gp.ground_object("widget", asset, ("top",), tmp_path / "b", "oracle", None, write=True)
    assert json.loads((tmp_path / "b" / "widget" / "frames.json").read_text())[
        "axes"]["up_axis"]["sigma_deg"] == 0.0


def test_grounding_error_is_typed_not_systemexit(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "render_depth_views", _fake_render)
    empty = tmp_path / "assets" / "nomesh"
    empty.mkdir(parents=True)
    with pytest.raises(gp.GroundingError, match="missing"):
        gp.ground_object("nomesh", empty, ("top",), tmp_path / "g", "oracle", None, write=False)
    asset = tmp_path / "assets" / "widget"
    _write_box_asset(asset)
    with pytest.raises(gp.GroundingError, match="no masks"):
        gp.ground_object("widget", asset, ("top",), tmp_path / "g", "masks",
                         tmp_path / "masks", write=False)
    with pytest.raises(gp.GroundingError, match="unknown provider"):
        gp.ground_object("widget", asset, ("top",), tmp_path / "g", "psychic", None, write=False)
    assert issubclass(gp.GroundingError, RuntimeError)
    assert not issubclass(gp.GroundingError, SystemExit)


def test_cli_still_exits_with_the_message(tmp_path, monkeypatch):
    empty = tmp_path / "assets" / "widget"
    empty.mkdir(parents=True)
    scene_json = tmp_path / "s.json"
    scene_json.write_text(json.dumps({
        "name": "s", "task": "t", "table": {"size": [1, 1, 0.05], "top_z": 0.8},
        "objects": {"widget": {"asset": str(empty), "placement": {"xy": [0, 0]}}}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"object_parts": {"widget": ["top"]}}))
    monkeypatch.setattr(sys, "argv", ["ground_parts.py", "--stage-plan", str(plan),
                                      "--scene", str(scene_json), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as e:
        gp.main()
    assert "missing" in str(e.value.code)
