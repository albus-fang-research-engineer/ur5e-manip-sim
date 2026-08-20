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
