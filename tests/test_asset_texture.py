"""Colour textures on converted assets (convert_asset.wire_texture) and the
two model builders that must carry them: render_candidates.build_model
(canonical views for SAM3 and call #2 menus; textures resolve through the
injected texturedir) and render_stage_frames.build_scene_model (call #3's
framed arm; copies texture + material with absolutized files).

The sim assets rendered flat grey (rgba 0.8 0.8 0.85) although their
visual OBJs carry UVs: MuJoCo ignores .mtl, so SAM3 on canonical views was
segmenting an untextured shape. Built on a synthetic quad whose texture is
half red, half blue, so the render check is a colour, not a judgment."""

import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.convert_asset import wire_texture  # noqa: E402

PIL = pytest.importorskip("PIL")


def _asset(tmp_path: Path, uv: bool = True) -> Path:
    """A converted-layout asset: a closed 4 x 4 x 1 cm box (MuJoCo needs
    volume) with planar UVs u = x, v = y across the top face, visual geom in
    group 1 with the converter's default rgba, one (unused) collision mesh."""
    from PIL import Image
    d = tmp_path / "quad"
    (d / "meshes").mkdir(parents=True)
    V = [(x, y, z) for z in (-0.005, 0.005) for x, y in
         ((-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02))]
    F = [(1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
    obj = [f"v {x} {y} {z}" for x, y, z in V]
    if uv:
        obj += [f"vt {(x + 0.02) / 0.04} {(y + 0.02) / 0.04}" for x, y, _ in V]
        obj += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in F]
    else:
        obj += [f"f {a} {b} {c}" for a, b, c in F]
    (d / "meshes" / "quad_visual.obj").write_text("\n".join(obj) + "\n")
    (d / "meshes" / "quad_col_0.obj").write_text(
        "\n".join([f"v {x} {y} {z}" for x, y, z in V]
                  + [f"f {a} {b} {c}" for a, b, c in F]) + "\n")
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = (220, 20, 20)                  # u < 0.5 red
    img[:, 32:] = (20, 20, 220)                  # u > 0.5 blue
    Image.fromarray(img).save(tmp_path / "texture_map.png")
    (d / "quad.xml").write_text(textwrap.dedent("""\
        <mujoco model="quad">
          <asset>
            <mesh file="meshes/quad_visual.obj" name="quad_visual_mesh" />
            <mesh file="meshes/quad_col_0.obj" name="quad_col_mesh_0" />
          </asset>
          <worldbody>
            <body>
              <body name="object">
                <geom type="mesh" mesh="quad_visual_mesh" group="1" conaffinity="0" contype="0" rgba="0.8 0.8 0.85 1" mass="0.0001" />
                <geom type="mesh" mesh="quad_col_mesh_0" group="0" rgba="0 1 0 0.0" density="0" mass="0.01" />
              </body>
            </body>
          </worldbody>
        </mujoco>
        """))
    return d


def test_wire_texture_puts_material_on_visual_geom_only(tmp_path):
    d = _asset(tmp_path)
    col_before = [g.attrib for g in ET.parse(d / "quad.xml").iter("geom")
                  if g.get("group") == "0"]
    dst = wire_texture(d / "quad.xml", tmp_path / "texture_map.png")
    assert dst == d / "meshes" / "quad_texture.png" and dst.exists()
    root = ET.parse(d / "quad.xml").getroot()
    tex = root.find("asset/texture")
    assert tex.get("file") == "meshes/quad_texture.png"      # relative, like meshes
    assert root.find("asset/material").get("texture") == tex.get("name")
    vis = [g for g in root.iter("geom") if g.get("group") == "1"]
    assert len(vis) == 1 and vis[0].get("material") == "quad_mat"
    assert "rgba" not in vis[0].attrib
    assert [g.attrib for g in root.iter("geom") if g.get("group") == "0"] == col_before
    # idempotent: a second wire replaces, never duplicates
    wire_texture(d / "quad.xml", tmp_path / "texture_map.png")
    root = ET.parse(d / "quad.xml").getroot()
    assert len(root.findall("asset/texture")) == len(root.findall("asset/material")) == 1


def test_wire_texture_refuses_a_mesh_without_uvs(tmp_path):
    d = _asset(tmp_path, uv=False)
    with pytest.raises(SystemExit, match="no vt"):
        wire_texture(d / "quad.xml", tmp_path / "texture_map.png")


_RENDER = textwrap.dedent("""\
    import sys, numpy as np
    sys.path.insert(0, {root!r})
    from scripts.render_candidates import build_model
    from pathlib import Path
    import mujoco
    cams = {{"top": {{"pos": [0, 0, 0.12], "quat": [1, 0, 0, 0]}}}}
    m = build_model("quad", Path({d!r}), cams, px=64)
    r = mujoco.Renderer(m, 64, 64); dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
    r.update_scene(dd, camera="top"); im = r.render().astype(int)
    row = im[32]
    print(int(row[20, 0] - row[20, 2]), int(row[44, 2] - row[44, 0]))
""")


def _render(d: Path) -> tuple[int, int] | None:
    p = subprocess.run([sys.executable, "-c", _RENDER.format(root=str(ROOT), d=str(d))],
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return None
    return tuple(int(x) for x in p.stdout.split()[-2:])


def test_build_model_renders_the_texture(tmp_path):
    """Grey before wiring; red on the u<0.5 half (x<0) and blue on the other after,
    through build_model's texturedir (run in a subprocess: a GL failure must
    not abort the session)."""
    pytest.importorskip("mujoco")
    d = _asset(tmp_path)
    before = _render(d)
    if before is None:
        pytest.skip("no working MuJoCo GL backend")
    assert abs(before[0]) < 20 and abs(before[1]) < 20          # neutral grey
    wire_texture(d / "quad.xml", tmp_path / "texture_map.png")
    red, blue = _render(d)
    assert red > 80 and blue > 80, (red, blue)


def test_stage_frames_scene_model_compiles_with_a_textured_object(tmp_path):
    """GL-free: build_scene_model copies the object's texture + material (the
    geom references the material by name, so without them it cannot compile)."""
    pytest.importorskip("mujoco")
    from scripts.render_stage_frames import build_scene_model
    d = _asset(tmp_path)
    wire_texture(d / "quad.xml", tmp_path / "texture_map.png")
    scene = SimpleNamespace(
        table_top_z=0.8, table_size=(1.2, 1.2, 0.05),
        objects={"quad": SimpleNamespace(xml=d / "quad.xml", asset=d)},
        fixed_poses=lambda: {"quad": (np.array([0, 0, 0.81]), np.array([1.0, 0, 0, 0]))})
    m = build_scene_model(scene, {})
    assert m.ntex >= 2 and m.nmat >= 1      # skybox + the object's texture
