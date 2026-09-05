"""scripts/render_stage_frames.py: the drawn w IS the compiled w, the
drawn triads ARE the licensed #3 alphabet, and the drawn pixels are the
projection of the compiler's basis. Geometry tests run without GL (they
need mujoco only for the camera quaternion math); the render smoke test
skips when no offscreen GL context is available or the visual meshes
are not converted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manip_sim.compile_tsr import compile_stage
from manip_sim.frames import load_symbols
from manip_sim.scene import load_scene
from manip_sim.tsr import pose_from_pos_quat_wxyz
from manip_sim.vlm import (CANONICAL_DIRS, StageEmission, TSRSpec, TransTerm,
                           RotRow, Vocabulary)
from scripts.render_candidates import canonical_cameras, project
from scripts.render_stage_frames import (VIEWS, DrawnFrame, compute_stage_frames,
                                         draw_frame, render, scene_extent)
from scripts.emit_constraints import STAGES

from test_selection import ASSET_DIRS, _HAVE_POOLS, _pour_tea_selections

pytestmark = pytest.mark.skipif(not _HAVE_POOLS,
                                reason="candidate pools not written")
RADIUS = 0.33      # a plausible two-object scene extent, m


def _setup():
    scene = load_scene()
    symbols = {n: load_symbols(d) for n, d in scene.asset_dirs.items()}
    poses = {n: pose_from_pos_quat_wxyz(*pq)
             for n, pq in scene.fixed_poses().items()}
    sels, objects, role_index, _ = _pour_tea_selections()
    frames = compute_stage_frames(STAGES, sels, objects, role_index,
                                  scene.asset_dirs, symbols, poses, RADIUS)
    return scene, symbols, poses, frames


def _free_emission(stage):
    free = TSRSpec(rot=tuple(RotRow(None, "free", None, None, r)
                             for r in ("roll", "pitch", "yaw")),
                   trans=(TransTerm("free"),))
    return StageEmission(stage=stage.index, name=stage.name,
                         active=stage.active, passive=stage.passive,
                         path_tsr=free, subgoal_tsr=free, verify="")


def test_drawn_w_is_the_compiled_w_for_every_stage():
    """The w triad's origin and axes equal compile_stage's T0_w — same
    _canonical_w, same poses, same call-#2 point — and the recorded x
    route is the one the compiler reports in its provenance note."""
    from manip_sim.selection import selected_points
    scene, symbols, poses, frames = _setup()
    sels, objects, role_index, _ = _pour_tea_selections()
    for stage, role in STAGES:
        sf = frames[role]
        w_point, e_point = selected_points(role, stage, sels, objects,
                                           role_index, scene.asset_dirs, symbols)
        cs = compile_stage(_free_emission(stage), symbols, poses,
                           w_point=w_point, e_point=e_point)
        w = next(f for f in sf.frames if f.label == "w")
        np.testing.assert_allclose(w.origin, cs.path.T0_w[:3, 3], atol=1e-12)
        for k, col in zip("xyz", range(3)):
            np.testing.assert_allclose(w.axes[k], cs.path.T0_w[:3, col],
                                       atol=1e-12)
        assert f"x = {sf.w_x_route}" in cs.notes[0]
        np.testing.assert_allclose(sf.w_point_body, w_point, atol=1e-5)


def test_fallback_x_is_labeled_and_never_presented_as_a_mug_direction():
    """The sim mug has no front: w.x comes from the compiler's fallback
    ladder (the teapot's lateral at entry). The render must say so in
    the arrow label and the manifest, and the mug's own triad must show
    only what the mug licenses (+up) — not the borrowed x."""
    _, _, _, frames = _setup()
    for role in ("transport_active", "pour"):
        sf = frames[role]
        assert sf.w_object == "mug" and sf.w_fallback
        assert "teapot lateral" in sf.w_x_route
        w = next(f for f in sf.frames if f.label == "w")
        assert w.triad[0][1] == "w.x (fallback)"
        owner = next(t for t in sf.triads if t["role_in_stage"] == "w_owner")
        assert owner["object"] == "mug"
        assert owner["directions"] == ["+up", "-up"]
        assert owner["axes_drawn"] == ["up_axis"]
    sf = frames["grasp"]
    assert sf.w_object == "teapot" and not sf.w_fallback
    w = next(f for f in sf.frames if f.label == "w")
    assert w.triad[0][1] == "w.x"


def test_drawn_triads_are_exactly_the_licensed_direction_alphabet():
    """Picture == accept set: each drawn triad's directions equal
    Vocabulary.canonical_directions for that object, and its axes are
    the columns those directions map to. Two triads on two-object
    stages (w owner + active), one on the grasp."""
    _, symbols, _, frames = _setup()
    vocab = Vocabulary.from_symbols(symbols)
    for role, sf in frames.items():
        roles = [t["role_in_stage"] for t in sf.triads]
        assert roles == (["w_owner", "active"] if sf.passive else ["w_owner"])
        for t in sf.triads:
            assert tuple(t["directions"]) == vocab.canonical_directions(t["object"])
    active = next(t for t in frames["pour"].triads if t["role_in_stage"] == "active")
    assert active["object"] == "teapot"
    assert tuple(active["directions"]) == CANONICAL_DIRS
    assert active["axes_drawn"] == ["front_axis", "lateral_axis", "up_axis"]
    # the active triad sits at the selected feature point (spout tip, #3)
    from manip_sim.selection import load_pool
    pool = load_pool(ASSET_DIRS["teapot"])
    assert active["point_body"] == pytest.approx(pool[3]["xyz"], abs=1e-5)


def test_drawn_pixels_are_the_projection_of_the_compiler_basis():
    """draw_frame's returned endpoints equal project() of origin and
    origin + length * axis for the camera actually used, with no
    culling and no rescaling — what the model sees is the basis."""
    from PIL import Image
    _, _, _, frames = _setup()
    w = next(f for f in frames["pour"].frames if f.label == "w")
    cam = canonical_cameras(w.origin, RADIUS)["iso"]
    ends = draw_frame(Image.new("RGB", (1024, 1024)), cam, w, px=1024)
    assert set(ends) == {"x", "y", "z"}
    for k in "xyz":
        pts = np.vstack([w.origin, w.origin + w.length * w.axes[k]])
        uv, _ = project(pts, cam, px=1024)
        np.testing.assert_allclose(np.array(ends[k]), uv, atol=1e-9)


def _gl_available() -> bool:
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
        mujoco.Renderer(m, 16, 16).close()
        return True
    except Exception:
        return False


_HAVE_MESHES = all((d / "meshes" / f"{n}_visual.obj").exists()
                   for n, d in ASSET_DIRS.items())


@pytest.mark.skipif(not (_HAVE_MESHES and _gl_available()),
                    reason="needs converted visual meshes + offscreen GL")
def test_render_writes_three_views_per_stage_and_a_manifest(tmp_path):
    scene, symbols, poses, _ = _setup()
    center, radius = scene_extent(scene, poses)
    sels, objects, role_index, _ = _pour_tea_selections()
    frames = compute_stage_frames(STAGES, sels, objects, role_index,
                                  scene.asset_dirs, symbols, poses, radius)
    manifest = render(scene, frames, tmp_path, center, radius)
    assert manifest["views"] == list(VIEWS) and len(VIEWS) == 3
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    for _, role in STAGES:
        rec = manifest["roles"][role]
        assert set(rec["views"]) == set(VIEWS)
        for p in rec["views"].values():
            assert Path(p).exists() and Path(p).stat().st_size > 0
        assert set(rec["w"]) == {"object", "point_body", "candidate_id",
                                 "x_route", "fallback"}
    assert manifest["roles"]["pour"]["w"]["candidate_id"] == 2
    assert manifest["roles"]["pour"]["w"]["fallback"] is True
