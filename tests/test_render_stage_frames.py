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
from scripts.render_stage_frames import (END_ON_DEG, VIEWS, LabelRequest,
                                         compose_view, compute_stage_frames,
                                         draw_frame, place_labels, render,
                                         scene_extent, _area)
from manip_sim.compile_tsr import ALIGN_TOL_RAD
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
    assert w.triad == ()           # every w axis merged into the teapot's


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
    sf = frames["pour"]
    w = next(f for f in sf.frames if f.label == "w")
    cam = canonical_cameras(w.origin, RADIUS)["iso"]
    ends, modes, segs, labels = draw_frame(Image.new("RGB", (1024, 1024)),
                                           cam, w, px=1024)
    assert set(ends) == {"x", "y"}          # z merged into mug.+up
    assert all(m == "arrow" for m in modes.values()) and len(segs) == 2
    for k in ends:
        pts = np.vstack([w.origin, w.origin + w.length * w.axes[k]])
        uv, _ = project(pts, cam, px=1024)
        np.testing.assert_allclose(np.array(ends[k]), uv, atol=1e-9)
    # the merged arrow is the mug's up, drawn from the same origin as w
    mug = next(f for f in sf.frames if f.label == "mug")
    np.testing.assert_allclose(mug.origin, w.origin, atol=1e-12)
    np.testing.assert_allclose(mug.axes["up_axis"], w.axes["z"], atol=1e-12)


def test_anchor_dots_are_labeled_with_point_tokens():
    """The dots carry the token the model writes as a trans anchor, not
    the call-#2 candidate id (which call #3 never saw)."""
    _, _, _, frames = _setup()
    labels = {role: [lab for lab, _ in sf.anchors] for role, sf in frames.items()}
    assert labels["grasp"] == ["teapot.handle_center"]
    assert labels["pour"] == ["mug.opening_center", "teapot.spout_tip"]
    assert labels["transport_active"] == labels["pour"]
    for sf in frames.values():
        assert "#" not in "".join(l for l, _ in sf.anchors)
        assert all(t["point"] for t in sf.triads)


def _drawn_arrows(sf):
    return [(fr.label, k, fr.origin, fr.axes[k], lab)
            for fr in sf.frames for k, lab, _ in fr.triad]


def test_one_arrow_per_distinct_direction_at_an_origin():
    """Merge invariant: no two drawn arrows share an origin (1 mm) and
    a direction (ALIGN_TOL). w.z always merges into the owner's +up; on
    the gripper stage all of w merges, leaving three arrows whose
    labels carry both names; the fallback w.x / w.y stay separate."""
    _, _, _, frames = _setup()
    for role, sf in frames.items():
        arrows = _drawn_arrows(sf)
        for i, (_, _, o1, d1, _) in enumerate(arrows):
            for (_, _, o2, d2, _) in arrows[i + 1:]:
                if np.linalg.norm(o1 - o2) < 1e-3:
                    assert float(d1 @ d2) < np.cos(ALIGN_TOL_RAD)
    g = frames["grasp"]
    assert len(_drawn_arrows(g)) == 3
    assert g.merged == {"x": "teapot.front_axis", "y": "teapot.lateral_axis",
                        "z": "teapot.up_axis"}
    labels = {lab for _, _, _, _, lab in _drawn_arrows(g)}
    assert labels == {"w.x = teapot.+front", "w.y = teapot.+left",
                      "w.z = teapot.+up"}
    for role in ("transport_active", "pour"):
        sf = frames[role]
        assert sf.merged == {"z": "mug.up_axis"}
        labels = {lab for _, _, _, _, lab in _drawn_arrows(sf)}
        assert "w.z = mug.+up" in labels and "w.x (fallback)" in labels
        assert "w.y" in labels and "mug.up" not in labels


def test_end_on_axes_become_glyphs_only_in_the_top_view():
    from PIL import Image
    _, _, _, frames = _setup()
    sf = frames["pour"]
    cams = canonical_cameras(np.zeros(3), RADIUS)
    teapot = next(f for f in sf.frames if f.label == "teapot")
    for vname in ("iso", "iso-opp"):
        _, modes, _, _ = draw_frame(Image.new("RGB", (1024, 1024)),
                                    cams[vname], teapot)
        assert all(m == "arrow" for m in modes.values())
    _, modes, segs, labels = draw_frame(Image.new("RGB", (1024, 1024)),
                                        cams["top"], teapot)
    assert modes["up_axis"] == "toward"        # +up points at a top camera
    assert modes["front_axis"] == modes["lateral_axis"] == "arrow"
    assert len(segs) == 2                       # glyphs add no segment
    assert len(labels) == 3                     # but still get a label
    assert 0 < END_ON_DEG < 45


def test_place_labels_separates_colliding_requests():
    """Two labels on one anchor, an arrow running through the obvious
    spot: the greedy placer returns non-overlapping boxes that avoid
    the segment; and it is deterministic."""
    from PIL import Image
    reqs = [LabelRequest((500.0, 500.0), "teapot.up", (0, 0, 255), 200.0,
                         t=np.array([0.0, -1.0])),
            LabelRequest((500.0, 500.0), "teapot.left", (200, 150, 0), 150.0,
                         t=np.array([1.0, 0.0])),
            LabelRequest((500.0, 500.0), "teapot #3", (0, 0, 0), 0.4)]
    segs = [((500.0, 500.0), (500.0, 300.0)), ((500.0, 500.0), (700.0, 500.0))]
    boxes = place_labels(Image.new("RGB", (1024, 1024)), list(reqs), segs)
    assert len(boxes) == 3
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert _area(a, b) == 0.0
        assert a[0] >= 0 and a[1] >= 0 and a[2] <= 1024 and a[3] <= 1024
    again = place_labels(Image.new("RGB", (1024, 1024)), list(reqs), segs)
    assert again == boxes


def _probe_gl() -> bool:
    """Probed ONCE: a failed offscreen context attempt can leave the
    process unable to survive a second one (MuJoCo aborts, not raises)."""
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
        mujoco.Renderer(m, 16, 16).close()
        return True
    except Exception:
        return False


_HAVE_GL = _probe_gl()


_HAVE_MESHES = all((d / "meshes" / f"{n}_visual.obj").exists()
                   for n, d in ASSET_DIRS.items())


@pytest.mark.skipif(not (_HAVE_MESHES and _HAVE_GL),
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
        assert set(rec["w"]) == {"object", "point", "point_body",
                                 "candidate_id", "x_route", "fallback",
                                 "merged"}
    assert manifest["roles"]["pour"]["w"]["point"] == "mug.opening_center"
    assert manifest["roles"]["pour"]["w"]["candidate_id"] == 2
    assert manifest["roles"]["pour"]["w"]["fallback"] is True
    assert manifest["roles"]["pour"]["w"]["merged"] == {"z": "mug.up_axis"}
    assert manifest["roles"]["pour"]["end_on"]["top"] == {
        "mug": ["up_axis"], "teapot": ["up_axis"]}
    assert manifest["roles"]["pour"]["end_on"]["iso"] == {}


@pytest.mark.skipif(not (_HAVE_MESHES and _HAVE_GL),
                    reason="needs converted visual meshes + offscreen GL")
def test_no_label_overlaps_in_any_rendered_view():
    import mujoco
    from scripts.render_stage_frames import (CAM_RADIUS_FRAC, PX,
                                             build_scene_model)
    scene, symbols, poses, _ = _setup()
    center, radius = scene_extent(scene, poses)
    sels, objects, role_index, _ = _pour_tea_selections()
    frames = compute_stage_frames(STAGES, sels, objects, role_index,
                                  scene.asset_dirs, symbols, poses, radius)
    cams = {k: v for k, v in
            canonical_cameras(center, CAM_RADIUS_FRAC * radius).items()
            if k in VIEWS}
    model = build_scene_model(scene, cams)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, PX, PX)
    for vname in VIEWS:
        r.update_scene(data, camera=vname)
        base = r.render().copy()
        for role, sf in frames.items():
            _, info = compose_view(base, cams[vname], sf, vname)
            boxes = info["labels"]
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    assert _area(a, b) == 0.0, (role, vname, a, b)
    r.close()
