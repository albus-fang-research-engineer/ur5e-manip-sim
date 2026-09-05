"""Render, per emit stage, the frames the touchpoint-#3 emission is
written against — the image-conditioned arm's input, and the reason a
model composing 'teapot.-front antiparallel world.z' is not composing
blind.

One scene render per stage at the manifest SPAWN poses (the entry
attitude the compile gate freezes w and the goal attitude on — what the
model sees is what the gate tests), from three cameras, with two frames
drawn:

    w (black, labeled w.x / w.y / w.z)
        the constraint frame compile_tsr roots the stage on:
        compile_tsr._canonical_w at the w-owning object's call-#2 point
        — the same function, same poses, same point the gate uses, so
        the drawn w and the compiled w are one construction. Drawn as
        ITS OWN frame, not as the owner's canonical triad: for a
        front-less owner w.x is the fallback azimuth (mover lateral /
        world x), which the owner does not license as a direction; the
        label says so ("w.x (fallback: ...)").
    canonical triads (front red, left gold, up blue; labels
    "<obj>.front" ...)
        the ACTIVE object's at its selected feature point, and the
        w-owner's at the w point, each drawing ONLY the directions
        vlm.py licenses for that object (Vocabulary.canonical_directions
        — the sim mug has no front, so it shows +up alone). A '-'
        direction is the drawn arrow reversed; negatives are not drawn
        (an arrowhead carries polarity; six arrows double clutter). The
        picture and the #3 accept set are therefore the same alphabet:
        no token is sayable that is not in the picture.
    anchor dots
        the resolved call-#2 points, labeled by object and pool id.

Three views, not eight: render_candidates' eight exist so every SURFACE
mark is unoccluded in >=1 view; triads are drawn without culling, and
their only legibility failure is an end-on projection, which two
opposed three-quarter views plus the top already cover. View names are
render_candidates.canonical_cameras' — VIEWS is a constant, not a
flag (an ablation on view count edits it under a --tag).

Output: outputs/frames/<role>/<view>.png + outputs/frames/manifest.json
recording, per role, the w construction (object, point, x route,
fallback flag), each drawn triad's object / point / axes / licensed
directions, and the view paths — the artifact emit_constraints.py
--frames turns into Client.emit_constraints view_paths.

Run from the repo root (headless; pick the backend as usual):

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. \\
        python scripts/render_stage_frames.py \\
            --selections outputs/selections/pour_tea.json
    ... --stage-plan outputs/stage_plan/pour_tea.marks.json   # call-#1 stages
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from manip_sim.compile_tsr import _canonical_w
from manip_sim.frames import load_symbols
from manip_sim.scene import add_scene_arg, load_scene
from manip_sim.selection import (load_pool, load_selections, resolve_selection,
                                 selected_points, selection_objects)
from manip_sim.tsr import pose_from_pos_quat_wxyz
from manip_sim.vlm import FRAME_AXES, Vocabulary, _DIR_AXIS
from scripts.render_candidates import (FOVY_DEG, TRIAD_AXES, _font,
                                       canonical_cameras, load_visual_mesh,
                                       project)

VIEWS = ("iso", "iso-opp", "top")     # constant, not a flag (see docstring)
PX = 1024
FONT_PX = 22
TRIAD_LEN_FRAC = 0.30                 # of the scene radius
W_LEN_FRAC = 0.40                     # w drawn longer than the owner's triad
                                      # so coincident axes (w.z = owner up)
                                      # stay distinguishable
CAM_RADIUS_FRAC = 1.4                 # camera distance uses the extent
                                      # inflated by the longest arrow so
                                      # tips and labels stay in frame
W_COLOR = (20, 20, 20)
DOT_R = 9
FRAMES_DIR = Path("outputs/frames")

W_TRIAD = (("x", "w.x", W_COLOR), ("y", "w.y", W_COLOR), ("z", "w.z", W_COLOR))


# --------------------------------------------------------------- geometry

@dataclass
class DrawnFrame:
    """One labeled frame in WORLD coords: origin + named unit axes."""
    label: str                        # "w" | object name
    origin: np.ndarray                # world, m
    axes: dict[str, np.ndarray]       # axis key -> world unit direction
    triad: tuple                      # (key, label, color) rows to draw
    length: float


@dataclass
class StageFrames:
    role: str
    stage: int
    name: str
    active: str
    passive: str | None
    w_object: str
    w_point_body: list[float]
    w_candidate_id: int | None
    w_x_route: str
    w_fallback: bool
    frames: list[DrawnFrame] = field(default_factory=list)
    anchors: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (label, world)
    triads: list[dict] = field(default_factory=list)   # manifest records


def _world(T: np.ndarray, p_body) -> np.ndarray:
    return (T @ np.append(np.asarray(p_body, float).reshape(3), 1.0))[:3]


def _canonical_triad(obj: str, symbols, T: np.ndarray, point_body,
                     vocab: Vocabulary, radius: float) -> tuple[DrawnFrame, dict]:
    """The object's canonical triad at a body point, drawing only the
    axes whose signed directions vlm.py licenses for it."""
    licensed = vocab.canonical_directions(obj)
    cols = {_DIR_AXIS[d[1:]] for d in licensed}
    R = T[:3, :3]
    axes = {a: R @ (np.asarray(symbols[obj].axes[a], float)
                    / np.linalg.norm(symbols[obj].axes[a]))
            for a in FRAME_AXES if a in cols}
    triad = tuple((a, f"{obj}.{lab}", col) for a, lab, col in TRIAD_AXES
                  if a in axes)
    fr = DrawnFrame(label=obj, origin=_world(T, point_body), axes=axes,
                    triad=triad, length=TRIAD_LEN_FRAC * radius)
    rec = {"object": obj,
           "point_body": np.round(np.asarray(point_body, float), 5).tolist(),
           "axes_drawn": sorted(axes), "directions": list(licensed)}
    return fr, rec


def compute_stage_frames(stages, sels, objects, role_index, asset_dirs,
                         symbols, poses, radius: float) -> dict[str, StageFrames]:
    """Pure geometry (no GL): per role, the w frame via compile_tsr's
    own constructor and the licensed canonical triads, all in world
    coords at `poses`. Tested against compile_stage's T0_w directly."""
    vocab = Vocabulary.from_symbols(symbols)
    cand = {r: int(sels[r].candidate_id) for r in sels}
    out: dict[str, StageFrames] = {}
    for stage, role in stages:
        w_point, e_point = selected_points(role, stage, sels, objects,
                                           role_index, asset_dirs, symbols)
        w_obj = stage.passive or stage.active
        active = stage.active if stage.passive else None
        w_frame, route = _canonical_w(w_obj, active, symbols, poses,
                                      np.asarray(w_point, float),
                                      "passive" if stage.passive else "active")
        T0_w = poses[w_obj] @ w_frame.T()
        fallback = w_frame.name.endswith("(fallback)")
        x_lab = "w.x (fallback)" if fallback else "w.x"   # route: manifest
        sf = StageFrames(
            role=role, stage=stage.index, name=stage.name, active=stage.active,
            passive=stage.passive, w_object=w_obj,
            w_point_body=np.round(w_frame.point, 5).tolist(),
            w_candidate_id=next((cand[r] for r in sels
                                 if objects[r] == w_obj
                                 and np.allclose(resolve_selection(
                                     sels[r], load_pool(asset_dirs[w_obj]),
                                     symbols[w_obj]).frame.point, w_point)),
                                None),
            w_x_route=route, w_fallback=fallback)
        sf.frames.append(DrawnFrame(
            label="w", origin=T0_w[:3, 3],
            axes={"x": T0_w[:3, 0], "y": T0_w[:3, 1], "z": T0_w[:3, 2]},
            triad=(("x", x_lab, W_COLOR),) + W_TRIAD[1:],
            length=W_LEN_FRAC * radius))
        # the w-owner's own licensed triad at the w point
        fr, rec = _canonical_triad(w_obj, symbols, poses[w_obj], w_point,
                                   vocab, radius)
        sf.frames.append(fr)
        sf.triads.append({"role_in_stage": "w_owner", **rec})
        sf.anchors.append((f"{w_obj} #{sf.w_candidate_id}", fr.origin))
        # the active object's licensed triad at its feature point
        if active is not None and e_point is not None:
            fr, rec = _canonical_triad(active, symbols, poses[active],
                                       e_point, vocab, radius)
            sf.frames.append(fr)
            sf.triads.append({"role_in_stage": "active", **rec})
            sf.anchors.append((f"{active} #{cand[role]}", fr.origin))
        out[role] = sf
    return out


# ---------------------------------------------------------------- drawing

def draw_frame(img, cam: dict, fr: DrawnFrame, px: int = PX,
               font_px: int = FONT_PX) -> dict[str, tuple]:
    """Labeled positive arrows for one frame; no occlusion culling
    (directions are frame metadata). Returns the pixel endpoints per
    axis key — the projection test compares these against project() of
    the compiler's basis."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    font = _font(font_px)
    ends_px: dict[str, tuple] = {}
    for key, label, col in fr.triad:
        d = fr.axes[key]
        ends = np.vstack([fr.origin, fr.origin + fr.length * d])
        uv, _ = project(ends, cam, px=px)
        (u0, v0), (u1, v1) = uv
        ends_px[key] = ((float(u0), float(v0)), (float(u1), float(v1)))
        width = 4 if fr.label == "w" else 3
        dr.line([u0, v0, u1, v1], fill=(255, 255, 255), width=width + 3)
        dr.line([u0, v0, u1, v1], fill=col, width=width)
        t = np.array([u1 - u0, v1 - v0], float)
        n = np.linalg.norm(t)
        if n > 1e-6:
            t = t / n
            p = np.array([-t[1], t[0]])
            for side in (1.0, -1.0):
                b = np.array([u1, v1]) - 14.0 * t + side * 8.0 * p
                dr.line([b[0], b[1], u1, v1], fill=(255, 255, 255),
                        width=width + 3)
                dr.line([b[0], b[1], u1, v1], fill=col, width=width)
        if fr.label == "w":            # w labels left of / above the tip:
            tw = dr.textlength(label, font=font)   # away from the canonical
            xy = (u1 - tw - 8, v1 - font_px - 6)   # labels drawn tip-right
        else:
            xy = (u1 + 8, v1 - 10)
        dr.text(xy, label, fill=col, font=font, stroke_width=3,
                stroke_fill=(255, 255, 255))
    return ends_px


def draw_anchor(img, cam: dict, label: str, xyz: np.ndarray,
                px: int = PX, font_px: int = FONT_PX) -> None:
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    uv, _ = project(xyz[None, :], cam, px=px)
    u, v = uv[0]
    r = DOT_R
    dr.ellipse([u - r, v - r, u + r, v + r], fill=(0, 0, 0),
               outline=(255, 255, 255), width=3)
    font = _font(font_px)
    tw = dr.textlength(label, font=font)
    dr.text((u - tw - r, v + r), label, fill=(0, 0, 0), font=font,
            stroke_width=3, stroke_fill=(255, 255, 255))


# ------------------------------------------------------------------ scene

def build_scene_model(scene, cams: dict[str, dict], px: int = PX):
    """One MJCF with every scene object's VISUAL mesh at its spawn pose,
    a table-top slab for a ground cue, headlight, and the cameras.
    Collision hulls are dropped; mesh paths are made absolute so a
    single model spans per-object mesh dirs; body names are prefixed so
    the objects' identical 'object' bodies do not collide."""
    import mujoco
    root = ET.Element("mujoco", model="stage_frames")
    ET.SubElement(root, "compiler", angle="radian")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(px), offheight=str(px))
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45",
                  diffuse="0.6 0.6 0.6")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.96 0.96 0.98", rgb2="0.78 0.80 0.86",
                  width="32", height="32")
    wb = ET.SubElement(root, "worldbody")
    tz = scene.table_top_z
    sx, sy, _ = scene.table_size
    ET.SubElement(wb, "geom", type="box", pos=f"0 0 {tz - 0.01:.4f}",
                  size=f"{sx / 2:.3f} {sy / 2:.3f} 0.01",
                  rgba="0.88 0.86 0.82 1", group="1", contype="0",
                  conaffinity="0")
    for name, obj in scene.objects.items():
        tree = ET.parse(obj.xml)
        oroot = tree.getroot()
        for m in oroot.iter("mesh"):
            if "_col_" in m.get("name", ""):
                continue
            f = Path(m.get("file"))
            m.set("file", str((obj.asset / f).resolve()))
            asset.append(m)
        pos, quat = scene.fixed_poses()[name]
        body = ET.SubElement(wb, "body", name=name,
                             pos=" ".join(f"{v:.6f}" for v in pos),
                             quat=" ".join(f"{v:.6f}" for v in quat))
        for g in oroot.iter("geom"):
            if "_col_" in g.get("mesh", ""):
                continue
            g.attrib.pop("mass", None)
            body.append(g)
    for cname, c in cams.items():
        ET.SubElement(wb, "camera", name=cname, fovy=str(FOVY_DEG),
                      pos=" ".join(f"{v:.6f}" for v in c["pos"]),
                      quat=" ".join(f"{v:.6f}" for v in c["quat"]))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def scene_extent(scene, poses) -> tuple[np.ndarray, float]:
    """(center, radius) of all objects' visual vertices at spawn."""
    pts = []
    for name in scene.objects:
        V, _ = load_visual_mesh(name, scene.asset_dirs[name])
        pts.append((poses[name][:3, :3] @ V.T).T + poses[name][:3, 3])
    P = np.vstack(pts)
    center = 0.5 * (P.min(0) + P.max(0))
    return center, float(np.linalg.norm(P - center, axis=1).max())


def render(scene, frames: dict[str, StageFrames], out_dir: Path,
           center: np.ndarray, radius: float) -> dict:
    import mujoco
    from PIL import Image, ImageDraw
    cams = {k: v for k, v in
            canonical_cameras(center, CAM_RADIUS_FRAC * radius).items()
            if k in VIEWS}
    model = build_scene_model(scene, cams)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 0
    vopt.geomgroup[1] = 1
    renderer = mujoco.Renderer(model, PX, PX)
    base = {}
    for vname in VIEWS:
        renderer.update_scene(data, camera=vname, scene_option=vopt)
        base[vname] = renderer.render().copy()
    renderer.close()

    manifest_roles = {}
    for role, sf in frames.items():
        rdir = out_dir / role
        rdir.mkdir(parents=True, exist_ok=True)
        views = {}
        for vname in VIEWS:
            img = Image.fromarray(base[vname])
            cam = cams[vname]
            for lab, xyz in sf.anchors:
                draw_anchor(img, cam, lab, xyz)
            for fr in sf.frames:          # w last: drawn on top
                if fr.label != "w":
                    draw_frame(img, cam, fr)
            draw_frame(img, cam, next(f for f in sf.frames if f.label == "w"))
            ImageDraw.Draw(img).text(
                (14, 10), f"stage {sf.stage} {sf.name} — {vname}",
                fill=(30, 30, 30), font=_font(FONT_PX),
                stroke_width=2, stroke_fill=(255, 255, 255))
            path = rdir / f"{vname}.png"
            img.save(path)
            views[vname] = str(path)
        manifest_roles[role] = {
            "stage": sf.stage, "name": sf.name, "active": sf.active,
            "passive": sf.passive,
            "w": {"object": sf.w_object, "point_body": sf.w_point_body,
                  "candidate_id": sf.w_candidate_id, "x_route": sf.w_x_route,
                  "fallback": sf.w_fallback},
            "triads": sf.triads,
            "views": views,
        }
    manifest = {"scene": str(scene.path), "poses": "spawn",
                "views": list(VIEWS), "px": PX, "roles": manifest_roles}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", required=True, metavar="JSON",
                    help="touchpoint-#2 artifact (role-keyed)")
    ap.add_argument("--stage-plan", default=None, metavar="JSON",
                    help="plan_stages.py artifact; its bound stages replace "
                         "the fixed pour-tea table (same binding "
                         "emit_constraints.py uses)")
    ap.add_argument("--out-dir", default=str(FRAMES_DIR))
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene, getattr(args, "grounding", None))
    asset_dirs = scene.asset_dirs
    symbols = {n: load_symbols(d) for n, d in asset_dirs.items()}
    poses = {n: pose_from_pos_quat_wxyz(*pq)
             for n, pq in scene.fixed_poses().items()}

    from scripts.emit_constraints import STAGES
    stages = STAGES
    if args.stage_plan:
        from scripts.plan_stages import load_bindings
        b = load_bindings(args.stage_plan)
        by_idx = {s.index: s for s in b.plan.stages}
        stages = tuple((by_idx[b.roles[r][1].index], r)
                       for r in ("grasp", "transport_active", "pour"))
        role_index = {r: st.index for r, (_, st) in b.roles.items()}
    else:
        from scripts.select_frames import ROLES
        role_index = {r: st.index for r, (_, st) in ROLES.items()}

    sels = load_selections(args.selections)
    objects = selection_objects(args.selections, sels)
    center, radius = scene_extent(scene, poses)
    frames = compute_stage_frames(stages, sels, objects, role_index,
                                  asset_dirs, symbols, poses, radius)
    for role, sf in frames.items():
        print(f"[frames] {role}: w = {sf.w_object} canonical at "
              f"{sf.w_point_body} (x = {sf.w_x_route}"
              f"{', FALLBACK' if sf.w_fallback else ''}); triads: "
              + ", ".join(f"{t['object']}[{','.join(t['directions'])}]"
                          for t in sf.triads))
    out_dir = Path(args.out_dir)
    render(scene, frames, out_dir, center, radius)
    print(f"[frames] wrote {out_dir}/<role>/<view>.png + "
          f"{out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
