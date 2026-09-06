"""Render, per emit stage, the frames the touchpoint-#3 emission is
written against — the image-conditioned arm's input, and the reason a
model composing 'teapot.-front antiparallel world.z' is not composing
blind.

One scene render per stage at the manifest SPAWN poses (the entry
attitude the compile gate freezes w and the goal attitude on — what the
model sees is what the gate tests), from three cameras, with two frames
drawn:

    w (DASHED, axis-colored x red / y gold / z blue, labeled w.x / w.y / w.z)
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
    one arrow per distinct direction at an origin
        w.z is the owner's +up by construction, and on a gripper stage
        w is the owner's whole canonical frame; drawing both is the
        same arrow twice. A w axis that shares an origin with a
        licensed canonical axis and agrees with it within the
        compiler's ALIGN_TOL is MERGED into the canonical arrow, whose
        label then carries both names ("w.z = mug.+up") — the
        equivalence stated in the token the model will write. Decided
        in the geometry stage (world vectors, compile_tsr's tolerance),
        never per view; the manifest records the merges.
    end-on axes
        per view, an axis within END_ON_DEG of the viewing direction is
        not drawn as a (vanishing) arrow but as the physics-diagram
        glyph in its color: (.) toward the camera, (x) away. The three
        views guarantee every axis is a real arrow somewhere.
    labels
        placed greedily — longest projected arrow first — over eight
        offsets around the tip plus one further along the arrow,
        choosing the candidate with the least overlap against labels
        already placed and every drawn segment. Deterministic.
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

from manip_sim.compile_tsr import ALIGN_TOL_RAD, _canonical_w
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
CAM_RADIUS_FRAC = 1.15                # camera distance uses the extent
                                      # inflated by the longest arrow so
                                      # tips and labels stay in frame
END_ON_DEG = 15.0                     # axis within this of the view axis
                                      # -> glyph, not arrow (per view)
DOT_R = 9
DASH_PX, GAP_PX = 12, 8               # w line style
GLYPH_R = 20
FRAMES_DIR = Path("outputs/frames")

AXIS_COLOR = {a: col for a, _, col in TRIAD_AXES}     # canonical columns
W_TRIAD = (("x", "w.x", AXIS_COLOR["front_axis"]),
           ("y", "w.y", AXIS_COLOR["lateral_axis"]),
           ("z", "w.z", AXIS_COLOR["up_axis"]))
_DIR_OF = {"front_axis": "+front", "lateral_axis": "+left", "up_axis": "+up"}


# --------------------------------------------------------------- geometry

@dataclass
class DrawnFrame:
    """One labeled frame in WORLD coords: origin + named unit axes."""
    label: str                        # "w" | object name
    origin: np.ndarray                # world, m
    axes: dict[str, np.ndarray]       # axis key -> world unit direction
    triad: tuple                      # (key, label, color) rows to draw
    length: float
    style: str = "solid"              # "solid" (canonical) | "dashed" (w)


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
    merged: dict[str, str] = field(default_factory=dict)  # w key -> "obj.axis"


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
            triad=(("x", x_lab, W_TRIAD[0][2]),) + W_TRIAD[1:],
            length=W_LEN_FRAC * radius, style="dashed"))
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
        _merge_coincident(sf)
        out[role] = sf
    return out


def _merge_coincident(sf: StageFrames, origin_tol: float = 1e-3) -> None:
    """One arrow per distinct direction at an origin: a w axis sharing
    an origin with a licensed canonical axis and agreeing with it within
    compile_tsr's ALIGN_TOL is dropped from the w frame and its name
    appended to the canonical arrow's label. Pure geometry on world
    vectors; recorded in sf.merged."""
    w = next(f for f in sf.frames if f.label == "w")
    keep = []
    for key, lab, col in w.triad:
        d = w.axes[key]
        hit = None
        for fr in sf.frames:
            if fr is w or np.linalg.norm(fr.origin - w.origin) > origin_tol:
                continue
            for a, dv in fr.axes.items():
                if float(np.dot(d, dv)) > np.cos(ALIGN_TOL_RAD):
                    hit = (fr, a)
                    break
            if hit:
                break
        if hit is None:
            keep.append((key, lab, col))
            continue
        fr, a = hit
        fr.triad = tuple((k, (f"w.{key} = {fr.label}.{_DIR_OF[k]}"
                              if k == a else l), c) for k, l, c in fr.triad)
        sf.merged[key] = f"{fr.label}.{a}"
    w.triad = tuple(keep)


# ---------------------------------------------------------------- drawing

@dataclass
class LabelRequest:
    """A label to place: its anchor pixel, text, color and a priority
    (projected arrow length; anchors and glyphs get a low fixed one).
    `t` is the arrow's image-plane unit direction (None for dots)."""
    anchor: tuple[float, float]
    text: str
    color: tuple
    priority: float
    t: np.ndarray | None = None


def _cam_forward(cam: dict) -> np.ndarray:
    import mujoco
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, cam["quat"])
    return -R.reshape(3, 3)[:, 2]           # MuJoCo cameras look along -z


def _line(dr, a, b, col, width, style):
    """Solid, or dashed (DASH_PX on / GAP_PX off) in `col` over a white
    underlay so it reads on any background."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if style == "solid":
        dr.line([*a, *b], fill=(255, 255, 255), width=width + 3)
        dr.line([*a, *b], fill=col, width=width)
        return
    v = b - a
    n = np.linalg.norm(v)
    if n < 1e-6:
        return
    t = v / n
    pos = 0.0
    segs = []
    while pos < n:
        segs.append((a + pos * t, a + min(pos + DASH_PX, n) * t))
        pos += DASH_PX + GAP_PX
    for p0, p1 in segs:
        dr.line([*p0, *p1], fill=(255, 255, 255), width=width + 3)
    for p0, p1 in segs:
        dr.line([*p0, *p1], fill=col, width=width)


def _arrowhead(dr, u0, v0, u1, v1, col, width):
    t = np.array([u1 - u0, v1 - v0], float)
    n = np.linalg.norm(t)
    if n < 1e-6:
        return
    t = t / n
    pn = np.array([-t[1], t[0]])
    for side in (1.0, -1.0):
        b = np.array([u1, v1]) - 14.0 * t + side * 8.0 * pn
        dr.line([b[0], b[1], u1, v1], fill=(255, 255, 255), width=width + 3)
        dr.line([b[0], b[1], u1, v1], fill=col, width=width)


def _glyph(dr, u, v, col, toward: bool):
    """(.) toward the camera, (x) away — in the axis color."""
    r = GLYPH_R
    dr.ellipse([u - r - 2, v - r - 2, u + r + 2, v + r + 2],
               fill=(255, 255, 255))                     # white underlay
    dr.ellipse([u - r, v - r, u + r, v + r], fill=(255, 255, 255),
               outline=col, width=5)
    if toward:
        k = r * 0.35
        dr.ellipse([u - k, v - k, u + k, v + k], fill=col)
    else:
        k = r * 0.6
        dr.line([u - k, v - k, u + k, v + k], fill=col, width=5)
        dr.line([u - k, v + k, u + k, v - k], fill=col, width=5)


def draw_frame(img, cam: dict, fr: DrawnFrame, px: int = PX
               ) -> tuple[dict[str, tuple], dict[str, str], list, list]:
    """Arrows (or end-on glyphs) for one frame; no occlusion culling
    (directions are frame metadata). Returns (pixel endpoints per axis
    key — the projection test compares these against project() of the
    compiler's basis; per-axis mode "arrow" | "toward" | "away";
    drawn segments; label requests). Labels are NOT drawn here — see
    place_labels."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    fwd = _cam_forward(cam)
    ends_px: dict[str, tuple] = {}
    modes: dict[str, str] = {}
    segments: list = []
    labels: list[LabelRequest] = []
    width = 3
    for key, label, col in fr.triad:
        d = fr.axes[key]
        ends = np.vstack([fr.origin, fr.origin + fr.length * d])
        uv, _ = project(ends, cam, px=px)
        (u0, v0), (u1, v1) = uv
        ends_px[key] = ((float(u0), float(v0)), (float(u1), float(v1)))
        c = float(np.dot(d, fwd))
        if abs(c) > np.cos(np.deg2rad(END_ON_DEG)):
            toward = c < 0
            modes[key] = "toward" if toward else "away"
            _glyph(dr, u0, v0, col, toward)
            labels.append(LabelRequest((float(u0), float(v0)), label, col,
                                       priority=0.5))
            continue
        modes[key] = "arrow"
        _line(dr, (u0, v0), (u1, v1), col, width, fr.style)
        _arrowhead(dr, u0, v0, u1, v1, col, width)
        segments.append(((u0, v0), (u1, v1)))
        t = np.array([u1 - u0, v1 - v0], float)
        n = float(np.linalg.norm(t))
        labels.append(LabelRequest((float(u1), float(v1)), label, col,
                                   priority=n, t=t / n if n > 1e-6 else None))
    return ends_px, modes, segments, labels


def draw_anchor(img, cam: dict, label: str, xyz: np.ndarray,
                px: int = PX) -> LabelRequest:
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    uv, _ = project(xyz[None, :], cam, px=px)
    u, v = uv[0]
    r = DOT_R
    dr.ellipse([u - r, v - r, u + r, v + r], fill=(0, 0, 0),
               outline=(255, 255, 255), width=3)
    return LabelRequest((float(u), float(v)), label, (0, 0, 0), priority=0.4)


# ---------------------------------------------------------- label placement

_OFFSETS = ((1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (-1, -1), (1, 1), (-1, 1))
LABEL_PAD = 10


def _bbox_at(dr, font, text, x, y):
    l, t, r, b = dr.textbbox((x, y), text, font=font, stroke_width=3)
    return (float(l), float(t), float(r), float(b))


def _area(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _segment_hits_box(seg, box) -> bool:
    """Liang-Barsky: does segment (p0, p1) intersect the box?"""
    (x0, y0), (x1, y1) = seg
    l, t, r, b = box
    dx, dy = x1 - x0, y1 - y0
    u0, u1 = 0.0, 1.0
    for p, q in ((-dx, x0 - l), (dx, r - x0), (-dy, y0 - t), (dy, b - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        u = q / p
        if p < 0:
            u0 = max(u0, u)
        else:
            u1 = min(u1, u)
        if u0 > u1:
            return False
    return True


def place_labels(img, requests: list[LabelRequest], segments: list,
                 px: int = PX, font_px: int = FONT_PX) -> list[tuple]:
    """Greedy point-feature label placement, highest priority (longest
    projected arrow) first. Candidates: eight compass offsets around
    the anchor and one further along the arrow. Cost: overlap area with
    labels already placed, plus a fixed cost per drawn segment the box
    crosses, plus a large out-of-frame penalty. Deterministic — ties
    break in candidate order. Returns the placed boxes (tests assert
    none overlap)."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    font = _font(font_px)
    placed: list[tuple] = []
    order = sorted(range(len(requests)), key=lambda i: -requests[i].priority)
    for i in order:
        rq = requests[i]
        w_txt = dr.textlength(rq.text, font=font)
        h_txt = font_px + 6
        cands = []
        ax, ay = rq.anchor
        for ox, oy in _OFFSETS:
            x = ax + ox * LABEL_PAD - (w_txt if ox < 0 else w_txt / 2 if ox == 0 else 0)
            y = ay + oy * LABEL_PAD - (h_txt if oy < 0 else h_txt / 2 if oy == 0 else 0)
            cands.append((x, y))
        if rq.t is not None:
            tip = np.array(rq.anchor) + 26.0 * rq.t
            cands.append((tip[0] - (w_txt if rq.t[0] < 0 else 0),
                          tip[1] - (h_txt if rq.t[1] < 0 else 0)))
        best, best_cost = None, None
        for (x, y) in cands:
            box = _bbox_at(dr, font, rq.text, x, y)
            cost = sum(_area(box, pb) for pb in placed)
            cost += 60.0 * sum(_segment_hits_box(sg, box) for sg in segments)
            if box[0] < 0 or box[1] < 0 or box[2] > px or box[3] > px:
                cost += 1e6
            if best_cost is None or cost < best_cost:
                best, best_cost = (x, y, box), cost
        x, y, box = best
        dr.text((x, y), rq.text, fill=rq.color, font=font, stroke_width=3,
                stroke_fill=(255, 255, 255))
        placed.append(box)
    return placed


def compose_view(base: np.ndarray, cam: dict, sf: StageFrames,
                 vname: str) -> tuple:
    """One view of one stage: arrows / glyphs / anchor dots, then the
    labels placed over everything. Returns (image, info) with info =
    {"end_on": {label -> [axis keys]}, "labels": placed boxes}."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(base)
    segments, requests, end_on = [], [], {}
    for lab, xyz in sf.anchors:
        requests.append(draw_anchor(img, cam, lab, xyz))
    for fr in sorted(sf.frames, key=lambda f: f.label == "w"):   # w on top
        _, modes, segs, labs = draw_frame(img, cam, fr)
        segments += segs
        requests += labs
        eo = [k for k, m in modes.items() if m != "arrow"]
        if eo:
            end_on[fr.label] = eo
    placed = place_labels(img, requests, segments)
    ImageDraw.Draw(img).text(
        (14, 10), f"stage {sf.stage} {sf.name} — {vname}",
        fill=(30, 30, 30), font=_font(FONT_PX),
        stroke_width=2, stroke_fill=(255, 255, 255))
    return img, {"end_on": end_on, "labels": placed}


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
        views, end_on = {}, {}
        for vname in VIEWS:
            img, info = compose_view(base[vname], cams[vname], sf, vname)
            path = rdir / f"{vname}.png"
            img.save(path)
            views[vname] = str(path)
            end_on[vname] = info["end_on"]
        manifest_roles[role] = {
            "stage": sf.stage, "name": sf.name, "active": sf.active,
            "passive": sf.passive,
            "w": {"object": sf.w_object, "point_body": sf.w_point_body,
                  "candidate_id": sf.w_candidate_id, "x_route": sf.w_x_route,
                  "fallback": sf.w_fallback, "merged": sf.merged},
            "triads": sf.triads,
            "end_on": end_on,
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
