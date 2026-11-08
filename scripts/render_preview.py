"""Preview images for the critic (VLM call #4): what the specification
allows, rendered before planning. See docs/deferred.md ("Evidence per
task") for the decisions behind each choice.

Two kinds of image, written in the order the critic prompt attaches them:

  state images     one per state: the initial scene, the grasp goal (the
                   gripper at the grasp), then each mover stage's goal drawn
                   at its subgoal NOMINAL (stage k+1 starts where stage k's
                   goal is, because the compiler builds stage k+1 against
                   stage k's subgoal.nominal()). Each image is a montage of
                   the iso and top views; the grounded points the critic may
                   name are labeled.
  path boundaries  one montage per stage with a path TSR: for each path row,
                   the mover at that row's bound with every other row held at
                   the representative pose (never joint corners). Bounded
                   rows show both bounds; free rotation rows show the mover
                   turned +45 and +90 degrees; free translation rows show it
                   shifted +-10 cm; pinned rows are listed, not drawn.

Representative pose: the midpoint, in the path TSR's displacement
coordinates, of the stage's entry pose and its subgoal nominal, clamped
into the path box. The boundary is shown mid-motion, away from the start
and the goal, which have their own state images.

Boundary style: "panels" (one solid render per bound value; the default,
chosen on the 11/05 hardware check) or "overlay" (the representative pose
solid, the bound poses as translucent copies in one render).

The gripper is drawn at the grasp and follows the held object afterward.
Its source is pluggable (GripperModel, a mesh in the TCP frame): the sim's
robosuite Robotiq85, whose TCP is the grip site the sim's IK targets
(sim_gripper), or a URDF's fixed chain below a root link, e.g. the
hardware's 2F-85 below tool0, the frame the hardware IK targets
(gripper_from_urdf). Place it with the same FK the planner used.

Inputs as a JSON document (load_inputs / render_from_inputs, or the CLI):

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_preview.py \\
        --inputs preview_inputs.json --out <dir> [--style panels|overlay]

Geometry (states, boundary poses, camera choice, the manifest) needs no GL
and imports MuJoCo only where it renders.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROW_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
STATE_VIEWS = ("iso", "top")
STATE_PX = 768
PANEL_PX = 448
FREE_ROT_SAMPLES_DEG = (45.0, 90.0)
FREE_TRANS_SHOW_M = 0.10
PINNED_EPS = 1e-9
GHOST_RGBA = (0.85, 0.35, 0.10, 0.35)
GRIPPER_RGBA = (0.25, 0.25, 0.28, 1.0)
EXTENT_MARGIN = 1.05
BOUNDARY_STYLES = ("panels", "overlay")
# cameras a boundary panel may use (the canonical set minus "bottom")
BOUNDARY_VIEWS = ("iso", "iso-opp", "+x", "-x", "+y", "-y", "top")


# ------------------------------------------------------------------ inputs

@dataclass
class StageInput:
    """One mover stage as compiled for the preview. `path`/`subgoal` are
    TSRs (None when the stage did not compile; `error` says why).
    `entry` is the mover's pose the stage was compiled at."""
    role: str
    name: str
    mover: str
    passive: str | None
    path: object | None
    subgoal: object | None
    entry: np.ndarray | None = None
    error: str | None = None
    row_labels: dict[int, str] = field(default_factory=dict)   # optional per-row
                                                                # slot/token text


@dataclass
class GraspInput:
    """The grasp: the held object and the TCP pose at the grasp (world),
    from the same FK the planner's IK used."""
    obj: str
    tcp_pose: np.ndarray


@dataclass
class State:
    key: str
    title: str
    poses: dict[str, np.ndarray]
    gripper_pose: np.ndarray | None = None
    role: str | None = None


# ------------------------------------------------------------------ gripper

@dataclass
class GripperModel:
    """A gripper as one triangle mesh in its TCP frame."""
    V: np.ndarray
    F: np.ndarray
    source: str

    def write_obj(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.V]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.F]
        path.write_text("\n".join(lines) + "\n")
        return path


def _box_mesh(half: np.ndarray):
    import trimesh
    return trimesh.creation.box(extents=2.0 * np.asarray(half, float))


def sim_gripper(xml: str | Path | None = None, site: str = "grip_site") -> GripperModel:
    """robosuite's Robotiq85 at its spawn (open) configuration, expressed in
    the grip-site frame: the frame the sim's IK targets
    (gripper0_right_grip_site)."""
    import mujoco
    import trimesh
    if xml is None:
        import robosuite
        xml = Path(robosuite.__file__).parent / "models/assets/grippers/robotiq_gripper_85.xml"
    import xml.etree.ElementTree as ET
    root = ET.parse(str(xml)).getroot()
    # robosuite's gripper XMLs repeat mesh names (robosuite dedups when it
    # merges models); standalone, keep the first definition of each name
    seen = set()
    for asset in root.iter("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") in seen:
                asset.remove(mesh)
            seen.add(mesh.get("name"))
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("meshdir", str(Path(xml).parent.resolve()))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"{xml}: no site {site!r}")
    T_site = np.eye(4)
    T_site[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    T_site[:3, 3] = d.site_xpos[sid]
    T_inv = np.linalg.inv(T_site)
    parts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 1:                    # visual geoms only
            continue
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            mesh = trimesh.Trimesh(m.mesh_vert[va:va + vn].copy(),
                                   m.mesh_face[fa:fa + fn].copy(), process=False)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = _box_mesh(m.geom_size[g])
        else:
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        mesh.apply_transform(T_inv @ T)
        parts.append(mesh)
    whole = trimesh.util.concatenate(parts)
    return GripperModel(np.asarray(whole.vertices), np.asarray(whole.faces),
                        f"robosuite {Path(xml).name} @ {site}")


def _urdf_origin(el) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
    T[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()     # URDF: fixed-axis rpy
    T[:3, 3] = xyz
    return T


def gripper_from_urdf(urdf: str | Path, root_link: str = "tool0",
                      mesh_dir: str | Path | None = None,
                      geometry: str = "collision") -> GripperModel:
    """The meshes of every link below `root_link` along FIXED joints, in
    `root_link`'s frame. Mesh files that do not exist as written (e.g.
    container paths) are looked up by file name under `mesh_dir`. Use
    "collision" geometry: those are STL, which this loader and MuJoCo read;
    the visual DAE files are not."""
    import xml.etree.ElementTree as ET

    import trimesh
    root = ET.parse(str(urdf)).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    if root_link not in links:
        raise ValueError(f"{urdf}: no link {root_link!r}")
    children: dict[str, list] = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    parts, stack = [], [(root_link, np.eye(4))]
    while stack:
        name, T_link = stack.pop()
        for g in links[name].findall(geometry):
            mesh_el = g.find("geometry/mesh")
            if mesh_el is None:
                continue
            f = Path(mesh_el.get("filename").replace("file://", ""))
            if not f.is_file():
                if mesh_dir is None or not (Path(mesh_dir) / f.name).is_file():
                    raise FileNotFoundError(f"{urdf}: mesh {f} not found "
                                            f"(mesh_dir={mesh_dir})")
                f = Path(mesh_dir) / f.name
            mesh = trimesh.load(str(f), force="mesh", process=False)
            scale = mesh_el.get("scale")
            if scale:
                mesh.apply_scale([float(v) for v in scale.split()])
            mesh.apply_transform(T_link @ _urdf_origin(g.find("origin")))
            parts.append(mesh)
        for j in children.get(name, ()):
            if j.get("type") != "fixed":
                continue                                # the frozen chain is all fixed
            stack.append((j.find("child").get("link"),
                          T_link @ _urdf_origin(j.find("origin"))))
    if not parts:
        raise ValueError(f"{urdf}: no {geometry} meshes below {root_link!r}")
    whole = trimesh.util.concatenate(parts)
    return GripperModel(np.asarray(whole.vertices), np.asarray(whole.faces),
                        f"{Path(urdf).name} below {root_link} ({geometry})")


# ------------------------------------------------------------------ geometry

def chain_states(initial: dict[str, np.ndarray], stages: list[StageInput],
                 grasp: GraspInput | None = None) -> tuple[list[State], list[dict]]:
    """The states the critic sees, in order, and the stages that could not
    be drawn. A mover's goal is its subgoal NOMINAL; the gripper (if a
    grasp is given) follows the held object rigidly. A stage that did not
    compile ends the chain: later goals were compiled from its goal."""
    poses = {k: np.asarray(v, float) for k, v in initial.items()}
    states = [State("initial", "initial scene", dict(poses))]
    skipped: list[dict] = []
    T_obj_tcp = None
    if grasp is not None:
        T_obj_tcp = np.linalg.inv(poses[grasp.obj]) @ np.asarray(grasp.tcp_pose, float)
        states.append(State("grasp", "grasp goal (gripper at the grasp)", dict(poses),
                            np.asarray(grasp.tcp_pose, float), "grasp"))
    gripper = states[-1].gripper_pose
    for i, st in enumerate(stages):
        if st.subgoal is None:
            skipped.append({"role": st.role, "name": st.name,
                            "reason": st.error or "not compiled"})
            skipped += [{"role": s.role, "name": s.name,
                         "reason": f"follows {st.role}, which was not compiled"}
                        for s in stages[i + 1:]]
            break
        poses = dict(poses)
        poses[st.mover] = np.asarray(st.subgoal.nominal(), float)
        if T_obj_tcp is not None and grasp is not None and st.mover == grasp.obj:
            gripper = poses[st.mover] @ T_obj_tcp
        states.append(State(f"goal:{st.role}", f"{st.role} goal ({st.name})",
                            poses, gripper, st.role))
    return states, skipped


def representative_pose(path, entry: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(displacement, pose): the midpoint of the entry's and the goal's
    displacements in the path TSR, clamped into its box."""
    from manip_sim.tsr import displacement_to_pose
    de = path.displacement(np.asarray(entry, float))
    dg = path.displacement(np.asarray(goal, float))
    step = dg - de
    step[3:] = (step[3:] + np.pi) % (2 * np.pi) - np.pi     # angles: the short way round
    d = np.clip(de + 0.5 * step, path.Bw[:, 0], path.Bw[:, 1])
    return d, path.T0_w @ displacement_to_pose(d) @ path.Tw_e


def _rot_token(half_width: float) -> str | None:
    from manip_sim.compile_tsr import ROT_TOL_RAD
    for tok, v in ROT_TOL_RAD.items():
        if abs(half_width - v) < 1e-6:
            return tok
    return None


def _fmt(i: int, v: float) -> str:
    return f"{np.degrees(v):+.0f} deg" if i >= 3 else f"{100 * v:+.1f} cm"


@dataclass
class Boundary:
    row: int
    kind: str                     # bounded | one-sided | free
    label: str                    # the row, e.g. "pitch (about w.y)"
    values: list[float]           # the row value at each drawn pose
    captions: list[str]
    poses: list[np.ndarray]
    token: str | None = None


def boundary_samples(path, d_rep: np.ndarray,
                     row_labels: dict[int, str] | None = None) -> tuple[list[Boundary], list[str]]:
    """Per path row, the poses to draw: every other row at d_rep. Returns
    (boundaries, pinned row names)."""
    from manip_sim.tsr import displacement_to_pose
    row_labels = row_labels or {}
    out, pinned = [], []
    for i in range(6):
        lo, hi = float(path.Bw[i, 0]), float(path.Bw[i, 1])
        axis = f"about w.{'xyz'[i - 3]}" if i >= 3 else f"along w.{'xyz'[i]}"
        label = f"{ROW_NAMES[i]} ({axis})"
        one_sided = np.isfinite(lo) != np.isfinite(hi)
        free = (not np.isfinite(lo) and not np.isfinite(hi)) or \
            (i >= 3 and hi - lo >= 2 * np.pi - 1e-9)       # as TSR's rotation-free rule
        if not free and not one_sided and hi - lo < PINNED_EPS:
            pinned.append(ROW_NAMES[i])
            continue
        if one_sided:
            # the finite bound, and the open side shown FREE_TRANS_SHOW_M past
            # the representative value (never past the finite bound)
            if np.isfinite(lo):
                vals = [lo, max(d_rep[i], lo) + FREE_TRANS_SHOW_M]
                caps = [f"{ROW_NAMES[i]} at lower bound {_fmt(i, lo)}",
                        f"{ROW_NAMES[i]} open above: {_fmt(i, vals[1])} shown"]
            else:
                vals = [hi, min(d_rep[i], hi) - FREE_TRANS_SHOW_M]
                caps = [f"{ROW_NAMES[i]} at upper bound {_fmt(i, hi)}",
                        f"{ROW_NAMES[i]} open below: {_fmt(i, vals[1])} shown"]
            token, kind = None, "one-sided"
        elif free:
            vals = ([d_rep[i] + np.radians(a) for a in FREE_ROT_SAMPLES_DEG] if i >= 3
                    else [d_rep[i] - FREE_TRANS_SHOW_M, d_rep[i] + FREE_TRANS_SHOW_M])
            caps = [f"{ROW_NAMES[i]} free: {_fmt(i, v - d_rep[i])} shown" for v in vals]
            token, kind = None, "free"
        else:
            vals = [lo, hi]
            token = _rot_token(0.5 * (hi - lo)) if i >= 3 else None
            tag = f" [{token}]" if token else ""
            caps = [f"{ROW_NAMES[i]} at {'lower' if k == 0 else 'upper'} bound "
                    f"{_fmt(i, v)}{tag}" for k, v in enumerate(vals)]
            kind = "bounded"
        if i in row_labels:
            caps = [f"{c} -- {row_labels[i]}" for c in caps]
        poses = []
        for v in vals:
            d = np.array(d_rep, float)
            d[i] = v
            poses.append(path.T0_w @ displacement_to_pose(d) @ path.Tw_e)
        out.append(Boundary(i, kind, label, [float(v) for v in vals], caps, poses, token))
    return out, pinned


def row_world_axis(path, row: int) -> np.ndarray:
    """The world direction a row moves along (translation) or about (rotation)."""
    return path.T0_w[:3, row % 3]


def pick_view(cams: dict[str, dict], center: np.ndarray, axis: np.ndarray,
              rotation: bool, views=BOUNDARY_VIEWS) -> str:
    """The camera that shows a row's motion in the image plane: a shift is
    seen from the side (view most perpendicular to the axis), a rotation
    from along its axis (the turn then happens in the image plane). Ties
    keep the earlier view (iso first)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    best, score = None, -1.0
    for v in views:
        if v not in cams:
            continue
        f = center - np.asarray(cams[v]["pos"], float)
        f = f / np.linalg.norm(f)
        s = abs(float(f @ a)) if rotation else float(np.linalg.norm(np.cross(f, a)))
        if s > score + 1e-9:
            best, score = v, s
    return best


# ------------------------------------------------------------------ rendering

_MESH_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _visual_vertices(name: str, d: Path) -> np.ndarray:
    key = (name, str(d))
    if key not in _MESH_CACHE:
        from scripts.render_candidates import load_visual_mesh
        _MESH_CACHE[key] = load_visual_mesh(name, d)[0]
    return _MESH_CACHE[key]


def _extent(scene, pose_sets: list[dict[str, np.ndarray]], gripper: GripperModel | None,
            gripper_poses: list[np.ndarray]) -> tuple[np.ndarray, float]:
    pts = []
    for poses in pose_sets:
        for name, T in poses.items():
            if name not in scene.objects:
                continue
            V = _visual_vertices(name, scene.asset_dirs[name])
            pts.append(V @ T[:3, :3].T + T[:3, 3])
    if gripper is not None:
        for T in gripper_poses:
            pts.append(gripper.V @ T[:3, :3].T + T[:3, 3])
    P = np.vstack(pts)
    c = 0.5 * (P.min(0) + P.max(0))
    return c, EXTENT_MARGIN * float(np.linalg.norm(P - c, axis=1).max())


def _render(scene, poses, extras, view_names, px, center, radius) -> dict:
    import mujoco

    from scripts.render_candidates import canonical_cameras
    from scripts.render_stage_frames import CAM_RADIUS_FRAC, build_scene_model
    cams = {k: v for k, v in canonical_cameras(center, CAM_RADIUS_FRAC * radius).items()
            if k in view_names}
    model = build_scene_model(scene, cams, px=px, poses=poses, extras=extras)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 0
    vopt.geomgroup[1] = 1
    r = mujoco.Renderer(model, px, px)
    out = {}
    for v in view_names:
        r.update_scene(data, camera=v, scene_option=vopt)
        out[v] = (r.render().copy(), cams[v])
    r.close()
    return out


def label_points(labels: dict[str, dict[str, np.ndarray]],
                 poses: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """(text, world xyz) for every labeled point of an object present."""
    out = []
    for obj, pts in labels.items():
        if obj not in poses:
            continue
        T = poses[obj]
        for name, xyz in pts.items():
            out.append((f"{obj}.{name}", T[:3, :3] @ np.asarray(xyz, float) + T[:3, 3]))
    return out


def _draw_labels(img, cam, pts, px) -> list[dict]:
    from scripts.render_stage_frames import draw_anchor, place_labels
    reqs = [draw_anchor(img, cam, text, xyz, px=px) for text, xyz in pts]
    place_labels(img, reqs, [], px=px)
    return [{"text": r.text, "pixel": [round(r.anchor[0], 1), round(r.anchor[1], 1)]}
            for r in reqs]


def _caption(img, text):
    from PIL import ImageDraw

    from scripts.render_candidates import _font
    ImageDraw.Draw(img).text((10, 8), text, fill=(25, 25, 25), font=_font(18),
                             stroke_width=2, stroke_fill=(255, 255, 255))


def _grid(images: list, cols: int, title: str):
    from PIL import Image, ImageDraw

    from scripts.render_candidates import _font
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    top = 36
    out = Image.new("RGB", (cols * w, rows * h + top), (255, 255, 255))
    ImageDraw.Draw(out).text((12, 8), title, fill=(20, 20, 20), font=_font(20))
    for k, im in enumerate(images):
        out.paste(im, ((k % cols) * w, top + (k // cols) * h))
    return out


def _gripper_extra(gripper_obj: Path | None, T) -> list[dict]:
    if gripper_obj is None or T is None:
        return []
    return [{"mesh": str(gripper_obj), "pose": T, "rgba": GRIPPER_RGBA}]


def _r(T) -> list:
    return np.round(np.asarray(T, float), 5).tolist()


def render_preview(scene, initial: dict[str, np.ndarray], stages: list[StageInput],
                   labels: dict[str, dict[str, np.ndarray]], out_dir: Path,
                   grasp: GraspInput | None = None, gripper: GripperModel | None = None,
                   style: str = "panels") -> dict:
    """Render every state image and path-boundary montage; write the PNGs
    and manifest.json to `out_dir`. Returns the manifest."""
    from PIL import Image
    if style not in BOUNDARY_STYLES:
        raise ValueError(f"style {style!r}: one of {BOUNDARY_STYLES}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gobj = gripper.write_obj(out_dir / "gripper.obj") if gripper is not None else None
    states, skipped = chain_states(initial, stages, grasp)
    manifest = {"style": style, "images": [], "skipped": skipped,
                "gripper": gripper.source if gripper is not None else None}

    for n, s in enumerate(states):
        c, rad = _extent(scene, [s.poses], gripper, [s.gripper_pose] if s.gripper_pose is not None else [])
        views = _render(scene, s.poses, _gripper_extra(gobj, s.gripper_pose),
                        STATE_VIEWS, STATE_PX, c, rad)
        panels, drawn = [], {}
        for v in STATE_VIEWS:
            img = Image.fromarray(views[v][0])
            drawn[v] = _draw_labels(img, views[v][1], label_points(labels, s.poses), STATE_PX)
            _caption(img, v)
            panels.append(img)
        f = f"{n:02d}_state_{s.key.replace(':', '_')}.png"
        _grid(panels, len(panels), s.title).save(out_dir / f)
        manifest["images"].append({
            "file": f, "kind": "state", "state": s.key, "title": s.title,
            "panels": list(STATE_VIEWS), "panel_px": STATE_PX,
            "labels": drawn,                 # per panel; pixels within that panel
            "poses": {k: _r(T) for k, T in s.poses.items()},
            "gripper_pose": _r(s.gripper_pose) if s.gripper_pose is not None else None})

    # path boundaries, for stages whose entry state was drawn
    entry_state = {s.role: states[k - 1] for k, s in enumerate(states)
                   if k > 0 and s.key.startswith("goal:")}
    for st in stages:
        if st.subgoal is None or st.path is None or st.role not in entry_state:
            continue
        prev = entry_state[st.role]
        base = dict(prev.poses)
        entry = st.entry if st.entry is not None else base[st.mover]
        d_rep, T_rep = representative_pose(st.path, entry, st.subgoal.nominal())
        bounds, pinned = boundary_samples(st.path, d_rep, st.row_labels)
        T_obj_tcp = None
        if grasp is not None and prev.gripper_pose is not None and st.mover == grasp.obj:
            T_obj_tcp = np.linalg.inv(base[st.mover]) @ prev.gripper_pose
        panels, rec_rows = [], []
        for b in bounds:
            frames = b.poses if style == "panels" else [T_rep]
            ghosts = [] if style == "panels" else b.poses
            pose_sets = [{**base, st.mover: T} for T in frames + ghosts]
            grips = [T @ T_obj_tcp for T in frames + ghosts] if T_obj_tcp is not None else []
            c, rad = _extent(scene, pose_sets, gripper if grips else None, grips)
            from scripts.render_candidates import canonical_cameras
            from scripts.render_stage_frames import CAM_RADIUS_FRAC
            cams = canonical_cameras(c, CAM_RADIUS_FRAC * rad)
            view = pick_view(cams, c, row_world_axis(st.path, b.row), rotation=b.row >= 3)
            captions = b.captions if style == "panels" else [
                f"{ROW_NAMES[b.row]} {b.kind}: " + ", ".join(_fmt(b.row, v) for v in b.values)
                + (f" [{b.token}]" if b.token else "")]
            for k, T in enumerate(frames):
                extras = _gripper_extra(gobj, T @ T_obj_tcp if T_obj_tcp is not None else None)
                extras += [{"object": st.mover, "pose": G, "rgba": GHOST_RGBA} for G in ghosts]
                extras += [e for G in ghosts
                           for e in _gripper_extra(gobj, G @ T_obj_tcp if T_obj_tcp is not None else None)]
                img = Image.fromarray(_render(scene, {**base, st.mover: T}, extras,
                                              (view,), PANEL_PX, c, rad)[view][0])
                _caption(img, captions[k])
                panels.append(img)
            rec_rows.append({"row": ROW_NAMES[b.row], "label": b.label, "kind": b.kind,
                             "token": b.token, "values": b.values, "captions": captions,
                             "view": view, "poses": [_r(T) for T in b.poses]})
        if not panels:
            continue
        f = f"{len(manifest['images']):02d}_path_{st.role}.png"
        _grid(panels, 4 if style == "panels" else 3,
              f"{st.role} path region: each row at its bound, others at the "
              f"representative pose ({st.mover} {'shown solid' if style == 'panels' else 'solid = representative, orange = bounds'})"
              ).save(out_dir / f)
        manifest["images"].append({
            "file": f, "kind": "path_boundary", "stage": st.role, "name": st.name,
            "rows": rec_rows, "pinned_rows": pinned,
            "representative": {"displacement": np.round(d_rep, 5).tolist(), "pose": _r(T_rep),
                               "rule": "midpoint of entry and subgoal nominal in path "
                                       "displacement, clamped into the path box"}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# ------------------------------------------------------------------ from critic inputs

def from_preview_inputs(previews) -> tuple[dict[str, np.ndarray], list[StageInput],
                                            dict[str, dict[str, np.ndarray]]]:
    """(initial poses, mover stages, labels) from manip_sim.critic
    StagePreviewInput records (duck-typed: role, emission, compiled, poses,
    symbols, editable). Context stages (editable False: the grasp) are left
    out; the grasp is passed separately as a GraspInput. A record whose
    `compiled` is None, or an editable stage without a passive object
    (gripper frame), cannot be drawn and ends the chain (later stages were
    compiled from its goal); it appears in the manifest's `skipped`.
    Labels: every grounded point of each mover."""
    if not previews:
        raise ValueError("no stages")
    initial = {k: np.asarray(v, float) for k, v in previews[0].poses.items()}
    stages, labels = [], {}
    for si in previews:
        em = si.emission
        if not getattr(si, "editable", True):
            continue                     # context (the grasp): passed as a GraspInput
        cs = si.compiled
        if not em.passive:               # a gripper-frame stage: no mover goal to draw
            stages.append(StageInput(role=si.role, name=em.name, mover=em.active,
                                     passive=None, path=None, subgoal=None,
                                     entry=np.asarray(si.poses[em.active], float),
                                     error="gripper-frame stage (no passive object): "
                                           "not drawn"))
            continue
        stages.append(StageInput(
            role=si.role, name=em.name, mover=em.active, passive=em.passive,
            path=None if cs is None else cs.path, subgoal=None if cs is None else cs.subgoal,
            entry=np.asarray(si.poses[em.active], float),
            error=None if cs is not None else "not compiled"))
        labels.setdefault(em.active, {k: np.asarray(v, float)
                                      for k, v in si.symbols[em.active].points.items()})
    return initial, stages, labels


# ------------------------------------------------------------------ JSON inputs

def _tsr_from(doc):
    from manip_sim.tsr import TSR
    if doc is None:
        return None
    return TSR(T0_w=np.asarray(doc["T0_w"], float), Bw=np.asarray(doc["Bw"], float),
               Tw_e=np.asarray(doc.get("Tw_e", np.eye(4)), float), name=doc.get("name", ""))


@dataclass
class _Obj:
    name: str
    asset: Path

    @property
    def xml(self) -> Path:
        return self.asset / f"{self.name}.xml"


class PreviewScene:
    """What the renderer reads: per object an asset dir holding <name>.xml
    and meshes/<name>_visual.obj, and a table slab for the ground cue."""

    def __init__(self, asset_dirs: dict, table_top_z: float, table_size):
        self.objects = {n: _Obj(n, Path(d)) for n, d in asset_dirs.items()}
        self.asset_dirs = {n: Path(d) for n, d in asset_dirs.items()}
        self.table_top_z = float(table_top_z)
        self.table_size = tuple(float(v) for v in table_size)

    def fixed_poses(self):
        raise RuntimeError("PreviewScene has no spawn poses: pass the poses")


def load_inputs(path: Path) -> dict:
    """preview_inputs.json -> render_preview kwargs. Schema:
    {"scene": {"asset_dirs": {obj: dir}, "table_top_z": z, "table_size": [x, y, z]},
     "initial": {obj: 4x4},
     "stages": [{"role", "name", "mover", "passive", "path": {T0_w, Tw_e, Bw} | null,
                 "subgoal": {...} | null, "entry": 4x4 | null, "error": str | null}],
     "grasp": {"obj": name, "tcp_pose": 4x4} | null,
     "labels": {obj: {point: [x, y, z]}},
     "gripper": {"kind": "sim"} | {"kind": "urdf", "urdf": path, "root_link": "tool0",
                 "mesh_dir": path} | null}"""
    doc = json.loads(Path(path).read_text())
    sc = doc["scene"]
    stages = [StageInput(role=s["role"], name=s["name"], mover=s["mover"],
                         passive=s.get("passive"), path=_tsr_from(s.get("path")),
                         subgoal=_tsr_from(s.get("subgoal")),
                         entry=None if s.get("entry") is None else np.asarray(s["entry"], float),
                         error=s.get("error"),
                         row_labels={int(k): v for k, v in (s.get("row_labels") or {}).items()})
              for s in doc["stages"]]
    g = doc.get("grasp")
    gr = doc.get("gripper")
    gripper = None
    if gr and gr.get("kind") == "sim":
        gripper = sim_gripper(gr.get("xml"))
    elif gr and gr.get("kind") == "urdf":
        gripper = gripper_from_urdf(gr["urdf"], gr.get("root_link", "tool0"), gr.get("mesh_dir"))
    return {"scene": PreviewScene(sc["asset_dirs"], sc["table_top_z"], sc["table_size"]),
            "initial": {k: np.asarray(v, float) for k, v in doc["initial"].items()},
            "stages": stages,
            "grasp": None if g is None else GraspInput(g["obj"], np.asarray(g["tcp_pose"], float)),
            "labels": {o: {p: np.asarray(x, float) for p, x in pts.items()}
                       for o, pts in (doc.get("labels") or {}).items()},
            "gripper": gripper}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, help="preview_inputs.json (see load_inputs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", choices=BOUNDARY_STYLES, default="panels")
    a = ap.parse_args(argv)
    kw = load_inputs(Path(a.inputs))
    m = render_preview(out_dir=Path(a.out), style=a.style, **kw)
    print(f"[preview] {len(m['images'])} image(s) -> {a.out}; skipped: "
          f"{[s['role'] for s in m['skipped']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
