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
  path boundaries  two images per stage with a path TSR, translation rows
                   and rotation rows (<= 6 panels each, under the API's
                   1568 px limit): for each path row, the mover at that
                   row's bound with every other row held at the
                   representative pose (never joint corners). Bounded rows
                   show both bounds; free rotation rows show the mover
                   turned +45 and +90 degrees; free translation rows show it
                   shifted +-10 cm; pinned rows are listed, not drawn.

Collisions: objects and gripper only, against the other objects and the
table (CollisionChecker; the planner checks the arm). A goal whose nominal
collides is drawn at the nearest collision-free pose of its subgoal region
(nearest_free_pose: translation first, then all rows), a path likewise
around the nearest free center; none found keeps the colliding pose. A
path panel that collides is dropped when the center is free (the planner
avoids that pose); with no free center every panel is kept. None of this
is drawn on the images: each image's manifest `notes` carry the facts,
and the caption beside the image is their single home.

Labels: grounded points by NAME (the critic's re-pick token), anchors that
project within LABEL_MERGE_PX merged into one label; each object tagged
with its name. Path panels carry the object tags only.

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
# path images: translation rows and rotation rows apart, <= 6 panels each
PATH_GROUPS = (("translation", (0, 1, 2)), ("rotation", (3, 4, 5)))
PATH_COLS = 3
# the API scales an image down past this long edge; stay under it
MAX_IMAGE_EDGE = 1568
# labels: anchors closer than this (pixels) share one label
LABEL_MERGE_PX = 12.0
TAG_COLOR = (20, 40, 150)
# collision check (objects only, no arm): penetration past COLLISION_TOL_M
# at COLLISION_MIN_POINTS or more surface samples counts; less is resting
# contact or sampling noise
COLLISION_TOL_M = 0.002
COLLISION_MIN_POINTS = 3
COLLISION_SAMPLES = 30000
# nearest collision-free pose (nearest_free_pose): when the drawn pose (a
# goal's nominal, a path's center) collides, FPS samples over the region's
# box, free rows clipped to the display range, checked nearest first. Two
# passes: translation rows only (rotation held at the nominal: a tilted
# object reads as spilling), then, only if that finds nothing, all rows
FREE_ROT_SHOW_RAD = np.pi / 2           # free rotation rows: +-90 deg
ROT_DIST_M_PER_RAD = 0.1                # metric: translation m + 0.1 m per rad
FREE_POSE_SAMPLES_TRANS = 64            # pass (a) budget: translation only
FREE_POSE_SAMPLES_FULL = 64             # pass (b) budget: all rows
FREE_POSE_SAMPLES = FREE_POSE_SAMPLES_FULL   # fps_displacements' default n
FREE_POSE_POOL = 4096                   # uniform pool FPS picks from
FREE_POSE_SEED = 0
FREE_POSE_BISECT = 6                    # refinement checks toward the nominal


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
    mover: str | None = None          # the object a goal state moved


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
                            poses, gripper, st.role, st.mover))
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
    descs: list[str] = field(default_factory=list)   # short per-value text,
                                                     # e.g. "x -10.0 cm (free)"


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
                descs = [f"{ROW_NAMES[i]} {_fmt(i, lo)} (lower bound)",
                         f"{ROW_NAMES[i]} {_fmt(i, vals[1])} (open above)"]
            else:
                vals = [hi, min(d_rep[i], hi) - FREE_TRANS_SHOW_M]
                caps = [f"{ROW_NAMES[i]} at upper bound {_fmt(i, hi)}",
                        f"{ROW_NAMES[i]} open below: {_fmt(i, vals[1])} shown"]
                descs = [f"{ROW_NAMES[i]} {_fmt(i, hi)} (upper bound)",
                         f"{ROW_NAMES[i]} {_fmt(i, vals[1])} (open below)"]
            token, kind = None, "one-sided"
        elif free:
            vals = ([d_rep[i] + np.radians(a) for a in FREE_ROT_SAMPLES_DEG] if i >= 3
                    else [d_rep[i] - FREE_TRANS_SHOW_M, d_rep[i] + FREE_TRANS_SHOW_M])
            caps = [f"{ROW_NAMES[i]} free: {_fmt(i, v - d_rep[i])} shown" for v in vals]
            descs = [f"{ROW_NAMES[i]} {_fmt(i, v - d_rep[i])} (free)" for v in vals]
            token, kind = None, "free"
        else:
            vals = [lo, hi]
            token = _rot_token(0.5 * (hi - lo)) if i >= 3 else None
            tag = f" [{token}]" if token else ""
            caps = [f"{ROW_NAMES[i]} at {'lower' if k == 0 else 'upper'} bound "
                    f"{_fmt(i, v)}{tag}" for k, v in enumerate(vals)]
            descs = [f"{ROW_NAMES[i]} {_fmt(i, v)} ({'lower' if k == 0 else 'upper'} bound)"
                     for k, v in enumerate(vals)]
            kind = "bounded"
        if i in row_labels:
            caps = [f"{c} -- {row_labels[i]}" for c in caps]
        poses = []
        for v in vals:
            d = np.array(d_rep, float)
            d[i] = v
            poses.append(path.T0_w @ displacement_to_pose(d) @ path.Tw_e)
        out.append(Boundary(i, kind, label, [float(v) for v in vals], caps, poses, token,
                            descs))
    return out, pinned


def search_box(tsr, d0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The region's box in displacement coordinates, free sides clipped to
    the display range around d0 (+-FREE_TRANS_SHOW_M, +-FREE_ROT_SHOW_RAD)."""
    lo, hi = np.array(tsr.Bw[:, 0], float), np.array(tsr.Bw[:, 1], float)
    for i in range(6):
        show = FREE_TRANS_SHOW_M if i < 3 else FREE_ROT_SHOW_RAD
        if i >= 3 and hi[i] - lo[i] >= 2 * np.pi - 1e-9:        # rotation-free
            lo[i], hi[i] = d0[i] - show, d0[i] + show
            continue
        if not np.isfinite(lo[i]):
            lo[i] = min(d0[i], hi[i]) - show
        if not np.isfinite(hi[i]):
            hi[i] = max(d0[i], lo[i]) + show
    return lo, hi


def pose_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between displacements (rows of a vs b): translation in m,
    rotation scaled by ROT_DIST_M_PER_RAD (angles the short way round)."""
    d = np.atleast_2d(np.asarray(a, float)) - np.asarray(b, float)
    d[:, 3:] = ROT_DIST_M_PER_RAD * ((d[:, 3:] + np.pi) % (2 * np.pi) - np.pi)
    return np.linalg.norm(d, axis=1)


def fps_displacements(lo, hi, d0, n: int = FREE_POSE_SAMPLES,
                      seed: int = FREE_POSE_SEED, pool: int = FREE_POSE_POOL) -> np.ndarray:
    """n farthest-point samples of the box (pose_distance metric) from a
    seeded uniform pool, then ordered nearest d0 first. Pinned rows stay
    at their value."""
    rng = np.random.default_rng(seed)
    P = lo + rng.random((pool, 6)) * (hi - lo)
    picked = [int(np.argmin(pose_distance(P, d0)))]
    dmin = pose_distance(P, P[picked[0]])
    for _ in range(min(n, pool) - 1):
        k = int(np.argmax(dmin))
        if dmin[k] <= 1e-12:              # the rest repeat picked samples
            break
        picked.append(k)
        dmin = np.minimum(dmin, pose_distance(P, P[k]))
    S = P[picked]
    S = S[pose_distance(S, d0) > 1e-12]   # d0 itself is already known
    return S[np.argsort(pose_distance(S, d0), kind="stable")]


def nearest_free_pose(tsr, d0: np.ndarray, contacts_at,
                      n_trans: int = FREE_POSE_SAMPLES_TRANS,
                      n_full: int = FREE_POSE_SAMPLES_FULL,
                      seed: int = FREE_POSE_SEED, contacts0: list | None = None,
                      n_bisect: int = FREE_POSE_BISECT) -> tuple[np.ndarray, dict]:
    """(displacement to draw, record). `contacts_at(T)` collision-checks a
    pose of the region (T0_w @ D(d) @ Tw_e); `contacts0` (optional) are the
    caller's contacts at d0, so d0 is not checked twice. d0 collision-free:
    d0 itself. Otherwise two passes, each checking FPS samples nearest d0
    first and stopping at the first free one:
      (a) "translation": the translation rows only, rotation held at d0
          (n_trans samples; skipped when every translation row is pinned);
      (b) "full": all rows, only if (a) found nothing (n_full samples;
          skipped when every rotation row is pinned: (a) covered the box).
    The free sample is then refined toward d0 by bisection on the segment
    d0 -> sample (n_bisect checks; the last free point is kept), so the
    shown pose is near the collision boundary, not one FPS spacing away.
    None free: d0 is kept (the region has no free pose the check can find).
    Record: nominal_collides, contacts (at d0), shown (nominal |
    nearest_free | none_free), pass (translation | full | None), n_checked
    (FPS samples, both passes), n_refine (bisection checks), delta (shown
    - d0)."""
    from manip_sim.tsr import displacement_to_pose
    pose = lambda d: tsr.T0_w @ displacement_to_pose(d) @ tsr.Tw_e      # noqa: E731
    d0 = np.asarray(d0, float)
    c0 = contacts_at(pose(d0)) if contacts0 is None else contacts0
    rec = {"nominal_collides": bool(c0), "contacts": c0, "shown": "nominal",
           "pass": None, "n_checked": 0, "n_refine": 0, "delta": None}
    if not c0:
        return d0, rec
    lo, hi = search_box(tsr, d0)
    lo_t, hi_t = lo.copy(), hi.copy()
    lo_t[3:] = hi_t[3:] = d0[3:]
    trans_open = bool(np.any(hi_t[:3] > lo_t[:3]))
    rot_open = bool(np.any(hi[3:] > lo[3:]))
    passes = []
    if trans_open:
        passes.append(("translation", lo_t, hi_t, n_trans))
    if rot_open or not trans_open:
        passes.append(("full", lo, hi, n_full))
    shown = d0
    rec["shown"] = "none_free"
    for name, plo, phi, n in passes:
        for d in fps_displacements(plo, phi, d0, n, seed):
            rec["n_checked"] += 1
            if not contacts_at(pose(d)):
                shown, rec["shown"], rec["pass"] = d, "nearest_free", name
                break
        if rec["shown"] == "nearest_free":
            break
    if rec["shown"] == "nearest_free":
        step = shown - d0
        step[3:] = (step[3:] + np.pi) % (2 * np.pi) - np.pi
        a, b = 0.0, 1.0                    # d0 + a*step collides, d0 + b*step is free
        for _ in range(n_bisect):
            m = 0.5 * (a + b)
            rec["n_refine"] += 1
            if contacts_at(pose(d0 + m * step)):
                a = m
            else:
                b = m
        shown = d0 + b * step
    step = shown - d0
    step[3:] = (step[3:] + np.pi) % (2 * np.pi) - np.pi
    rec["delta"] = {"trans_cm": round(100 * float(np.linalg.norm(step[:3])), 1),
                    "rot_deg": round(float(np.degrees(np.linalg.norm(step[3:]))), 1),
                    "displacement": np.round(step, 5).tolist()}
    return shown, rec


def search_note(rec: dict, verb_contacts: str) -> list[str]:
    """Caption lines for a nearest_free_pose record."""
    if not rec or not rec["nominal_collides"]:
        return []
    bodies = _bodies(rec["contacts"])
    if rec["shown"] == "nearest_free":
        dl = rec["delta"]
        rot = f", {dl['rot_deg']:g} deg" if dl["rot_deg"] >= 0.5 else ""
        if rec.get("pass") == "full":
            rot += "; needed rotation"
        return [f"{verb_contacts} pose collides with {bodies}; nearest collision-free "
                f"pose shown (\u0394 {dl['trans_cm']:g} cm{rot})"]
    return [f"no collision-free pose found in the region ({rec['n_checked']} samples): "
            f"{contact_text(rec['contacts'])}"]


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


# ------------------------------------------------------------------ collisions

@dataclass
class _Surface:
    """Surface samples of one mesh in its own frame, with outward normals."""
    P: np.ndarray
    N: np.ndarray
    tree: object
    lo: np.ndarray
    hi: np.ndarray


def mesh_surface(V: np.ndarray, F: np.ndarray, n: int = COLLISION_SAMPLES,
                 seed: int = 0) -> _Surface:
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, int), process=False)
    P, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    P = np.asarray(P, float)
    return _Surface(P, np.asarray(m.face_normals[fi], float), cKDTree(P),
                    P.min(0), P.max(0))


def penetration(points_world: np.ndarray, surf: _Surface, T: np.ndarray) -> np.ndarray:
    """Depth (m, > 0 inside) of world points in the solid bounded by `surf`
    at pose T, from the nearest surface sample and its outward normal.
    Works on open visual meshes (no fill needed); only points inside the
    mesh's bounding box are tested, the rest get 0."""
    T = np.asarray(T, float)
    Pb = (np.asarray(points_world, float) - T[:3, 3]) @ T[:3, :3]
    m = np.all((Pb > surf.lo - COLLISION_TOL_M) & (Pb < surf.hi + COLLISION_TOL_M), axis=1)
    out = np.zeros(len(Pb))
    if m.any():
        _, i = surf.tree.query(Pb[m])
        out[m] = -np.einsum("ij,ij->i", Pb[m] - surf.P[i], surf.N[i])
    return out


class CollisionChecker:
    """Objects-only collision check for preview poses (no arm): the moved
    objects and the gripper against every other scene object and the table.
    The planner checks the arm; this only keeps the critic from reading a
    pose the planner will never use as an allowed one.

    Each object is its visual mesh's surface samples with outward normals;
    a pair collides when at least COLLISION_MIN_POINTS samples of either
    lie deeper than COLLISION_TOL_M inside the other (both directions, so a
    thin wall is caught by the thick side). The table is the half-space
    below table_top_z over the table's footprint. Pairs inside the held
    unit (the gripper and the object it holds) are not checked."""

    def __init__(self, scene, gripper: GripperModel | None = None,
                 tol: float = COLLISION_TOL_M, min_points: int = COLLISION_MIN_POINTS):
        self.scene, self.gripper = scene, gripper
        self.tol, self.min_points = float(tol), int(min_points)
        self._surf: dict[str, _Surface] = {}

    def surface(self, name: str) -> _Surface:
        if name not in self._surf:
            if name == "gripper":
                V, F = self.gripper.V, self.gripper.F
            else:
                from scripts.render_candidates import load_visual_mesh
                V, F = load_visual_mesh(name, self.scene.asset_dirs[name])
            self._surf[name] = mesh_surface(V, F)
        return self._surf[name]

    def _pair(self, a: str, Ta, b: str, Tb) -> float:
        sa, sb = self.surface(a), self.surface(b)
        Ta, Tb = np.asarray(Ta, float), np.asarray(Tb, float)
        best = 0.0
        for s1, T1, s2, T2 in ((sa, Ta, sb, Tb), (sb, Tb, sa, Ta)):
            d = penetration(s1.P @ T1[:3, :3].T + T1[:3, 3], s2, T2)
            deep = d[d > self.tol]
            if len(deep) >= self.min_points:
                best = max(best, float(deep.max()))
        return best

    def _table(self, name: str, T) -> float:
        tz = getattr(self.scene, "table_top_z", None)
        if tz is None:
            return 0.0
        sx, sy, _ = self.scene.table_size
        T = np.asarray(T, float)
        P = self.surface(name).P @ T[:3, :3].T + T[:3, 3]
        on = (np.abs(P[:, 0]) <= sx / 2) & (np.abs(P[:, 1]) <= sy / 2)
        d = (tz - P[on, 2])
        deep = d[d > self.tol]
        return float(deep.max()) if len(deep) >= self.min_points else 0.0

    def check(self, poses: dict[str, np.ndarray], movers: list[str],
              gripper_pose: np.ndarray | None = None,
              held: str | None = None) -> list[dict]:
        """Contacts [{"a": moved body, "b": other body, "depth_mm"}], sorted;
        empty when collision-free. `movers` are the objects this pose moved;
        `held` is the object the gripper holds: the gripper is not checked
        against it, but a mover that is not the held object is."""
        bodies = [(m, poses[m]) for m in movers if m in poses]
        if gripper_pose is not None and self.gripper is not None:
            bodies.append(("gripper", gripper_pose))
        others = [o for o in poses if o not in movers and o in self.scene.objects]
        out = []
        for a, Ta in bodies:
            for b in others:
                if a == "gripper" and b == held:
                    continue              # the gripper holds it
                dep = self._pair(a, Ta, b, poses[b])
                if dep > 0:
                    out.append({"a": a, "b": b, "depth_mm": round(1000 * dep, 1)})
            dep = self._table(a, Ta)
            if dep > 0:
                out.append({"a": a, "b": "table", "depth_mm": round(1000 * dep, 1)})
        return sorted(out, key=lambda c: (c["a"], c["b"]))


def _bodies(contacts: list[dict]) -> str:
    """"mug and table": the bodies hit, once each."""
    return " and ".join(dict.fromkeys(c["b"] for c in contacts))


def contact_text(contacts: list[dict], verb: str = "collides with") -> str:
    """"teapot collides with mug and table; gripper collides with mug"."""
    by: dict[str, list[str]] = {}
    for c in contacts:
        bs = by.setdefault(c["a"], [])
        if c["b"] not in bs:
            bs.append(c["b"])
    return "; ".join(f"{a} {verb} {' and '.join(bs)}" for a, bs in by.items())


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


def merge_anchors(anchors: list[tuple[str, tuple[float, float]]],
                  tol_px: float = LABEL_MERGE_PX) -> list[tuple[str, tuple[float, float]]]:
    """Anchors of the SAME object closer than tol_px to a group's first
    anchor join that group: one label "a / b" at the group's mean pixel
    (e.g. a mug's rim and opening centers, which project almost onto each
    other). Points of different objects stay apart: one object's point over
    another's is evidence (e.g. a handle over a rim). Order kept."""
    groups: list[list] = []
    for text, uv in anchors:
        for g in groups:
            if (text.split(".")[0] == g[0][0].split(".")[0]
                    and np.hypot(uv[0] - g[0][1][0], uv[1] - g[0][1][1]) < tol_px):
                g.append((text, uv))
                break
        else:
            groups.append([(text, uv)])
    return [(" / ".join(t for t, _ in g),
             (float(np.mean([u for _, (u, _) in g])), float(np.mean([v for _, (_, v) in g]))))
            for g in groups]


def tag_points(scene, poses: dict[str, np.ndarray], n: int = 400,
               gripper: GripperModel | None = None,
               gripper_pose: np.ndarray | None = None) -> list[tuple[str | None, np.ndarray]]:
    """(object name, world points) for each drawn object: a subsample of its
    visual vertices, to anchor the object-name tag. The gripper, when given,
    is an untagged obstacle (name None): tags avoid sitting on it."""
    out: list[tuple[str | None, np.ndarray]] = []
    if scene is None:
        return out
    for name, T in poses.items():
        if name not in scene.objects:
            continue
        V = _visual_vertices(name, scene.asset_dirs[name])
        V = V[:: max(1, len(V) // n)]
        out.append((name, V @ T[:3, :3].T + T[:3, 3]))
    if gripper is not None and gripper_pose is not None:
        V = gripper.V[:: max(1, len(gripper.V) // n)]
        T = np.asarray(gripper_pose, float)
        out.append((None, V @ T[:3, :3].T + T[:3, 3]))
    return out


TAG_TOP_PX = 48          # tags stay below a panel's caption line


def _tag_anchor(box, others, px) -> tuple[float, float] | None:
    """Above the object's projected box; else below, left or right of it,
    the first spot not inside another drawn body's box (e.g. a mug under a
    teapot, a teapot under the gripper). None when every spot is covered:
    the object is mostly hidden, and a tag would sit on another body."""
    l, t, r, b = box
    u, m = 0.5 * (l + r), 0.5 * (t + b)
    cands = [(u, t - 14.0), (u, b + 14.0), (l - 40.0, m), (r + 40.0, m)]
    for cu, cv in cands:
        if TAG_TOP_PX <= cv <= px - 10 and 10 <= cu <= px - 10 and not any(
                o[0] <= cu <= o[2] and o[1] <= cv <= o[3] for o in others):
            return cu, cv
    return None


def leader_segment(anchor, box, gap: float = 11.0):
    """A short line from a dot (radius ~gap) to the nearest point of its
    label's box, or None when the label touches the dot."""
    ax, ay = anchor
    l, t, r, b = box
    qx, qy = min(max(ax, l), r), min(max(ay, t), b)
    d = float(np.hypot(qx - ax, qy - ay))
    if d <= gap + 2.0:
        return None
    k = gap / d
    return (ax + k * (qx - ax), ay + k * (qy - ay)), (qx, qy)


def _draw_labels(img, cam, pts, px, tags=(), leaders: bool = False) -> tuple[list[dict], list[dict]]:
    """Point labels (names, never numbers: the critic's re-pick token is the
    point name) and object-name tags; one greedy placement for both so they
    do not cover each other. `leaders`: a short line from each point label
    to its dot. Returns (labels, tags) with anchor pixels."""
    from PIL import ImageDraw

    from scripts.render_candidates import project
    from scripts.render_stage_frames import DOT_R, LabelRequest, place_labels
    dr = ImageDraw.Draw(img)
    anchors = []
    for text, xyz in pts:
        uv, _ = project(np.asarray(xyz, float)[None, :], cam, px=px)
        anchors.append((text, (float(uv[0][0]), float(uv[0][1]))))
    reqs = []
    for text, (u, v) in merge_anchors(anchors):
        dr.ellipse([u - DOT_R, v - DOT_R, u + DOT_R, v + DOT_R], fill=(0, 0, 0),
                   outline=(255, 255, 255), width=3)
        reqs.append(LabelRequest((u, v), text, (0, 0, 0), priority=0.4))
    boxes = []
    for name, P in tags:
        uv, depth = project(np.asarray(P, float), cam, px=px)
        uv = uv[depth > 0]
        inside = uv[(uv[:, 0] >= 0) & (uv[:, 0] < px) & (uv[:, 1] >= 0) & (uv[:, 1] < px)]
        if len(inside):
            boxes.append((name, (float(inside[:, 0].min()), float(inside[:, 1].min()),
                                 float(inside[:, 0].max()), float(inside[:, 1].max()))))
    tag_reqs = []
    for k, (name, box) in enumerate(boxes):
        if name is None:
            continue                      # an obstacle only (the gripper)
        others = [b for j, (_, b) in enumerate(boxes) if j != k]
        at = _tag_anchor(box, others, px)
        if at is not None:
            tag_reqs.append(LabelRequest(at, name, TAG_COLOR, priority=1.0))
    allreq = tag_reqs + reqs
    placed = place_labels(img, allreq, [], px=px, with_index=True)
    if leaders:
        for i, box in placed:
            if i < len(tag_reqs):
                continue                  # tags have no dot
            seg = leader_segment(allreq[i].anchor, box, gap=DOT_R + 2)
            if seg is not None:
                dr.line([seg[0], seg[1]], fill=(40, 40, 40), width=2)
    return ([{"text": r.text, "pixel": [round(r.anchor[0], 1), round(r.anchor[1], 1)]}
             for r in reqs],
            [{"text": r.text, "pixel": [round(r.anchor[0], 1), round(r.anchor[1], 1)]}
             for r in tag_reqs])


def _caption(img, text):
    from PIL import ImageDraw

    from scripts.render_candidates import _font
    ImageDraw.Draw(img).text((10, 8), text, fill=(25, 25, 25), font=_font(18),
                             stroke_width=2, stroke_fill=(255, 255, 255))


def _fit(img):
    """Scale an image down so its long edge is at most MAX_IMAGE_EDGE."""
    from PIL import Image
    w, h = img.size
    k = MAX_IMAGE_EDGE / max(w, h)
    return img if k >= 1 else img.resize((int(w * k), int(h * k)), Image.LANCZOS)


def _grid(images: list, cols: int, title: str):
    """Panels in a grid under a title. Collision and search facts are not
    drawn: they live in the manifest's `notes`, and the caption the driver
    writes beside the image is their single home."""
    from PIL import Image, ImageDraw

    from scripts.render_candidates import _font
    w, h = images[0].size
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    top = 36
    out = Image.new("RGB", (cols * w, rows * h + top), (255, 255, 255))
    ImageDraw.Draw(out).text((12, 8), title, fill=(20, 20, 20), font=_font(20))
    for k, im in enumerate(images):
        out.paste(im, ((k % cols) * w, top + (k // cols) * h))
    return _fit(out)


def _gripper_extra(gripper_obj: Path | None, T) -> list[dict]:
    if gripper_obj is None or T is None:
        return []
    return [{"mesh": str(gripper_obj), "pose": T, "rgba": GRIPPER_RGBA}]


def _r(T) -> list:
    return np.round(np.asarray(T, float), 5).tolist()


def _attached(T_obj_tcp, T):
    return T @ T_obj_tcp if T_obj_tcp is not None else None


def render_preview(scene, initial: dict[str, np.ndarray], stages: list[StageInput],
                   labels: dict[str, dict[str, np.ndarray]], out_dir: Path,
                   grasp: GraspInput | None = None, gripper: GripperModel | None = None,
                   style: str = "panels", collisions: bool | CollisionChecker = True) -> dict:
    """Render every state image and the path-boundary images; write the PNGs
    and manifest.json to `out_dir`. Returns the manifest.

    Collisions (objects and gripper only; `collisions=False` or no scene
    turns the check off and the manifest records None). Facts go to each
    image's manifest `notes` (and per-panel records), never onto the image:
      - each goal state (and the grasp) is checked; a colliding goal nominal
        is drawn at the nearest collision-free pose of its subgoal region
        (or kept when none is found), e.g. "nominal pose collides with mug;
        nearest collision-free pose shown (delta 2 cm)";
      - each stage's path center (the representative pose, or the nearest
        free pose when it collides) and each path panel are checked. Free
        center: a colliding panel is dropped and listed as a note line (the
        planner avoids that pose). No free center: every panel is kept and
        the colliding ones are recorded (the critic must see it).
    Path panels come in two images per stage, translation rows and rotation
    rows; an image with nothing left to draw is not written and its lines
    go to the manifest's `path_text_only`."""
    from PIL import Image
    if style not in BOUNDARY_STYLES:
        raise ValueError(f"style {style!r}: one of {BOUNDARY_STYLES}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gobj = gripper.write_obj(out_dir / "gripper.obj") if gripper is not None else None
    states, skipped = chain_states(initial, stages, grasp)
    if isinstance(collisions, CollisionChecker):
        checker = collisions
    elif collisions and scene is not None:
        checker = CollisionChecker(scene, gripper)
    else:
        checker = None
    held = grasp.obj if grasp is not None else None
    manifest = {"style": style, "images": [], "skipped": skipped, "path_text_only": [],
                "gripper": gripper.source if gripper is not None else None,
                "collision_check": None if checker is None else {
                    "bodies": "scene objects, gripper, table (no arm)",
                    "tol_mm": round(1000 * checker.tol, 1),
                    "min_points": checker.min_points}}

    stage_of = {st.role: st for st in stages}
    for n, s in enumerate(states):
        contacts, search = None, None
        # what is DRAWN; the chain (later stages' entries) stays on
        # subgoal.nominal(), as the compiler builds stage k+1 against it
        poses, grip = dict(s.poses), s.gripper_pose
        if checker is not None and s.key != "initial":
            contacts = checker.check(s.poses, [s.mover] if s.mover else [],
                                     s.gripper_pose, held)
            st = stage_of.get(s.role) if s.key.startswith("goal:") else None
            if contacts and st is not None and st.subgoal is not None:
                # a colliding nominal: draw the nearest collision-free pose of
                # the subgoal region (or keep the nominal when none is found)
                # the gripper moves with the mover only when it holds it (as
                # in chain_states); otherwise it stays where the state has it
                T_obj_tcp = (np.linalg.inv(s.poses[s.mover]) @ s.gripper_pose
                             if s.gripper_pose is not None and s.mover == held else None)

                def grip_at(T, s=s, T_obj_tcp=T_obj_tcp):
                    return _attached(T_obj_tcp, T) if T_obj_tcp is not None else s.gripper_pose

                def contacts_at(T, s=s, grip_at=grip_at):
                    return checker.check({**s.poses, s.mover: T}, [s.mover], grip_at(T), held)
                d0 = st.subgoal.displacement(np.asarray(s.poses[s.mover], float))
                d, search = nearest_free_pose(st.subgoal, d0, contacts_at, contacts0=contacts)
                if search["shown"] == "nearest_free":
                    from manip_sim.tsr import displacement_to_pose
                    T = st.subgoal.T0_w @ displacement_to_pose(d) @ st.subgoal.Tw_e
                    poses[s.mover] = T
                    grip = grip_at(T)
        if search is not None:
            notes = search_note(search, "nominal")
        else:
            notes = [contact_text(contacts, "intersects")] if contacts else []
        c, rad = _extent(scene, [poses], gripper, [grip] if grip is not None else [])
        views = _render(scene, poses, _gripper_extra(gobj, grip),
                        STATE_VIEWS, STATE_PX, c, rad)
        tags = tag_points(scene, poses, gripper=gripper, gripper_pose=grip)
        panels, drawn, drawn_tags = [], {}, {}
        for v in STATE_VIEWS:
            img = Image.fromarray(views[v][0])
            drawn[v], drawn_tags[v] = _draw_labels(img, views[v][1],
                                                   label_points(labels, poses), STATE_PX, tags,
                                                   leaders=True)
            _caption(img, v)
            panels.append(img)
        f = f"{n:02d}_state_{s.key.replace(':', '_')}.png"
        _grid(panels, len(panels), s.title).save(out_dir / f)
        manifest["images"].append({
            "file": f, "kind": "state", "state": s.key, "title": s.title,
            "role": s.role, "mover": s.mover,
            "panels": list(STATE_VIEWS), "panel_px": STATE_PX,
            "labels": drawn,                 # per panel; pixels within that panel
            "tags": drawn_tags,
            "collisions": contacts,          # at the nominal (the chained state)
            "search": search,                # nearest_free_pose record, or None
            "notes": notes,
            "poses": {k: _r(T) for k, T in poses.items()},           # as drawn
            "gripper_pose": _r(grip) if grip is not None else None})

    # path boundaries, for stages whose entry state was drawn
    from scripts.render_candidates import canonical_cameras
    from scripts.render_stage_frames import CAM_RADIUS_FRAC
    entry_state = {s.role: states[k - 1] for k, s in enumerate(states)
                   if k > 0 and s.key.startswith("goal:")}
    for st in stages:
        if st.subgoal is None or st.path is None or st.role not in entry_state:
            continue
        prev = entry_state[st.role]
        base = dict(prev.poses)
        entry = st.entry if st.entry is not None else base[st.mover]
        d_rep, T_rep = representative_pose(st.path, entry, st.subgoal.nominal())
        T_obj_tcp = None
        if grasp is not None and prev.gripper_pose is not None and st.mover == grasp.obj:
            T_obj_tcp = np.linalg.inv(base[st.mover]) @ prev.gripper_pose

        def grip_at(T):
            # the gripper moves with the mover only when it holds it (as in
            # chain_states); otherwise it stays at the entry state's pose
            return _attached(T_obj_tcp, T) if T_obj_tcp is not None else prev.gripper_pose

        def check(T):
            if checker is None:
                return None
            return checker.check({**base, st.mover: T}, [st.mover], grip_at(T), held)
        # the center the panels are taken around: the representative pose,
        # or (when it collides) the nearest collision-free pose of the path
        # region; none found -> the representative pose, every panel kept
        if checker is not None:
            d_c, search = nearest_free_pose(st.path, d_rep, check)
        else:
            d_c, search = d_rep, None
        from manip_sim.tsr import displacement_to_pose
        T_c = st.path.T0_w @ displacement_to_pose(d_c) @ st.path.Tw_e
        center = None if search is None else (search["contacts"] if search["shown"] == "none_free"
                                              else [])
        center_hit = bool(center)
        bounds, pinned = boundary_samples(st.path, d_c, st.row_labels)
        rep_rec = {"displacement": np.round(d_rep, 5).tolist(), "pose": _r(T_rep),
                   "rule": "midpoint of entry and subgoal nominal in path "
                           "displacement, clamped into the path box",
                   "collisions": None if search is None else search["contacts"],
                   "search": search,
                   "shown_displacement": np.round(d_c, 5).tolist(), "shown_pose": _r(T_c)}
        for group, rows in PATH_GROUPS:
            gb = [b for b in bounds if b.row in rows]
            if not gb:
                continue
            panels, rec_rows = [], []
            notes = search_note(search, "center") if search else []
            in_collision = []             # center in collision: panels kept, named here
            for b in gb:
                hits = [check(T) for T in b.poses]
                if center_hit:
                    in_collision += [f"{b.descs[k]} with {_bodies(h)}"
                                     for k, h in enumerate(hits) if h]
                dropped = [bool(h) and not center_hit for h in hits]
                for k, h in enumerate(hits):
                    if dropped[k]:
                        notes.append(f"{b.descs[k]}: {contact_text(h)} (the planner avoids this)")
                keep = [T for T, dr_ in zip(b.poses, dropped) if not dr_]
                frames = keep if style == "panels" else ([T_c] if keep else [])
                ghosts = [] if style == "panels" else keep
                kept_vals = [v for v, dr_ in zip(b.values, dropped) if not dr_]
                captions = b.captions if style == "panels" else [
                    f"{ROW_NAMES[b.row]} {b.kind}: " + ", ".join(_fmt(b.row, v) for v in kept_vals)
                    + (f" [{b.token}]" if b.token else "")]
                if frames:
                    pose_sets = [{**base, st.mover: T} for T in frames + ghosts]
                    grips = [g for g in (grip_at(T) for T in frames + ghosts) if g is not None]
                    c, rad = _extent(scene, pose_sets, gripper if grips else None, grips)
                    cams = canonical_cameras(c, CAM_RADIUS_FRAC * rad)
                    view = pick_view(cams, c, row_world_axis(st.path, b.row), rotation=b.row >= 3)
                else:
                    view = None
                kept_caps = ([cap for cap, dr_ in zip(captions, dropped) if not dr_]
                             if style == "panels" else captions)   # overlay: kept values only
                for k, T in enumerate(frames):
                    extras = _gripper_extra(gobj, grip_at(T))
                    extras += [{"object": st.mover, "pose": G, "rgba": GHOST_RGBA} for G in ghosts]
                    extras += [e for G in ghosts
                               for e in _gripper_extra(gobj, _attached(T_obj_tcp, G))]
                    pose = {**base, st.mover: T}
                    rend = _render(scene, pose, extras, (view,), PANEL_PX, c, rad)[view]
                    img = Image.fromarray(rend[0])          # no labels, no tags
                    _caption(img, kept_caps[k])
                    panels.append(img)
                rec_rows.append({
                    "row": ROW_NAMES[b.row], "label": b.label, "kind": b.kind,
                    "token": b.token, "values": b.values, "captions": captions,
                    "view": view, "poses": [_r(T) for T in b.poses],
                    "panels": [{"value": v, "desc": d, "collides": None if h is None else bool(h),
                                "collisions": h, "dropped": dr_}
                               for v, d, h, dr_ in zip(b.values, b.descs, hits, dropped)]})
            if center_hit:
                notes = notes + ["center pose in collision; all panels kept; panels in collision: "
                                 + ("; ".join(in_collision) if in_collision else "none")]
            rec = {"kind": "path_boundary", "stage": st.role, "name": st.name, "group": group,
                   "rows": rec_rows, "pinned_rows": pinned, "notes": notes,
                   # the center the panels use collides (no free pose found)
                   "center_collides": None if center is None else center_hit,
                   "representative": rep_rec}
            if not panels:
                manifest["path_text_only"].append(rec)
                continue
            f = f"{len(manifest['images']):02d}_path_{st.role}_{group}.png"
            title = (f"{st.role} path region, {group} rows: each row at its bound, others at "
                     f"the center pose ({st.mover} "
                     + ("shown solid)" if style == "panels"
                        else "solid = center, orange = bounds)"))
            _grid(panels, PATH_COLS, title).save(out_dir / f)
            manifest["images"].append({"file": f, "title": title, **rec})
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
