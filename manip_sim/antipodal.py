"""Analytic antipodal grasp synthesis on a metric object mesh.

No learned model: grasps come from the mesh geometry alone, so the same code
serves sim meshes and perception's reconstructed metric-scale meshes.

    1  sample surface points p1 (area-weighted) with outward normals n1
    2  cast a ray from p1 inward along -n1; its first hit p2 is the opposite
       contact. Keep the pair when the width fits the gripper and both
       normals lie inside the friction cone of the closing axis a = p1->p2
    3  around each pair: approach directions perpendicular to a, and depths
       (where along the pad the contact sits); drop approaches from below
       and poses whose gripper boxes cross the table, with the fingers in
       the approach state (Gripper.approach_state) and closed on the object
       The gripper's geometry comes from the sim's measured states
       (manip_sim/data/sim_2f85_closing.json, scripts/measure_gripper_closing.py):
       every collision geom as an oriented box in the grip-site frame. The
       closed state at a grasp width is the one measured with the gripper
       closed on a cylinder of that diameter. The sim 2F-85's pads do not
       stay parallel: closed on a 19 mm cylinder they are toed in by 19-22
       deg (fingertip ends closer together), so the contact face is the
       pad's tilted inner face, not a plane across the closing axis
    4  score each pose in [0, 1] by three terms, combined as a geometric mean
       so a grasp that fails one term ranks low:
         antipodality  1 - worst contact angle / friction-cone angle
         COM distance  1 - (distance from the center of mass to the closing
                       axis) / (half the object's longest extent)
         pad support   the smaller fraction, over the two pads, of a grid on
                       the pad's inner face that has object surface within
                       PAD_SUPPORT_M of the pad once it touches. Rays run
                       along each pad's own normal (with its measured tilt);
                       the pad lands where the object first meets its plane.
                       A pose is dropped when, inside a pad's footprint, the
                       object stands more than PAD_LAND_M proud of its own
                       tangent plane at the sampled contact (a step, e.g. a
                       cap edge): tapers and curved faces pass, steps do not
    5  drop poses whose gripper geoms (boxes; mesh geoms as the convex hulls
       MuJoCo collides with) pass through the object: in the approach state,
       and the closed fingers with each finger moved to where its pad lands
       (pads and fingertips may press FINGER_CONTACT_TOL_M in). Then pick a
       spatially spread set (best first; contact midpoints at least SPREAD_M
       apart within each approach bin: top-down, oblique, side) so the
       planner's lookahead can choose a grasp that suits the downstream
       goals, not only the best-scored spot

Poses are grip-site frames (the frame the planner's IK targets), as in
grasping.py: +z = approach (out of the palm), +x = closing axis, +y = z x x.
Everything is numpy (trimesh only loads and samples the mesh); the ray caster
is a vectorized Moller-Trumbore with a per-chunk bounding-box prefilter,
since trimesh's own needs rtree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The sim 2F-85's collision boxes in the planner's default (partly open) state,
# site-frame AABBs (mm -> m), as measured before the closing table existed.
# Kept as the reference the measured table is validated against
# (tests/test_antipodal.py); the planner itself reads the table.
_MM = 1e-3
SIM_2F85_BOXES = tuple((np.array(lo) * _MM, np.array(hi) * _MM) for lo, hi in (
    ((-43.3, -37.5, -149.5), (43.8, 37.5, -53.6)),     # hand
    ((20.3, -12.0, -104.0), (71.6, 12.0, -68.3)),      # left outer knuckle
    ((51.5, -13.5, -93.8), (68.8, 13.5, -33.4)),       # left outer finger
    ((22.5, -7.5, -52.5), (65.6, 7.5, 7.6)),           # left inner finger
    ((32.3, -11.0, -26.5), (39.0, 11.0, 11.0)),        # left fingertip
    ((32.0, -10.0, -24.5), (34.3, 10.0, 9.0)),         # left pad
    ((4.2, -19.5, -92.0), (58.3, 19.5, -31.7)),        # left inner knuckle
    ((-71.6, -12.0, -104.0), (-20.3, 12.0, -68.4)),    # right outer knuckle
    ((-68.8, -13.5, -93.8), (-51.5, 13.5, -33.4)),     # right outer finger
    ((-65.6, -7.5, -52.4), (-22.6, 7.5, 7.6)),         # right inner finger
    ((-39.1, -11.0, -26.5), (-32.4, 11.0, 11.0)),      # right fingertip
    ((-34.4, -10.0, -24.5), (-32.1, 10.0, 9.0)),       # right pad
    ((-58.3, -19.5, -92.0), (-4.2, 19.5, -31.7)),      # right inner knuckle
))
SIM_2F85_BOX_NAMES = ("hand", "left_outer_knuckle", "left_outer_finger", "left_inner_finger",
                      "left_fingertip", "left_fingerpad", "left_inner_knuckle",
                      "right_outer_knuckle", "right_outer_finger", "right_inner_finger",
                      "right_fingertip", "right_fingerpad", "right_inner_knuckle")

CLOSING_TABLE = Path(__file__).parent / "data" / "sim_2f85_closing.json"
SIDES = ("left", "right")            # left = site +x side, right = site -x side
_DISTAL = ("inner_finger", "fingertip", "fingerpad")   # the link that carries the pad


# ---------------------------------------------------------------- finger geometry

def _rot_y(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _log_so3(R: np.ndarray) -> np.ndarray:
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * ang / (2.0 * np.sin(ang))


def _exp_so3(w: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(w))
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * K @ K


@dataclass(frozen=True, eq=False)
class PadFace:
    """A pad's inner (object-side) face, site frame."""
    side: str
    center: np.ndarray        # face center
    normal: np.ndarray        # unit, out of the pad toward the object (about -x on the left)
    e_len: np.ndarray         # unit, along the pad, toward its fingertip end (+z)
    e_wid: np.ndarray         # unit, across the pad (about site y)
    half_len: float
    half_wid: float

    @property
    def tilt_deg(self) -> float:
        """Rotation of the face about site y away from parallel to the approach;
        positive = toed in (the fingertip end nearer the other pad)."""
        sx = -1.0 if self.side == "left" else 1.0
        return float(np.degrees(np.arctan2(sx * self.e_len[0], self.e_len[2])))

    def corners(self) -> np.ndarray:
        return np.array([self.center + a * self.half_len * self.e_len + b * self.half_wid * self.e_wid
                         for a in (-1, 1) for b in (-1, 1)])

    def z_extent(self) -> tuple:
        r = self.half_len * abs(self.e_len[2]) + self.half_wid * abs(self.e_wid[2])
        return float(self.center[2] - r), float(self.center[2] + r)


@dataclass(frozen=True, eq=False)
class FingerState:
    """Every gripper collision geom as an oriented box in the grip-site frame:
    center c, axes R (columns), half-sizes h. The pads and fingertips are true
    boxes; a mesh geom's box is its mesh's own bounding box (conservative,
    used for the table check), and `hulls` holds the convex hull MuJoCo
    collides with (used for the object checks)."""
    names: tuple
    c: np.ndarray             # (G, 3)
    R: np.ndarray             # (G, 3, 3)
    h: np.ndarray             # (G, 3)
    hulls: dict = field(default_factory=dict)   # mesh geom -> (A, d): A q + d <= 0 inside, box frame

    @classmethod
    def from_json(cls, obbs: dict, hulls: dict | None = None) -> "FingerState":
        names = tuple(obbs)
        hs = {k: (np.array(v, float)[:, :3], np.array(v, float)[:, 3]) for k, v in (hulls or {}).items()}
        return cls(names, np.array([obbs[k]["c"] for k in names], float),
                   np.array([obbs[k]["R"] for k in names], float).reshape(-1, 3, 3),
                   np.array([obbs[k]["h"] for k in names], float), hs)

    def index(self, part: str) -> int:
        return next(i for i, k in enumerate(self.names) if k.startswith(part))

    def side_of(self, i: int):
        s = self.names[i].split("_")[0]
        return s if s in SIDES else None

    def corners(self) -> np.ndarray:
        """The boxes' corners (8 per geom), site frame."""
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return (self.c[:, None, :] + np.einsum("gij,gkj->gki", self.R, s[None] * self.h[:, None, :])
                ).reshape(-1, 3)

    def aabbs(self) -> list:
        """Site-frame AABBs (lo, hi) of the oriented boxes (conservative)."""
        k = self.corners().reshape(-1, 8, 3)
        return [(b.min(axis=0), b.max(axis=0)) for b in k]

    def pad(self, side: str) -> PadFace:
        i = self.index(f"{side}_fingerpad")
        c, R, h = self.c[i], self.R[i], self.h[i]
        kn = int(np.argmin(h))                                  # the pad's thickness axis
        n = R[:, kn] * (-np.sign(c[0] * R[0, kn]) or 1.0)       # toward the other pad
        rest = [k for k in range(3) if k != kn]
        kw = max(rest, key=lambda k: abs(R[1, k]))              # across: about site y
        kl = next(k for k in rest if k != kw)
        e_len = R[:, kl] * (np.sign(R[2, kl]) or 1.0)
        e_wid = R[:, kw] * (np.sign(R[1, kw]) or 1.0)
        return PadFace(side, c + n * h[kn], n, e_len, e_wid, float(h[kl]), float(h[kw]))

    @property
    def gap(self) -> float:
        """Distance between the pads' inner faces along the closing axis at
        their closest (an object wider than this cannot pass between them)."""
        return float(self.pad("left").corners()[:, 0].min() - self.pad("right").corners()[:, 0].max())

    def gap_at(self, z: float) -> float:
        """Distance between the inner-face planes along the closing axis at
        site height z (in the closing plane, y = 0)."""
        x = []
        for side in SIDES:
            pf = self.pad(side)
            x.append(pf.center[0] + (z - pf.center[2]) * pf.e_len[0] / pf.e_len[2])
        return float(x[0] - x[1])

    @property
    def gap_center(self) -> float:
        """Distance between the inner faces' centers along the closing axis."""
        return float(self.pad("left").center[0] - self.pad("right").center[0])

    def tilt_deg(self) -> float:
        """Mean pad tilt over the two pads (positive = toed in)."""
        return 0.5 * (self.pad("left").tilt_deg + self.pad("right").tilt_deg)

    def moved(self, offsets: dict) -> "FingerState":
        """Each side's geoms translated by offsets[side] (site frame)."""
        c = self.c.copy()
        for i in range(len(self.names)):
            s = self.side_of(i)
            if s in offsets:
                c[i] = c[i] + np.asarray(offsets[s], float)
        return FingerState(self.names, c, self.R, self.h, self.hulls)

    def parallel(self) -> "FingerState":
        """The same state with each pad's distal link (inner finger, fingertip,
        pad) rotated about site y around its pad face center to zero tilt: the
        parallel-pad gripper the planner assumed before the tilt was measured."""
        c, R = self.c.copy(), self.R.copy()
        for side in SIDES:
            pf = self.pad(side)
            sx = -1.0 if side == "left" else 1.0
            Ry = _rot_y(-sx * np.radians(pf.tilt_deg))    # undoes the toe-in
            for i, k in enumerate(self.names):
                if k.startswith(side) and any(d in k for d in _DISTAL):
                    c[i] = pf.center + Ry @ (c[i] - pf.center)
                    R[i] = Ry @ R[i]
        return FingerState(self.names, c, R, self.h, self.hulls)


def _interp(s0: FingerState, s1: FingerState, t: float) -> FingerState:
    if t <= 0.0:
        return s0
    if t >= 1.0:
        return s1
    R = np.array([R0 @ _exp_so3(t * _log_so3(R0.T @ R1)) for R0, R1 in zip(s0.R, s1.R)])
    return FingerState(s0.names, (1 - t) * s0.c + t * s1.c, R, (1 - t) * s0.h + t * s1.h, s0.hulls)


def _interp_rows(rows: tuple, key: float) -> FingerState:
    """Linear interpolation over (key, state) rows sorted by key; clamped."""
    keys = np.array([k for k, _ in rows])
    g = float(np.clip(key, keys[0], keys[-1]))
    i = int(np.clip(np.searchsorted(keys, g), 1, len(rows) - 1))
    (k0, s0), (k1, s1) = rows[i - 1], rows[i]
    return _interp(s0, s1, 0.0 if k1 == k0 else (g - k0) / (k1 - k0))


@lru_cache(maxsize=4)
def load_table(path: str) -> SimpleNamespace:
    """The measured gripper states: default (the planner's partly open
    state), free_air ((gap, state), gap ascending), open (the widest free-air
    state), on_object ((object width, state), width ascending; only closings held
    by the pads and fingertips alone)."""
    d = json.loads(Path(path).read_text())
    hulls = d.get("hulls", {})
    free = sorted(((r["gap_m"], FingerState.from_json(r["obbs"], hulls)) for r in d["free_air"]
                   if r["gap_m"] > 0.0), key=lambda r: r[0])
    held = sorted(((r["width_m"], FingerState.from_json(r["obbs"], hulls)) for r in d["on_object"]
                   if r["pad_grasp"]), key=lambda r: r[0])
    return SimpleNamespace(default=FingerState.from_json(d["default"]["obbs"], hulls),
                           free_air=tuple(free), open=free[-1][1], on_object=tuple(held))


@dataclass(frozen=True)
class Gripper:
    """A gripper as its measured finger states; SIM_2F85 (all defaults) is the
    sim's Robotiq 2F-85. approach_state picks the fingers the approach is
    checked with (table and object): "open" is the default; "default" is the
    planner's partly open state, the earlier behavior."""
    table: Path = CLOSING_TABLE        # measured finger states (see load_table)
    approach_state: str = "open"       # fingers while approaching: "open" (widest
                                       # measured) or "default" (the planner's partly open)
    width_margin: float = 0.004        # per side, between pad and object
    parallel_pads: bool = False        # analysis variant: closed pads forced to zero tilt

    def _t(self) -> SimpleNamespace:
        return load_table(str(self.table))

    def approach(self) -> FingerState:
        if self.approach_state not in ("open", "default"):
            raise ValueError(f"approach_state must be 'open' or 'default', not {self.approach_state!r}")
        return getattr(self._t(), self.approach_state)

    def closed(self, width: float) -> FingerState:
        """The fingers closed on an object `width` wide (interpolated between
        the closings measured on cylinders of nearby diameters)."""
        s = _interp_rows(self._t().on_object, width)
        return s.parallel() if self.parallel_pads else s

    def free_air(self, gap: float) -> FingerState:
        """The fingers closed in free air to this pad gap (FingerState.gap)."""
        return _interp_rows(self._t().free_air, gap)

    @property
    def max_width(self) -> float:
        t = self._t()
        return min(self.approach().gap - 2 * self.width_margin, t.on_object[-1][0])


SIM_2F85 = Gripper()

MU = 0.5                       # pad / object friction (cone half-angle ~26.6 deg)
MIN_WIDTH_M = 0.003
N_DEPTHS = 4                  # contact positions along the contact face (site z)
DEPTH_MARGIN_M = 0.001        # kept from either end of the contact face
N_APPROACH = 12
MAX_UP_SIN = np.sin(np.radians(10.0))     # approach may point up at most 10 deg
TABLE_CLEAR_M = 0.002
OBJECT_CLEAR_M = 0.001
FINGER_CONTACT_TOL_M = 0.0015  # pads / fingertips may press this far into the object
PAD_SUPPORT_M = 0.003
PAD_LAND_M = 0.001            # object proud of its contact plane by more is a step
PAD_RAY_START_M = 0.010       # pad rays start this far behind the pad plane at the contact
PAD_GRID = (5, 7)              # rays across x along each pad
SPREAD_M = 0.02
POSES_PER_CELL = 24            # poses kept per 1 cm cell before the pad-support rays
APPROACH_BINS = (-0.7, -0.25)  # world approach z: top-down <= -0.7 < oblique <= -0.25 < side


@dataclass
class Grasp:
    T_body_site: np.ndarray        # grip-site pose in the object's body frame
    width: float
    contacts: tuple                # (p1, p2), body frame; p2 is on the left pad
    depth: float                   # contact z in the site frame
    antipodality: float
    com: float
    pad_support: float
    score: float
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------- rays

def first_hits(V: np.ndarray, F: np.ndarray, origins: np.ndarray, dirs: np.ndarray,
               max_t: float, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit of each ray (unit dirs) with the triangle mesh within
    (1e-6, max_t]: (t, face), t = inf and face = -1 for a miss. Rays are
    processed in spatially sorted chunks; each chunk tests only triangles
    whose bounding box meets the chunk's segment bounding box."""
    origins, dirs = np.asarray(origins, float), np.asarray(dirs, float)
    n = len(origins)
    t_out, f_out = np.full(n, np.inf), np.full(n, -1)
    if n == 0:
        return t_out, f_out
    tri = V[F]                                       # (T, 3, 3)
    tlo, thi = tri.min(axis=1), tri.max(axis=1)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    order = np.lexsort(np.round(origins / 0.01).T[::-1])
    for k in range(0, n, chunk):
        idx = order[k:k + chunk]
        o, d = origins[idx], dirs[idx]
        ends = o + d * max_t
        lo = np.minimum(o, ends).min(axis=0) - 1e-6
        hi = np.maximum(o, ends).max(axis=0) + 1e-6
        sel = np.where(np.all((thi >= lo) & (tlo <= hi), axis=1))[0]
        if len(sel) == 0:
            continue
        v0, E1, E2 = tri[sel, 0], e1[sel], e2[sel]                 # (S, 3)
        p = np.cross(d[:, None, :], E2[None])                      # (R, S, 3)
        det = np.einsum("rsk,sk->rs", p, E1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o[:, None, :] - v0[None]
        u = np.einsum("rsk,rsk->rs", s, p) * inv
        q = np.cross(s, E1[None])
        v = np.einsum("rk,rsk->rs", d, q) * inv
        t = np.einsum("sk,rsk->rs", E2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t <= max_t)
        t = np.where(hit, t, np.inf)
        j = np.argmin(t, axis=1)
        tb = t[np.arange(len(idx)), j]
        t_out[idx] = tb
        f_out[idx] = np.where(np.isfinite(tb), sel[j], -1)
    return t_out, f_out


# ---------------------------------------------------------------- helpers

def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _angle(a, b):
    return np.arccos(np.clip(np.sum(_unit(a) * _unit(b), axis=-1), -1.0, 1.0))


def _perp(a):
    """A unit vector perpendicular to each row of a."""
    ref = np.where(np.abs(a[:, [2]]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    return _unit(np.cross(a, ref))


def center_of_mass(mesh) -> np.ndarray:
    """Volume centroid; the convex hull's when the mesh is not watertight."""
    return np.asarray(mesh.center_mass if mesh.is_watertight
                      else mesh.convex_hull.center_mass, float)


def _contact_z(S: FingerState) -> tuple:
    """The contact face's extent along the approach (site z): where both pads'
    inner faces reach."""
    zl, zr = S.pad("left").z_extent(), S.pad("right").z_extent()
    return max(zl[0], zr[0]), min(zl[1], zr[1])


def intrusions(S: FingerState, local: np.ndarray, clear: float = OBJECT_CLEAR_M,
               contact_tol: float | None = FINGER_CONTACT_TOL_M) -> list:
    """Names of the geoms of S that object points (site frame) lie inside,
    each shrunk by `clear`: mesh geoms as the convex hull MuJoCo collides with
    (their box when no hull was measured), the rest as boxes. With contact_tol
    set (closed fingers), pads and fingertips touch the object at the
    contacts: a point counts there only when it is more than contact_tol
    behind the box's inner face."""
    hit = []
    for i, k in enumerate(S.names):
        q = (local - S.c[i]) @ S.R[i]                          # in the box's own axes
        h = S.h[i]
        side = S.side_of(i)
        if contact_tol is not None and side and ("fingerpad" in k or "fingertip" in k):
            n = S.pad(side).normal @ S.R[i]                    # inner-face normal, box axes
            kn = int(np.argmax(np.abs(n)))
            lat = [j for j in range(3) if j != kn]
            depth = h[kn] - np.sign(n[kn]) * q[:, kn]           # behind the inner face
            inside = (np.all(np.abs(q[:, lat]) < h[lat] - clear, axis=1)
                      & (depth > contact_tol) & (depth < 2 * h[kn]))
        elif k in S.hulls:                                     # the box first: cheap
            q = q[np.all(np.abs(q) < h - clear, axis=1)]
            A, d = S.hulls[k]
            inside = np.all(q @ A.T + d < -clear, axis=1) if len(q) else q[:, 0] > 0
        else:
            inside = np.all(np.abs(q) < h - clear, axis=1)
        if np.any(inside):
            hit.append(k)
    return hit


def closed_state_of(g: Grasp, gripper: Gripper = SIM_2F85) -> FingerState:
    """The closed fingers of a returned grasp as the synthesis checked them:
    the state at its width, each finger moved to where its pad lands."""
    return gripper.closed(g.width).moved(g.detail["finger_offset_m"])


# ---------------------------------------------------------------- synthesis

def synthesize(V: np.ndarray, F: np.ndarray, T0_body: np.ndarray, table_top_z: float,
               gripper: Gripper = SIM_2F85, n_samples: int = 1500, mu: float = MU,
               top_k: int = 30, seed: int = 0) -> tuple[list[Grasp], dict]:
    """Antipodal grasps on the mesh (body frame, meters) of an object at pose
    T0_body above a table at table_top_z. Returns (spread, best-first grasps;
    funnel counts)."""
    import trimesh
    V, F = np.asarray(V, float), np.asarray(F, int)
    mesh = trimesh.Trimesh(V, F, process=False)
    rng_seed = int(seed)
    max_w = gripper.max_width
    p1, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=rng_seed)
    n1 = mesh.face_normals[fi]
    t, f2 = first_hits(V, F, p1 - n1 * 1e-5, -n1, max_w)
    t = t + 1e-5                                          # measured from p1, not the ray start
    keep = np.isfinite(t) & (t >= MIN_WIDTH_M) & (t <= max_w)
    p1, n1, t, f2 = p1[keep], n1[keep], t[keep], f2[keep]
    p2, n2 = p1 - n1 * t[:, None], mesh.face_normals[f2]
    a = _unit(p2 - p1)                                    # closing axis, p1 -> p2
    th1, th2 = _angle(a, -n1), _angle(a, n2)              # force along +a at p1, -a at p2
    cone = np.arctan(mu)
    ok = (th1 <= cone) & (th2 <= cone)
    funnel = {"surface_samples": n_samples, "opposite_hit": int(keep.sum()),
              "in_friction_cone": int(ok.sum())}
    p1, p2, n1, n2, a = p1[ok], p2[ok], n1[ok], n2[ok], a[ok]
    th = np.maximum(th1[ok], th2[ok])
    if len(a) == 0:
        return [], funnel | {"clear_of_table": 0, "support_tested": 0, "scored": 0,
                             "clear_of_object": 0, "returned": 0}

    com = center_of_mass(mesh)
    d_max = 0.5 * float(np.max(mesh.extents))
    mid = 0.5 * (p1 + p2)
    to_com = com - mid
    d_com = np.linalg.norm(to_com - np.sum(to_com * a, axis=1, keepdims=True) * a, axis=1)
    s_anti = 1.0 - th / cone
    s_com = 1.0 - np.minimum(d_com / d_max, 1.0)
    widths = np.linalg.norm(p2 - p1, axis=1)

    # the closed geometry, once per pair (pairs of equal width share it)
    A = gripper.approach()
    a_corners = A.corners()
    closed_by_w: dict = {}

    def closed_geom(w: float):
        key = float(w)
        if key not in closed_by_w:
            S = gripper.closed(w)
            closed_by_w[key] = (S, S.corners(), _contact_z(S), {s: S.pad(s) for s in SIDES})
        return closed_by_w[key]

    # poses: pair x approach angle x depth (body frame)
    u = _perp(a)
    w = np.cross(a, u)
    phis = np.arange(N_APPROACH) * 2 * np.pi / N_APPROACH
    R0 = T0_body[:3, :3]
    poses, meta = [], []
    for i in range(len(a)):
        _, c_corners, (z_lo, z_hi), _ = closed_geom(widths[i])
        both = np.vstack([a_corners, c_corners])          # approaching, then closed
        for phi in phis:
            z = np.cos(phi) * u[i] + np.sin(phi) * w[i]
            if (R0 @ z)[2] > MAX_UP_SIN:                  # approach from below
                continue
            x = a[i]
            y = np.cross(z, x)
            Rb = np.column_stack([x, y, z])
            # the depth (contact's site z) range: on the contact face, and with
            # every gripper box above the table. A corner's height is linear in
            # the depth: h = h0 - dep * az, az = the approach's world z.
            az = float((R0 @ z)[2])
            h0 = (T0_body @ np.c_[(both @ Rb.T) + mid[i], np.ones(len(both))].T)[2]
            lo_d, hi_d = z_lo + DEPTH_MARGIN_M, z_hi - DEPTH_MARGIN_M
            need = table_top_z + TABLE_CLEAR_M - h0           # required increase in height
            if abs(az) < 1e-9:
                if need.max() > 0:
                    continue
            elif az < 0:                                      # deeper raises the gripper
                lo_d = max(lo_d, float(need.max() / -az))
            else:
                hi_d = min(hi_d, float(-need.max() / az))
            if lo_d > hi_d:
                continue
            for dep in np.linspace(lo_d, hi_d, N_DEPTHS) if hi_d - lo_d > 1e-4 else [lo_d]:
                o = mid[i] - dep * z                      # contact at site z = dep
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = Rb, o
                poses.append(T)
                meta.append((i, float(dep)))
    funnel["clear_of_table"] = len(poses)
    if not poses:
        return [], funnel | {"support_tested": 0, "scored": 0, "clear_of_object": 0, "returned": 0}
    # pad support costs 70 rays a pose: keep the best few poses (by the other two
    # terms) in each 1 cm cell of contact midpoint, so the cap does not pull the
    # set toward the center of mass
    pre = np.array([s_anti[i] * s_com[i] for i, _ in meta])
    cell = [tuple(np.floor(mid[i] / 0.01).astype(int)) for i, _ in meta]
    picked, per_cell = [], {}
    for j in np.argsort(-pre):
        if per_cell.get(cell[j], 0) < POSES_PER_CELL:
            per_cell[cell[j]] = per_cell.get(cell[j], 0) + 1
            picked.append(j)
    poses, meta = [poses[j] for j in picked], [meta[j] for j in picked]
    funnel["support_tested"] = len(poses)

    # pad support for each pose: rays along each pad's own normal from a grid on
    # its inner face, the face first placed through the sampled contact
    n_grid = PAD_GRID[0] * PAD_GRID[1]
    GW, GL = [g.ravel() for g in np.meshgrid(np.linspace(-1, 1, PAD_GRID[0]),
                                             np.linspace(-1, 1, PAD_GRID[1]))]
    origins, dirs, plane_t, shifts = [], [], [], []
    for T, (i, dep) in zip(poses, meta):
        _, _, _, pads = closed_geom(widths[i])
        Rb, o = T[:3, :3], T[:3, 3]
        for side, p, ns in (("left", p2[i], n2[i]), ("right", p1[i], n1[i])):
            pf = pads[side]
            grid = pf.center + (GW * pf.half_wid)[:, None] * pf.e_wid + (GL * pf.half_len)[:, None] * pf.e_len
            shift = float(pf.normal @ (Rb.T @ (p - o) - pf.center))   # face plane -> through p
            org = (grid + pf.normal * (shift - PAD_RAY_START_M)) @ Rb.T + o
            dn = Rb @ pf.normal
            den = float(ns @ dn)                                      # < 0: facing the surface
            den = den if den < -0.2 else -0.2
            origins.append(org)
            dirs.append(np.tile(dn, (n_grid, 1)))
            plane_t.append(((p - org) @ ns) / den)                    # ray distance to p's tangent plane
            shifts.append(shift)
    t_hit, _ = first_hits(V, F, np.vstack(origins), np.vstack(dirs),
                          PAD_RAY_START_M + PAD_SUPPORT_M, chunk=2 * n_grid)
    t_hit = t_hit.reshape(len(poses), 2, n_grid)
    plane_t = np.array(plane_t).reshape(len(poses), 2, n_grid)
    shifts = np.array(shifts).reshape(len(poses), 2)
    hit = np.isfinite(t_hit)
    rel = t_hit - PAD_RAY_START_M                 # along the pad normal, from its plane through p
    land = np.where(hit, rel, np.inf).min(axis=2)                     # where each pad touches
    # a step: object standing proud of its own tangent plane at the contact
    proud = np.where(hit, t_hit - plane_t, np.inf).min(axis=2)
    lands_off = (proud < -PAD_LAND_M).any(axis=1)
    sup = (hit & (rel - land[:, :, None] <= PAD_SUPPORT_M)).mean(axis=2).min(axis=1)
    sup = np.where(lands_off | ~np.isfinite(land).all(axis=1), 0.0, sup)
    funnel["pad_lands_off_contact"] = int(lands_off.sum())
    idx = np.array([m[0] for m in meta])
    score = np.cbrt(np.clip(s_anti[idx], 0, 1) * np.clip(s_com[idx], 0, 1) * np.clip(sup, 0, 1))
    funnel["scored"] = int((score > 0).sum())

    # best first: keep a spread set of poses whose gripper does not pass through
    # the object (approach state; closed fingers moved to where their pads land).
    # The spread test is cheap and needs only the kept poses, so it runs first.
    cloud = trimesh.sample.sample_surface(mesh, 4000, seed=rng_seed + 1)[0]
    out, kept, n_tested, n_clear, n_closed_hit = [], [], 0, 0, 0
    for j in np.argsort(-score):
        if score[j] <= 0 or len(out) >= top_k:
            break
        T = poses[j]
        i, dep = meta[j]
        az = float((R0 @ T[:3, 2])[2])
        abin = int(np.searchsorted(APPROACH_BINS, az, side="left"))
        if any(b == abin and np.linalg.norm(mid[i] - m) < SPREAD_M for m, b in kept):
            continue
        n_tested += 1
        local = (cloud - T[:3, 3]) @ T[:3, :3]                     # cloud in the site frame
        if intrusions(A, local, contact_tol=None):
            continue
        S, _, _, pads = closed_geom(widths[i])
        offs = {s: (pads[s].normal * (shifts[j, k] + land[j, k])).tolist()
                for k, s in enumerate(SIDES)}
        if intrusions(S.moved(offs), local):
            n_closed_hit += 1
            continue
        n_clear += 1
        kept.append((mid[i], abin))
        out.append(Grasp(T_body_site=T, width=float(widths[i]),
                         contacts=(p1[i].copy(), p2[i].copy()), depth=float(dep),
                         antipodality=float(s_anti[i]), com=float(s_com[i]),
                         pad_support=float(sup[j]), score=float(score[j]),
                         detail={"approach_bin": ("top-down", "oblique", "side")[abin],
                                 "pad_tilt_deg": round(S.tilt_deg(), 2),
                                 "finger_offset_m": offs}))
    funnel["collision_tested"] = n_tested
    funnel["closed_fingers_hit_object"] = n_closed_hit
    funnel["clear_of_object"] = n_clear
    funnel["returned"] = len(out)
    return out, funnel


def approach_joints(gripper: "Gripper | None" = None) -> dict:
    """The gripper joint positions of the approach state (the sim 2F-85's widest
    free-air state for "open", the planner's spawn state for "default"), for the
    planners to set on their grasp-phase kinematics, so the arm is collision-checked
    with the fingers as execution opens them."""
    g = gripper or Gripper()
    doc = json.loads(CLOSING_TABLE.read_text())
    if g.approach_state == "default":
        return dict(doc["default"]["joints"])
    return dict(max(doc["free_air"], key=lambda r: r["gap_m"])["joints"])


def set_gripper_joints(kin, joints: dict) -> None:
    """Set these gripper joints on a kinematics object's own MjData (each planner
    kinematics object has its own copy, so other phases are unaffected)."""
    import mujoco
    for name, v in joints.items():
        j = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:                       # the table stores names without the robot prefix
            j = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_right_{name}")
        if j < 0:
            raise KeyError(f"gripper joint {name!r} not in the model")
        kin.data.qpos[kin.model.jnt_qposadr[j]] = v
    mujoco.mj_forward(kin.model, kin.data)
