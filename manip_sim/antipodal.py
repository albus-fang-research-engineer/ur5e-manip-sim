"""Analytic antipodal grasp synthesis on a metric object mesh.

No learned model: grasps come from the mesh geometry alone, so the same code
serves sim meshes and perception's reconstructed metric-scale meshes.

    1  sample surface points p1 (area-weighted) with outward normals n1
    2  cast a ray from p1 inward along -n1; its first hit p2 is the opposite
       contact. Keep the pair when the width fits the gripper and both
       normals lie inside the friction cone of the closing axis a = p1->p2
    3  around each pair: approach directions perpendicular to a, and depths
       (where along the pad the contact sits); drop approaches from below
       and poses whose gripper boxes cross the table
    4  score each pose in [0, 1] by three terms, combined as a geometric mean
       so a grasp that fails one term ranks low:
         antipodality  1 - worst contact angle / friction-cone angle
         COM distance  1 - (distance from the center of mass to the closing
                       axis) / (half the object's longest extent)
         pad support   the smaller fraction, over the two pads, of a grid on
                       the pad face that has object surface within
                       PAD_SUPPORT_M of the pad plane
    5  drop poses whose gripper boxes pass through the object, then pick a
       spatially spread set (contact midpoints at least SPREAD_M apart,
       best first) so the planner's lookahead can choose a grasp that suits
       the downstream goals, not only the best-scored spot

Poses are grip-site frames (the frame the planner's IK targets), as in
grasping.py: +z = approach (out of the palm), +x = closing axis, +y = z x x.
Everything is numpy (trimesh only loads and samples the mesh); the ray caster
is a vectorized Moller-Trumbore with a per-chunk bounding-box prefilter,
since trimesh's own needs rtree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Robotiq 2F-85 as the sim's collision model has it in the planner's default
# gripper state, measured in the grip-site frame (mm -> m). Pad inner faces at
# x = +-32.0 mm, so the widest graspable object is 64 mm less a margin.
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


@dataclass(frozen=True)
class Gripper:
    boxes: tuple = SIM_2F85_BOXES          # site-frame AABBs (lo, hi), meters
    pad_x: float = 0.032                   # pad inner face, |x| in the site frame
    pad_y: tuple = (-0.010, 0.010)         # pad extent across the closing plane
    pad_z: tuple = (-0.0245, 0.009)        # pad extent along the approach
    width_margin: float = 0.004            # per side, between pad and object

    @property
    def max_width(self) -> float:
        return 2 * (self.pad_x - self.width_margin)


SIM_2F85 = Gripper()

MU = 0.5                       # pad / object friction (cone half-angle ~26.6 deg)
MIN_WIDTH_M = 0.003
DEPTHS_M = (-0.016, -0.008, 0.0, 0.005)   # contact's z in the site frame (on the pad)
N_APPROACH = 12
MAX_UP_SIN = np.sin(np.radians(10.0))     # approach may point up at most 10 deg
TABLE_CLEAR_M = 0.002
OBJECT_CLEAR_M = 0.001
PAD_SUPPORT_M = 0.003
PAD_GRID = (5, 7)              # rays across (y) x along (z) each pad
SPREAD_M = 0.02
POSES_PER_CELL = 24            # poses kept per 1 cm cell before the pad-support rays


@dataclass
class Grasp:
    T_body_site: np.ndarray        # grip-site pose in the object's body frame
    width: float
    contacts: tuple                # (p1, p2), body frame
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


def _box_corners(gripper: Gripper) -> np.ndarray:
    c = []
    for lo, hi in gripper.boxes:
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    c.append([(lo, hi)[sx][0], (lo, hi)[sy][1], (lo, hi)[sz][2]])
    return np.array(c)                                             # (8 * boxes, 3)


def center_of_mass(mesh) -> np.ndarray:
    """Volume centroid; the convex hull's when the mesh is not watertight."""
    return np.asarray(mesh.center_mass if mesh.is_watertight
                      else mesh.convex_hull.center_mass, float)


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
    p1, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=rng_seed)
    n1 = mesh.face_normals[fi]
    t, f2 = first_hits(V, F, p1 - n1 * 1e-5, -n1, gripper.max_width)
    t = t + 1e-5                                          # measured from p1, not the ray start
    keep = np.isfinite(t) & (t >= MIN_WIDTH_M) & (t <= gripper.max_width)
    p1, n1, t, f2 = p1[keep], n1[keep], t[keep], f2[keep]
    p2, n2 = p1 - n1 * t[:, None], mesh.face_normals[f2]
    a = _unit(p2 - p1)                                    # closing axis, p1 -> p2
    th1, th2 = _angle(a, -n1), _angle(a, n2)              # force along +a at p1, -a at p2
    cone = np.arctan(mu)
    ok = (th1 <= cone) & (th2 <= cone)
    funnel = {"surface_samples": n_samples, "opposite_hit": int(keep.sum()),
              "in_friction_cone": int(ok.sum())}
    p1, p2, a, th = p1[ok], p2[ok], a[ok], np.maximum(th1[ok], th2[ok])
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

    # poses: pair x approach angle x depth (body frame)
    u = _perp(a)
    w = np.cross(a, u)
    phis = np.arange(N_APPROACH) * 2 * np.pi / N_APPROACH
    R0 = T0_body[:3, :3]
    corners = _box_corners(gripper)
    poses, meta = [], []
    for i in range(len(a)):
        for phi in phis:
            z = np.cos(phi) * u[i] + np.sin(phi) * w[i]
            if (R0 @ z)[2] > MAX_UP_SIN:                  # approach from below
                continue
            x = a[i]
            y = np.cross(z, x)
            Rb = np.column_stack([x, y, z])
            for dep in DEPTHS_M:
                o = mid[i] - dep * z                      # contact at site z = dep
                cw = (T0_body @ np.c_[(corners @ Rb.T) + o, np.ones(len(corners))].T)[2]
                if cw.min() < table_top_z + TABLE_CLEAR_M:
                    continue
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = Rb, o
                poses.append(T)
                meta.append((i, dep))
    funnel["clear_of_table"] = len(poses)
    if not poses:
        return [], funnel | {"scored": 0, "clear_of_object": 0, "returned": 0}
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

    # pad support for each pose: rays from a grid on each pad plane toward the object
    gy = np.linspace(*gripper.pad_y, PAD_GRID[0])
    gz = np.linspace(*gripper.pad_z, PAD_GRID[1])
    GY, GZ = [g.ravel() for g in np.meshgrid(gy, gz)]
    origins, dirs = [], []
    for T, (i, dep) in zip(poses, meta):
        y, z = T[:3, 1], T[:3, 2]
        off = GY[:, None] * y + (GZ - dep)[:, None] * z    # pad grid relative to the contact
        origins += [p1[i] + off - a[i] * 1e-3, p2[i] + off + a[i] * 1e-3]
        dirs += [np.tile(a[i], (len(GY), 1)), np.tile(-a[i], (len(GY), 1))]
    th_hit, _ = first_hits(V, F, np.vstack(origins), np.vstack(dirs), PAD_SUPPORT_M + 1e-3,
                           chunk=2 * len(GY))
    sup = np.isfinite(th_hit).reshape(len(poses), 2, len(GY)).mean(axis=2).min(axis=1)
    idx = np.array([m[0] for m in meta])
    score = np.cbrt(np.clip(s_anti[idx], 0, 1) * np.clip(s_com[idx], 0, 1) * np.clip(sup, 0, 1))
    funnel["scored"] = int((score > 0).sum())

    # best first: drop poses whose gripper passes through the object, keep a spread set
    cloud = trimesh.sample.sample_surface(mesh, 4000, seed=rng_seed + 1)[0]
    out, mids, n_clear = [], [], 0
    for j in np.argsort(-score):
        if score[j] <= 0 or len(out) >= top_k:
            break
        T = poses[j]
        local = (cloud - T[:3, 3]) @ T[:3, :3]                     # cloud in the site frame
        hit = False
        for lo, hi in gripper.boxes:
            if np.any(np.all((local > lo + OBJECT_CLEAR_M) & (local < hi - OBJECT_CLEAR_M),
                             axis=1)):
                hit = True
                break
        if hit:
            continue
        n_clear += 1
        i, dep = meta[j]
        if any(np.linalg.norm(mid[i] - m) < SPREAD_M for m in mids):
            continue
        mids.append(mid[i])
        out.append(Grasp(T_body_site=T, width=float(np.linalg.norm(p2[i] - p1[i])),
                         contacts=(p1[i].copy(), p2[i].copy()), depth=float(dep),
                         antipodality=float(s_anti[i]), com=float(s_com[i]),
                         pad_support=float(sup[j]), score=float(score[j])))
    funnel["clear_of_object"] = n_clear
    funnel["returned"] = len(out)
    return out, funnel
