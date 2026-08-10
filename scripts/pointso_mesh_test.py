"""Camera-free PointSO smoke test: sample the object MESHES, ship the cloud
to the pointso sidecar, score the predicted semantic directions against the
calibrated frames.json symbols.

No RGB-D camera, no MuJoCo render, no robot — just:

    visual .obj  --area-weighted-sample-->  Nx6 cloud  --ZMQ-->  PointSO
                                                                   |
    frames.json axes  <------- angular error ----------------------+

Because the converter writes meshes in the body frame of mjcf body 'object'
(same frame frames.json declares), the sampled points ARE body coordinates,
and PointSO's prediction comes back in that same frame — directly comparable
to pour_axis / up_axis / handle geometry. PointSO normalizes xyz to the unit
sphere internally (pc_norm), so metric scale is irrelevant; ORIENTATION of
the canonical frame is what it sees. Our assets rest z-up, matching the
renders PointSO was trained on.

Two cloud modes:
  full (default)   points from the whole surface — the clean upper bound
  --partial        Katz hidden-point-removal from a virtual viewpoint —
                   the single-depth-camera ablation (what the RealSense
                   will actually hand SoFar on hardware)

Usage (inside the sim container, pointso sidecar up on the host):

    PYTHONPATH=. python scripts/pointso_mesh_test.py
    PYTHONPATH=. python scripts/pointso_mesh_test.py --objects teapot
    PYTHONPATH=. python scripts/pointso_mesh_test.py --partial --azimuth 45
    PYTHONPATH=. python scripts/pointso_mesh_test.py --save-npz outputs/pointso

Start the sidecar from ur5e-manip-hardware:
    docker compose up -d pointso
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from manip_sim.perception.pointso_client import PointSOClient

ASSETS = Path("assets/objects")

# Per-object instruction sets, and how each instruction maps to ground truth.
# GT spec forms:
#   ("axis", name)            frames.json axes[name].xyz
#   ("toward_point", name)    unit vector from cloud centroid toward
#                             frames.json points[name].xyz (z zeroed for
#                             lateral symbols like handles/spouts)
#   None                      no GT — prediction printed, not scored
INSTRUCTIONS = {
    "teapot": [
        ("spout",         ("axis", "pour_axis")),
        ("pouring water", ("axis", "pour_axis")),
        ("handle",        ("toward_point", "handle_center")),
        ("lid",           ("axis", "up_axis")),
        ("upright",       ("axis", "up_axis")),
    ],
    "mug": [
        ("opening",       ("axis", "up_axis")),
        ("upright",       ("axis", "up_axis")),
        ("handle",        ("derived", "mug_handle")),
    ],
}


# --------------------------------------------------------------------------
# cloud construction
# --------------------------------------------------------------------------

def load_visual_mesh(obj: str) -> trimesh.Trimesh:
    path = ASSETS / obj / "meshes" / f"{obj}_visual.obj"
    if not path.exists():
        raise SystemExit(f"[pointso-test] missing mesh {path} — run the "
                         "asset converter first (meshes/ is gitignored).")
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise SystemExit(f"[pointso-test] {path} did not load as a "
                         "triangle mesh")
    return mesh


def sample_cloud(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    """Area-weighted surface sample -> Nx6 (xyz + rgb in [0,1])."""
    pts, face_idx = trimesh.sample.sample_surface(
        mesh, n, seed=seed)
    # Colors: PointSO input is xyz+rgb. Our converted visual meshes carry no
    # trustworthy per-vertex color, and the semantic-orientation queries we
    # care about are geometry-dominant, so a neutral constant is fine. If
    # the mesh does expose colors, use them.
    rgb = np.full((len(pts), 3), 0.5, np.float32)
    try:
        vc = mesh.visual.to_color().vertex_colors  # may raise / be default
        if vc is not None and len(vc) == len(mesh.vertices):
            face_verts = mesh.faces[face_idx]
            rgb = (np.asarray(vc[:, :3], np.float32)[face_verts]
                   .mean(axis=1) / 255.0)
    except Exception:
        pass
    return np.hstack([pts.astype(np.float32), rgb.astype(np.float32)])


def hidden_point_removal(pts: np.ndarray, cam: np.ndarray,
                         gamma: float = 3.0) -> np.ndarray:
    """Katz et al. spherical-flip HPR: indices of points visible from cam.

    gamma scales the flip radius; larger = more permissive (more points
    kept). 2-4 is the usual range for object-scale clouds.
    """
    from scipy.spatial import ConvexHull
    p = pts - cam[None, :]
    d = np.linalg.norm(p, axis=1, keepdims=True)
    R = d.max() * (10.0 ** gamma) ** 0.1  # gentle exponential radius
    flipped = p + 2.0 * (R - d) * (p / d)
    hull = ConvexHull(np.vstack([flipped, np.zeros(3)]))
    vis = set(hull.vertices.tolist())
    vis.discard(len(pts))  # the camera point itself
    return np.fromiter(vis, dtype=np.int64)


def virtual_camera(cloud_xyz: np.ndarray, azimuth_deg: float,
                   elevation_deg: float, dist: float) -> np.ndarray:
    c = cloud_xyz.mean(axis=0)
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    offset = dist * np.array([np.cos(el) * np.cos(az),
                              np.cos(el) * np.sin(az),
                              np.sin(el)])
    return c + offset


# --------------------------------------------------------------------------
# ground truth from frames.json
# --------------------------------------------------------------------------

def load_frames(obj: str) -> dict:
    return json.loads((ASSETS / obj / "frames.json").read_text())


def derived_mug_handle(mesh: trimesh.Trimesh, frames: dict) -> np.ndarray:
    """Mug handle direction is not annotated; derive it as the mean xy
    direction of the radially FARTHEST vertex band (top 3%). The handle's
    outer loop is the most radially extreme geometry on a mug, and unlike
    an absolute rim-radius threshold this is immune to the centroid being
    dragged toward the handle. Returns None if the band is not meaningfully
    outside the rim (e.g. a handleless cup)."""
    rim_r = frames.get("quantities", {}).get("rim_radius", {}).get("value")
    # Center on the mug's body axis, not the vertex mean: the annotated
    # opening_center IS the axis (and vertex means are density-biased —
    # a finely-meshed handle drags them). Fall back to the area-weighted
    # trimesh centroid.
    oc = frames.get("points", {}).get("opening_center", {}).get("xyz")
    center = np.asarray(oc, float) if oc is not None else mesh.centroid
    v = mesh.vertices - center[None, :]
    r = np.linalg.norm(v[:, :2], axis=1)
    band = v[r >= np.quantile(r, 0.97)]
    if len(band) < 10:
        return None
    if rim_r is not None and band[:, :2].__abs__().max() < 1.1 * rim_r:
        return None  # nothing sticks out past the rim -> no handle
    d = band[:, :2].mean(axis=0)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return None  # extremes are radially symmetric -> no handle
    return np.array([d[0] / n, d[1] / n, 0.0])


def resolve_gt(spec, frames: dict, mesh: trimesh.Trimesh,
               cloud_xyz: np.ndarray):
    if spec is None:
        return None
    kind, name = spec
    if kind == "axis":
        ax = frames.get("axes", {}).get(name)
        if ax is None:
            return None
        v = np.asarray(ax["xyz"], float)
    elif kind == "toward_point":
        pt = frames.get("points", {}).get(name)
        if pt is None:
            return None
        v = np.asarray(pt["xyz"], float) - cloud_xyz.mean(axis=0)
        v[2] = 0.0  # lateral symbol: compare in the horizontal plane
    elif kind == "derived" and name == "mug_handle":
        v = derived_mug_handle(mesh, frames)
        if v is None:
            return None
    else:
        return None
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    c = float(np.clip(np.dot(pred, gt), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--objects", nargs="+",
                    default=sorted(INSTRUCTIONS.keys()),
                    choices=sorted(INSTRUCTIONS.keys()))
    ap.add_argument("--n-points", type=int, default=20000,
                    help="surface samples (PointSO votes on 10k subsets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partial", action="store_true",
                    help="single-viewpoint cloud via hidden point removal")
    ap.add_argument("--azimuth", type=float, default=30.0)
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--cam-dist", type=float, default=0.6)
    ap.add_argument("--addr", default=None,
                    help="override POINTSO_ADDR (tcp://pointso:5668)")
    ap.add_argument("--save-npz", default=None, metavar="DIR",
                    help="dump cloud + predictions per object")
    args = ap.parse_args()

    client = PointSOClient(addr=args.addr)
    client.ping()
    print(f"[pointso-test] server ok at {client.addr}")

    overall = []
    for obj in args.objects:
        mesh = load_visual_mesh(obj)
        frames = load_frames(obj)
        cloud = sample_cloud(mesh, args.n_points, args.seed)

        mode = "full-surface"
        if args.partial:
            cam = virtual_camera(cloud[:, :3], args.azimuth,
                                 args.elevation, args.cam_dist)
            keep = hidden_point_removal(cloud[:, :3], cam)
            cloud = cloud[keep]
            mode = (f"partial az={args.azimuth:g} el={args.elevation:g} "
                    f"({len(cloud)} pts visible)")

        instructions = [i for i, _ in INSTRUCTIONS[obj]]
        preds = client.orient_batch(cloud, instructions)

        print(f"\n=== {obj}  [{mode}, {len(cloud)} pts] ===")
        print(f"{'instruction':<16} {'prediction (body frame)':<28} "
              f"{'gt':<12} {'ang err':>8}")
        for (ins, gt_spec), pred in zip(INSTRUCTIONS[obj], preds):
            gt = resolve_gt(gt_spec, frames, mesh, cloud[:, :3])
            pstr = "[" + " ".join(f"{x:+.3f}" for x in pred) + "]"
            if gt is None:
                print(f"{ins:<16} {pstr:<28} {'—':<12} {'—':>8}")
            else:
                err = angular_error_deg(pred, gt)
                gt_name = gt_spec[1]
                flag = "" if err < 30 else "  <-- check"
                print(f"{ins:<16} {pstr:<28} {gt_name:<12} "
                      f"{err:7.1f}°{flag}")
                overall.append((obj, ins, err))

        if args.save_npz:
            out = Path(args.save_npz)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"pointso_{obj}.npz",
                     cloud=cloud, instructions=np.array(instructions),
                     predictions=preds)
            print(f"[pointso-test] wrote {out / f'pointso_{obj}.npz'}")

    if overall:
        errs = np.array([e for _, _, e in overall])
        print(f"\n[pointso-test] scored {len(errs)} instruction(s): "
              f"mean {errs.mean():.1f}°, median {np.median(errs):.1f}°, "
              f"max {errs.max():.1f}°")


if __name__ == "__main__":
    main()
