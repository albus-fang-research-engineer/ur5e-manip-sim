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

Scoring is on the SIGNED angle: semantic orientations are directed vectors
(upright vs. upside-down differ), so a 180-degree antipode counts as wrong.
A separate diagnostic flag marks near-antipodes ("sign-flip?") so axis-right/
sign-wrong failures are distinguishable from genuinely wrong directions.

Two cloud modes:
  full (default)   points from the whole surface — the clean upper bound
  --partial        Katz hidden-point-removal from a virtual viewpoint —
                   the single-depth-camera ablation (what the RealSense
                   will actually hand SoFar on hardware). The mesh is
                   oversampled 8x before HPR and the visible subset is
                   resampled back to --n-points, so PointSO always sees a
                   training-distribution-sized cloud.

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
from manip_sim.refine import refine_axis

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

# --refine: which fitter snaps each instruction's prediction onto the
# geometry (manip_sim.refine). Only up-type instructions map to a fitter
# here: both bodies are surfaces of revolution about up_axis, so the
# whole unsegmented cloud is a valid fit input (robust loss absorbs the
# handle/spout). Lateral symbols (spout, handle) would need a part-
# segmented subcloud to fit against — that arrives with the segmentation
# stage, so they stay unrefined rather than fit against the wrong region.
REFINE = {
    "lid": "revolution",
    "upright": "revolution",
    "opening": "revolution",
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
                         gamma: float = 2.0) -> np.ndarray:
    """Katz et al. spherical-flip HPR: indices of points visible from cam.

    gamma sets the flip radius as R = d_max * 10**gamma — the standard
    Katz/Open3D parameterization (Open3D's hidden_point_removal takes the
    radius directly, typically 100-1000x the object diameter). 2-4 is the
    usual range; larger keeps more points. NOTE: the previous version
    computed R = d_max * (10**gamma)**0.1 ~= 2*d_max, far too small — the
    spherical flip barely inverted the geometry and the hull kept only a
    ~3% crust of extreme points, which is what tanked the predictions.
    """
    from scipy.spatial import ConvexHull
    p = pts - cam[None, :]
    d = np.linalg.norm(p, axis=1, keepdims=True)
    R = d.max() * (10.0 ** gamma)
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


def camera_frame_rotation(cloud_xyz: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Rows of R map body -> OpenCV camera frame (x right, y down,
    z forward) for a camera at `cam` looking at the cloud centroid.

    Probably unnecessary: OrienText300K training applies random SO(3)
    rotation augmentation, so PointSO is meant to be frame-agnostic. Kept
    as an ablation switch. Send (p - cam) @ R.T, then map predictions back
    with R.T @ d.
    """
    c = cloud_xyz.mean(axis=0)
    z = c - cam
    z = z / np.linalg.norm(z)                      # forward
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))     # right (world z-up hint)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                             # down
    return np.stack([x, y, z])


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
# debug rendering
# --------------------------------------------------------------------------

def save_debug_view(obj: str, dense_xyz: np.ndarray, keep: np.ndarray,
                    cam: np.ndarray, out_dir: Path) -> Path:
    """Three orthographic scatter panels (xy / xz / yz): culled points in
    gray, HPR-visible points in orange, camera as a star with the view ray
    to the cloud centroid. Answers at a glance whether the camera is where
    you think it is (and at a sane scale) and whether the visible set is a
    clean camera-facing shell or degenerate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = np.zeros(len(dense_xyz), bool)
    mask[keep] = True
    c = dense_xyz.mean(axis=0)
    d = np.linalg.norm(dense_xyz - cam[None, :], axis=1)
    lo, hi = dense_xyz.min(axis=0), dense_xyz.max(axis=0)

    panels = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (nx, ny, i, j) in zip(axes, panels):
        ax.scatter(dense_xyz[~mask, i], dense_xyz[~mask, j],
                   s=0.5, c="0.82", rasterized=True, label="culled")
        ax.scatter(dense_xyz[mask, i], dense_xyz[mask, j],
                   s=0.7, c="tab:orange", rasterized=True, label="visible")
        ax.scatter([cam[i]], [cam[j]], marker="*", s=260,
                   c="tab:blue", zorder=5, label="camera")
        ax.plot([cam[i], c[i]], [cam[j], c[j]], "b--", lw=1, zorder=4)
        ax.set_xlabel(nx)
        ax.set_ylabel(ny)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", markerscale=8, fontsize=9)
    fig.suptitle(
        f"{obj}: cam={np.array2string(cam, precision=3)}  "
        f"centroid={np.array2string(c, precision=3)}  "
        f"bbox={np.array2string(hi - lo, precision=3)}\n"
        f"|cam-centroid|={np.linalg.norm(cam - c):.3f}  "
        f"d_min={d.min():.3f}  d_max={d.max():.3f}  "
        f"d_max/d_min={d.max() / max(d.min(), 1e-9):.2f}  "
        f"visible={mask.mean():.1%}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"hpr_debug_{obj}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--objects", nargs="+",
                    default=sorted(INSTRUCTIONS.keys()),
                    choices=sorted(INSTRUCTIONS.keys()))
    ap.add_argument("--n-points", type=int, default=10000,
                    help="points sent to PointSO (10k matches the "
                         "OrienText300K training/eval point count)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partial", action="store_true",
                    help="single-viewpoint cloud via hidden point removal; "
                         "oversamples 8x then resamples the visible subset "
                         "back to --n-points")
    ap.add_argument("--hpr-gamma", type=float, default=2.0,
                    help="HPR flip-radius exponent: R = d_max * 10**gamma "
                         "(Katz/Open3D convention, usual range 2-4)")
    ap.add_argument("--camera-frame", action="store_true",
                    help="express the cloud in an OpenCV camera frame "
                         "(y down, z forward) before sending and rotate "
                         "predictions back to body frame for scoring. "
                         "Likely unneeded (training uses random SO(3) "
                         "augmentation); kept as an ablation.")
    ap.add_argument("--azimuth", type=float, default=30.0)
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--cam-dist", type=float, default=0.6)
    ap.add_argument("--addr", default=None,
                    help="override POINTSO_ADDR (tcp://pointso:5668)")
    ap.add_argument("--save-npz", default=None, metavar="DIR",
                    help="dump cloud + predictions per object")
    ap.add_argument("--tta", type=int, default=1,
                    help="rotation test-time augmentation: query N copies "
                         "of the cloud under random SO(3) rotations, "
                         "average the back-rotated predictions, and report "
                         "per-query spread (training used SO(3) "
                         "augmentation, so single queries sample a "
                         "pose-dependent answer distribution)")
    ap.add_argument("--refine", action="store_true",
                    help="snap each refinable prediction onto a primitive "
                         "fitted to the same cloud (manip_sim.refine) and "
                         "score raw vs refined side by side — the "
                         "semantic layer initializes and signs, the fit "
                         "supplies the metric direction")
    ap.add_argument("--debug-view", default=None, metavar="DIR",
                    help="save an orthographic HPR debug render per object "
                         "(culled vs visible points, camera position, "
                         "scale stats)")
    args = ap.parse_args()

    client = PointSOClient(addr=args.addr)
    client.ping()
    print(f"[pointso-test] server ok at {client.addr}")

    overall = []
    overall_refined = []
    for obj in args.objects:
        mesh = load_visual_mesh(obj)
        frames = load_frames(obj)

        if args.partial:
            # Oversample so the visible subset covers n_points, then trim
            # back down: PointSO's FPS/KNN tokenization expects ~10k-point
            # clouds (FPS to 512 seeds x KNN 32); sending more than that
            # buys nothing, sending less is OOD. The multiplier adapts:
            # low-visibility geometry (large objects, close cameras) just
            # gets sampled denser until the visible set is big enough.
            mult = 8
            while True:
                dense = sample_cloud(mesh, args.n_points * mult, args.seed)
                cam = virtual_camera(dense[:, :3], args.azimuth,
                                     args.elevation, args.cam_dist)
                keep = hidden_point_removal(dense[:, :3], cam,
                                            gamma=args.hpr_gamma)
                if len(keep) >= args.n_points or mult >= 64:
                    break
                mult *= 2
            vis_frac = len(keep) / len(dense)
            cloud = dense[keep]
            if len(cloud) > args.n_points:
                rng = np.random.default_rng(args.seed)
                sel = rng.choice(len(cloud), args.n_points, replace=False)
                cloud = cloud[sel]
            mode = (f"partial az={args.azimuth:g} el={args.elevation:g} "
                    f"({len(keep)}/{len(dense)} visible = {vis_frac:.0%}, "
                    f"sent {len(cloud)})")
            hpr_cam = cam
            if args.debug_view:
                out = save_debug_view(obj, dense[:, :3], keep, cam,
                                      Path(args.debug_view))
                print(f"[pointso-test] wrote {out}")
            if vis_frac < 0.15:
                print(f"[pointso-test] WARNING {obj}: only {vis_frac:.0%} "
                      "of points visible — camera likely too close to the "
                      "object for Katz HPR; try --cam-dist 1.0 or a larger "
                      "--hpr-gamma")
            if len(cloud) < args.n_points:
                print(f"[pointso-test] WARNING {obj}: sent {len(cloud)} < "
                      f"{args.n_points} points even at {mult}x "
                      "oversampling — below PointSO's training point "
                      "count; fix the viewpoint before trusting scores")
        else:
            cloud = sample_cloud(mesh, args.n_points, args.seed)
            mode = "full-surface"
            hpr_cam = None

        # GT geometry stays in body frame regardless of what we send.
        body_xyz = cloud[:, :3].copy()

        R_bc = None
        if args.camera_frame:
            # Reuse the HPR camera when there is one: recomputing from the
            # visible subset's centroid (which shifts toward the camera
            # after culling) made the frame-rotation camera differ from
            # the culling camera.
            cam = hpr_cam if hpr_cam is not None else virtual_camera(
                cloud[:, :3], args.azimuth, args.elevation, args.cam_dist)
            R_bc = camera_frame_rotation(cloud[:, :3], cam)
            cloud = cloud.copy()
            cloud[:, :3] = (cloud[:, :3] - cam[None, :]) @ R_bc.T
            mode += " camera-frame"

        instructions = [i for i, _ in INSTRUCTIONS[obj]]
        spread = None
        if args.tta > 1:
            from scipy.spatial.transform import Rotation
            samples = [np.asarray(client.orient_batch(cloud, instructions),
                                  np.float32)]
            rots = Rotation.random(args.tta - 1, random_state=args.seed)
            for R in (rots if args.tta > 2 else [rots]):
                M = R.as_matrix().astype(np.float32)
                rc = cloud.copy()
                rc[:, :3] = rc[:, :3] @ M.T   # rotate the sent cloud by M
                p = np.asarray(client.orient_batch(rc, instructions),
                               np.float32)
                samples.append(p @ M)         # rotate prediction back
            stack = np.stack(samples)                       # (T, K, 3)
            stack /= np.maximum(
                np.linalg.norm(stack, axis=-1, keepdims=True), 1e-9)
            preds = stack.sum(axis=0)
            preds /= np.maximum(
                np.linalg.norm(preds, axis=-1, keepdims=True), 1e-9)
            cosang = np.clip(np.einsum("tkd,kd->tk", stack, preds), -1, 1)
            spread = np.degrees(np.arccos(cosang)).mean(axis=0)
            mode += f" tta={args.tta}"
        else:
            preds = np.asarray(client.orient_batch(cloud, instructions),
                               np.float32)
        if R_bc is not None:
            preds = preds @ R_bc          # (R_bc.T @ d.T).T back to body

        print(f"\n=== {obj}  [{mode}, {len(cloud)} pts] ===")
        print(f"{'instruction':<16} {'prediction (body frame)':<28} "
              f"{'gt':<12} {'ang err':>8}"
              + ("  {:>7}".format("spread") if spread is not None else ""))
        for k, ((ins, gt_spec), pred) in enumerate(
                zip(INSTRUCTIONS[obj], preds)):
            gt = resolve_gt(gt_spec, frames, mesh, body_xyz)
            pstr = "[" + " ".join(f"{x:+.3f}" for x in pred) + "]"
            sp = (f"  {spread[k]:5.1f}°"
                  if spread is not None else "")
            if gt is None:
                print(f"{ins:<16} {pstr:<28} {'—':<12} {'—':>8}{sp}")
            else:
                # Signed angle is the metric: semantic orientation is a
                # directed vector, so the antipode is WRONG. The flip flag
                # is diagnostic only — it tells you the model found the
                # right axis but the wrong sign (e.g. "upright" pointing
                # down), which is a different failure from a random miss.
                err = angular_error_deg(pred, gt)
                flip = min(err, 180.0 - err)
                if err < 30:
                    flag = ""
                elif flip < 30:
                    flag = "  <-- sign-flip?"
                else:
                    flag = "  <-- check"
                gt_name = gt_spec[1]
                print(f"{ins:<16} {pstr:<28} {gt_name:<12} "
                      f"{err:7.1f}°{sp}{flag}")
                overall.append((obj, ins, err))
            if args.refine and ins in REFINE:
                res = refine_axis(body_xyz, pred, REFINE[ins])
                rstr = ("[" + " ".join(f"{x:+.3f}" for x in res.direction)
                        + "]")
                if res.accepted:
                    detail = (f"snap {res.snap_deg:4.1f}°  "
                              f"rms {res.residual_rms * 1000:.1f} mm  "
                              f"σ {res.sigma_deg:.2f}°")
                else:
                    detail = f"REJECTED: {res.note}"
                if gt is not None:
                    rerr = angular_error_deg(res.direction, gt)
                    print(f"{'  └ refined':<16} {rstr:<28} {'':<12} "
                          f"{rerr:7.1f}°  {detail}")
                    overall_refined.append((obj, ins, rerr))
                else:
                    print(f"{'  └ refined':<16} {rstr:<28} {'—':<12} "
                          f"{'—':>8}  {detail}")

        if args.save_npz:
            out = Path(args.save_npz)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"pointso_{obj}.npz",
                     cloud=cloud, instructions=np.array(instructions),
                     predictions=preds)
            print(f"[pointso-test] wrote {out / f'pointso_{obj}.npz'}")

    if overall:
        errs = np.array([e for _, _, e in overall])
        n_flip = int(np.sum((errs >= 150.0)))
        print(f"\n[pointso-test] scored {len(errs)} instruction(s): "
              f"mean {errs.mean():.1f}°, median {np.median(errs):.1f}°, "
              f"max {errs.max():.1f}°, <45°: {np.mean(errs < 45):.0%}"
              + (f", near-antipodes: {n_flip}" if n_flip else ""))
    if overall_refined:
        raw = {(o, i): e for o, i, e in overall}
        pairs = [(raw[(o, i)], e) for o, i, e in overall_refined
                 if (o, i) in raw]
        r = np.array([a for a, _ in pairs])
        f = np.array([b for _, b in pairs])
        print(f"[pointso-test] refined {len(f)} of them: raw mean "
              f"{r.mean():.1f}° -> refined mean {f.mean():.1f}°, "
              f"max {r.max():.1f}° -> {f.max():.1f}° (sign errors, if "
              "any, pass through refinement by design)")


if __name__ == "__main__":
    main()