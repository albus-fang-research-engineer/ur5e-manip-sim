"""Render-test refine_axis with the TRUE revolution axis horizontal,
pointing from handle to spout (the frames.json pour_axis direction) —
i.e. the mid-pour configuration where the teapot's up axis has tipped
to horizontal — and coarse seeds that are nearby and also horizontal.

Run from repo root:

    PYTHONPATH=. python scripts/test_horizontal_axis_render.py
    PYTHONPATH=. python scripts/test_horizontal_axis_render.py --mesh

Synthetic mode (default) uses the teapot-like cloud from
tests/test_refine.py (belly profile + handle blob) and expects clean
sub-degree acceptances — the well-conditioned tier.

--mesh mode rotates the REAL teapot_visual.obj so its calibrated
up_axis lands on the horizontal pour direction, samples the surface,
and runs the full documented route: whole-cloud revolution refine
first; on a typed rejection (the expected outcome for the real teapot
— near-degenerate body, attachments not band-separable) fall through
to extract_ring_from_mesh + feature="rim". The two-tier contract from
test_refine_on_converted_meshes applies per stage: an ACCEPT must land
within 3 deg of truth; otherwise the rejection must be typed with the
coarse passed back — never a confident wrong axis.

Renders one 3D panel per coarse seed to outputs/axis_fit/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_refine import make_revolution_cloud, _tilt  # noqa: E402

from manip_sim.refine import extract_ring_from_mesh, refine_axis  # noqa: E402


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def horiz_tilt(axis, deg):
    """A coarse seed `deg` away from `axis` but still HORIZONTAL
    (rotated about world z), matching 'nearby horizontal'."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ axis


def rot_between(u, v):
    """Rotation matrix taking unit u to unit v (Rodrigues)."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    s = np.linalg.norm(w)
    c = float(u @ v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any perpendicular
        p = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            p = np.array([0.0, 1.0, 0.0])
        p = np.cross(u, p)
        p /= np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def load_pour_axis():
    frames = json.loads(
        (ROOT / "assets/objects/teapot/frames.json").read_text())
    pour = np.asarray(frames["axes"]["pour_axis"]["xyz"], float)
    pour /= np.linalg.norm(pour)
    assert abs(pour[2]) < 1e-6, "pour_axis should be horizontal"
    up = np.asarray(frames["axes"]["up_axis"]["xyz"], float)
    return pour, up / np.linalg.norm(up)


def refine_with_fallback(P, coarse, mesh_rot=None):
    """Whole-cloud revolution refine; on rejection, the documented
    terminal route: ring extraction + rim fit (mesh mode only)."""
    res = refine_axis(P, coarse, "revolution")
    route = "revolution"
    if not res.accepted and mesh_rot is not None:
        ring = extract_ring_from_mesh(mesh_rot, coarse, side="top")
        if len(ring) >= 50:
            rim = refine_axis(np.asarray(ring), coarse, "rim")
            if rim.accepted:
                return rim, "rim(top)", np.asarray(ring)
            res, route = rim, "rim(top) rejected"
        else:
            route = "revolution rejected; ring < 50 pts"
    return res, route, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="store_true",
                    help="use the real teapot_visual.obj instead of the "
                         "synthetic cloud")
    ap.add_argument("--axis", choices=["pour", "up-rotated"],
                    default="pour",
                    help="pour: teapot UPRIGHT, refine the horizontal "
                         "handle->spout pour_axis itself (whole-cloud "
                         "revolution must reject typed; spout-tip rim "
                         "circle is the recovering route). up-rotated: "
                         "rotate the mesh so up_axis lies along the "
                         "pour direction and refine that (rotation-"
                         "equivariance check only - same problem as "
                         "the vertical fit in a rotated frame)")
    ap.add_argument("--samples", type=int, default=6000)
    args = ap.parse_args()

    true_axis, up = load_pour_axis()
    mesh_rot = None
    if args.mesh:
        import trimesh
        mesh_path = ROOT / "assets/objects/teapot/meshes/teapot_visual.obj"
        if not mesh_path.exists():
            print(f"[skip] {mesh_path} not converted — run "
                  "scripts/convert_asset.py or drop --mesh")
            return 0
        mesh_rot = trimesh.load(mesh_path, force="mesh", process=False)
        if args.axis == "up-rotated":
            R = rot_between(up, true_axis)  # tip up_axis onto pour dir
            mesh_rot.apply_transform(
                np.block([[R, np.zeros((3, 1))],
                          [np.zeros((1, 3)), 1.0]]))
        # pour mode: mesh stays in its calibrated upright pose; the
        # target IS the horizontal pour_axis from frames.json
        P, _ = trimesh.sample.sample_surface(mesh_rot, args.samples,
                                             seed=0)
        P = np.asarray(P)
        tag = f"mesh_{args.axis}"
    else:
        if args.axis == "pour":
            print("[note] synthetic cloud is a revolution body; there "
                  "is no synthetic pour_axis analog. Falling back to "
                  "the up-rotated equivariance check. Use --mesh for "
                  "the real pour-axis test.")
        P = make_revolution_cloud(true_axis, n=4000, noise=0.001,
                                  handle=True, seed=3)
        tag = "synthetic"

    cases = [
        ("in-plane +15 deg (horizontal)", horiz_tilt(true_axis, 15.0)),
        ("in-plane -25 deg (horizontal)", horiz_tilt(true_axis, -25.0)),
        ("in-plane +40 deg (horizontal)", horiz_tilt(true_axis, 40.0)),
        ("out-of-plane 20 deg", _tilt(true_axis, 20.0, seed=7)),
    ]

    fig = plt.figure(figsize=(20, 5.5))
    if args.mesh and args.axis == "pour":
        print("[expect] upright teapot, target = pour_axis: whole-cloud "
              "revolution should REJECT typed (the body is not a "
              "revolution surface about pour); the spout-tip rim circle "
              "is the recovering route - or a typed rejection if its "
              "edge candidates come in under MIN_POINTS, which routes "
              "to segmented-spout pca instead.")
    print(f"[{tag}] true axis (pour_axis, handle->spout): "
          f"[{true_axis[0]:+.4f} {true_axis[1]:+.4f} {true_axis[2]:+.4f}]\n")
    hdr = (f"{'case':<32} {'coarse->true':>12} {'refined->true':>13} "
           f"{'snap':>7} {'sigma':>7} {'rms mm':>7} {'accept':>6}  route")
    print(hdr)
    print("-" * len(hdr))

    all_ok = True
    for i, (name, coarse) in enumerate(cases):
        res, route, ring = refine_with_fallback(P, coarse, mesh_rot)
        err_c = angle_deg(coarse, true_axis)
        err_r = angle_deg(res.direction, true_axis)
        # two-tier contract: accepted -> close to truth; rejected ->
        # coarse passed back unchanged (typed, not confidently wrong)
        ok = (err_r < 3.0) if res.accepted else (
            angle_deg(res.direction, coarse) < 1e-6)
        all_ok &= ok
        print(f"{name:<32} {err_c:11.1f}d {err_r:12.2f}d "
              f"{res.snap_deg:6.1f}d {res.sigma_deg:6.2f}d "
              f"{res.residual_rms * 1e3:7.2f} {str(res.accepted):>6}"
              f"  {route}" + ("" if ok else "   <-- FAIL"))
        if res.note:
            print(f"    note: {res.note}")

        ax = fig.add_subplot(1, len(cases), i + 1, projection="3d")
        if mesh_rot is not None:
            # shaded mesh surface: far clearer structure than a sparse
            # scatter. Subsample faces if the visual mesh is heavy.
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            tri = np.asarray(mesh_rot.triangles)
            if len(tri) > 40000:
                tri = tri[::len(tri) // 40000 + 1]
            # simple lambertian shading from face normals
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            light = np.array([0.4, -0.5, 0.77])
            shade = 0.55 + 0.45 * np.abs(n @ light)
            cols = np.column_stack([0.62 * shade, 0.66 * shade,
                                    0.72 * shade,
                                    np.full(len(tri), 0.28)])
            pc = Poly3DCollection(tri, facecolors=cols,
                                  edgecolors="none")
            ax.add_collection3d(pc)
        else:
            step = max(len(P) // 4000, 1)
            Q = P[::step]
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=1.5, c="#9aa6b2",
                       alpha=0.45, linewidths=0)
        if ring is not None:
            ax.scatter(ring[:, 0], ring[:, 1], ring[:, 2], s=5.0,
                       c="#f28e2b", alpha=1.0, linewidths=0,
                       label="extracted rim (fed to rim fit)",
                       depthshade=False)
        c0 = P.mean(axis=0)
        # true axis drawn LONGER than refined: when the fit lands, blue
        # covers green except at the tips — green tips = coincidence.
        for v, col, lab, lw, L in [
                (true_axis, "#1a9850", "true", 3.5, 0.115),
                (coarse / np.linalg.norm(coarse), "#d73027", "coarse",
                 2.0, 0.10),
                (res.direction, "#4575b4", "refined", 2.5, 0.10)]:
            seg = np.array([c0 - L * v, c0 + L * v])
            ax.plot(*seg.T, c=col, lw=lw, label=lab,
                    ls="--" if lab == "coarse" else "-")
        ax.set_title(f"{name}\n{route}  err {err_r:.2f}d  "
                     f"{'ACCEPT' if res.accepted else 'REJECT'}",
                     fontsize=9,
                     color="#1a9850" if ok else "#d73027")
        ax.set_box_aspect((1, 1, 1))
        # limits from the cloud extent, 35% margin so the axis lines
        # (which extend past the body) stay inside the frame
        lim = 1.35 * float(np.abs(P - c0).max())
        ax.set_xlim(c0[0] - lim, c0[0] + lim)
        ax.set_ylim(c0[1] - lim, c0[1] + lim)
        ax.set_zlim(c0[2] - lim, c0[2] + lim)
        # camera perpendicular to the true axis: the horizontal
        # handle->spout direction reads across the frame
        az_axis = np.degrees(np.arctan2(true_axis[1], true_axis[0]))
        ax.view_init(elev=18, azim=az_axis - 90.0)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    fig.suptitle(f"refine_axis [{tag}]: horizontal true axis "
                 "(handle->spout pour direction), nearby horizontal "
                 "coarse seeds", fontsize=12)
    out = ROOT / "outputs" / "axis_fit" / f"horizontal_axis_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nrender -> {out}")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())