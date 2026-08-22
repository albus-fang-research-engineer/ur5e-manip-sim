"""End-to-end frame refinement demo: coarse semantic directions in, a
full w-frame with per-row-coupled Bw out, on the converted teapot and
mug assets.

Run from repo root:

    PYTHONPATH=. python scripts/demo_refine_frame.py teapot --render out/teapot_frame.png
    PYTHONPATH=. python scripts/demo_refine_frame.py mug --tilt-deg 40 --render out/mug_frame.png

Per object this exercises the whole ladder:

  1. UP: refine_axis(feature="revolution") from a tilted coarse seed;
     on typed rejection, the terminal ring route
     (extract_ring_from_mesh -> feature="rim") — i.e. exactly the
     routing the orchestrator will do with the fitter's RefineResult.
  2. AZIMUTH: the object's candidate list, in its declared fallback
     order (see below), each candidate printed with its accept/reject
     verdict, route, sigma, and conditioning.
  3. ASSEMBLY: assemble_frame -> FrameResult; orthonormality and the
     winning route reported.
  4. COUPLING: a pour-grade authored Bw (roll/pitch +-2 deg, yaw
     +-5 deg) pushed through couple_rot_bounds, printed row by row with
     which estimator's sigma set each floor.
  5. RENDER (--render): three-azimuth cloud view with true/coarse/
     refined up, the refined front, the part sliver and ring points
     that fed the fits, and the assembled triad at the frame origin.

Fallback orders (the stage-code decision this script pins down; when
the VLM orchestrator lands, these lists lift into it verbatim — the
routes are already perception-shaped: constructed points become
grounded-manifest symbols, the sliver becomes the segmented spout
mask, the semantic front becomes the Orient Anything output):

  teapot   constructed(handle_center -> spout_tip)   ~0.8 deg over the
                                                      21 cm lever
           part_pca(spout sliver along the coarse front)
           semantic(coarse front)                     20 deg regime
  mug      semantic only — no metric front feature exists, and the
           honest outcome for a receiving vessel is a loose/FREE yaw.

The coarse inputs are simulated as truth + tilt: up from frames.json
up_axis tilted by --tilt-deg,
front from pour_axis (teapot; ground truth for error reporting, NOT a
route) or an arbitrary horizontal (mug, which has no front truth)
tilted by --front-tilt-deg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from manip_sim.refine import RefineResult, extract_ring_from_mesh, refine_axis
from manip_sim.refine_frame import (AzimuthResult, assemble_frame,
                                    azimuth_from_part_cloud,
                                    azimuth_from_points,
                                    azimuth_from_semantic)
from manip_sim.tsr import bounds

# Calibrated-point trust for the constructed route: the residual regime
# of calibrate_frames_from_mesh's sliver/band centroids on converted
# meshes. Deployment-side this becomes the grounding pipeline's point
# uncertainty (mask lift + primitive fit residual).
POINT_SIGMA_M = 0.002
# Extremal-band fraction along the coarse front used to carve the spout
# sliver for the part_pca route — the calibration script's sliver idea,
# but deliberately selected via the COARSE front (selection is the
# semantic layer's job and must survive its error; the geometry then
# measures). Deployment-side this is the segmented spout mask.
SLIVER_FRAC = 0.22


def _unit(v):
    v = np.asarray(v, float).reshape(3)
    return v / np.linalg.norm(v)


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(
        np.dot(_unit(u), _unit(v)), -1, 1))))


def tilt(axis, deg, seed=0):
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(seed)
    perp = np.cross(axis, rng.normal(size=3))
    perp /= np.linalg.norm(perp)
    return R.from_rotvec(perp * np.deg2rad(deg)).as_matrix() @ _unit(axis)


# ------------------------------------------------------------- up pipeline

def refine_up(P: np.ndarray, mesh, coarse_up: np.ndarray
              ) -> tuple[RefineResult, np.ndarray]:
    """Revolution fit, then the ring terminal route on typed rejection.
    Returns (result, ring_points_used) — ring_points empty when the
    whole-cloud fit accepted."""
    res = refine_axis(P, coarse_up, "revolution")
    print(f"[up] revolution: accepted={res.accepted} "
          f"snap {res.snap_deg:.1f} deg  rms {res.residual_rms * 1e3:.1f} mm"
          f"  sigma {res.sigma_deg:.2f} deg")
    if res.accepted:
        return res, np.empty((0, 3))
    print(f"[up]   note: {res.note}")
    for side in ("top", "bottom"):
        rpts = extract_ring_from_mesh(mesh, coarse_up, side)
        if len(rpts) < 50:
            print(f"[up] ring/{side}: only {len(rpts)} edge points")
            continue
        rres = refine_axis(rpts, coarse_up, "rim")
        print(f"[up] ring/{side}: accepted={rres.accepted} n={len(rpts)} "
              f"rms {rres.residual_rms * 1e3:.2f} mm  "
              f"sigma {rres.sigma_deg:.3f} deg")
        if rres.accepted:
            return rres, rpts
        print(f"[up]   note: {rres.note}")
    print("[up] all routes rejected — coarse kept, roll/pitch stay "
          "authored (no trusted sigma)")
    return res, np.empty((0, 3))


# ------------------------------------------- per-object azimuth candidates

def spout_sliver(P: np.ndarray, up_dir: np.ndarray,
                 coarse_front: np.ndarray) -> np.ndarray:
    """Extremal band of the cloud along the in-plane coarse front — the
    sim/calibration stand-in for the segmented spout mask."""
    f = coarse_front - (coarse_front @ up_dir) * up_dir
    f = f / max(np.linalg.norm(f), 1e-12)
    h = (P - P.mean(axis=0)) @ f
    return P[h >= h.max() - SLIVER_FRAC * (h.max() - h.min())]


def azimuth_candidates_teapot(sym_points: dict, up_dir: np.ndarray,
                              P: np.ndarray, coarse_front: np.ndarray
                              ) -> list[AzimuthResult]:
    """The declared fallback order for the teapot's pouring frame."""
    cands = [
        azimuth_from_points(sym_points["handle_center"],
                            sym_points["spout_tip"], up_dir,
                            point_sigma_m=POINT_SIGMA_M),
        azimuth_from_part_cloud(spout_sliver(P, up_dir, coarse_front),
                                up_dir, coarse_front),
        azimuth_from_semantic(coarse_front, up_dir),
    ]
    return cands


def azimuth_candidates_mug(sym_points: dict, up_dir: np.ndarray,
                           P: np.ndarray, coarse_front: np.ndarray
                           ) -> list[AzimuthResult]:
    """No metric front feature exists on the mug; semantic-only, and the
    coupling's job is to make the resulting looseness explicit."""
    return [azimuth_from_semantic(coarse_front, up_dir)]


CANDIDATE_BUILDERS = {
    "teapot": azimuth_candidates_teapot,
    "mug": azimuth_candidates_mug,
}


# ----------------------------------------------------------------- render

def render_frame(P, sliver, ring_pts, true_up, coarse_up, fr, origin,
                 out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = P.mean(axis=0)
    L = 0.7 * float(np.linalg.norm(P.max(0) - P.min(0)))
    fig = plt.figure(figsize=(15, 5.4))
    for k, azim in enumerate((30, 120, 210)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(*P.T, s=1.5, c="0.6", alpha=0.3, linewidths=0)
        if len(sliver):
            ax.scatter(*sliver.T, s=3.0, c="purple", alpha=0.7,
                       linewidths=0, label="spout sliver")
        if len(ring_pts):
            ax.scatter(*ring_pts.T, s=5.0, c="magenta", alpha=0.9,
                       linewidths=0, label="ring points")
        for vec, color, style, lw, lbl in (
                (true_up, "green", "-", 4.5, "true up"),
                (coarse_up, "darkorange", "--", 1.6, "coarse up"),
                (fr.R[:, 2], "blue", "-", 2.2, "refined up (z)")):
            seg = np.array([c - 0.45 * L * vec, c + 0.65 * L * vec])
            ax.plot(*seg.T, color=color, ls=style, lw=lw, label=lbl)
        # assembled triad at the frame origin
        for col, color, lbl in ((0, "red", "front (x)"),
                                (1, "goldenrod", "left (y)")):
            seg = np.array([origin, origin + 0.5 * L * fr.R[:, col]])
            ax.plot(*seg.T, color=color, lw=2.4, label=lbl)
        ax.scatter(*np.atleast_2d(origin).T, s=40, c="black", marker="o",
                   label="frame origin")
        ax.view_init(elev=18, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        lim = np.array([c - 0.6 * L, c + 0.6 * L])
        ax.set_xlim(lim[:, 0]); ax.set_ylim(lim[:, 1])
        ax.set_zlim(lim[:, 2])
        ax.set_axis_off()
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[demo] wrote {out_path}")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object", choices=sorted(CANDIDATE_BUILDERS),
                    help="object name under assets/objects/")
    ap.add_argument("--tilt-deg", type=float, default=25.0,
                    help="coarse UP error to simulate (default 25)")
    ap.add_argument("--front-tilt-deg", type=float, default=20.0,
                    help="coarse FRONT error to simulate (default 20)")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--render", default=None, metavar="PNG")
    args = ap.parse_args()

    obj_dir = Path("assets/objects") / args.object
    spec = json.loads((obj_dir / "frames.json").read_text())
    points = {k: np.asarray(v["xyz"], float)
              for k, v in spec.get("points", {}).items()}
    true_up = _unit(spec["axes"]["up_axis"]["xyz"])
    mesh = trimesh.load(obj_dir / "meshes" / f"{args.object}_visual.obj",
                        force="mesh", process=False)
    P, _ = trimesh.sample.sample_surface(mesh, args.samples, seed=0)
    P = np.asarray(P)

    # simulated coarse inputs (PointSO / Orient Anything stand-ins)
    coarse_up = tilt(true_up, args.tilt_deg)
    if "pour_axis" in spec["axes"]:            # front truth exists (teapot)
        true_front = _unit(spec["axes"]["pour_axis"]["xyz"])
    else:                                      # mug: no front truth; pick
        true_front = _unit(np.cross(true_up, [0.0, 1.0, 0.0]))
    coarse_front = tilt(true_front, args.front_tilt_deg, seed=1)
    print(f"[demo] {args.object}: {len(P)} samples, coarse up "
          f"{args.tilt_deg:.0f} deg off, coarse front "
          f"{args.front_tilt_deg:.0f} deg off")

    up, ring_pts = refine_up(P, mesh, coarse_up)
    print(f"[up] error vs true: {angle_deg(up.direction, true_up):.2f} deg"
          f"  (route sigma {up.sigma_deg:.3f} deg)")

    builder = CANDIDATE_BUILDERS[args.object]
    cands = builder(points, up.direction, P, coarse_front)
    print("\n[azimuth] candidates in fallback order:")
    for cd in cands:
        verdict = ("sigma {:.2f} deg, conditioning {:.2f}".format(
            cd.sigma_deg, cd.conditioning) if cd.accepted
            else f"REJECTED — {cd.note}")
        print(f"  {cd.route:<12} {verdict}")

    fr = assemble_frame(up, cands)
    print(f"\n[frame] accepted={fr.accepted} via {fr.azimuth.route}; "
          f"{fr.note}")
    yaw_err = angle_deg(fr.R[:, 0],
                        true_front - (true_front @ fr.R[:, 2]) * fr.R[:, 2])
    print(f"[frame] front error vs true (in-plane): {yaw_err:.2f} deg"
          + ("" if "pour_axis" in spec["axes"] else
             "  (mug: vs the arbitrary simulated truth)"))

    authored = bounds(roll=(-np.deg2rad(2), np.deg2rad(2)),
                      pitch=(-np.deg2rad(2), np.deg2rad(2)),
                      yaw=(-np.deg2rad(5), np.deg2rad(5)))
    Bw = fr.couple_rot_bounds(authored, k=3.0)
    print("\n[coupling] authored -> coupled (k=3), half-widths in deg:")
    src = {3: f"up fit sigma {up.sigma_deg:.3f}",
           4: f"up fit sigma {up.sigma_deg:.3f}",
           5: (f"azimuth ({fr.azimuth.route}) sigma "
               f"{fr.azimuth.sigma_deg:.2f}" if fr.azimuth.accepted
               else "azimuth rejected -> FREE")}
    for row, name in ((3, "roll"), (4, "pitch"), (5, "yaw")):
        h0 = np.degrees(0.5 * (authored[row, 1] - authored[row, 0]))
        h1 = np.degrees(0.5 * (Bw[row, 1] - Bw[row, 0]))
        floor = "FREE" if h1 >= 179.9 else f"{h1:7.2f}"
        print(f"  {name:<5} {h0:6.2f} -> {floor:<8} [{src[row]}]")

    origin = points.get("spout_tip", points.get("opening_center",
                                                P.mean(axis=0)))
    f = fr.to_frame(f"{args.object}.demo_frame", origin)
    print(f"\n[frame] packaged: {f.name} status={f.status} — {f.comment}")

    if args.render:
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        sliver = (spout_sliver(P, up.direction, coarse_front)
                  if args.object == "teapot" else np.empty((0, 3)))
        render_frame(P, sliver, ring_pts, true_up, coarse_up, fr, origin,
                     args.render,
                     f"{args.object}: up via {up.method} "
                     f"({'ok' if up.accepted else 'rejected'}), azimuth "
                     f"via {fr.azimuth.route} "
                     f"({'ok' if fr.accepted else 'rejected -> yaw FREE'})")


if __name__ == "__main__":
    main()