"""Diagnose why the revolution-axis refinement accepts, rejects, or
mislands on a given object.

Run from repo root:

    PYTHONPATH=. python scripts/diagnose_axis_fit.py teapot
    PYTHONPATH=. python scripts/diagnose_axis_fit.py mug --tilt-deg 40

Prints, in order:

  1. cloud stats about the CALIBRATED axis (frames.json), including a
     per-height-band radius table with the fraction each band's gate
     would keep — this shows directly whether attachments (spout,
     handle) are band-separable radial outliers, which is the
     assumption the gate machinery rests on;
  2. the same band table about the tilted coarse seed the fit will
     actually start from;
  3. a multistart dump: every start's converged axis (angle to truth
     and to coarse), data rms, retained count — the mode structure the
     cluster arbitration sees;
  4. the final refine_axis verdict with the quality numbers
     (rms / median radius) next to the gates that judge them.

Interpreting the outcome: if even the true-axis band table shows the
attachments blending into the body bands (keep fraction ~1.0 in bands
that contain spout/handle), the whole-cloud fit cannot be rescued by
tuning — the object needs a segmented feature fit (opening rim circle
via Kasa, as calibrate_frames_from_mesh does for the mug) and the typed
rejection is the correct terminal behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from manip_sim.refine import (MAX_REL_RMS, MAX_SIGMA_DEG, _band_gate,
                              _basis_perp, _fit_revolution_once,
                              refine_axis)


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def tilt(axis, deg, seed=0):
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(seed)
    perp = np.cross(axis, rng.normal(size=3))
    perp /= np.linalg.norm(perp)
    return R.from_rotvec(perp * np.deg2rad(deg)).as_matrix() @ axis


def band_table(P, axis, label, nbins=12):
    c = P.mean(axis=0)
    d = P - c
    h = d @ axis
    radial = np.linalg.norm(d - np.outer(h, axis), axis=1)
    keep = _band_gate(P, axis, c, nbins=nbins)
    span = float(h.max() - h.min()) + 1e-9
    bins = np.clip(((h - h.min()) / span * nbins).astype(int), 0, nbins - 1)
    print(f"\n-- band table about {label} "
          f"(gate keeps {keep.mean():.0%} overall) --")
    print(f"{'band':>4} {'n':>5} {'h range (mm)':>16} {'med r':>7} "
          f"{'MAD r':>7} {'max r':>7} {'keep':>5}")
    for b in range(nbins):
        m = bins == b
        if not m.any():
            continue
        print(f"{b:>4} {m.sum():>5} "
              f"[{h[m].min() * 1e3:6.1f},{h[m].max() * 1e3:6.1f}] "
              f"{np.median(radial[m]) * 1e3:7.1f} "
              f"{np.median(np.abs(radial[m] - np.median(radial[m]))) * 1e3:7.1f} "
              f"{radial[m].max() * 1e3:7.1f} {keep[m].mean():5.0%}")
    return keep


def render_fit(P, true_axis, coarse, res, raw_axis, kept_idx, out_path,
               title=""):
    """Three-azimuth 3D render of the cloud and the axis story:
    retained points gray, gated/trimmed points red, TRUE axis green,
    coarse dashed orange, the raw coarse-seeded fit light blue, and
    the final verdict axis solid blue (accepted) or dashed red (typed
    rejection, i.e. the coarse handed back)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = np.zeros(len(P), dtype=bool)
    kept[kept_idx] = True
    c = P.mean(axis=0)
    L = 0.75 * float(np.linalg.norm(P.max(0) - P.min(0)))

    fig = plt.figure(figsize=(15, 5.4))
    for k, azim in enumerate((30, 120, 210)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(*P[kept].T, s=1.5, c="0.55", alpha=0.35, linewidths=0)
        if (~kept).any():
            ax.scatter(*P[~kept].T, s=2.5, c="crimson", alpha=0.6,
                       linewidths=0, label="gated/trimmed")
        for vec, color, style, lw, alpha, lbl in (
                (true_axis, "green", "-", 5.5, 0.45, "true"),
                (coarse, "darkorange", "--", 2.0, 1.0, "coarse"),
                (raw_axis, "deepskyblue", "-", 1.4, 1.0, "raw fit"),
                (res.direction, "blue" if res.accepted else "red",
                 "-" if res.accepted else "--", 2.2, 1.0,
                 "refined" if res.accepted else "rejected->coarse")):
            seg = np.array([c - 0.5 * L * vec, c + 0.7 * L * vec])
            ax.plot(*seg.T, color=color, ls=style, lw=lw, alpha=alpha,
                    label=lbl)
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
    print(f"[diagnose] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object", help="object name under assets/objects/")
    ap.add_argument("--axis", default="up_axis",
                    help="axis name in frames.json (default up_axis)")
    ap.add_argument("--tilt-deg", type=float, default=25.0)
    ap.add_argument("--tilt-seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--render", default=None, metavar="PNG",
                    help="also render the cloud + axes (retained vs "
                         "gated points, true / coarse / raw-fit / "
                         "verdict axes) to this path")
    args = ap.parse_args()

    obj_dir = Path("assets/objects") / args.object
    spec = json.loads((obj_dir / "frames.json").read_text())
    true_axis = np.asarray(spec["axes"][args.axis]["xyz"], float)
    true_axis /= np.linalg.norm(true_axis)
    mesh = trimesh.load(obj_dir / "meshes" / f"{args.object}_visual.obj",
                        force="mesh", process=False)
    P, _ = trimesh.sample.sample_surface(mesh, args.samples, seed=0)
    P = np.asarray(P)

    d = P - P.mean(axis=0)
    r_true = np.linalg.norm(d - np.outer(d @ true_axis, true_axis), axis=1)
    print(f"[diagnose] {args.object}: {len(P)} samples, bbox "
          f"{np.round(P.max(0) - P.min(0), 3)}, r about {args.axis} "
          f"[{r_true.min():.3f}, {r_true.max():.3f}] "
          f"median {np.median(r_true):.3f}")

    band_table(P, true_axis, f"TRUE {args.axis}")
    coarse = tilt(true_axis, args.tilt_deg, args.tilt_seed)
    band_table(P, coarse, f"coarse ({args.tilt_deg:.0f} deg tilt)")

    print("\n-- multistart dump --")
    e1, e2 = _basis_perp(coarse)
    starts = [coarse]
    j = np.deg2rad(20.0)
    for az in np.arange(8) * (np.pi / 4):
        dd = np.cos(az) * e1 + np.sin(az) * e2
        s = coarse * np.cos(j) + dd * np.sin(j)
        starts.append(s / np.linalg.norm(s))
    print(f"{'start':>5} {'->true':>7} {'->coarse':>8} {'rms mm':>7} "
          f"{'sigma':>7} {'kept':>5}")
    for i, s in enumerate(starts):
        a, rms, sig, nin = _fit_revolution_once(P, s)
        if a @ coarse < 0:
            a = -a
        print(f"{i:>5} {angle_deg(a, true_axis):7.1f} "
              f"{angle_deg(a, coarse):8.1f} {rms * 1e3:7.1f} "
              f"{sig:7.2f} {nin:>5}")

    res = refine_axis(P, coarse, "revolution")
    err = angle_deg(res.direction, true_axis)
    d = P - P.mean(axis=0)
    r_med = float(np.median(np.linalg.norm(
        d - np.outer(d @ res.direction, res.direction), axis=1)))
    print(f"\n-- verdict --\naccepted={res.accepted} err_vs_true "
          f"{err:.1f} deg  snap {res.snap_deg:.1f} deg\n"
          f"rms {res.residual_rms * 1e3:.1f} mm = "
          f"{res.residual_rms / max(r_med, 1e-9):.0%} of median radius "
          f"(gate {MAX_REL_RMS:.0%})   sigma {res.sigma_deg:.2f} deg "
          f"(gate {MAX_SIGMA_DEG:.0f})\nnote: {res.note or '—'}")

    if args.render:
        raw_axis, raw_rms, _, _, kept_idx = _fit_revolution_once(
            P, coarse, return_kept=True)
        if raw_axis @ coarse < 0:
            raw_axis = -raw_axis
        render_fit(
            P, true_axis, coarse, res, raw_axis, kept_idx, args.render,
            title=(f"{args.object} / {args.axis}: "
                   f"{'ACCEPTED' if res.accepted else 'REJECTED'} — "
                   f"err {err:.1f} deg, raw fit "
                   f"{angle_deg(raw_axis, true_axis):.1f} deg off, "
                   f"rms {res.residual_rms * 1e3:.1f} mm "
                   f"({res.residual_rms / max(r_med, 1e-9):.0%} of "
                   f"radius), sigma {res.sigma_deg:.1f} deg"))


if __name__ == "__main__":
    main()
