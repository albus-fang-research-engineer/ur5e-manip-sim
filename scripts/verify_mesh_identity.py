#!/usr/bin/env python3
"""Decide whether a candidate raw mesh IS the mesh the tracked grounding
artifacts were authored against — and, if it is up to scale/yaw, emit the
corrected raw mesh to feed scripts/convert_asset.py.

Why this exists: meshes/ and *.obj are gitignored, so a `git clean -fdx`
destroys the geometry while leaving every artifact DERIVED from it in the
index — the MJCF (whose three sites record the post-scale bounding half
extents), frames.json (calibrated symbols), candidates.json (the pool the
VLM mark IDs index into), and outputs/selections/*.json. Those derived
files are only meaningful if the restored mesh reproduces the exact same
body frame. This script checks that, using the tracked files as the
reference measurement rather than trusting a filename.

Three fingerprints, in increasing strictness:

  1. SHAPE   bounding-box aspect ratio (scale-invariant, yaw-sensitive
             only through the box). Rules out "a different teapot".
  2. SCALE   the single factor s taking the raw extents onto the extents
             recorded by the MJCF sites. Solved, not assumed.
  3. FRAME   re-derive the frames.json symbols from the scaled, recentred
             mesh using the SAME code calibrate_frames_from_mesh.py uses,
             and diff against the committed values. This is the one that
             actually certifies candidates.json IDs and the committed VLM
             selections are still valid.

Usage:

    # is this the teapot, and at what scale?
    PYTHONPATH=. python scripts/verify_mesh_identity.py \
        ~/Downloads/teapot.glb --name teapot

    # same, but write a scale/yaw-corrected raw OBJ ready for conversion
    PYTHONPATH=. python scripts/verify_mesh_identity.py \
        ~/Downloads/teapot.glb --name teapot --fix-yaw \
        --export assets/raw/teapot.obj

Exit status is 0 only when SHAPE and SCALE pass; FRAME residuals are
always printed and never auto-corrected — a frame mismatch is a decision
(re-source the mesh, or accept and regenerate the whole grounding chain),
not something a script should make for you.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_frames_from_mesh as cal  # noqa: E402  (same symbol code)

ASSETS = Path("assets/objects")

# per-axis aspect agreement; loosened only by mesh decimation, not by a
# genuinely different model
ASPECT_TOL = 2e-3
# frames.json stores 4 decimals, so anything under ~0.2 mm / 0.2 deg is
# quantisation, not disagreement
POINT_TOL_M = 2e-4
AXIS_TOL_DEG = 0.5


def target_extents(name: str) -> np.ndarray:
    """Post-scale bounding extents, read back out of the committed MJCF.

    convert_asset.py writes bottom_site at (0,0,-hz), top_site at (0,0,+hz)
    and horizontal_radius_site at (hx,hy,0), so the sites ARE the half
    extents of the mesh it converted. Nothing else in the repo records
    absolute size.
    """
    xml = ASSETS / name / f"{name}.xml"
    sites = {s.get("name"): np.fromstring(s.get("pos"), sep=" ")
             for s in ET.parse(xml).getroot().iter("site")}
    hx, hy, _ = sites["horizontal_radius_site"]
    hz = sites["top_site"][2]
    return 2.0 * np.array([hx, hy, hz])


def load_raw(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"[verify] {path} did not load as a single mesh")
    return m


def yaw_of(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def measured_yaw(name: str, V: np.ndarray, spec: dict) -> tuple[float, str]:
    """A yaw observable that does not depend on the bounding box.

    teapot: the spout protrusion direction (same iteration the calibrator
    uses, seeded off the committed axis so a wrong seed still converges).
    mug:    the handle bearing, i.e. the vertex centroid's offset from the
            rim-circle centre — the mug's only azimuthal feature.
    """
    if name == "teapot":
        seed = np.asarray(spec["axes"]["pour_axis"]["xyz"], float)
        return yaw_of(cal.mesh_pour_axis(V, seed)), "spout direction"
    rim = cal.calibrate_mug(V)
    d = V.mean(axis=0)[:2] - rim["opening_center"][:2]
    return yaw_of(np.r_[d, 0.0]), "handle bearing"


def frame_residuals(name: str, V: np.ndarray, spec: dict) -> list[tuple]:
    """(label, committed, recomputed, error, unit, tol) rows."""
    rows = []
    if name == "teapot":
        pour = np.asarray(spec["axes"]["pour_axis"]["xyz"], float)
        got = cal.calibrate_teapot(V, pour / np.linalg.norm(pour))
        for key, sec in [("spout_tip", "points"), ("handle_center", "points"),
                         ("handle_axis", "axes"), ("tilt_axis", "axes")]:
            ref = np.asarray(spec[sec][key]["xyz"], float)
            cur = np.asarray(got[key], float)
            if sec == "points":
                rows.append((key, ref, cur, float(np.linalg.norm(cur - ref)),
                             "m", POINT_TOL_M))
            else:
                cos = np.clip(cur @ ref / (np.linalg.norm(cur)
                                           * np.linalg.norm(ref)), -1, 1)
                rows.append((key, ref, cur,
                             float(np.degrees(np.arccos(cos))),
                             "deg", AXIS_TOL_DEG))
    else:
        got = cal.calibrate_mug(V)
        ref = np.asarray(spec["points"]["opening_center"]["xyz"], float)
        cur = np.asarray(got["opening_center"], float)
        rows.append(("opening_center", ref, cur,
                     float(np.linalg.norm(cur - ref)), "m", POINT_TOL_M))
        r_ref = float(spec["quantities"]["rim_radius"]["value"])
        rows.append(("rim_radius", np.array([r_ref]),
                     np.array([got["rim_radius"]]),
                     abs(got["rim_radius"] - r_ref), "m", POINT_TOL_M))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--name", required=True, choices=["teapot", "mug"])
    ap.add_argument("--scale", type=float, default=None,
                    help="force a scale instead of solving for it")
    ap.add_argument("--fix-yaw", action="store_true",
                    help="rotate about +z to null the yaw error before "
                         "the frame check and export")
    ap.add_argument("--export", type=Path, default=None,
                    help="write the scaled/recentred(/rotated) mesh here; "
                         "feed it to convert_asset.py with --scale 1.0")
    args = ap.parse_args()

    import json
    spec = json.loads((ASSETS / args.name / "frames.json").read_text())
    want = target_extents(args.name)

    mesh = load_raw(args.mesh)
    raw = mesh.bounds[1] - mesh.bounds[0]

    # --- 1. SHAPE ---------------------------------------------------------
    per_axis = want / raw
    spread = float(per_axis.max() / per_axis.min() - 1.0)
    shape_ok = spread <= ASPECT_TOL
    print(f"\n[shape] raw extents      {np.round(raw, 6).tolist()}")
    print(f"[shape] committed extents {np.round(want, 6).tolist()}")
    print(f"[shape] aspect (x-norm)   raw {np.round(raw / raw[0], 5).tolist()}"
          f"  committed {np.round(want / want[0], 5).tolist()}")
    print(f"[shape] per-axis scale spread {spread * 100:.3f}%  "
          f"(tol {ASPECT_TOL * 100:.1f}%)  -> {'PASS' if shape_ok else 'FAIL'}")
    if not shape_ok:
        print("[shape] the axis ratios disagree: different model, a "
              "non-uniform export scale, or a permuted axis convention. "
              "Try the permutations below before giving up:")
        import itertools
        for p in itertools.permutations(range(3)):
            s = want / raw[list(p)]
            if s.max() / s.min() - 1.0 <= ASPECT_TOL:
                print(f"        axis order {p} would match at "
                      f"scale {s.mean():.6f}")

    # --- 2. SCALE ---------------------------------------------------------
    s = args.scale if args.scale is not None else float(per_axis.mean())
    print(f"\n[scale] using s = {s:.6f}"
          + ("  (forced)" if args.scale is not None else "  (solved)"))
    mesh.apply_scale(s)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))  # convert_asset does this
    got = mesh.bounds[1] - mesh.bounds[0]
    err_mm = float(np.abs(got - want).max() * 1e3)
    print(f"[scale] residual extent error {err_mm:.4f} mm")

    # --- 3. FRAME ---------------------------------------------------------
    # The symbol residuals are the authority; yaw is only a REPAIR HINT for
    # when they fail. The mug's yaw observable in particular is weak: it
    # reads the heading off the vertex-centroid-to-rim-centre offset, which
    # assumes roughly uniform vertex density. On a google_16k mesh the
    # cylinder wall's tessellation outnumbers the handle and drags the
    # centroid, giving a ~12deg phantom offset on a mesh whose symbols
    # reproduce exactly. So: residuals first, and if they pass, the heading
    # is correct by definition and the yaw complaint is suppressed.
    V = np.asarray(mesh.vertices, float)
    rows = frame_residuals(args.name, V, spec)
    worst = max(err / tol for _, _, _, err, _, tol in rows)

    yaw_now, how = measured_yaw(args.name, V, spec)
    if args.name == "teapot":
        yaw_ref = yaw_of(np.asarray(spec["axes"]["pour_axis"]["xyz"], float))
    else:
        ref_c = np.asarray(spec["points"]["opening_center"]["xyz"], float)
        yaw_ref = yaw_of(np.r_[-ref_c[:2], 0.0])
    dyaw = (yaw_ref - yaw_now + 180.0) % 360.0 - 180.0
    print(f"\n[yaw]   {how}: mesh {yaw_now:+.3f}deg, "
          f"committed {yaw_ref:+.3f}deg, delta {dyaw:+.3f}deg")

    if worst <= 1.0:
        if abs(dyaw) > 0.5:
            print("[yaw]   ignored — the frame residuals below already pass, "
                  "so the heading is correct and this observable is simply "
                  "weak on this mesh. Do NOT rerun with --fix-yaw.")
    elif args.fix_yaw and abs(dyaw) > 1e-6:
        mesh.apply_transform(rot_z(dyaw))
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        V = np.asarray(mesh.vertices, float)
        rows = frame_residuals(args.name, V, spec)
        worst = max(err / tol for _, _, _, err, _, tol in rows)
        print(f"[yaw]   applied Rz({dyaw:+.3f}deg); residuals recomputed")
    elif abs(dyaw) > 0.5:
        print("[yaw]   NOT applied — the residuals below fail AND the "
              "heading disagrees, so this may be the same model re-exported "
              "at a different heading. Rerun with --fix-yaw to test that.")

    print(f"\n[frame] recomputed symbols vs committed frames.json")
    for label, ref, cur, err, unit, tol in rows:
        flag = "ok " if err <= tol else "OFF"
        print(f"        {flag} {label:<15} committed {np.round(ref, 4)}  "
              f"got {np.round(cur, 4)}  err {err:.5f} {unit}")
    verdict = ("IDENTICAL — candidates.json IDs and "
               "outputs/selections/*.json remain valid"
               if worst <= 1.0 else
               "DIVERGENT — the grounding chain must be regenerated "
               "(calibrate --write, propose --write, re-run select_frames.py)")
    print(f"[frame] {verdict}")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(args.export)
        print(f"\n[export] wrote {args.export} — already metric and "
              f"recentred, so convert it with --scale 1.0")

    sys.exit(0 if (shape_ok and err_mm < 1.0) else 1)


if __name__ == "__main__":
    main()