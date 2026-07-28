"""Calibrate frames.json interaction symbols FROM THE CONVERTED MESHES —
the mechanized replacement for the double-click-in-viewer loop, and the
first concrete piece of the mesh-native grounding pipeline (interaction
points computed on the mesh, in body coordinates, exactly the artifact the
VLM point-selection stage will later produce).

What it derives, and from what:

  teapot.spout_tip       centroid of the vertices extremal along the
                         CALIBRATED pour_axis (the one symbol that is
                         already trusted anchors the rest)
  teapot.handle_center   the handle is the geometry extremal along
                         -pour_axis (opposite the spout); handle_center is
                         the centroid of the outer bar band, at the
                         region's mid height — the mid-grip point
  teapot.handle_axis     principal component of the handle region
                         (sign-fixed toward +z); warns if it strays far
                         from vertical, since the grasp TSR's slide axis
                         and the vertical-loop assumption ride on it
  teapot.tilt_axis       recomputed as up_axis x pour_axis, signed so
                         positive rotation tips the spout DOWN (keeps the
                         sidecar's auditability convention)
  mug.opening_center     centroid of the top rim band
  mug.rim_radius         mean radial distance of the rim band vertices

Coordinates: the converter writes meshes in the body frame of mjcf body
'object' (geoms carry no pos offset), so mesh vertices ARE body
coordinates — the same frame PoseReader returns and frames.json declares.

Dry run prints current vs computed side by side; --write updates the
sidecars in place, flips statuses to "calibrated", and records the
provenance in each symbol's comment. Re-run scripts/plan_pour_tea.py
afterwards: plans aimed at placeholder symbols are aimed at guesses.

    PYTHONPATH=. python scripts/calibrate_frames_from_mesh.py
    PYTHONPATH=. python scripts/calibrate_frames_from_mesh.py --write
"""

import argparse
import json
from pathlib import Path

import numpy as np

TEAPOT_DIR = Path("assets/objects/teapot")
MUG_DIR = Path("assets/objects/mug")


def load_obj_vertices(path: Path) -> np.ndarray:
    """Vertices of a wavefront OBJ (positions only; faces/normals/uvs
    ignored — interaction symbols are point statistics)."""
    vs = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    if not vs:
        raise SystemExit(f"[calibrate] no vertices in {path}")
    return np.asarray(vs)


def band(values: np.ndarray, quantile: float) -> np.ndarray:
    """Boolean mask of entries at or above the given quantile."""
    return values >= np.quantile(values, quantile)


def mesh_pour_axis(V: np.ndarray, a_seed: np.ndarray) -> np.ndarray:
    """Horizontal spout direction implied by the mesh itself: from the
    vertex centroid to the extremal tip, iterated so a poor seed axis
    still converges on the true protrusion. Cross-checks (or replaces)
    the sidecar pour_axis."""
    a = np.asarray(a_seed, dtype=float).copy()
    a[2] = 0.0
    a /= np.linalg.norm(a)
    c = V.mean(axis=0)
    for _ in range(3):
        tip = V[band(V @ a, 0.999)].mean(axis=0)
        d = tip - c
        d[2] = 0.0
        a = d / np.linalg.norm(d)
    return a


def calibrate_teapot(V: np.ndarray, pour_axis: np.ndarray) -> dict:
    a = pour_axis / np.linalg.norm(pour_axis)
    proj = V @ a

    # spout tip: centroid of the extreme sliver along the pour axis
    tip = V[band(proj, 0.999)].mean(axis=0)

    # handle: geometry extremal the OTHER way. Take a generous back band,
    # then keep its outer half (the bar, not the wall attachment)
    back = V[band(-proj, 0.97)]
    back_out = back[band(-(back @ a), 0.5)]

    # the extremal band sees only the bar's OUTER SKIN, biasing the
    # centroid radially outward by about a bar radius; refine by fitting
    # the bar as a line and re-collecting ALL vertices near that line, so
    # the centroid straddles the tube instead of hugging one side
    pts = back_out
    for _ in range(4):
        c = pts.mean(axis=0)
        C = pts - c
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        axis = Vt[0]
        d_line = np.linalg.norm(C - (C @ axis)[:, None] * axis, axis=1)
        r_bar = max(0.006, 3.0 * float(np.median(d_line)))
        Call = V - c
        near = np.linalg.norm(
            Call - (Call @ axis)[:, None] * axis, axis=1) <= r_bar
        # stay on the handle side: admit the full tube (one bar radius
        # inward of the current band) but not body/spout vertices the
        # line sweep may graze
        near &= (-proj) >= float((-(pts @ a)).min()) - r_bar
        if near.sum() < 20:
            break
        pts = V[near]
    if axis[2] < 0:
        axis = -axis
    vert_dev = np.rad2deg(np.arccos(np.clip(abs(axis[2]), -1, 1)))

    # mid-grip point: centroid of the refined bar around its median height
    h = pts[:, 2]
    mid = pts[np.abs(h - np.median(h)) <= 0.25 * (h.max() - h.min() + 1e-9)]
    center = mid.mean(axis=0)

    up = np.array([0.0, 0.0, 1.0])
    tilt = np.cross(up, a)                  # positive roll tips spout down
    tilt /= np.linalg.norm(tilt)

    return {
        "pour_axis": a,
        "spout_tip": tip,
        "handle_center": center,
        "handle_axis": axis,
        "tilt_axis": tilt,
        "_handle_extent_z": (float(h.min()), float(h.max())),
        "_handle_vertical_dev_deg": float(vert_dev),
    }


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Kasa least-squares circle fit: x^2+y^2 + a x + b y + c = 0."""
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    (ka, kb, kc), *_ = np.linalg.lstsq(A, b, rcond=None)
    center = np.array([-ka / 2.0, -kb / 2.0])
    r = float(np.sqrt(max(center @ center - kc, 1e-12)))
    return center, r


def calibrate_mug(V: np.ndarray) -> dict:
    rim = V[band(V[:, 2], 0.98)]
    # the mug HANDLE's top often reaches the rim band and drags a plain
    # centroid sideways; fit a circle and trim radial outliers so the
    # center lands on the cylinder axis
    xy = rim[:, :2]
    for _ in range(3):
        center, r = fit_circle(xy)
        keep = np.abs(np.linalg.norm(xy - center, axis=1) - r) <= 0.25 * r
        if keep.sum() < 10:
            break
        xy = xy[keep]
    z = float(rim[:, 2].mean())
    return {"opening_center": np.array([center[0], center[1], z]),
            "rim_radius": r}


def _fmt(v) -> str:
    return np.array2string(np.asarray(v), precision=4, suppress_small=True)


def _update(spec: dict, section: str, name: str, value, note: str):
    entry = spec[section].setdefault(name, {})
    if section == "quantities":
        entry["value"] = round(float(value), 4)
    else:
        entry["xyz"] = [round(float(x), 4) for x in np.asarray(value)]
    entry["status"] = "calibrated"
    entry["comment"] = note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="update the frames.json sidecars in place")
    ap.add_argument("--recalibrate-pour-axis", action="store_true",
                    help="trust the mesh over the sidecar pour_axis: derive "
                         "the spout direction from the geometry and base "
                         "every other symbol (and tilt_axis) on it")
    args = ap.parse_args()

    # ---- teapot ------------------------------------------------------------
    tp_spec = json.loads((TEAPOT_DIR / "frames.json").read_text())
    mesh = TEAPOT_DIR / "meshes" / "teapot_visual.obj"
    if not mesh.exists():
        raise SystemExit(f"[calibrate] {mesh} missing — run "
                         "scripts/convert_asset.py first.")
    V = load_obj_vertices(mesh)
    pour = np.asarray(tp_spec["axes"]["pour_axis"]["xyz"], dtype=float)
    pour_mesh = mesh_pour_axis(V, pour)
    dev = np.rad2deg(np.arccos(np.clip(
        pour_mesh @ (pour / np.linalg.norm(pour)), -1, 1)))
    tp = calibrate_teapot(V, pour_mesh if args.recalibrate_pour_axis else pour)

    print(f"[calibrate] teapot: {len(V)} vertices, bbox "
          f"{_fmt(V.min(0))} .. {_fmt(V.max(0))}")
    print(f"  pour_axis cross-check: sidecar {_fmt(pour)} vs mesh-derived "
          f"{_fmt(pour_mesh)} ({dev:.1f} deg apart)")
    if dev > 5.0 and not args.recalibrate_pour_axis:
        print(f"  WARNING: the mesh's own spout direction disagrees with "
              f"the sidecar pour_axis by {dev:.1f} deg — the tip/handle "
              "found below are the true mesh features, but tilt_axis and "
              "the scene-facing yaw inherit the sidecar value. Rerun with "
              "--recalibrate-pour-axis to base everything on the mesh.")
    for name in ("spout_tip", "handle_center"):
        cur = tp_spec["points"][name]["xyz"]
        print(f"  points.{name:15s} {_fmt(cur)} -> {_fmt(tp[name])}")
    for name in ("handle_axis", "tilt_axis"):
        cur = tp_spec["axes"][name]["xyz"]
        print(f"  axes.{name:17s} {_fmt(cur)} -> {_fmt(tp[name])}")
    # cross-check against the COLLISION hulls: the symbols are computed on
    # the visual mesh, but the fingers grasp the hulls; VHACD can blob or
    # displace thin features like the handle bar
    col_files = sorted((TEAPOT_DIR / "meshes").glob("*col*.obj"))
    if col_files:
        Vc = np.vstack([load_obj_vertices(f) for f in col_files])
        for name in ("handle_center", "spout_tip"):
            dmin = float(np.min(np.linalg.norm(Vc - tp[name], axis=1)))
            flag = ("  <-- WARNING: nearest collision geometry is far from "
                    "the visual feature; the fingers may close on air here"
                    if dmin > 0.015 else "")
            print(f"  hull cross-check {name:14s}: nearest collision vertex "
                  f"{dmin * 1000:.1f} mm away{flag}")
    lo, hi = tp["_handle_extent_z"]
    print(f"  handle bar z extent [{lo:.3f}, {hi:.3f}] "
          f"({(hi - lo) * 1000:.0f} mm of bar; grasp TSR slide is +-20 mm)")
    dev = tp["_handle_vertical_dev_deg"]
    if dev > 30.0:
        print(f"  WARNING: handle_axis deviates {dev:.0f} deg from vertical "
              "— the vertical-loop assumption in the grasp TSR may not fit "
              "this handle; inspect in the viewer before trusting.")
    else:
        print(f"  handle_axis within {dev:.0f} deg of vertical (loop "
              "assumption holds)")

    # ---- mug ---------------------------------------------------------------
    mg_spec = json.loads((MUG_DIR / "frames.json").read_text())
    mesh = MUG_DIR / "meshes" / "mug_visual.obj"
    if not mesh.exists():
        raise SystemExit(f"[calibrate] {mesh} missing — run "
                         "scripts/convert_asset.py first.")
    Vm = load_obj_vertices(mesh)
    mg = calibrate_mug(Vm)
    print(f"[calibrate] mug: {len(Vm)} vertices, bbox "
          f"{_fmt(Vm.min(0))} .. {_fmt(Vm.max(0))}")
    cur = mg_spec["points"]["opening_center"]["xyz"]
    print(f"  points.opening_center {_fmt(cur)} -> {_fmt(mg['opening_center'])}")
    cur = mg_spec["quantities"]["rim_radius"]["value"]
    print(f"  quantities.rim_radius {cur} -> {mg['rim_radius']:.4f}")

    if not args.write:
        print("\n[calibrate] dry run — rerun with --write to update the "
              "sidecars, then REPLAN (plans aimed at placeholders are aimed "
              "at guesses).")
        return

    note = "auto-calibrated from mesh (calibrate_frames_from_mesh.py)"
    if args.recalibrate_pour_axis:
        _update(tp_spec, "axes", "pour_axis", tp["pour_axis"],
                "spout direction derived from mesh geometry; " + note)
    _update(tp_spec, "points", "spout_tip", tp["spout_tip"], note)
    _update(tp_spec, "points", "handle_center", tp["handle_center"], note)
    _update(tp_spec, "axes", "handle_axis", tp["handle_axis"], note)
    _update(tp_spec, "axes", "tilt_axis", tp["tilt_axis"],
            "up_axis x pour_axis, positive tips spout down; " + note)
    (TEAPOT_DIR / "frames.json").write_text(json.dumps(tp_spec, indent=2))
    _update(mg_spec, "points", "opening_center", mg["opening_center"], note)
    _update(mg_spec, "quantities", "rim_radius", mg["rim_radius"], note)
    (MUG_DIR / "frames.json").write_text(json.dumps(mg_spec, indent=2))
    print("\n[calibrate] sidecars updated and statuses flipped to "
          "'calibrated'. Now: PYTHONPATH=. python scripts/plan_pour_tea.py")


if __name__ == "__main__":
    main()