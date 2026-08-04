"""Propose the interaction point candidate POOL for the pour-tea objects —
the 3D-propose half of the 3D-propose / 2D-select grounding pipeline
(algorithm in manip_sim/proposal.py; this script is the driver, following
the calibrate_frames_from_mesh.py dry-run / --write convention).

For each object (teapot, mug) it:

  1. loads the visual mesh (vertices + faces) in body coordinates
  2. generates the four candidate classes:
       fps          surface coverage (farthest point sampling)
       curvature    angle-defect saliency samples (rims, tips, bars)
       part         per-part quota inside geometric part bands derived
                    from the calibrated frames.json symbols
       constructed  primitive-derived points surface sampling cannot
                    produce (opening_center, mid_cavity, spout_tip, ...)
                    — sourced from frames.json where already calibrated
  3. unions them, greedy-NMS deduplicates with class priority
     (constructed > part > curvature > fps), caps the pool at the 30-50
     budget, and enforces per-part coverage
  4. dry run: prints the pool with per-class raw->kept counts
     --write: saves assets/objects/<name>/candidates.json — the artifact
     the marked-render generator (SoM views for VLM multiple choice)
     consumes next

Determinism: fixed seed constant; same mesh + same frames.json -> the
same pool, so candidate IDs are stable across reruns (required for the
typed-repair loop to reference selections by ID).

Prerequisites: converted meshes present (scripts/convert_asset.py) and
frames.json calibrated (scripts/calibrate_frames_from_mesh.py --write) —
the part bands and constructed points anchor on those symbols.

Run from the repo root:

    PYTHONPATH=. python scripts/propose_interaction_points.py
    PYTHONPATH=. python scripts/propose_interaction_points.py --write
    PYTHONPATH=. python scripts/propose_interaction_points.py --object mug --write
"""

import argparse
import json
from pathlib import Path

import numpy as np

from manip_sim.proposal import load_obj, pool_to_json, propose

OBJECTS = {
    "teapot": Path("assets/objects/teapot"),
    "mug": Path("assets/objects/mug"),
}


def _fmt(v) -> str:
    return np.array2string(np.asarray(v, dtype=float), precision=4,
                           suppress_small=True)


def run(name: str, obj_dir: Path, write: bool) -> None:
    mesh = obj_dir / "meshes" / f"{name}_visual.obj"
    if not mesh.exists():
        raise SystemExit(f"[proposal] {mesh} missing — run "
                         "scripts/convert_asset.py first.")
    frames = obj_dir / "frames.json"
    spec = json.loads(frames.read_text())
    V, F = load_obj(mesh)

    pool = propose(name, V, F, spec)
    doc = pool_to_json(name, pool)

    print(f"[proposal] {name}: {len(V)} vertices / {len(F)} faces, bbox "
          f"{_fmt(V.min(0))} .. {_fmt(V.max(0))}")
    raw, kept = pool.class_counts_raw, pool.class_counts_kept
    for cls in ("constructed", "part", "curvature", "fps"):
        print(f"  {cls:12s} {raw.get(cls, 0):3d} raw -> "
              f"{kept.get(cls, 0):3d} kept")
    print(f"  pool: {len(pool.candidates)} candidates "
          f"(budget {doc['params']['pool_budget']}, "
          f"nms {doc['params']['nms_radius_m'] * 1000:.0f} mm)")
    if pool.readmitted:
        print(f"  coverage guarantee re-admitted parts: "
              f"{', '.join(pool.readmitted)}")
    for c in pool.candidates:
        tag = c.get("symbol") or c.get("part") or ""
        sal = (f"  saliency {c['saliency']:.3f}"
               if "saliency" in c else "")
        print(f"  [{c['id']:2d}] {c['source']:11s} {tag:14s} "
              f"{_fmt(c['xyz'])}{sal}")

    out = obj_dir / "candidates.json"
    if write:
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"  wrote {out}")
    else:
        print(f"  dry run (no file written) — pass --write to save {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", choices=sorted(OBJECTS),
                    help="restrict to one object (default: both)")
    ap.add_argument("--write", action="store_true",
                    help="save assets/objects/<name>/candidates.json")
    args = ap.parse_args()

    for name, obj_dir in OBJECTS.items():
        if args.object and name != args.object:
            continue
        run(name, obj_dir, args.write)


if __name__ == "__main__":
    main()