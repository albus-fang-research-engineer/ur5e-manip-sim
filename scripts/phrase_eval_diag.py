"""Read-only diagnostic for a masks root of phrase-prompted SAM3 masks, one
subdir per stage plan (<root>/<plan>/<obj>/{manifest.json,<view>/<part>.png}),
as make_part_masks --phrases writes them. No SAM3, no API, writes nothing.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/phrase_eval_diag.py \\
        ~/ur5e-manip-hardware/outputs/runs/phrase_eval

Per run and object:
  1. views hit per part, and which prompt produced each mask (phrase or bare name)
  2. pairwise mask IoU between the object's parts, over views where both exist
  3. each part lifted + fitted TOGETHER (the pipeline's exclusive vote) and
     ALONE, with the ring-gate numbers (plane rms/R, circle rms/R, arc;
     ring needs <=0.08, <=0.08, >=0.60)
  4. line fits: the far end (`<part>_tip` in frames.json) and its y
"""
from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

from manip_sim.part_grounding import _circle, _plane, fit_part, lift_masks
from manip_sim.proposal import DENSE_SAMPLES, SEED, load_obj, surface_samples
from scripts.ground_parts import UP_BODY, read_mask_dir, render_depth_views

if len(sys.argv) != 2:
    sys.exit(__doc__)
R = Path(sys.argv[1]).expanduser()
GEOM = {}


def geom(obj):
    if obj not in GEOM:
        d = Path(f"assets/objects/{obj}")
        V, F = load_obj(d / "meshes" / f"{obj}_visual.obj")
        P, _ = surface_samples(V, F, DENSE_SAMPLES, np.random.default_rng(SEED))
        cams, project, _ = render_depth_views(obj, d, V, None)
        uv, vis = project(P)
        GEOM[obj] = (V, P, cams, uv, vis)
    return GEOM[obj]


def fit(obj, parts, masks_root, p):
    V, P, cams, uv, vis = geom(obj)
    mv = read_mask_dir(masks_root, obj, cams, parts)
    lab = lift_masks(uv, vis, mv, len(P))
    Q = P[lab[p]] if p in lab else P[:0]
    g = fit_part(p, Q, UP_BODY, V.mean(0))
    s = f"{g.primitive:10s} n={len(Q):5d}"
    if g.center is not None:
        s += f" c={np.round(g.center, 4).tolist()}"
    if g.radius is not None:
        s += f" r={g.radius:.4f}"
    if len(Q) >= 30:
        c, n, _, prms, _ = _plane(Q)
        _, Rr, crms, arc = _circle(Q, n, c)
        s += f"  gate plane {prms / Rr:.2f} circle {crms / Rr:.2f} arc {arc:.2f} (R {Rr:.4f})"
    if g.primitive == "line":
        s += f"  far end y={g.tip[1]:+.4f}"
    return s


for K in sorted(p.name for p in R.iterdir() if p.is_dir()):
    print(f"\n######## {K}")
    for obj in ("marker", "cup"):
        man = json.load(open(R / K / obj / "manifest.json"))
        parts = man["parts"]
        print(f"  {obj}")
        for p in parts:
            recs = [v["parts"][p] for v in man["views"].values()]
            by = {}
            for r in recs:
                if r["written"]:
                    by[r["prompt"]] = by.get(r["prompt"], 0) + 1
            print(f"    {p:14s} phrase {man['prompts'][p]['prompt']!r:24s} "
                  f"hit {sum(by.values())}/{len(recs)}  by prompt {by}")
        for a, b in combinations(parts, 2):
            ious = []
            for v in man["views"].values():
                fa, fb = (R / K / obj / v["file"] / f"{x}.png" for x in (a, b))
                if fa.exists() and fb.exists():
                    ma = np.asarray(Image.open(fa).convert("L")) > 0
                    mb = np.asarray(Image.open(fb).convert("L")) > 0
                    ious.append((ma & mb).sum() / max((ma | mb).sum(), 1))
            if ious:
                print(f"    IoU {a} / {b}: {np.mean(ious):.2f} over {len(ious)} views")
        for p in parts:
            print(f"    together {p:14s} {fit(obj, parts, R / K, p)}")
            print(f"    alone    {p:14s} {fit(obj, [p], R / K, p)}")
