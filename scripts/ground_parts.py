"""Runtime part grounding driver: the part names VLM call #1 emitted ->
per-view part masks -> lifted surface points -> fitted primitives ->
outputs/grounding/<scene>/<object>/{frames.json, candidates.json}.

This closes the circularity plan_stages.py inherited: until now call #1's
part names were checked against candidates.json built from the HAND-
AUTHORED frames.json, so the symbol table preceded the call that was
supposed to produce it. After --write, every downstream script run with
`--grounding outputs/grounding/<scene>` reads the runtime tables instead
(manip_sim.scene.Scene.asset_dirs), and the authored sidecars become the
ground-truth arm of the grounding ablation.

Per object (only those in the stage plan's object_parts):

  1. views    eight canonical renders of the object (render_candidates'
              cameras, depth-tested) -> <out>/<obj>/views/<view>.png —
              the images a 2D segmenter labels
  2. masks    provider-specific:
                --provider oracle  rasterize the geometric part bands
                                   (proposal.part_masks, which READ the
                                   authored frames.json) into per-view
                                   masks. Exercises the whole lift path
                                   with perfect masks; it is NOT
                                   grounding from names — it is the
                                   segmentation-oracle arm, and it
                                   cannot run on an object without an
                                   authored sidecar.
                --provider masks   read <masks-root>/<obj>/<view>/<part>.png
                                   (8-bit, nonzero = part) written by
                                   GroundedSAM / SAM3 on the views from
                                   step 1 (or, on hardware, on the
                                   registration frame with its own
                                   camera — then pass --views-json).
  3. lift     manip_sim.part_grounding.lift_masks: depth-tested
              projection of dense surface samples, majority vote
  4. fit      one primitive per part -> <part>_center/_axis/_tip/...
  5. pool     proposal.propose with the lifted part point sets, so
              candidates.json's `part` labels are the grounded ones and
              call #2's menus / plan_stages' grounds[] check read them

Dry run prints the fits; --write writes frames.json + candidates.json and
symlinks the authored <obj>.xml and meshes/ into the grounding dir so
every consumer that expects an asset dir layout works unchanged.

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/ground_parts.py \\
        --stage-plan outputs/stage_plan/pour_tea.marks.json --provider oracle --write
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/ground_parts.py \\
        --stage-plan ... --provider masks --masks-root outputs/grounding/pour_tea/sam --write
    # then everything downstream with:
    ... --grounding outputs/grounding/pour_tea
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from manip_sim.part_grounding import (fit_part, lift_masks, raster_masks,
                                      symbols_from_parts)
from manip_sim.proposal import (DENSE_SAMPLES, SEED, load_obj, part_masks,
                                pool_to_json, propose, surface_samples)
from manip_sim.scene import add_scene_arg, load_scene

VIEW_PX = 640
UP_BODY = np.array([0.0, 0.0, 1.0])     # sim: assets are converted upright


def render_depth_views(name: str, obj_dir: Path, V: np.ndarray,
                       out_views: Path | None, px: int = VIEW_PX):
    """Eight canonical views: rgb (saved), depth buffers, camera dicts.
    Imports mujoco lazily so the fit path stays importable without GL."""
    import mujoco
    from PIL import Image

    from scripts.render_candidates import (build_model, canonical_cameras,
                                           project, OCCLUSION_TOL_M)

    center = 0.5 * (V.min(0) + V.max(0))
    radius = float(np.linalg.norm(V - center, axis=1).max())
    cams = canonical_cameras(center, radius)
    model = build_model(name, obj_dir, cams, px=px)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 0
    vopt.geomgroup[1] = 1
    renderer = mujoco.Renderer(model, px, px)
    depths, paths = {}, {}
    for vname in cams:
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=vname, scene_option=vopt)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=vname, scene_option=vopt)
        depths[vname] = renderer.render().copy()
        if out_views is not None:
            out_views.mkdir(parents=True, exist_ok=True)
            p = out_views / f"{vname.replace('+', 'p').replace('-', 'n')}.png"
            Image.fromarray(rgb).save(p)
            paths[vname] = p
    renderer.close()

    def project_samples(P):
        uv_by, vis_by = {}, {}
        for vname, cam in cams.items():
            uv, pdepth = project(P, cam, px=px)
            pix = np.clip(uv.round().astype(int), 0, px - 1)
            buf = depths[vname][pix[:, 1], pix[:, 0]]
            uv_by[vname] = uv
            vis_by[vname] = pdepth <= buf + OCCLUSION_TOL_M
        return uv_by, vis_by

    return cams, project_samples, paths


def read_mask_dir(root: Path, obj: str, views, parts) -> dict[str, dict[str, np.ndarray]]:
    from PIL import Image
    out = {}
    for vname in views:
        vdir = root / obj / vname.replace('+', 'p').replace('-', 'n')
        if not vdir.exists():
            continue
        out[vname] = {}
        for part in parts:
            f = vdir / f"{part}.png"
            if f.exists():
                out[vname][part] = np.asarray(Image.open(f).convert("L")) > 0
    return out


def _oracle_band(part: str, bands: dict, spec: dict) -> str | None:
    """Free-text part name -> oracle band, by the same substring rule
    selection._part_match uses, falling back through the authored
    symbol names (\"opening\" -> opening_center -> rim)."""
    from manip_sim.proposal import _part_of_symbol
    t = part.strip().lower()
    for b in bands:
        if t == b or t in b or b in t:
            return b
    for sym in spec.get("points", {}):
        if t in sym.lower() or sym.lower() in t:
            b = _part_of_symbol(sym, tuple(bands))
            if b in bands:
                return b
    return None


def ground_object(name: str, obj_dir: Path, parts: tuple[str, ...],
                  out_dir: Path, provider: str, masks_root: Path | None,
                  write: bool) -> dict:
    mesh = obj_dir / "meshes" / f"{name}_visual.obj"
    if not mesh.exists():
        raise SystemExit(f"[ground] {mesh} missing — run scripts/convert_asset.py first")
    V, F = load_obj(mesh)
    P, _ = surface_samples(V, F, DENSE_SAMPLES, np.random.default_rng(SEED))
    obj_out = out_dir / name
    cams, project_samples, view_paths = render_depth_views(
        name, obj_dir, V, obj_out / "views")
    uv_by, vis_by = project_samples(P)

    if provider == "oracle":
        spec = json.loads((obj_dir / "frames.json").read_text())
        bands = part_masks(name, P, spec)            # authored-anchored
        labels, missing = {}, []
        for p in parts:
            b = _oracle_band(p, bands, spec)
            (labels.__setitem__(p, bands[b]) if b else missing.append(p))
        if missing:
            print(f"[ground] {name}: oracle has no band for {missing} — "
                  f"those parts will come back ungrounded")
        masks_by_view = {v: raster_masks(uv_by[v], vis_by[v], labels, VIEW_PX, VIEW_PX)
                         for v in cams}
    elif provider == "masks":
        masks_by_view = read_mask_dir(masks_root, name, cams, parts)
        if not masks_by_view:
            raise SystemExit(f"[ground] no masks under {masks_root / name}; "
                             f"expected <view>/<part>.png for views {sorted(cams)}")
    else:
        raise SystemExit(f"unknown provider {provider}")

    lifted = lift_masks(uv_by, vis_by, masks_by_view, len(P))
    body_center = V.mean(axis=0)
    grounded = {p: fit_part(p, P[lifted[p]] if p in lifted else P[:0],
                            UP_BODY, body_center) for p in parts}
    print(f"[ground] {name} ({provider}): {len(P)} samples, "
          f"{sum(int(m.sum()) for m in lifted.values())} labelled")
    for g in grounded.values():
        print(f"    {g.summary()}")

    frames = symbols_from_parts(name, grounded, UP_BODY, provider)
    frames["provenance"]["views"] = {v: str(p) for v, p in view_paths.items()}
    part_points = {p: P[lifted[p]] for p in parts
                   if p in lifted and grounded[p].primitive != "ungrounded"}
    pool = propose(name, V, F, frames, part_points=part_points)
    doc = pool_to_json(name, pool)
    print(f"    symbols: points {sorted(frames['points'])}, axes "
          f"{sorted(frames['axes'])}, quantities {sorted(frames['quantities'])}")
    print(f"    pool: {len(pool.candidates)} candidates, part classes "
          f"{sorted({c.get('part') for c in pool.candidates if c.get('part')})}")

    if write:
        obj_out.mkdir(parents=True, exist_ok=True)
        (obj_out / "frames.json").write_text(json.dumps(frames, indent=2) + "\n")
        (obj_out / "candidates.json").write_text(json.dumps(doc, indent=2) + "\n")
        for link, target in ((f"{name}.xml", obj_dir / f"{name}.xml"),
                             ("meshes", obj_dir / "meshes")):
            lp = obj_out / link
            if lp.is_symlink() or lp.exists():
                lp.unlink()
            os.symlink(os.path.relpath(target.resolve(), obj_out.resolve()), lp)
        print(f"    wrote {obj_out}/{{frames.json,candidates.json}} (+ xml/mesh links)")
    return {p: g.primitive for p, g in grounded.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-plan", required=True, metavar="JSON",
                    help="plan_stages.py artifact; its object_parts name "
                         "what to ground")
    ap.add_argument("--provider", choices=("oracle", "masks"), default="oracle")
    ap.add_argument("--masks-root", default=None, metavar="DIR")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="default outputs/grounding/<scene>")
    ap.add_argument("--write", action="store_true")
    add_scene_arg(ap)
    args = ap.parse_args()
    if args.provider == "masks" and not args.masks_root:
        ap.error("--provider masks needs --masks-root")
    scene = load_scene(args.scene)                 # authored dirs, on purpose
    out = Path(args.out or f"outputs/grounding/{scene.name}")
    parts = json.loads(Path(args.stage_plan).read_text())["object_parts"]

    results = {}
    for name, ps in parts.items():
        if not ps:
            continue
        if name not in scene.objects:
            print(f"[ground] {name!r} not in scene manifest — skipped")
            continue
        results[name] = ground_object(name, scene.asset_dirs[name], tuple(ps),
                                      out, args.provider,
                                      Path(args.masks_root) if args.masks_root else None,
                                      args.write)
    ung = [f"{o}.{p}" for o, r in results.items() for p, k in r.items() if k == "ungrounded"]
    if ung:
        raise SystemExit(f"[ground] ungrounded parts {ung} — call #1 named a part "
                         "the masks do not cover; that is a typed failure for "
                         "the repair touchpoint, not something to paper over")
    if args.write:
        print(f"\n[ground] next: re-run plan_stages.py --replay <artifact> --grounding {out}, "
              f"then select_frames.py / plan_pour_tea.py --grounding {out}")
    else:
        print("\n[ground] dry run — --write to emit the grounding dir")


if __name__ == "__main__":
    main()
