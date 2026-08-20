"""Produce the object mark set for VLM call #1 from a scene manifest.

Sim default provider is the MuJoCo segmentation buffer (exact, per body);
the output format is identical to what the hardware SAM node writes, so
everything downstream — the marked image the model sees, the mark-ID
accept set, the verifier — is provider-agnostic. See
manip_sim/perception/marks.py for the layout.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/mark_scene.py
    PYTHONPATH=. python scripts/mark_scene.py --scene scenes/X.json --out outputs/marks/X
    PYTHONPATH=. python scripts/mark_scene.py --from-masks rgb.png masks/ --source sam

--from-masks ingests any provider's per-object PNGs (a SAM dump) into
the same layout; that is the segmentation-ablation arm on sim renders
and the only path on hardware.

Default out: outputs/marks/<scene name>/. The sim provider also writes
marks.gt.json (id -> manifest name) for the verifier; no prompt code
reads it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manip_sim.perception.marks import build_marks, from_mask_dir
from manip_sim.scene import add_scene_arg, load_scene

CAMERA = "agentview"
H, W = 480, 640


def segbuffer_marks(scene, out: Path, camera: str = CAMERA, h: int = H, w: int = W,
                    robot: str = "UR5e"):
    import robosuite.utils.camera_utils as CU

    from manip_sim.scene import make_env
    from scripts.capture_rgbd_packet import masks_from_segmentation, project

    env, objs = make_env(scene, robot=robot, has_renderer=False,
                         has_offscreen_renderer=True, use_camera_obs=True,
                         camera_names=[camera], camera_heights=h, camera_widths=w)
    obs = env._get_observations(force_update=True)
    rgb = np.asarray(obs[f"{camera}_image"], np.uint8)[::-1].copy()
    try:
        seg = CU.get_camera_segmentation(sim=env.sim, camera_name=camera,
                                         camera_height=h, camera_width=w)
    except TypeError:
        seg = env.sim.render(camera_name=camera, height=h, width=w, segmentation=True)
    seg = np.asarray(seg)[::-1]
    names = list(objs)
    masks = masks_from_segmentation(env, seg, names)

    # same raster self-check as capture_rgbd_packet: body origin must
    # project inside its mask, else flip
    K = CU.get_camera_intrinsic_matrix(env.sim, camera, h, w)
    T_wc = CU.get_camera_extrinsic_matrix(env.sim, camera)

    def ok(mk):
        for n in names:
            bid = env.obj_body_ids[n]
            uv = project(K, T_wc, env.sim.data.body_xpos[bid])
            if uv is None:
                return False
            u, v = int(round(uv[0])), int(round(uv[1]))
            if not (0 <= v < h and 0 <= u < w and mk[n][max(0, v - 6):v + 7,
                                                          max(0, u - 6):u + 7].any()):
                return False
        return True

    if not ok(masks):
        flipped = {k: m[::-1].copy() for k, m in masks.items()}
        if ok(flipped):
            masks = flipped
        else:
            raise RuntimeError("segmentation/raster conventions inconsistent")
    env.close()
    return build_marks(rgb, [masks[n] for n in names], "segbuffer", out, gt_names=names)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_arg(ap)
    ap.add_argument("--out", default=None, metavar="DIR")
    ap.add_argument("--camera", default=CAMERA)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--from-masks", nargs=2, metavar=("RGB", "MASK_DIR"),
                    help="ingest another provider's per-object PNGs")
    ap.add_argument("--source", default="sam",
                    help="provider tag recorded with --from-masks")
    args = ap.parse_args()
    scene = load_scene(args.scene)
    out = Path(args.out or f"outputs/marks/{scene.name}")

    if args.from_masks:
        ms = from_mask_dir(Path(args.from_masks[0]), Path(args.from_masks[1]),
                           args.source, out)
    else:
        ms = segbuffer_marks(scene, out, camera=args.camera, robot=args.robot)
    print(f"[mark-scene] {ms.source}: {len(ms.marks)} marks -> {out}")
    for i, m in sorted(ms.marks.items()):
        print(f"  mark {i}: bbox {list(m.bbox)} area {m.area}")
    print(f"  next: PYTHONPATH=. python scripts/plan_stages.py --scene {args.scene} "
          f"--marks {out}")


if __name__ == "__main__":
    main()
