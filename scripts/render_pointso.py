"""Render PointSO's semantic-orientation predictions as arrow overlays on
canonical views of each object — the montage companion to
scripts/pointso_mesh_test.py, in the render_candidates.py house style (and
reusing its camera / projection / model-building machinery verbatim, so the
views match the interaction-point montages pixel-for-pixel).

Per object, eight canonical views of the visual mesh with, per instruction:

    solid arrow    PointSO prediction        (per-instruction color)
    dashed arrow   frames.json ground truth  (same color)
    label          "<instruction>  <angular error>°" at the solid tip

Both arrows are anchored at the sampled-cloud centroid; matching colors
diverging is the whole readout — a glance tells you which semantics the
model has and which it doesn't. Views where an arrow points into the
camera foreshorten to a dot; read those axes from the orthogonal views.

Predictions come from either
  (a) live queries against the pointso sidecar (default) — flags mirror
      pointso_mesh_test.py (--partial / --camera-frame / --azimuth ...),
      so what you render is exactly what that harness scores; or
  (b) an .npz saved by pointso_mesh_test.py --save-npz, via --npz DIR
      (no server needed; GT is recomputed from frames.json).

Output: outputs/pointso/<name>_montage.png

    PYTHONPATH=. python scripts/render_pointso.py
    PYTHONPATH=. python scripts/render_pointso.py --partial --camera-frame
    PYTHONPATH=. python scripts/render_pointso.py --npz outputs/pointso
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scripts.render_candidates import (VIEW_PX, _font, build_model,
                                       canonical_cameras, project)
from scripts.pointso_mesh_test import (ASSETS, INSTRUCTIONS,
                                       angular_error_deg,
                                       camera_frame_rotation,
                                       hidden_point_removal, load_frames,
                                       load_visual_mesh, resolve_gt,
                                       sample_cloud, virtual_camera)

OUT_DIR = Path("outputs/pointso")

# Okabe-Ito colorblind-safe palette, cycled per instruction
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (204, 121, 167),
           (213, 94, 0), (0, 114, 178), (240, 228, 66)]


# ---------------------------------------------------------------------------
# arrow drawing (2D, after projection)
# ---------------------------------------------------------------------------

def _draw_segment(dr, p0, p1, color, width, dashed=False):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if not dashed:
        dr.line([tuple(p0), tuple(p1)], fill=color, width=width)
        return
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return
    v = v / L
    dash, gap, s = 9.0, 6.0, 0.0
    while s < L:
        e = min(s + dash, L)
        dr.line([tuple(p0 + s * v), tuple(p0 + e * v)],
                fill=color, width=width)
        s = e + gap


def draw_arrow(dr, p0, p1, color, width=4, dashed=False):
    """2D arrow p0 -> p1 with a chevron head; degenerates to a dot when
    the 3D axis points into the camera."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 6.0:  # foreshortened: axis ~parallel to view axis
        dr.ellipse([p1[0] - 5, p1[1] - 5, p1[0] + 5, p1[1] + 5],
                   outline=color, width=3)
        return
    _draw_segment(dr, p0, p1, color, width, dashed)
    v = v / L
    n = np.array([-v[1], v[0]])
    h = min(14.0, 0.35 * L)
    for side in (+1, -1):
        tip = p1 - h * v + side * 0.55 * h * n
        dr.line([tuple(p1), tuple(tip)], fill=color, width=width)


def overlay_axes(img, cam, origin, entries, arrow_len):
    """entries: list of (label, color, pred_dir | None, gt_dir | None,
    err_deg | None). Directions are unit vectors in body frame."""
    dr = ImageDraw.Draw(img)
    font = _font(14)
    for label, color, pred, gt, err in entries:
        if gt is not None:
            (uv, _) = project(np.stack([origin, origin + arrow_len * gt]),
                              cam)
            draw_arrow(dr, uv[0], uv[1], color, width=3, dashed=True)
        if pred is not None:
            (uv, _) = project(np.stack([origin, origin + arrow_len * pred]),
                              cam)
            draw_arrow(dr, uv[0], uv[1], color, width=4, dashed=False)
            txt = label if err is None else f"{label}  {err:.0f}\u00b0"
            dr.text((uv[1][0] + 6, uv[1][1] - 8), txt, fill=color,
                    font=font, stroke_width=2, stroke_fill=(255, 255, 255))


def montage(views, name, entries):
    font = _font(16)
    legend_h = 34
    cols = 4
    rows = (len(views) + cols - 1) // cols
    out = Image.new("RGB", (cols * VIEW_PX, rows * VIEW_PX + legend_h),
                    (255, 255, 255))
    for i, (vname, im) in enumerate(views.items()):
        x, y = (i % cols) * VIEW_PX, (i // cols) * VIEW_PX
        ImageDraw.Draw(im).text((10, 8), f"{name} \u2014 {vname}",
                                fill=(30, 30, 30), font=font,
                                stroke_width=2, stroke_fill=(255, 255, 255))
        out.paste(im, (x, y))
    dr = ImageDraw.Draw(out)
    x, y = 12, rows * VIEW_PX + legend_h // 2
    for label, color, _, _, err in entries:
        txt = label if err is None else f"{label} ({err:.0f}\u00b0)"
        dr.line([x, y, x + 22, y], fill=color, width=4)
        dr.text((x + 28, y - 9), txt, fill=(30, 30, 30), font=font)
        x += 28 + 9 * len(txt) + 26
    dr.text((x + 14, y - 9), "solid = PointSO, dashed = frames.json GT",
            fill=(90, 90, 90), font=font)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def get_predictions(obj, cloud, args):
    """(instructions, preds-in-body-frame). Live sidecar query unless
    --npz points at a pointso_mesh_test.py --save-npz dump."""
    instructions = [i for i, _ in INSTRUCTIONS[obj]]
    if args.npz:
        f = Path(args.npz) / f"pointso_{obj}.npz"
        if not f.exists():
            raise SystemExit(f"[render-pointso] {f} missing — run "
                             "pointso_mesh_test.py --save-npz first, or "
                             "drop --npz for a live query.")
        d = np.load(f, allow_pickle=True)
        return [str(s) for s in d["instructions"]], \
            np.asarray(d["predictions"], np.float32)

    from manip_sim.perception.pointso_client import PointSOClient
    client = PointSOClient(addr=args.addr)

    send = cloud
    R_bc = None
    if args.partial:
        cam = virtual_camera(send[:, :3], args.azimuth, args.elevation,
                             args.cam_dist)
        send = send[hidden_point_removal(send[:, :3], cam)]
    if args.camera_frame:
        cam = virtual_camera(send[:, :3], args.azimuth, args.elevation,
                             args.cam_dist)
        R_bc = camera_frame_rotation(send[:, :3], cam)
        send = send.copy()
        send[:, :3] = (send[:, :3] - cam[None, :]) @ R_bc.T
    preds = client.orient_batch(send, instructions)
    if R_bc is not None:
        preds = preds @ R_bc
    return instructions, preds


def run(obj, args):
    mesh = load_visual_mesh(obj)
    frames = load_frames(obj)
    cloud = sample_cloud(mesh, args.n_points, args.seed)
    instructions, preds = get_predictions(obj, cloud, args)

    origin = cloud[:, :3].mean(axis=0)
    spec_by_ins = dict(INSTRUCTIONS[obj])
    entries = []
    for k, ins in enumerate(instructions):
        gt = resolve_gt(spec_by_ins.get(ins), frames, mesh, cloud[:, :3])
        pred = np.asarray(preds[k], float)
        err = None if gt is None else angular_error_deg(pred, gt)
        entries.append((ins, PALETTE[k % len(PALETTE)], pred, gt, err))

    V = np.asarray(mesh.vertices)
    center = 0.5 * (V.min(0) + V.max(0))
    radius = float(np.linalg.norm(V - center, axis=1).max())
    arrow_len = 1.35 * radius

    cams = canonical_cameras(center, radius)
    model = build_model(obj, ASSETS / obj, cams)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 0
    vopt.geomgroup[1] = 1

    renderer = mujoco.Renderer(model, VIEW_PX, VIEW_PX)
    views = {}
    for vname, cam in cams.items():
        renderer.update_scene(data, camera=vname, scene_option=vopt)
        img = Image.fromarray(renderer.render().copy())
        overlay_axes(img, cam, origin, entries, arrow_len)
        views[vname] = img
    renderer.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{obj}_montage.png"
    montage(views, obj, entries).save(out)
    scored = [e for _, _, _, g, e in entries if g is not None]
    tag = "npz" if args.npz else "live"
    print(f"[render-pointso] {obj} ({tag}): {len(entries)} instructions"
          + (f", mean err {np.mean(scored):.1f}\u00b0" if scored else "")
          + f" -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--objects", nargs="+",
                    default=sorted(INSTRUCTIONS.keys()),
                    choices=sorted(INSTRUCTIONS.keys()))
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz", default=None, metavar="DIR",
                    help="render saved predictions instead of querying live")
    ap.add_argument("--addr", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--camera-frame", action="store_true")
    ap.add_argument("--azimuth", type=float, default=30.0)
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--cam-dist", type=float, default=0.6)
    args = ap.parse_args()
    for obj in args.objects:
        run(obj, args)


if __name__ == "__main__":
    main()
