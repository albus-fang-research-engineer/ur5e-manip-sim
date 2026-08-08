"""Render the proposed interaction point pool onto canonical views of each
object — the inspection step BEFORE writing candidates.json, and the first
concrete piece of the marked-render generator (the same camera projection
and depth-based occlusion test later drive SoM mark culling for the VLM).

For each object it regenerates the pool in memory with the exact same
deterministic parameters as scripts/propose_interaction_points.py (same
seed, same constants — what you see here is byte-for-byte what --write
would save), renders the VISUAL mesh (geom group 1 only, matching the
render_plan.py convention) from eight canonical cameras covering the full
view sphere — the four horizontal semi-axes (slightly elevated), straight
down, straight up, and two opposed three-quarter views. Full coverage is
required for the occlusion flags to mean anything: with cameras on one
side only, far-side candidates are hidden in every view by construction.

and overlays every candidate as a colored mark with its pool ID:

    red     constructed   (primitive-derived; opening_center, spout_tip, ...)
    blue    part          (per-part quota samples)
    orange  curvature     (angle-defect saliency samples)
    teal    fps           (surface coverage)

Filled mark: the point is VISIBLE in that view (its projected depth
matches the rendered depth buffer). Hollow mark: occluded behind the mesh
in that view — drawn anyway because 3D inspection needs it, but this is
exactly the per-view legibility signal the SoM generator will later use
to cull marks (every candidate should be filled in at least one view).
Constructed off-surface points (mid_cavity, opening_center) will show
hollow in side views and filled from the top — that is correct behavior,
not a bug.

Output: outputs/candidates/<name>.png (2x2 montage + legend). No sidecar
is touched; write candidates.json afterwards with
scripts/propose_interaction_points.py --write once the pool looks right.

Run from the repo root (headless; pick the backend as usual):

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
        PYTHONPATH=. python scripts/render_candidates.py
    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_candidates.py
    ... python scripts/render_candidates.py --object mug
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manip_sim.proposal import load_obj, propose

OBJECTS = {
    "teapot": Path("assets/objects/teapot"),
    "mug": Path("assets/objects/mug"),
}
OUT_DIR = Path("outputs/candidates")

VIEW_PX = 640                # per-view render size (square)
FOVY_DEG = 40.0
CLASS_COLOR = {"constructed": (230, 57, 70), "part": (69, 123, 157),
               "curvature": (244, 162, 97), "fps": (42, 157, 143)}
OCCLUSION_TOL_M = 0.004      # depth-buffer agreement tolerance


# ---------------------------------------------------------------------------
# model: the object's own MJCF, meshdir absolutized, cameras + lights added
# ---------------------------------------------------------------------------

def lookat_quat(pos: np.ndarray, target: np.ndarray,
                up_hint: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a camera at pos looking at target (MuJoCo
    convention: camera looks along its -z, +y is image-up)."""
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(fwd, up_hint)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def canonical_cameras(center: np.ndarray, radius: float) -> dict[str, dict]:
    """Eight canonical views covering the full view sphere: the four
    horizontal semi-axes (slightly elevated), straight down, straight up,
    and two opposed three-quarter views. Full coverage is what makes the
    >=1-view legibility invariant satisfiable for every surface point —
    with cameras on one side only, far-side candidates are unavoidably
    hidden everywhere and the occlusion flags stop meaning anything."""
    d = 2.8 * radius
    eps = 1e-4                      # breaks up-hint colinearity for top/bottom
    views = {
        "+x": center + d * np.array([1.0, 0.0, 0.35]),
        "-x": center + d * np.array([-1.0, 0.0, 0.35]),
        "+y": center + d * np.array([0.0, 1.0, 0.35]),
        "-y": center + d * np.array([0.0, -1.0, 0.35]),
        "top": center + d * np.array([0.0, eps, 1.0]),
        "bottom": center + d * np.array([0.0, eps, -1.0]),
        "iso": center + d * np.array([0.7, 0.7, 0.6]),
        "iso-opp": center + d * np.array([-0.7, -0.7, 0.6]),
    }
    z_up = np.array([0.0, 0.0, 1.0])
    return {name: {"pos": pos, "quat": lookat_quat(pos, center, z_up)}
            for name, pos in views.items()}


def build_model(name: str, obj_dir: Path,
                cams: dict[str, dict]) -> mujoco.MjModel:
    tree = ET.parse(obj_dir / f"{name}.xml")
    root = tree.getroot()
    # visual-only render: drop the collision hulls (mesh assets + geoms)
    # entirely instead of merely hiding them — no reason to load 32 hulls
    # per object for a visualization that never shows them
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh" and "_col_" in child.get("name", ""):
                parent.remove(child)
            elif child.tag == "geom" and "_col_" in child.get("mesh", ""):
                parent.remove(child)
    ET.SubElement(root, "compiler", meshdir=str(obj_dir.resolve()))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(VIEW_PX),
                  offheight=str(VIEW_PX))
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45",
                  diffuse="0.6 0.6 0.6")
    wb = root.find("worldbody")
    for cname, c in cams.items():
        ET.SubElement(wb, "camera", name=cname, fovy=str(FOVY_DEG),
                      pos=" ".join(f"{v:.6f}" for v in c["pos"]),
                      quat=" ".join(f"{v:.6f}" for v in c["quat"]))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


# ---------------------------------------------------------------------------
# pinhole projection matching the MJCF cameras exactly
# ---------------------------------------------------------------------------

def project(points: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) pixels + camera depth (meters along the view axis) for body
    points, given the same pos/quat written into the MJCF."""
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, cam["quat"])
    R = R.reshape(3, 3)
    pc = (points - cam["pos"]) @ R                 # world -> camera frame
    depth = -pc[:, 2]
    f = (VIEW_PX / 2.0) / np.tan(np.deg2rad(FOVY_DEG) / 2.0)
    u = VIEW_PX / 2.0 + f * pc[:, 0] / depth
    v = VIEW_PX / 2.0 - f * pc[:, 1] / depth
    return np.column_stack([u, v]), depth


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_marks(img: Image.Image, uv: np.ndarray, visible: np.ndarray,
               candidates: list[dict]) -> None:
    dr = ImageDraw.Draw(img)
    font = _font(13)
    r = 6
    for c, (u, v), vis in zip(candidates, uv, visible):
        col = CLASS_COLOR[c["source"]]
        if not (0 <= u < VIEW_PX and 0 <= v < VIEW_PX):
            continue
        if vis:
            dr.ellipse([u - r, v - r, u + r, v + r], fill=col,
                       outline=(255, 255, 255), width=2)
        else:
            dr.ellipse([u - r, v - r, u + r, v + r],
                       outline=col, width=2)
        dr.text((u + r + 2, v - r - 4), str(c["id"]),
                fill=col if vis else col + (140,), font=font,
                stroke_width=2, stroke_fill=(255, 255, 255))


def montage(views: dict[str, Image.Image], name: str) -> Image.Image:
    font = _font(16)
    legend_h = 34
    cols = 4
    rows = (len(views) + cols - 1) // cols
    out = Image.new("RGB", (cols * VIEW_PX, rows * VIEW_PX + legend_h),
                    (255, 255, 255))
    for i, (vname, im) in enumerate(views.items()):
        x, y = (i % cols) * VIEW_PX, (i // cols) * VIEW_PX
        ImageDraw.Draw(im).text((10, 8), f"{name} — {vname}",
                                fill=(30, 30, 30), font=font)
        out.paste(im, (x, y))
    dr = ImageDraw.Draw(out)
    x = 12
    for cls, col in CLASS_COLOR.items():
        y = rows * VIEW_PX + legend_h // 2
        dr.ellipse([x, y - 6, x + 12, y + 6], fill=col)
        dr.text((x + 18, y - 9), cls, fill=(30, 30, 30), font=font)
        x += 18 + 10 * len(cls) + 30
    dr.text((x + 20, rows * VIEW_PX + legend_h // 2 - 9),
            "filled = visible in view, hollow = occluded", fill=(90, 90, 90),
            font=font)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(name: str, obj_dir: Path) -> None:
    mesh = obj_dir / "meshes" / f"{name}_visual.obj"
    if not mesh.exists():
        raise SystemExit(f"[render-candidates] {mesh} missing — run "
                         "scripts/convert_asset.py first.")
    spec = json.loads((obj_dir / "frames.json").read_text())
    V, F = load_obj(mesh)
    pool = propose(name, V, F, spec)
    X = np.array([c["xyz"] for c in pool.candidates])

    center = 0.5 * (V.min(0) + V.max(0))
    radius = float(np.linalg.norm(V - center, axis=1).max())
    cams = canonical_cameras(center, radius)
    model = build_model(name, obj_dir, cams)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 0        # visual mesh only, no collision hulls
    vopt.geomgroup[1] = 1

    renderer = mujoco.Renderer(model, VIEW_PX, VIEW_PX)
    views: dict[str, Image.Image] = {}
    vis_count = np.zeros(len(X), dtype=int)
    for vname, cam in cams.items():
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=vname, scene_option=vopt)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=vname, scene_option=vopt)
        depth = renderer.render().copy()

        uv, pdepth = project(X, cam)
        px = np.clip(uv.round().astype(int), 0, VIEW_PX - 1)
        buf = depth[px[:, 1], px[:, 0]]
        visible = pdepth <= buf + OCCLUSION_TOL_M
        vis_count += visible.astype(int)

        img = Image.fromarray(rgb)
        draw_marks(img, uv, visible, pool.candidates)
        views[vname] = img
    renderer.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    montage(views, name).save(out)

    print(f"[render-candidates] {name}: {len(X)} candidates -> {out}")
    hidden = [c["id"] for c, n in zip(pool.candidates, vis_count) if n == 0]
    if hidden:
        print(f"  note: candidates {hidden} are occluded in ALL eight "
              "views — expected only for interior constructed points "
              "(mid_cavity) or points inside closed cavities; a SURFACE "
              "sample here violates the >=1-view legibility invariant "
              "and needs attention before SoM marking")
    print("  pool matches the dry run of propose_interaction_points.py "
          "exactly (same seed/constants); --write there when satisfied.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", choices=sorted(OBJECTS),
                    help="restrict to one object (default: both)")
    args = ap.parse_args()
    for name, obj_dir in OBJECTS.items():
        if args.object and name != args.object:
            continue
        run(name, obj_dir)


if __name__ == "__main__":
    main()