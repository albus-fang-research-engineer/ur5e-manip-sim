#!/usr/bin/env python3
"""Convert a mesh (OBJ/STL/GLB/PLY/...) into a robosuite-loadable MJCF object.

This is the "one script, reused for every future object" from the plan.

Pipeline:
  1. load with trimesh, optionally rescale / recenter
  2. CoACD convex decomposition -> collision geoms (MuJoCo needs convex
     collision geometry; a raw non-convex mesh would be silently convexified
     into its hull, destroying handles, cavities, spouts, sockets)
  3. write  assets/objects/<name>/<name>.xml  +  meshes/  in the exact
     structure robosuite's MujocoXMLObject expects (visual geom in group 1,
     collision geoms in group 0, and the three required sites: bottom_site,
     top_site, horizontal_radius_site)

Usage:
  python scripts/convert_asset.py assets/raw/mug.obj --name mug --mass 0.3
  python scripts/convert_asset.py model.glb --name widget --scale 0.001   # mm -> m
  python scripts/convert_asset.py mesh.obj --name part --threshold 0.02   # finer decomp

Then in code:
  from robosuite.models.objects import MujocoXMLObject
  obj = MujocoXMLObject("assets/objects/mug/mug.xml", name="mug")
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd

coacd.set_log_level("error")
import numpy as np
import trimesh


def decompose(mesh: trimesh.Trimesh, threshold: float, max_hulls: int) -> list[trimesh.Trimesh]:
    cm = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_hulls)
    return [trimesh.Trimesh(v, f) for v, f in parts]


def convert(
    src: Path,
    name: str,
    out_root: Path,
    scale: float,
    mass: float,
    threshold: float,
    max_hulls: int,
    rgba: str,
) -> Path:
    out_dir = out_root / name
    mesh_dir = out_dir / "meshes"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    mesh_dir.mkdir(parents=True)

    mesh = trimesh.load(src, force="mesh")
    if scale != 1.0:
        mesh.apply_scale(scale)
    # Center at the bounding-box centroid so the body frame == object center.
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    lo, hi = mesh.bounds
    half = (hi - lo) / 2.0
    print(f"[{name}] extents (m): {np.round(hi - lo, 4).tolist()}, faces: {len(mesh.faces)}")

    # --- visual mesh (full detail) ---
    visual_path = mesh_dir / f"{name}_visual.obj"
    mesh.export(visual_path)

    # --- collision meshes (convex pieces) ---
    pieces = decompose(mesh, threshold, max_hulls)
    print(f"[{name}] CoACD -> {len(pieces)} convex pieces")
    col_files = []
    for i, p in enumerate(pieces):
        f = mesh_dir / f"{name}_col_{i}.obj"
        p.export(f)
        col_files.append(f.name)

    # --- MJCF in robosuite MujocoXMLObject layout ---
    root = ET.Element("mujoco", model=name)
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", file=f"meshes/{visual_path.name}", name=f"{name}_visual_mesh")
    for i, fn in enumerate(col_files):
        ET.SubElement(asset, "mesh", file=f"meshes/{fn}", name=f"{name}_col_mesh_{i}")

    wb = ET.SubElement(root, "worldbody")
    outer = ET.SubElement(wb, "body")
    obj_body = ET.SubElement(outer, "body", name="object")

    # visual geom: group 1, no collision
    ET.SubElement(
        obj_body, "geom",
        type="mesh", mesh=f"{name}_visual_mesh",
        group="1", conaffinity="0", contype="0",
        rgba=rgba, mass="0.0001",
    )
    # collision geoms: group 0, share the mass evenly
    piece_mass = mass / max(len(col_files), 1)
    for i in range(len(col_files)):
        ET.SubElement(
            obj_body, "geom",
            type="mesh", mesh=f"{name}_col_mesh_{i}",
            group="0", rgba="0 1 0 0.0",
            solimp="0.998 0.998 0.001", solref="0.001 1",
            density="0", mass=f"{piece_mass:.6f}",
            friction="0.95 0.3 0.1", condim="4",
        )

    # the three sites robosuite requires on every XML object
    ET.SubElement(outer, "site", name="bottom_site", pos=f"0 0 {-half[2]:.5f}",
                  rgba="0 0 0 0", size="0.005")
    ET.SubElement(outer, "site", name="top_site", pos=f"0 0 {half[2]:.5f}",
                  rgba="0 0 0 0", size="0.005")
    ET.SubElement(outer, "site", name="horizontal_radius_site",
                  pos=f"{half[0]:.5f} {half[1]:.5f} 0", rgba="0 0 0 0", size="0.005")

    xml_path = out_dir / f"{name}.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path)
    print(f"[{name}] wrote {xml_path}")
    return xml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=Path("assets/objects"))
    ap.add_argument("--scale", type=float, default=1.0, help="uniform scale (e.g. 0.001 for mm->m)")
    ap.add_argument("--mass", type=float, default=0.2, help="total mass in kg")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD concavity threshold; lower = tighter fit, more pieces")
    ap.add_argument("--max-hulls", type=int, default=32)
    ap.add_argument("--rgba", default="0.8 0.8 0.85 1")
    args = ap.parse_args()
    convert(args.mesh, args.name, args.out, args.scale, args.mass,
            args.threshold, args.max_hulls, args.rgba)


if __name__ == "__main__":
    main()
