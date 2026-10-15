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

Texture. MuJoCo ignores .mtl files: trimesh's export writes `vt` and a
material.mtl -> material_0.png beside the visual OBJ, but the geom only
shows the texture when the MJCF declares <texture>/<material> and puts
the material on it. --texture PNG does that (copied to
meshes/<name>_texture.png). Pass the image the SOURCE mesh's UVs index —
the PNG next to the scan's own textured.obj, never whatever material_0.png
trimesh resolved from a shared assets/raw/: scans exported as
textured.obj/textured.mtl/texture_map.png collide there. --texture-xml
wires an already-converted asset in place (no reconversion, no CoACD).

Usage:
  python scripts/convert_asset.py assets/raw/mug.obj --name mug --mass 0.3
  python scripts/convert_asset.py --texture-xml assets/objects/marker/marker.xml --texture SCAN/texture_map.png
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

import numpy as np
import trimesh


def decompose(mesh: trimesh.Trimesh, threshold: float, max_hulls: int) -> list[trimesh.Trimesh]:
    import coacd                      # lazy: --texture-xml / --reweight-xml need no CoACD
    coacd.set_log_level("error")
    cm = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_hulls)
    return [trimesh.Trimesh(v, f) for v, f in parts]


def hull_masses(pieces: list[trimesh.Trimesh], total_mass: float) -> np.ndarray:
    """Split total_mass across convex pieces proportionally to hull volume
    (uniform density). CoACD hulls overlap slightly at the cuts, so the
    volumes are approximate partition weights, not an exact partition —
    the induced COM error is sub-mm, versus tens of mm for an even split."""
    vols = np.array([max(abs(p.volume), 1e-12) for p in pieces])
    return total_mass * vols / vols.sum()


def wire_texture(xml_path: Path, png: Path) -> Path:
    """Put a colour texture on an existing converted asset's visual geom, in
    place: copy `png` to meshes/<name>_texture.png, declare <texture> +
    <material> (file RELATIVE to the XML, like the meshes: robosuite's
    MujocoXMLObject absolutizes every asset file= against the XML's folder,
    and render_candidates.build_model injects texturedir), set the material
    on the group-1 geom and drop its rgba. Refuses when the visual OBJ has
    no `vt` (a texture would map nowhere). Collision geoms, masses, sites
    and frames.json are untouched; re-running replaces the texture."""
    name = xml_path.stem
    mesh_dir = xml_path.parent / "meshes"
    visual = mesh_dir / f"{name}_visual.obj"
    if not visual.exists():
        raise SystemExit(f"[{name}] missing {visual}")
    with open(visual) as f:
        n_vt = sum(1 for line in f if line.startswith("vt "))
    if not n_vt:
        raise SystemExit(f"[{name}] {visual.name} has no vt lines — no UVs "
                         "to map a texture with; reconvert from a textured source")
    from PIL import Image
    Image.open(png).verify()                         # a readable image, not a stub
    dst = mesh_dir / f"{name}_texture.png"
    if Path(png).resolve() != dst.resolve():
        shutil.copyfile(png, dst)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    tex, mat = f"{name}_tex", f"{name}_mat"
    for child in list(asset):                        # idempotent re-wire
        if child.get("name") in (tex, mat):
            asset.remove(child)
    ET.SubElement(asset, "texture", name=tex, type="2d",
                  file=f"meshes/{dst.name}")
    ET.SubElement(asset, "material", name=mat, texture=tex,
                  specular="0.1", shininess="0.1")
    geoms = [g for g in root.iter("geom")
             if g.get("mesh") == f"{name}_visual_mesh"]
    if len(geoms) != 1:
        raise SystemExit(f"[{name}] expected one visual geom, found {len(geoms)}")
    geoms[0].attrib.pop("rgba", None)
    geoms[0].set("material", mat)
    ET.indent(root)
    tree.write(xml_path)
    print(f"[{name}] texture {Path(png).name} -> {dst} ({n_vt} vt); "
          f"material {mat} on {name}_visual_mesh")
    return dst


def reweight_xml(xml_path: Path) -> None:
    """Rewrite ONLY the mass="..." attributes of the collision geoms in an
    existing converted XML, in place, weighting by hull volume read from the
    sibling meshes/ directory. Everything else — solref, solimp, friction,
    formatting — is preserved byte-for-byte (targeted regex, no ET rewrite).
    Total mass is the sum of the existing collision-geom masses, so whatever
    --mass the original conversion used is preserved exactly."""
    import re

    text = xml_path.read_text()
    name = xml_path.stem
    pat = re.compile(
        rf'(<geom[^>]*mesh="{name}_col_mesh_(\d+)"[^>]*mass=")([0-9.eE+-]+)(")')
    hits = list(pat.finditer(text))
    if not hits:
        raise SystemExit(f"[{name}] no collision-geom mass attributes found "
                         f"in {xml_path}")
    total = sum(float(m.group(3)) for m in hits)
    pieces = []
    for m in hits:
        f = xml_path.parent / "meshes" / f"{name}_col_{m.group(2)}.obj"
        if not f.exists():
            raise SystemExit(f"[{name}] missing {f} — reweighting needs the "
                             "hull meshes next to the XML")
        pieces.append(trimesh.load(f, force="mesh"))
    masses = hull_masses(pieces, total)

    # report the COM this rewrite moves (hull centroids, both weightings)
    C = np.array([p.center_mass for p in pieces])
    com_even = C.mean(axis=0)
    com_vol = (masses[:, None] * C).sum(axis=0) / masses.sum()
    print(f"[{name}] {len(pieces)} hulls, total mass {total:.6g} kg "
          "(preserved)")
    print(f"[{name}] COM even-split -> volume-weighted moves "
          f"{np.linalg.norm(com_vol - com_even) * 1000:.1f} mm "
          f"(even {np.round(com_even * 1000, 1).tolist()} mm, "
          f"weighted {np.round(com_vol * 1000, 1).tolist()} mm)")

    by_idx = {int(m.group(2)): masses[k] for k, m in enumerate(hits)}
    text = pat.sub(
        lambda m: f"{m.group(1)}{by_idx[int(m.group(2))]:.6g}{m.group(4)}",
        text)
    xml_path.write_text(text)
    print(f"[{name}] rewrote mass attributes in place -> {xml_path}")


def convert(
    src: Path,
    name: str,
    out_root: Path,
    scale: float,
    mass: float,
    threshold: float,
    max_hulls: int,
    rgba: str,
    texture: Path | None = None,
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
    # collision geoms: group 0, mass split BY HULL VOLUME. An even split
    # (the previous behavior) makes the body's COM/inertia an artifact of
    # where CoACD spent hulls — high-curvature parts (handle, spout) get
    # many small hulls and drag the COM toward them (~28 mm error measured
    # on a teapot-like mesh vs ~0.6 mm volume-weighted). With density="0"
    # MuJoCo takes each geom's mass attribute and computes inertia from
    # the hull shape, so volume weighting recovers the uniform-density
    # body the visual mesh implies.
    piece_masses = hull_masses(pieces, mass)
    for i in range(len(col_files)):
        ET.SubElement(
            obj_body, "geom",
            type="mesh", mesh=f"{name}_col_mesh_{i}",
            group="0", rgba="0 1 0 0.0",
            solimp="0.998 0.998 0.001", solref="0.01 1",
            density="0", mass=f"{piece_masses[i]:.6g}",
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
    if texture is not None:
        wire_texture(xml_path, texture)
    return xml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path, nargs="?",
                    help="source mesh (omit with --reweight-xml)")
    ap.add_argument("--reweight-xml", type=Path, default=None,
                    help="rewrite ONLY the mass attributes of an existing "
                         "converted XML in place, volume-weighted from its "
                         "meshes/ hulls; no re-conversion, meshes untouched")
    ap.add_argument("--name", default=None,
                    help="object name (required for conversion; unused with "
                         "--reweight-xml, which takes it from the XML stem)")
    ap.add_argument("--out", type=Path, default=Path("assets/objects"))
    ap.add_argument("--scale", type=float, default=1.0, help="uniform scale (e.g. 0.001 for mm->m)")
    ap.add_argument("--mass", type=float, default=0.2, help="total mass in kg")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD concavity threshold; lower = tighter fit, more pieces")
    ap.add_argument("--max-hulls", type=int, default=32)
    ap.add_argument("--rgba", default="0.8 0.8 0.85 1",
                    help="visual colour when no --texture is given")
    ap.add_argument("--texture", type=Path, default=None, metavar="PNG",
                    help="colour texture the source mesh's UVs index (see "
                         "module docstring); replaces --rgba")
    ap.add_argument("--texture-xml", type=Path, default=None, metavar="XML",
                    help="wire --texture onto an existing converted XML in "
                         "place; no reconversion, meshes and hulls untouched")
    args = ap.parse_args()
    if args.texture_xml is not None:
        if args.texture is None:
            ap.error("--texture-xml needs --texture")
        wire_texture(args.texture_xml, args.texture)
        return
    if args.reweight_xml is not None:
        reweight_xml(args.reweight_xml)
        return
    if args.mesh is None or args.name is None:
        ap.error("mesh and --name are required unless --reweight-xml is given")
    convert(args.mesh, args.name, args.out, args.scale, args.mass,
            args.threshold, args.max_hulls, args.rgba, args.texture)


if __name__ == "__main__":
    main()
