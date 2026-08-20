"""Object marks: the provider-neutral mask contract that VLM call #1 is
addressed against (OmniManip/ReKep order: mark every foreground object
on RGB -> VLM picks by ID -> mesh/register only the picked ones).

One format, two providers, identical downstream:

  segbuffer   sim default — MuJoCo segmentation buffer, grouped per
              body (scripts/capture_rgbd_packet.py). Exact; the oracle
              segmentation arm.
  sam         hardware — GroundedSAM/SAM3 on the same RGB. Also runnable
              on a sim render as an ablation arm that isolates
              segmentation-induced failures from planning failures.

Layout of a mark set directory (what both providers must produce):

  rgb.png          the frame the marks live on
  marked.png       rgb with numbered labels (what the VLM sees)
  mask_<id>.png    HxW 8-bit {0,255}, one per mark
  marks.json       {"source": ..., "image": "rgb.png", "marked": ...,
                    "marks": {"1": {"mask": "mask_1.png",
                                    "bbox": [u0,v0,u1,v1],
                                    "centroid": [u,v], "area": px}}}
  marks.gt.json    SIM ONLY: {"1": "teapot", ...}. Kept in a separate
                   file so nothing that builds a prompt can read it by
                   accident; scripts/plan_stages.py's verifier is the
                   only consumer.

IDs are 1-based ints in reading order of mask centroid (top-left first)
so the same scene segmented twice gets the same numbering up to mask
jitter. Marks below MIN_AREA_PX are dropped as noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MIN_AREA_PX = 150


@dataclass(frozen=True)
class Mark:
    id: int
    mask: str                      # file name relative to the set dir
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]  # (u, v) pixels
    area: int


@dataclass(frozen=True)
class MarkSet:
    source: str                    # "segbuffer" | "sam"
    image: str                     # rgb file name relative to dir
    marks: dict[int, Mark] = field(default_factory=dict)
    marked: str | None = None
    dir: Path | None = None

    # -- accept set / prompt text ---------------------------------------
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.marks))

    def describe(self) -> str:
        return "\n".join(f"  mark {i}: bbox {list(m.bbox)}, area {m.area} px"
                         for i, m in sorted(self.marks.items()))

    # -- files -----------------------------------------------------------
    def mask_path(self, mid: int) -> Path:
        return (self.dir or Path(".")) / self.marks[mid].mask

    def load_mask(self, mid: int) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(self.mask_path(mid))) > 127

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = {"source": self.source, "image": self.image, "marked": self.marked,
               "marks": {str(i): {k: v for k, v in asdict(m).items() if k != "id"}
                         for i, m in sorted(self.marks.items())}}
        p = out_dir / "marks.json"
        p.write_text(json.dumps(doc, indent=2) + "\n")
        return p


def load_marks(path: str | Path) -> MarkSet:
    """`path` is marks.json or its directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "marks.json"
    doc = json.loads(path.read_text())
    marks = {int(i): Mark(id=int(i), mask=m["mask"], bbox=tuple(m["bbox"]),
                          centroid=tuple(m["centroid"]), area=int(m["area"]))
             for i, m in doc["marks"].items()}
    return MarkSet(source=doc["source"], image=doc["image"], marks=marks,
                   marked=doc.get("marked"), dir=path.parent)


def load_gt(path: str | Path) -> dict[int, str]:
    """Sim-only id -> manifest object name (marks.gt.json). Verifier use."""
    path = Path(path)
    if path.is_dir():
        path = path / "marks.gt.json"
    return {int(i): n for i, n in json.loads(path.read_text()).items()}


# ---------------------------------------------------------------- builders

def _stats(mask: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[float, float], int]:
    vs, us = np.nonzero(mask)
    return ((int(us.min()), int(vs.min()), int(us.max()), int(vs.max())),
            (float(us.mean()), float(vs.mean())), int(mask.sum()))


def build_marks(rgb: np.ndarray, masks: list[np.ndarray], source: str,
                out_dir: Path, gt_names: list[str] | None = None,
                min_area: int = MIN_AREA_PX) -> MarkSet:
    """Assign IDs, write masks/rgb/marked/marks.json (+ marks.gt.json when
    `gt_names` parallels `masks`). Empty or tiny masks are dropped."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [(m, gt_names[k] if gt_names else None)
            for k, m in enumerate(masks) if m.any() and int(m.sum()) >= min_area]
    # reading order: row-major by centroid, quantised to 1/8 image height
    # so near-equal rows don't flip order between providers
    h = rgb.shape[0]
    keyed = []
    for m, name in keep:
        bbox, (cu, cv), area = _stats(m)
        keyed.append(((round(cv / (h / 8)), cu), m, name, bbox, (cu, cv), area))
    keyed.sort(key=lambda t: t[0])

    Image.fromarray(rgb).save(out_dir / "rgb.png")
    marks, gt = {}, {}
    for i, (_, m, name, bbox, c, area) in enumerate(keyed, start=1):
        fn = f"mask_{i}.png"
        Image.fromarray((m.astype(np.uint8) * 255)).save(out_dir / fn)
        marks[i] = Mark(id=i, mask=fn, bbox=bbox, centroid=c, area=area)
        if name is not None:
            gt[i] = name
    ms = MarkSet(source=source, image="rgb.png", marks=marks, marked="marked.png",
                 dir=out_dir)
    render_marks(rgb, ms, out_dir / "marked.png")
    ms.write(out_dir)
    if gt:
        (out_dir / "marks.gt.json").write_text(
            json.dumps({str(i): n for i, n in gt.items()}, indent=2) + "\n")
    return ms


def from_mask_dir(rgb_path: Path, mask_dir: Path, source: str, out_dir: Path,
                  pattern: str = "*.png") -> MarkSet:
    """Any provider that dumped one binary PNG per object (a SAM node)
    -> canonical mark set. File order does not matter; IDs are
    reassigned in reading order."""
    from PIL import Image
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    masks = [np.asarray(Image.open(p).convert("L")) > 127
             for p in sorted(mask_dir.glob(pattern))
             if p.resolve() != Path(rgb_path).resolve()]
    return build_marks(rgb, masks, source, out_dir)


def from_packet(packet_dir: Path, out_dir: Path) -> MarkSet:
    """segbuffer provider: scripts/capture_rgbd_packet.py output
    (packet.npz with rgb + mask_<obj> + meta.objects)."""
    z = np.load(Path(packet_dir) / "packet.npz")
    meta = json.loads(str(z["meta"]))
    names = list(meta["objects"])
    masks = [np.asarray(z[f"mask_{n}"], bool) for n in names]
    return build_marks(np.asarray(z["rgb"], np.uint8), masks, "segbuffer",
                       out_dir, gt_names=names)


# --------------------------------------------------------------- rendering

def render_marks(rgb: np.ndarray, ms: MarkSet, out_png: Path,
                 label_px: int = 22) -> None:
    """Numbered label at each mask centroid + mask outline. Style matches
    render_candidates.py's marks so calls #1 and #2 look alike to the
    model: white disc, dark ring, dark numeral."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_px)
    except OSError:
        font = ImageFont.load_default()
    for i, m in sorted(ms.marks.items()):
        try:
            mask = ms.load_mask(i)
            edge = mask & ~_erode(mask)
            vs, us = np.nonzero(edge)
            draw.point(list(zip(us.tolist(), vs.tolist())), fill=(255, 230, 0))
        except Exception:
            pass                       # outline is cosmetic
        u, v = m.centroid
        r = label_px * 0.75
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 255, 255),
                     outline=(20, 20, 20), width=3)
        t = str(i)
        tw, th = draw.textbbox((0, 0), t, font=font)[2:]
        draw.text((u - tw / 2, v - th / 2 - label_px * 0.1), t,
                  fill=(20, 20, 20), font=font)
    img.save(out_png)


def _erode(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    m[1:] &= mask[:-1]; m[:-1] &= mask[1:]
    m[:, 1:] &= mask[:, :-1]; m[:, :-1] &= mask[:, 1:]
    return m
