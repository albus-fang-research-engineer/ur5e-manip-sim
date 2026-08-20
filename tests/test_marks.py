"""manip_sim.perception.marks — provider-neutral mark sets (no MuJoCo)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manip_sim.perception.marks import (MIN_AREA_PX, build_marks,  # noqa: E402
                                        from_mask_dir, load_gt, load_marks)
from manip_sim.vlm import Vocabulary  # noqa: E402

pytest.importorskip("PIL")


def _scene():
    rgb = np.full((120, 160, 3), 200, np.uint8)
    a = np.zeros((120, 160), bool); a[70:110, 10:50] = True      # bottom-left
    b = np.zeros((120, 160), bool); b[10:50, 100:150] = True     # top-right
    tiny = np.zeros((120, 160), bool); tiny[60, 80] = True       # noise
    return rgb, [a, b, tiny]


def test_ids_are_reading_order_and_noise_dropped(tmp_path):
    rgb, masks = _scene()
    ms = build_marks(rgb, masks, "segbuffer", tmp_path, gt_names=["left", "right", "noise"])
    assert ms.ids() == (1, 2)
    assert load_gt(tmp_path) == {1: "right", 2: "left"}        # top row first
    assert ms.marks[1].bbox == (100, 10, 149, 49)
    assert masks[2].sum() < MIN_AREA_PX
    assert {p.name for p in tmp_path.iterdir()} >= {
        "rgb.png", "marked.png", "mask_1.png", "mask_2.png", "marks.json", "marks.gt.json"}


def test_roundtrip_and_vocab_sees_no_names(tmp_path):
    rgb, masks = _scene()
    build_marks(rgb, masks, "segbuffer", tmp_path, gt_names=["left", "right", "noise"])
    ms = load_marks(tmp_path)
    assert ms.source == "segbuffer" and ms.dir == tmp_path
    assert (ms.load_mask(2) == masks[0]).all()
    v = Vocabulary.from_marks(ms)
    assert set(v.marks) == {1, 2}
    text = v.describe_marks() + json.dumps(json.loads((tmp_path / "marks.json").read_text()))
    assert "left" not in text and "right" not in text


def test_sam_dump_ingests_to_same_layout(tmp_path):
    from PIL import Image
    rgb, masks = _scene()
    build_marks(rgb, masks, "segbuffer", tmp_path / "ref", gt_names=["l", "r", "n"])
    dump = tmp_path / "dump"; dump.mkdir()
    Image.fromarray(rgb).save(dump / "rgb.png")
    # provider writes in arbitrary order / names
    Image.fromarray(masks[1].astype(np.uint8) * 255).save(dump / "obj_zz.png")
    Image.fromarray(masks[0].astype(np.uint8) * 255).save(dump / "obj_aa.png")
    ms = from_mask_dir(dump / "rgb.png", dump, "sam", tmp_path / "sam")
    ref = load_marks(tmp_path / "ref")
    assert ms.source == "sam" and not (tmp_path / "sam/marks.gt.json").exists()
    assert {i: m.bbox for i, m in ms.marks.items()} == {i: m.bbox for i, m in ref.marks.items()}
