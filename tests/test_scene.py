"""scenes/<name>.json loader (manip_sim.scene) — no MuJoCo needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manip_sim.scene import load_scene  # noqa: E402

SCENE = ROOT / "scenes/pour_tea.json"
pytestmark = pytest.mark.skipif(
    not (ROOT / "assets/objects/teapot/frames.json").exists(),
    reason="needs committed frames.json")


def test_manifest_matches_retired_constants():
    s = load_scene(SCENE)
    assert s.table_size == (1.2, 1.2, 0.05) and s.table_top_z == 0.8
    assert s.asset_dirs == {"teapot": Path("assets/objects/teapot"),
                            "mug": Path("assets/objects/mug")}
    assert s.object_xmls["mug"] == "assets/objects/mug/mug.xml"
    np.testing.assert_allclose(s.xy("teapot"), [0.0, -0.25])
    np.testing.assert_allclose(s.xy("mug"), [0.0, 0.25])
    assert s.spawn_z == pytest.approx(0.86)


def test_face_yaw_matches_pour_axis_derivation():
    s = load_scene(SCENE)
    # retired SPOUT_YAW_OFFSET = 2.381 was a hand copy of atan2(pour_axis)
    assert s.yaw("teapot") == pytest.approx(np.pi / 2 - 2.381, abs=1e-3)
    assert s.yaw("mug") == 0.0
    pos, quat = s.fixed_poses()["teapot"]
    assert pos.shape == (3,) and quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_relative_paths_resolve_from_repo_root(tmp_path):
    doc = (SCENE.read_text().replace('"yaw": {"face": "mug", "along": "pour_axis"}',
                                     '"yaw": 0.5'))
    p = tmp_path / "s.json"
    p.write_text(doc)
    s = load_scene(p)
    assert s.yaw("teapot") == 0.5 and s.path == p
