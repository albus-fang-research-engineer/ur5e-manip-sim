"""Authored part bands from frames.json symbols (proposal._symbol_bands):
the ground-truth part labels the oracle provider and the authored pools use
for objects without a hand-written branch (marker, cup). Synthetic geometry
with known labels, so no converted mesh is needed."""

import json
from pathlib import Path

import numpy as np
import pytest

from manip_sim.proposal import _part_of_symbol, part_masks

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def _marker_points(n=4000):
    """Cylinder r=8.7 mm along y, tail at -60.5 mm, tip at +60.5 mm."""
    y = RNG.uniform(-0.0605, 0.0605, n)
    th = RNG.uniform(0, 2 * np.pi, n)
    return np.column_stack([0.0087 * np.cos(th), y, 0.0087 * np.sin(th)]), y


def test_elongated_bands_follow_the_tail_to_tip_segment():
    spec = json.loads((ROOT / "assets/objects/marker/frames.json").read_text())
    P, y = _marker_points()
    m = part_masks("marker", P, spec)
    assert set(m) == {"tip", "body", "tail"}
    s = (y + 0.0605) / 0.121
    assert np.array_equal(m["tip"], s >= 0.8)
    assert np.array_equal(m["tail"], s < 0.2)
    assert np.array_equal(m["body"], (s >= 0.2) & (s < 0.8))
    assert not (m["tip"] & m["body"]).any() and not (m["tail"] & m["body"]).any()


def _cup(n=2000):
    """Flared open cup, axis +z: outer wall r 35->42 mm over z -35..37 mm,
    inner wall 2 mm inside it, inner floor at z=-33 mm, underside at
    z=-35 mm, rim top annulus at z=37 mm. Returns points, OUTWARD normals,
    and the true label of each sample."""
    P, N, lab = [], [], []

    def wall(offset, sign, label):
        z = RNG.uniform(-0.033, 0.037, n)
        r = 0.035 + (z + 0.035) / 0.072 * 0.007 - offset
        th = RNG.uniform(0, 2 * np.pi, n)
        P.append(np.column_stack([r * np.cos(th), r * np.sin(th), z]))
        N.append(sign * np.column_stack([np.cos(th), np.sin(th), np.zeros(n)]))
        lab.extend([label] * n)

    def disk(z0, r0, r1, nz, label):
        r = np.sqrt(RNG.uniform(r0 ** 2, r1 ** 2, n))
        th = RNG.uniform(0, 2 * np.pi, n)
        P.append(np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, z0)]))
        N.append(np.tile([0.0, 0.0, nz], (n, 1)))
        lab.extend([label] * n)

    wall(0.0, +1, "outer")
    wall(0.002, -1, "inner")
    disk(-0.033, 0.0, 0.033, +1, "floor")
    disk(-0.035, 0.0, 0.035, -1, "underside")
    disk(0.037, 0.040, 0.042, +1, "rimtop")
    return np.vstack(P), np.vstack(N), np.array(lab)


@pytest.mark.parametrize("winding", [+1, -1])
def test_container_interior_is_inner_wall_and_floor_either_winding(winding):
    spec = json.loads((ROOT / "assets/objects/cup/frames.json").read_text())
    P, N, lab = _cup()
    m = part_masks("cup", P, spec, normals=winding * N)
    assert {"rim", "interior", "base"} <= set(m)
    inside = np.isin(lab, ["inner", "floor"])
    got = m["interior"]
    assert got[inside].mean() > 0.97             # inner wall + floor
    assert not got[np.isin(lab, ["outer", "underside", "rimtop"])].any()
    assert (lab[m["base"]] == "underside").mean() > 0.9


def test_container_without_normals_has_no_interior_band():
    spec = json.loads((ROOT / "assets/objects/cup/frames.json").read_text())
    P, _, _ = _cup()
    assert set(part_masks("cup", P, spec)) == {"rim", "base"}


def test_object_without_band_symbols_keeps_top_base_fallback():
    P, _ = _marker_points()
    spec = {"points": {"grip_center": {"xyz": [0, 0, 0]}}, "axes": {}, "quantities": {}}
    assert set(part_masks("widget", P, spec)) == {"top", "base"}


def test_authored_point_named_for_its_part_labels_that_part():
    assert _part_of_symbol("tip", ("tip", "body", "tail")) == "tip"
    assert _part_of_symbol("tail_center", ("tip", "body", "tail")) == "tail"
    assert _part_of_symbol("grip_center", ("tip", "body", "tail")) is None
    assert _part_of_symbol("opening_center", ("rim", "interior", "base")) == "rim"


def test_oracle_resolves_call1_names_on_the_new_bands():
    from scripts.ground_parts import _oracle_band
    cup = json.loads((ROOT / "assets/objects/cup/frames.json").read_text())
    marker = json.loads((ROOT / "assets/objects/marker/frames.json").read_text())
    cb = {"rim": 0, "interior": 0, "base": 0}
    mb = {"tip": 0, "body": 0, "tail": 0}
    assert _oracle_band("opening", cb, cup) == "rim"
    assert _oracle_band("interior", cb, cup) == "interior"
    assert _oracle_band("rim", cb, cup) == "rim"
    assert _oracle_band("body", mb, marker) == "body"
    assert _oracle_band("tip", mb, marker) == "tip"
    assert _oracle_band("cap", mb, marker) is None       # no synonym table, by design
