"""Runtime part grounding (manip_sim.part_grounding): lift, fit, symbols,
and the grounded pool, on synthetic geometry — no GL, no meshes."""

import numpy as np
import pytest

from manip_sim import part_grounding as pg
from manip_sim.proposal import propose

UP = np.array([0.0, 0.0, 1.0])
RNG = np.random.default_rng(0)


def ring(r=0.04, z=0.05, n=400, noise=3e-4):
    t = RNG.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t), np.full(n, z)]) + \
        RNG.normal(0, noise, (n, 3))


def bar(p0, p1, r=0.006, n=400):
    t = RNG.uniform(0, 1, n)[:, None]
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    e1 = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    ph = RNG.uniform(0, 2 * np.pi, n)[:, None]
    return p0 + t * (p1 - p0) + r * (np.cos(ph) * e1 + np.sin(ph) * e2)


# ------------------------------------------------------------------ fits

def test_ring_fit():
    g = pg.fit_part("rim", ring(), UP, np.zeros(3))
    assert g.primitive == "ring"
    assert abs(g.radius - 0.04) < 1e-3
    assert np.allclose(g.center, [0, 0, 0.05], atol=1e-3)
    assert g.axis @ UP > 0.99


def test_line_fit_signed_away_from_body_and_tip_at_far_end():
    body = np.zeros(3)
    P = bar(np.array([0.05, 0, 0.02]), np.array([0.14, 0, 0.04]))
    g = pg.fit_part("spout", P, UP, body)
    assert g.primitive == "line"
    assert g.axis[0] > 0.9                          # points away from body
    assert np.allclose(g.tip, [0.14, 0, 0.04], atol=4e-3)
    assert 0.08 < g.length < 0.1


def test_blob_fit_and_ungrounded():
    P = RNG.normal(0, 0.01, (300, 3)) + [0, 0, 0.08]
    g = pg.fit_part("lid", P, UP, np.zeros(3))
    assert g.primitive == "blob" and g.axis @ UP > 0.9
    assert pg.fit_part("knob", P[:5], UP, np.zeros(3)).primitive == "ungrounded"


# --------------------------------------------------------------- symbols

def test_symbols_are_part_named_and_loadable(tmp_path):
    import json
    from manip_sim.frames import load_symbols
    parts = {"rim": pg.fit_part("rim", ring(), UP, np.zeros(3)),
             "handle": pg.fit_part("handle", bar(np.array([-0.05, 0, 0]),
                                                 np.array([-0.05, 0, 0.08])), UP, np.zeros(3)),
             "spout": pg.fit_part("spout", bar(np.array([0.05, 0, 0.02]),
                                               np.array([0.14, 0, 0.04])), UP, np.zeros(3)),
             "gone": pg.GroundedPart("gone", "ungrounded", 0)}
    doc = pg.symbols_from_parts("pot", parts, UP, "test")
    assert set(doc["points"]) == {"rim_center", "handle_center", "handle_tip",
                                  "spout_center", "spout_tip"}
    assert "rim_radius" in doc["quantities"] and "spout_length" in doc["quantities"]
    # canonical frame: up always; no front supplied -> no front/lateral
    assert "up_axis" in doc["axes"]
    assert "front_axis" not in doc["axes"] and "lateral_axis" not in doc["axes"]
    assert not any(k.endswith("_lateral_axis") for k in doc["axes"])
    # per-symbol sigma is structured: fit sigma for line parts
    assert doc["axes"]["spout_axis"]["sigma_deg"] > 0
    (tmp_path / "frames.json").write_text(json.dumps(doc))
    sym = load_symbols(tmp_path)
    assert sym.sigmas["axes.spout_axis"] == doc["axes"]["spout_axis"]["sigma_deg"]
    f = sym.frame("spout_tip", "spout_axis")
    assert abs(f.axis @ sym.axes["spout_axis"]) > 0.999


def test_canonical_front_is_caller_supplied_and_orthogonalized(tmp_path):
    import json
    from manip_sim.frames import load_symbols
    parts = {"spout": pg.fit_part("spout", bar(np.array([0.05, 0, 0.02]),
                                               np.array([0.14, 0, 0.04])), UP, np.zeros(3))}
    # a front tipped 20 deg off horizontal is projected into the up-plane
    doc = pg.symbols_from_parts("pot", parts, UP, "test",
                                front=np.array([np.cos(0.35), 0.0, np.sin(0.35)]),
                                up_sigma_deg=2.0, front_sigma_deg=3.0)
    assert np.allclose(doc["axes"]["front_axis"]["xyz"], [1, 0, 0], atol=1e-3)
    assert np.allclose(doc["axes"]["lateral_axis"]["xyz"], [0, 1, 0], atol=1e-3)
    assert doc["axes"]["front_axis"]["sigma_deg"] == 3.0
    assert doc["axes"]["lateral_axis"]["sigma_deg"] == pytest.approx(np.hypot(2, 3), abs=0.01)
    (tmp_path / "frames.json").write_text(json.dumps(doc))
    sym = load_symbols(tmp_path)
    assert sym.sigmas["axes.up_axis"] == 2.0
    # a front (anti)parallel to up is unusable: canonical frame is up only
    doc = pg.symbols_from_parts("pot", parts, UP, "test", front=UP)
    assert "front_axis" not in doc["axes"] and "lateral_axis" not in doc["axes"]


# ------------------------------------------------------------------ lift

def test_lift_majority_vote_and_depth_test():
    N = 6
    uv = {"a": np.array([[5, 5]] * N, float), "b": np.array([[5, 5]] * N, float)}
    vis = {"a": np.array([1, 1, 1, 0, 1, 0], bool),
           "b": np.array([1, 1, 0, 1, 1, 0], bool)}
    m_on = np.zeros((10, 10), bool); m_on[5, 5] = True
    m_off = np.zeros((10, 10), bool)
    masks = {"a": {"x": m_on, "y": m_off}, "b": {"x": m_off, "y": m_on}}
    # sample 0,1,4: visible in both, one vote each -> tie, no winner
    # sample 2: visible in a only -> x ; sample 3: visible in b only -> y
    # sample 5: visible nowhere -> none
    out = pg.lift_masks(uv, vis, masks, N)
    assert out["x"].tolist() == [False, False, True, False, False, False]
    assert out["y"].tolist() == [False, False, False, True, False, False]


def test_raster_roundtrips_through_lift():
    N = 50
    uv = np.column_stack([RNG.uniform(5, 60, N), RNG.uniform(5, 60, N)])
    vis = np.ones(N, bool)
    lab = {"p": np.arange(N) < 20}
    masks = pg.raster_masks(uv, vis, lab, 64, 64, dilate_px=0)
    out = pg.lift_masks({"v": uv}, {"v": vis}, {"v": masks}, N)
    assert out["p"].tolist() == lab["p"].tolist()


# -------------------------------------------------------- grounded pool

def _box_mesh(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def test_propose_with_part_points_uses_grounded_labels_only():
    V, F = _box_mesh([-0.05, -0.05, 0], [0.05, 0.05, 0.1])
    top = np.column_stack([RNG.uniform(-0.05, 0.05, 300),
                           RNG.uniform(-0.05, 0.05, 300), np.full(300, 0.1)])
    spec = {"points": {"cap_center": {"xyz": [0, 0, 0.1]}},
            "axes": {"up_axis": {"xyz": [0, 0, 1]}}, "quantities": {}}
    pool = propose("widget", V, F, spec, part_points={"cap": top})
    parts = {c.get("part") for c in pool.candidates}
    assert "cap" in parts and parts <= {"cap", None}
    con = [c for c in pool.candidates if c["source"] == "constructed"]
    assert [c["symbol"] for c in con] == ["cap_center"]
    assert con[0]["part"] == "cap"
