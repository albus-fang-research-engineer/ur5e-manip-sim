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


# ------------------------------------------------------- depth semantics
#
# depth_check is decided by the generalized winding number of the point
# w.r.t. the mesh -- "is this point inside material" -- never by primitive.
# The cases below are closed tubes swept along planar curves, fitted from
# their own surface samples exactly as ground_object does.

def _tube(center_line, r, closed=False, n_ring=16):
    """Closed tube mesh around a centerline in the xz plane (end caps when
    open), oriented so the inside reads w = +1."""
    C = np.asarray(center_line, float)
    n = len(C)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.array([0, 1.0, 0])
    N = np.cross(B, T)
    ph = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    V = (C[:, None] + r * (np.cos(ph)[None, :, None] * N[:, None]
                           + np.sin(ph)[None, :, None] * B)).reshape(-1, 3)
    F = []
    for i in range(n if closed else n - 1):
        i2 = (i + 1) % n
        for j in range(n_ring):
            j2 = (j + 1) % n_ring
            a, b, c, d = i * n_ring + j, i * n_ring + j2, i2 * n_ring + j, i2 * n_ring + j2
            F += [[a, b, d], [a, d, c]]
    if not closed:
        c0, c1, last = len(V), len(V) + 1, (n - 1) * n_ring
        V = np.vstack([V, C[0], C[-1]])
        F += [[c0, (j + 1) % n_ring, j] for j in range(n_ring)]
        F += [[c1, last + j, last + (j + 1) % n_ring] for j in range(n_ring)]
    F = np.asarray(F)
    if pg.winding_number(C[n // 2], V, F)[0] < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def _arc(rx, rz, deg, n=60):
    s = np.linspace(-np.radians(deg) / 2, np.radians(deg) / 2, n)
    return np.column_stack([rx * np.cos(s), np.zeros(n), rz * np.sin(s)])


def _ground_one(V, F, part="handle"):
    from manip_sim.proposal import surface_samples
    P, _ = surface_samples(V, F, 4000, np.random.default_rng(0))
    g = pg.fit_part(part, P, UP, np.array([-0.1, 0.0, 0.0]))
    doc = pg.symbols_from_parts("obj", {part: g}, UP, "test")
    return g, pg.annotate_depth_checks(doc, V, F)["points"]


@pytest.mark.parametrize("curve, r, closed, primitive, check", [
    # straight spout: center on the centerline -> inside the tube
    (np.linspace([0, 0, 0], [0.08, 0, 0.03], 40), 0.006, False, "line", "interior"),
    # 180-deg C handle, taller than wide: still a LINE fit, but the center
    # (PCA axis through the centroid) sits ~8 mm inside the curve, outside
    # the 6 mm tube -- the case a primitive-keyed label gets wrong
    (_arc(0.022, 0.028, 180), 0.006, False, "line", "skip"),
    # the same primitive with its center ON the material: a narrow, tall C
    (_arc(0.018, 0.035, 180), 0.006, False, "line", "interior"),
    # round 180-deg C: blob fit, centroid in the handle's opening
    (_arc(0.025, 0.025, 180), 0.006, False, "blob", "skip"),
    # closed thin ring (rim): center is the free-space opening
    (_arc(0.04, 0.04, 360, n=96)[:-1], 0.002, True, "ring", "skip"),
])
def test_depth_check_is_geometry_not_primitive(curve, r, closed, primitive, check):
    V, F = _tube(curve, r, closed)
    g, pts = _ground_one(V, F)
    assert g.primitive == primitive
    assert pts["handle_center"]["depth_check"] == check
    w = pts["handle_center"]["winding"]
    assert (w > 0.9) if check == "interior" else (abs(w) < 0.1)    # saturated, not marginal
    if primitive == "line":
        assert pts["handle_tip"]["depth_check"] in ("interior", "skip")


def test_winding_saturates_off_the_surface():
    """2 mm either side of a face already reads ~1 / ~0, so the label is
    only arbitrary AT the surface, where the value is undefined (0 or 1 by
    a floating-point sign). That a point at the surface passes the
    consumer's gate under either label is pinned on the hardware side,
    where that gate lives (check_symbols' interior band is [-5, +30] mm)."""
    V, F = _box_mesh([-0.05, -0.05, 0], [0.05, 0.05, 0.1])
    on, inside, outside = [0.01, 0.02, 0.1], [0.01, 0.02, 0.098], [0.01, 0.02, 0.102]
    w = pg.winding_number(np.array([on, inside, outside]), V, F)
    assert min(abs(w[0]), abs(w[0] - 1.0)) < 1e-6         # either side, never in between
    assert w[1] > 0.95 and abs(w[2]) < 0.05
    assert abs(pg.winding_number([0, 0, 0.05], V, F)[0] - 1.0) < 1e-9
    assert abs(pg.winding_number([0.3, 0, 0.05], V, F)[0]) < 1e-9


def test_winding_sign_follows_orientation_label_does_not():
    V, F = _box_mesh([-0.05, -0.05, 0], [0.05, 0.05, 0.1])
    Fi = F[:, [0, 2, 1]]                                   # inward-wound copy
    assert pg.winding_number([0, 0, 0.05], V, Fi)[0] == pytest.approx(-1.0)
    doc = {"points": {"c": {"xyz": [0, 0, 0.05]}}}
    assert pg.annotate_depth_checks(doc, V, Fi)["points"]["c"]["depth_check"] == "interior"


def test_annotate_only_touches_points_and_is_loadable(tmp_path):
    from manip_sim.frames import load_symbols
    V, F = _tube(_arc(0.022, 0.028, 180), 0.006)
    g, _ = _ground_one(V, F)
    doc = pg.symbols_from_parts("obj", {"handle": g}, UP, "test", up_sigma_deg=2.0)
    before = {k: dict(v) for k, v in doc["axes"].items()}, dict(doc["quantities"])
    pg.annotate_depth_checks(doc, V, F)
    assert ({k: dict(v) for k, v in doc["axes"].items()}, dict(doc["quantities"])) == before
    assert all({"depth_check", "winding"} <= set(p) for p in doc["points"].values())
    assert doc["provenance"]["depth_semantics"]["inconsistent_edges"] == 0   # _tube is clean
    assert doc["provenance"]["depth_semantics"]["boundary_edges"] == 0
    empty = pg.annotate_depth_checks({"points": {}}, V, F)
    assert empty["points"] == {} and "depth_semantics" in empty["provenance"]
    (tmp_path / "frames.json").write_text(__import__("json").dumps(doc))
    sym = load_symbols(tmp_path)                           # unknown keys ignored downstream
    assert set(sym.points) == set(doc["points"])


def test_mesh_orientation_counts_flips_and_holes():
    V, F = _box_mesh([-0.05, -0.05, 0], [0.05, 0.05, 0.1])
    clean = {"inconsistent_edges": 0, "flipped_edges": 0, "pinch_edges": 0,
             "boundary_edges": 0, "non_manifold_edges": 0,
             "merged_vertices": 8, "vertices": 8}
    assert pg.mesh_orientation(V, F) == clean
    Ff = F.copy()
    Ff[0] = Ff[0, [0, 2, 1]]                                # one flipped face
    assert pg.mesh_orientation(V, Ff)["inconsistent_edges"] == 3
    assert pg.mesh_orientation(V, Ff)["flipped_edges"] == 3
    assert pg.mesh_orientation(V, F[1:])["boundary_edges"] == 3
    # a flipped face moves the winding number off 1 inside: what the count warns about
    assert abs(pg.winding_number([0, 0, 0.05], V, Ff)[0] - 1.0) > 0.05


def test_mesh_orientation_tells_a_pinch_from_a_flip():
    """Two consistently wound boxes touching along one edge: four faces on
    it, two traversals each way -- 2 'inconsistent' directed edges, but no
    flip, and the winding number is exact inside both. Flip a face of one
    box and only then does flipped_edges count."""
    A, FA = _box_mesh([0, 0, 0], [0.1, 0.1, 0.1])
    B, FB = _box_mesh([0.1, 0.1, 0], [0.2, 0.2, 0.1])
    V, F = np.vstack([A, B]), np.vstack([FA, FB + len(A)])
    m = pg.mesh_orientation(V, F)
    assert (m["inconsistent_edges"], m["non_manifold_edges"]) == (2, 1)
    assert (m["flipped_edges"], m["pinch_edges"]) == (0, 1)
    assert pg.winding_number([[0.05, 0.05, 0.05], [0.15, 0.15, 0.05]], V, F) == \
        pytest.approx([1.0, 1.0])
    Ff = F.copy()
    Ff[0] = Ff[0, [0, 2, 1]]
    mf = pg.mesh_orientation(V, Ff)
    assert mf["flipped_edges"] >= 1 and mf["pinch_edges"] <= 1


def _soup(V, F):
    """Every face gets its own vertex copies: the extreme of a UV-seam
    split, where by index no two faces share an edge."""
    return V[F].reshape(-1, 3), np.arange(3 * len(F)).reshape(-1, 3)


def test_mesh_orientation_sees_across_seams():
    """By index, a seam-split mesh looks like all boundary, and a flipped
    face is invisible; merged by position, it is a closed box again and the
    flip is counted. The winding number never needed the merge."""
    V, F = _box_mesh([-0.05, -0.05, 0], [0.05, 0.05, 0.1])
    Vs, Fs = _soup(V, F)
    m = pg.mesh_orientation(Vs, Fs)
    assert m["boundary_edges"] == 0 and m["inconsistent_edges"] == 0
    assert m["merged_vertices"] == 8 and m["vertices"] == 36
    Ff = F.copy()
    Ff[0] = Ff[0, [0, 2, 1]]
    Vsf, Fsf = _soup(V, Ff)
    E = np.concatenate([Fsf[:, [0, 1]], Fsf[:, [1, 2]], Fsf[:, [2, 0]]])
    assert len(np.unique(E, axis=0)) == len(E)              # by index: no flip visible
    assert pg.mesh_orientation(Vsf, Fsf)["inconsistent_edges"] == 3   # merged: flip counted
    assert pg.winding_number([0, 0, 0.05], Vs, Fs)[0] == pytest.approx(1.0)
