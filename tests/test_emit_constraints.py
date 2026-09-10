"""scripts/emit_constraints.py: the framed arm's inputs come from the
render_stage_frames manifest and never degrade silently."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.emit_constraints import STAGES, _framed_inputs


def _manifest(tmp_path: Path) -> dict:
    roles = {}
    for stage, role in STAGES:
        d = tmp_path / role
        d.mkdir(parents=True)
        views = {}
        for v in ("iso", "iso-opp", "top"):
            (d / f"{v}.png").write_bytes(b"png")
            views[v] = str(d / f"{v}.png")
        roles[role] = {"stage": stage.index, "name": stage.name,
                       "active": stage.active, "passive": stage.passive,
                       "w": {"object": stage.passive or stage.active},
                       "triads": [], "views": views}
    return {"views": ["iso", "iso-opp", "top"], "roles": roles}


def test_framed_inputs_returns_views_in_manifest_order_with_the_record(tmp_path):
    m = _manifest(tmp_path)
    paths, rec = _framed_inputs("pour", m)
    assert [p.name for p in paths] == ["iso.png", "iso-opp.png", "top.png"]
    assert all(p.exists() for p in paths)
    assert rec is m["roles"]["pour"]


def test_framed_inputs_refuses_a_missing_role_or_view(tmp_path):
    m = _manifest(tmp_path)
    del m["roles"]["pour"]
    with pytest.raises(SystemExit, match="role 'pour' absent"):
        _framed_inputs("pour", m)
    m = _manifest(tmp_path / "b")
    Path(m["roles"]["grasp"]["views"]["top"]).unlink()
    with pytest.raises(SystemExit, match="views missing on disk"):
        _framed_inputs("grasp", m)


def test_stage_table_roles_key_selections_and_frames_alike():
    assert [r for _, r in STAGES] == ["grasp", "transport_active", "pour"]


# ---------------------------------------------------------- framed-chained

def _canned(stage):
    """Grounding emissions (ablation-modality-v3 framed.3's rows): transport
    upright with a free path, pour hinge + spout-down with a path that
    nests transport's subgoal band."""
    docs = {
        "grasp": {"stage": 1, "name": "grasp", "active": "teapot", "passive": None,
                  "path_tsr": {"rot": "free", "trans": "free"},
                  "subgoal_tsr": {"rot": "free", "trans": [
                      {"term": "inside", "anchor": "teapot.handle_center", "slack": "snug"}]},
                  "verify": ""},
        "transport": {"stage": 2, "name": "transport", "active": "teapot", "passive": "mug",
                      "path_tsr": {"rot": [{"axis": "teapot.+up", "points": "world.z", "tol": "moderate"}],
                                   "trans": "free"},
                      "subgoal_tsr": {"rot": [{"axis": "teapot.+up", "points": "world.z", "tol": "moderate"}],
                                      "trans": [
                          {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"},
                          {"term": "above", "anchor": "mug.opening_center", "clearance": "medium", "slack": "moderate"}]},
                      "verify": ""},
        "pour": {"stage": 3, "name": "pour", "active": "teapot", "passive": "mug",
                 "path_tsr": {"rot": [{"axis": "teapot.+left", "perpendicular_to": "mug.+up", "tol": "moderate"}],
                              "trans": [
                     {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"},
                     {"term": "above", "anchor": "mug.opening_center", "clearance": "small", "slack": "loose"}]},
                 "subgoal_tsr": {"rot": [{"axis": "teapot.+front", "points": "mug.-up", "tol": "moderate"}],
                                 "trans": [
                     {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"},
                     {"term": "above", "anchor": "mug.opening_center", "clearance": "small", "slack": "snug"}]},
                 "verify": ""},
    }
    return json.dumps(docs[stage.name])


def test_render_chained_renders_each_stage_after_the_previous_gate_at_its_nominal(
        tmp_path, monkeypatch):
    """The ordering dependency that interleaving exists for: stage 3's
    render happens after stage 2's gate passed, with the teapot at stage
    2's subgoal nominal; stages 1 and 2 render at spawn. The fake client
    also sees the entry line agreeing with the render's entry_from."""
    import sys
    import numpy as np
    import scripts.render_stage_frames as rsf
    from manip_sim.vlm import CallLog, Vocabulary, parse_emission
    from scripts import emit_constraints as ec
    from test_selection import _HAVE_POOLS
    if not _HAVE_POOLS or not Path("outputs/selections/pour_tea.json").exists():
        pytest.skip("needs the pour_tea selections artifact + candidate pools")

    events: list[tuple] = []          # ("render", role, entry_from, teapot pose) / ("emit", name, entry_line)
    gate_passed: list[str] = []

    def fake_render_role(scene, role, sf, poses, out_dir, entry_from, cams=None, base=None):
        events.append(("render", role, entry_from, np.asarray(poses["teapot"]).copy(),
                       list(gate_passed)))
        d = Path(out_dir) / role
        d.mkdir(parents=True, exist_ok=True)
        views = {}
        for v in rsf.VIEWS:
            p = d / f"{v}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")          # exists, non-empty
            views[v] = str(p)
        return {"stage": sf.stage, "name": sf.name, "active": sf.active,
                "passive": sf.passive, "w": {"object": sf.w_object,
                "point_body": sf.w_point_body, "point": "mug.opening_center",
                "candidate_id": sf.w_candidate_id, "x_route": sf.w_x_route,
                "fallback": sf.w_fallback, "merged": sf.merged},
                "triads": sf.triads, "end_on": {}, "views": views,
                "poses": {n: np.asarray(T).tolist() for n, T in poses.items()},
                "entry_from": entry_from}
    monkeypatch.setattr(rsf, "render_role", fake_render_role)
    # extent needs converted visual meshes; not what this test is about
    monkeypatch.setattr(rsf, "scene_extent",
                        lambda scene, poses: (np.zeros(3), 0.33))

    class FakeClient:
        def __init__(self):
            self.logs = []
        def emit_constraints(self, stage, vocab, selection=None, view_paths=None,
                             rejections=None, frames=None, entry=None):
            events.append(("emit", stage.name, entry, len(view_paths or []),
                           frames is not None))
            raw = _canned(stage)
            self.logs.append(CallLog("emit_constraints", attempts=1, raw=raw,
                                     views=len(view_paths or []),
                                     legend=frames is not None, entry=entry is not None))
            return parse_emission(raw, vocab)
    monkeypatch.setattr(ec, "Client", FakeClient)

    # observe gate passes by wrapping compile_stage (the loop chains poses after it)
    real_compile = ec.compile_stage
    def spy_compile(em, *a, **k):
        cs = real_compile(em, *a, **k)
        gate_passed.append(em.name)
        return cs
    monkeypatch.setattr(ec, "compile_stage", spy_compile)

    out = tmp_path / "chained.json"
    monkeypatch.setattr(sys, "argv", [
        "emit_constraints.py", "--selections", "outputs/selections/pour_tea.json",
        "--render-chained", str(tmp_path / "frames"), "--out", str(out)])
    ec.main()

    renders = [e for e in events if e[0] == "render"]
    emits = [e for e in events if e[0] == "emit"]
    assert [r[1] for r in renders] == ["grasp", "transport_active", "pour"]
    assert [r[2] for r in renders] == ["spawn", "spawn", "transport.subgoal.nominal"]
    # stage 3 rendered only after stage 2 compiled (its pose is stage 2's nominal)
    assert renders[2][4] == ["grasp", "transport"]
    assert renders[1][4] == ["grasp"]
    spawn_T = renders[0][3]
    np.testing.assert_allclose(renders[1][3], spawn_T)
    assert np.linalg.norm(renders[2][3][:3, 3] - spawn_T[:3, 3]) > 0.1   # moved over the mug
    # every stage was emitted framed (3 views, legend), with the entry line
    assert all(e[3] == 3 and e[4] for e in emits)
    assert "at its spawn pose" in emits[1][2]
    assert "at the previous stage's goal center" in emits[2][2]
    # ordering inside the stream: render precedes emit for each stage, and the
    # pour render comes after the transport emit
    order = [(e[0], e[1]) for e in events]
    assert order.index(("render", "pour")) > order.index(("emit", "transport"))
    art = json.loads(out.read_text())
    assert art["arm"] == "framed-chained" and art["stages_reached"] == 3
    assert all(c["grounded"] for c in art["compiled"])
    # the chained pose the render used equals the entry pose the gate recorded
    np.testing.assert_allclose(renders[2][3], np.array(art["compiled"][2]["entry_poses"]["teapot"]),
                               atol=1e-6)
