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
