"""manip_sim.fault_injection: one-edit faults for the repair study. Offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manip_sim import fault_injection as fi
from manip_sim.critic import row_text
from tests.test_critic import CARRY, POOLS, POUR, _emissions
from manip_sim.vlm import PointAxisSelection

CATALOG = Path(__file__).resolve().parents[1] / "tasks/repair_study/faults.json"


def _spec() -> fi.Spec:
    sels = {"grasp": PointAxisSelection(2, None, "+", None, ""),
            "transport_active": PointAxisSelection(4, None, "+", None, ""),   # spout
            "pour": PointAxisSelection(4, None, "+", None, ""),
            "transport_passive": PointAxisSelection(1, None, "+", None, "")}
    objs = {"grasp": "teapot", "transport_active": "teapot", "pour": "teapot",
            "transport_passive": "mug"}
    ems = _emissions()
    return fi.Spec(sels, objs, ems, ("grasp", "transport_active", "pour"),
                   {"task": "pour tea", "compiled": [{"stage": 0, "grounded": True}]})


def _fault(**kw) -> fi.Fault:
    base = dict(id="f", task="pour_tea", category="clearance", role="transport_active")
    return fi.Fault(**{**base, **kw})


def test_set_field_by_match_records_pointer_and_texts():
    f = _fault(op="set_field", ctx="subgoal", table="trans", match={"term": "above"},
               field="clearance", value="contact")
    spec, rec = fi.inject(_spec(), f)
    row = spec.emissions["transport_active"].subgoal_tsr.trans[0]
    assert row.clearance == "contact"
    assert rec.pointer == f"{CARRY}.subgoal.trans[0]"
    assert "clearance small" in rec.before and "clearance contact" in rec.after
    # nothing else changed
    diff = fi.spec_diff(_spec(), spec)
    assert [d["pointer"] for d in diff] == [rec.pointer]


def test_set_point_maps_symbol_to_candidate():
    f = _fault(op="set_point", category="wrong_anchor", point="handle_center",
               unfixable_by=("render_check",))
    spec, rec = fi.inject(_spec(), f, POOLS)
    assert spec.selections["transport_active"].candidate_id == 2
    assert rec.pointer == f"{CARRY}.point.teapot"
    assert (rec.before, rec.after) == ("teapot.spout_center", "teapot.handle_center")
    assert rec.unfixable_by == ("render_check",)


def test_add_row_appends_and_points_at_the_new_index():
    f = _fault(op="add_row", category="contradictory_rows", role="pour", ctx="subgoal",
               table="rot", row={"axis": "teapot.up_axis", "relation": "parallel",
                                 "reference": "world.z", "tol": "tight"})
    spec, rec = fi.inject(_spec(), f)
    rot = spec.emissions["pour"].subgoal_tsr.rot
    assert len(rot) == 2 and rec.pointer == f"{POUR}.subgoal.rot[1]"
    assert rec.after == row_text(rot[1])


@pytest.mark.parametrize("kw, why", [
    (dict(role="nope", op="set_point", point="handle_center"), "no emission"),
    (dict(op="set_field", ctx="subgoal", table="trans", match={"term": "below"},
          field="clearance", value="contact"), "matching"),
    (dict(op="set_field", ctx="subgoal", table="trans", match={"term": "above"},
          field="clearance", value="small"), "already"),
    (dict(op="set_field", ctx="subgoal", table="trans", match={"term": "centered"},
          field="clearance", value="large"), "does not use"),
    (dict(op="set_point", point="spout_center"), "already selected"),
    (dict(op="set_point", point="no_such_point"), "no candidate"),
    (dict(role="pour", op="add_row", ctx="subgoal", table="trans",
          row={"term": "above", "anchor": "mug.rim_center", "clearance": "small",
               "slack": "snug"}), "single free term"),
])
def test_not_applicable_is_loud(kw, why):
    with pytest.raises(fi.FaultNotApplicable, match=why):
        fi.inject(_spec(), _fault(**kw), POOLS)


def test_add_row_beside_a_free_rot_table_is_refused():
    f = _fault(op="add_row", category="contradictory_rows", role="grasp", ctx="path",
               table="rot", row={"axis": "teapot.up_axis", "relation": "parallel",
                                 "reference": "world.z", "tol": "tight"})
    spec = _spec()
    free = tuple(fi.RotRow(None, "free", None, None, r) for r in ("roll", "pitch", "yaw"))
    em = spec.emissions["grasp"]
    spec = fi.replace(spec, emissions={**spec.emissions, "grasp": fi.replace(
        em, path_tsr=fi.replace(em.path_tsr, rot=free))})
    with pytest.raises(fi.FaultNotApplicable, match="declared free"):
        fi.inject(spec, f)


def test_bad_values_are_catalog_errors_not_skips():
    with pytest.raises(ValueError):
        _fault(op="set_field", category="nonsense")
    f = _fault(op="set_field", ctx="subgoal", table="trans", match={"term": "above"},
               field="clearance", value="huge")
    with pytest.raises(ValueError, match="not in"):
        fi.inject(_spec(), f)


def test_write_and_load_roundtrip_drops_stale_compile_gate(tmp_path):
    f = _fault(op="set_field", ctx="subgoal", table="trans", match={"term": "above"},
               field="clearance", value="contact")
    spec, _ = fi.inject(_spec(), f)
    ps, pe = fi.write_spec(spec, tmp_path)
    doc = json.loads(pe.read_text())
    assert "compiled" not in doc and doc["task"] == "pour tea"
    back = fi.load_spec(ps, pe)
    assert back.emissions == spec.emissions and back.selections == spec.selections
    assert back.objects == spec.objects and back.roles == spec.roles


def test_catalog_is_valid_and_ids_unique():
    faults = fi.load_catalog(CATALOG)
    assert {f.task for f in faults} == {"pour_tea", "insert_marker"}
    assert {f.category for f in faults} == set(fi.CATEGORIES)
    for f in faults:
        assert f.expect, f.id
        if f.category == "wrong_anchor":
            assert f.unfixable_by == ("render_check",)
