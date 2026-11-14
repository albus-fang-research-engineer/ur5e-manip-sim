"""manip_sim.repair_loop: the offline parts (role indices, row text, stage
images, captions, report splicing). Rendering and compiling run in the
dry-run smoke check, not here."""

from __future__ import annotations

from pathlib import Path

from manip_sim import fault_injection as fi
from manip_sim import repair_loop as rl
from tests.test_critic import CARRY, POUR, _state
from tests.test_fault_injection import _spec


def test_role_index_pairs_passive_with_active():
    idx = rl.role_index_for(_spec())
    assert idx == {"grasp": 0, "transport_active": 1, "pour": 2, "transport_passive": 1}
    assert rl.role_index_for(_spec(), {"grasp": 0, "insert": 3}) == {"grasp": 0, "insert": 3}


def test_stage_rows_name_every_row_by_pointer():
    rows = rl.stage_rows(_spec().emissions["transport_active"])
    assert rows[0].startswith(f"{CARRY}.path.rot[0]: teapot.up_axis parallel world.z")
    assert any(r.startswith(f"{CARRY}.subgoal.trans[0]: above") for r in rows)


def test_stage_images_pick_one_stage_without_captions(tmp_path):
    man = {"images": [{"kind": "state", "state": "initial", "file": "a.png"},
                      {"kind": "state", "state": "goal:pour", "role": "pour", "file": "b.png"},
                      {"kind": "path", "stage": "pour", "file": "c.png"},
                      {"kind": "path", "stage": "transport_active", "file": "d.png"}]}
    got = rl.stage_images(man, tmp_path, "pour")
    assert [p.name for p in got] == ["b.png", "c.png"]


def test_caption_for_goal_without_free_pose_says_so():
    c = rl.ChainStage("pour", _spec().emissions["pour"], {})
    img = {"kind": "state", "state": "goal:pour", "role": "pour",
           "collisions": [{"a": "teapot", "b": "mug", "depth_mm": 8.0}],
           "search": {"nominal_collides": True, "shown": "none_free"},
           "notes": ["no collision-free pose found in the region (128 samples): "
                     "teapot collides with mug"]}
    cap = rl.caption_for(img, {}, 2, 3, [c], {"teapot": "white"})
    assert f'stage "{POUR}" (role pour)' in cap
    assert "no collision-free pose found" in cap and "Image 2 of 3" in cap


def test_uncompiled_stage_is_spliced_into_the_report():
    st = _state()
    chain = [rl.ChainStage("transport_active", st.emissions["transport_active"], {},
                           error="compile rejected: x")]
    pv = [type("P", (), {"editable": True})()]
    rep = rl.preview_report(st, chain, pv)
    rec = rep["stages"][0]
    assert rec["compile_error"] == "compile rejected: x"
    assert rec["subgoal"]["rows"][0]["pointer"] == f"{CARRY}.subgoal.rot[0]"


def test_world_axis_tokens_keep_their_axis():
    assert rl._dir_token("world.z") == "world.z"
    assert rl._dir_token("world.z", "-") == "world.-z"
    assert rl._dir_token("world.x", "-") == "world.-x"
    assert rl._dir_token("teapot.lateral_axis") == "teapot.+left"


def test_spec_state_round_trip_from_a_fault_spec(tmp_path):
    spec, _ = fi.inject(_spec(), fi.Fault(
        id="f", task="pour_tea", category="clearance", role="transport_active",
        op="set_field", ctx="subgoal", table="trans", match={"term": "above"},
        field="clearance", value="contact"))
    assert spec.roles == ("grasp", "transport_active", "pour")
    assert Path(rl.__file__).name == "repair_loop.py"
