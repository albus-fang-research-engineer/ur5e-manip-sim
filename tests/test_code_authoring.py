"""manip_sim/code_authoring.py: the code media of call #3 (A3 constrained
code, A4 free code). Offline: real pour-tea sidecars, a fake transport, no
network, no simulator."""

import json

import numpy as np
import pytest

from manip_sim import code_authoring as ca
from manip_sim.compile_tsr import ROT_TOL_RAD, check_pair_consistency, compile_stage
from manip_sim.frames import load_symbols
from manip_sim.tsr import make_pose
from manip_sim.vlm import Client, ParseRejection, StageSpec, Vocabulary, parse_emission

SYMBOLS = {"teapot": load_symbols("assets/objects/teapot"),
           "mug": load_symbols("assets/objects/mug")}
VOCAB = Vocabulary.from_asset_dirs({"teapot": "assets/objects/teapot",
                                    "mug": "assets/objects/mug"})
POSES = {"teapot": make_pose((0.3, 0.1, 0.05)),
         "mug": make_pose((0.55, -0.2, 0.02))}
TIP = SYMBOLS["teapot"].points["spout_tip"]
OPENING = SYMBOLS["mug"].points["opening_center"]
STAGE = StageSpec(index=2, name="transport", active="teapot", passive="mug",
                  parts={"teapot": ("spout",), "mug": ("rim",)})

UPRIGHT = {"axis": "teapot.+up", "points": "world.z", "tol": "moderate"}
TRANSPORT = {   # schema emission: upright, spout 3-8 cm over the opening
    "stage": 2, "name": "transport", "active": "teapot", "passive": "mug",
    "path_tsr": {"rot": [UPRIGHT], "trans": "free"},
    "subgoal_tsr": {"rot": [UPRIGHT], "trans": [
        {"term": "above", "anchor": "mug.opening_center", "clearance": "medium",
         "slack": "moderate"},
        {"term": "centered", "anchor": "mug.opening_center", "tol": "moderate"}]},
    "verify": ""}

A3_TRANSPORT = '''# goal: the spout tip hovers 3-8 cm over the mug opening, pot upright
def author():
    o = point("mug.opening_center")
    up = [points_to("teapot.+up", "world.z", 15.0)]
    return {"path": tsr(rot=up),
            "subgoal": tsr(x=(o[0] - 0.02, o[0] + 0.02), y=(o[1] - 0.02, o[1] + 0.02),
                           z=(o[2] + 0.03, o[2] + 0.08), rot=up)}
'''


def _code(medium, src, name="transport"):
    return ca.CodeEmission(stage=2, name=name, active="teapot", passive="mug",
                           medium=medium, source=src, goal="g")


def _schema():
    return compile_stage(parse_emission(json.dumps(TRANSPORT), VOCAB), SYMBOLS,
                         POSES, w_point=OPENING, e_point=TIP)


# ------------------------------------------------------------ source checks

@pytest.mark.parametrize("src, why", [
    ("import os\ndef author():\n    pass\n", "Import"),
    ("def author():\n    return np.load('x')\n", "'load'"),
    ("def author():\n    return open('x')\n", "'open'"),
    ("def author():\n    return (1).__class__\n", "underscore"),
    ("def author():\n    return __import__('os')\n", "__import__"),
    ("def other():\n    pass\n", "top-level function `author`"),
    ("def author(:\n", "syntax error"),
])
def test_check_source_rejects_with_a_reason(src, why):
    with pytest.raises(ParseRejection, match=why):
        ca.check_source(src, "author")


def test_parse_takes_one_block_and_its_goal_line():
    raw = "Here:\n```python\n" + A3_TRANSPORT + "```\n"
    em = ca.parse_code_response(raw, STAGE, "constrained-code")
    assert em.goal.startswith("the spout tip hovers")
    assert em.source == A3_TRANSPORT and em.medium == "constrained-code"
    with pytest.raises(ParseRejection, match="exactly one"):
        ca.parse_code_response(raw + raw, STAGE, "constrained-code")
    with pytest.raises(ParseRejection, match="# goal:"):
        ca.parse_code_response("```python\ndef author():\n    pass\n```", STAGE,
                               "constrained-code")


# --------------------------------------------------------- constrained-code

def test_a3_reproduces_the_schema_pair_through_the_same_compiler():
    ref = _schema()
    cs = ca.compile_any(_code("constrained-code", A3_TRANSPORT), SYMBOLS, POSES,
                        w_point=OPENING, e_point=TIP, vocab=VOCAB)
    for k in ("path", "subgoal"):
        a, b = getattr(cs, k), getattr(ref, k)
        assert np.allclose(a.Bw, b.Bw, atol=1e-6), k
        assert np.allclose(a.T0_w, b.T0_w) and np.allclose(a.Tw_e, b.Tw_e)
    check_pair_consistency(cs)


def test_a3_numeric_tolerance_keys_do_not_leak_into_the_frozen_table():
    before = dict(ROT_TOL_RAD)
    src = A3_TRANSPORT.replace("15.0", "7.5")
    cs = ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES,
                        w_point=OPENING, e_point=TIP, vocab=VOCAB)
    assert ROT_TOL_RAD == before
    assert np.isclose(cs.subgoal.Bw[3, 1], np.deg2rad(7.5))


def test_a3_compile_errors_name_the_models_rows():
    bad = A3_TRANSPORT.replace('"subgoal": tsr(x=', '"subgoal": tsr(x=None, xx=')
    with pytest.raises(ca.CodeError, match="unexpected keyword"):
        ca.compile_any(_code("constrained-code", bad), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)
    # a path z band the entry pose is far outside of: blamed on "path.z"
    far = A3_TRANSPORT.replace("tsr(rot=up)", "tsr(z=(o[2] + 0.5, o[2] + 0.6), rot=up)")
    with pytest.raises(ca.CodeError) as e:
        ca.compile_any(_code("constrained-code", far), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)
    assert any(x.slot == "transport.path.z" for x in e.value.all())


def test_a3_subgoal_rows_must_be_closed():
    half = A3_TRANSPORT.replace("z=(o[2] + 0.03, o[2] + 0.08)", "z=(o[2] + 0.03, None)")
    with pytest.raises(ca.CodeError, match="half-open"):
        ca.compile_any(_code("constrained-code", half), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


def test_a3_rotation_rows_use_the_schema_direction_alphabet():
    bad = A3_TRANSPORT.replace('"teapot.+up", "world.z"', '"teapot.spout_axis", "world.z"')
    with pytest.raises(ca.CodeError, match="spout_axis"):
        ca.compile_any(_code("constrained-code", bad), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


# --------------------------------------------------------------- free-code

def _literal_a4(ref) -> str:
    def arr(a):
        return json.dumps(np.where(np.isinf(a), np.sign(a) * 1e9, a).tolist())
    body = ", ".join(f'"{k}": {{"T0_w": np.array({arr(getattr(ref, k).T0_w)}), '
                     f'"Tw_e": np.array({arr(getattr(ref, k).Tw_e)}), '
                     f'"Bw": np.array({arr(getattr(ref, k).Bw)})}}'
                     for k in ("path", "subgoal"))
    return f"# goal: same as the schema\ndef author(scene):\n    return {{{body}}}\n"


def test_a4_accepts_numeric_tsrs_and_runs_the_same_feasibility_checks():
    ref = _schema()
    cs = ca.compile_any(_code("free-code", _literal_a4(ref)), SYMBOLS, POSES,
                        w_point=OPENING, e_point=TIP)
    assert np.allclose(cs.subgoal.Bw, ref.subgoal.Bw)
    check_pair_consistency(cs)
    # the same TSR with a path that excludes the entry pose fails, typed
    far = POSES | {"teapot": make_pose((0.3, 0.1, 0.95))}
    src = _literal_a4(ref).replace("1000000000.0", "0.2")   # bound the path z
    with pytest.raises(ca.CodeError, match="entry pose lies outside"):
        ca.compile_any(_code("free-code", src), SYMBOLS, far, w_point=OPENING,
                       e_point=TIP)


def test_a4_reads_the_scene_at_the_pose_it_is_run_at():
    src = '''# goal: anywhere within 5 cm of the default w, any attitude
def author(scene):
    Tw_e = np.linalg.inv(scene.w_default) @ scene.T_active   # entry is zero displacement
    free = [[-np.inf, np.inf]] * 3 + [[-np.pi, np.pi]] * 3
    box = [[-0.05, 0.05]] * 3 + [[-np.pi, np.pi]] * 3
    d = Tw_e[:3, 3]
    box = [[d[i] - 0.05, d[i] + 0.05] for i in range(3)] + [[-np.pi, np.pi]] * 3
    return {"path": {"T0_w": scene.w_default, "Tw_e": np.eye(4), "Bw": free},
            "subgoal": {"T0_w": scene.w_default, "Tw_e": np.eye(4), "Bw": box}}
'''
    a = ca.compile_any(_code("free-code", src), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)
    moved = POSES | {"teapot": make_pose((0.1, 0.0, 0.1))}
    b = ca.compile_any(_code("free-code", src), SYMBOLS, moved, w_point=OPENING, e_point=TIP)
    assert not np.allclose(a.subgoal.Bw, b.subgoal.Bw)


@pytest.mark.parametrize("subgoal_tw_e, subgoal_bw, why", [
    ("np.diag([2.0, 1.0, 1.0, 1.0])", "free", "not a rotation"),
    ("np.eye(4)", "[[0, 1]]", "6x2"),
    ("np.eye(4)", "[[0, 1]] * 3 + [[0.5, -0.5]] * 3", "hi < lo"),
])
def test_a4_shape_and_rotation_errors_are_typed(subgoal_tw_e, subgoal_bw, why):
    src = f'''# goal: x
def author(scene):
    free = [[-np.inf, np.inf]] * 3 + [[-np.pi, np.pi]] * 3
    return {{"path": {{"T0_w": scene.w_default, "Tw_e": np.eye(4), "Bw": free}},
            "subgoal": {{"T0_w": scene.w_default, "Tw_e": {subgoal_tw_e}, "Bw": {subgoal_bw}}}}}
'''
    with pytest.raises(ca.CodeError, match=why):
        ca.compile_any(_code("free-code", src), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)


# ------------------------------------------------------------------ sandbox

def test_runtime_errors_carry_the_models_line_number():
    src = "# goal: x\ndef author():\n    a = 1\n    return quantity('mug.nope')\n"
    with pytest.raises(ca.CodeError, match=r"KeyError.*line 4"):
        ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


def test_runaway_code_is_stopped(monkeypatch):
    monkeypatch.setattr(ca, "CHILD_CPU_S", 1)
    monkeypatch.setattr(ca, "CHILD_WALL_S", 10)
    src = "# goal: x\ndef author():\n    while True:\n        pass\n"
    with pytest.raises(ca.CodeError, match="CPU limit|did not finish"):
        ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


def test_the_child_cannot_write_files(tmp_path):
    src = ("# goal: x\ndef author():\n    np.array([1.0]).tofile('x')\n")
    with pytest.raises(ParseRejection, match="tofile"):
        ca.check_source(src, "author")


# ----------------------------------------------------------- prompt + client

def test_emit_code_goes_through_the_client_and_logs_the_call():
    seen = []

    def transport(payload):
        seen.append(payload)
        return "```python\n" + A3_TRANSPORT + "```"

    client = Client(transport=transport, model="fake", max_tokens=ca.MAX_TOKENS)
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    em = ca.emit_code(client, STAGE, VOCAB, "constrained-code", facts,
                      entry="At entry: x.", rejections=[("old", "bad z")])
    assert em.source == A3_TRANSPORT
    assert client.logs[-1].touchpoint == "emit_code" and client.logs[-1].entry
    msgs = seen[0]["messages"]
    assert "rejected: bad z" in msgs[-1]["content"][0]["text"]
    user = msgs[0]["content"][0]["text"]
    assert "point('mug.opening_center')" in user and "At entry: x." in user
    assert "points_to" in seen[0]["system"] and "T0_w" not in seen[0]["system"]
    s4, _ = ca.build_code_prompt(STAGE, VOCAB, "free-code", facts)
    assert "Tw_e" in s4 and "points_to" not in s4


def test_artifacts_round_trip_and_refuse_a_failed_gate(tmp_path):
    em = _code("free-code", "# goal: g\ndef author(scene):\n    pass\n")
    doc = {"roles": ["transport_active"], "emissions": [ca.as_record(em)],
           "compiled": [{"stage": "transport", "grounded": True}]}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(doc))
    assert ca.load_any_emissions(p) == {"transport_active": em}
    doc["compiled"][0]["grounded"] = False
    p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="gate failed"):
        ca.load_any_emissions(p)


# ------------------------------------------------- code-review regressions

def test_a_rejected_code_reply_is_asked_for_code_not_json():
    replies = iter(["```python\ndef author():\n    pass\n```",       # no goal line
                    "```python\n" + A3_TRANSPORT + "```"])
    seen = []

    def transport(payload):
        seen.append(payload)
        return next(replies)

    client = Client(transport=transport, model="fake")
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    em = ca.emit_code(client, STAGE, VOCAB, "constrained-code", facts)
    assert em.source == A3_TRANSPORT
    retry = seen[1]["messages"][-1]["content"][0]["text"]
    assert "```python block" in retry and "JSON" not in retry
    assert client.logs[-1].attempts == 2 and client.logs[-1].rejections


def test_a4_path_and_subgoal_may_use_different_frames():
    src = '''# goal: spout 3-8 cm over the opening, any attitude
def author(scene):
    W = scene.w_default
    Tw_e = np.linalg.inv(np.eye(4))
    T_ef = np.eye(4)
    T_ef[:3, 3] = scene.e_point
    free = [[-np.inf, np.inf]] * 3 + [[-np.pi, np.pi]] * 3
    path = {"T0_w": np.eye(4), "Tw_e": np.linalg.inv(T_ef), "Bw": free}
    sub = {"T0_w": W, "Tw_e": np.linalg.inv(T_ef),
           "Bw": [[-0.02, 0.02], [-0.02, 0.02], [0.03, 0.08]] + [[-np.pi, np.pi]] * 3}
    return {"path": path, "subgoal": sub}
'''
    cs = ca.compile_any(_code("free-code", src), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)
    assert any("different frames" in n for n in cs.notes)
    check_pair_consistency(cs)


@pytest.mark.parametrize("ret", [
    '{"path": {"T0_w": [[1, 0, 0], [0, 1, 0, 0]], "Tw_e": np.eye(4), "Bw": free}, "subgoal": 1}',
    '{"path": {"T0_w": np.eye(4), "Tw_e": np.eye(4), "Bw": "free"}, "subgoal": 1}',
])
def test_a4_malformed_returns_are_typed_errors(ret):
    src = ("# goal: x\ndef author(scene):\n"
           "    free = [[-np.inf, np.inf]] * 3 + [[-np.pi, np.pi]] * 3\n"
           f"    return {ret}\n")
    with pytest.raises(ca.CodeError):
        ca.compile_any(_code("free-code", src), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)


def test_a3_malformed_rotation_rows_are_typed_errors():
    src = A3_TRANSPORT.replace('rot=up)}', 'rot=[{"axis": "teapot.+up", "tol_deg": "wide"}])}')
    with pytest.raises(ca.CodeError):
        ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


def test_model_prints_do_not_break_the_result_channel(monkeypatch):
    monkeypatch.setattr(ca, "CHILD_CPU_S", 1)
    printed = A3_TRANSPORT.replace('    o = point', '    print([1, 2])\n    o = point')
    cs = ca.compile_any(_code("constrained-code", printed), SYMBOLS, POSES,
                        w_point=OPENING, e_point=TIP, vocab=VOCAB)
    assert cs.subgoal is not None
    loop = "# goal: x\ndef author():\n    print(1)\n    while True:\n        pass\n"
    with pytest.raises(ca.CodeError, match="CPU limit|did not finish"):
        ca.compile_any(_code("constrained-code", loop), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


def test_memory_is_limited(monkeypatch):
    monkeypatch.setattr(ca, "CHILD_MEM_B", 1 << 30)
    src = "# goal: x\ndef author():\n    a = np.ones((20000, 20000))\n    return a\n"
    with pytest.raises(ca.CodeError, match="MemoryError|memory"):
        ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)


def test_repeat_compiles_reuse_the_result():
    ca._RESULTS.clear()
    for _ in range(2):
        ca.compile_any(_code("constrained-code", A3_TRANSPORT), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)
    assert len(ca._RESULTS) == 1


def test_code_seam_uses_the_schema_seam_check():
    a = ca.compile_any(_code("constrained-code", A3_TRANSPORT), SYMBOLS, POSES,
                       w_point=OPENING, e_point=TIP, vocab=VOCAB)
    assert ca.check_seam(a, a)[0].startswith("transport.path admits")
