"""manip_sim/code_authoring.py: the code media of call #3 (A3 constrained
code, A4 free code). Offline: real pour-tea sidecars, a fake transport, no
network, no simulator."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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


# ------------------------------------------------ browser-review follow-ups

def test_shared_guidance_is_verbatim_in_the_schema_prompt():
    from manip_sim.vlm import build_emission_prompt
    schema, _ = build_emission_prompt(STAGE, VOCAB)
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    a3, _ = ca.build_code_prompt(STAGE, VOCAB, "constrained-code", facts)
    a4, _ = ca.build_code_prompt(STAGE, VOCAB, "free-code", facts)
    for key, sentence in ca.SHARED_WITH_SCHEMA.items():
        assert sentence in schema, key
        assert sentence in a3, key
    for key in ("dirs", "path", "rotate"):
        assert ca.SHARED_WITH_SCHEMA[key] in a4, key


def test_every_listed_value_is_readable_through_the_api():
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    _, msgs = ca.build_code_prompt(STAGE, VOCAB, "constrained-code", facts)
    user = msgs[0]["content"][0]["text"]
    assert "active_point('teapot.spout_tip')" in user
    src = A3_TRANSPORT.replace('    o = point("mug.opening_center")',
                               '    o = point("mug.opening_center")\n'
                               '    t = active_point("teapot.spout_tip")')
    ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES, w_point=OPENING,
                   e_point=TIP, vocab=VOCAB)
    s4, m4 = ca.build_code_prompt(STAGE, VOCAB, "free-code", facts)
    assert "scene.points_world['teapot.spout_tip']" in m4[0]["content"][0]["text"]


def test_a4_world_positions_match_the_body_ones():
    src = '''# goal: x
def author(scene):
    p = scene.T_active @ np.append(scene.points["teapot.spout_tip"], 1.0)
    if np.abs(p[:3] - scene.points_world["teapot.spout_tip"]).max() > 1e-5:
        raise ValueError("world and body disagree")
    free = [[-np.inf, np.inf]] * 3 + [[-np.pi, np.pi]] * 3
    box = [[-1.0, 1.0]] * 3 + [[-np.pi, np.pi]] * 3
    return {"path": {"T0_w": np.eye(4), "Tw_e": np.eye(4), "Bw": free},
            "subgoal": {"T0_w": np.eye(4), "Tw_e": np.eye(4), "Bw": box}}
'''
    ca.compile_any(_code("free-code", src), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)


def test_a4_unbounded_subgoal_is_rejected_at_compile():
    src = '''# goal: x
def author(scene):
    free = [[-np.inf, np.inf]] * 3 + [[-np.pi, np.pi]] * 3
    return {"path": {"T0_w": np.eye(4), "Tw_e": np.eye(4), "Bw": free},
            "subgoal": {"T0_w": np.eye(4), "Tw_e": np.eye(4), "Bw": free}}
'''
    with pytest.raises(ca.CodeError, match="unbounded"):
        ca.compile_any(_code("free-code", src), SYMBOLS, POSES, w_point=OPENING, e_point=TIP)


def test_code_seam_rejects_a_path_that_excludes_the_previous_goals():
    prev = ca.compile_any(_code("constrained-code", A3_TRANSPORT), SYMBOLS, POSES,
                          w_point=OPENING, e_point=TIP, vocab=VOCAB)
    entry = POSES | {"teapot": prev.subgoal.nominal()}
    narrow = '''# goal: x
def author():
    f = feature
    return {"path": tsr(z=(f[2] - 0.001, f[2] + 0.001)),
            "subgoal": tsr(x=(-0.05, 0.05), y=(-0.05, 0.05),
                           z=(f[2] - 0.001, f[2] + 0.001))}
'''
    cur = ca.compile_any(_code("constrained-code", narrow, name="pour"), SYMBOLS, entry,
                         w_point=OPENING, e_point=TIP, vocab=VOCAB)
    with pytest.raises(ca.CodeError, match="does not admit every"):
        ca.check_seam(prev, cur)


def test_one_max_tokens_knob_for_every_method():
    import os
    import subprocess
    import sys
    code = ("import manip_sim.vlm as v, manip_sim.code_authoring as c; "
            "print(v.MAX_TOKENS, c.MAX_TOKENS)")
    env = {**os.environ, "MANIP_VLM_MAX_TOKENS": "4000"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True).stdout.split()
    assert out == ["4000", "4000"]
    env.pop("MANIP_VLM_MAX_TOKENS")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True).stdout.split()
    assert out == ["2000", "4000"]     # defaults unchanged when unset


def test_numpy_methods_that_import_lazily_work_in_the_child():
    src = A3_TRANSPORT.replace('    o = point("mug.opening_center")',
                               '    o = point("mug.opening_center")\n'
                               '    _m = (np.ones(3).max(), np.ones(3).sum(), np.ones(3).mean())')
    src = src.replace("_m = ", "m_ = ")
    ca.compile_any(_code("constrained-code", src), SYMBOLS, POSES, w_point=OPENING,
                   e_point=TIP, vocab=VOCAB)


def test_call_log_keeps_rejected_replies_and_stop_reasons():
    """Logging only: every attempt's stop reason and usage, and the text of each
    rejected reply, land on the CallLog; the reply that is returned is unchanged."""
    from manip_sim import vlm
    from manip_sim.vlm import CallLog
    replies = iter([("no code here", "max_tokens"), ("ok", "end_turn")])

    def transport(payload):
        text, stop = next(replies)
        vlm.LAST_RESPONSE.update(stop_reason=stop, usage={"output_tokens": len(text)})
        return text

    def parse(raw):
        if raw != "ok":
            raise ParseRejection("expected ok")
        return raw

    client = Client(transport=transport)
    assert client._ask("t", "sys", [{"role": "user", "content": []}], parse) == "ok"
    log = client.logs[-1]
    assert log.attempts == 2 and log.rejections == ["expected ok"]
    assert log.rejected_raw == ["no code here"] and log.raw == "ok"
    assert log.stop_reasons == ["max_tokens", "end_turn"]
    assert log.usage == [{"output_tokens": 12}, {"output_tokens": 2}]

    # a transport that reports no metadata logs None, not a stale value
    client = Client(transport=lambda payload: "ok")
    client._ask("t", "sys", [], parse)
    assert client.logs[-1].stop_reasons == [None] and client.logs[-1].usage == [None]
    assert isinstance(CallLog("x").gate_rejection, str)


# ------------------------------------------- A4 / A5: given frames, Bw only

A3_A4PILOT_PROMPT_SHA256 = "277922f41e8191f2b1ce33522075a2b484684257a657ec6c886c1ead9b5124ed"

BOUNDS = {   # spout tip 3-8 cm over the opening, teapot upright within 15 deg
    "path": [[None, None], [None, None], [None, None], [-15, 15], [-15, 15], [-180, 180]],
    "subgoal": [[-0.02, 0.02], [-0.02, 0.02], [0.03, 0.08], [-15, 15], [-15, 15],
                [-180, 180]]}


def _json(b=BOUNDS, goal="the spout tip hovers 3-8 cm over the opening"):
    return "```json\n" + json.dumps({"goal": goal, **b}) + "\n```"


def _bounds_code(b=BOUNDS):
    return ("# goal: the spout tip hovers 3-8 cm over the opening\n"
            "def author(scene):\n    return " + json.dumps(b).replace("null", "None") + "\n")


def _compile(em, poses=POSES):
    return ca.compile_any(em, SYMBOLS, poses, w_point=OPENING, e_point=TIP)


def test_a3_and_pilot_a4_prompt_text_is_unchanged():
    """A3 and the pilot's A4 ran; the bounds media must not move a byte of them."""
    import hashlib
    t = ca._COMMON + ca._A3 + ca._A4 + ca._FORMAT
    assert hashlib.sha256(t.encode()).hexdigest() == A3_A4PILOT_PROMPT_SHA256
    assert ca.REPAIR_TEXT["code"] == (
        "Re-emit the corrected ```python block only.",
        "That code was rejected: {r}. Re-emit the corrected code block only, "
        "changing as little as needed.")
    for m in ("constrained-code", "free-code"):
        assert ca.system_text(m, VOCAB) == (
            ca._COMMON + (ca._A3 if m == "constrained-code" else ca._A4)
            + f"Grounded symbols:\n{VOCAB.describe_directions()}\n\n" + ca._FORMAT)


def test_a4_and_a5_get_the_same_values_and_are_never_asked_for_a_frame():
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    s4, m4 = ca.build_code_prompt(STAGE, VOCAB, "bounds-code", facts)
    s5, m5 = ca.build_code_prompt(STAGE, VOCAB, "bounds-json", facts)
    shared = ca._BD_COMMON + ca._BD_GIVEN
    assert s4.startswith(shared) and s5.startswith(shared)
    assert m4[0]["content"][0]["text"] == m5[0]["content"][0]["text"]
    for s in (s4, s5):
        assert "frames are given; you write only the bounds" in s
        assert "Tw_e" not in s and "T0_w" not in s
    assert "```python" in s4 and "```json" not in s4
    assert "```json" in s5 and "python" not in s5.lower()
    user = m5[0]["content"][0]["text"]
    assert "w = mug's canonical frame at mug.opening_center" in user
    assert "e = teapot's canonical frame at teapot.spout_tip: e.x = teapot.+front" in user
    assert "e's displacement at entry" in user


def test_given_frames_are_the_schemas_w_and_the_active_canonical_frame():
    cs = _compile(ca.parse_bounds_json(_json(), STAGE))
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    assert np.allclose(cs.subgoal.T0_w, facts["w_default"], atol=1e-6)    # the schema's w
    assert np.allclose(cs.subgoal.Bw[2], [0.03, 0.08])
    assert np.allclose(cs.subgoal.Bw[3], np.radians([-15, 15]))           # degrees -> radians
    assert np.all(np.isinf(cs.path.Bw[:3]))                               # null -> unbounded
    T = cs.subgoal.T0_w @ cs.subgoal.Tw_e        # zero displacement: canonical axes on w's
    up_w = cs.subgoal.T0_w[:3, 2]
    assert np.allclose(T[:3, :3] @ SYMBOLS["teapot"].axes["up_axis"], up_w, atol=1e-6)
    assert np.allclose(T[:3, :3] @ TIP + T[:3, 3], cs.subgoal.T0_w[:3, 3], atol=1e-9)
    check_pair_consistency(cs)


def test_entry_displacement_is_in_bws_coordinates():
    """The listed entry displacement is exactly where the entry pose sits in Bw."""
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    cs = _compile(ca.parse_bounds_json(_json(), STAGE))
    d = cs.path.displacement(POSES["teapot"])
    assert np.allclose(facts["entry_d"][:3], d[:3], atol=1e-6)
    assert np.allclose(facts["entry_d"][3:], np.degrees(d[3:]), atol=1e-4)


def test_a4_code_matches_a5_and_reads_the_scene_at_run_time():
    cs5 = _compile(ca.parse_bounds_json(_json(), STAGE))
    cs4 = _compile(_code("bounds-code", _bounds_code()))
    for k in ("path", "subgoal"):
        a, b = getattr(cs4, k), getattr(cs5, k)
        assert np.allclose(a.T0_w, b.T0_w) and np.allclose(a.Tw_e, b.Tw_e)
        assert np.array_equal(np.isinf(a.Bw), np.isinf(b.Bw))
        assert np.allclose(np.nan_to_num(a.Bw), np.nan_to_num(b.Bw))
    src = '''# goal: spout tip 3 cm over its entry height, at most
def author(scene):
    h = scene.entry[2]
    return {"path": [[None, None]] * 3 + [[-15, 15], [-15, 15], [-180, 180]],
            "subgoal": [[-0.02, 0.02], [-0.02, 0.02], [0.03, max(0.08, h + 0.01)],
                        [-15, 15], [-15, 15], [-180, 180]]}
'''
    hi = [_compile(_code("bounds-code", src), POSES | {"teapot": make_pose((0.3, 0.1, z))}
                   ).subgoal.Bw[2, 1] for z in (0.05, 0.30)]
    assert hi[1] > hi[0]


@pytest.mark.parametrize("change, why", [
    ({"subgoal": [[-0.02, 0.02], [-0.02, 0.02], [0.03, None], [-15, 15], [-15, 15],
                  [-180, 180]]}, "unbounded"),
    ({"path": [[None, None]] * 3 + [[None, 15], [-15, 15], [-180, 180]]}, "free rotation"),
    ({"path": [[None, None]] * 3 + [[-400, 15], [-15, 15], [-180, 180]]}, "within"),
    ({"path": [[None, None]] * 3 + [[-15, 15], [-200, 200], [-180, 180]]}, "360 wide"),
    ({"subgoal": [[float("nan"), 0.02]] + BOUNDS["subgoal"][1:]}, "NaN"),
    ({"path": [[None, None]] * 5}, "6 rows"),
    ({"subgoal": [[0.02, -0.02]] + BOUNDS["subgoal"][1:]}, "hi < lo"),
    ({"path": [[None, None]] * 3 + [["a", 15], [-15, 15], [-180, 180]]}, "finite numbers"),
])
def test_bound_errors_are_typed_and_say_what_is_allowed(change, why):
    with pytest.raises(ca.CodeError, match=why):
        _compile(_code("bounds-json", json.dumps({"goal": "g", **BOUNDS, **change})))


def test_a5_parse_takes_one_json_block():
    em = ca.parse_bounds_json("Here:\n" + _json(), STAGE)
    assert em.medium == "bounds-json" and em.goal.startswith("the spout tip")
    for bad, why in ((_json() + _json(), "exactly one"), ("no block", "exactly one"),
                     ("```json\n{nope}\n```", "invalid JSON"),
                     ("```json\n[1, 2]\n```", "one JSON object"),
                     ("```json\n" + json.dumps({"goal": "g", "path": []}) + "\n```", "missing"),
                     (_json(goal=" "), "goal")):
        with pytest.raises(ParseRejection, match=why):
            ca.parse_bounds_json(bad, STAGE)


def test_a5_retries_ask_for_json_and_the_artifact_round_trips(tmp_path):
    replies = iter(["```python\ndef author(scene):\n    pass\n```", _json()])
    seen = []

    def transport(payload):
        seen.append(payload)
        return next(replies)

    client = Client(transport=transport, model="fake")
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    em = ca.emit_code(client, STAGE, VOCAB, "bounds-json", facts,
                      rejections=[("old", "bad z")])
    assert "That answer was rejected: bad z" in seen[0]["messages"][-1]["content"][0]["text"]
    retry = seen[1]["messages"][-1]["content"][0]["text"]
    assert "```json block" in retry and "python" not in retry
    assert client.logs[-1].attempts == 2
    doc = {"roles": ["transport_active"], "emissions": [ca.as_record(em)],
           "compiled": [{"stage": "transport", "grounded": True}]}
    p = tmp_path / "a5.json"
    p.write_text(json.dumps(doc))
    assert ca.load_any_emissions(p) == {"transport_active": em}


def test_unbounded_translation_sides_are_none_or_inf_and_nan_is_an_error():
    """None (JSON null) and a same-side inf are unbounded; numpy keeps None in an
    object array; a NaN can only come from a computation, so it is rejected."""
    inf = float("inf")
    for path in ([[None, None]] * 3, [[-inf, inf]] * 3):
        b = {**BOUNDS, "path": path + BOUNDS["path"][3:]}
        src = _bounds_code(b).replace("Infinity", "float('inf')")
        assert np.all(np.isinf(_compile(_code("bounds-code", src)).path.Bw[:3]))
    obj = _bounds_code().replace('return {"path": ', 'return {"path": np.array(').replace(
        ', "subgoal"', '), "subgoal"')        # None mixed with numbers: numpy picks object
    assert np.all(np.isinf(_compile(_code("bounds-code", obj)).path.Bw[:3]))
    em = ca.parse_bounds_json('```json\n{"goal": "g", "path": [[-Infinity, Infinity], '
                              '[null, null], [null, null], [-15, 15], [-15, 15], [-180, 180]], '
                              '"subgoal": ' + json.dumps(BOUNDS["subgoal"]) + '}\n```', STAGE)
    assert np.all(np.isinf(_compile(em).path.Bw[:3]))
    nan = _bounds_code().replace('"path": [[None, None]', '"path": [[float("nan"), None]')
    with pytest.raises(ca.CodeError, match="NaN"):
        _compile(_code("bounds-code", nan))


def test_rotation_bands_are_read_as_written_past_90_pitch_and_past_180():
    """No clipping: the TSR reads pitch past +-90 (dual RPY) and roll / yaw past
    +-180 (2pi shifts), so natural answers keep their meaning."""
    free = [[None, None]] * 3 + [[-180, 180]] * 3

    def tsr_of(sub):
        b = {"path": free, "subgoal": BOUNDS["subgoal"][:3] + sub}
        return _compile(_code("bounds-json", json.dumps({"goal": "g", **b}))).subgoal

    def body_at(t, rpy_deg):          # body pose at displacement (0, 0, 0.05, rpy) from w
        d = make_pose([0, 0, 0.05], Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix())
        return t.T0_w @ d @ t.Tw_e

    t = tsr_of([[-5, 5], [-98, -82], [-180, 180]])           # vertical within 8 deg
    assert np.allclose(t.Bw[4], np.radians([-98, -82]))
    assert t.contains(body_at(t, [0, -95, 0])) and t.contains(body_at(t, [0, -84, 30]))
    assert not t.contains(body_at(t, [0, -70, 0]))
    t = tsr_of([[-180, 180], [100, 120], [-180, 180]])       # wholly past +90: with roll
    assert t.contains(body_at(t, [0, 110, 0]))               # and yaw free it is the same
    assert t.contains(body_at(t, [0, 70, 0]))                # set as pitch [60, 80]
    assert not t.contains(body_at(t, [0, 30, 0]))
    t = tsr_of([[-5, 5], [-5, 5], [170, 190]])               # turned around, crossing 180
    assert t.contains(body_at(t, [0, 0, 180])) and t.contains(body_at(t, [0, 0, -175]))
    assert not t.contains(body_at(t, [0, 0, 150]))


def test_bounds_media_survive_a_missing_e_point_and_others_a_missing_up_axis():
    from dataclasses import replace
    em = ca.parse_bounds_json(_json(), STAGE)
    cs = ca.compile_any(em, SYMBOLS, POSES, w_point=OPENING, e_point=None)
    assert np.allclose((cs.subgoal.T0_w @ cs.subgoal.Tw_e)[:3, 3], cs.subgoal.T0_w[:3, 3])
    no_up = {**SYMBOLS, "teapot": replace(SYMBOLS["teapot"], axes={
        k: v for k, v in SYMBOLS["teapot"].axes.items() if k != "up_axis"})}
    facts = ca.stage_facts("teapot", "mug", no_up, POSES, OPENING, TIP)   # A3 / A4-pilot fine
    assert facts["Tw_e"] is None and ca.describe_facts(facts, "constrained-code")
    with pytest.raises(ca.CodeError, match="no up_axis"):
        ca.describe_facts(facts, "bounds-json")


def test_the_values_table_lists_world_z_in_w_as_a4s_scene_does():
    facts = ca.stage_facts("teapot", "mug", SYMBOLS, POSES, OPENING, TIP)
    assert "world.z = " in ca.describe_facts(facts, "bounds-json")
