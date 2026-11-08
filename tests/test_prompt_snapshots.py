"""Prompt snapshot ("golden") tests for VLM calls #1-#3 (manip_sim/vlm.py).

Every prompt the pipeline sends for plan_stages (#1), select_point_axis
(#2) and emit_constraints (#3) is built here from fixed inputs and compared
byte-for-byte against a golden JSON file in tests/golden/prompts/. Any edit
to the prompt text fails a test, so a run's numbers stay attributable to
the prompt it ran under (see PROMPT_DELTAS in manip_sim/vlm.py).

Image blocks are pinned by media type and the sha256 of their base64 data;
the image bytes are tiny deterministic stand-ins written by the test.

The request settings are pinned in tests/golden/prompts/_request.json:
the module defaults (MODEL, MAX_TOKENS, TEMPERATURE, MAX_PARSE_RETRIES,
HTTP_RETRIES, HTTP_BACKOFF_S, API_URL, API_VERSION) from a fresh copy of
the module loaded with MANIP_VLM_MODEL / MANIP_VLM_TEMPERATURE unset; the
MANIP_VLM_TEMPERATURE parsing; the non-message request fields Client
sends; and, read statically from _urllib_transport's source, the HTTP
method, header names (with every value except the API key), timeout and
retried status codes. Not pinned: the transport's control flow beyond
those literals.

PROMPT_DELTAS itself is pinned in tests/golden/prompts/_prompt_deltas.json
(append-only).

Regenerate prompt goldens after an intended prompt change:

    1. append a PROMPT_DELTAS entry in manip_sim/vlm.py;
    2. UPDATE_PROMPT_GOLDENS=1 PYTHONPATH=. python -m pytest -q \\
           tests/test_prompt_snapshots.py

A regeneration that changes a prompt without a new PROMPT_DELTAS entry is
refused (each golden records the PROMPT_DELTAS tuple it was written
under), and outside update mode the recorded tuple must equal the current
one. Case labels (call / builder / variant) are metadata, not compared,
so relabeling a case needs no delta.

The request golden is rewritten only with UPDATE_REQUEST_GOLDEN=1 (never
by UPDATE_PROMPT_GOLDENS alone): those fields change run behavior, so a
change to them is deliberate and belongs in the commit message.

Known limit: deleting a golden file and regenerating writes it fresh,
with nothing to compare against. That bypass cannot be prevented from
inside the test; git history is the record.

Offline only: no network, no API key; transports are injected fakes.
"""

from __future__ import annotations

import ast
import base64
import copy
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import manip_sim.vlm as vlm
from manip_sim.vlm import (CANONICAL_DIRS, PROMPT_DELTAS, Client,
                           PointAxisSelection, StageSpec, Vocabulary,
                           build_emission_prompt, build_point_axis_prompt,
                           build_point_axis_prompt_directions,
                           build_point_axis_prompt_named,
                           build_stage_plan_prompt, entry_line)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "prompts"
REQUEST_GOLDEN = GOLDEN_DIR / "_request.json"
DELTAS_GOLDEN = GOLDEN_DIR / "_prompt_deltas.json"
META_GOLDENS = {REQUEST_GOLDEN.stem, DELTAS_GOLDEN.stem}
UPDATE = os.environ.get("UPDATE_PROMPT_GOLDENS") == "1"
UPDATE_REQUEST = os.environ.get("UPDATE_REQUEST_GOLDEN") == "1"

HOW_TO_FIX = (
    "If this prompt change is intended, add a PROMPT_DELTAS entry in "
    "manip_sim/vlm.py (append; never edit an entry), then regenerate the "
    "goldens: UPDATE_PROMPT_GOLDENS=1 PYTHONPATH=. python -m pytest -q "
    "tests/test_prompt_snapshots.py")

# ------------------------------------------------------------ fixed inputs
# Copied (not imported) from the existing tests so the goldens depend only
# on this file and manip_sim/vlm.py: an edit to another test's fixture must
# not read as a prompt change.

# tests/test_vlm.py: the synthetic vocabulary (the fixture's fallback
# branch; the real-asset branch would tie the goldens to frames.json) and
# its candidate menu.
MENU = {3: "constructed opening_center", 7: "part handle",
        12: "curvature spout ridge"}
VOCAB = Vocabulary(objects={
    "teapot": {"points": ("spout_tip", "handle_center"),
               "axes": ("pour_axis", "up_axis", "handle_axis",
                        "tilt_axis", "front_axis", "lateral_axis"),
               "quantities": ()},
    "mug": {"points": ("opening_center",), "axes": ("up_axis",),
            "quantities": ("rim_radius",)},
}, menu=MENU)

# scenes/pour_tea.json["task"] (scripts/plan_stages.py TASK).
TASK = "pour tea from the teapot into the mug"

# tests/test_marks.py _scene(): the two marks build_marks keeps, as a
# duck-typed MarkSet (Vocabulary.from_marks reads only .marks[i].bbox/.area).
MARKSET = SimpleNamespace(marks={
    1: SimpleNamespace(bbox=(100, 10, 149, 49), area=2000),
    2: SimpleNamespace(bbox=(10, 70, 49, 109), area=1600),
})

# scripts/emit_constraints.py STAGES: the fixed pour-tea stage table.
GRASP = StageSpec(index=1, name="grasp", active="teapot", passive=None,
                  parts={"teapot": ("handle",)})
TRANSPORT = StageSpec(index=2, name="transport", active="teapot",
                      passive="mug", parts={"teapot": ("spout",),
                                            "mug": ("rim",)})
POUR = StageSpec(index=3, name="pour", active="teapot", passive="mug",
                 parts={"teapot": ("spout",), "mug": ("rim",)})

# tests/test_selection.py _pour_tea_selections(): point-only selections.
SEL_GRASP = PointAxisSelection(1, None, "+", None, "")
SEL_TRANSPORT = PointAxisSelection(3, None, "+", None, "")
SEL_POUR = PointAxisSelection(3, None, "+", None, "")

# tests/test_vlm.py entry_line tests: canonical axes and poses.
AXES = {"up_axis": np.array([0, 0, 1.0]), "front_axis": np.array([1.0, 0, 0]),
        "lateral_axis": np.array([0, 1.0, 0])}
TOL_RAD = float(np.deg2rad(10))
TIP = np.array([0.09, 0.0, 0.04])            # +front, +up of center
OPENING = np.array([0.0, 0.0, 0.10])


def _pose(xyz, R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = xyz
    if R is not None:
        T[:3, :3] = R
    return T


T_TEAPOT_SPAWN = _pose([0.0, 0.0, 0.50])
T_MUG = _pose([0.3, 0.0, 0.40])
_c, _s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
# 30 deg about +y: front tilts down 30 deg, up tilts 60 deg off level
T_TEAPOT_TILTED = _pose([0.25, 0.0, 0.55],
                        np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]]))


def _pour_frames_record(fallback=True):
    """tests/test_vlm.py _pour_frames_record: a render_stage_frames role
    record (w on the mug, z merged into mug.+up, fallback x; the teapot's
    full triad)."""
    return {
        "stage": 3, "name": "pour", "active": "teapot", "passive": "mug",
        "w": {"object": "mug", "point_body": [-0.0118, 0.0, 0.0404],
              "candidate_id": 2,
              "x_route": "teapot lateral (up x front) at entry",
              "fallback": fallback, "merged": {"z": "mug.up_axis"}},
        "triads": [
            {"role_in_stage": "w_owner", "object": "mug",
             "point_body": [-0.0118, 0.0, 0.0404],
             "axes_drawn": ["up_axis"], "directions": ["+up", "-up"]},
            {"role_in_stage": "active", "object": "teapot",
             "point_body": [-0.086, 0.085, 0.038],
             "axes_drawn": ["front_axis", "lateral_axis", "up_axis"],
             "directions": list(CANONICAL_DIRS)}],
        "end_on": {"iso": {}, "iso-opp": {},
                   "top": {"mug": ["up_axis"], "teapot": ["up_axis"]}},
        "views": {"iso": "outputs/frames/pour/iso.png",
                  "iso-opp": "outputs/frames/pour/iso-opp.png",
                  "top": "outputs/frames/pour/top.png"},
    }


def _transport_frames_record():
    """Transport: w anchored at a named point token, no fallback x
    (tests/test_vlm.py test_frames_legend_names_the_anchor_point_token)."""
    rec = _pour_frames_record(fallback=False)
    rec.update(stage=2, name="transport")
    rec["w"]["point"] = "mug.opening_center"
    return rec


def _grasp_frames_record():
    """Grasp: the whole w frame merged into the teapot's triad, w owner
    only (tests/test_vlm.py test_frames_legend_grasp_stage_...)."""
    rec = _pour_frames_record(fallback=False)
    rec.update(stage=1, name="grasp", passive=None)
    rec["w"].update(object="teapot", candidate_id=1,
                    merged={"x": "teapot.front_axis",
                            "y": "teapot.lateral_axis",
                            "z": "teapot.up_axis"})
    rec["triads"] = [dict(rec["triads"][1], role_in_stage="w_owner")]
    return rec


# Tiny deterministic image stand-ins: distinct bytes per view so the
# golden pins which view goes where.
VIEW_NAMES = ("iso", "iso-opp", "top", "scene", "cand0", "cand1", "cand2")


@pytest.fixture(scope="module")
def views(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("views")
    out = {}
    for name in VIEW_NAMES:
        p = d / f"{name}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() * 4)
        out[name] = p
    return out


def _framed(views):
    return [views["iso"], views["iso-opp"], views["top"]]


def _cands(views):
    return [views["cand0"], views["cand1"], views["cand2"]]


def _entry_transport_spawn() -> str:
    return entry_line("teapot", T_TEAPOT_SPAWN, AXES, chained=False,
                      tol_rad=TOL_RAD, anchor=("teapot.spout_tip", TIP),
                      w_anchor=("mug.opening_center", OPENING, T_MUG))


def _entry_pour_chained() -> str:
    return entry_line("teapot", T_TEAPOT_TILTED, AXES, chained=True,
                      tol_rad=TOL_RAD, anchor=("teapot.spout_tip", TIP),
                      w_anchor=("mug.opening_center", OPENING, T_MUG))


def _entry_grasp_spawn() -> str:
    return entry_line("teapot", T_TEAPOT_SPAWN, AXES, chained=False,
                      tol_rad=TOL_RAD)


# ------------------------------------------------------------------ cases

def _fake_transport(responses: list[str]):
    """(transport, seen): returns the canned responses in order and fails
    loudly, not with a stray StopIteration, if asked for one more."""
    seen: list[dict] = []

    def transport(payload):
        seen.append(payload)
        if len(seen) > len(responses):
            raise AssertionError(
                f"fake transport: request #{len(seen)} but only "
                f"{len(responses)} canned response(s)")
        return responses[len(seen) - 1]
    return transport, seen


def _capture(call, responses: list[str], n_requests: int):
    """Run a Client touchpoint against canned responses, check it sent
    exactly n_requests requests, and return the (system, messages) of the
    last one."""
    transport, seen = _fake_transport(responses)
    call(Client(transport=transport))
    assert len(seen) == n_requests, (
        f"expected {n_requests} request(s), the client sent {len(seen)}")
    return seen[-1]["system"], seen[-1]["messages"]


_EMISSION_OK = json.dumps({
    "stage": 2, "name": "transport", "active": "teapot", "passive": "mug",
    "goal": "teapot.spout_tip above mug.opening_center",
    "path_tsr": {"rot": "free", "trans": "free"},
    "subgoal_tsr": {"rot": "free", "trans": "free"}, "verify": "x"})


def _case_c2_parse_retry(views, monkeypatch):
    """#2 point-only with one off-menu answer: the parse-rejection turn
    the retry loop (Client._ask) appends."""
    return _capture(
        lambda c: c.select_point_axis(TRANSPORT, VOCAB, _cands(views)),
        [json.dumps({"candidate_id": 99, "rationale": "r"}),
         json.dumps({"candidate_id": 3, "rationale": "r"})], n_requests=2)


def _case_c3_compile_rejection(views, monkeypatch):
    """#3 with one compile-gate rejection replayed (Client.emit_constraints
    `rejections`): the assistant/user repair turns."""
    return _capture(
        lambda c: c.emit_constraints(
            TRANSPORT, VOCAB, selection=SEL_TRANSPORT,
            rejections=[(_EMISSION_OK, "transport.subgoal_tsr.rot[0]: bad")],
            entry=_entry_transport_spawn()),
        [_EMISSION_OK], n_requests=1)


def _case_c3_relation_syntax(views, monkeypatch):
    """The rot-row syntax ablation edits the ROT_ROW_SYNTAX constant."""
    monkeypatch.setattr(vlm, "ROT_ROW_SYNTAX", "relation")
    return build_emission_prompt(TRANSPORT, VOCAB, SEL_TRANSPORT,
                                 entry=_entry_transport_spawn())


# name -> (call, builder, variant, build(views, monkeypatch))
CASES = {
    # #1 plan_stages ---------------------------------------------------
    "c1_text_only": (
        "#1 plan_stages", "build_stage_plan_prompt",
        "text-only (vocab.objects; ablation baseline), no image",
        lambda v, m: build_stage_plan_prompt(TASK, VOCAB)),
    "c1_marks": (
        "#1 plan_stages", "build_stage_plan_prompt",
        "mark-addressed (Vocabulary.from_marks) with the marked scene image",
        lambda v, m: build_stage_plan_prompt(
            TASK, Vocabulary.from_marks(MARKSET), [v["scene"]])),
    # #2 select_point_axis ---------------------------------------------
    "c2_point_only_transport": (
        "#2 select_point_axis", "build_point_axis_prompt",
        "point-only scheme (default), two-object stage, 3 candidate views",
        lambda v, m: build_point_axis_prompt(TRANSPORT, VOCAB, _cands(v))),
    "c2_point_only_grasp": (
        "#2 select_point_axis", "build_point_axis_prompt",
        "point-only scheme (default), single-object stage (passive=None)",
        lambda v, m: build_point_axis_prompt(GRASP, VOCAB, _cands(v))),
    "c2_directions": (
        "#2 select_point_axis", "build_point_axis_prompt_directions",
        "canonical-directions ablation scheme (full teapot triad)",
        lambda v, m: build_point_axis_prompt_directions(
            TRANSPORT, VOCAB, _cands(v))),
    "c2_named": (
        "#2 select_point_axis", "build_point_axis_prompt_named",
        "named-axis ablation scheme (describe_symbols menu)",
        lambda v, m: build_point_axis_prompt_named(
            TRANSPORT, VOCAB, _cands(v))),
    "c2_parse_retry": (
        "#2 select_point_axis", "Client.select_point_axis -> Client._ask",
        "point-only, second request after one off-menu parse rejection",
        _case_c2_parse_retry),
    # #3 emit_constraints ----------------------------------------------
    "c3_text_passive_bare": (
        "#3 emit_constraints", "build_emission_prompt",
        "schema-only (text) variant, passive object, no selection, no entry",
        lambda v, m: build_emission_prompt(TRANSPORT, VOCAB)),
    "c3_text_no_passive_bare": (
        "#3 emit_constraints", "build_emission_prompt",
        "schema-only (text) variant, no passive object (grasp)",
        lambda v, m: build_emission_prompt(GRASP, VOCAB)),
    "c3_text_passive": (
        "#3 emit_constraints", "build_emission_prompt",
        "schema-only (text) variant as driven: passive, selection, entry line "
        "(spawn, anchor side + height)",
        lambda v, m: build_emission_prompt(
            TRANSPORT, VOCAB, SEL_TRANSPORT, entry=_entry_transport_spawn())),
    "c3_text_no_passive": (
        "#3 emit_constraints", "build_emission_prompt",
        "schema-only (text) variant as driven: no passive, selection, entry "
        "line without anchor",
        lambda v, m: build_emission_prompt(
            GRASP, VOCAB, SEL_GRASP, entry=_entry_grasp_spawn())),
    "c3_framed_pour": (
        "#3 emit_constraints", "build_emission_prompt + frames_legend",
        "image-conditioned (framed) variant: passive, 3 views, legend with "
        "fallback x and candidate-id anchor, chained tilted entry line",
        lambda v, m: build_emission_prompt(
            POUR, VOCAB, SEL_POUR, _framed(v), _pour_frames_record(),
            entry=_entry_pour_chained())),
    "c3_framed_transport": (
        "#3 emit_constraints", "build_emission_prompt + frames_legend",
        "framed variant: passive, legend anchored at a point token, no "
        "fallback",
        lambda v, m: build_emission_prompt(
            TRANSPORT, VOCAB, SEL_TRANSPORT, _framed(v),
            _transport_frames_record(), entry=_entry_transport_spawn())),
    "c3_framed_grasp": (
        "#3 emit_constraints", "build_emission_prompt + frames_legend",
        "framed variant: no passive object, fully merged w (w owner only)",
        lambda v, m: build_emission_prompt(
            GRASP, VOCAB, SEL_GRASP, _framed(v), _grasp_frames_record(),
            entry=_entry_grasp_spawn())),
    "c3_relation_syntax": (
        "#3 emit_constraints", "build_emission_prompt",
        "ROT_ROW_SYNTAX='relation' (rot-row syntax ablation constant)",
        _case_c3_relation_syntax),
    "c3_compile_rejection": (
        "#3 emit_constraints", "Client.emit_constraints",
        "text variant with one replayed compile-gate rejection",
        _case_c3_compile_rejection),
}


# ---------------------------------------------------------------- helpers

def _normalize_block(block: dict) -> dict:
    """Image data -> its sha256; everything else verbatim."""
    if block.get("type") != "image":
        return copy.deepcopy(block)
    src = dict(block["source"])
    data = src.pop("data")
    base64.b64decode(data, validate=True)        # must be real base64
    src["data_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    return {**{k: v for k, v in block.items() if k != "source"},
            "source": src}


def _normalize(system: str, messages: list) -> dict:
    return {"system": system,
            "messages": [{**{k: v for k, v in m.items() if k != "content"},
                          "content": [_normalize_block(b)
                                      for b in m["content"]]}
                         for m in messages]}


def _render(doc: dict) -> list[tuple[str, str]]:
    """(label, text) per message, the system text first, for diffs."""
    out = [("system", doc["system"])]
    for i, m in enumerate(doc["messages"]):
        parts = []
        for b in m["content"]:
            if b.get("type") == "text":
                parts.append(b["text"])
            else:
                parts.append(json.dumps(b, sort_keys=True))
        out.append((f"messages[{i}] ({m.get('role')})",
                    "\n[block]\n".join(parts)))
    return out


_SENTENCE = re.compile(r"(?<=[.;:!?])\s+(?=\S)")


def _diff_lines(text: str) -> list[str]:
    """Split at newlines and sentence ends so a one-word edit in a long
    paragraph shows as a short diff."""
    return [s for line in text.split("\n") for s in _SENTENCE.split(line)]


def _has_changes(diff: list[str]) -> bool:
    return any(l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
               for l in diff)


def _first_diff(want: dict, got: dict) -> str:
    w, g = _render(want), _render(got)
    for (lw, tw), (lg, tg) in zip(w, g):
        if tw != tg or lw != lg:
            diff = list(difflib.unified_diff(
                _diff_lines(tw), _diff_lines(tg), f"golden {lw}",
                f"current {lg}", lineterm=""))
            if not _has_changes(diff):           # whitespace-only change
                diff = list(difflib.unified_diff(
                    [repr(x) for x in tw.splitlines(True)],
                    [repr(x) for x in tg.splitlines(True)],
                    f"golden {lw} (repr)", f"current {lg} (repr)",
                    lineterm=""))
            return f"first differing message: {lw}\n" + "\n".join(diff)
    if len(w) != len(g):
        return (f"message count differs: golden has {len(w) - 1}, current "
                f"has {len(g) - 1}")
    # the rendered text is identical: the difference is in the structure
    # (extra keys, block types), so diff the normalized JSON
    items = lambda d: ([("system", d["system"])]
                       + [(f"messages[{i}]", m)
                          for i, m in enumerate(d["messages"])])
    for (label, xw), (_, xg) in zip(items(want), items(got)):
        if xw != xg:
            dump = lambda x: json.dumps(x, indent=2, sort_keys=True,
                                        ensure_ascii=False).splitlines()
            return (f"first differing message: {label} (structure)\n"
                    + "\n".join(difflib.unified_diff(
                        dump(xw), dump(xg), f"golden {label}",
                        f"current {label}", lineterm="")))
    return "no differing message found"


def _read(path: Path) -> dict | None:
    return (json.loads(path.read_text(encoding="utf-8"))
            if path.exists() else None)


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _recorded_deltas(golden: dict, path: Path) -> list[str]:
    """The PROMPT_DELTAS a golden was written under; a missing key is an
    error, never an empty tuple that would license an overwrite."""
    assert "prompt_deltas" in golden, (
        f"{path.name} has no 'prompt_deltas' key; restore it from git "
        "rather than regenerating over it")
    return golden["prompt_deltas"]


def _check_append_only(old: list[str], path: Path) -> None:
    now = list(PROMPT_DELTAS)
    assert now[:len(old)] == old, (
        f"PROMPT_DELTAS is append-only, but {old} (recorded in {path.name}) "
        f"is not a prefix of the current {now}; restore the edited entry")


# ------------------------------------------------------------------ tests

@pytest.mark.parametrize("case", sorted(CASES))
def test_prompt_matches_golden(case, views, monkeypatch):
    call, builder, variant, build = CASES[case]
    system, messages = build(views, monkeypatch)
    got = _normalize(system, messages)
    path = GOLDEN_DIR / f"{case}.json"
    golden = _read(path)
    deltas = list(PROMPT_DELTAS)
    # only the prompt payload is compared; call/builder/variant are labels
    same = golden is not None and {k: golden.get(k) for k in got} == got

    if UPDATE:
        if golden is not None:
            old = _recorded_deltas(golden, path)
            _check_append_only(old, path)
            assert same or deltas != old, (
                f"{case}: refusing to regenerate a changed prompt without a "
                f"new PROMPT_DELTAS entry (still {old}).\n{HOW_TO_FIX}\n"
                + _first_diff(golden, got))
        _write(path, {"case": case, "call": call, "builder": builder,
                      "variant": variant, "prompt_deltas": deltas, **got})
        return

    assert golden is not None, (
        f"no golden for {case} at {path}; generate it with "
        "UPDATE_PROMPT_GOLDENS=1")
    if not same:
        pytest.fail(f"{case} ({call}, {builder}: {variant}): the prompt "
                    f"differs from {path.name}.\n{HOW_TO_FIX}\n\n"
                    + _first_diff(golden, got), pytrace=False)
    # a new PROMPT_DELTAS entry must come with regenerated goldens (and a
    # regeneration with a new entry, enforced in update mode above)
    assert _recorded_deltas(golden, path) == deltas, (
        f"{case}: the golden was written under PROMPT_DELTAS "
        f"{golden['prompt_deltas']}, the code now has {deltas}. Regenerate "
        "the goldens with UPDATE_PROMPT_GOLDENS=1 so they record the prompt "
        "text this delta describes.")


def test_prompt_deltas_are_pinned():
    """The current tuple itself, in a file only this test reads and
    writes, so an edited or removed entry fails even when every golden is
    regenerated alongside it."""
    golden = _read(DELTAS_GOLDEN)
    if UPDATE:
        if golden is not None:
            _check_append_only(_recorded_deltas(golden, DELTAS_GOLDEN),
                               DELTAS_GOLDEN)
        _write(DELTAS_GOLDEN, {"prompt_deltas": list(PROMPT_DELTAS)})
        return
    assert golden is not None, (
        f"no {DELTAS_GOLDEN.name}; run with UPDATE_PROMPT_GOLDENS=1")
    recorded = _recorded_deltas(golden, DELTAS_GOLDEN)
    _check_append_only(recorded, DELTAS_GOLDEN)
    assert recorded == list(PROMPT_DELTAS), (
        f"PROMPT_DELTAS changed: pinned {recorded}, now "
        f"{list(PROMPT_DELTAS)}. After appending an entry, regenerate the "
        "goldens with UPDATE_PROMPT_GOLDENS=1.")


def test_no_orphan_goldens():
    names = {p.stem for p in GOLDEN_DIR.glob("*.json")} - META_GOLDENS
    assert names <= set(CASES), (
        f"golden files without a case: {sorted(names - set(CASES))}")


def _fresh_vlm(monkeypatch):
    """A private copy of manip_sim/vlm.py imported with the run-level env
    knobs unset, so the pinned defaults do not depend on the caller's
    environment (manip_sim.vlm itself is left untouched)."""
    monkeypatch.delenv("MANIP_VLM_MODEL", raising=False)
    monkeypatch.delenv("MANIP_VLM_TEMPERATURE", raising=False)
    spec = importlib.util.spec_from_file_location("_vlm_snapshot_defaults",
                                                  vlm.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod     # dataclasses look the module up here
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def _request_fields(mod, monkeypatch, temperature) -> dict:
    monkeypatch.setattr(mod, "TEMPERATURE", temperature)
    transport, seen = _fake_transport(
        [json.dumps({"candidate_id": 3, "rationale": "r"})])
    vocab = mod.Vocabulary(objects=VOCAB.objects, menu=MENU)
    stage = mod.StageSpec(**{k: getattr(TRANSPORT, k) for k in
                             ("index", "name", "active", "passive", "parts")})
    mod.Client(transport=transport).select_point_axis(stage, vocab, [])
    assert len(seen) == 1, f"expected 1 request, the client sent {len(seen)}"
    return {k: v for k, v in seen[0].items() if k not in ("system", "messages")}


def _transport_literals(mod) -> dict:
    """The HTTP literals in _urllib_transport, read from its source (the
    function is never called: it needs the API key). Header values that
    are expressions are recorded as their source text, so the key itself
    is only ever the name of a local variable."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(mod._urllib_transport)))
    out: dict = {}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id.isupper():
            return {"const": node.id, "value": getattr(mod, node.id)}
        return {"expr": ast.unparse(node)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = ast.unparse(node.func)
        kw = {k.arg: k.value for k in node.keywords}
        if fn == "urllib.request.Request":
            out["request_url"] = value(node.args[0])
            out["method"] = value(kw["method"])
            out["headers"] = {value(k): value(v)
                              for k, v in zip(kw["headers"].keys,
                                              kw["headers"].values)}
        elif fn == "urllib.request.urlopen":
            out["timeout"] = value(kw["timeout"])
    codes = [ast.literal_eval(n.comparators[0]) for n in ast.walk(tree)
             if isinstance(n, ast.Compare) and ast.unparse(n.left) == "e.code"]
    out["retried_http_status"] = [list(c) for c in codes]
    return out


def test_request_payload_matches_golden(monkeypatch):
    mod = _fresh_vlm(monkeypatch)
    default_temperature = mod.TEMPERATURE
    got = {
        "constants": {
            "MODEL": mod.MODEL, "MAX_TOKENS": mod.MAX_TOKENS,
            "TEMPERATURE": default_temperature,
            "MAX_PARSE_RETRIES": mod.MAX_PARSE_RETRIES,
            "HTTP_RETRIES": mod.HTTP_RETRIES,
            "HTTP_BACKOFF_S": mod.HTTP_BACKOFF_S,
            "API_URL": mod.API_URL, "API_VERSION": mod.API_VERSION},
        # MANIP_VLM_TEMPERATURE parsing: unset, "none" (omit the key), float
        "temperature_env": {
            "<unset>": mod._temperature(None),
            "none": mod._temperature("none"),
            " None ": mod._temperature(" None "),
            "0.7": mod._temperature("0.7")},
        # every request field other than system/messages
        "request_default": _request_fields(mod, monkeypatch,
                                           default_temperature),
        "request_temperature_none": _request_fields(mod, monkeypatch, None),
        "transport": _transport_literals(mod),
    }
    golden = _read(REQUEST_GOLDEN)
    if UPDATE_REQUEST:
        _write(REQUEST_GOLDEN, got)
        return
    assert golden is not None, (
        f"no {REQUEST_GOLDEN.name}; run with UPDATE_REQUEST_GOLDEN=1")
    if golden != got:
        diff = difflib.unified_diff(
            json.dumps(golden, indent=2).splitlines(),
            json.dumps(got, indent=2).splitlines(),
            f"golden {REQUEST_GOLDEN.name}", "current", lineterm="")
        pytest.fail(
            f"the request settings differ from {REQUEST_GOLDEN.name}. These "
            "change run behavior (model, token budget, sampling, retries, "
            "transport) rather than prompt text, so UPDATE_PROMPT_GOLDENS "
            "does not rewrite them. If the change is intended, rewrite the "
            "golden with UPDATE_REQUEST_GOLDEN=1 PYTHONPATH=. python -m "
            "pytest -q tests/test_prompt_snapshots.py and record the change "
            "in the commit message.\n\n" + "\n".join(diff), pytrace=False)
