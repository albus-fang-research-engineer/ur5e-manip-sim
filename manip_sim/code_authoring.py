"""Call #3 authored as code: the two code-medium variants of the authoring
ablation, beside the token schema (vlm.build_emission_prompt + compile_tsr).

    constrained-code (A3)  The model writes `author()` against a helper API.
                           Same frame w as the schema (fixed by rule), same
                           anchor points, same feature pinned by Tw_e, same
                           rule table for rotations (`points_to` /
                           `perpendicular_to` only), but every magnitude is
                           a number the code computes: rotation tolerances
                           in degrees, translation bounds as [lo, hi]
                           intervals in w. The result is lowered to a
                           StageEmission (numeric `expr` rows, numeric
                           tolerance keys) and compiled by compile_stage,
                           so it meets exactly the schema's checks.
    free-code (A4)         The model writes `author(scene)` returning raw
                           numeric TSRs: T0_w, Tw_e and the 6x2 B^w for the
                           path and the subgoal. It may choose any frame
                           and any feature; the same feasibility checks
                           (entry inside the path, path contains the
                           subgoal, subgoal samples on the path, seam)
                           apply to the numbers it returns.

The code is the authored artifact, not its output: it is stored verbatim and
re-run wherever the schema would be recompiled (the emit gate at the nominal
entry, plan_task.py at the actual entry), so values read through the API
track the pose they are evaluated at. compile_any() is the single entry
point both call sites use.

Running model-written code. The source is parsed and checked here (no
imports, no names beginning with a double underscore, no attributes
beginning with an underscore, no file or
reflection builtins, no numpy file I/O), then executed in a separate
`python -I -B` process with restricted builtins, CPU, wall-clock and
address-space limits, no file writes (RLIMIT_FSIZE = 0), a scratch working
directory and a minimal environment. Its result is one tagged JSON line on
stdout; the model's own print() goes to stderr. This is defense in depth for
a research harness, not a security boundary against an adversary.

numpy/scipy only in this module; the child process imports numpy (and
scipy's Rotation for free-code) and nothing from this repo.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .compile_tsr import (ROT_TOL_RAD, CompiledStage, CompileError, _Box,
                          _canonical_w, _check_path_containment, check_stage_seam,
                          compile_stage)
from .frames import Frame, Symbols
from .tsr import TSR
from . import vlm
from .vlm import (_DIR_AXIS, CallLog, ParseRejection, RotRow, StageEmission, VLMError,
                  StageSpec, TransTerm, TSRSpec, Vocabulary, _parse_rot_rows,
                  _text, emission_from_json, frames_legend, image_block,
                  load_emissions)

MEDIA = ("constrained-code", "free-code")
LABEL = {"constrained-code": "A3", "free-code": "A4"}
PROMPT_VERSION = "code-authoring-2026-11-07"   # bump on any prompt text change
MAX_TOKENS = int(os.environ.get("MANIP_VLM_MAX_TOKENS", "4000"))   # a cap, not a target;
                           # the ablation runner sets one value for every method
MAX_SOURCE_CHARS = 20000
CHILD_CPU_S = 15
CHILD_WALL_S = 30
CHILD_MEM_B = 4 << 30   # address space: numpy + scipy import fits well under this
UNBOUNDED_M = 100.0        # a half-open path row (lo or hi None) lowers to +-this


class CodeError(CompileError):
    """A code emission that could not be run or grounded. Same shape as
    CompileError (slot + instructive reason) so the emit loop feeds it back
    the same way."""


@dataclass(frozen=True)
class CodeEmission:
    """Call #3 output in a code medium: the source is the artifact."""
    stage: int
    name: str
    active: str
    passive: str | None
    medium: str
    source: str
    goal: str = ""


# ------------------------------------------------------------ source checks

_BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "input", "breakpoint", "help", "exit",
    "quit", "memoryview", "type", "object", "super", "classmethod",
    "staticmethod", "property",
})
# numpy entry points to files, processes or native code
_BANNED_ATTRS = frozenset({
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "fromregex", "ctypeslib",
    "ctypes", "lib", "f2py", "testing", "distutils", "os", "sys", "dump",
    "dumps", "frombuffer", "DataSource", "show_config", "info",
})
_BANNED_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                 ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
                 ast.ClassDef)
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def check_source(src: str, entry: str) -> None:
    """ParseRejection unless `src` parses, defines top-level `entry`, and
    uses none of the banned constructs. The rejection names the line."""
    if len(src) > MAX_SOURCE_CHARS:
        raise ParseRejection(f"code is {len(src)} characters; keep it under "
                             f"{MAX_SOURCE_CHARS}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ParseRejection(f"syntax error at line {e.lineno}: {e.msg}")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, _BANNED_NODES):
            raise ParseRejection(f"line {line}: {type(node).__name__} is not "
                                 "allowed (no imports, classes, globals or "
                                 "async; np and math are predefined)")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ParseRejection(f"line {line}: attribute {node.attr!r} "
                                     "is not allowed (no names beginning "
                                     "with an underscore)")
            if node.attr in _BANNED_ATTRS:
                raise ParseRejection(f"line {line}: {node.attr!r} is not "
                                     "allowed (no file, process or native "
                                     "access)")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ParseRejection(f"line {line}: name {node.id!r} is not "
                                     "allowed")
            if node.id in _BANNED_NAMES:
                raise ParseRejection(f"line {line}: builtin {node.id!r} is "
                                     "not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry
               for n in tree.body):
        raise ParseRejection(f"the code must define a top-level function "
                             f"`{entry}`")


def entry_name(medium: str) -> str:
    return "author"


def parse_code_response(raw: str, stage: StageSpec, medium: str) -> CodeEmission:
    """Exactly one fenced python block; its `# goal:` comment is the goal."""
    blocks = _FENCE.findall(raw)
    if len(blocks) != 1:
        raise ParseRejection(f"expected exactly one ```python block, got "
                             f"{len(blocks)}")
    src = blocks[0].strip("\n") + "\n"
    check_source(src, entry_name(medium))
    m = re.search(r"^\s*#\s*goal:\s*(.+)$", src, re.M)
    if not m:
        raise ParseRejection("the code's first line must be a comment "
                             "'# goal: <one short sentence>'")
    return CodeEmission(stage=stage.index, name=stage.name, active=stage.active,
                        passive=stage.passive, medium=medium, source=src,
                        goal=m.group(1).strip())


# ------------------------------------------------------------- stage facts

def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _dirs_body(obj: str, sym: Symbols) -> dict[str, np.ndarray]:
    """Licensed signed canonical directions of one object, body coords."""
    out = {}
    for tok, ax in _DIR_AXIS.items():
        if ax in sym.axes:
            u = _unit(sym.axes[ax])
            out[f"{obj}.+{tok}"] = u
            out[f"{obj}.-{tok}"] = -u
    return out


def stage_facts(active: str, passive: str, symbols: dict[str, Symbols],
                body_poses: dict[str, np.ndarray], w_point, e_point) -> dict:
    """Every number the code may read, at these entry poses, as plain
    lists. The w frame is compile_tsr's rule (same call, same fallback), so
    both media see the frame the renders draw."""
    w_point = np.asarray(w_point, float).reshape(3)
    e_point = (np.zeros(3) if e_point is None
               else np.asarray(e_point, float).reshape(3))
    w_frame, route = _canonical_w(passive, active, symbols, body_poses,
                                  w_point, "passive")
    T0_w = body_poses[passive] @ w_frame.T()
    W = np.linalg.inv(T0_w)

    def in_w(obj, p):
        return (W @ body_poses[obj] @ np.append(p, 1.0))[:3]

    dirs_w, dirs_body = {}, {}
    for obj in (active, passive):
        for tok, u in _dirs_body(obj, symbols[obj]).items():
            dirs_body[tok] = u
            dirs_w[tok] = W[:3, :3] @ body_poses[obj][:3, :3] @ u
    for tok, z in (("world.z", 1.0), ("world.-z", -1.0)):
        dirs_w[tok] = W[:3, :3] @ np.array([0.0, 0.0, z])
    points_body = {f"{o}.{n}": np.asarray(p, float)
                   for o in (active, passive) for n, p in symbols[o].points.items()}
    facts = {
        "active": active, "passive": passive, "w_route": route,
        # constrained-code view: w coordinates
        "points_w": {k: in_w(passive, p) for k, p in points_body.items()
                     if k.startswith(passive + ".")},
        "active_points_w": {k: in_w(active, p) for k, p in points_body.items()
                            if k.startswith(active + ".")},
        "feature_w": in_w(active, e_point),
        "dirs_w": dirs_w,
        "quantities": {f"{o}.{q}": float(v) for o, s in symbols.items()
                       for q, v in s.quantities.items()},
        # free-code view: world poses and body coordinates
        "T_active": body_poses[active], "T_passive": body_poses[passive],
        "points_world": {k: (body_poses[k.split(".")[0]] @ np.append(p, 1.0))[:3]
                         for k, p in points_body.items()},
        "dirs_world": {k: body_poses[k.split(".")[0]][:3, :3] @ u
                       for k, u in dirs_body.items()},
        "w_default": T0_w, "points_body": points_body, "dirs_body": dirs_body,
        "w_point": w_point, "e_point": e_point,
    }
    return _plain(facts)


def _plain(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 6).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def _vec(v) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


def _mat(T) -> str:
    return "[" + ", ".join(_vec(r) for r in T) + "]"


def describe_facts(facts: dict, medium: str) -> str:
    """The numbers, as text for the user turn (the code reads the same
    values at run time; these are for the model to reason with)."""
    a, p = facts["active"], facts["passive"]
    q = "; ".join(f"{k} = {v:.4f}" for k, v in sorted(facts["quantities"].items()))
    if medium == "constrained-code":
        lines = ["Grounded values at stage entry, in w (meters; directions "
                 "are unit vectors in w):"]
        if facts["w_route"] != f"{p}.front_axis":
            lines.append(f"  w.x is fixed by rule from {facts['w_route']} ({p} has no "
                         "+front direction)")
        lines += [f"  point({k!r}) = {_vec(v)}" for k, v in sorted(facts["points_w"].items())]
        lines.append(f"  feature (the {a} point the TSR pins at w) = "
                     f"{_vec(facts['feature_w'])}")
        lines += [f"  active_point({k!r}) = {_vec(v)}"
                  for k, v in sorted(facts["active_points_w"].items())]
        lines += [f"  direction({k!r}) = {_vec(v)}"
                  for k, v in sorted(facts["dirs_w"].items()) if not k.startswith(p + ".-")
                  and not k.startswith(a + ".-")]
        lines.append(f"  quantities: {q or '(none)'}")
        return "\n".join(lines)
    lines = ["Grounded values at stage entry (world frame, meters):",
             f"  scene.T_active ({a} body pose) = {_mat(facts['T_active'])}",
             f"  scene.T_passive ({p} body pose) = {_mat(facts['T_passive'])}",
             f"  scene.w_default (the dashed frame w in the images) = "
             f"{_mat(facts['w_default'])}",
             *([f"  scene.w_default's x is fixed by rule from {facts['w_route']} ({p} "
                "has no +front direction)"] if facts["w_route"] != f"{p}.front_axis" else []),
             f"  scene.w_point (selected point on {p}, {p} body coords) = "
             f"{_vec(facts['w_point'])}",
             f"  scene.e_point (selected point on {a}, {a} body coords) = "
             f"{_vec(facts['e_point'])}"]
    lines += [f"  scene.points[{k!r}] (body coords) = {_vec(v)}"
              for k, v in sorted(facts["points_body"].items())]
    lines += [f"  scene.dirs[{k!r}] (body coords) = {_vec(v)}"
              for k, v in sorted(facts["dirs_body"].items()) if ".-" not in k]
    lines += [f"  scene.points_world[{k!r}] = {_vec(v)}"
              for k, v in sorted(facts["points_world"].items())]
    lines += [f"  scene.dirs_world[{k!r}] = {_vec(v)}"
              for k, v in sorted(facts["dirs_world"].items()) if ".-" not in k]
    lines.append(f"  scene.quantities: {q or '(none)'}")
    return "\n".join(lines)


# ------------------------------------------------------------------ prompts

# Sentences shared VERBATIM with the schema prompt (vlm.build_emission_prompt
# and vlm._ROT_ROW_TEXT["target"]), so a wording difference cannot pose as a
# medium difference. tests/test_code_authoring.py asserts each one still
# appears in the schema's system text.
SHARED_WITH_SCHEMA = {
    "dirs": ("Axes are SIGNED CANONICAL DIRECTIONS of an object: "
             "obj.+front, obj.-front, obj.+left, obj.-left, obj.+up, obj.-up. "
             "Each object licenses only the directions listed for it below; "
             "a '-' direction is the exact opposite of its '+'. world.z is up "
             "(gravity); world.-z is down."),
    "w": ("The constraint frame w is fixed by rule, not emitted: z = the "
          "passive object's +up, x = its +front, origin = the interaction "
          "point selected on it; it is frozen at stage entry."),
    "free": "Degrees of freedom no row mentions are FREE.",
    "goal": ("The subgoal is centered on the attitude that satisfies its rows, "
             "reached by the smallest rotation from the entry attitude."),
    "rotate": ("A direction the stage rotates must be left free on the path "
               "\u2014 a path row on it forbids the rotation \u2014 and "
               "constrained only in the subgoal."),
    "limit": ("Rule-table limit per TSR: ONE row of either kind, or TWO aligning "
              "rows on independent active directions (which fully determine the "
              "attitude); an aligning row plus a perpendicular row is redundant "
              "at best (dropped) or rejected; two perpendicular rows are "
              "rejected."),
    "path": ("Path terms hold for the WHOLE motion from entry to subgoal, so the "
             "entry pose must satisfy them (the planner closes at most a 0.03 m "
             "gap at the start) and the path must contain the subgoal; leave "
             "rows the stage does not need free \u2014 clearance while moving "
             "comes from collision checking."),
    "hinge": ("The direction an object tilts about does not move during the "
              "tilt, so never 'point' it at a new target \u2014 that rotates the "
              "object about a different axis. Hold it where it already is, with "
              "the relation it already has: 'perpendicular_to' a reference it is "
              "perpendicular to at entry (a level hinge stays level), or "
              "'points' at a reference it already lies along (a vertical "
              "hinge). Then 'point' the direction that must end up along the "
              "target."),
}
_S = SHARED_WITH_SCHEMA

_COMMON = (
    "You are the constraint-authoring module. For one stage you write Python "
    "code that defines a path TSR (holds along the whole motion) and a "
    "subgoal TSR (goal region) for the ACTIVE (moving) object. A planner "
    "samples the subgoal, keeps samples that lie on the path, and plans "
    "with CBiRRT inside the path TSR; collision checking is separate. "
    "Here a 'term' or 'row' is a bound your code sets. "
    + _S["path"] + " " + _S["rotate"] + " The subgoal must bound x, y and "
    "z, since the planner samples it.\n\n" + _S["dirs"] + "\n\n")

_A3 = (
    _S["w"] + " Both TSRs pin the active object's selected feature point at "
    "w: translation bounds are where that feature may be, in w.\n\n"
    "Write `def author():` returning {\"path\": tsr(...), \"subgoal\": "
    "tsr(...)}. Available, with no imports (np and math are predefined):\n"
    "  point(name)         -> np.array, a passive-object point in w\n"
    "  active_point(name)  -> np.array, an active-object point in w at entry\n"
    "  feature             -> np.array, the active feature in w at entry\n"
    "  direction(name)     -> np.array, a unit direction in w at entry "
    "(active or passive direction, or 'world.z' / 'world.-z')\n"
    "  quantity(name)      -> float, a grounded quantity\n"
    "  points_to(axis, target, tol_deg)  aligning row: the active direction "
    "`axis` ends up pointing the way `target` points (a passive direction or "
    "world.z / world.-z), within tol_deg; rotation about it stays free\n"
    "  perpendicular_to(axis, reference, tol_deg)  perpendicular row: `axis` "
    "ends up perpendicular to `reference`, within tol_deg (one tilt)\n"
    "  tsr(x=None, y=None, z=None, rot=())  x, y, z are (lo, hi) in meters "
    "in w, or None for free; either end may be None for a half-open path "
    "row. rot is a list of rotation rows.\n"
    "Rotation is stated ONLY through points_to and perpendicular_to "
    "('points' below means points_to). " + _S["free"] + " " + _S["goal"]
    + " " + _S["limit"] + " " + _S["hinge"] + "\n\n")

_A4 = (
    "You write the TSRs as numbers. A TSR is (T0_w, Tw_e, Bw): T0_w is the "
    "4x4 world pose of a reference frame w (any frame you choose); Tw_e is "
    "the 4x4 offset from w to the active object's BODY frame e at zero "
    "displacement; Bw is a 6x2 array of [lo, hi] bounds on the displacement "
    "(x, y, z in meters, roll, pitch, yaw in radians) expressed in w. A "
    "candidate body pose T0_e has displacement d from Tw = inv(T0_w) @ T0_e "
    "@ inv(Tw_e): d[:3] = Tw[:3, 3], d[3:] = the extrinsic xyz Euler angles "
    "of Tw[:3, :3] (rotation = Rz(yaw) @ Ry(pitch) @ Rx(roll)). The pose is "
    "in the TSR when every d[i] is in Bw[i]. Free translation is "
    "(-np.inf, np.inf); free rotation is (-np.pi, np.pi). Example: if "
    "T_ef is a 4x4 pose in the active body frame whose origin is a feature "
    "point p, then Tw_e = np.linalg.inv(T_ef) makes that feature frame "
    "coincide with w at zero displacement: p sits at w's origin and the "
    "body attitude is T0_w[:3, :3] @ T_ef[:3, :3].T.\n\n"
    "Write `def author(scene):` returning {\"path\": {\"T0_w\": 4x4, "
    "\"Tw_e\": 4x4, \"Bw\": 6x2}, \"subgoal\": {...same...}} (numpy arrays "
    "or nested lists). `scene` has: T_active, T_passive (4x4 world body "
    "poses at entry), w_default (4x4, the rule-based frame drawn dashed in "
    "the images: z = passive +up, x = passive +front, origin = the selected "
    "point on the passive), w_point (selected passive point, passive body "
    "coords), e_point (selected active point, active body coords), "
    "points[name] and dirs[name] (body coords of each object's points and "
    "unit directions), points_world[name] and dirs_world[name] (the same in "
    "world coordinates at entry), quantities[name]. np, math and Rotation "
    "(scipy.spatial.transform.Rotation) are predefined; no imports.\n\n")

_FORMAT = (
    "Read values from the API at run time rather than copying numbers: the "
    "code is re-run at the actual entry pose when the stage is planned. "
    "Respond with exactly one ```python fenced block and nothing else. Its "
    "first line is a comment '# goal: <one short plain sentence saying "
    "where the stage must end, naming the objects' points and "
    "directions>'.")


def build_code_prompt(stage: StageSpec, vocab: Vocabulary, medium: str,
                      facts: dict, selection=None,
                      view_paths: list[Path] | None = None,
                      frames: dict | None = None, entry: str | None = None
                      ) -> tuple[str, list]:
    """System text: shared preamble + the medium's API + format. User turn:
    the same stage line, entry line, legend and renders as the schema
    prompt, plus the grounded values."""
    if medium not in MEDIA:
        raise ValueError(f"unknown medium {medium!r}")
    system = (_COMMON + (_A3 if medium == "constrained-code" else _A4)
              + f"Grounded symbols:\n{vocab.describe_directions()}\n\n"
              + _FORMAT)
    text = (f"Write the TSR pair for stage {stage.index} ({stage.name}): "
            f"active={stage.active}, passive={stage.passive}.")
    if selection:
        text += f" Selected interaction point candidate {selection.candidate_id}."
    if entry:
        text += "\n\n" + entry
    text += "\n\n" + describe_facts(facts, medium)
    if frames is not None:
        text += "\n\n" + frames_legend(stage, frames)
    parts = [_text(text)] + [image_block(p) for p in (view_paths or [])]
    return system, [{"role": "user", "content": parts}]


def _ask_code(client, system: str, messages: list, parse) -> CodeEmission:
    """Client._ask with a code-shaped repair turn: _ask's says "Re-emit the
    corrected JSON object", which contradicts the code format. Same budget
    (vlm.MAX_PARSE_RETRIES), payload, transport and CallLog as _ask, so the
    two media are logged and retried alike; vlm.py is left untouched."""
    log = CallLog("emit_code")
    client.logs.append(log)
    msgs = list(messages)
    for _ in range(1 + vlm.MAX_PARSE_RETRIES):
        log.attempts += 1
        payload = {"model": client.model, "max_tokens": client.max_tokens}
        if vlm.TEMPERATURE is not None:
            payload["temperature"] = vlm.TEMPERATURE
        raw = client._transport({**payload, "system": system, "messages": msgs})
        try:
            result = parse(raw)
        except ParseRejection as rej:
            log.rejections.append(rej.reason)
            msgs = msgs + [
                {"role": "assistant", "content": [_text(raw)]},
                {"role": "user", "content": [_text(
                    f"Your response was rejected: {rej.reason}. Re-emit the "
                    "corrected ```python block only.")]}]
            continue
        log.raw = raw
        return result
    raise VLMError(f"emit_code: parse retry budget exhausted after "
                   f"{log.attempts} attempts; rejections: {log.rejections}")


def emit_code(client, stage: StageSpec, vocab: Vocabulary, medium: str,
              facts: dict, selection=None, view_paths=None, frames=None,
              entry=None, rejections: list[tuple[str, str]] | None = None
              ) -> CodeEmission:
    """One call #3 in a code medium (parse-retry loop and call log as
    the schema's, see _ask_code). `rejections`: (raw, reason) pairs
    from earlier attempts at this stage, replayed so the model repairs its
    own code."""
    system, messages = build_code_prompt(stage, vocab, medium, facts,
                                         selection, view_paths, frames, entry)
    for raw, r in rejections or []:
        messages = messages + [
            {"role": "assistant", "content": [_text(raw)]},
            {"role": "user", "content": [_text(
                f"That code was rejected: {r}. Re-emit the corrected code "
                "block only, changing as little as needed.")]}]
    em = _ask_code(client, system, messages,
                   lambda raw: parse_code_response(raw, stage, medium))
    log = client.logs[-1]
    log.views = len(view_paths or [])
    log.legend = frames is not None
    log.entry = entry is not None
    return em


# ------------------------------------------------------------------ sandbox

_CHILD = r'''
import json, math, sys, traceback
import numpy as np

TAG = "@@author-result@@"   # = _TAG in the parent
req = json.loads(sys.stdin.read())
F, medium = req["facts"], req["medium"]
SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
        for k in ("abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
                  "float", "int", "bool", "list", "dict", "tuple", "set", "round",
                  "sorted", "reversed", "map", "filter", "any", "all", "isinstance",
                  "pow", "divmod", "print", "str", "repr", "ValueError",
                  "TypeError", "KeyError", "IndexError", "Exception",
                  "ZeroDivisionError", "ArithmeticError", "None", "True", "False")
        if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))}

def _lookup(table, name, what):
    if name not in table:
        raise KeyError(f"unknown {what} {name!r}; known: {sorted(table)}")
    return table[name]

import builtins as _b
def _import(name, globals=None, locals=None, fromlist=(), level=0):
    # numpy's C methods (ndarray.max, .min, .sum, ...) import their Python
    # helpers lazily through the CALLER's __import__; allow numpy / scipy /
    # math internals only. Model code cannot call this: the parent's AST
    # check bans import statements and double-underscore names.
    root = (name or "").split(".")[0]
    if level == 0 and root in ("numpy", "scipy", "math"):
        return _b.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not allowed")
SAFE["__import__"] = _import
def _print(*a, **k):
    k["file"] = sys.stderr          # stdout carries only the result line
    print(*a, **k)
SAFE["print"] = _print
ns = {"__builtins__": SAFE, "np": np, "math": math}
if medium == "constrained-code":
    def point(name):
        return np.array(_lookup(F["points_w"], name, "point"), float)
    def direction(name):
        return np.array(_lookup(F["dirs_w"], name, "direction"), float)
    def quantity(name):
        return float(_lookup(F["quantities"], name, "quantity"))
    def _tol(t):
        t = float(t)
        if not (0.0 < t <= 180.0):
            raise ValueError(f"tol_deg must be in (0, 180], got {t}")
        return t
    def points_to(axis, target, tol_deg):
        return {"axis": str(axis), "points": str(target), "tol_deg": _tol(tol_deg)}
    def perpendicular_to(axis, reference, tol_deg):
        return {"axis": str(axis), "perpendicular_to": str(reference), "tol_deg": _tol(tol_deg)}
    def _row(v, k):
        if v is None:
            return None
        if len(v) != 2:
            raise ValueError(f"{k} must be (lo, hi) or None")
        lo = None if v[0] is None else float(v[0])
        hi = None if v[1] is None else float(v[1])
        for e in (lo, hi):
            if e is not None and not math.isfinite(e):
                raise ValueError(f"{k} bounds must be finite or None")
        if lo is not None and hi is not None and hi < lo:
            raise ValueError(f"{k} = ({lo}, {hi}): hi < lo")
        return [lo, hi]
    def tsr(x=None, y=None, z=None, rot=()):
        rot = list(rot)
        for r in rot:
            if not (isinstance(r, dict) and "tol_deg" in r):
                raise ValueError("rot rows must come from points_to / perpendicular_to")
        return {"x": _row(x, "x"), "y": _row(y, "y"), "z": _row(z, "z"), "rot": rot}
    def active_point(name):
        return np.array(_lookup(F["active_points_w"], name, "active point"), float)
    ns.update(point=point, active_point=active_point, direction=direction, quantity=quantity,
              points_to=points_to, perpendicular_to=perpendicular_to, tsr=tsr,
              feature=np.array(F["feature_w"], float))
    args = ()
else:
    from scipy.spatial.transform import Rotation
    class _Scene:
        pass
    s = _Scene()
    s.T_active = np.array(F["T_active"], float)
    s.T_passive = np.array(F["T_passive"], float)
    s.w_default = np.array(F["w_default"], float)
    s.w_point = np.array(F["w_point"], float)
    s.e_point = np.array(F["e_point"], float)
    s.points = {k: np.array(v, float) for k, v in F["points_body"].items()}
    s.dirs = {k: np.array(v, float) for k, v in F["dirs_body"].items()}
    s.points_world = {k: np.array(v, float) for k, v in F["points_world"].items()}
    s.dirs_world = {k: np.array(v, float) for k, v in F["dirs_world"].items()}
    s.quantities = dict(F["quantities"])
    ns["Rotation"] = Rotation
    args = (s,)

def _out(o):
    if isinstance(o, np.ndarray):
        return o.astype(float).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _out(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_out(v) for v in o]
    return o

try:
    exec(compile(req["source"], "<author>", "exec"), ns)
    res = ns["author"](*args)
    sys.stdout.write(TAG + json.dumps({"ok": True, "result": _out(res)}) + "\n")
except BaseException as e:
    tb = traceback.extract_tb(e.__traceback__)
    lines = [f.lineno for f in tb if f.filename == "<author>"]
    where = f" (line {lines[-1]})" if lines else ""
    sys.stdout.write(TAG + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}{where}"}) + "\n")
'''


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (CHILD_CPU_S, CHILD_CPU_S))
    resource.setrlimit(resource.RLIMIT_AS, (CHILD_MEM_B, CHILD_MEM_B))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


_TAG = "@@author-result@@"
_RESULTS: dict[str, object] = {}     # (source, medium, facts) -> result, per process


def run_source(source: str, medium: str, facts: dict, slot: str) -> object:
    """Run `author` in the restricted child; its return value as JSON data.
    CodeError (slot-named) on any failure, with the model's line number.
    Results are cached per process by (source, medium, facts): the planners
    compile the same stage at the same pose more than once (lookahead,
    goal ranking, re-anchoring), and each run is a fresh interpreter."""
    check_source(source, entry_name(medium))
    req = json.dumps({"source": source, "medium": medium, "facts": facts},
                     sort_keys=True)
    if req in _RESULTS:
        return copy.deepcopy(_RESULTS[req])
    with tempfile.TemporaryDirectory(prefix="author-") as tmp:
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            p = subprocess.run([sys.executable, "-I", "-B", "-c", _CHILD],
                               input=req, capture_output=True, text=True, cwd=tmp,
                               env=env, timeout=CHILD_WALL_S, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            raise CodeError(slot, f"the code did not finish within {CHILD_WALL_S} s")
    lines = [l[len(_TAG):] for l in p.stdout.splitlines() if l.startswith(_TAG)]
    out = None
    if lines:
        try:
            out = json.loads(lines[-1])
        except json.JSONDecodeError:
            out = None
    if not isinstance(out, dict) or "ok" not in out:
        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
        why = ("CPU limit exceeded" if p.returncode in (-9, -24, 152)
               else "memory limit exceeded" if "MemoryError" in p.stderr
               else err)
        raise CodeError(slot, f"the code crashed: {why}")
    if not out["ok"]:
        raise CodeError(slot, f"running author raised {out['error']}")
    _RESULTS[req] = copy.deepcopy(out["result"])
    return out["result"]


# ------------------------------------------------------- constrained-code

_NUM = re.compile(r"^deg:")
# what a malformed return value raises while being read (ragged lists,
# strings where numbers belong, missing keys): reported as a CodeError
_SHAPE_ERRORS = (TypeError, ValueError, KeyError, AttributeError, IndexError)


@contextmanager
def numeric_tols(keys):
    """Make 'deg:<x>' tolerance keys resolvable by compile_stage for the
    duration of a compile. Keys already present are left alone; added keys
    are removed afterwards, so the frozen enum table is unchanged outside.
    Deliberate: it lets A3 use compile_stage exactly as A0 does without
    changing compile_tsr (which hardware imports). Not reentrant; every
    caller here is single-threaded."""
    added = [k for k in set(keys) if k not in ROT_TOL_RAD]
    for k in added:
        ROT_TOL_RAD[k] = float(np.deg2rad(float(k[4:])))
    try:
        yield
    finally:
        for k in added:
            ROT_TOL_RAD.pop(k, None)


def _fmt(v: float) -> str:
    return f"{v:.9f}"          # _eval_expr's tokenizer reads no exponents


def lower_constrained(result: object, em: CodeEmission, vocab: Vocabulary
                      ) -> tuple[StageEmission, dict]:
    """author()'s return value -> (StageEmission for compile_stage, the
    slot map that renames lowered slots back to the model's terms)."""
    if not (isinstance(result, dict) and set(result) == {"path", "subgoal"}):
        raise CodeError(f"{em.name}", "author() must return {'path': tsr(...), "
                                      "'subgoal': tsr(...)}")
    specs, slot_map = {}, {}
    for ctx in ("path", "subgoal"):
        t = result[ctx]
        if not (isinstance(t, dict) and {"x", "y", "z", "rot"} <= set(t)):
            raise CodeError(f"{em.name}.{ctx}", f"{ctx} must be built with tsr(...)")
        rows = []
        for r in t["rot"]:
            r = dict(r)
            tol = float(r.pop("tol_deg"))
            rows.append((r, f"deg:{tol:.6g}"))
        try:
            parsed = _parse_rot_rows([{**r, "tol": "tight"} for r, _ in rows],
                                     vocab, f"{em.name}.{ctx}")
        except ParseRejection as e:
            raise CodeError(f"{em.name}.{ctx}.rot", e.reason)
        rot = tuple(RotRow(axis=p.axis, relation=p.relation, reference=p.reference,
                           tol=tol) for p, (_, tol) in zip(parsed, rows))
        trans = []
        for k in ("x", "y", "z"):
            v = t[k]
            if v is None:
                continue
            lo, hi = v
            if ctx == "subgoal" and (lo is None or hi is None):
                raise CodeError(f"{em.name}.subgoal.{k}", (
                    f"subgoal {k} is half-open; the planner samples the "
                    "subgoal, so give both ends"))
            slot_map[f"{em.name}.{ctx}.trans[{len(trans)}]"] = f"{em.name}.{ctx}.{k}"
            trans.append(TransTerm(term="expr", row=k,
                                   expr_lo=_fmt(-UNBOUNDED_M if lo is None else lo),
                                   expr_hi=_fmt(UNBOUNDED_M if hi is None else hi)))
        specs[ctx] = TSRSpec(rot=rot, trans=tuple(trans) or (TransTerm(term="free"),))
    return (StageEmission(stage=em.stage, name=em.name, active=em.active,
                          passive=em.passive, path_tsr=specs["path"],
                          subgoal_tsr=specs["subgoal"], verify="", goal=em.goal),
            slot_map)


def _renamed(e: CompileError, slot_map: dict) -> CodeError:
    def fix(s: str) -> str:
        for a, b in slot_map.items():
            s = s.replace(a, b)
        return s
    flat = [CodeError(fix(x.slot), fix(x.reason)) for x in e.all()]
    return CodeError(flat[0].slot, flat[0].reason, tuple(flat[1:]))


def _tol_keys(lowered: StageEmission) -> list[str]:
    return [r.tol for spec in (lowered.path_tsr, lowered.subgoal_tsr)
            for r in spec.rot if r.tol and _NUM.match(r.tol)]


# ------------------------------------------------------------- free-code

def _as_pose(v, slot: str, what: str) -> np.ndarray:
    T = np.asarray(v, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise CodeError(slot, f"{what} must be a finite 4x4 array")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise CodeError(slot, f"{what}'s last row must be [0, 0, 0, 1]")
    Rm = T[:3, :3]
    err = float(np.abs(Rm.T @ Rm - np.eye(3)).max())
    if err > 1e-4 or np.linalg.det(Rm) < 0:
        raise CodeError(slot, (f"{what}'s upper-left 3x3 is not a rotation "
                               f"(|R^T R - I| = {err:.2g}, det = "
                               f"{np.linalg.det(Rm):+.3f})"))
    U, _, Vt = np.linalg.svd(Rm)
    T = T.copy()
    T[:3, :3] = U @ Vt
    return T


def _as_bw(v, slot: str) -> np.ndarray:
    B = np.asarray(v, float)
    if B.shape != (6, 2) or np.isnan(B).any():
        raise CodeError(slot, "Bw must be a 6x2 array of [lo, hi] without NaN")
    if not np.all(np.isfinite(B[3:])):
        raise CodeError(slot, "rotation bounds must be finite (free is "
                              "(-np.pi, np.pi))")
    bad = [("x", "y", "z", "roll", "pitch", "yaw")[i] for i in range(6)
           if B[i, 1] < B[i, 0]]
    if bad:
        raise CodeError(slot, f"Bw rows {bad} have hi < lo")
    return B


def build_free(result: object, em: CodeEmission, body_poses) -> tuple[TSR, TSR, Frame]:
    if not (isinstance(result, dict) and set(result) == {"path", "subgoal"}):
        raise CodeError(em.name, "author(scene) must return {'path': {...}, "
                                 "'subgoal': {...}}")
    tsrs = {}
    for ctx in ("path", "subgoal"):
        t = result[ctx]
        slot = f"{em.name}.{ctx}"
        if not (isinstance(t, dict) and {"T0_w", "Tw_e", "Bw"} <= set(t)):
            raise CodeError(slot, f"{ctx} must have T0_w, Tw_e and Bw")
        tsrs[ctx] = TSR(T0_w=_as_pose(t["T0_w"], slot, f"{ctx} T0_w"),
                        Tw_e=_as_pose(t["Tw_e"], slot, f"{ctx} Tw_e"),
                        Bw=_as_bw(t["Bw"], slot), name=f"{em.name}/{ctx}(code)")
    open_rows = [k for k in "xyz" if not np.all(np.isfinite(tsrs["subgoal"].Bw["xyz".index(k)]))]
    if open_rows:
        raise CodeError(f"{em.name}.subgoal", (
            f"subgoal row(s) {open_rows} are unbounded; the planner samples the "
            "subgoal, so bound x, y and z"))
    T_bw = np.linalg.inv(body_poses[em.passive]) @ tsrs["path"].T0_w
    w = Frame(name=f"{em.passive}.authored", point=T_bw[:3, 3].copy(),
              axis=T_bw[:3, 2].copy(), secondary=T_bw[:3, 0].copy())
    return tsrs["path"], tsrs["subgoal"], w


# --------------------------------------------------------------- dispatch

def compile_any(emission, symbols: dict[str, Symbols],
                body_poses: dict[str, np.ndarray], w_point, e_point=None,
                entry_synthesized: bool = False, vocab: Vocabulary | None = None
                ) -> CompiledStage:
    """compile_stage for a StageEmission; for a CodeEmission, run its code
    at these entry poses and ground the result with the same checks. The
    one call site both the emit gate and plan_task.py use."""
    if isinstance(emission, StageEmission):
        return compile_stage(emission, symbols, body_poses, w_point=w_point,
                             e_point=e_point, entry_synthesized=entry_synthesized)
    em = emission
    if em.passive is None:
        raise CodeError(em.name, "code media author object-mover stages only")
    for o in (em.active, em.passive):
        if o not in body_poses or o not in symbols:
            raise CodeError(em.name, f"no symbols or entry pose for {o!r}")
    facts = stage_facts(em.active, em.passive, symbols, body_poses, w_point, e_point)
    result = run_source(em.source, em.medium, facts, em.name)
    if em.medium == "constrained-code":
        vocab = vocab or Vocabulary.from_symbols(symbols)
        try:
            lowered, slot_map = lower_constrained(result, em, vocab)
        except _SHAPE_ERRORS as e:
            raise CodeError(em.name, f"author() returned an unexpected value: "
                                     f"{type(e).__name__}: {e}")
        try:
            with numeric_tols(_tol_keys(lowered)):
                cs = compile_stage(lowered, symbols, body_poses, w_point=w_point,
                                   e_point=e_point, entry_synthesized=entry_synthesized)
        except CompileError as e:
            raise _renamed(e, slot_map)
        return cs
    try:
        path, subgoal, w = build_free(result, em, body_poses)
    except _SHAPE_ERRORS as e:
        raise CodeError(em.name, f"author(scene) returned an unexpected value: "
                                 f"{type(e).__name__}: {e}")
    notes: list[str] = []
    T_entry = None if entry_synthesized else body_poses[em.active]
    # compile_tsr's containment check compares path and subgoal rows
    # directly, which is right only when both share T0_w (always so in the
    # schema). With two frames, check the entry alone here; whether subgoal
    # samples lie on the path is check_pair_consistency's sampling test,
    # which is frame-free.
    same_w = np.allclose(path.T0_w, subgoal.T0_w, atol=1e-9)
    errs = _check_path_containment(em.name, path, subgoal if same_w else path,
                                   _Box(), T_entry, notes)
    if not same_w:
        notes.append(f"{em.name}: path and subgoal use different frames; their "
                     "overlap is checked by sampling, not row by row")
    if errs:
        flat = [x for e in errs for x in e.all()]
        raise CodeError(flat[0].slot, flat[0].reason,
                        tuple(CodeError(x.slot, x.reason) for x in flat[1:]))
    if entry_synthesized:
        notes.append(f"{em.name}.path: entry synthesized; entry containment "
                     "not checked")
    return CompiledStage(stage=em.stage, name=em.name, path=path, subgoal=subgoal,
                         w_frame=w, notes=tuple(notes), x_source="authored")


def check_seam(prev: CompiledStage, cur: CompiledStage) -> list[str]:
    """compile_tsr.check_stage_seam for a code stage: the same sampled test
    (every goal of `prev` inside `cur`'s path, translation rows), with the
    blame on the path as a whole since there are no schema terms to name."""
    stub = SimpleNamespace(path_tsr=SimpleNamespace(trans="free"))
    try:
        return check_stage_seam(prev, cur, stub)
    except CompileError as e:
        raise CodeError(f"{cur.name}.path", e.reason)


# ---------------------------------------------------------------- artifacts

def emission_from_any(d: dict):
    if d.get("medium") in MEDIA:
        return CodeEmission(**{k: d[k] for k in ("stage", "name", "active",
                                                 "passive", "medium", "source")},
                            goal=d.get("goal", ""))
    return emission_from_json(d)


def load_any_emissions(path) -> dict:
    """vlm.load_emissions for a schema artifact; for a code artifact the same
    refusal rule (any failed gate) and role keying, with CodeEmissions."""
    doc = json.loads(Path(path).read_text())
    if (doc.get("medium") not in MEDIA
            and not any(e.get("medium") in MEDIA for e in doc.get("emissions", []))):
        return load_emissions(path)
    bad = [c["stage"] for c in doc.get("compiled", []) if not c["grounded"]]
    if bad:
        raise SystemExit(f"[authoring] {path}: gate failed for {bad}; refusing "
                         "to plan on it")
    return {r: emission_from_any(e) for r, e in zip(doc["roles"], doc["emissions"])}


def as_record(em) -> dict:
    return asdict(em)
