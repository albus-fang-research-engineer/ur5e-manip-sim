"""The repair study's loop pieces in sim (docs/deferred.md D9): compile a
specification's stage chain, build what the critic (VLM call #4) and the
OmniManip-style checker (render_check) see, and caption the preview images.
The driver is scripts/run_repair_case.py.

Ported from the hardware critic loop (ur5e-manip-hardware,
outputs/runs/critic_scene.py: compile_chain, preview_inputs, preview_report,
caption_for, views_from_manifest), with the hardware parts removed: no
measured poses (the sim's settled spawn poses), no taught grasp (the sim's
grasp is found by the planner, so the preview draws no grasp state; the
grasp is context, never editable), no compat imports. The sim must not
depend on the hardware repo, so this is a port, not an import; hardware
can later import these through its critic_compat instead of keeping its own.

The chain is compiled the way the emit gate and plan_task's lookahead do:
the first mover at its spawn pose, each later mover against the previous
mover's subgoal.nominal(). A stage that does not compile ends the chain
for the stages after it (they have no entry pose).
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import critic as cr
from .compile_tsr import CompileError, check_pair_consistency
from .selection import ResolutionError, load_pool, selected_points

_VIEW_NAMES = {"iso": "iso", "iso-opp": "iso (opposite side)", "top": "top",
               "+x": "side (+x)", "-x": "side (-x)", "+y": "side (+y)", "-y": "side (-y)"}
_TRANS_ROWS = {"x", "y", "z"}


# ------------------------------------------------------------------ the spec as the critic edits it

def role_index_for(spec, bound_index: dict | None = None) -> dict[str, int]:
    """role -> stage index. A stage plan's bindings win (insert marker);
    otherwise (pour tea, fixed roles) the emissions' order, with a role
    '<x>_passive' sharing its '<x>_active' stage."""
    if bound_index:
        return dict(bound_index)
    idx = {r: i for i, r in enumerate(spec.roles)}
    for r in spec.selections:
        if r not in idx and r.endswith("_passive"):
            act = r[: -len("_passive")] + "_active"
            if act in idx:
                idx[r] = idx[act]
    return idx


def spec_state(spec, asset_dirs: dict, offered=None) -> cr.SpecState:
    pools = {o: load_pool(asset_dirs[o]) for o in sorted(set(spec.objects.values()))}
    return cr.SpecState(emissions={r: spec.emissions[r] for r in spec.roles},
                        selections=dict(spec.selections), objects=dict(spec.objects),
                        pools=pools, offered=offered)


# ------------------------------------------------------------------ the compiled chain

@dataclass
class ChainStage:
    role: str
    emission: object                   # StageEmission
    poses: dict                        # every object's pose at this stage's entry
    w_point: np.ndarray | None = None
    e_point: np.ndarray | None = None
    compiled: object | None = None     # CompiledStage
    error: str | None = None
    compile_error: object | None = None
    synthesized: bool = False
    no_entry: bool = False


def compile_chain(state: cr.SpecState, roles, poses: dict, symbols: dict, asset_dirs: dict,
                  role_index: dict, synthesized=frozenset()) -> list[ChainStage]:
    from .code_authoring import compile_any
    out, cur, broken = [], dict(poses), None
    for role in roles:
        em = state.emissions.get(role)
        if em is None:
            continue
        rec = ChainStage(role, em, dict(cur), synthesized=role in synthesized)
        out.append(rec)
        if em.passive is None:
            continue                     # the grasp: context; the planner never compiles it
        if broken is not None:
            rec.error = (f"not compiled: stage {broken!r} did not compile, so this stage "
                         "has no entry pose")
            rec.no_entry = True
            continue
        try:
            rec.w_point, rec.e_point = selected_points(
                role, em, state.selections, state.objects, role_index, asset_dirs, symbols)
        except (SystemExit, ResolutionError) as e:
            rec.error = f"anchor points do not resolve: {e}"
        else:
            try:
                cs = compile_any(em, symbols, rec.poses, w_point=rec.w_point,
                                 e_point=rec.e_point, entry_synthesized=rec.synthesized)
                check_pair_consistency(cs)
                rec.compiled = cs
            except CompileError as e:
                rec.error = f"compile rejected: {e.text()}"
                rec.compile_error = e
        if rec.compiled is None:
            broken = em.name
        else:
            cur = {**cur, em.active: np.asarray(rec.compiled.subgoal.nominal(), float)}
    return out


def preview_inputs(state: cr.SpecState, chain: list[ChainStage], symbols: dict) -> list:
    context = state.context_roles()
    return [cr.StagePreviewInput(
        role=c.role, emission=c.emission, compiled=c.compiled, poses=c.poses, symbols=symbols,
        w_point=np.zeros(3) if c.w_point is None else c.w_point, e_point=c.e_point,
        editable=c.role not in context) for c in chain]


def _uncompiled_record(c: ChainStage, state, editable: bool) -> dict:
    em = c.emission
    rec = {"stage": em.name, "role": c.role, "active": em.active, "passive": em.passive,
           "editable": editable, "goal": em.goal, "compile_error": c.error}
    if state.objects.get(c.role) == em.active and c.role in state.selections:
        rec["point_choice"] = {"pointer": f"{em.name}.point.{em.active}",
                               "current": state.point_of(c.role)}
    if not editable:                     # context (the grasp): trimmed, as stage_report trims it
        rec.pop("passive")
        if rec["compile_error"] is None:
            rec.pop("compile_error")
        return rec
    for ctx_name, spec in (("path", em.path_tsr), ("subgoal", em.subgoal_tsr)):
        rows = [{"pointer": f"{em.name}.{ctx_name}.rot[{i}]", "text": cr.row_text(r)}
                for i, r in enumerate(spec.rot)]
        rows += [{"pointer": f"{em.name}.{ctx_name}.trans[{i}]", "text": cr.row_text(t)}
                 for i, t in enumerate(spec.trans)]
        rec[ctx_name] = {"rows": rows}
    return rec


def preview_report(state, chain: list[ChainStage], previews: list) -> dict:
    """build_preview_report over the compiled stages, with a record for each
    stage that did not compile spliced in at its place (its rows as text and
    the compile error)."""
    ok = [p for p, c in zip(previews, chain) if c.compiled is not None]
    rep = cr.build_preview_report(ok, state)
    by_role = {r["role"]: r for r in rep["stages"]}
    rep["stages"] = [by_role[c.role] if c.compiled is not None
                     else _uncompiled_record(c, state, p.editable)
                     for p, c in zip(previews, chain)]
    return rep


def stage_rows(em) -> tuple[str, ...]:
    """'<pointer>: <row text>' for every row of a stage, path then subgoal
    (the render_check condition's "planned constraint")."""
    out = []
    for ctx_name, spec in (("path", em.path_tsr), ("subgoal", em.subgoal_tsr)):
        out += [f"{em.name}.{ctx_name}.rot[{i}]: {cr.row_text(r)}" for i, r in enumerate(spec.rot)]
        out += [f"{em.name}.{ctx_name}.trans[{i}]: {cr.row_text(t)}"
                for i, t in enumerate(spec.trans)]
    return tuple(out)


# ------------------------------------------------------------------ rendering

@dataclass
class _Obj:
    name: str
    asset: Path

    @property
    def xml(self) -> Path:
        return self.asset / f"{self.name}.xml"


class PreviewScene:
    """What the renderer reads from a sim scene: per object an asset dir and
    the table slab."""

    def __init__(self, asset_dirs: dict, table_top_z: float, table_size):
        self.objects = {n: _Obj(n, Path(d)) for n, d in asset_dirs.items()}
        self.asset_dirs = {n: Path(d) for n, d in asset_dirs.items()}
        self.table_top_z = float(table_top_z)
        self.table_size = tuple(float(v) for v in table_size)

    def fixed_poses(self):
        raise RuntimeError("PreviewScene has no spawn poses: pass the poses")


def render_round(scene, chain: list[ChainStage], previews: list, image_dir: Path,
                 symbols: dict, gripper=None) -> dict:
    """Render the preview (GL at call time) and return the manifest. The
    passive's anchor point is labeled by name."""
    import scripts.render_preview as rp
    from scripts.render_stage_frames import point_token
    initial, stages, labels = rp.from_preview_inputs(previews)
    for c in chain:
        em = c.emission
        if em.passive and c.w_point is not None:
            tok = point_token(em.passive, symbols, c.w_point)
            if tok:
                pt = tok.split(".", 1)[1]
                labels.setdefault(em.passive, {})[pt] = np.asarray(
                    symbols[em.passive].points[pt], float)
    pscene = PreviewScene(scene.asset_dirs, scene.table_top_z, scene.table_size)
    return rp.render_preview(pscene, initial, stages, labels, image_dir, grasp=None,
                             gripper=gripper)


# ------------------------------------------------------------------ captions

def color_name(rgb) -> str:
    r, g, b = (float(v) for v in rgb[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.15:
        return "black"
    if s < 0.2:
        return "white" if v > 0.85 else "gray"
    for top, name in ((0.05, "red"), (0.11, "orange"), (0.19, "yellow"), (0.5, "green"),
                      (0.7, "blue"), (0.85, "purple"), (0.94, "pink"), (1.01, "red")):
        if h < top:
            return name
    return "red"


def object_colors(asset_dirs: dict) -> dict:
    out = {}
    for obj, d in asset_dirs.items():
        md = Path(d, "meshes")
        tex = sorted(md.glob("*_texture.png")) + sorted(md.glob("material_*.png")) \
            if md.is_dir() else []
        try:
            from PIL import Image
            a = np.asarray(Image.open(tex[0]).convert("RGB"), float).reshape(-1, 3) / 255.0
            a = a[a.max(1) > 0.08]
            out[obj] = color_name(np.median(a, 0)) if len(a) else "textured"
        except (IndexError, OSError, ImportError):
            out[obj] = "textured"
    return out


def _contact_text(contacts, verb: str = "collides with") -> str:
    import scripts.render_preview as rp
    return rp.contact_text(contacts or [], verb)


def _bodies(contacts) -> str:
    import scripts.render_preview as rp
    return rp._bodies(contacts or [])


def _legend(manifest: dict, img: dict, colors: dict) -> str:
    import scripts.render_preview as rp
    leg = img.get("legend") or manifest.get("legend") or {}
    merged = {**colors, **(leg if isinstance(leg, dict) else {})}
    if manifest.get("gripper") and "gripper" not in merged:
        merged["gripper"] = color_name(rp.GRIPPER_RGBA)
    return ", ".join(f"{k} = {v}" for k, v in merged.items())


def _path_kind(img: dict) -> str:
    g = img.get("group")
    if g in ("translation", "rotation"):
        return f"path {g} rows"
    rows = {r.get("row") for r in img.get("rows") or []}
    if rows and rows <= _TRANS_ROWS:
        return "path translation rows"
    if rows and not rows & _TRANS_ROWS:
        return "path rotation rows"
    return "path region rows"


def _path_panels(img: dict) -> tuple[list[str], list[str]]:
    drawn, dropped = [], []
    for row in img.get("rows") or []:
        view = _VIEW_NAMES.get(row.get("view"), row.get("view"))
        caps = row.get("captions") or []
        panels = row.get("panels") if isinstance(row.get("panels"), list) else []
        for j, cap in enumerate(caps):
            p = panels[j] if j < len(panels) and isinstance(panels[j], dict) else {}
            if p.get("dropped"):
                why = _contact_text(p.get("collisions")) or "in collision"
                dropped.append(f"{p.get('desc') or cap}: {why} (the planner avoids this)")
                continue
            note = ""
            if p.get("collides"):
                b = _bodies(p.get("collisions"))
                note = f" -- in collision with {b}" if b else " -- in collision"
            drawn.append(f"{cap}" + (f" ({view} view)" if view else "") + note)
    return drawn, dropped


def _stage_label(c) -> str:
    return f'stage "{c.emission.name}" (role {c.role})'


def caption_for(img: dict, manifest: dict, k: int, n: int, chain: list, colors: dict,
                extra: list[str] | None = None) -> str:
    """The text block placed before image k (1-based) of n."""
    by_role = {c.role: c for c in chain}
    lines = []
    if img.get("kind") == "state":
        st = str(img.get("state", ""))
        if st == "initial":
            head = "initial scene (spawn poses, before any stage)"
        else:
            role = img.get("role") or st.split(":", 1)[-1]
            c = by_role.get(role)
            shown = (img.get("search") or {}).get("shown")
            where = ("the nearest collision-free pose of its subgoal region (the region's "
                     "center collides)" if shown == "nearest_free" else
                     "the center of its subgoal region (no collision-free pose found)"
                     if shown == "none_free" else "the center of its subgoal region")
            head = (f"stage goal -- {_stage_label(c)}: the {c.emission.active} at {where}"
                    if c else f"stage goal (role {role})")
        lines.append(f"Image {k} of {n}: {head}.")
        if img.get("panels"):
            lines.append("Panels, left to right: " + "; ".join(
                f"{_VIEW_NAMES.get(p, p)} view" for p in img["panels"]) + ".")
        search = img.get("search")
        if "collisions" in img and img["collisions"] is not None:
            txt = _contact_text(img["collisions"], "intersects")
            label = ("Collisions at the subgoal center" if search and search.get("nominal_collides")
                     else "Collisions")
            lines.append(f"{label}: {txt}." if txt else "Collisions: none.")
        if search and search.get("nominal_collides"):
            lines += [f"Renderer: {nt}." for nt in img.get("notes") or []]
    else:
        role = img.get("stage")
        c = by_role.get(role)
        label = _stage_label(c) if c else f'stage "{img.get("name", role)}" (role {role})'
        mover = c.emission.active if c else "object"
        lines.append(f"Image {k} of {n}: {_path_kind(img)} -- {label}: the {mover} with one "
                     "path row at a time at its bound, the other rows at the center pose "
                     "(midpoint of the stage's entry and goal, inside the path region).")
        col = colors.get(mover)
        lines.append(f"Moving object: {mover}" + (f" ({col})" if col and col != "textured" else "")
                     + ". No point labels in path images; every object keeps its color from "
                     "the state images.")
        drawn, dropped = _path_panels(img)
        if drawn:
            lines.append("Panels, left to right then top to bottom:")
            lines += [f"  {i + 1}. {d}" for i, d in enumerate(drawn)]
        if img.get("notes") is not None:
            own = [nt for nt in img["notes"] if not nt.endswith("(the planner avoids this)")]
            if own:
                lines.append("Renderer: " + "; ".join(own) + ".")
        else:
            center = (img.get("representative") or {}).get("collisions")
            if img.get("center_collides") or center:
                txt = _contact_text(center)
                lines.append("The center pose itself is in collision"
                             + (f" ({txt})" if txt else "")
                             + ": no collision-free pose to show, so every panel is kept.")
        if dropped:
            lines.append("Not drawn: " + "; ".join(dropped) + ".")
        if img.get("pinned_rows"):
            lines.append("Pinned rows (fixed, not drawn): " + ", ".join(img["pinned_rows"]) + ".")
    lines += extra or []
    leg = _legend(manifest, img, colors)
    if leg:
        lines.append(f"Colors: {leg}.")
    return "\n".join(lines)


def _text_only_lines(rec: dict) -> str:
    _, dropped = _path_panels(rec)
    return (f"{_path_kind(rec).capitalize()} of this stage: no image, every panel collides "
            "while the center pose is free -- " + "; ".join(dropped) + ".")


def views_from_manifest(manifest: dict, image_dir: Path, chain: list,
                        colors: dict) -> list[tuple[str, Path]]:
    """(caption, image) pairs in the renderer's order, for the critic."""
    imgs = manifest.get("images") or []
    extra: dict[str, list[str]] = {}
    for rec in manifest.get("path_text_only") or []:
        extra.setdefault(rec.get("stage"), []).append(_text_only_lines(rec))
    out = []
    for k, img in enumerate(imgs):
        more = []
        if img.get("kind") == "state" and str(img.get("state", "")).startswith("goal:"):
            more = extra.get(img.get("role") or img["state"].split(":", 1)[1], [])
        if k == 0 and manifest.get("collision_check"):
            chk = manifest["collision_check"]
            more = more + [f"Collision checks in these images: {chk.get('bodies', 'scene')}"
                           + (f", tolerance {chk['tol_mm']} mm" if chk.get("tol_mm") is not None
                              else "") + "."]
        out.append((caption_for(img, manifest, k + 1, len(imgs), chain, colors, more),
                    Path(image_dir) / img["file"]))
    return out


def withheld_views(manifest: dict, image_dir: Path, chain: list,
                   colors: dict) -> list[tuple[str, Path]]:
    """The untyped condition's captions: the same images in the same order,
    each captioned with what it shows (stage, panels, the moving object,
    colors) and nothing about why a stage fails: no collision facts, no
    nearest-free-pose or no-free-pose wording, no renderer notes, no dropped
    panels."""
    by_role = {c.role: c for c in chain}
    imgs = manifest.get("images") or []
    out = []
    for k, img in enumerate(imgs):
        lines = []
        if img.get("kind") == "state":
            st = str(img.get("state", ""))
            if st == "initial":
                head = "initial scene (spawn poses, before any stage)"
            else:
                role = img.get("role") or st.split(":", 1)[-1]
                c = by_role.get(role)
                head = (f"stage goal -- {_stage_label(c)}: the {c.emission.active} in its "
                        "subgoal region" if c else f"stage goal (role {role})")
            lines.append(f"Image {k + 1} of {len(imgs)}: {head}.")
            if img.get("panels"):
                lines.append("Panels, left to right: " + "; ".join(
                    f"{_VIEW_NAMES.get(p, p)} view" for p in img["panels"]) + ".")
        else:
            c = by_role.get(img.get("stage"))
            label = _stage_label(c) if c else f'stage (role {img.get("stage")})'
            mover = c.emission.active if c else "object"
            lines.append(f"Image {k + 1} of {len(imgs)}: {_path_kind(img)} -- {label}: the "
                         f"{mover} with one path row at a time at its bound, the other rows "
                         "at the center pose.")
            caps = [cap for row in img.get("rows") or [] for cap in row.get("captions") or []]
            if caps:
                lines.append("Panels: " + "; ".join(caps) + ".")
        leg = _legend(manifest, img, colors)
        if leg:
            lines.append(f"Colors: {leg}.")
        out.append(("\n".join(lines), Path(image_dir) / img["file"]))
    return out


def withheld_report(report: dict) -> dict:
    """The untyped condition's report: a stage that did not compile says only
    that it failed."""
    stages = []
    for rec in report["stages"]:
        if rec.get("compile_error"):
            rec = {**rec, "compile_error": "this stage failed"}
        stages.append(rec)
    return {**report, "stages": stages}


# ------------------------------------------------------------------ call #3 again

_AXIS_DIR = {"front_axis": "front", "lateral_axis": "left", "up_axis": "up"}


def _dir_token(axis: str, sign: str = "+") -> str:
    if axis.startswith("world."):
        return axis if sign == "+" else f"world.-{axis.split('.', 1)[1]}"
    obj, ax = axis.split(".", 1)
    return f"{obj}.{sign}{_AXIS_DIR.get(ax, ax)}"


def emission_raw(em) -> str:
    """A StageEmission written in call #3's output schema (the hardware
    loop's emission_raw): the assistant turn replayed when a stage is
    written again from its current emission."""
    import json

    def rot(rows):
        if rows and all(r.relation == "free" for r in rows) and {r.row for r in rows} == {
                "roll", "pitch", "yaw"}:
            return "free"
        out = []
        for r in rows:
            if r.relation == "free":
                out.append({"relation": "free", "row": r.row})
            else:
                out.append({"axis": _dir_token(r.axis), "relation": r.relation,
                            "reference": _dir_token(r.reference), "tol": r.tol})
        return out

    def trans(terms):
        out = []
        for t in terms:
            if t.term == "along":
                out.append({"term": "along", "axis": _dir_token(t.axis, t.sign or "+")})
            elif t.term == "expr":
                out.append({"term": "expr", "row": t.row, "lo": t.expr_lo, "hi": t.expr_hi})
            else:
                out.append({k: v for k, v in (("term", t.term), ("anchor", t.anchor),
                                              ("clearance", t.clearance), ("slack", t.slack),
                                              ("tol", t.tol)) if v is not None})
        return out
    return json.dumps({"stage": em.stage, "name": em.name, "active": em.active,
                       "passive": em.passive, "goal": em.goal,
                       "path_tsr": {"rot": rot(em.path_tsr.rot), "trans": trans(em.path_tsr.trans)},
                       "subgoal_tsr": {"rot": rot(em.subgoal_tsr.rot),
                                       "trans": trans(em.subgoal_tsr.trans)},
                       "verify": em.verify})


# NEW PROMPT TEXT (call #3 follow-ups in the repair study; the first two are
# the hardware loop's REEMIT_COMPILED / REEMIT_UNCOMPILED, unchanged)
REEMIT_COMPILED = ("A preview critic reviewed this stage before planning and rejected it: "
                   "{diagnosis}. Re-emit the corrected JSON object only, changing as little "
                   "as needed.")
REEMIT_UNCOMPILED = ("The compiler rejected that emission: {error}. A preview critic also "
                     "reviewed the stage before planning: {diagnosis}. Re-emit the corrected "
                     "JSON object only, changing as little as needed.")


def stage_specs(stage_plan: str | None) -> dict:
    """role -> StageSpec as call #3 saw it: a stage plan's bound stages, or
    emit_constraints' fixed pour-tea stages."""
    from scripts.emit_constraints import STAGES, bound_stages
    if not stage_plan:
        return {r: st for st, r in STAGES}
    from scripts.plan_stages import load_bindings
    b = load_bindings(stage_plan)
    by_idx = {s.index: s for s in b.plan.stages}
    return {r: st for st, r in bound_stages(b, by_idx)}


def call3_inputs(role: str, stage, c: ChainStage, state, chain: list, scene, symbols: dict,
                 asset_dirs: dict, role_index: dict, arm: str, render_dir: Path) -> dict:
    """The inputs call #3 had for this stage (emit_constraints.py): the
    stage's selection, the framed-chained views rendered at the stage's
    entry in the current chain, the frames record and the entry line."""
    from manip_sim.compile_tsr import ALIGN_TOL_RAD
    from manip_sim.vlm import entry_line
    from scripts.render_stage_frames import point_token
    poses = c.poses
    views = frec = None
    if arm in ("framed-chained", "framed"):
        from scripts.render_stage_frames import compute_stage_frames, render_role, scene_extent
        _, radius = scene_extent(scene, poses)
        sf = compute_stage_frames(((stage, role),), state.selections, state.objects,
                                  role_index, asset_dirs, symbols, poses, radius)[role]
        prev = [x for x in chain[:chain.index(c)] if x.emission.passive is not None]
        entry_from = f"{prev[-1].emission.name}.subgoal.nominal" if prev else "spawn"
        frec = render_role(scene, role, sf, poses, render_dir, entry_from)
        views = [Path(frec["views"][v]) for v in frec["views"]] or None
    chained = any(x.emission.passive is not None and x.emission.active == stage.active
                  for x in chain[:chain.index(c)])
    anchor = w_anchor = None
    if stage.passive and c.e_point is not None:
        e_tok = point_token(stage.active, symbols, c.e_point) or f"{stage.active}.anchor"
        w_tok = point_token(stage.passive, symbols, c.w_point) or f"{stage.passive}.anchor"
        anchor, w_anchor = (e_tok, c.e_point), (w_tok, c.w_point, poses[stage.passive])
    entry = None if c.synthesized else entry_line(
        stage.active, poses[stage.active], symbols[stage.active].axes, chained=chained,
        tol_rad=ALIGN_TOL_RAD, anchor=anchor, w_anchor=w_anchor)
    return {"selection": state.selections.get(role), "view_paths": views, "frames": frec,
            "entry": entry}


def rerun_call3(client, stage, vocab, inputs: dict, c: ChainStage, symbols: dict,
                feedback=None, compile_retries: int = 2,
                room=lambda: True) -> tuple[object | None, dict]:
    """Call #3 for one stage with its original inputs (plus `feedback` turns,
    if any), through call #3's own compile retries (typed compile errors go
    back to call #3 in every condition, as in the pipeline). `room()` is
    asked before every attempt (the run's call cap). Returns (emission or
    None, record); the emission is kept even if its compile failed (the next
    preview shows that)."""
    from manip_sim.code_authoring import compile_any
    from manip_sim.vlm import VLMError
    rec = {"stage": stage.name, "attempts": [], "feedback": [t for _, t in feedback or []]}
    rejections, em = [], None
    for _ in range(1 + compile_retries):
        if not room():
            rec["stopped"] = "call cap"
            break
        try:
            em = client.emit_constraints(stage, vocab, rejections=rejections,
                                         feedback=feedback, **inputs)
        except VLMError as e:
            rec["error"] = f"call #3 failed: {e}"
            break
        raw = client.logs[-1].raw
        try:
            cs = compile_any(em, symbols, c.poses, w_point=c.w_point, e_point=c.e_point,
                             entry_synthesized=c.synthesized)
            check_pair_consistency(cs)
        except CompileError as e:
            rec["attempts"].append({"raw": raw, "compile_error": e.text()})
            rejections.append((raw, e.text()))
            continue
        rec["attempts"].append({"raw": raw, "compiled": True})
        break
    return em, rec


# ------------------------------------------------------------------ plan and grade

def plan_and_grade(task: str, spec_dir: Path, seed: int, stage_names: dict,
                   mover_roles: list | None = None) -> dict:
    """Plan the specification in spec_dir with the task's planner (chord
    checks on) for one seed, then grade it. Returns {seed, planned, success,
    failure (repair_context record or None), eval}. plan_pour_tea writes no
    report: its last FAILED line is recorded as the failure."""
    import json
    import subprocess
    from manip_sim.repair_context import failure_from_report
    from scripts.score_authoring_ablation import (grade_command, last_failure, plan_command,
                                                  success)
    spec_dir = Path(spec_dir)
    plan = spec_dir / f"s{seed}.plan.npz"
    # a rerun into the same folder must never read an earlier run's plan,
    # report or grade (score_authoring_ablation does the same)
    for old in spec_dir.glob(f"s{seed}.plan.*"):
        old.unlink()
    cmd = plan_command(task, spec_dir / "emissions.json", plan, seed)
    # plan_command plans TASKS[task]'s selections; this spec has its own
    if cmd.count("--selections") != 1:
        raise RuntimeError(f"plan_command for {task} must pass --selections exactly once: {cmd}")
    cmd[cmd.index("--selections") + 1] = str(spec_dir / "selections.json")
    outp = plan.with_suffix(".out")
    with open(outp, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    res = {"seed": seed, "planned": plan.exists(), "success": False, "failure": None}
    if not plan.exists():
        rp = plan.with_suffix(".report.json")
        rep = (json.loads(rp.read_text()) if rp.exists()
               else {"status": "failed", "failure": last_failure(outp)})
        try:
            res["failure"] = failure_from_report(rep, stage_names, mover_roles)
        except ValueError as e:
            res["failure"] = {"stage": None, "type": None, "message": str(e)}
        return res
    with open(plan.with_suffix(".eval.out"), "w") as f:
        subprocess.run(grade_command(task, plan), stdout=f, stderr=subprocess.STDOUT)
    ev = json.loads(plan.with_suffix(".eval.json").read_text())
    _, ok = success(task, ev)
    res.update(success=bool(ok), eval={k: ev.get(k) for k in
                                       ("goal_pass", "path_pass", "max_penetration_mm")})
    return res


def stage_images(manifest: dict, image_dir: Path, role: str) -> tuple[Path, ...]:
    """The images of one stage (its goal state and its path images), no
    captions: what the render_check condition shows its checker."""
    out = []
    for img in manifest.get("images") or []:
        st = str(img.get("state", ""))
        if (img.get("kind") == "state" and (img.get("role") == role or st == f"goal:{role}")) \
                or (img.get("kind") != "state" and img.get("stage") == role):
            out.append(Path(image_dir) / img["file"])
    return tuple(out)
