"""Robot-free scores of an authored specification: computed from the emission
alone, once per emission, with no arm, grasp or planner involved.

So far: handoff consistency. For each pair of chained stages (the same mover,
the next stage starting where this one ends), how much of this stage's goal set
the next stage cannot start from. The goal set is what the planner can actually
pick a goal from -- the subgoal intersected with the stage's own path -- compiled
at the nominal entry, as the emit gate compiles it. Each sampled goal is tested
with the compile the planner runs at the actual handoff (the next stage compiled
with that pose as its start, including its entry-admission check).

Reported per stage pair: the rejected fraction, and two binaries that do not
depend on the sampling measure -- "all admitted" and "none admitted". The
fraction inherits the Euler-box measure of TSR sampling, so it is not
comparable across parameterizations (bounds near zero vs near pitch +-90);
"none admitted" is the case no planner can recover from.

Inputs mirror the planners' (scene, optional grounding, optional stage plan,
selections, emissions); object poses are the scene's spawn poses, as the gate's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_HANDOFF = 64
HANDOFF_SEED = 0


def _context(scene_path, grounding, stage_plan, selections, emissions):
    """(symbols, spawn poses, movers, points) built the way the planners build them."""
    from manip_sim.code_authoring import load_any_emissions
    from manip_sim.frames import load_symbols
    from manip_sim.scene import load_scene
    from manip_sim.selection import load_selections, selected_points, selection_objects
    from manip_sim.task_plan import mover_stages
    from scripts.emit_constraints import STAGES, _spawn_poses, bound_stages

    scene = load_scene(scene_path, grounding)
    symbols = {n: load_symbols(d) for n, d in scene.asset_dirs.items()}
    poses = _spawn_poses(scene)
    if stage_plan:
        from scripts.plan_stages import load_bindings
        b = load_bindings(stage_plan)
        stages = bound_stages(b, {s.index: s for s in b.plan.stages})
        synthesized = b.entry_synthesized
        role_index = {r: st.index for r, (_, st) in b.roles.items()}
    else:
        from scripts.select_frames import ROLES
        stages, synthesized = STAGES, frozenset()
        role_index = {r: st.index for r, (_, st) in ROLES.items()}
    sels = load_selections(selections)
    objects = selection_objects(selections, sels)
    points = {role: selected_points(role, st, sels, objects, role_index, scene.asset_dirs, symbols)
              for st, role in stages}
    movers = mover_stages(stages, load_any_emissions(emissions), synthesized)
    return symbols, poses, movers, points


def handoff_scores(scene_path, grounding, stage_plan, selections, emissions,
                   n: int = N_HANDOFF, seed: int = HANDOFF_SEED) -> list[dict]:
    """One row per chained stage pair: {"from", "to", "tested", "rejected",
    "all_admitted", "none_admitted", "reasons"}; a row with "error" when the
    stage itself does not compile at its nominal entry."""
    from manip_sim.code_authoring import compile_any
    from manip_sim.task_plan import chained, handoff_rejection
    from manip_sim.tsr import sample_intersection

    symbols, poses, movers, points = _context(scene_path, grounding, stage_plan,
                                              selections, emissions)
    rows, T_ent = [], None
    for i, m in enumerate(movers):
        active, passive = m.stage.active, m.stage.passive
        T_ent = poses[active] if T_ent is None else T_ent
        w_pt, e_pt = points[m.role]
        try:
            cs = compile_any(m.emission, symbols, {active: T_ent, passive: poses[passive]},
                             w_point=w_pt, e_point=e_pt, entry_synthesized=m.synthesized)
        except Exception as e:                       # the gate rejected it: nothing to chain
            rows.append({"from": m.role, "error": f"{type(e).__name__}: {e}"[:200]})
            break
        nxt = chained(movers, i)
        if nxt is not None:
            rng = np.random.default_rng(seed)
            goals = sample_intersection(cs.subgoal, [cs.path], n=n, rng=rng).accepted
            wn, en = points[nxt.role]
            rejected, reasons = 0, {}
            for T in goals:
                why = handoff_rejection(compile_any, nxt, symbols,
                                        {nxt.stage.passive: poses[nxt.stage.passive],
                                         nxt.stage.active: T}, wn, en)
                if why is not None:
                    rejected += 1
                    k = why.split(";")[0][:100]
                    reasons[k] = reasons.get(k, 0) + 1
            tested = len(goals)
            rows.append({"from": m.role, "to": nxt.role, "tested": tested, "rejected": rejected,
                         "all_admitted": tested > 0 and rejected == 0,
                         "none_admitted": tested > 0 and rejected == tested,
                         "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:3])})
        T_ent = cs.subgoal.nominal()
    return rows
