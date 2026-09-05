"""Preview a selections artifact: resolve every role through
manip_sim.selection and render, per stage role, WHAT THE PLANNER
CONSUMES on the object clouds — the inspection step for the OUTPUT side
of the wiring, the complement of render_candidates.py (which inspects
the INPUT menu).

Since the canonical-frame redesign the compiler roots every stage's w
on the w-owning object's canonical frame at the call-#2 POINT; the
selected axes are consumed only by the grasp classifier (hand arm's
slide direction). The panels reflect that:

    gray cloud          mesh vertices (proposal.load_obj — numpy only,
                        no trimesh/mujoco dependency)
    gray dots + IDs     the full candidate pool from candidates.json
    black dot           each SELECTED candidate (the anchor point —
                        always consumed)
    solid RGB triad     EVERY panel: the anchor object's CANONICAL frame
                        at the role's selected point (via
                        compile_tsr._canonical_w at the manifest spawn
                        poses, so the fallback-x route matches the
                        compile gate exactly). One orientation per
                        object, anchored per role — the canonical-frame
                        redesign's invariant made visible; mismatched
                        triads across panels of one object mean a route
                        or frames.json-vs-refined divergence. It is w
                        for the w-owner roles and (by the passive-less
                        rule, Tw_e = I) for the grasp — stage 1 still
                        routes handle_grasp_tsr, see the arrow below —
                        and the relation-row axis source on mover roles.
                        x/front red, y/left gold, z/axis blue
    dashed gray triad   mover roles (transport_active, pour) only: the
                        resolved call-#2 selection frame. NOT consumed
                        under the emitted arm — only its anchor point is
                        (the feature Tw_e pins); kept as hand-arm
                        tilt_frame context.
    magenta arrow       grasp only: the classifier slide/wrap axis (the
                        resolved handle axis — the one direction
                        grasp_tsr's slide corridor and yaw wrap live
                        on). The corridor is symmetric, so the call-#2
                        sign fixes only the arrow's direction, not the
                        accept set.

so a wrong candidate, a wrong canonical route, or a frames.json-vs-
refined divergence is visible before the frame ever reaches the planner
or the preview critic. Provenance (which source resolved each
ingredient) is printed per role — the same comment string the Frame
carries.

With --refine, the Orient-Anything ladder is exercised the same way
demo_refine_frame.py does (truth + tilt as the coarse stand-in), and
resolution runs against the refined basis instead of frames.json — so
this doubles as the visual check that refined-column resolution and
frames.json agree where they should.

Run from the repo root:

    PYTHONPATH=. python scripts/preview_selections.py \
        outputs/selections/pour_tea.json --render outputs/selections/preview.png
    PYTHONPATH=. python scripts/preview_selections.py \
        outputs/selections/pour_tea.json --refine --tilt-deg 25 \
        --render outputs/selections/preview_refined.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manip_sim.compile_tsr import _canonical_w
from manip_sim.frames import load_symbols
from manip_sim.proposal import load_obj
from manip_sim.selection import (extremal_band, load_pool, load_selections,
                                 refine_body_basis, resolve_selection)
from manip_sim.scene import add_scene_arg, load_scene
from manip_sim.tsr import pose_from_pos_quat_wxyz

# stage-role semantics (select_frames.ROLES / the task contract): which
# roles select the w-owning object's point vs the mover's feature point
W_OWNER_ROLES = ("transport_passive",)
MOVER_ROLES = ("transport_active", "pour")


def _unit(v):
    v = np.asarray(v, float).reshape(3)
    return v / np.linalg.norm(v)


def _tilt(axis, deg, seed=0):
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(seed)
    perp = np.cross(axis, rng.normal(size=3))
    perp /= np.linalg.norm(perp)
    return R.from_rotvec(perp * np.deg2rad(deg)).as_matrix() @ _unit(axis)


def build_basis(name: str, V: np.ndarray, spec: dict, sym, tilt_deg: float,
                front_tilt_deg: float):
    """Simulated-coarse refined basis, mirroring demo_refine_frame's
    conventions (truth + tilt stands in for Orient Anything)."""
    true_up = _unit(spec["axes"]["up_axis"]["xyz"])
    coarse_up = _tilt(true_up, tilt_deg)
    if "pour_axis" in spec["axes"]:
        true_front = _unit(spec["axes"]["pour_axis"]["xyz"])
    else:
        true_front = _unit(np.cross(true_up, [0.0, 1.0, 0.0]))
    coarse_front = _tilt(true_front, front_tilt_deg, seed=1)
    front_pair = None
    if {"handle_center", "spout_tip"} <= set(sym.points):
        front_pair = (sym.points["handle_center"], sym.points["spout_tip"])
    # up fit gets the cloud; ring terminal route needs a mesh object,
    # which this numpy-only script does not construct — the revolution
    # fit alone covers the converted assets (see demo_refine_frame for
    # the full ladder with trimesh)
    fr = refine_body_basis(
        V, coarse_up, coarse_front, front_pair=front_pair,
        part_cloud=(extremal_band(V, coarse_up, coarse_front)
                    if front_pair else None))
    print(f"[{name}] basis: up {fr.up.method}/"
          f"{'ok' if fr.up.accepted else 'REJECTED'} "
          f"sigma {fr.up.sigma_deg:.3f} deg; azimuth {fr.azimuth.route}/"
          f"{'ok' if fr.accepted else 'REJECTED'} "
          f"sigma {fr.azimuth.sigma_deg:.2f} deg")
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("selections", help="role-keyed selections JSON")
    ap.add_argument("--refine", action="store_true",
                    help="resolve against a refined basis (simulated "
                         "coarse) instead of frames.json")
    ap.add_argument("--tilt-deg", type=float, default=25.0)
    ap.add_argument("--front-tilt-deg", type=float, default=20.0)
    ap.add_argument("--render", default=None, metavar="PNG")
    add_scene_arg(ap)
    args = ap.parse_args()
    scene = load_scene(args.scene, getattr(args, "grounding", None))

    sels = load_selections(args.selections)
    by_obj: dict[str, dict] = {}
    for role, s in sorted(sels.items()):
        by_obj.setdefault(s.axis.partition(".")[0], {})[role] = s

    panels = []
    symbols_all: dict = {}
    for name, roles in sorted(by_obj.items()):
        obj_dir = scene.asset_dirs[name]
        V, _ = load_obj(obj_dir / "meshes" / f"{name}_visual.obj")
        spec = json.loads((obj_dir / "frames.json").read_text())
        sym = load_symbols(obj_dir)
        symbols_all[name] = sym
        pool = load_pool(obj_dir)
        basis = (build_basis(name, V, spec, sym, args.tilt_deg,
                             args.front_tilt_deg) if args.refine else None)
        resolved = {}
        for role, s in roles.items():
            rf = resolve_selection(s, pool, sym, basis=basis)
            resolved[role] = rf
            print(f"[{name}] {role}: {rf.frame.comment} "
                  f"(status {rf.frame.status})")
        panels.append((name, V, pool, resolved))

    if not args.render:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # one panel per STAGE ROLE (not per object): overlaid triads at a
    # shared anchor (pour + transport_active both at spout_tip) are
    # unreadable; per-role panels show each frame alone on its object.
    clouds = {name: V for name, V, _, _ in panels}
    pools = {name: pool for name, _, pool, _ in panels}
    # the mover (fallback-x donor for a frontless w-owner) and the spawn
    # poses the emit gate freezes the canonical frame on
    mover_name = next((s.axis.partition(".")[0] for r, s in sorted(sels.items())
                       if r in MOVER_ROLES), None)
    body_poses = {n: pose_from_pos_quat_wxyz(*pq)
                  for n, pq in scene.fixed_poses().items()}
    roles = [(role, name, rf) for name, _, _, resolved in panels
             for role, rf in resolved.items()]
    roles.sort(key=lambda t: t[0])
    ncol = 2
    nrow = (len(roles) + ncol - 1) // ncol
    fig = plt.figure(figsize=(6.6 * ncol, 5.8 * nrow))
    for k, (role, name, rf) in enumerate(roles):
        V, pool = clouds[name], pools[name]
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        sub = V[:: max(1, len(V) // 4000)]
        ax.scatter(*sub.T, s=1.2, c="0.65", alpha=0.25, linewidths=0)
        sel_id = rf.selection.candidate_id
        for i, c in sorted(pool.items()):
            if i == sel_id:
                continue
            ax.scatter(*np.atleast_2d(c["xyz"]).T, s=8, c="0.35",
                       linewidths=0)
            ax.text(*c["xyz"], f"{i}", fontsize=6, color="0.25")
        L = 0.35 * float(np.linalg.norm(V.max(0) - V.min(0)))
        sel_T = rf.frame.T()
        o = sel_T[:3, 3]
        ax.scatter(*np.atleast_2d(o).T, s=60, c="black", marker="o",
                   label=f"anchor: mark {sel_id}")
        axis_desc = (f"z = {rf.selection.sign}"
                     f"{rf.selection.axis.partition('.')[2]} "
                     f"({rf.axis_source})")
        # EVERY role: the anchor object's canonical frame at the mark —
        # the redesign's invariant (one orientation per object, anchored
        # per role) made visible. Same helper + spawn poses as the emit
        # gate, so the fallback-x route cannot drift from the compiler.
        try:
            w_frame, route = _canonical_w(
                name, mover_name if mover_name != name else None,
                symbols_all, body_poses, rf.frame.point, role)
            T = w_frame.T()
            canon = (f"{name} canonical @ mark {sel_id}: "
                     f"z = up_axis, x = {route}")
        except Exception as e:                   # no up_axis etc.: fall back
            print(f"[preview] {role}: canonical frame unavailable ({e}); "
                  "drawing the selection frame")
            T, canon = sel_T, None
        overlay = None                # extra per-role glyph on the anchor
        if role in W_OWNER_ROLES:
            title = ((f"{role} — w = {canon}\n"
                      "(one frame, shared by transport & pour)")
                     if canon else
                     f"{role} — {name} selection frame, {axis_desc}")
        elif role in MOVER_ROLES:
            # the POINT is consumed (the feature Tw_e pins) and the
            # canonical axes displace in relation rows; the call-#2
            # selection frame binds nothing under the emitted arm —
            # kept dashed for the hand arm's tilt_frame
            overlay = "selection" if canon else None
            title = ((f"{role} — {canon}\n"
                      f"anchor = Tw_e pin; selection axes hand-arm "
                      f"only: {axis_desc}")
                     if canon else
                     f"{role} — {name} selection frame, {axis_desc}")
        else:                                    # grasp
            overlay = "slide" if canon else None
            title = ((f"{role} — {canon}\n"
                      f"classifier slide/wrap axis: {axis_desc}")
                     if canon else
                     f"{role} — {name} classifier frame, {axis_desc}")
        for col, color, lbl in ((0, "red", "x/front"),
                                (1, "goldenrod", "y/left"),
                                (2, "blue", "z/axis")):
            seg = np.array([T[:3, 3], T[:3, 3] + L * T[:3, col]])
            ax.plot(*seg.T, color=color, lw=2.6, label=lbl)
        if overlay == "selection":
            for col in range(3):
                seg = np.array([o, o + 0.85 * L * sel_T[:3, col]])
                ax.plot(*seg.T, color="0.55", lw=1.2, ls="--",
                        label="selection axes (unconsumed)"
                        if col == 0 else None)
        elif overlay == "slide":
            seg = np.array([o, o + 1.15 * L * sel_T[:3, 2]])
            ax.plot(*seg.T, color="darkmagenta", lw=2.0,
                    label="classifier slide/wrap axis")
        c0 = V.mean(axis=0)
        lim = np.array([c0 - 1.7 * L, c0 + 1.7 * L])
        ax.set_xlim(lim[:, 0]); ax.set_ylim(lim[:, 1])
        ax.set_zlim(lim[:, 2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("refined basis" if args.refine else "frames.json arm",
                 fontsize=10)
    Path(args.render).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.render, dpi=140)
    print(f"[preview] wrote {args.render}")


if __name__ == "__main__":
    main()