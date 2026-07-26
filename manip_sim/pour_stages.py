"""Hand-authored TSR pairs for the three-stage pour task.

This file is the ground-truth arm of the eventual emission ablation
(schema-emitted / constrained-Python / unconstrained-Python / raw-numeric
vs. these hand instances) — a deliverable, not scaffolding.

Frame bookkeeping convention (keep this invariant everywhere): every TSR in
a stage constrains the SAME physical frame e = the teapot BODY frame, so
that intersection sampling and downstream IK all speak one pose. Which
*feature* of the teapot a TSR pins is encoded in Tw_e:

    "spout tip at the mug opening"  ->  Tw_e = inv(T_body_spout_tip)
    "body upright"                  ->  Tw_e = I

End-effector goals are recovered afterwards by composing sampled body poses
with the measured in-hand transform T_body_ee (grasp transform), never by
constraining the gripper directly — constraints stay embodiment-agnostic.

Stage summary (from the TSR design discussions):

  1. GRASP     w = handle frame. Used as a *classifier* over grasp
               proposals (AnyGrasp in the real pipeline; antipodal/hand
               samples in sim), not as a sampler.
  2. TRANSPORT subgoal: spout tip in a region above the mug opening,
               approach azimuth free; PATH: teapot near-upright the whole
               way, translation free. Goals = subgoal INTERSECT path.
  3. POUR      w at the spout tip *frozen at stage entry*; the tip is the
               pivot (translation pinned), the sidecar tilt axis is the
               single free rotational DoF. Path TSR spans the tilt range;
               subgoal pins the target tilt. Termination of the pour is
               perceptual (liquid-proxy flow / dwell), not a pose event —
               the subgoal here is only "reached pouring attitude".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import Frame
from .tsr import FREE_ROT, FREE_TRANS, TSR, bounds, make_pose


def _world_upright_frame(at: np.ndarray) -> np.ndarray:
    """World-axis-aligned frame at a point: the reference for gravity
    constraints. Roll/pitch displacements in this frame are tilts w.r.t.
    gravity regardless of where the object travels (translation is free)."""
    return make_pose(at, np.eye(3))


# ------------------------------------------------------------------ stage 1


def grasp_tsr(
    T0_body: np.ndarray,
    handle: Frame,
    slide: float = 0.02,
    lateral_tol: float = 0.005,
    wrap_rot: tuple[float, float] = (-np.pi / 4, np.pi / 4),
) -> TSR:
    """Grasp-region TSR on the handle frame, in the ROLE of a classifier:
    score/filter externally proposed grasp poses (their implied handle-frame
    displacement), do not sample from it. e here is the GRIPPER frame — the
    one stage-1 exception to the body-frame convention, because the grasp
    *defines* T_body_ee rather than consuming it.

    Axes (handle frame, +z = handle long axis): slide along the handle
    (z, +-slide), tight laterally (x, y), rotation about the handle axis
    within wrap_rot (yaw), tight roll/pitch. Widen wrap_rot if your gripper
    approach study needs it; the lookahead feasibility probe — not this TSR
    — is what should prefer grasps that keep stages 2-3 reachable.
    """
    return TSR(
        T0_w=T0_body @ handle.T(),
        Bw=bounds(
            x=(-lateral_tol, lateral_tol),
            y=(-lateral_tol, lateral_tol),
            z=(-slide, slide),
            roll=(-0.09, 0.09),
            pitch=(-0.09, 0.09),
            yaw=wrap_rot,
        ),
        name="grasp/handle",
    )


# ------------------------------------------------------------------ stage 2


@dataclass
class TransportPair:
    subgoal: TSR   # sample this ...
    path: TSR      # ... check containment in this (and enforce along the motion)


def transport_pair(
    T0_mug_body: np.ndarray,
    mug_opening: Frame,
    spout_tip: Frame,
    teapot_body_pos_now: np.ndarray,
    rim_margin: float = 0.02,
    height: tuple[float, float] = (0.03, 0.08),
    upright_tol: float = np.deg2rad(15.0),
    z_corridor: tuple[float, float] = (0.02, 0.45),
) -> TransportPair:
    """Stage-2 pair.

    SUBGOAL — w = mug opening frame (world pose from mug body pose +
    sidecar); e = teapot body. The tip offset in Tw_e is TRANSLATION-ONLY
    (inv of the tip *point*, not the tip frame): translation bounds then pin
    the spout-tip position in the opening frame while rotation bounds bind
    the BODY attitude. Using the full tip frame here is a trap this module
    fell into once — the tip frame's +z is the spout axis, horizontal at
    upright, so roll/pitch~0 on it demands the spout point at the ceiling
    and the intersection with the upright path TSR is empty. (The
    intersection sampler's per-constraint rejection counts are what catch
    this class of bug; keep them on for every VLM-emitted pair.)
    Bounds: x/y within the rim disc (square approximation, rim_margin
    inset), z in the pour-standoff band above the opening, yaw FREE
    (approach from any azimuth), roll/pitch within upright_tol (still not
    pouring at the subgoal — tilt happens in stage 3).

    NOTE the deliberate approximation flagged for later: roll/pitch here are
    measured in the OPENING frame, which equals world-upright only while the
    mug stands upright on the table. True in this scene; revisit if a task
    ever tilts the passive object.

    PATH — w = world-upright frame at the teapot's current position; e =
    teapot body directly (Tw_e = I). x/y FREE, z within a table-clearance
    corridor (relative to the frame origin at the teapot's start pose),
    roll/pitch within upright_tol, yaw FREE. This is the constraint CBiRRT
    enforces at every configuration; the subgoal-only baseline arm of the
    headline experiment simply omits it during the motion.
    """
    T0_opening = T0_mug_body @ mug_opening.T()
    tip_offset = make_pose(spout_tip.point)          # translation-only
    subgoal = TSR(
        T0_w=T0_opening,
        Tw_e=np.linalg.inv(tip_offset),
        Bw=bounds(
            x=(-rim_margin, rim_margin),
            y=(-rim_margin, rim_margin),
            z=height,
            roll=(-upright_tol, upright_tol),
            pitch=(-upright_tol, upright_tol),
            yaw=FREE_ROT,
        ),
        name="transport/subgoal(spout_tip@opening)",
    )
    path = TSR(
        T0_w=_world_upright_frame(teapot_body_pos_now),
        Bw=bounds(
            x=FREE_TRANS,
            y=FREE_TRANS,
            z=z_corridor,
            roll=(-upright_tol, upright_tol),
            pitch=(-upright_tol, upright_tol),
            yaw=FREE_ROT,
        ),
        name="transport/path(upright)",
    )
    return TransportPair(subgoal=subgoal, path=path)


# ------------------------------------------------------------------ stage 3


@dataclass
class PourPair:
    subgoal: TSR
    path: TSR


def pour_pair(
    T0_body_at_entry: np.ndarray,
    tilt_frame: Frame,
    tilt_target: float = np.deg2rad(95.0),
    tilt_tol: float = np.deg2rad(5.0),
    tilt_range: tuple[float, float] = (np.deg2rad(-3.0), np.deg2rad(110.0)),
    pivot_tol: float = 0.005,
    off_axis_tol: float = np.deg2rad(3.0),
) -> PourPair:
    """Stage-3 pair, built ONCE at stage entry (poses frozen then — the pour
    is defined relative to where the transport actually ended, mirroring
    'Tw_e fixed at grasp completion' from stage 1's handoff).

    w = the tilt-axis frame at the spout tip, in WORLD, at entry. The
    sidecar maps the tilt axis to frame +z... no: frames.py maps the primary
    axis to +z, but displacement 'roll' is rotation about w's +x. So we
    REORDER here: build w with +x = tilt axis, which makes the free DoF
    exactly the roll component of the displacement. Positive roll tips the
    spout down (sidecar sign convention).

    Translation pinned to +-pivot_tol: the spout tip is the pivot. This is
    the frame-placement move that encodes the pour geometry — root w at the
    body center instead and the same rotational bounds would sweep the tip
    through an arc over the table.

    PATH: roll in tilt_range (slight negative allowance for settle),
    pitch/yaw within off_axis_tol. SUBGOAL: roll = tilt_target +- tilt_tol.
    Both constrain e = teapot body via Tw_e = inv(T_w_body_at_entry).
    """
    T_frame = tilt_frame.T()          # body -> tilt frame, tilt axis on +z
    # rotate the frame so the tilt axis lands on +x: new_x = old_z,
    # new_y = old_y, new_z = -old_x (right-handed)
    swap = np.eye(4)
    # columns = new axes in old coords: new_x = old_z (tilt axis),
    # new_y = old_y, new_z = -old_x  (det +1, right-handed)
    swap[:3, :3] = np.array([[0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0]]).T
    T0_w = T0_body_at_entry @ T_frame @ swap
    Tw_e = np.linalg.inv(np.linalg.inv(T0_body_at_entry) @ T0_w)

    def _tsr(roll_bounds, name):
        return TSR(
            T0_w=T0_w,
            Tw_e=Tw_e,
            Bw=bounds(
                x=(-pivot_tol, pivot_tol),
                y=(-pivot_tol, pivot_tol),
                z=(-pivot_tol, pivot_tol),
                roll=roll_bounds,
                pitch=(-off_axis_tol, off_axis_tol),
                yaw=(-off_axis_tol, off_axis_tol),
            ),
            name=name,
        )

    return PourPair(
        subgoal=_tsr((tilt_target - tilt_tol, tilt_target + tilt_tol),
                     "pour/subgoal(tilt_target)"),
        path=_tsr(tilt_range, "pour/path(tilt_corridor)"),
    )