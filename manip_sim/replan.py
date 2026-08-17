"""Slip-triggered stage re-anchoring — the execution-time re-plan loop.

Three loops revise this architecture's behavior, and only this one is
numeric. The call-#4 critic edits B^w rows because the constraints were
AUTHORED wrong; call-#5 repair edits schema because a typed failure says
the constraints don't intersect reality. This module handles the third
case: the constraints are CORRECT and only the measured in-hand
transform T_ee_body went stale (slip). No symbolic decision changed, so
no VLM belongs in this loop — re-measure the transform, re-sample the
same TSRs, re-plan the remaining stages. Routing a perception update
through the authoring layer would be exactly the confusion the critic
packet's authored_in pointers exist to prevent.

Trigger discipline (both properties are the design, not conveniences):

  boundary-only   re-anchoring inside a stage would mean re-entering a
                  path TSR frozen at entry — a different, messier
                  contract. The residual is checked between stages.
  tolerance from  the threshold is what the NEXT stage's subgoal B^w
  the constraints tolerates (slip_tolerance below), not a fresh magic
                  number: re-plan fires exactly when measured slip
                  exceeds what the authored constraint grants. The
                  STANDING residual (last SlipMonitor entry) is what is
                  compared — the transient max includes recoverable
                  wobble; what invalidates downstream goals is where
                  the object settled.

replan_from_stage(k) is stage-generic and CHAINS: re-planning stage 2
re-plans stage 3 too, because the pour pair freezes wherever the new
transport ends. Planning runs against a scratch MjData snapshotted from
the live env (ArmKinematics semantics), so the re-plan sees the
measured scene — teapot and mug where they really are — without
touching the running sim.

Failures raise typed ReplanError (stage + reason), never SystemExit:
the executor decides whether a failed re-plan aborts or falls through
to the stale plan, and the failure record is the input the call-#5
repair loop will eventually consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import Frame
from .planning import (ArmKinematics, AttachedArmKinematics, AttachedObject,
                       MinkIK, plan_constrained)
from .pour_stages import pour_pair, transport_pair
from .tsr import TSR, sample_intersection

_FREE_ROT_WIDTH = 2.0 * np.pi - 1e-9


class ReplanError(RuntimeError):
    """A re-plan attempt that found no feasible path — typed, routable
    (stage + reason), never a SystemExit: the caller owns the policy."""

    def __init__(self, stage: int, reason: str):
        super().__init__(f"[replan stage {stage}] {reason}")
        self.stage = stage
        self.reason = reason


# ------------------------------------------------------------- trigger


def slip_tolerance(subgoal: TSR) -> tuple[float, float]:
    """(pos_m, rot_rad) the subgoal's B^w tolerates: the tightest finite
    half-width among translation rows and among rotation rows. Rows that
    are free (infinite translation, full-circle rotation) express no
    preference and are excluded; a subgoal free in ALL rows of a block
    tolerates any slip in that block (inf)."""
    Bw = np.asarray(subgoal.Bw, dtype=float)
    half = 0.5 * (Bw[:, 1] - Bw[:, 0])
    t = half[:3][np.isfinite(half[:3])]
    r = half[3:][half[3:] < 0.5 * _FREE_ROT_WIDTH]
    return (float(t.min()) if len(t) else np.inf,
            float(r.min()) if len(r) else np.inf)


def slip_exceeds(standing: tuple[float, float],
                 subgoal: TSR) -> tuple[bool, float, float]:
    """Standing rigid-attachment residual (dpos_m, drot_rad) vs. what
    the next stage's subgoal tolerates. -> (fire, tol_pos, tol_rot)."""
    tol_pos, tol_rot = slip_tolerance(subgoal)
    return standing[0] > tol_pos or standing[1] > tol_rot, tol_pos, tol_rot


# ------------------------------------------------------------- machinery


def default_seeds(q_now: np.ndarray, joint_range: np.ndarray,
                  rng: np.random.Generator, n_random: int = 8) -> list:
    """IK seed set anchored at the CURRENT config (the re-plan must be
    continuable from where the arm stands), plus the planner's base-joint
    azimuth sweeps and a few uniform draws."""
    seeds = [q_now]
    for dj0 in (-1.0, -0.5, 0.5, 1.0):
        s = q_now.copy()
        s[0] += dj0
        seeds.append(s)
    lo = np.maximum(joint_range[:, 0], -np.pi)
    hi = np.minimum(joint_range[:, 1], np.pi)
    seeds += [rng.uniform(lo, hi) for _ in range(n_random)]
    return seeds


def goal_funnel(rep, ik, kin, attached, seeds, containment, label, q_ref,
                ik_kw=None):
    """Sampled body poses -> IK -> collision -> containment of the
    ACHIEVED config; sorted nearest q_ref first. The same funnel the
    offline planner runs (single shared home once the branches merge)."""
    goals, n_ik, n_col = [], 0, 0
    for T_body in rep.accepted:
        T_ee = T_body @ np.linalg.inv(attached.T_ee_body)
        q, ok = ik.solve_multiseed(T_ee, seeds, **(ik_kw or {}))
        if not ok:
            continue
        n_ik += 1
        if kin.in_collision(q):
            continue
        n_col += 1
        T_ach = attached.body_pose(kin.fk(q))
        if all(t.contains(T_ach, tol=5e-3) for t in containment):
            goals.append(q)
    print(f"[replan:{label}] goal funnel: {len(rep.accepted)} sampled -> "
          f"{n_ik} IK-feasible -> {n_col} collision-free -> "
          f"{len(goals)} contained")
    goals.sort(key=lambda q: float(np.linalg.norm(q - q_ref)))
    return goals


@dataclass
class TaskFrames:
    """The passive/active task frames stages 2-3 consume — resolved once
    (hand-authored or VLM arm, the re-plan is arm-agnostic) and reused
    verbatim: slip does not change WHAT the constraints say."""
    spout_tip: Frame
    tilt_frame: Frame
    opening: Frame
    rim_margin: float = 0.02


@dataclass
class ReplanResult:
    paths: list = field(default_factory=list)     # [(stage_id, np.ndarray)]
    pairs: dict = field(default_factory=dict)     # stage_id -> stage pair
    stats: dict = field(default_factory=dict)

    @property
    def path(self) -> np.ndarray:
        return np.vstack([p for _, p in self.paths])

    @property
    def stage_ids(self) -> np.ndarray:
        return np.concatenate([np.full(len(p), s) for s, p in self.paths])


def replan_from_stage(stage: int, *, env, q_now: np.ndarray,
                      T_ee_body: np.ndarray, T_teapot_now: np.ndarray,
                      T0_mug: np.ndarray, frames: TaskFrames,
                      tilt_target: float, object_joint: str,
                      object_prefix: str = "teapot",
                      n_goal_samples: int = 30, timeout: float = 20.0,
                      seed: int = 0,
                      transport_kw: dict | None = None) -> ReplanResult:
    """Re-plan stages k..3 from the arm's CURRENT config with the
    RE-MEASURED in-hand transform. stage=3: pour pair frozen at the
    measured pose (q_now IS the entry). stage=2: transport re-aimed from
    the measured teapot position, then the pour chained from the new
    transport end — the pour-freezes-at-entry convention applied to the
    re-planned entry, identical to the offline planner's handoff.
    """
    if stage not in (2, 3):
        raise ValueError("replan_from_stage: stage must be 2 or 3")
    rng = np.random.default_rng(seed)
    attached = AttachedObject(T_ee_body)
    kin = AttachedArmKinematics(env, attached, object_joint, object_prefix)
    ik = MinkIK(kin)
    seeds = default_seeds(q_now, kin.joint_range, rng)
    out = ReplanResult()
    q_entry = q_now
    T_entry = T_teapot_now

    if stage == 2:
        tpair = transport_pair(
            T0_mug_body=T0_mug, mug_opening=frames.opening,
            spout_tip=frames.spout_tip,
            teapot_body_pos_now=T_teapot_now[:3, 3],
            rim_margin=frames.rim_margin, **(transport_kw or {}))
        rep = sample_intersection(tpair.subgoal, [tpair.path],
                                  n=n_goal_samples, rng=rng)
        goals = goal_funnel(rep, ik, kin, attached, seeds,
                            [tpair.path, tpair.subgoal], "transport", q_now)
        if not goals:
            raise ReplanError(2, "no feasible transport goal configs")
        res = plan_constrained(kin, attached, [tpair.path], q_now, goals[0],
                               timeout=timeout, rng=rng)
        if not res.ok:
            raise ReplanError(2, f"transport planning failed: {res.reason}")
        out.paths.append((2, res.path))
        out.pairs[2] = tpair
        out.stats["transport"] = res.stats
        q_entry = res.path[-1]
        T_entry = attached.body_pose(kin.fk(q_entry))

    ppair = pour_pair(T_entry, frames.tilt_frame, tilt_target=tilt_target)
    rep = sample_intersection(ppair.subgoal, [ppair.path],
                              n=n_goal_samples, rng=rng)
    goals = goal_funnel(rep, ik, kin, attached, [q_entry] + seeds,
                        [ppair.path, ppair.subgoal], "pour", q_entry,
                        ik_kw={"iters": 200})
    if not goals:
        raise ReplanError(3, "no feasible pour configs at the re-anchored "
                             "entry — the slip may have carried the pose "
                             "outside the reachable tilt corridor")
    res = plan_constrained(kin, attached, [ppair.path], q_entry, goals[0],
                           timeout=timeout, eps=0.05, rng=rng)
    if not res.ok:
        raise ReplanError(3, f"pour planning failed: {res.reason}")
    out.paths.append((3, res.path))
    out.pairs[3] = ppair
    out.stats["pour"] = res.stats
    return out
