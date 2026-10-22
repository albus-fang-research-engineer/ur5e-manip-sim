"""Task-generic planning blocks for scripts/plan_task.py: the parts of the
pour-tea planner that are not about teapots, plus the two things a
non-pour task needs that pour-tea never did (a synthesized stage start, a
grasp on an arbitrary fitted part). Pure numpy over duck-typed kinematics,
so it tests without MuJoCo; plan_task.py does the wiring.

Typed failures: every way a stage can fail raises TaskPlanError(kind,
stage, detail). `detail` carries the evidence (funnel counts, collision
offender histogram, the reason histogram of the planner, the excess of a
start pose outside its path TSR) so the plan report states WHY, and an
infeasible specification surfaces as a typed failure rather than a crash
or a silent best-effort plan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------- constants
# Synthesized stage start (a stage whose emitted carry does not exist):
# the subgoal's nominal pose translated along the passive object's +up until
# the active object's lowest point clears the passive object's highest
# point by this margin. A named constant, not derived from anything.
SYNTH_START_CLEARANCE_M = 0.05

# Sim grasp stand-in (AnyGrasp's role on hardware, where the approach comes
# from the grasp network and this parameter does not exist): approach
# elevation below horizontal. Top-down, the one sensible approach for a part
# lying on the table.
GRASP_ELEVATION_RAD = np.pi / 2

# Sim grasp stand-in, same status as the elevation: how far the grip site is
# backed off along the approach from the grasp part's axis, so the open
# fingertip pads stop above the support surface. Measured 2026-10-21 on the
# insert_marker scene (UR5e + Robotiq 2F-85, fingers as spawned = open): the
# nominal top-down grasp at the marker centreline (9.9 mm above the table)
# collides inner_finger <-> table at 0 and 1 mm and is clear from 2 mm; +3 mm
# margin for the proposals' roll/pitch noise. pad_offset is the hardware
# equivalent; on hardware the grasp network sets the depth.
GRASP_PAD_BACKOFF_M = 0.005

# containment tolerance for the synthesized start against the stage's path
# TSR (distance mixes m and rad; this is "contained", not a slack)
START_CONTAINMENT_TOL = 1e-6


class TaskPlanError(RuntimeError):
    """A typed planning failure. kind: compile | grasp_part | grasp |
    synthesized_start_outside_path | synthesized_start_unreachable |
    free_reach | lift | empty_goal_set | planning."""

    def __init__(self, kind: str, stage: str, message: str,
                 detail: dict | None = None):
        self.kind, self.stage, self.detail = kind, stage, dict(detail or {})
        super().__init__(f"[{stage}] {kind}: {message}")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "stage": self.stage,
                "message": str(self), "detail": self.detail}


# --------------------------------------------------------------- geometry

def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations, degrees."""
    c = (np.trace(np.asarray(R1).T @ np.asarray(R2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def transform_points(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.asarray(V, float) @ T[:3, :3].T + T[:3, 3]


def synthesized_start(T_sub_nominal: np.ndarray, mover_V: np.ndarray,
                      T_passive: np.ndarray, passive_V: np.ndarray,
                      passive_up_body: np.ndarray,
                      clearance: float = SYNTH_START_CLEARANCE_M
                      ) -> tuple[np.ndarray, float]:
    """The subgoal's nominal mover pose, translated along the passive
    object's +up (world) until the mover's lowest point clears the
    passive's highest point by `clearance`. Never translated down: a
    nominal already clear stays where it is. -> (T_start, shift_m)."""
    up = T_passive[:3, :3] @ np.asarray(passive_up_body, float)
    up = up / np.linalg.norm(up)
    low = float(np.min(transform_points(T_sub_nominal, mover_V) @ up))
    high = float(np.max(transform_points(T_passive, passive_V) @ up))
    shift = max(0.0, high + clearance - low)
    T = np.array(T_sub_nominal, float, copy=True)
    T[:3, 3] = T[:3, 3] + shift * up
    return T, shift


def lowest_end(T_body: np.ndarray, V_body: np.ndarray,
               points: dict[str, np.ndarray], up_world=(0.0, 0.0, 1.0)) -> dict:
    """Which end of the object is lowest at pose T_body: the lowest mesh
    vertex (body coords) and the nearest grounded point symbol to it."""
    up = np.asarray(up_world, float)
    k = int(np.argmin(transform_points(T_body, V_body) @ up))
    v = np.asarray(V_body, float)[k]
    rec = {"lowest_point_body": np.round(v, 4).tolist(),
           "lowest_point_world_z": round(float(transform_points(T_body, v[None])[0] @ up), 4)}
    if points:
        name = min(points, key=lambda n: float(np.linalg.norm(np.asarray(points[n]) - v)))
        rec.update(nearest_symbol=name,
                   nearest_symbol_distance_m=round(float(np.linalg.norm(
                       np.asarray(points[name]) - v)), 4))
    return rec


# ------------------------------------------------------------------ grasp

@dataclass(frozen=True)
class GraspPart:
    part: str
    primitive: str            # "line" | "ring"
    center: np.ndarray        # body coords
    axis: np.ndarray          # body coords, unit
    half_length: float        # slide range along the axis (0 for a ring)


def grasp_part(frames_doc: dict, part: str | None, stage: str = "grasp") -> GraspPart:
    """The grasp region from a grounded frames.json: the selected part's
    fitted axis and, for a line, its fitted length (the slide range). A
    blob or an unlabelled point has no axis to close across -> typed."""
    if not part:
        raise TaskPlanError("grasp_part", stage,
                            "the selected grasp point carries no part label")
    prim = frames_doc.get("provenance", {}).get("parts", {}).get(part, {}).get("primitive")
    pts, axes, qty = (frames_doc.get(k, {}) for k in ("points", "axes", "quantities"))
    c, a = pts.get(f"{part}_center"), axes.get(f"{part}_axis")
    if prim not in ("line", "ring") or c is None or a is None:
        raise TaskPlanError("grasp_part", stage,
                            f"part {part!r} has no line/ring fit (primitive {prim!r})",
                            {"part": part, "primitive": prim})
    axis = np.asarray(a["xyz"], float)
    half = 0.0
    if prim == "line":
        L = qty.get(f"{part}_length", {}).get("value")
        if L is None:
            raise TaskPlanError("grasp_part", stage, f"line part {part!r} has no _length")
        half = 0.5 * float(L)
    return GraspPart(part, prim, np.asarray(c["xyz"], float),
                     axis / np.linalg.norm(axis), half)


def backoff_along_approach(tsr, d: float):
    """The same grasp TSR with the gripper frame e moved back by d along
    its own approach axis (e's -z): Tw_e @ T(0, 0, -d). Applies to any
    approach direction; for a top-down grasp it raises the grip site."""
    from manip_sim.tsr import TSR, make_pose
    return TSR(T0_w=tsr.T0_w, Bw=tsr.Bw, name=tsr.name,
               Tw_e=tsr.Tw_e @ make_pose([0.0, 0.0, -float(d)], np.eye(3)))


def topdown_approach_h(T0_w: np.ndarray, stage: str = "grasp") -> np.ndarray:
    """Approach direction for handle_grasp_tsr, in the w frame, giving a
    top-down grasp that closes across the bar (w.z). nominal_grip_in_world
    takes the approach's horizontal part, sets the closing axis to
    approach x bar, and pitches the approach down about that axis: with the
    approach along the bar's horizontal direction the closing axis is
    horizontal and perpendicular to the bar, and a 90 deg pitch points the
    approach straight down. (An approach perpendicular to a horizontal bar
    would make the closing axis vertical and the "pitch" a yaw.)"""
    bar = T0_w[:3, 2]
    a = bar - np.dot(bar, [0.0, 0.0, 1.0]) * np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(a)
    if n < 1e-3:
        raise TaskPlanError("grasp_part", stage,
                            "the grasp axis is vertical: a top-down approach "
                            "along it has no closing direction")
    return T0_w[:3, :3].T @ (a / n)


# --------------------------------------------------------------- funnels

@dataclass
class FunnelReport:
    n_sampled: int = 0
    accept_rate: float = 0.0
    n_ik: int = 0
    n_collision_free: int = 0
    n_contained: int = 0
    offenders: dict = field(default_factory=dict)
    n_rounds: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("n_sampled", "accept_rate", "n_ik",
                                              "n_collision_free", "n_contained",
                                              "offenders", "n_rounds")}


def goal_funnel(samples, ik, kin, attached, seeds, containment, offender,
                report: FunnelReport, ik_kw=None) -> list[np.ndarray]:
    """sampled mover body poses -> IK -> collision -> containment of the
    ACHIEVED config (the pour-tea funnel, returning its counts)."""
    goals = []
    counts = Counter(report.offenders)
    for T_body in samples:
        T_ee = T_body @ np.linalg.inv(attached.T_ee_body)
        q, ok = ik.solve_multiseed(T_ee, seeds, **(ik_kw or {}))
        if not ok:
            continue
        report.n_ik += 1
        if kin.in_collision(q):
            counts[offender(kin)] += 1
            continue
        report.n_collision_free += 1
        T_ach = attached.body_pose(kin.fk(q))
        if all(t.contains(T_ach, tol=5e-3) for t in containment):
            goals.append(q)
    report.n_contained += len(goals)
    report.offenders = dict(counts.most_common())
    return goals


def sample_goals(subgoal, path, n, n_max, min_accept, rng, ik, kin, attached,
                 seeds, offender, q_ref, sampler=None) -> tuple[list, FunnelReport]:
    """Intersection sampling + funnel, doubling n on funnel starvation up
    to n_max, but only while the intersection itself is healthy (the
    pour-tea escalation rule). Goals sorted nearest q_ref."""
    if sampler is None:
        from manip_sim.tsr import sample_intersection as sampler
    rep = FunnelReport()
    goals: list[np.ndarray] = []
    while True:
        inter = sampler(subgoal, [path], n=n, rng=rng)
        rep.n_rounds += 1
        rep.n_sampled += len(inter.accepted)
        rep.accept_rate = round(inter.acceptance_rate, 4)
        goals += goal_funnel(inter.accepted, ik, kin, attached, seeds,
                             [path, subgoal], offender, rep)
        if goals or n >= n_max or inter.acceptance_rate < min_accept:
            break
        n = min(2 * n, n_max)
    goals.sort(key=lambda q: float(np.linalg.norm(q - q_ref)))
    return goals, rep


def probe_fractions(ik, kin, T0_ee_grasp, T0_body_now, stage_samples, seeds):
    """Lookahead for one grasp candidate over ANY number of downstream
    stages (pour-tea's probe is fixed at transport x pour): the IK-feasible,
    collision-free fraction of each stage's goal samples under the in-hand
    transform this grasp would measure. Score = product."""
    T_ee_body = np.linalg.inv(T0_ee_grasp) @ T0_body_now
    T_body_ee = np.linalg.inv(T_ee_body)
    fr = []
    for samples in stage_samples:
        if not samples:
            fr.append(0.0)
            continue
        ok = 0
        for T_b in samples:
            q, hit = ik.solve_multiseed(T_b @ T_body_ee, seeds)
            if hit and not kin.in_collision(q):
                ok += 1
        fr.append(ok / len(samples))
    return fr, float(np.prod(fr)) if fr else 0.0


def walk_goals(plan, kin, attached, path_tsrs, q_start, goals, k_max, timeout,
               stage: str, **kw):
    """Plan to the ranked goals in turn (a failure on one goal is
    attrition); all failed -> typed failure with the reason histogram."""
    fails = []
    for k, g in enumerate(goals[:k_max]):
        r = plan(kin, attached, path_tsrs, q_start, g, timeout=timeout, **kw)
        if r.ok:
            return r, k, fails
        fails.append(r.reason)
    raise TaskPlanError("planning", stage,
                        f"planning failed on all {len(fails)} ranked goals",
                        {"reasons": dict(Counter(fails).most_common())})


# ------------------------------------------------------------------ stages

@dataclass(frozen=True)
class MoverStage:
    role: str
    stage: object             # full StageSpec (both objects' parts)
    emission: object          # StageEmission
    synthesized: bool         # starts at a synthesized pose (no emitted carry)


def mover_stages(stage_roles, emissions: dict, synthesized_roles) -> list[MoverStage]:
    """The object-mover stages in order, from emit_constraints.bound_stages
    output ((stage, role) pairs) and the role-keyed emissions. The grasp
    (no passive) is not a mover stage: it is planned from the grasp
    stand-in, as in pour-tea."""
    out = []
    for stage, role in stage_roles:
        if stage.passive is None:
            continue
        if role not in emissions:
            raise TaskPlanError("compile", role,
                                f"emissions artifact has no role {role!r} "
                                f"(has {sorted(emissions)})")
        out.append(MoverStage(role, stage, emissions[role], role in synthesized_roles))
    return out


def decomposition(stage_reports: list[dict]) -> str:
    """Where the big reorientation happened, as a result line."""
    parts = []
    for s in stage_reports:
        free = s.get("reorient_free_reach_deg")
        inside = s.get("reorient_in_stage_deg")
        if free is not None:
            parts.append(f"{s['role']}: {free:.0f} deg in the free-space reach "
                         f"to the synthesized start, {inside:.0f} deg in the stage")
        elif inside is not None:
            parts.append(f"{s['role']}: {inside:.0f} deg inside the stage "
                         "(under its path TSR)")
    return "; ".join(parts)
