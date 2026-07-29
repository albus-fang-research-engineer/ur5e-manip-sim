"""Stage-1 (grasp) machinery: proposer stand-in, TSR classifier, lookahead.

Architecture recap (from the TSR design discussions): stage 1 is the one
stage where the TSR acts as a CLASSIFIER over externally proposed grasp
poses, not as a sampler — AnyGrasp proposes in the real pipeline, and the
grasp TSR on the handle frame scores/filters the proposals. The grasp that
survives classification + IK + collision is then ranked by the LOOKAHEAD
FEASIBILITY PROBE: each candidate's implied in-hand transform T_ee_body is
propagated through the downstream stage TSRs (transport subgoal∩path, pour
subgoal) and scored by the fraction of IK-feasible samples it admits. The
probe — not the grasp TSR — is what prefers grasps that keep stages 2-3
reachable; no cup-relative structure leaks into the stage-1 constraint.

Two refinements over pour_stages.grasp_tsr, which this module completes
rather than replaces:

  * Tw_e made explicit. grasp_tsr leaves Tw_e = I, i.e. nominal gripper
    frame == handle frame — but the grip site's +z is the APPROACH axis
    (out of the palm), so identity puts the palm facing up, approaching the
    handle from below. handle_grasp_tsr inserts the nominal grip-in-handle
    transform: approach pitched down from horizontal by `elevation` (an
    oblique over-the-handle grasp; the UR5e reachability scan showed pure
    horizontal approaches are mostly outside the wrist's reach at this
    table pose), closing axis horizontal and perpendicular to the handle
    bar. Displacement axes keep grasp_tsr's semantics: z slides along the
    bar, yaw spins the approach azimuth about the bar within wrap_rot,
    roll/pitch tight, x/y lateral tight.
  * Parallel-jaw wrist symmetry. A 180 deg flip about the approach axis is
    the same physical grasp; wrist_flip() exposes it so IK can try both
    branches. Proposals are emitted canonical (unflipped); the flip is an
    IK-time trick, not a distinct proposal.

Everything below planning-time IK is numpy-only; ArmKinematics / MinkIK are
passed in, never imported, so this module stays simulator-agnostic like
tsr.py and frames.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from .frames import Frame
from .tsr import (
    FREE_ROT,
    FREE_TRANS,
    TSR,
    bounds,
    displacement_to_pose,
    make_pose,
)

# ------------------------------------------------------------- nominal frames


def nominal_grip_in_handle(
    approach_h: np.ndarray, elevation: float
) -> np.ndarray:
    """Nominal gripper pose expressed IN THE HANDLE FRAME (this is Tw_e).

    Default elevation 35 deg: the shallowest broadly IK-reachable band at
    the calibrated handle pose (pure horizontal approaches are outside the
    UR5e wrist's workspace with the handle ~0.83 m from the base), and far
    enough from vertical that the bar seats between the fingers instead of
    being pinched end-on.

    approach_h  horizontal approach direction in handle-frame coordinates
                (component along the handle axis is dropped); by convention
                it points from outside the handle toward the object body.
    elevation   downward pitch of the approach from horizontal, radians
                (0 = horizontal, pi/2 = straight down).

    Gripper axes, measured on robosuite's Robotiq85 (finger bodies sit at
    site-frame x = +-0.069 and displace along x when closing): +z =
    approach (grip-site convention: out of the palm), +x = CLOSING axis,
    kept horizontal and perpendicular to the handle bar so the pads
    straddle the bar, +y = z cross x. An earlier revision assumed the
    closing axis was +y; the resulting 90-deg-rotated closing plane is
    exactly the 'grasping it vertically' artifact — at steep elevations
    the pads still happened to catch the bar, at side-grasp elevations
    they sweep past it. Origin at the handle center.
    """
    z = np.asarray(approach_h, dtype=float).copy()
    z[2] = 0.0
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("approach direction is parallel to the handle axis")
    z /= n
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))     # closing axis: horizontal,
    x /= np.linalg.norm(x)                         # perpendicular to the bar
    z = R.from_rotvec(-elevation * x).apply(z)     # pitch approach downward
    y = np.cross(z, x)
    return make_pose(np.zeros(3), np.column_stack([x, y, z]))


def wrist_flip(T0_ee: np.ndarray) -> np.ndarray:
    """The parallel-jaw twin: same fingers on the same bar, wrist rotated
    180 deg about the approach (+z) axis. Try both at IK time."""
    F = np.diag([-1.0, -1.0, 1.0, 1.0])
    return T0_ee @ F


# ------------------------------------------------------------------ grasp TSR


def handle_grasp_tsr(
    T0_body: np.ndarray,
    handle: Frame,
    approach_h: np.ndarray,
    elevation: float = np.deg2rad(35.0),
    slide: float = 0.02,
    lateral_tol: float = 0.005,
    wrap_rot: tuple[float, float] = (-np.pi / 4, np.pi / 4),
    rp_tol: float = 0.09,
) -> TSR:
    """Grasp-region TSR on the handle frame with the gripper nominal made
    explicit in Tw_e (see module docstring). e = the GRIPPER grip-site
    frame — the stage-1 exception to the body-frame convention, because the
    grasp *defines* T_ee_body rather than consuming it.

    Role: CLASSIFIER. Feed externally proposed gripper poses to
    .contains()/.distance(); do not sample from it as the goal generator
    (sampling it inside propose_handle_grasps to *synthesize* proposals in
    sim is the stand-in for AnyGrasp, not a change of role — the classifier
    still sees arbitrary poses and rejects the off-manifold ones).
    """
    return TSR(
        T0_w=T0_body @ handle.T(),
        Tw_e=nominal_grip_in_handle(approach_h, elevation),
        Bw=bounds(
            x=(-lateral_tol, lateral_tol),
            y=(-lateral_tol, lateral_tol),
            z=(-slide, slide),
            roll=(-rp_tol, rp_tol),
            pitch=(-rp_tol, rp_tol),
            yaw=wrap_rot,
        ),
        name="grasp/handle",
    )


# ------------------------------------------------------------------ proposals


@dataclass
class GraspProposal:
    T0_ee: np.ndarray            # proposed world grip-site pose
    provenance: str              # "handle" | "junk" — sim bookkeeping only;
                                 # the classifier never sees this field
    tsr_distance: float = np.nan


def propose_handle_grasps(
    tsr: TSR,
    rng: np.random.Generator,
    n: int = 80,
    azimuth_span: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    slide_span: float = 0.035,
    lateral_sigma: float = 0.004,
    rp_sigma: float = 0.05,
    junk_frac: float = 0.25,
    junk_points: list[np.ndarray] | None = None,
) -> list[GraspProposal]:
    """AnyGrasp stand-in: emit a DIVERSE cloud of candidate gripper poses
    around the handle — deliberately wider than the TSR bounds on every
    axis (azimuth beyond wrap_rot, slide beyond the bar tolerance, lateral
    and roll/pitch noise) plus a fraction of junk proposals at other body
    locations (spout, lid, ...) — so that classification against the grasp
    TSR is a real filter with a nontrivial rejection rate, exercising the
    classifier role end to end. Swap this function for
    perception.grasp_client.GraspClient output and nothing downstream
    changes.
    """
    out: list[GraspProposal] = []
    n_junk = int(round(n * junk_frac)) if junk_points else 0
    for _ in range(n - n_junk):
        d = np.zeros(6)
        d[0:2] = rng.normal(0.0, lateral_sigma, size=2)
        d[2] = rng.uniform(-slide_span, slide_span)
        d[3:5] = rng.normal(0.0, rp_sigma, size=2)
        d[5] = rng.uniform(*azimuth_span)
        T = tsr.T0_w @ displacement_to_pose(d) @ tsr.Tw_e
        out.append(GraspProposal(T, "handle"))
    for _ in range(n_junk):
        p = junk_points[rng.integers(len(junk_points))]
        T = make_pose(p + rng.normal(0, 0.01, 3),
                      R.random(random_state=rng).as_matrix())
        out.append(GraspProposal(T, "junk"))
    rng.shuffle(out)
    return out


def classify_grasps(
    tsr: TSR, proposals: list[GraspProposal], tol: float = 1e-6
) -> tuple[list[GraspProposal], dict[str, int]]:
    """TSR-as-classifier: keep proposals contained in the grasp TSR.
    Returns (survivors, tally by provenance for the funnel printout)."""
    kept: list[GraspProposal] = []
    tally = {"handle_kept": 0, "handle_rejected": 0,
             "junk_kept": 0, "junk_rejected": 0}
    for p in proposals:
        p.tsr_distance = tsr.distance(p.T0_ee)
        ok = p.tsr_distance <= tol
        key = f"{p.provenance}_{'kept' if ok else 'rejected'}"
        tally[key] = tally.get(key, 0) + 1     # any provenance, e.g. anygrasp
        if ok:
            kept.append(p)
    return kept, tally


# --------------------------------------------------------------- free motion


def free_tsr(name: str = "free") -> TSR:
    """The trivially satisfied TSR: every axis free. Passing [free_tsr()]
    as the path constraint turns plan_constrained into plain bidirectional
    RRT — projection is a no-op because the excess is identically zero —
    which is how stage 1 plans its unconstrained reach-to-pregrasp motion
    without a second planner code path."""
    return TSR(
        T0_w=np.eye(4),
        Bw=bounds(x=FREE_TRANS, y=FREE_TRANS, z=FREE_TRANS,
                  roll=FREE_ROT, pitch=FREE_ROT, yaw=FREE_ROT),
        name=name,
    )


# ------------------------------------------------------------ lookahead probe


@dataclass
class ProbeReport:
    frac_transport: float
    frac_pour: float
    n_transport: int
    n_pour: int

    @property
    def score(self) -> float:
        """Both downstream stages must be feasible: the product punishes a
        grasp that aces transport but dead-ends the pour."""
        return self.frac_transport * self.frac_pour

    def summary(self) -> str:
        return (f"transport {self.frac_transport:.2f} "
                f"({self.n_transport} probes), "
                f"pour {self.frac_pour:.2f} ({self.n_pour} probes), "
                f"score {self.score:.3f}")


@dataclass
class ProbeContext:
    """Shared body-pose sample sets, drawn ONCE so every grasp candidate is
    scored against the same probes (fair ranking, cheaper too).

    transport_samples  stage-2 subgoal INTERSECT path body poses
    pour_samples       stage-3 subgoal body poses, built at a common entry
                       pose (a representative transport sample) — an
                       approximation shared by all candidates, fine for
                       ranking
    """
    transport_samples: list[np.ndarray]
    pour_samples: list[np.ndarray]
    seeds: list[np.ndarray] = field(default_factory=list)


def lookahead_probe(
    ik,                                   # planning.MinkIK
    kin,                                  # planning.ArmKinematics
    T0_ee_grasp: np.ndarray,
    T0_body_now: np.ndarray,
    ctx: ProbeContext,
) -> ProbeReport:
    """Score one surviving grasp candidate by propagating its implied
    in-hand transform through the downstream TSR samples:

        T_ee_body = inv(T0_ee_grasp) @ T0_body_now      (what the grasp
                                                         would measure)
        eef target for body sample T_b = T_b @ inv(T_ee_body)

    and counting the IK-feasible, collision-free fraction. This is the
    inter-stage coupling: cup-facing-away teapots, awkward azimuths, etc.
    are penalized HERE, not encoded into the stage-1 TSR.
    """
    T_ee_body = np.linalg.inv(T0_ee_grasp) @ T0_body_now
    T_body_ee = np.linalg.inv(T_ee_body)

    def frac(samples: list[np.ndarray]) -> float:
        if not samples:
            return 0.0
        ok = 0
        for T_b in samples:
            q, hit = ik.solve_multiseed(T_b @ T_body_ee, ctx.seeds)
            if hit and not kin.in_collision(q):
                ok += 1
        return ok / len(samples)

    return ProbeReport(
        frac_transport=frac(ctx.transport_samples),
        frac_pour=frac(ctx.pour_samples),
        n_transport=len(ctx.transport_samples),
        n_pour=len(ctx.pour_samples),
    )