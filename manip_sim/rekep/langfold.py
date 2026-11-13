"""LangFold's pour-tea TSRs as ReKep constraints, so ReKep's solvers can plan
under LangFold's constraints and only the solver differs from LangFold's plans.

Each constraint is a TSR's distance (pour_stages.TSR.distance: the norm of the
per-axis excess outside the bounds, metres and radians, 0 inside) at the held
object's pose, which is recovered from its keypoints. That is ReKep's contract:
a function of (end_effector, keypoints), satisfied at <= 0.

The TSRs are the ones scripts/plan_pour_tea.py plans the Table 1 runs with
(ours10g: --upright-deg 10.5, hand-authored frames, tilt 95 deg) and are built
when it builds them: transport from the start poses, pour from the teapot pose
where transport ended. LangFold's planner also ranks transport goals by whether
the pour works from them and accepts any pour goal over the rim; those belong
to its solver and are not here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..frames import Symbols
from ..pour_stages import TaskFrames

UPRIGHT_DEG = 10.5                   # plan_pour_tea.py --upright-deg, Table 1
TILT_DEG = 95.0                      # plan_pour_tea.py --tilt-deg default
TRANSPORT_HEIGHT = (0.04, 0.10)      # plan_pour_tea.py stage 2, hand arm
Z_CORRIDOR = (-0.02, 0.45)


def pose_from_points(world: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Rigid pose T with world ~ T @ body (Kabsch), from >= 3 non-collinear points."""
    cw, cb = world.mean(0), body.mean(0)
    U, _, Vt = np.linalg.svd((body - cb).T @ (world - cw))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    T = np.eye(4)
    T[:3, :3] = Vt.T @ D @ U.T
    T[:3, 3] = cw - T[:3, :3] @ cb
    return T


def tsr_constraint(tsr, idx: list[int], body: np.ndarray) -> Callable:
    """ReKep constraint: tsr's distance at the pose of the body whose points
    keypoints[idx] are (body-frame coordinates `body`)."""
    def constraint(end_effector, keypoints):
        return tsr.distance(pose_from_points(keypoints[idx], body))
    constraint.__name__ = tsr.name
    return constraint


class PourConstraints:
    """stage index (ReKep's, 1 transport, 2 pour) and held body pose at the
    stage's start -> (subgoal constraints, path constraints)."""

    def __init__(self, teapot: Symbols, mug: Symbols, T0_teapot: np.ndarray,
                 T0_mug: np.ndarray, idx: list[int], body: np.ndarray):
        self.frames = TaskFrames(spout_tip=teapot.frame("spout_tip", "pour_axis"),
                                 tilt_frame=teapot.frame("spout_tip", "tilt_axis", secondary="pour_axis"),
                                 opening=mug.frame("opening_center", "up_axis"),
                                 rim_margin=mug.quantities.get("rim_radius", 0.04) * 0.5)
        self.T0_teapot, self.T0_mug = T0_teapot, T0_mug
        self.idx, self.body = list(idx), np.asarray(body, float)

    def tsrs(self, index: int, T_held: np.ndarray):
        if index == 1:
            return self.frames.transport(self.T0_teapot, self.T0_mug, z_corridor=Z_CORRIDOR,
                                         upright_tol=np.deg2rad(UPRIGHT_DEG), height=TRANSPORT_HEIGHT)
        if index == 2:
            return self.frames.pour(T_held, self.T0_mug, np.deg2rad(TILT_DEG))
        raise ValueError(f"no LangFold constraints for stage index {index}")

    def __call__(self, index: int, T_held: np.ndarray) -> tuple[list[Callable], list[Callable]]:
        pair = self.tsrs(index, T_held)
        return ([tsr_constraint(pair.subgoal, self.idx, self.body)],
                [tsr_constraint(pair.path, self.idx, self.body)])
