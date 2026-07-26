"""Task Space Regions (Berenson et al., IJRR 2011) — the geometric core.

A TSR is (T0_w, Tw_e, Bw):

    T0_w  world pose of the TSR reference frame w (in our architecture, a
          task frame on the *passive* object, or a world-anchored frame for
          gravity-referenced constraints like "stay upright"),
    Tw_e  fixed offset from w to the *constrained* frame e when the
          displacement is zero (in our architecture, e is the active
          object's body frame; Tw_e encodes which feature of the active
          object sits at w — e.g. inv(T_body_spout_tip) says "the spout tip
          coincides with w"),
    Bw    6x2 per-axis bounds [lo, hi] over displacements
          (x, y, z, roll, pitch, yaw), expressed in w.

Displacement of a candidate pose T0_e:

    Tw = inv(T0_w) @ T0_e @ inv(Tw_e)
    d  = [trans(Tw); rpy(Tw)]           rotation = Rz(yaw) Ry(pitch) Rx(roll)
                                        (scipy extrinsic "xyz")

The RPY decomposition is not unique: both the +-2pi shifts of each angle and
the dual solution (r+pi, pi-p, y+pi) represent the same rotation. displacement()
returns the representative *nearest to Bw* (Berenson's min-over-solutions
distance), so bounds like yaw in [170deg, 190deg] behave correctly.

Everything here is numpy/scipy only — no simulator dependency — so it is
unit-testable in isolation. Two deliberate scope limits, documented so they
do not surprise later:

  * sample() is uniform in displacement coordinates, which is not the Haar
    measure on SO(3). For the tight rotational bounds our subgoal TSRs use,
    the bias is negligible; if a future TSR frees all three rotations,
    sample the rotation with R.random() instead.
  * Bw intervals on angles wider than 2pi are treated as free.

Quaternions never appear in this file's public API; poses are 4x4 numpy
arrays. Use manip_sim.frames / manip_sim.state to build them from MuJoCo's
wxyz quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------------------------- helpers

FREE_ROT: tuple[float, float] = (-np.pi, np.pi)
FREE_TRANS: tuple[float, float] = (-np.inf, np.inf)

_TWO_PI = 2.0 * np.pi


def make_pose(pos=(0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> np.ndarray:
    """(pos[3], 3x3 rot) -> 4x4 homogeneous pose."""
    T = np.eye(4)
    if rot is not None:
        T[:3, :3] = rot
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def pose_from_pos_quat_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo-convention (pos, wxyz quat) -> 4x4. Matches state.PoseReader."""
    return make_pose(pos, R.from_quat(quat_wxyz, scalar_first=True).as_matrix())


def rpy_to_matrix(rpy) -> np.ndarray:
    """[roll, pitch, yaw] -> 3x3, rotation = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return R.from_euler("xyz", rpy).as_matrix()


def displacement_to_pose(d: np.ndarray) -> np.ndarray:
    """6-vector displacement -> 4x4 pose of e relative to w (pre-Tw_e)."""
    return make_pose(d[:3], rpy_to_matrix(d[3:]))


def _wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return a - _TWO_PI * np.floor((a + np.pi) / _TWO_PI)


def _interval_excess(v: float, lo: float, hi: float) -> float:
    """Signed distance outside [lo, hi]; 0 inside."""
    if v < lo:
        return v - lo
    if v > hi:
        return v - hi
    return 0.0


def _nearest_angle_rep(a: float, lo: float, hi: float) -> tuple[float, float]:
    """Among {a, a+2pi, a-2pi}, the representative closest to [lo, hi].

    Returns (representative, excess). Handles bounds authored outside
    (-pi, pi], e.g. yaw in [170deg, 190deg].
    """
    best_v, best_e = a, _interval_excess(a, lo, hi)
    for shift in (_TWO_PI, -_TWO_PI):
        v = a + shift
        e = _interval_excess(v, lo, hi)
        if abs(e) < abs(best_e):
            best_v, best_e = v, e
    return best_v, best_e


def _rpy_solutions(Rm: np.ndarray) -> list[np.ndarray]:
    """Both RPY decompositions of a rotation matrix (extrinsic xyz)."""
    r, p, y = R.from_matrix(Rm).as_euler("xyz")
    primary = np.array([r, p, y])
    dual = np.array([_wrap_pi(r + np.pi), _wrap_pi(np.pi - p), _wrap_pi(y + np.pi)])
    return [primary, dual]


# ------------------------------------------------------------------------- TSR


def bounds(
    x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0
) -> np.ndarray:
    """Build a 6x2 Bw. Each argument is either a scalar (axis pinned at that
    displacement) or a (lo, hi) pair. Use FREE_ROT / FREE_TRANS for free axes.

        bounds(z=(0.03, 0.08), yaw=FREE_ROT)
    """
    rows = []
    for v in (x, y, z, roll, pitch, yaw):
        if np.isscalar(v):
            rows.append((float(v), float(v)))
        else:
            lo, hi = float(v[0]), float(v[1])
            if hi < lo:
                raise ValueError(f"bound hi < lo: {(lo, hi)}")
            rows.append((lo, hi))
    return np.array(rows)


@dataclass
class TSR:
    """A Task Space Region. See module docstring for conventions."""

    T0_w: np.ndarray                      # 4x4 world pose of frame w
    Bw: np.ndarray                        # 6x2 bounds
    Tw_e: np.ndarray = field(default_factory=lambda: np.eye(4))
    name: str = ""

    def __post_init__(self):
        self.T0_w = np.asarray(self.T0_w, dtype=float)
        self.Tw_e = np.asarray(self.Tw_e, dtype=float)
        self.Bw = np.asarray(self.Bw, dtype=float)
        assert self.T0_w.shape == (4, 4) and self.Tw_e.shape == (4, 4)
        assert self.Bw.shape == (6, 2)
        self._T0_w_inv = np.linalg.inv(self.T0_w)
        self._Tw_e_inv = np.linalg.inv(self.Tw_e)
        # angle axes wider than 2pi are fully free
        self._rot_free = (self.Bw[3:, 1] - self.Bw[3:, 0]) >= _TWO_PI - 1e-9

    # -- core queries -------------------------------------------------------

    def displacement(self, T0_e: np.ndarray) -> np.ndarray:
        """6-vector displacement of candidate pose T0_e, using the RPY
        representative nearest to Bw (min over dual solution and 2pi shifts).
        """
        Tw = self._T0_w_inv @ T0_e @ self._Tw_e_inv
        t = Tw[:3, 3]
        best_d, best_cost = None, np.inf
        for sol in _rpy_solutions(Tw[:3, :3]):
            d = np.empty(6)
            d[:3] = t
            excess_sq = 0.0
            for i in range(3):
                excess_sq += _interval_excess(t[i], *self.Bw[i]) ** 2
            for i in range(3):
                if self._rot_free[i]:
                    d[3 + i] = sol[i]
                    continue
                rep, exc = _nearest_angle_rep(sol[i], *self.Bw[3 + i])
                d[3 + i] = rep
                excess_sq += exc ** 2
            if excess_sq < best_cost:
                best_cost, best_d = excess_sq, d
        return best_d

    def excess(self, T0_e: np.ndarray) -> np.ndarray:
        """Per-axis signed excess of the displacement outside Bw (zeros iff
        contained). This 6-vector is the constraint residual the planner's
        projection operator drives to zero."""
        d = self.displacement(T0_e)
        return np.array([_interval_excess(d[i], *self.Bw[i]) for i in range(6)])

    def distance(self, T0_e: np.ndarray) -> float:
        """Norm of the per-axis excess outside Bw (0 iff contained).
        Note: mixes meters and radians; fine as a feasibility check,
        weight the axes if you need a calibrated metric."""
        return float(np.linalg.norm(self.excess(T0_e)))

    def contains(self, T0_e: np.ndarray, tol: float = 1e-9) -> bool:
        return self.distance(T0_e) <= tol

    def project(self, T0_e: np.ndarray) -> np.ndarray:
        """Nearest-in-displacement-coordinates pose inside the TSR: clamp the
        displacement into Bw and recompose. This is the operator CBiRRT calls
        on every tree extension (there via IK on the projected pose)."""
        d = self.displacement(T0_e)
        lo, hi = self.Bw[:, 0], self.Bw[:, 1]
        d_cl = np.clip(d, lo, hi)
        return self.T0_w @ displacement_to_pose(d_cl) @ self.Tw_e

    # -- sampling -----------------------------------------------------------

    def sample_displacement(self, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.Bw[:, 0].copy(), self.Bw[:, 1].copy()
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError(
                f"TSR '{self.name}': cannot sample unbounded axes. Free "
                "translations are for path/containment TSRs; sample the "
                "subgoal TSR and check containment in this one instead."
            )
        return lo + rng.uniform(size=6) * (hi - lo)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A world pose of the constrained frame e, uniform over Bw."""
        return self.T0_w @ displacement_to_pose(self.sample_displacement(rng)) @ self.Tw_e

    def zero(self) -> np.ndarray:
        """The pose at zero displacement (w coincident with its offset e)."""
        return self.T0_w @ self.Tw_e


# ----------------------------------------------------------------- intersection


@dataclass
class IntersectionReport:
    accepted: list[np.ndarray]
    n_tried: int
    per_constraint_rejections: dict[str, int]

    @property
    def acceptance_rate(self) -> float:
        return len(self.accepted) / self.n_tried if self.n_tried else 0.0

    def summary(self) -> str:
        rej = ", ".join(f"{k}: {v}" for k, v in self.per_constraint_rejections.items())
        return (
            f"{len(self.accepted)} accepted / {self.n_tried} tried "
            f"(rate {self.acceptance_rate:.3f}); rejections by [{rej}]"
        )


def sample_intersection(
    sampler: TSR,
    constraints: list[TSR],
    n: int,
    rng: np.random.Generator,
    max_tries: int | None = None,
    tol: float = 1e-9,
) -> IntersectionReport:
    """Rejection-sample the intersection: draw from `sampler` (typically the
    subgoal TSR — the tighter, bounded one), keep samples contained in every
    TSR in `constraints` (typically the stage's path TSR, plus any auxiliary
    conjunct such as an upright constraint).

    All TSRs must constrain the *same physical frame e* (same active-object
    body frame); they may place w and Tw_e differently.

    The report's acceptance rate is a first-class diagnostic: a rate near
    zero on a VLM-emitted pair is the earliest machine-checkable signal that
    the emission is internally inconsistent (subgoal not reachable on the
    path manifold) and should be escalated for repair, not brute-forced.
    """
    max_tries = max_tries or 50 * n
    rejections = {c.name or f"constraint_{i}": 0 for i, c in enumerate(constraints)}
    accepted: list[np.ndarray] = []
    tried = 0
    while len(accepted) < n and tried < max_tries:
        T = sampler.sample(rng)
        tried += 1
        ok = True
        for i, c in enumerate(constraints):
            if not c.contains(T, tol=tol):
                rejections[c.name or f"constraint_{i}"] += 1
                ok = False
                break
        if ok:
            accepted.append(T)
    return IntersectionReport(accepted, tried, rejections)