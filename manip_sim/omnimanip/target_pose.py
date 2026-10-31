"""OmniManip's target-pose optimization (Pan et al., CVPR 2025, Sec. 3.3, Eqs. 2-5).

    P_ee* = argmin L_C + L_collision + L_path            over ONE end-effector pose
    L_C         = rho(C, P_active, P_passive), P_active = Phi(P_ee)
    L_collision = sum_j max(0, d_min - d(P_ee, O_j))^2
    L_path      = lambda_1 d_trans(P_ee_t, P_ee) + lambda_2 d_rot(P_ee_t, P_ee)

The constraint is the paper's template: "Active axis and passive axis are
aligned. Active point is <d> along passive axis to passive point."

The paper leaves these open, so each is this implementation's choice:

  solver      SLSQP over translation + rotation vector from the stage's start
              pose, warm-started there. A local solve, like the rest.
  rho         squared position error plus (rot_radius * angle)^2, written with
              2 (1 - cos) so its gradient is defined at alignment. In m^2, so it
              is comparable to Eq. 4 with no weight the paper does not have.
  lambda      small enough that L_path breaks ties (spin about the aligned axis)
              rather than trading alignment for a shorter move: at these values
              the bias is about 0.5 mm and 0.3 deg. Larger values trade it away.
  obstacles   per Eq. 4, the end-effector only: the held object is not checked,
              and nothing here knows about reachability, which is why the paper's
              A.5 lists unsolvable IK and execution collisions as failure causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class Constraint:
    """C: the active primitive in its body frame, the passive one in the world."""

    point_active_body: np.ndarray
    axis_active_body: np.ndarray
    point_passive: np.ndarray
    axis_passive: np.ndarray
    d: float                       # signed, along axis_passive


@dataclass(frozen=True)
class Weights:
    lambda_trans: float = 1e-3     # m^2 / m
    lambda_rot: float = 1e-4       # m^2 / rad
    rot_radius: float = 0.1        # m, scales angle into rho's position units
    d_min: float = 0.02            # m


@dataclass
class Obstacle:
    """O_j: one object, as the convex hulls (A x <= b inside) of its geoms."""

    name: str
    hulls: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def distance(self, points: np.ndarray) -> float:
        """Smallest face-plane distance from any point to any hull: negative
        inside, and a lower bound on the true distance near a hull's corners."""
        return min(float(np.min(np.max(points @ A.T - b, axis=1))) for A, b in self.hulls)


def errors(T_ee: np.ndarray, T_ee_body: np.ndarray, c: Constraint) -> tuple[float, float]:
    """(position error m, angle rad) of the active primitive against C."""
    T = T_ee @ T_ee_body
    point = T[:3, :3] @ c.point_active_body + T[:3, 3]
    axis = T[:3, :3] @ c.axis_active_body
    pos = float(np.linalg.norm(point - (c.point_passive + c.d * c.axis_passive)))
    # atan2 stays accurate near alignment, where arccos loses half the digits
    ang = float(np.arctan2(np.linalg.norm(np.cross(axis, c.axis_passive)), axis @ c.axis_passive))
    return pos, ang


def rho(T_ee: np.ndarray, T_ee_body: np.ndarray, c: Constraint, rot_radius: float) -> float:
    T = T_ee @ T_ee_body
    point = T[:3, :3] @ c.point_active_body + T[:3, 3]
    axis = T[:3, :3] @ c.axis_active_body
    e = point - (c.point_passive + c.d * c.axis_passive)
    return float(e @ e + 2.0 * rot_radius ** 2 * (1.0 - axis @ c.axis_passive))


def collision_loss(T_ee: np.ndarray, ee_points: np.ndarray, obstacles: list[Obstacle],
                   d_min: float) -> float:
    """Eq. 4. ee_points are in the end-effector frame."""
    world = ee_points @ T_ee[:3, :3].T + T_ee[:3, 3]
    return float(sum(max(0.0, d_min - o.distance(world)) ** 2 for o in obstacles))


def pose_from_x(x: np.ndarray, T_start: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(x[3:]).as_matrix() @ T_start[:3, :3]
    T[:3, 3] = T_start[:3, 3] + x[:3]
    return T


def losses(x, T_start, T_ee_body, c, ee_points, obstacles, w: Weights) -> dict[str, float]:
    T = pose_from_x(x, T_start)
    trans, rot = np.sqrt(x[:3] @ x[:3] + 1e-12), np.sqrt(x[3:] @ x[3:] + 1e-12)
    return {"constraint": rho(T, T_ee_body, c, w.rot_radius),
            "collision": collision_loss(T, ee_points, obstacles, w.d_min),
            "path": float(w.lambda_trans * trans + w.lambda_rot * rot)}


def solve_target_pose(T_start, T_ee_body, c: Constraint, ee_points, obstacles,
                      w: Weights = Weights(), x0=None, maxiter: int = 500):
    """-> (T_ee*, report). The caller decides what counts as a failed solve."""
    args = (T_start, T_ee_body, c, ee_points, obstacles, w)
    result = minimize(lambda x: sum(losses(x, *args).values()),
                      np.zeros(6) if x0 is None else np.asarray(x0, float), method="SLSQP",
                      options={"maxiter": maxiter, "ftol": 1e-10, "eps": 1e-6})
    T = pose_from_x(result.x, T_start)
    pos, ang = errors(T, T_ee_body, c)
    return T, {"position_error_mm": round(pos * 1000, 2), "angle_error_deg": round(float(np.degrees(ang)), 3),
               "losses": {k: float(f"{v:.3e}") for k, v in losses(result.x, *args).items()},
               "move_m": round(float(np.linalg.norm(result.x[:3])), 3),
               "rotation_deg": round(float(np.degrees(np.linalg.norm(result.x[3:]))), 1),
               "n_iterations": int(result.nit), "solver_message": str(result.message)}
