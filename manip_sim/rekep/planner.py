"""ReKep's stage loop, planned once and written as a joint path.

Upstream re-solves while executing, from live keypoints, and backtracks when a
path constraint breaks. This runs the same subgoal and path solvers once per
stage, from the state the previous stage planned, so it is OPEN-LOOP ReKep: it
suits undisturbed comparisons and says nothing about reactivity.

Other differences from upstream's main.py, each because of this sim:

  grasp       upstream backs off grasp_depth/2 and then moves grasp_depth, which
              ends past the keypoint; Fetch's EE frame allows that. The grip
              site sits between the pads, so this ends at the solved pose.
  grasping    is a friction grasp in execution, not OmniGibson's assisted mode.
              Planning assumes it holds; execute_task_plan measures whether it did.
  failures    upstream executes whatever the solvers return. Here a solve must
              pass its constraints and IK or the stage retries with a new seed,
              then raises TaskPlanError.
  joint path  upstream sends poses to an operational-space controller. The
              executor tracks joint paths, so poses become a continuous IK chain
              (bisecting a pose step whose joint step exceeds the limit).
  collisions  ReKep's cost is soft and skips most arm links, so plans are not
              rejected for collision: the report counts colliding waypoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from ..frames import load_symbols
from ..planning import ArmKinematics, AttachedArmKinematics, AttachedObject
from ..task_plan import TaskPlanError
from .adapter import (COLLISION_POINT_PREFIXES, ReKepIK, body_pose, collision_vertices,
                      rekep_to_site, scene_sdf, site_to_rekep)
from .program import load_program
from .vendor import (PathSolver, SubgoalSolver, get_linear_interpolation_steps,
                     spline_interpolate_poses)

UPSTREAM_COMMIT = "63c43fd"
FINGER_OFFSET = 0.05                 # m, virtual keypoint along the closing axis


@dataclass
class Config:
    """Solver values are upstream's configs/config.yaml, except the tolerance:
    its 1e-4 rejects sub-millimetre misses, and the costs here are in metres."""

    sampling_maxfun: int = 5000
    minimizer_maxiter: int = 200
    max_collision_points: int = 60
    constraint_tolerance: float = 0.005
    opt_pos_step_size: float = 0.20
    opt_rot_step_size: float = 0.78
    opt_interpolate_pos_step_size: float = 0.02
    opt_interpolate_rot_step_size: float = 0.10
    interpolate_pos_step_size: float = 0.05
    interpolate_rot_step_size: float = 0.34
    sdf_voxel_size: float = 0.01
    grasp_standoff: float = 0.05     # upstream's grasp_depth / 2
    workspace_height: float = 0.55   # m above the table
    workspace_xy: tuple[float, float] = (0.45, 0.55)
    attempts: int = 5                # seeds per stage
    max_joint_step: float = 0.35     # rad between IK chain waypoints
    max_bisections: int = 4
    path_pos_tol: float = 1e-3       # tracked waypoints, tighter than the cost's IK
    path_rot_tol: float = 5e-3
    seed: int = 0


class _ChainFailure(RuntimeError):
    pass


@dataclass
class Plan:
    path: np.ndarray
    stage_ids: np.ndarray
    stage_labels: np.ndarray
    q_home: np.ndarray
    q_grasp: np.ndarray
    T_ee_body: np.ndarray
    grasped: str
    T0: dict[str, np.ndarray]
    target: dict | None


class ReKepPlanner:
    def __init__(self, env, scene, program_dir, cfg: Config | None = None):
        self.env, self.scene, self.cfg = env, scene, cfg or Config()
        self.kin = ArmKinematics(env)
        self.q_home = self.kin.data.qpos[self.kin.qpos_ids].copy()
        self.T0 = {n: body_pose(self.kin, n) for n in scene.objects}
        self.symbols = {n: load_symbols(scene.asset_dirs[n]) for n in scene.objects}
        self.held: str | None = None
        self.T_ee_body: np.ndarray | None = None
        self.program = load_program(program_dir, self._grasping_cost)
        self.keypoints_body = [
            k.xyz if k.xyz is not None else self.symbols[k.object].points[k.point]
            for k in self.program.keypoints]
        self.bounds_min, self.bounds_max = self._workspace()
        self.ik = ReKepIK(self.kin, self.q_home)
        solver_cfg = {
            "bounds_min": self.bounds_min, "bounds_max": self.bounds_max,
            "sampling_maxfun": self.cfg.sampling_maxfun,
            "max_collision_points": self.cfg.max_collision_points,
            "constraint_tolerance": self.cfg.constraint_tolerance,
            "minimizer_options": {"maxiter": self.cfg.minimizer_maxiter},
            "opt_pos_step_size": self.cfg.opt_pos_step_size,
            "opt_rot_step_size": self.cfg.opt_rot_step_size,
            "opt_interpolate_pos_step_size": self.cfg.opt_interpolate_pos_step_size,
            "opt_interpolate_rot_step_size": self.cfg.opt_interpolate_rot_step_size,
        }
        self.subgoal_solver = SubgoalSolver(solver_cfg, self.ik, self.q_home)
        self.path_solver = PathSolver(solver_cfg, self.ik, self.q_home)
        self.report: dict = {"planner": "rekep", "mode": "open-loop",
                             "upstream_commit": UPSTREAM_COMMIT,
                             "program": str(program_dir), "stages": []}

    def _workspace(self) -> tuple[np.ndarray, np.ndarray]:
        p, c = self.program, self.cfg
        lo = np.array([-c.workspace_xy[0], -c.workspace_xy[1], self.scene.table_top_z])
        hi = np.array([c.workspace_xy[0], c.workspace_xy[1],
                       self.scene.table_top_z + c.workspace_height])
        return (p.bounds_min if p.bounds_min is not None else lo,
                p.bounds_max if p.bounds_max is not None else hi)

    # ------------------------------------------------------------ plan state

    def _grasping_cost(self, keypoint_idx: int) -> float:
        """0 if the keypoint's object is held in the planned state, else 1."""
        return 0.0 if self.program.keypoints[keypoint_idx].object == self.held else 1.0

    def _object_pose(self, name: str, T_site: np.ndarray) -> np.ndarray:
        return T_site @ self.T_ee_body if name == self.held else self.T0[name]

    def _keypoints(self, T_site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[ee, *scene keypoints, finger] and upstream's movable mask."""
        scene = [self._object_pose(k.object, T_site)[:3, :3] @ b
                 + self._object_pose(k.object, T_site)[:3, 3]
                 for k, b in zip(self.program.keypoints, self.keypoints_body)]
        finger = T_site[:3, 3] + FINGER_OFFSET * T_site[:3, 0]
        points = np.vstack([T_site[:3, 3], *scene, finger])
        mask = np.array([True, *(k.object == self.held for k in self.program.keypoints), True])
        return points, mask

    def _collision_points(self, q: np.ndarray, T_site: np.ndarray) -> np.ndarray:
        self.kin.set_q(q)
        points = [collision_vertices(self.kin, COLLISION_POINT_PREFIXES)]
        if self.held is not None:
            verts = collision_vertices(self.kin, (self.held,))     # at T0 in the scratch world
            T = self._object_pose(self.held, T_site) @ np.linalg.inv(self.T0[self.held])
            points.append(verts @ T[:3, :3].T + T[:3, 3])
        self.kin.set_q(self.q_home)
        return np.vstack(points)

    # ------------------------------------------------------------ joint path

    def _ik_step(self, T_a, T_b, q_a, depth) -> list[np.ndarray]:
        c = self.cfg
        r = self.ik.solve_site(T_b, c.path_pos_tol, c.path_rot_tol, 200, q_a)
        if not r.success:
            raise _ChainFailure(f"no IK within tolerance ({r.position_error * 1000:.1f} mm)")
        step = float(np.max(np.abs(r.cspace_position - q_a)))
        if step <= c.max_joint_step:
            return [r.cspace_position]
        if depth == 0:
            raise _ChainFailure(f"joint step {step:.2f} rad survives "
                                f"{c.max_bisections} bisections")
        T_m = np.eye(4)
        T_m[:3, 3] = (T_a[:3, 3] + T_b[:3, 3]) / 2
        T_m[:3, :3] = Slerp([0, 1], R.from_matrix([T_a[:3, :3], T_b[:3, :3]]))(0.5).as_matrix()
        head = self._ik_step(T_a, T_m, q_a, depth - 1)
        return head + self._ik_step(T_m, T_b, head[-1], depth - 1)

    def _ik_chain(self, poses_site: list[np.ndarray], q_start: np.ndarray) -> np.ndarray:
        chain = [q_start]
        for T_a, T_b in zip(poses_site[:-1], poses_site[1:]):
            chain += self._ik_step(T_a, T_b, chain[-1], self.cfg.max_bisections)
        return np.array(chain)

    def _dense_poses(self, T_start: np.ndarray, rekep_path: np.ndarray) -> list[np.ndarray]:
        c = self.cfg
        control = np.vstack([site_to_rekep(T_start), rekep_path])
        steps = get_linear_interpolation_steps(control[0], control[-1],
                                               c.interpolate_pos_step_size,
                                               c.interpolate_rot_step_size)
        dense = spline_interpolate_poses(control, steps)
        return [T_start] + [rekep_to_site(p) for p in dense[1:]]

    # ---------------------------------------------------------------- stages

    def _violations(self, debug: dict, *keys) -> list[float]:
        out: list[float] = []
        for key in keys:
            out += [float(v) for v in (debug.get(key) or [])]
        return out

    def _solve_stage(self, index: int, label: str, T_start, q_start) -> tuple[np.ndarray, np.ndarray, dict]:
        """-> (joint path including q_start, T_site at its end, report row)."""
        c, stage = self.cfg, self.program.stages[index]
        exclude = (self.held,) if self.held else ()
        sdf = scene_sdf(self.kin, self.bounds_min, self.bounds_max, c.sdf_voxel_size, exclude)
        points, mask = self._keypoints(T_start)
        collision = self._collision_points(q_start, T_start)
        start = site_to_rekep(T_start)
        failures = []
        for attempt in range(c.attempts):
            np.random.seed(c.seed + 1000 * index + attempt)
            t0 = time.time()
            sub, dbg = self.subgoal_solver.solve(
                start, points, mask, stage.subgoal, stage.path, sdf, collision,
                stage.is_grasp, q_start, from_scratch=True)
            viol = self._violations(dbg, "subgoal_violation", "path_violation")
            if not dbg["ik_feasible"] or any(v > c.constraint_tolerance for v in viol):
                failures.append({"attempt": attempt, "kind": "subgoal", "ik": bool(dbg["ik_feasible"]),
                                 "violations": np.round(viol, 4).tolist()})
                continue
            T_goal = rekep_to_site(sub)
            T_pre = T_goal.copy()
            if stage.is_grasp:                 # back off along the approach axis
                T_pre[:3, 3] -= T_goal[:3, 2] * c.grasp_standoff
            path, pdbg = self.path_solver.solve(
                start, site_to_rekep(T_pre), points, mask, stage.path, sdf, collision,
                q_start, from_scratch=True)
            pviol = self._violations(pdbg, "path_violation")
            if not np.all(pdbg["ik_feasible"]) or any(v > c.constraint_tolerance for v in pviol):
                failures.append({"attempt": attempt, "kind": "path",
                                 "ik": np.asarray(pdbg["ik_feasible"]).tolist(),
                                 "max_violation": round(max(pviol, default=0.0), 4)})
                continue
            poses = self._dense_poses(T_start, path)
            if stage.is_grasp:
                poses.append(T_goal)
            try:
                chain = self._ik_chain(poses, q_start)
            except _ChainFailure as e:
                failures.append({"attempt": attempt, "kind": "joint_path", "reason": str(e)})
                continue
            row = {"stage": label, "attempts": attempt + 1, "failures": failures,
                   "solve_seconds": round(time.time() - t0, 1),
                   "subgoal_violations": np.round(viol, 5).tolist(),
                   "path_max_violation": round(max(pviol, default=0.0), 5),
                   "n_waypoints": int(len(chain)),
                   "max_joint_step_rad": round(float(np.max(np.abs(np.diff(chain, axis=0)))), 3)}
            print(f"[{label}] attempt {attempt + 1}: {len(chain)} waypoints, "
                  f"violations {row['subgoal_violations']}, {row['solve_seconds']} s")
            return chain, self.kin.fk(chain[-1]), row
        self.kin.set_q(self.q_home)
        raise TaskPlanError(failures[-1]["kind"], label,
                            f"no solution in {c.attempts} attempts", {"failures": failures})

    def _colliding(self, chain: np.ndarray) -> int:
        kin = self.kin
        if self.held is not None:
            kin = AttachedArmKinematics(self.env, AttachedObject(self.T_ee_body),
                                        self.env.objects[self.held].joints[0], self.held)
        n = sum(kin.in_collision(q) for q in chain)
        self.kin.set_q(self.q_home)
        return int(n)

    def plan(self) -> Plan:
        program = self.program
        grasped = program.grasped_object
        q, T_site = self.q_home, self.kin.fk(self.q_home)
        segments, q_grasp = [], None
        for i, stage in enumerate(program.stages):
            label = "reach+approach" if stage.is_grasp else f"stage{i + 1}"
            chain, T_site, row = self._solve_stage(i, label, T_site, q)
            row["colliding_waypoints"] = self._colliding(chain)
            self.report["stages"].append(row)
            segments.append((label, chain))
            q = chain[-1]
            if stage.is_grasp:
                q_grasp = q
                self.held = grasped
                self.T_ee_body = np.linalg.inv(T_site) @ self.T0[grasped]
        self.report["status"] = "ok"
        return Plan(path=np.vstack([s for _, s in segments]),
                    stage_ids=np.concatenate([np.full(len(s), i)
                                              for i, (_, s) in enumerate(segments)]),
                    stage_labels=np.array([lab for lab, _ in segments]),
                    q_home=self.q_home, q_grasp=q_grasp, T_ee_body=self.T_ee_body,
                    grasped=grasped, T0=self.T0, target=self._target())

    def _target(self) -> dict | None:
        t = self.program.target
        if not t:
            return None
        symbols = self.symbols[t["object"]]
        return {"target_object": np.array(t["object"]),
                "target_point_body": symbols.points[t["point"]],
                "target_up_body": symbols.axes[t["up_axis"]]}


def default_program_dir(scene_name: str) -> Path:
    return Path(__file__).resolve().parent / "programs" / scene_name
