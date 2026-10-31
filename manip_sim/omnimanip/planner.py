"""OmniManip's execution side, given privileged primitives instead of a VLM.

Only what the paper describes: Eq. 2's single target pose for the place stage
(target_pose.py), and the atomic actions of its supplement A.4, as straight
end-effector moves:

    grasp   "Move to 8cm in front of the grasp pose, move forward and close"
    place   "move to v cm in front of the target pose, move to the target pose
            and open the gripper"

There is deliberately no motion planner, lift, reachability term or retry: the
paper describes none, and adding them would measure something else. Unreachable
targets and broken IK lines are typed failures, and collisions are left for
eval_plan to report. A.5 lists both as failure causes of the original system.

Differences from the paper, each because of this comparison:

  grasp       a given grasp pose stands in for its grasp model, so every baseline
              compares motion from the same grip.
  primitives  come from the program, so RRC and its VLM calls do not run.
  loop        planned once. The paper re-solves Eq. 2 as a 6D tracker updates the
              object poses; undisturbed and kinematic, that has nothing to correct.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from ..frames import load_symbols
from ..planning import ArmKinematics
from ..rekep.adapter import (ROBOT_PREFIXES, ReKepIK, body_pose, collision_vertices,
                             farthest_point_sampling, geom_vertices, hull_planes)
from ..task_plan import TaskPlanError
from .target_pose import Constraint, Obstacle, Weights, solve_target_pose

EE_PREFIX = "gripper0"


@dataclass
class Config:
    weights: Weights = field(default_factory=Weights)
    max_ee_points: int = 60
    pos_tol: float = 0.005           # a target solve must reach these, or it failed
    rot_tol_deg: float = 5.0
    line_pos_step: float = 0.005     # an IK waypoint every 5 mm / 1 deg
    line_rot_step_deg: float = 1.0
    ik_pos_tol: float = 1e-3
    ik_rot_tol: float = 5e-3
    max_joint_step: float = 0.2      # rad between line waypoints: a flip, or fast joints near a wrist singularity
    x0_jitter: tuple[float, float] = (0.01, 0.05)  # m, rad; seed 0 starts exactly at the pose
    seed: int = 0


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


def line_poses(T_a: np.ndarray, T_b: np.ndarray, pos_step: float, rot_step: float) -> list[np.ndarray]:
    """Straight translation plus slerp, both ends included."""
    rotations = R.from_matrix([T_a[:3, :3], T_b[:3, :3]])
    angle = (rotations[0].inv() * rotations[1]).magnitude()
    n = max(1, int(np.ceil(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]) / pos_step)),
            int(np.ceil(angle / rot_step)))
    slerp, out = Slerp([0, 1], rotations), []
    for s in np.linspace(0.0, 1.0, n + 1):
        T = np.eye(4)
        T[:3, :3] = slerp(s).as_matrix()
        T[:3, 3] = (1 - s) * T_a[:3, 3] + s * T_b[:3, 3]
        out.append(T)
    return out


def grasp_pose(T_object: np.ndarray, point_body, closing_body, approach_world,
               T_reference: np.ndarray) -> np.ndarray:
    """Site pose: +z along the approach, +x along the closing axis. A parallel jaw
    closes the same either way round, so take the sign nearer T_reference."""
    z = np.asarray(approach_world, float)
    z = z / np.linalg.norm(z)
    x = T_object[:3, :3] @ np.asarray(closing_body, float)
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    if x @ T_reference[:3, 0] < 0:
        x = -x
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = T_object[:3, :3] @ np.asarray(point_body, float) + T_object[:3, 3]
    return T


def load_program(path: str | Path) -> dict:
    program = json.loads(Path(path).read_text())
    stages = program["stages"]
    if len(stages) != 1 or stages[0]["action"] != "place":
        raise ValueError(f"{path}: only grasp then one place is implemented")
    return program


def default_program(scene_name: str) -> Path:
    return Path(__file__).resolve().parent / "programs" / f"{scene_name}.json"


class OmniManipPlanner:
    def __init__(self, env, scene, program_path, cfg: Config | None = None):
        self.env, self.scene, self.cfg = env, scene, cfg or Config()
        self.program = load_program(program_path)
        self.kin = ArmKinematics(env)
        self.q_home = self.kin.data.qpos[self.kin.qpos_ids].copy()
        self.T_home = self.kin.fk(self.q_home)
        self.T0 = {n: body_pose(self.kin, n) for n in scene.objects}
        self.symbols = {n: load_symbols(scene.asset_dirs[n]) for n in scene.objects}
        self.ik = ReKepIK(self.kin, self.q_home)
        self.report: dict = {"planner": "omnimanip", "mode": "open-loop",
                             "program": str(program_path), "config": asdict(self.cfg),
                             "stages": []}

    def _ee_points(self) -> np.ndarray:
        """End-effector collision vertices in its own frame, gripper open."""
        self.kin.set_q(self.q_home)
        world = collision_vertices(self.kin, (EE_PREFIX,))
        local = (world - self.T_home[:3, 3]) @ self.T_home[:3, :3]
        return farthest_point_sampling(local, self.cfg.max_ee_points)

    def _obstacles(self, active: str) -> list[Obstacle]:
        """Every object but the robot and the held one, at its initial pose."""
        m, d = self.kin.model, self.kin.data
        skip, found = tuple(ROBOT_PREFIXES) + (active,), {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if not (m.geom_contype[g] or m.geom_conaffinity[g]) or name.startswith(skip) \
                    or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            found.setdefault(name, Obstacle(name)).hulls.append(hull_planes(geom_vertices(m, d, g)))
        return list(found.values())

    def _constraint(self, stage: dict) -> Constraint:
        a, p = stage["active"], stage["passive"]
        sa, sp = self.symbols[a["object"]], self.symbols[p["object"]]
        T_p = self.T0[p["object"]]
        axis_a = sa.axes[a["axis"]] / np.linalg.norm(sa.axes[a["axis"]])
        axis_p = T_p[:3, :3] @ sp.axes[p["axis"]]
        return Constraint(sa.points[a["point"]], axis_a, T_p[:3, :3] @ sp.points[p["point"]] + T_p[:3, 3],
                          axis_p / np.linalg.norm(axis_p), float(stage["d_m"]))

    def _line(self, label: str, T_a, T_b, q_a) -> np.ndarray:
        """IK along the straight move. It takes the first solution from the last
        waypoint and fails on a jump, as a Cartesian move would."""
        c = self.cfg
        poses = line_poses(T_a, T_b, c.line_pos_step, np.radians(c.line_rot_step_deg))
        chain = [q_a]
        for i, T in enumerate(poses[1:], 1):
            r = self.ik.solve_site(T, c.ik_pos_tol, c.ik_rot_tol, 200, chain[-1])
            step = float(np.max(np.abs(r.cspace_position - chain[-1])))
            if not r.success or step > c.max_joint_step:
                raise TaskPlanError("ik_line", label, f"at waypoint {i}/{len(poses) - 1}: " + (
                    f"no IK ({r.position_error * 1000:.1f} mm)" if not r.success
                    else f"joint step {step:.2f} rad"), {"T": np.round(T, 4).tolist()})
            chain.append(r.cspace_position)
        return np.array(chain)

    def plan(self) -> Plan:
        c, program = self.cfg, self.program
        g = program["grasp"]
        grasped = g["object"]

        T_grasp = grasp_pose(self.T0[grasped], g["point_body"], g["closing_body"],
                             g["approach_world"], self.T_home)
        T_pre = T_grasp.copy()
        T_pre[:3, 3] -= T_grasp[:3, 2] * g["standoff_m"]
        to_pre = self._line("grasp", self.T_home, T_pre, self.q_home)
        forward = self._line("grasp", T_pre, T_grasp, to_pre[-1])
        reach = np.vstack([to_pre, forward[1:]])
        q_grasp = reach[-1]
        T_grasp = self.kin.fk(q_grasp)
        T_ee_body = np.linalg.inv(T_grasp) @ self.T0[grasped]
        self.report["stages"].append({"stage": "grasp", "n_waypoints": len(reach)})

        stage = program["stages"][0]
        constraint = self._constraint(stage)
        rng = np.random.default_rng(c.seed)
        x0 = None if c.seed == 0 else np.r_[rng.normal(0.0, c.x0_jitter[0], 3),
                                            rng.normal(0.0, c.x0_jitter[1], 3)]
        ee_points, obstacles = self._ee_points(), self._obstacles(grasped)
        T_target, solve = solve_target_pose(T_grasp, T_ee_body, constraint, ee_points, obstacles,
                                            c.weights, x0)
        row = {"stage": "place", "solve": solve, "obstacles": [o.name for o in obstacles]}
        self.report["stages"].append(row)
        print(f"[place] target: {solve['position_error_mm']} mm, {solve['angle_error_deg']} deg, "
              f"losses {solve['losses']}")
        if solve["position_error_mm"] > c.pos_tol * 1000 or solve["angle_error_deg"] > c.rot_tol_deg:
            raise TaskPlanError("target_pose", "place", "the solve missed the constraint", solve)

        T_above = T_target.copy()
        T_above[:3, 3] += constraint.axis_passive * stage["v_m"]
        move = self._line("place:move", T_grasp, T_above, q_grasp)
        insert = self._line("place:insert", T_above, T_target, move[-1])
        row["n_waypoints"] = len(move) + len(insert)

        segments = [("grasp", reach), ("place:move", move), ("place:insert", insert)]
        self.kin.set_q(self.q_home)
        self.report["status"] = "ok"
        target = program.get("target")
        return Plan(path=np.vstack([s for _, s in segments]),
                    stage_ids=np.concatenate([np.full(len(s), i) for i, (_, s) in enumerate(segments)]),
                    stage_labels=np.array([lab for lab, _ in segments]),
                    q_home=self.q_home, q_grasp=q_grasp, T_ee_body=T_ee_body, grasped=grasped, T0=self.T0,
                    target=None if not target else {
                        "target_object": np.array(target["object"]),
                        "target_point_body": self.symbols[target["object"]].points[target["point"]],
                        "target_up_body": self.symbols[target["object"]].axes[target["up_axis"]]})
