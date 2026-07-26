"""Verify task-space control paths for the chosen robot.

Robot support matrix discovered on robosuite 1.5.2:

  OSC_POSE   -> works for every stock robot (default controller). This is
                the day-to-day control path.
  IK_POSE    -> robosuite's built-in InverseKinematicsController hard-asserts
                a whitelist: {Panda, Sawyer, Baxter, GR1FixedLowerBody}.
                UR5e is NOT on it.
  mink       -> differential IK as a QP over the raw MjModel; robot-agnostic,
                works on UR5e (verified to ~1e-16 error below). This is the
                path for TSR-projection / constrained-IK experiments anyway,
                so losing built-in IK_POSE on UR5e costs nothing.

Run:  python scripts/ik_test.py              # UR5e (default)
      ROBOT=Panda python scripts/ik_test.py  # also exercises IK_POSE
"""

import os

import mujoco
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

IK_POSE_SUPPORTED = {"Panda", "Sawyer", "Baxter", "GR1FixedLowerBody"}


def run_env(robot: str, part_cfg: dict | None, label: str) -> np.ndarray:
    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    if part_cfg is not None:
        cfg["body_parts"]["right"] = part_cfg
    env = suite.make(
        env_name="Lift",
        robots=robot,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    obs = env.reset()
    p0 = obs["robot0_eef_pos"].copy()
    for _ in range(40):
        act = np.zeros(env.action_dim)
        act[0] = 0.15
        obs, *_ = env.step(act)
    delta = obs["robot0_eef_pos"] - p0
    env.close()
    print(f"{label:8s} eef delta over 40 steps: {np.round(delta, 3)}")
    return delta


def run_mink(robot: str) -> None:
    """Standalone differential IK on the robot's raw MjModel via mink."""
    import mink

    env = suite.make(
        env_name="Lift",
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    model = env.sim.model._model
    data = env.sim.data._data

    sites = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    grip = next(s for s in sites if s and "grip_site" in s)

    conf = mink.Configuration(model)
    conf.update(data.qpos.copy())
    task = mink.FrameTask(
        frame_name=grip, frame_type="site", position_cost=1.0, orientation_cost=1.0
    )
    T0 = conf.get_transform_frame_to_world(grip, "site")
    task.set_target(mink.SE3.from_translation(np.array([0.05, 0.0, 0.05])) @ T0)

    for _ in range(100):
        v = mink.solve_ik(conf, [task], dt=0.02, solver="quadprog")
        conf.integrate_inplace(v, 0.02)

    err = float(np.linalg.norm(task.compute_error(conf)))
    print(f"mink     IK on {robot} ({grip}): residual {err:.2e}")
    env.close()
    assert err < 1e-6, "mink IK failed to converge"


def main() -> None:
    robot = os.environ.get("ROBOT", "UR5e")
    print(f"robot = {robot}\n")

    # OSC_POSE — default, works for all robots.
    d_osc = run_env(robot, None, "OSC")
    assert np.linalg.norm(d_osc) > 0.02, "OSC produced no motion"

    # IK_POSE — whitelist-only. Gotcha: replace the whole part dict; leftover
    # OSC keys (input_max, ...) collide with the IK controller's kwargs.
    if robot in IK_POSE_SUPPORTED:
        ik_cfg = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
        d_ik = run_env(robot, ik_cfg, "IK_POSE")
        assert np.linalg.norm(d_ik) > 0.02, "IK produced no motion"
    else:
        print(f"IK_POSE  skipped: robosuite 1.5.2 supports only {sorted(IK_POSE_SUPPORTED)}")

    # mink — robot-agnostic differential IK, the TSR-projection workhorse.
    run_mink(robot)

    print("\nIK TEST PASSED")


if __name__ == "__main__":
    main()
