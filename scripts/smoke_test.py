"""End-to-end smoke test for the manip-sim environment.

Validates, in order:
  1. mujoco imports and reports the pinned version
  2. robosuite builds a Panda Lift env
  3. offscreen rendering works under the active MUJOCO_GL backend
  4. ground-truth state access (the sim-state path the architecture reads
     on the critical path instead of running pose estimators in-loop)
  5. a short random rollout steps without error

Run:  python scripts/smoke_test.py            # UR5e (default)
      ROBOT=Panda python scripts/smoke_test.py
"""

import os
import sys

import numpy as np


def main() -> int:
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    robot = os.environ.get("ROBOT", "UR5e")
    print(f"MUJOCO_GL = {backend}")
    print(f"robot     = {robot}")

    import mujoco

    print(f"mujoco    = {mujoco.__version__}")
    assert mujoco.__version__.startswith("3.3"), (
        "Expected mujoco 3.3.x — newer minors (>=3.10) break robosuite 1.5.2 "
        "(mj_fullM signature change)."
    )

    import robosuite as suite

    print(f"robosuite = {suite.__version__}")

    env = suite.make(
        env_name="Lift",
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,  # RGB-D, matching the eventual perception interface
        control_freq=20,
    )
    obs = env.reset()

    # --- rendering ---
    rgb = obs["agentview_image"]
    depth = obs["agentview_depth"]
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert depth.shape[:2] == (256, 256)
    assert (rgb > 0).mean() > 0.5, "Rendered frame is mostly black — GL backend broken."
    print(f"render    = rgb {rgb.shape} ok, depth {depth.shape} ok")

    # --- ground-truth state (critical-path pose source in sim) ---
    cube_body = env.sim.model.body_name2id("cube_main")
    cube_pos = env.sim.data.body_xpos[cube_body].copy()
    cube_quat = env.sim.data.body_xquat[cube_body].copy()
    print(f"gt pose   = cube pos {np.round(cube_pos, 3)}, quat {np.round(cube_quat, 3)}")

    # --- rollout ---
    for _ in range(20):
        a = np.random.uniform(-0.1, 0.1, env.action_dim)
        obs, reward, done, info = env.step(a)
    print(f"rollout   = 20 steps ok, eef at {np.round(obs['robot0_eef_pos'], 3)}")

    env.close()
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
