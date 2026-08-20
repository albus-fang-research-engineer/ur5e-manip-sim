"""Live interactive viewer for the POUR-TEA scene (fixed poses).

Same viewer plumbing as view_live.py, but the deterministic pour stage:
1.2 m table, teapot and mug pinned 0.5 m apart, teapot spout (body +x,
minus SPOUT_YAW_OFFSET) yawed to face the mug.

  left-drag   orbit camera        right-drag   pan
  scroll      zoom                double-click select a body
  Space       pause/resume        Tab          camera menu
  Esc / close window              quit

Run INSIDE the container:

  PYTHONPATH=. python scripts/view_pour_tea.py
  PYTHONPATH=. python scripts/view_pour_tea.py --jiggle

One-time on the HOST (not the container):  xhost +local:docker
"""

import argparse
import os
from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation as R

import manip_sim  # noqa: F401  registers TableTop

# ----------------------------------------------------------------- scene spec
# The scene now lives in scenes/pour_tea.json (manip_sim.scene). These
# module constants are kept as a compatibility shim for older scripts that
# import them; new code takes --scene and reads the Scene object.
from manip_sim.scene import (DEFAULT_SCENE, load_scene,  # noqa: E402
                             make_env as _make_env, yaw_quat_wxyz)  # noqa: F401  (re-export)

SCENE = load_scene(DEFAULT_SCENE)
TABLE_SIZE = SCENE.table_size
TABLE_TOP_Z = SCENE.table_top_z
TEAPOT_XY = SCENE.xy("teapot")
MUG_XY = SCENE.xy("mug")
DROP_HEIGHT = SCENE.drop_height
SETTLE_STEPS = SCENE.settle_steps
OBJECTS = SCENE.object_xmls


def make_env(robot: str = "UR5e", has_renderer: bool = True,
             settle: bool = True, scene=SCENE, **make_kwargs):
    """Compat wrapper: build `scene` (default scenes/pour_tea.json).
    Returns (env, objs)."""
    return _make_env(scene, robot=robot, has_renderer=has_renderer,
                     settle=settle, **make_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=os.environ.get("ROBOT", "UR5e"))
    ap.add_argument("--jiggle", action="store_true",
                    help="apply small sinusoidal eef motion instead of holding still")
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()

    if not os.environ.get("DISPLAY"):
        raise SystemExit("[view_pour_tea] $DISPLAY is empty -- X11 is not reaching "
                         "the container. Check compose env/volumes and "
                         "`xhost +local:docker`.")

    print(f"[view_pour_tea] running: {Path(__file__).resolve()}")

    env, objs = make_env(robot=args.robot, has_renderer=True)
    teapot_yaw = SCENE.yaw("teapot")

    # factory already settled physics; report the realized yaw
    if "teapot" in objs:
        q = env.sim.data.body_xquat[env.obj_body_ids["teapot"]]
        measured = R.from_quat(q, scalar_first=True).as_euler("zyx")[0]
        print(f"[view_pour_tea] commanded yaw {np.rad2deg(teapot_yaw):+.1f} deg, "
              f"measured settled yaw {np.rad2deg(measured):+.1f} deg")

    sep = np.linalg.norm(MUG_XY - TEAPOT_XY)
    print(f"[view_pour_tea] teapot->mug {sep:.2f} m. Space pauses, Esc quits.")

    t = 0.0
    try:
        for _ in range(args.steps):
            a = np.zeros(env.action_dim)
            if args.jiggle:
                a[0] = 0.06 * np.sin(2 * np.pi * 0.25 * t)
                a[1] = 0.06 * np.cos(2 * np.pi * 0.25 * t)
                t += 1.0 / 20.0
            env.step(a)
            env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()