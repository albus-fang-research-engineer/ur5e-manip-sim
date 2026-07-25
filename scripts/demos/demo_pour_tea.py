"""Live interactive viewer for the POUR-TEA scene (fixed poses).

Same viewer plumbing as view_live.py, but the deterministic pour stage:
1.2 m table, teapot and mug pinned 0.5 m apart, teapot spout (body +x,
minus the yaw offset) yawed to face the mug.

  left-drag   orbit camera        right-drag   pan
  scroll      zoom                double-click select a body
  Space       pause/resume        Tab          camera menu
  Esc / close window              quit

Run INSIDE the container:

  PYTHONPATH=. python scripts/view_pour_tea.py
  PYTHONPATH=. python scripts/view_pour_tea.py --yaw-offset-deg 45
  PYTHONPATH=. python scripts/view_pour_tea.py --jiggle

Calibrating the spout offset WITHOUT editing files:
  run with --yaw-offset-deg 0, then 90 (a quarter turn -- unmistakable),
  then bisect. The console prints BOTH the commanded yaw and the yaw
  measured back from the settled physics state; if you change the flag and
  the MEASURED number doesn't move, you are not running the file/process
  you think you are. Once found, bake the value into SPOUT_YAW_OFFSET here
  and in demo_pour_tea.py (stored in radians).

One-time on the HOST (not the container):  xhost +local:docker
"""

import argparse
import os
from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env
from scipy.spatial.transform import Rotation as R

import manip_sim  # noqa: F401
from manip_sim.envs.tabletop import TableTop

# ----------------------------------------------------------------- scene spec
TABLE_SIZE = (1.2, 1.2, 0.05)
TABLE_TOP_Z = 0.8
TEAPOT_XY = np.array([0.0, -0.25])
MUG_XY = np.array([0.0, 0.25])           # 0.5 m from the teapot
SPOUT_YAW_OFFSET = 0.3                    # rad; default for --yaw-offset-deg
DROP_HEIGHT = 0.06
SETTLE_STEPS = 20

OBJECTS = {
    "teapot": "assets/objects/teapot/teapot.xml",
    "mug": "assets/objects/mug/mug.xml",
}


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class PourTeaSceneLive(TableTop):
    """TableTop with deterministic object placement (viewer flavor)."""

    def __init__(self, robots, fixed_poses, **kwargs):
        self.fixed_poses = fixed_poses  # name -> (pos[3], quat_wxyz[4])
        super().__init__(robots=robots, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        for name, (pos, quat) in self.fixed_poses.items():
            if name in self.objects:
                self.sim.data.set_joint_qpos(
                    self.objects[name].joints[0], np.concatenate([pos, quat])
                )
        self.sim.forward()


register_env(PourTeaSceneLive)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=os.environ.get("ROBOT", "UR5e"))
    ap.add_argument("--yaw-offset-deg", type=float,
                    default=float(np.rad2deg(SPOUT_YAW_OFFSET)),
                    help="spout yaw offset in DEGREES (overrides the file constant)")
    ap.add_argument("--jiggle", action="store_true",
                    help="apply small sinusoidal eef motion instead of holding still")
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()

    if not os.environ.get("DISPLAY"):
        raise SystemExit("[view_pour_tea] $DISPLAY is empty -- X11 is not reaching "
                         "the container. Check compose env/volumes and "
                         "`xhost +local:docker`.")

    print(f"[view_pour_tea] running: {Path(__file__).resolve()}")

    objs = {}
    for name, path in OBJECTS.items():
        if Path(path).exists():
            objs[name] = path
        else:
            print(f"[view_pour_tea] skipping '{name}' (not converted yet: {path})")

    offset = np.deg2rad(args.yaw_offset_deg)
    bearing = MUG_XY - TEAPOT_XY
    teapot_yaw = float(np.arctan2(bearing[1], bearing[0])) - offset
    z0 = TABLE_TOP_Z + DROP_HEIGHT
    fixed_poses = {
        "teapot": (np.array([*TEAPOT_XY, z0]), yaw_quat_wxyz(teapot_yaw)),
        "mug": (np.array([*MUG_XY, z0]), yaw_quat_wxyz(0.0)),
    }

    env = suite.make(
        "PourTeaSceneLive",
        robots=args.robot,
        object_xmls=objs,
        fixed_poses=fixed_poses,
        table_full_size=TABLE_SIZE,
        table_offset=(0.0, 0.0, TABLE_TOP_Z),
        has_renderer=True,
        render_camera=None,          # start in free camera -> drag anywhere
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        ignore_done=True,
    )
    env.reset()

    # settle, then measure the yaw actually realized in physics
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(env.action_dim))
        env.render()
    if "teapot" in objs:
        q = env.sim.data.body_xquat[env.obj_body_ids["teapot"]]
        measured = R.from_quat(q, scalar_first=True).as_euler("zyx")[0]
        print(f"[view_pour_tea] offset {args.yaw_offset_deg:+.1f} deg -> "
              f"commanded yaw {np.rad2deg(teapot_yaw):+.1f} deg, "
              f"MEASURED settled yaw {np.rad2deg(measured):+.1f} deg")

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