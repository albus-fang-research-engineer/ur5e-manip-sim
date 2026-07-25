"""Live interactive viewer for the TableTop scene.

Opens MuJoCo's native window (via robosuite's mjviewer renderer):
  left-drag   orbit camera        right-drag   pan
  scroll      zoom                double-click select a body
  Space       pause/resume        Tab          camera menu
  Esc / close window              quit

Run INSIDE the container:

  PYTHONPATH=. python scripts/view_live.py
  PYTHONPATH=. python scripts/view_live.py --jiggle   # small arm motion

One-time on the HOST (not the container):  xhost +local:docker

Notes:
- MUJOCO_GL=glfw overrides the entrypoint's egl/osmesa choice; the window
  path and the headless camera-obs path use different GL plumbing, and
  mixing them is the flakiest corner of the stack — so this script keeps
  use_camera_obs=False and exists purely for eyeballing scenes.
- Objects listed in OBJECTS that haven't been converted yet are skipped
  with a warning, so the viewer always starts.
"""

import argparse
import os

# NB: no need to set MUJOCO_GL here — robosuite forcibly rewrites it to
# "egl" on Linux at import (binding_utils.py), and that only affects
# OFFSCREEN contexts anyway. The interactive window always uses GLFW.

from pathlib import Path

import numpy as np
import robosuite as suite

import manip_sim  # noqa: F401  registers TableTop

OBJECTS = {
    "mug": "assets/objects/mug/mug.xml",
    "teapot": "assets/objects/teapot/teapot.xml",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=os.environ.get("ROBOT", "UR5e"))
    ap.add_argument("--jiggle", action="store_true",
                    help="apply small sinusoidal eef motion instead of holding still")
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()

    if not os.environ.get("DISPLAY"):
        raise SystemExit("[view_live] $DISPLAY is empty — X11 is not reaching the "
                         "container. Check compose env/volumes and `xhost +local:docker`.")

    objs = {}
    for name, path in OBJECTS.items():
        if Path(path).exists():
            objs[name] = path
        else:
            print(f"[view_live] skipping '{name}' (not converted yet: {path})")

    env = suite.make(
        "TableTop",
        robots=args.robot,
        object_xmls=objs,
        placement_x_range=(-0.25, 0.25),
        placement_y_range=(-0.25, 0.25),
        has_renderer=True,
        render_camera=None,          # start in free camera -> drag anywhere
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        ignore_done=True
    )
    env.reset()
    print("[view_live] window up — drag to orbit, Space to pause, Esc to quit.")

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