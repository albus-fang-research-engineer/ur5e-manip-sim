"""Run plan_task.py with the "tight" rotational band overridden, for tolerance
comparisons without editing the frozen table in compile_tsr.py. --eps also
overrides how far apart plan_constrained takes its checked steps (rad), and
--moderate the "moderate" band (deg).

    MUJOCO_GL=egl PYTHONPATH=. python scripts/plan_task_tol.py 1 <plan_task args...>
    MUJOCO_GL=egl PYTHONPATH=. python scripts/plan_task_tol.py 5 --eps 0.03 --moderate 7.5 <plan_task args...>
"""

import functools
import runpy
import sys

import numpy as np

from manip_sim import planning
from manip_sim.compile_tsr import ROT_TOL_RAD

deg = float(sys.argv[1])
ROT_TOL_RAD["tight"] = np.deg2rad(deg)
print(f"[tol] tight = {deg} deg", flush=True)
rest = sys.argv[2:]
while rest[:1] in (["--eps"], ["--moderate"]):
    if rest[0] == "--eps":
        planning.plan_constrained = functools.partial(planning.plan_constrained, eps=float(rest[1]))
    else:
        ROT_TOL_RAD["moderate"] = np.deg2rad(float(rest[1]))
    print(f"[tol] {rest[0][2:]} = {rest[1]}", flush=True)
    rest = rest[2:]
sys.argv = ["scripts/plan_task.py"] + rest
runpy.run_path("scripts/plan_task.py", run_name="__main__")
