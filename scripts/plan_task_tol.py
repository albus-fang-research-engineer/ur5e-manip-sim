"""Run plan_task.py with the "tight" rotational band overridden, for tolerance
comparisons without editing the frozen table in compile_tsr.py.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/plan_task_tol.py 1 <plan_task args...>
"""

import runpy
import sys

import numpy as np

from manip_sim.compile_tsr import ROT_TOL_RAD

deg = float(sys.argv[1])
ROT_TOL_RAD["tight"] = np.deg2rad(deg)
print(f"[tol] tight = {deg} deg", flush=True)
sys.argv = ["scripts/plan_task.py"] + sys.argv[2:]
runpy.run_path("scripts/plan_task.py", run_name="__main__")
