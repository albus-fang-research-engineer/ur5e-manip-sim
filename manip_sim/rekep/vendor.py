"""ReKep's solvers, loaded unchanged from the upstream submodule.

upstream/ is https://github.com/huangwl18/ReKep at a pinned commit. It is a
submodule rather than a copy because that repository has no license, so its
code is not ours to redistribute. Fetch it with:

    git submodule update --init manip_sim/rekep/upstream

Keep it as upstream wrote it: anything specific to this sim goes in adapter.py
or a program, so the solvers stay comparable to the paper's. Upstream's modules
import each other at the top level and import open3d, which is not in the sim
image and which only farthest_point_sampling uses, so this module provides both.
Nothing else in the repo imports this, and the rest works without the submodule.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from .adapter import farthest_point_sampling

UPSTREAM = Path(__file__).resolve().parent / "upstream"
if not (UPSTREAM / "subgoal_solver.py").exists():
    raise ImportError(f"ReKep's solvers are not at {UPSTREAM}: "
                      "run git submodule update --init manip_sim/rekep/upstream")

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
sys.path.insert(0, str(UPSTREAM))
try:
    import path_solver
    import subgoal_solver
    import utils
finally:
    sys.path.remove(str(UPSTREAM))

# both solvers bound the name at import
path_solver.farthest_point_sampling = farthest_point_sampling
subgoal_solver.farthest_point_sampling = farthest_point_sampling

get_linear_interpolation_steps = utils.get_linear_interpolation_steps
load_functions_from_txt = utils.load_functions_from_txt
spline_interpolate_poses = utils.spline_interpolate_poses


def scale_collision_cost(scale: float) -> None:
    """Multiply the collision term of both solvers. Upstream's weights are literals
    inside its objectives, and both solvers bound this name at import."""
    def scaled(*args):
        return scale * utils.calculate_collision_cost(*args)
    path_solver.calculate_collision_cost = scaled
    subgoal_solver.calculate_collision_cost = scaled


class SubgoalSolver(subgoal_solver.SubgoalSolver):
    def _warmup(self):
        """Upstream's warmup is a full solve to compile for real-time use;
        planning once does not need it."""


class PathSolver(path_solver.PathSolver):
    def _warmup(self):
        """See SubgoalSolver._warmup."""
