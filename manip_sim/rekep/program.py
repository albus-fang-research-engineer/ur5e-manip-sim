"""A ReKep program: upstream's on-disk layout, with privileged keypoints.

    programs/<task>/metadata.json
    programs/<task>/stage<N>_subgoal_constraints.txt
    programs/<task>/stage<N>_path_constraints.txt       (optional)

Constraint files are upstream's: functions of (end_effector, keypoints) that
return a cost, satisfied at <= 0 up to the tolerance. Upstream's metadata gives
keypoints as world positions from a camera; here each is a named point from an
object's frames.json, or literal body-frame coordinates:

    "keypoints": [{"object": "marker", "point": "tip"},
                  {"object": "cup", "xyz": [0, 0, 0.03]}]

The planner appends one virtual keypoint, keypoints[-1], rigid with the gripper
and offset along its closing axis, so a constraint can see gripper orientation
(upstream's constraints only see positions). Scene keypoint indices are the
same as upstream's, so a generated program would load unchanged.

Constraint files are executed as Python: load only programs you trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .vendor import load_functions_from_txt


@dataclass
class Keypoint:
    object: str
    point: str | None = None
    xyz: np.ndarray | None = None


@dataclass
class Stage:
    subgoal: list[Callable] = field(default_factory=list)
    path: list[Callable] = field(default_factory=list)
    grasp_keypoint: int = -1
    release_keypoint: int = -1

    @property
    def is_grasp(self) -> bool:
        return self.grasp_keypoint != -1


@dataclass
class Program:
    name: str
    keypoints: list[Keypoint]
    stages: list[Stage]
    target: dict | None = None
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None

    @property
    def grasped_object(self) -> str:
        return self.keypoints[self.stages[0].grasp_keypoint].object


def load_program(directory: str | Path, grasping_cost: Callable[[int], float]) -> Program:
    """grasping_cost(keypoint_idx) backs get_grasping_cost_by_keypoint_idx."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    keypoints = [Keypoint(k["object"], k.get("point"),
                          np.asarray(k["xyz"], float) if "xyz" in k else None)
                 for k in meta["keypoints"]]
    for i, k in enumerate(keypoints):
        if (k.point is None) == (k.xyz is None):
            raise ValueError(f"{directory}: keypoint {i} needs exactly one of point, xyz")
    stages = []
    for n in range(1, meta["num_stages"] + 1):
        fns = {}
        for kind in ("subgoal", "path"):
            f = directory / f"stage{n}_{kind}_constraints.txt"
            fns[kind] = load_functions_from_txt(str(f), grasping_cost) if f.exists() else []
        stages.append(Stage(fns["subgoal"], fns["path"],
                            meta["grasp_keypoints"][n - 1], meta["release_keypoints"][n - 1]))
    # execute_task_plan's contract: one grasp on the first stage, release at the end
    if not stages[0].is_grasp or any(s.is_grasp for s in stages[1:]):
        raise ValueError(f"{directory}: the first stage, and only it, must be a grasp")
    if any(s.release_keypoint != -1 for s in stages[:-1]):
        raise ValueError(f"{directory}: only the last stage may release")
    return Program(directory.name, keypoints, stages, meta.get("target"),
                   *(np.asarray(meta[b], float) if b in meta else None
                     for b in ("bounds_min", "bounds_max")))
