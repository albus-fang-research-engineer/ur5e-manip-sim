"""ReKep's pour-tea program as TSRs, so LangFold's planner (CBiRRT) can plan
under ReKep's constraints and only the solver differs from ReKep's plans. The
mirror of langfold.py.

Every constraint in programs/pour_tea is a function of the teapot's pose (its
keypoints 1-5 are fixed on the pot), and each is a TSR on the teapot body:

    stage 2 path     up axis (kp2 -> kp3) along +z        roll = pitch = 0; x, y, z, yaw free
    stage 2 subgoal  spout (kp1) = opening (kp6) + 10 cm z  spout pinned; rotation free
    stage 3 path     spout x, y = opening x, y             spout x = y = 0; z, rotation free
    stage 3 subgoal  spout = opening + 5 cm z, and the     spout pinned; pour axis in the
                     pour axis (kp2 -> kp5) within 10 deg  10 deg cone; yaw free
                     of straight down

Each TSR contains a pose exactly where ReKep's costs are 0 (tests/test_rekep.py
checks it on sampled poses). The tolerances are the program's: none, apart from
the cone. A cone is not a box, so ConeTSR adds it to the box that circumscribes
it. The planner's own tolerances (projection 2e-3, goal containment 5e-3) apply
here as they do to LangFold's TSRs.

LangFold's planner samples a stage's goals from its subgoal TSR and keeps the
samples its path TSR contains. ReKep's stage-2 subgoal leaves rotation free and
its path pins roll and pitch, so the stage-2 subgoal here carries the path's
roll = pitch = 0: the same goal set, sampled directly.

Frames are world-aligned at the opening, as ReKep's offsets are in world z.
Where a free axis needs a reference, it is the pot's pose: the stage-3 subgoal
at zero yaw is the entry pose turned the least to point its pour axis down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..frames import Symbols
from ..pour_stages import PourPair, TransportPair
from ..tsr import FREE_ROT, FREE_TRANS, TSR, bounds, make_pose

TRANSPORT_HEIGHT = 0.10    # stage2_subgoal_constraints.txt
POUR_HEIGHT = 0.05         # stage3_subgoal_constraints.txt
POUR_CONE_DEG = 10.0       # stage3_subgoal_constraints.txt


@dataclass
class ConeTSR(TSR):
    """A TSR whose poses also keep the body axis `axis` within `max_angle` of
    the world direction `toward`. Only a subgoal: excess() and project(), which
    the planner's projection uses on path TSRs, do not see the cone."""

    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    toward: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    max_angle: float = np.pi

    def cone_excess(self, T0_e: np.ndarray) -> float:
        a = T0_e[:3, :3] @ self.axis
        angle = np.arccos(np.clip(a @ self.toward / np.linalg.norm(a), -1.0, 1.0))
        return max(0.0, float(angle) - self.max_angle)

    def distance(self, T0_e: np.ndarray) -> float:
        return float(np.hypot(super().distance(T0_e), self.cone_excess(T0_e)))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        while True:                       # the cone fills pi/4 of its box
            T = super().sample(rng)
            if self.cone_excess(T) == 0.0:
                return T


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The least rotation taking direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if a @ b > 0:
            return np.eye(3)
        perp = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return R.from_rotvec(np.pi * perp / np.linalg.norm(perp)).as_matrix()
    return R.from_rotvec(axis / s * np.arctan2(s, a @ b)).as_matrix()


def _pinned_at(feature: np.ndarray, R0: np.ndarray = np.eye(3)) -> np.ndarray:
    """Tw_e for a world-aligned w at the target: the body point `feature` sits
    at w, and the body has rotation R0 at zero displacement. A displacement's
    rotation then turns the body about the feature, in world axes."""
    return make_pose(-R0 @ feature, R0)


class ReKepTSRs:
    """programs/pour_tea as stage pairs, with plan_pour_tea.py's TaskFrames
    interface: transport(T_teapot_now, T0_mug), pour(T_entry, T0_mug).
    transport_height replaces the program's 10 cm, for the variant that raises
    it until the upright pot clears the mug."""

    def __init__(self, teapot: Symbols, mug: Symbols, program_dir: str | Path,
                 transport_height: float | None = None):
        kps = json.loads((Path(program_dir) / "metadata.json").read_text())["keypoints"]

        def point(i):
            k = kps[i]
            return np.asarray(k["xyz"], float) if "xyz" in k else {"teapot": teapot, "mug": mug}[k["object"]].points[k["point"]]

        assert [k["object"] for k in kps] == ["teapot"] * 6 + ["mug"], "keypoints are not pour_tea's"
        self.spout = point(1)
        self.up = point(3) - point(2)
        self.pour_axis = point(5) - point(2)
        self.opening = point(6)          # mug body frame
        self.transport_height = TRANSPORT_HEIGHT if transport_height is None else transport_height
        # roll and pitch measure the up axis from world z only if it is body z
        assert np.allclose(self.up / np.linalg.norm(self.up), [0.0, 0.0, 1.0])

    def transport(self, T_teapot_now: np.ndarray, T0_mug: np.ndarray, **_) -> TransportPair:
        """Keyword bounds, which LangFold's TSRs take, are ignored: the program has none."""
        at = (T0_mug @ np.r_[self.opening, 1.0])[:3]
        subgoal = TSR(T0_w=make_pose(at + [0.0, 0.0, self.transport_height]), Tw_e=_pinned_at(self.spout),
                      Bw=bounds(yaw=FREE_ROT),
                      name=f"rekep/stage2 subgoal(spout {100 * self.transport_height:g} cm above opening, upright)")
        path = TSR(T0_w=make_pose(T_teapot_now[:3, 3]),
                   Bw=bounds(x=FREE_TRANS, y=FREE_TRANS, z=FREE_TRANS, yaw=FREE_ROT),
                   name="rekep/stage2 path(upright)")
        return TransportPair(subgoal=subgoal, path=path)

    def pour(self, T_entry: np.ndarray, T0_mug: np.ndarray, tilt_target: float | None = None) -> PourPair:
        """tilt_target is unused: ReKep's pour subgoal is the cone."""
        at = (T0_mug @ np.r_[self.opening, 1.0])[:3]
        down = np.array([0.0, 0.0, -1.0])
        R0 = _align(T_entry[:3, :3] @ self.pour_axis, down) @ T_entry[:3, :3]
        cone = np.deg2rad(POUR_CONE_DEG)
        subgoal = ConeTSR(T0_w=make_pose(at + [0.0, 0.0, POUR_HEIGHT]), Tw_e=_pinned_at(self.spout, R0),
                          Bw=bounds(roll=(-cone, cone), pitch=(-cone, cone), yaw=FREE_ROT),
                          name="rekep/stage3 subgoal(spout 5 cm above opening, pour axis in 10 deg cone)",
                          axis=self.pour_axis, toward=down, max_angle=cone)
        path = TSR(T0_w=make_pose(at), Tw_e=_pinned_at(self.spout),
                   Bw=bounds(z=FREE_TRANS, roll=FREE_ROT, pitch=FREE_ROT, yaw=FREE_ROT),
                   name="rekep/stage3 path(spout over opening)")
        return PourPair(subgoal=subgoal, path=path)
