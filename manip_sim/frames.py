"""Grounding symbol tables and task-frame composition.

Each converted asset directory carries a frames.json declaring the object's
grounded interaction symbols in *body* coordinates (the frame of the mjcf
body named "object" -- i.e. exactly what state.PoseReader returns; the body
frame is arbitrary and never assumed canonical, only CONSISTENT with the
pose source):

    points      named interaction points  (spout_tip, opening_center, ...)
    axes        named semantic axes       (pour_axis, up_axis, tilt_axis, ...)
    quantities  compiler-owned scalars    (rim_radius, cavity_depth, ...)

Three axis names are reserved for the object's CANONICAL frame (vlm.
FRAME_AXES): up_axis, front_axis, lateral_axis = up x front. They are
caller-supplied (asset rest frame / scene facing in sim, Orient Anything
on hardware), not fitted, and the grounding compiler roots every stage's
w frame on them; part fits keep their own <part>_axis names. An axis
entry may carry "sigma_deg", the estimator's 1-sigma direction
uncertainty, which the compiler couples into the B^w rotation rows.

Points and axes are INDEPENDENT tables, mirroring the grounding pipeline
(point candidates -> VLM multiple choice; axes -> PointSO) and the emission
DSL, whose frame declarations compose them by reference:

    active teapot frame(origin=spout_tip, axis=pour_axis)
        -> symbols.frame("spout_tip", "pour_axis")

The hand-authored sidecars in assets/ are the ground-truth arm of the
emission ablation; the VLM grounding pipeline later becomes a *producer of
this same artifact* and nothing downstream changes.

Frame convention: axis -> +z; +x by Gram-Schmidt of a secondary direction
(default body -z) against the axis; +y = z cross x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tsr import make_pose


@dataclass
class Frame:
    name: str
    point: np.ndarray            # body coords, meters
    axis: np.ndarray             # body coords, unit
    secondary: np.ndarray        # body coords, unit
    status: str = "calibrated"
    comment: str = ""

    def T(self) -> np.ndarray:
        """4x4 body->frame transform: origin at point, +z = axis,
        +x = Gram-Schmidt(secondary against axis), +y = z cross x."""
        z = self.axis / np.linalg.norm(self.axis)
        s = self.secondary / np.linalg.norm(self.secondary)
        x = s - np.dot(s, z) * z
        n = np.linalg.norm(x)
        if n < 1e-6:  # secondary parallel to axis: pick any perpendicular
            s = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(s, z)) > 0.9:
                s = np.array([0.0, 1.0, 0.0])
            x = s - np.dot(s, z) * z
            n = np.linalg.norm(x)
        x = x / n
        y = np.cross(z, x)
        return make_pose(self.point, np.column_stack([x, y, z]))

    def world_T(self, T0_body: np.ndarray) -> np.ndarray:
        """World pose of this frame given the object's body world pose."""
        return T0_body @ self.T()


@dataclass
class Symbols:
    """One object's grounded symbol table."""

    object: str
    points: dict[str, np.ndarray]
    axes: dict[str, np.ndarray]
    quantities: dict[str, float] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    sigmas: dict[str, float] = field(default_factory=dict)   # "axes.<name>" -> deg

    def frame(self, origin: str, axis: str,
              secondary=None) -> Frame:
        """DSL-style composition: frame(origin=<point>, axis=<axis>).
        `secondary` may name another axis symbol, be a raw vector, or be
        None (body -z)."""
        if isinstance(secondary, str):
            sec = self.axes[secondary]
        elif secondary is None:
            sec = np.array([0.0, 0.0, -1.0])
        else:
            sec = np.asarray(secondary, dtype=float)
        status = "placeholder" if (
            self.statuses.get(f"points.{origin}") == "placeholder"
            or self.statuses.get(f"axes.{axis}") == "placeholder"
        ) else "calibrated"
        return Frame(
            name=f"{self.object}.frame({origin},{axis})",
            point=self.points[origin].copy(),
            axis=self.axes[axis].copy(),
            secondary=sec.copy(),
            status=status,
        )


def load_symbols(asset_dir):
    path = Path(asset_dir) / "frames.json"
    spec = json.loads(path.read_text())
    statuses, sigmas = {}, {}

    def _table(section):
        out = {}
        for name, entry in spec.get(section, {}).items():
            out[name] = np.asarray(entry["xyz"], dtype=float).reshape(3)
            statuses[f"{section}.{name}"] = entry.get("status", "calibrated")
            if entry.get("sigma_deg") is not None:
                sigmas[f"{section}.{name}"] = float(entry["sigma_deg"])
        return out

    sym = Symbols(
        object=spec.get("object", Path(asset_dir).name),
        points=_table("points"),
        axes=_table("axes"),
        quantities={k: float(v["value"])
                    for k, v in spec.get("quantities", {}).items()},
        statuses=statuses,
        sigmas=sigmas,
    )
    ph = [k for k, v in statuses.items() if v == "placeholder"]
    if ph:
        print(f"[frames] {sym.object}: PLACEHOLDER symbols needing "
              f"viewer calibration: {ph}")
    return sym