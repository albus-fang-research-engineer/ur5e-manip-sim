"""Object task frames from per-asset sidecar files (frames.json).

Each converted asset directory carries a frames.json declaring named
interaction frames as (point, primary axis) pairs in the object's *body*
coordinates (the frame of the mjcf body named "object", i.e. what
PoseReader returns for that object). A task frame is completed to a
right-handed basis by Gram-Schmidt of a secondary direction against the
primary axis, with the primary axis mapped to +z — exactly the frame
construction the grounding pipeline (mesh candidates -> canonical renders
-> VLM multiple choice + PointSO axes) will eventually emit. Keeping the
schema identical means the VLM later becomes a *producer of this same
artifact* and nothing downstream changes; the hand-authored sidecars in
assets/ double as the ground-truth arm of the emission ablation.

Schema:

{
  "object": "teapot",
  "frames": {
    "handle":    {"point": [x,y,z], "axis": [x,y,z],
                  "secondary": [x,y,z] | null,      # null -> body -z
                  "status": "calibrated" | "placeholder",
                  "comment": "free text"},
    ...
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
        Rm = np.column_stack([x, y, z])
        return make_pose(self.point, Rm)

    def world_T(self, T0_body: np.ndarray) -> np.ndarray:
        """World pose of this frame given the object's body world pose."""
        return T0_body @ self.T()


def load_frames(asset_dir: str | Path) -> dict[str, Frame]:
    path = Path(asset_dir) / "frames.json"
    spec = json.loads(path.read_text())
    out: dict[str, Frame] = {}
    for name, f in spec["frames"].items():
        sec = f.get("secondary")
        out[name] = Frame(
            name=name,
            point=np.asarray(f["point"], dtype=float),
            axis=np.asarray(f["axis"], dtype=float),
            secondary=np.asarray(sec if sec is not None else [0.0, 0.0, -1.0], float),
            status=f.get("status", "calibrated"),
            comment=f.get("comment", ""),
        )
    placeholders = [n for n, fr in out.items() if fr.status == "placeholder"]
    if placeholders:
        print(f"[frames] {spec.get('object', path)}: PLACEHOLDER frames "
              f"needing calibration: {placeholders}")
    return out