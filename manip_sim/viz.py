"""Scene overlays for the grounded interaction symbols.

One place owns the marker color table and the symbol->world math, so the
kinematic playback (`scripts/render_full_plan.py`) and the physical
execution (`scripts/execute_pour_tea.py`) draw the *same* points in the
*same* colors. Both consume `manip_sim.frames.Symbols`, so whatever
authors the sidecar — the hand-authored ground truth today, the VLM
grounding pipeline later — is what the video shows. The overlay is a
readout of the symbol table, never a second hand-placed copy of it.

    green   handle_center   (teapot)  the grasp point, stage-1 target
    red     spout_tip       (teapot)  the pouring feature, follows the pot
    blue    opening_center  (mug)     the transport subgoal target
    orange  teapot body origin        fallback when meshes are absent
    trail   spout-tip trace, colored by stage
            (grey = grasp, red = transport, purple = pour)

`opening_lift` raises the blue sphere along the mug's own up_axis. The
default is the center of the stage-2 standoff band (`height` in
`pour_stages.transport_pair`), i.e. the blue sphere marks where the spout
tip is actually AIMED rather than the rim plane it is aimed above. Pass
0.0 to draw the raw calibrated symbol instead. This is a drawing offset
only — it does not touch `frames.json` and does not move the TSR.
"""

from __future__ import annotations

import numpy as np

try:                                    # keeps the module importable for
    import mujoco                       # numpy-only unit tests
except ImportError:                     # pragma: no cover
    mujoco = None

from .frames import load_symbols

# center of transport_pair(height=(0.03, 0.08))
DEFAULT_OPENING_LIFT = 0.055

RGBA = {
    "tip": (0.9, 0.15, 0.15, 1.0),          # red    spout tip
    "handle": (0.15, 0.75, 0.25, 0.95),     # green  grasp point
    "opening": (0.2, 0.4, 0.95, 0.9),       # blue   mug opening
    "body": (0.95, 0.6, 0.1, 0.9),          # orange teapot origin
    "trace1": (0.55, 0.55, 0.55, 0.5),      # grey   grasp
    "trace2": (0.9, 0.15, 0.15, 0.55),      # red    transport
    "trace3": (0.6, 0.2, 0.85, 0.6),        # purple pour
}

RADIUS = {"tip": 0.014, "handle": 0.015, "opening": 0.018,
          "body": 0.03, "trace": 0.006}


def add_sphere(scene, pos, radius, rgba) -> bool:
    """Append one sphere to an MjvScene. False if the scene is full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0], float),
        np.asarray(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32))
    scene.ngeom += 1
    return True


class InteractionMarkers:
    """Symbol-table readout + scene overlay.

    Position queries are pure numpy (unit-testable without a renderer);
    `draw` is the only method that needs mujoco.
    """

    def __init__(self,
                 teapot_dir: str = "assets/objects/teapot",
                 mug_dir: str = "assets/objects/mug",
                 opening_lift: float = DEFAULT_OPENING_LIFT,
                 trail_stride: int = 2,
                 trail_max: int = 600):
        self.teapot_sym = load_symbols(teapot_dir)
        self.mug_sym = load_symbols(mug_dir)
        self.spout = self.teapot_sym.frame("spout_tip", "pour_axis")
        self.handle = self.teapot_sym.frame("handle_center", "handle_axis")
        self.opening = self.mug_sym.frame("opening_center", "up_axis")
        self.opening_lift = float(opening_lift)
        self.trail_stride = trail_stride
        self.trail_max = trail_max
        self.trail: list[tuple[np.ndarray, int]] = []

    # ------------------------------------------------------ symbol -> world

    def spout_tip(self, T0_teapot) -> np.ndarray:
        return (np.asarray(T0_teapot) @ self.spout.T())[:3, 3]

    def handle_center(self, T0_teapot) -> np.ndarray:
        return (np.asarray(T0_teapot) @ self.handle.T())[:3, 3]

    def opening_center(self, T0_mug) -> np.ndarray:
        """Mug opening center, lifted along the mug's own up_axis (the
        opening frame's +z), so the lift rides the mug if it ever tips."""
        T = np.asarray(T0_mug) @ self.opening.T()
        return T[:3, 3] + T[:3, 2] * self.opening_lift

    # ------------------------------------------------------------- trailing

    def push_trail(self, pos, stage: int) -> None:
        self.trail.append((np.asarray(pos, float).copy(), int(stage)))
        if len(self.trail) > self.trail_max:
            del self.trail[0]

    # ---------------------------------------------------------------- draw

    def draw(self, scene, T0_teapot, T0_mug, stage: int = 1,
             show_handle: bool = True, show_trail: bool = True,
             show_body: bool = False) -> None:
        """Overlay every marker for one frame. Trail first so the live
        spheres draw on top of it."""
        if show_trail:
            for p, s in self.trail[:-1][::self.trail_stride]:
                if not add_sphere(scene, p, RADIUS["trace"],
                                  RGBA.get(f"trace{s}", RGBA["trace2"])):
                    break
        if T0_mug is not None:
            add_sphere(scene, self.opening_center(T0_mug),
                       RADIUS["opening"], RGBA["opening"])
        if T0_teapot is None:
            return
        add_sphere(scene, self.spout_tip(T0_teapot),
                   RADIUS["tip"], RGBA["tip"])
        if show_handle:
            add_sphere(scene, self.handle_center(T0_teapot),
                       RADIUS["handle"], RGBA["handle"])
        if show_body:
            add_sphere(scene, np.asarray(T0_teapot)[:3, 3],
                       RADIUS["body"], RGBA["body"])