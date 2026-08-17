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

# ===================================================================== debug
# Frame / TSR overlay. Everything below is a READOUT of objects the planner
# and executor already hold — frames come from whatever authored them (hand
# sidecar or resolved VLM selections), TSR boxes come from the TSR's own
# T0_w and B^w. Nothing here re-derives geometry, so a disagreement on
# screen is a real disagreement in the pipeline.

AXIS_RGBA = ((0.95, 0.25, 0.25, 1.0),      # +x red
             (0.25, 0.90, 0.35, 1.0),      # +y green
             (0.35, 0.55, 1.00, 1.0))      # +z blue

DEBUG_RGBA = {
    "pred": (1.00, 0.85, 0.10, 0.95),      # yellow  rigid-prediction tip
    "slip": (1.00, 0.55, 0.00, 0.95),      # orange  measured <- predicted
    "tsr_transport": (0.20, 0.45, 0.95, 0.16),
    "tsr_pour": (0.65, 0.25, 0.90, 0.20),
    "stream_hit": (0.20, 0.85, 0.35, 0.85),
    "stream_miss": (0.95, 0.20, 0.20, 0.85),
}


def rot_to(d) -> np.ndarray:
    """Rotation whose +z is the unit direction d (MuJoCo arrows, capsules
    and lines all extend along the geom's local +z)."""
    z = np.asarray(d, float)
    n = np.linalg.norm(z)
    if n < 1e-12:
        return np.eye(3)
    z = z / n
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = a - np.dot(a, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _init(scene, gtype, size, pos, mat, rgba) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], gtype,
        np.asarray(size, float), np.asarray(pos, float),
        np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1
    return True


def add_arrow(scene, origin, direction, length, rgba, radius=0.0035) -> bool:
    """Arrow from `origin` along `direction`, `length` metres."""
    return _init(scene, mujoco.mjtGeom.mjGEOM_ARROW,
                 [radius, radius, length], origin,
                 rot_to(direction), rgba)


def add_segment(scene, a, b, rgba, radius=0.0025) -> bool:
    """Capsule spanning a->b. Version-safe substitute for mjv_connector."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return True
    return _init(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                 [radius, radius, 0.5 * L], 0.5 * (a + b), rot_to(d), rgba)


def add_frame(scene, T, length=0.05, radius=0.0035, alpha=1.0) -> None:
    """RGB triad for a 4x4 world pose: +x red, +y green, +z blue."""
    T = np.asarray(T, float)
    for i in range(3):
        r, g, b, a = AXIS_RGBA[i]
        add_arrow(scene, T[:3, 3], T[:3, i], length, (r, g, b, a * alpha),
                  radius)


def add_tsr_bounds(scene, tsr, rgba, max_extent=0.30) -> bool:
    """Translucent box over the TRANSLATION rows of B^w, drawn in w.

    d[:3] in TSR.displacement is the position of (T0_e @ inv(Tw_e)) in w,
    so this box is exactly the admissible region of the feature the TSR
    pins — the spout tip for the transport subgoal, the pivot for the
    pour pair. Infinite (FREE_TRANS) rows are clipped to max_extent so a
    path TSR still draws as a slab.
    """
    Bw = np.asarray(tsr.Bw, float)[:3]
    lo = np.where(np.isfinite(Bw[:, 0]), Bw[:, 0], -max_extent)
    hi = np.where(np.isfinite(Bw[:, 1]), Bw[:, 1], max_extent)
    half = np.maximum(0.5 * (hi - lo), 1e-4)
    ctr = np.asarray(tsr.T0_w, float) @ np.append(0.5 * (lo + hi), 1.0)
    return _init(scene, mujoco.mjtGeom.mjGEOM_BOX, half, ctr[:3],
                 np.asarray(tsr.T0_w, float)[:3, :3], rgba)


def stream_landing(tip, pour_dir_w, opening_pos, opening_up):
    """Where a stream leaving `tip` along `pour_dir_w` crosses the mug's
    opening plane. None if the spout is not pointing at that plane (i.e.
    nothing would pour). Gravity-free straight-line proxy — the honest
    static predicate for 'would the liquid land inside the rim'."""
    d = np.asarray(pour_dir_w, float)
    d = d / max(np.linalg.norm(d), 1e-12)
    up = np.asarray(opening_up, float)
    denom = float(d @ up)
    if denom > -1e-6:
        return None
    t = float((np.asarray(opening_pos, float) - np.asarray(tip, float)) @ up)
    return np.asarray(tip, float) + (t / denom) * d


class DebugOverlay:
    """Per-frame overlay of the frames and TSRs actually in force.

    Constructed with the SAME Frame objects the executor passes to
    pour_stages, so on the vlm arm it draws the VLM's selections and on
    the hand arm it draws the sidecar composition — there is no second
    copy of the symbol table here.
    """

    def __init__(self, spout_tip, tilt_frame, opening, handle=None,
                 rim_radius=0.044, axis_len=0.06):
        self.spout_tip = spout_tip
        self.tilt_frame = tilt_frame
        self.opening = opening
        self.handle = handle
        self.rim_radius = float(rim_radius)
        self.axis_len = float(axis_len)

    # ---------------------------------------------------------- readouts
    def tip_world(self, T0_teapot):
        return (np.asarray(T0_teapot, float) @ self.spout_tip.T())[:3, 3]

    def pour_dir_world(self, T0_teapot):
        """+z of the transport_active frame = the pour axis, signed as
        selected."""
        return (np.asarray(T0_teapot, float) @ self.spout_tip.T())[:3, 2]

    def opening_world(self, T0_mug):
        T = np.asarray(T0_mug, float) @ self.opening.T()
        return T[:3, 3], T[:3, 2]

    def report(self, T0_teapot, T0_mug) -> dict:
        """The pour success readout, decomposed. `tip_to_opening_mm` alone
        is uninterpretable: it folds in the transport subgoal's DELIBERATE
        standoff band, so a perfect run reports 40-100 mm."""
        o, up = self.opening_world(T0_mug)
        tip = self.tip_world(T0_teapot)
        d = tip - o
        standoff = float(d @ up)
        lateral = float(np.linalg.norm(d - standoff * up))
        pd = self.pour_dir_world(T0_teapot)
        land = stream_landing(tip, pd, o, up)
        out = {
            "tip_lateral_mm": lateral * 1000.0,
            "tip_standoff_mm": standoff * 1000.0,
            "tip_to_opening_mm": float(np.linalg.norm(d)) * 1000.0,
            "tip_over_rim": bool(lateral <= self.rim_radius),
            "spout_declination_deg": float(np.rad2deg(np.arccos(
                np.clip(np.dot(pd / max(np.linalg.norm(pd), 1e-12),
                               -up), -1.0, 1.0)))),
            "rim_radius_mm": self.rim_radius * 1000.0,
        }
        if land is None:
            out.update(stream_lands_in_mug=False, stream_lateral_mm=None)
        else:
            sl = float(np.linalg.norm(land - o))
            out.update(stream_lands_in_mug=bool(sl <= self.rim_radius),
                       stream_lateral_mm=sl * 1000.0)
        return out

    # -------------------------------------------------------------- draw
    def draw(self, scene, T0_teapot, T0_mug, T_body_pred=None, tsrs=()):
        """T_body_pred: the pose the frozen T_ee_body predicts. Drawing it
        next to the measured pose makes slip visible — the orange segment
        IS the slip, at the spout tip where it costs the task."""
        o, up = self.opening_world(T0_mug)
        for tsr, key in tsrs:
            add_tsr_bounds(scene, tsr, DEBUG_RGBA[key])
        add_frame(scene, np.asarray(T0_mug, float) @ self.opening.T(),
                  self.axis_len)
        if T0_teapot is None:
            return
        T0_teapot = np.asarray(T0_teapot, float)
        add_frame(scene, T0_teapot @ self.spout_tip.T(), self.axis_len)
        add_frame(scene, T0_teapot @ self.tilt_frame.T(),
                  self.axis_len * 0.8, alpha=0.65)
        if self.handle is not None:
            add_frame(scene, T0_teapot @ self.handle.T(),
                      self.axis_len * 0.7, alpha=0.5)
        tip = self.tip_world(T0_teapot)
        if T_body_pred is not None:
            tip_pred = self.tip_world(T_body_pred)
            add_sphere(scene, tip_pred, RADIUS["tip"] * 0.8,
                       DEBUG_RGBA["pred"])
            add_segment(scene, tip_pred, tip, DEBUG_RGBA["slip"])
        land = stream_landing(tip, self.pour_dir_world(T0_teapot), o, up)
        if land is not None:
            hit = float(np.linalg.norm(land - o)) <= self.rim_radius
            key = "stream_hit" if hit else "stream_miss"
            add_segment(scene, tip, land, DEBUG_RGBA[key], radius=0.0035)
            add_sphere(scene, land, 0.010, DEBUG_RGBA[key])
