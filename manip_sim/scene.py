"""Scene manifest: the one place that says which objects a sim scene
contains and where they start.

    scenes/pour_tea.json
      task         default task string for the VLM planner
      table        {size, top_z}
      drop_height  objects spawn this far above the table and settle
      objects      name -> {asset: dir with <name>.xml / frames.json /
                            candidates.json,
                            placement: {xy, yaw}}
      yaw          number (rad), or {"face": <object>, "along": <axis>}
                   = rotate so frames.json axis `along` points at the
                   other object (how the teapot spout faces the mug)

Two consumers, deliberately separate:

  load_scene()/make_env()   build the environment. Ground truth.
  Scene.asset_dirs          the per-object artifact dirs every script
                            used to hardcode as OBJECTS.

Neither is VLM-facing. When call #1 becomes image-conditioned the model
sees marks rendered from these bodies, never the names in this file; the
verifier (scripts/plan_stages.py) is the only place allowed to compare
an emitted plan against the manifest.

robosuite is imported lazily so `load_scene` works in test/CI contexts
without MuJoCo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SCENE = Path("scenes/pour_tea.json")


@dataclass(frozen=True)
class Placement:
    xy: tuple[float, float]
    yaw: float | dict          # rad, or {"face": obj, "along": axis}


@dataclass(frozen=True)
class SceneObject:
    name: str
    asset: Path
    placement: Placement

    @property
    def xml(self) -> Path:
        return self.asset / f"{self.name}.xml"


@dataclass(frozen=True)
class Scene:
    name: str
    task: str
    table_size: tuple[float, float, float]
    table_top_z: float
    drop_height: float
    settle_steps: int
    objects: dict[str, SceneObject] = field(default_factory=dict)
    path: Path | None = None

    # -- what the scripts consume --------------------------------------
    @property
    def asset_dirs(self) -> dict[str, Path]:
        return {n: o.asset for n, o in self.objects.items()}

    @property
    def object_xmls(self) -> dict[str, str]:
        return {n: str(o.xml) for n, o in self.objects.items()}

    def xy(self, name: str) -> np.ndarray:
        return np.asarray(self.objects[name].placement.xy, dtype=float)

    @property
    def spawn_z(self) -> float:
        return self.table_top_z + self.drop_height

    def yaw(self, name: str) -> float:
        """Resolved spawn yaw (rad). `face` yaws are resolved through
        frames.json so the manifest carries no hand-typed axis offsets."""
        y = self.objects[name].placement.yaw
        if not isinstance(y, dict):
            return float(y)
        spec = json.loads((self.objects[name].asset / "frames.json").read_text())
        axis = np.asarray(spec["axes"][y["along"]]["xyz"], dtype=float)
        body_yaw = float(np.arctan2(axis[1], axis[0]))
        bearing = self.xy(y["face"]) - self.xy(name)
        return float(np.arctan2(bearing[1], bearing[0])) - body_yaw

    def fixed_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """name -> (pos[3], quat_wxyz[4]) at spawn."""
        return {n: (np.array([*self.xy(n), self.spawn_z]), yaw_quat_wxyz(self.yaw(n)))
                for n in self.objects}


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def load_scene(path: str | Path = DEFAULT_SCENE) -> Scene:
    path = Path(path)
    doc = json.loads(path.read_text())
    objects = {}
    for name, o in doc["objects"].items():
        p = o["placement"]
        objects[name] = SceneObject(
            name=name, asset=Path(o["asset"]),
            placement=Placement(xy=tuple(p["xy"]), yaw=p.get("yaw", 0.0)))
    t = doc["table"]
    return Scene(name=doc["name"], task=doc["task"],
                 table_size=tuple(t["size"]), table_top_z=float(t["top_z"]),
                 drop_height=float(doc.get("drop_height", 0.06)),
                 settle_steps=int(doc.get("settle_steps", 20)),
                 objects=objects, path=path)


# ---------------------------------------------------------------- argparse

def add_scene_arg(ap) -> None:
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), metavar="JSON",
                    help="scene manifest (objects, assets, placement)")


# ----------------------------------------------------------------- env

_ENV_CLS = None


def _env_class():
    """Define + register the robosuite env once per process (lazy so this
    module imports without MuJoCo)."""
    global _ENV_CLS
    if _ENV_CLS is not None:
        return _ENV_CLS
    from robosuite.environments.base import register_env

    from manip_sim.envs.tabletop import TableTop

    class FixedPoseScene(TableTop):
        """TableTop with deterministic placement from the manifest."""

        def __init__(self, robots, fixed_poses, **kwargs):
            self.fixed_poses = fixed_poses
            super().__init__(robots=robots, **kwargs)

        def _reset_internal(self):
            super()._reset_internal()
            for name, (pos, quat) in self.fixed_poses.items():
                if name in self.objects:
                    self.sim.data.set_joint_qpos(
                        self.objects[name].joints[0], np.concatenate([pos, quat]))
            self.sim.forward()

    register_env(FixedPoseScene)
    _ENV_CLS = FixedPoseScene
    return _ENV_CLS


def make_env(scene: Scene, robot: str = "UR5e", has_renderer: bool = True,
             settle: bool = True, **make_kwargs):
    """THE scene factory. Planners, renderers, demos all build here so
    setup cannot drift. Returns (env, objs); objs maps loaded object
    names to xml paths (objects whose asset is missing are skipped)."""
    import robosuite as suite
    _env_class()

    objs = {}
    for name, o in scene.objects.items():
        if o.xml.exists():
            objs[name] = str(o.xml)
        else:
            print(f"[{scene.name}] skipping '{name}' (not converted yet: {o.xml})")
    poses = {n: p for n, p in scene.fixed_poses().items() if n in objs}

    def _try(obj_xmls, fixed):
        kw = dict(robots=robot, object_xmls=obj_xmls, fixed_poses=fixed,
                  table_full_size=scene.table_size,
                  table_offset=(0.0, 0.0, scene.table_top_z),
                  has_renderer=has_renderer, render_camera=None,
                  has_offscreen_renderer=False, use_camera_obs=False,
                  control_freq=20, ignore_done=True)
        kw.update(make_kwargs)
        try:
            return suite.make("FixedPoseScene", **kw)
        except ValueError as e:
            if obj_xmls and ("No such file" in str(e) or "Error opening file" in str(e)):
                return None
            raise

    env = _try(objs, poses)
    if env is None:
        print(f"[{scene.name}] mesh files missing -> building object-free scene")
        objs = {}
        env = _try({}, {})
    env.reset()
    if settle:
        for _ in range(scene.settle_steps):
            env.step(np.zeros(env.action_dim))
            if has_renderer:
                env.render()
    return env, objs
