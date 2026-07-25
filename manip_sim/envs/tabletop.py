"""TableTop: a minimal robosuite task that loads arbitrary converted objects.

This is the platform skeleton — the "house" the purpose-built constrained
scenes (pour, insertion, ...) will subclass. It accepts any objects produced
by scripts/convert_asset.py and places them on a table.

Usage:
    import robosuite as suite
    import manip_sim  # registers "TableTop"

    env = suite.make(
        "TableTop",
        robots="UR5e",
        object_xmls={"mug": "assets/objects/mug/mug.xml"},
        ...
    )

Every object gets standard observables: <name>_pos, <name>_quat (xyzw).
"""

from pathlib import Path

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial  # noqa: F401  (subclass convenience)
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class TableTop(ManipulationEnv):
    """Single-arm tabletop scene with user-supplied XML objects."""

    def __init__(
        self,
        robots,
        object_xmls: dict[str, str] | None = None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0.0, 0.0, 0.8),
        placement_x_range=(-0.15, 0.15),
        placement_y_range=(-0.15, 0.15),
        placement_rotation=None,  # None -> random z-rotation
        reward_scale=1.0,
        **kwargs,
    ):
        self.object_xmls = {k: Path(v) for k, v in (object_xmls or {}).items()}
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.placement_x_range = placement_x_range
        self.placement_y_range = placement_y_range
        self.placement_rotation = placement_rotation
        self.reward_scale = reward_scale
        self.objects: dict[str, MujocoXMLObject] = {}
        self.obj_body_ids: dict[str, int] = {}
        super().__init__(robots=robots, **kwargs)

    # ------------------------------------------------------------------ model
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.objects = {
            name: MujocoXMLObject(str(path.resolve()), name=name)
            for name, path in self.object_xmls.items()
        }

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=list(self.objects.values()),
            x_range=list(self.placement_x_range),
            y_range=list(self.placement_y_range),
            rotation=self.placement_rotation,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.objects.values()),
        )

    # ------------------------------------------------------------- references
    def _setup_references(self):
        super()._setup_references()
        self.obj_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in self.objects.items()
        }

    # ------------------------------------------------------------ observables
    def _setup_observables(self):
        observables = super()._setup_observables()
        modality = "object"

        def make_pos_sensor(body_id):
            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[body_id])

            return obj_pos

        def make_quat_sensor(body_id):
            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

            return obj_quat

        for name, body_id in self.obj_body_ids.items():
            for suffix, factory in (("pos", make_pos_sensor), ("quat", make_quat_sensor)):
                s = factory(body_id)
                s.__name__ = f"{name}_{suffix}"
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )
        return observables

    # ------------------------------------------------------------------ reset
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset and self.objects:
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    # ------------------------------------------------------------ task logic
    def reward(self, action=None):
        return 0.0  # subclasses define task rewards

    def _check_success(self):
        return False  # subclasses define success predicates
