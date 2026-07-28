"""AttachedArmKinematics tests: the attached object must participate in
collision checking (the closed v1 gap). Uses a primitive-geom box object
so no converted meshes are required — the same trick works for any future
attached-collision regression test."""

from pathlib import Path

import numpy as np
import pytest

robosuite = pytest.importorskip("robosuite")
mujoco = pytest.importorskip("mujoco")

from robosuite.environments.base import register_env  # noqa: E402

from manip_sim.envs.tabletop import TableTop  # noqa: E402
from manip_sim.planning import (  # noqa: E402
    ArmKinematics,
    AttachedArmKinematics,
    AttachedObject,
)
from manip_sim.tsr import make_pose  # noqa: E402

BOX_XML = """<mujoco model="box">
  <worldbody>
    <body>
      <body name="object">
        <geom type="box" size="0.03 0.03 0.03" group="0" rgba="0.8 0.3 0.3 1"
              mass="0.1" friction="0.95 0.3 0.1" condim="4"/>
      </body>
      <site name="bottom_site" pos="0 0 -0.03" rgba="0 0 0 0" size="0.005"/>
      <site name="top_site" pos="0 0 0.03" rgba="0 0 0 0" size="0.005"/>
      <site name="horizontal_radius_site" pos="0.03 0.03 0"
            rgba="0 0 0 0" size="0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    xml = tmp_path_factory.mktemp("assets") / "box.xml"
    xml.write_text(BOX_XML)
    try:
        register_env(TableTop)
    except AssertionError:      # already registered
        pass
    env = robosuite.make(
        "TableTop", robots="UR5e", object_xmls={"box": str(xml)},
        table_full_size=(1.2, 1.2, 0.05), table_offset=(0, 0, 0.8),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20, ignore_done=True,
    )
    env.reset()
    plain = ArmKinematics(env)
    q_home = env.sim.data.qpos[plain.qpos_ids].copy()
    yield env, plain, q_home
    env.close()


def _att(env, dz):
    """Attached kinematics with the box hanging dz below the grip site."""
    att = AttachedObject(make_pose([0.0, 0.0, dz]))
    return AttachedArmKinematics(env, att,
                                 env.objects["box"].joints[0], "box"), att


def test_object_rides_the_eef(scene):
    env, plain, q_home = scene
    kin, att = _att(env, -0.10)
    T_ee = kin.fk(q_home)
    # the scratch free joint was re-pinned to the attached pose
    T_box = np.eye(4)
    bid = [b for b in range(kin.model.nbody)
           if (mujoco.mj_id2name(kin.model, mujoco.mjtObj.mjOBJ_BODY, b)
               or "").startswith("box")][0]
    T_box[:3, 3] = kin.data.xpos[bid]
    assert np.allclose(T_box[:3, 3], att.body_pose(T_ee)[:3, 3], atol=1e-9)


def test_hanging_clear_is_collision_free(scene):
    env, plain, q_home = scene
    kin, _ = _att(env, -0.10)          # eef home z ~0.98 -> box at ~0.88
    assert not kin.in_collision(q_home)


def test_attached_object_table_hit_is_detected(scene):
    env, plain, q_home = scene
    # hang the box low enough that it penetrates the table top at home;
    # the ARM itself is collision-free, so the PLAIN checker misses it —
    # that miss is precisely the closed gap
    kin, _ = _att(env, -0.18)          # box center ~0.80 -> inside table
    assert not plain.in_collision(q_home)
    assert kin.in_collision(q_home)


def test_gripper_object_contact_is_the_grasp_not_a_collision(scene):
    env, plain, q_home = scene
    kin, _ = _att(env, -0.02)          # box overlapping the fingertips
    assert not kin.in_collision(q_home)