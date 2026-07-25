"""demo_pour_tea.py — deterministic pour-task scene setup.

Scene: a bigger table (1.2 m x 1.2 m, transport room for the path-constraint
demo), with the teapot and mug at FIXED poses: 0.5 m apart, teapot's spout
axis (body-frame +x) yawed to face the mug.

The fixed placement is done by subclassing TableTop and overriding
_reset_internal to write object joint qpos directly, bypassing the random
sampler. This is the pattern all deterministic evaluation scenes will use.

Run:
  PYTHONPATH=. python scripts/demo_pour_tea.py

Notes:
  * SPOUT_YAW_OFFSET: if your downloaded teapot mesh's spout does not point
    along body-frame +x, set this to the yaw (rad) that rotates body +x onto
    the spout direction, and the facing computation absorbs it.
  * Objects are dropped a few mm above the table and settled for a second of
    sim time so they rest on their CoACD collision geometry before poses are
    reported.
"""

import numpy as np
import robosuite as suite
from PIL import Image
from robosuite.environments.base import register_env

import manip_sim  # noqa: F401
from manip_sim.envs.tabletop import TableTop
from manip_sim.state import PoseReader

# ----------------------------------------------------------------- scene spec
TABLE_SIZE = (1.2, 1.2, 0.05)
TABLE_TOP_Z = 0.8

TEAPOT_XY = np.array([0.10, -0.25])
MUG_XY = np.array([0.10, 0.25])          # 0.5 m from the teapot
SPOUT_YAW_OFFSET = 0.0                    # rad; see module docstring
DROP_HEIGHT = 0.06                        # m above tabletop at reset
SETTLE_STEPS = 20                         # control steps (~1 s at 20 Hz)


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class PourTeaScene(TableTop):
    """TableTop with deterministic object placement."""

    def __init__(self, robots, fixed_poses: dict[str, tuple[np.ndarray, np.ndarray]], **kwargs):
        # name -> (pos[3], quat_wxyz[4])
        self.fixed_poses = fixed_poses
        super().__init__(robots=robots, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()  # samples randomly; we overwrite below
        for name, (pos, quat) in self.fixed_poses.items():
            obj = self.objects[name]
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([pos, quat])
            )
        self.sim.forward()


register_env(PourTeaScene)


def main() -> None:
    # teapot yaw: rotate body +x onto the bearing toward the mug
    bearing = MUG_XY - TEAPOT_XY
    teapot_yaw = np.arctan2(bearing[1], bearing[0]) - SPOUT_YAW_OFFSET

    z0 = TABLE_TOP_Z + DROP_HEIGHT
    fixed_poses = {
        "teapot": (np.array([*TEAPOT_XY, z0]), yaw_quat_wxyz(teapot_yaw)),
        "mug": (np.array([*MUG_XY, z0]), yaw_quat_wxyz(0.0)),
    }

    env = suite.make(
        "PourTeaScene",
        robots="UR5e",
        object_xmls={
            "teapot": "assets/objects/teapot/teapot.xml",
            "mug": "assets/objects/mug/mug.xml",
        },
        fixed_poses=fixed_poses,
        table_full_size=TABLE_SIZE,
        table_offset=(0.0, 0.0, TABLE_TOP_Z),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "birdview"],
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
    )
    obs = env.reset()

    # settle onto collision geometry
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(np.zeros(env.action_dim))

    reader = PoseReader(env, ["teapot_main", "mug_main"])
    tp_pos, tp_quat = reader.pose("teapot_main")
    mug_pos, _ = reader.pose("mug_main")

    dist = np.linalg.norm((mug_pos - tp_pos)[:2])
    # facing check: angle between teapot body +x (in world) and bearing to mug
    T = reader.pose_matrix("teapot_main")
    spout_dir = T[:3, 0][:2]
    spout_dir /= np.linalg.norm(spout_dir)
    bearing_w = (mug_pos - tp_pos)[:2]
    bearing_w /= np.linalg.norm(bearing_w)
    facing_err_deg = np.rad2deg(np.arccos(np.clip(spout_dir @ bearing_w, -1, 1)))

    print(f"teapot settled: pos {np.round(tp_pos, 3)}")
    print(f"mug    settled: pos {np.round(mug_pos, 3)}")
    print(f"separation    : {dist:.3f} m   (target 0.500)")
    print(f"facing error  : {facing_err_deg:.1f} deg  (spout +x vs bearing to mug)")

    for cam in ("agentview", "birdview"):
        Image.fromarray(obs[f"{cam}_image"][::-1]).save(f"pour_tea_{cam}.png")
    env.close()
    print("renders: pour_tea_agentview.png, pour_tea_birdview.png")

    ok = abs(dist - 0.5) < 0.03 and facing_err_deg < 10
    print("POUR SCENE OK" if ok else "POUR SCENE CHECK FAILED — see numbers above")


if __name__ == "__main__":
    main()