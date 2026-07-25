"""End-to-end demo of the full loop:

  converted mesh -> TableTop env -> real-time poses (gt + noisy) -> point cloud

Run after converting at least one asset:
  python scripts/convert_asset.py assets/raw/mug.obj --name mug --mass 0.3
  python scripts/demo_tabletop.py
"""

import numpy as np
import robosuite as suite
from PIL import Image

import manip_sim  # noqa: F401  registers TableTop
from manip_sim.state import (
    NoiseConfig,
    NoisyPoseSensor,
    PoseReader,
    depth_to_pointcloud,
)


def main() -> None:
    env = suite.make(
        "TableTop",
        robots="UR5e",
        object_xmls={"mug": "assets/objects/mug/mug.xml"},
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=True,
        control_freq=20,
    )
    obs = env.reset()

    # Observable route (per-control-step, built into obs dict):
    print("obs route     :", np.round(obs["mug_pos"], 3), "quat(xyzw)", np.round(obs["mug_quat"], 3))

    # Direct route (any time, exact):
    reader = PoseReader(env, ["mug_main"])
    sensor = NoisyPoseSensor(reader, NoiseConfig(trans_std=0.005, rot_std_deg=2.0,
                                                 latency_steps=2, dropout_prob=0.1, seed=0))

    print(f"\n{'step':>4} {'gt pos':>28} {'noisy pos':>28} {'valid':>6} {'err(mm)':>8}")
    for t in range(30):
        a = np.zeros(env.action_dim)
        a[0] = 0.1
        obs, *_ = env.step(a)
        sensor.step()
        gt_pos, gt_quat = reader.pose("mug_main")
        n_pos, n_quat, valid = sensor.pose("mug_main")
        if t % 5 == 0 or not valid:
            err = np.linalg.norm(gt_pos - n_pos) * 1000
            print(f"{t:>4} {np.round(gt_pos,3)!s:>28} {np.round(n_pos,3)!s:>28} {valid!s:>6} {err:>8.1f}")

    # 4x4 matrix + velocity access
    T = reader.pose_matrix("mug_main")
    lin, ang = reader.velocity("mug_main")
    print("\npose matrix det(R):", round(float(np.linalg.det(T[:3, :3])), 6))
    print("velocity |lin|, |ang|:", round(float(np.linalg.norm(lin)), 4), round(float(np.linalg.norm(ang)), 4))

    # Depth -> world point cloud (the estimated-pose arm's input)
    pc = depth_to_pointcloud(env, "agentview", obs["agentview_depth"])
    near_table = pc[(np.abs(pc[:, 0]) < 0.35) & (np.abs(pc[:, 1]) < 0.35) & (pc[:, 2] > 0.7) & (pc[:, 2] < 1.0)]
    print("point cloud   :", pc.shape[0], "pts total,", near_table.shape[0], "in table volume")
    print("z range near table:", np.round(near_table[:, 2].min(), 3), "-", np.round(near_table[:, 2].max(), 3))

    Image.fromarray(obs["agentview_image"][::-1]).save("tabletop_demo.png")
    env.close()
    print("\nDEMO PASSED — render saved to tabletop_demo.png")


if __name__ == "__main__":
    main()
