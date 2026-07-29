"""AnyGrasp reply -> GraspProposal adapter.

This is the real half of the swap the grasping module docstring promises:
propose_handle_grasps (the synthetic stand-in) and anygrasp_proposals emit
the same list[GraspProposal]; classifier, IK, lookahead probe, and the
approach machinery downstream are untouched.

Frame conversion (the one place conventions collide):

  AnyGrasp / graspnetAPI grasp frame:   +x = approach, +y = closing axis
  robosuite grip-site frame (this repo): +z = approach, +x = closing axis

  R_grip = R_ag @ AG_TO_GRIP   with AG_TO_GRIP mapping (x,y,z)_grip ->
  (y,z,x)_ag; right-handed (checked: x_grip x y_grip = z_grip).

Translation: graspnet's translation is the grasp center between the finger
contact regions, which is close to but not exactly robosuite's grip site
origin. `tcp_offset` shifts the pose along the approach axis (meters,
positive = deeper toward the object) to absorb the residual; calibrate it
by watching the classifier's lateral/slide displacement on the survivors
(test_anygrasp.py prints per-axis TSR displacement for exactly this).

Wrist symmetry: AnyGrasp's closing-axis SIGN is arbitrary for a
parallel-jaw gripper, and the 180-deg twin sits at yaw ~ pi in the grasp
TSR — outside wrap_rot, so the classifier would reject a physically
identical grasp if only one branch were emitted. Both branches are emitted
as separate proposals; the redundant IK-time wrist_flip downstream is
harmless (duplicate candidates, same physical grasp).
"""

from __future__ import annotations

import numpy as np

from ..grasping import GraspProposal, wrist_flip
from ..tsr import make_pose

# columns = grip-site axes expressed in the AnyGrasp grasp frame:
# x_grip = y_ag (closing), y_grip = z_ag, z_grip = x_ag (approach)
AG_TO_GRIP = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def anygrasp_pose_to_grip_site(
    t: np.ndarray, R_ag: np.ndarray, tcp_offset: float = 0.0
) -> np.ndarray:
    """One AnyGrasp (translation, rotation) in the CLOUD frame -> grip-site
    pose in the same frame."""
    approach = R_ag[:, 0]
    return make_pose(np.asarray(t, float) + tcp_offset * approach,
                     R_ag @ AG_TO_GRIP)


def anygrasp_proposals(
    reply: dict,
    T_world_cloud: np.ndarray,
    tcp_offset: float = 0.0,
    min_score: float | None = None,
    max_n: int | None = None,
    emit_flip: bool = True,
) -> list[GraspProposal]:
    """GraspClient reply -> world-frame GraspProposals, best score first.

    reply          dict from GraspClient.get_grasps (server sorts by score)
    T_world_cloud  pose of the frame the cloud was expressed in (the CV
                   camera pose from perception.cloud.render_cloud; identity
                   if you sent a world-frame cloud)
    """
    if reply.get("n", 0) == 0:
        return []
    ts, Rs, scores = reply["translations"], reply["rotations"], reply["scores"]
    out: list[GraspProposal] = []
    for i in range(len(ts)):
        if min_score is not None and scores[i] < min_score:
            break                              # sorted: rest are lower
        if max_n is not None and i >= max_n:
            break
        T_cloud_grip = anygrasp_pose_to_grip_site(ts[i], Rs[i], tcp_offset)
        T0 = T_world_cloud @ T_cloud_grip
        out.append(GraspProposal(T0, "anygrasp"))
        if emit_flip:
            out.append(GraspProposal(wrist_flip(T0), "anygrasp"))
    return out


def steer_toward(a_world: np.ndarray, T_world_cloud: np.ndarray) -> list:
    """Express a desired world-frame approach direction in the cloud frame,
    for the server's approach_steering param. Use the grasp TSR's nominal
    approach (+z of T0_w @ Tw_e) to bias AnyGrasp toward the reachable
    elevation band instead of rejecting its top-down favorites later."""
    a = np.asarray(a_world, float)
    a = a / np.linalg.norm(a)
    return list(T_world_cloud[:3, :3].T @ a)