"""Stage-2 (transport) pipeline, end to end, no gripper physics:

    sample transport subgoal INTERSECT upright path      (tsr)
      -> eef goals via measured grasp transform          (T_ee_body)
      -> multi-seed mink IK + collision filter           (goal funnel)
      -> RRTConnect on the projected TSR manifold        (planning)
      -> verify + save path; optional viewer replay

The grasp is kinematically faked: the teapot is attached hanging below the
gripper (T_ee_body chosen so the teapot is upright at the home config),
standing in for the measured post-grasp transform until stage 1 exists.
The printed goal FUNNEL (sampled -> IK-feasible -> collision-free) is the
architecture's goal-generation diagnostic; expect attrition at each step.

Headless run (works without meshes converted -- falls back to the demo
scene's synthetic mug pose and skips object collision geometry):

    PYTHONPATH=. python scripts/plan_transport.py

With viewer replay of the planned path (X11 container):

    PYTHONPATH=. python scripts/plan_transport.py --view
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import robosuite as suite  # noqa: F401  (env built via the scene factory)
from scipy.spatial.transform import Rotation as R

from scripts.demos.demo_pour_tea import MUG_XY, make_env
from manip_sim.frames import load_symbols
from manip_sim.planning import ArmKinematics, AttachedObject, MinkIK, plan_constrained
from manip_sim.pour_stages import transport_pair
from manip_sim.tsr import make_pose, pose_from_pos_quat_wxyz, sample_intersection

TABLE_TOP_Z = 0.8
HANG = 0.12                       # teapot body this far below the grip site
                                  # (0.16 clipped the pot ~2 cm into the
                                  #  table: mesh extends ~4 cm below its
                                  #  body origin, eef home z is 0.982)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--n-goal-samples", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--out", default="outputs/plans/transport_path.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # THE canonical scene: meshes, fixed poses, calibrated spout yaw --
    # built by the same factory as the demos so nothing can drift.
    env, objs = make_env(robot=args.robot, has_renderer=args.view)

    teapot_sym = load_symbols("assets/objects/teapot")
    mug_sym = load_symbols("assets/objects/mug")
    spout_tip = teapot_sym.frame("spout_tip", "pour_axis")
    opening = mug_sym.frame("opening_center", "up_axis")

    if "mug" in objs:
        from manip_sim.state import PoseReader
        T0_mug = pose_from_pos_quat_wxyz(*PoseReader(env, ["mug"]).pose("mug"))
        print(f"[plan_transport] mug pose from sim: {np.round(T0_mug[:3, 3], 3)}")
    else:
        T0_mug = make_pose([*MUG_XY, TABLE_TOP_Z + 0.06])
        print("[plan_transport] mug mesh absent -> synthetic mug pose "
              f"{np.round(T0_mug[:3, 3], 3)}")

    # ---- kinematics + fake grasp ------------------------------------------
    kin = ArmKinematics(env)
    q_home = env.sim.data.qpos[kin.qpos_ids].copy()
    T0_ee_home = kin.fk(q_home)
    # teapot upright, hanging HANG below the grip site at home
    T0_body_home = make_pose(T0_ee_home[:3, 3] - [0.0, 0.0, HANG])
    attached = AttachedObject(np.linalg.inv(T0_ee_home) @ T0_body_home)
    print(f"[plan_transport] fake grasp: teapot body at "
          f"{np.round(T0_body_home[:3, 3], 3)} (eef {np.round(T0_ee_home[:3, 3], 3)})")
    # the real teapot is now "in hand": park its mesh far ABOVE the scene in
    # the planner's scratch world so it isn't a phantom obstacle at its old
    # table spot. Above, not below: the floor is an infinite plane, and a
    # body parked under it is in permanent deep contact at every arm config,
    # which turned in_collision() into constant-True on machines with the
    # mesh loaded. (The attached copy itself still has no collision geometry
    # -- the known v1 gap until the weld lands.)
    if "teapot" in objs:
        jname = env.objects["teapot"].joints[0]
        jid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = kin.model.jnt_qposadr[jid]
        kin.data.qpos[adr: adr + 3] = [0.0, 0.0, 5.0]
        mujoco.mj_forward(kin.model, kin.data)

    # ---- stage-2 TSR pair + goal funnel -----------------------------------
    pair = transport_pair(
        T0_mug_body=T0_mug, mug_opening=opening, spout_tip=spout_tip,
        teapot_body_pos_now=T0_body_home[:3, 3],
        rim_margin=mug_sym.quantities.get("rim_radius", 0.04) * 0.5,
        # start hangs at the frame origin: corridor must contain z=0
        z_corridor=(-0.01, 0.45),
    )
    rep = sample_intersection(pair.subgoal, [pair.path],
                              n=args.n_goal_samples, rng=rng)
    print(f"[plan_transport] intersection: {rep.summary()}")

    ik = MinkIK(kin)
    lo, hi = kin.joint_range[:, 0], kin.joint_range[:, 1]
    seeds = [q_home] + [rng.uniform(np.maximum(lo, -np.pi),
                                    np.minimum(hi, np.pi)) for _ in range(3)]
    goals, n_ik_ok, n_col_ok = [], 0, 0
    t0 = time.time()
    for T_body_goal in rep.accepted:
        T_ee_goal = T_body_goal @ np.linalg.inv(attached.T_ee_body)
        q, ok = ik.solve_multiseed(T_ee_goal, seeds)
        if not ok:
            continue
        n_ik_ok += 1
        if kin.in_collision(q):
            continue
        n_col_ok += 1
        # containment of the *achieved* config (IK residual could drift out)
        T_body_ach = attached.body_pose(kin.fk(q))
        if pair.path.contains(T_body_ach, tol=5e-3) and \
           pair.subgoal.contains(T_body_ach, tol=5e-3):
            goals.append(q)
    print(f"[plan_transport] goal funnel: {len(rep.accepted)} sampled -> "
          f"{n_ik_ok} IK-feasible -> {n_col_ok} collision-free -> "
          f"{len(goals)} contained  ({time.time() - t0:.1f}s)")
    if not goals:
        raise SystemExit("[plan_transport] no feasible goal configs -- widen "
                         "bounds, add IK seeds, or check reachability.")
    # prefer the goal config nearest home: distant IK branches make RRT find
    # legal but silly paths that swing the arm the long way around
    goals.sort(key=lambda q: float(np.linalg.norm(q - q_home)))

    # ---- constrained plan --------------------------------------------------
    res = plan_constrained(kin, attached, [pair.path], q_home, goals[0],
                           timeout=args.timeout)
    if not res.ok:
        raise SystemExit(f"[plan_transport] planning failed: {res.reason}")
    print(f"[plan_transport] path: {res.stats['n_waypoints']} waypoints in "
          f"{res.solve_time:.2f}s; max path-TSR excess along path "
          f"{res.max_excess:.4f} (tol 2e-3 at plan time)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, path=res.path, q_home=q_home, q_goal=goals[0],
             T_ee_body=attached.T_ee_body)
    print(f"[plan_transport] saved -> {out}")

    # ---- optional replay ---------------------------------------------------
    if args.view:
        teapot_joint = (env.objects["teapot"].joints[0]
                        if "teapot" in env.objects else None)
        try:
            for q in res.path:
                env.sim.data.qpos[kin.qpos_ids] = q
                if teapot_joint is not None:
                    T_body = attached.body_pose(kin.fk(q))
                    quat = R.from_matrix(T_body[:3, :3]).as_quat(scalar_first=True)
                    env.sim.data.set_joint_qpos(
                        teapot_joint, np.concatenate([T_body[:3, 3], quat]))
                    env.sim.data.set_joint_qvel(teapot_joint, np.zeros(6))
                env.sim.forward()
                for _ in range(3):
                    env.render()
        except (KeyboardInterrupt, Exception):
            pass
        finally:
            env.close()
    else:
        env.close()


if __name__ == "__main__":
    main()