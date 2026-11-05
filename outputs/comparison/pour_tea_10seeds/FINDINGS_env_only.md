# Pour tea, seeds 0-9: findings, environment collisions only

This version counts collisions with the mug, box and table only. `FINDINGS.md` also counts
robot self-contact and arm-teapot contact.

Four planners on the same pour-tea task in the MuJoCo/robosuite UR5e sim. Every planner gets
privileged object poses, plans kinematics only and uses the same grasp. The task is to carry
a teapot upright to a mug and pour into it.

- **Ours.** TSR + CBiRRT with hand-authored TSRs and a goal region. It runs at two per-axis
  tilt bounds:
  - 10.5° per axis, which allows up to 14.8° of total tilt;
  - 5° per axis, which allows up to 7.1°.

  The total comes from cos(tilt) = cos(r)·cos(p). Both settings use a 60 s timeout and a
  0.03 rad collision-check step.
- **ReKep.** Upstream's own loop (`plan_rekep.py --upstream-loop`) with upstream's published
  cost weights, unless a section says otherwise.
- **OmniManip.** Its target pose, reached by a straight joint-space line.
- **OmniManip + RRT-Connect.** The same target pose, reached by our CBiRRT with every axis
  free (`plan_omnimanip.py --mover rrt`). The held pot is in the collision check, and the
  timeout and check step match ours. This variant separates what the representation does
  from what the straight-line move does.

Three scenes, 10 seeds each:

| Scene | Box |
|---|---|
| No box | none |
| On path | box at x = 0.03, near the line between the pot and the mug |
| Beside | box at x = 0.12, off that line |

## Metrics

| Metric | Definition |
|---|---|
| Success | The plan reaches the goal, holds the path limits and has no collision deeper than 1 mm. It is counted out of all 10 runs, so a run that returns no plan is a failure. |
| Path constraint violation | Tilt over 15° while carrying, or the spout outside the mug's rim (44.1 mm from its axis) while pouring. Carrying lasts until the spout first enters the rim column. |
| Per-plan path score | The maximum, over samples, of min(tilt/15°, off-axis/44.1 mm). A score above 1 means the plan violated the path limits. It matches the violation count on all 150 graded plans, so the counts do not depend on where carrying ends. |
| Collision | Contact with the environment: the robot or the held pot penetrating the mug, the box or the table by more than 1 mm. Robot self-contact (including the pedestal) and arm-teapot contact are not counted. The reasons are in section 3. |
| Medians and maxima | Taken across plans, of each plan's worst value. Rows other than success are over the plans that were returned. |

## 1. Main results

### No box

|  | Ours, 10.5° per axis | Ours, 5° per axis | ReKep | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | **10/10** | **10/10** | 1/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 9.9° | 3.9° / 5.2° | 1.1° / 9.3° | 76.8° / 87.9° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 25.8 / 32.3 mm | 25.1 / 32.3 mm | 20.9 / 245.8 mm | 42.9 / 328.0 mm | 42.7 / 42.8 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 9/10 | 0/10 | 0/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 3.1 / 3.2 mm | 0.0 / 0.3 mm | 0.0 / 0.0 mm |

### Box on path (x = 0.03)

|  | Ours, 10.5° per axis | Ours, 5° per axis | ReKep | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | **10/10** | **10/10** | 0/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 10.7° / 14.9° | 6.2° / 7.2° | 1.1° / 8.6° | 75.2° / 83.5° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 2/9 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 23.8 / 27.4 mm | 25.3 / 32.0 mm | 32.9 / 70.7 mm | 43.2 / 43.9 mm | 42.7 / 42.7 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 9/9 | 0/10 | 10/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 34.9 / 51.6 mm | 0.0 / 0.0 mm | 73.4 / 73.4 mm |

### Box beside (x = 0.12)

|  | Ours, 10.5° per axis | Ours, 5° per axis | ReKep | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | **10/10** | **10/10** | 0/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 12.7° | 4.9° / 6.5° | 2.0° / 10.5° | 76.8° / 87.9° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 21.7 / 32.4 mm | 21.8 / 32.1 mm | 20.5 / 72.3 mm | 43.5 / 259.9 mm | 42.7 / 42.8 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 10/10 | 0/10 | 0/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 3.2 / 39.9 mm | 0.0 / 0.5 mm | 0.0 / 0.0 mm |

In these tables, counting only environment contact changes two ReKep maxima and nothing
else. They were 18.3 mm (no box) and 55.7 mm (on path), both from IK-miss plans hitting the
robot's own links; they are now 3.2 mm and 51.6 mm. ReKep never touches the table at the
published weight.

These tables come from `scripts/eval_pour.py`, whose saved copies (`*.summary.md`) and the
README still count every collision category. They differ from this file only in those two
maxima.

**In short:**
- Ours succeeds in all 60 runs, across both bounds and all three scenes.
- ReKep stays upright but collides: it touches the mug in almost every plan and the box when
  the box is near the path.
- OmniManip has no notion of carrying upright. Its target pose is already tilted about 71°,
  so it violates the path limits in every plan, whichever mover reaches that pose.

## 2. OmniManip: straight line versus RRT-Connect

- **Path violations are the same with either mover.** Both fail 10/10 in every scene,
  because the violation comes from the target pose and not from the motion.
- **The straight line is deterministic.** It reaches the same target pose from every seed.
  The joint paths differ by up to 0.19 rad, but every metric is identical. That is why its
  median equals its max (71.2°).
- **RRT-Connect's paths vary.** It tilts more on the way (median about 76°, up to 88°), and
  in some seeds the spout swings outside the rim before it settles (328 mm with no box,
  260 mm beside).
- **RRT-Connect avoids the box; the line does not.** On path, the straight line drives
  through the box in all 10 plans, 73.4 mm deep. RRT-Connect has no contact deeper than
  1 mm in any scene.
- **Why RRT-Connect's deepest contact is above zero.** The 0.3 mm (mug, no box, seed 4) and
  0.5 mm (box, beside, seed 0) are single-sample grazes between configurations 0.03 rad
  apart, the same check step ours uses. Both are below the 1 mm tolerance.

## 3. Where ReKep's collisions come from

### Contact with the mug

Almost every ReKep plan presses about 3 mm into the mug, and that alone fails the 1 mm rule.

- **Where:** at the transport stage's subgoal, which puts the spout 10 cm above the opening
  with the pot upright (0.3 mm off the axis, 0.1° of tilt). The contact lasts into the start
  of the pour. It is not on the path between poses.
- **Why the cost allows it:** ReKep's subgoal solver does score collision there, but the cost
  is soft. It sums (sdf + 0.10 m) over the collision points that are within 10 cm.

ReKep's success if mug contact up to a threshold were allowed, with every other contact
still held to 1 mm:

| Mug contact allowed | No box | Box on path | Box beside |
|---|---|---|---|
| 1 mm (as published) | 1/10 | 0/10 | 0/10 |
| 3 mm | 4/10 | 2/10 | 2/10 |
| 5 mm | 9/10 | 3/10 | 6/10 |

### Contact with the box

The box is hit in 6 of 9 plans on path and 3 of 10 beside. Beside, the contact is with the
gripper and the held pot, not the arm:

| Seed (beside) | What hits the box | Depth |
|---|---|---|
| s6 | left inner finger; knuckles; teapot | 39.9 mm; 28-30 mm; 14.7 mm |
| s8 | finger | 29.2 mm |
| s9 | teapot | 7.0 mm |

What upstream ReKep can see:
- **Points:** only the gripper, the wrist and the held object
  (`environment.get_collision_points`). The forearm and upper arm are not modelled, and
  neither is self-collision.
- **Thresholds:** collision costs (sdf + threshold) > 0, with a threshold of 0.10 m in the
  subgoal solver and 0.20 m in the path solver.
- **Start and goal:** the path solver skips collision at both (`start_idx, end_idx = 1,
  num_poses-1`).

Our port keeps the same collision points (`gripper0`, `robot0_wrist`).

### The IK bridge, and why self-contact is not counted

ReKep outputs end-effector poses. Upstream sends them to an operational-space controller;
here mink IK (v0.0.5, default settings) turns them into joint configurations. In some plans
mink misses ReKep's waypoints, by 7 mm up to 1.06 m. Section 5 marks these plans as "IK miss".

What a large miss looks like, from the saved paths:
1. The arm is nearly fully stretched (92-97 cm from shoulder to gripper).
2. On the next waypoint, the elbow swings through the straight-arm singularity to its joint
   limit (+2.98 rad), folding the arm in half, 0.7 m from the target.
3. One waypoint later the arm unfolds.
4. The planner keeps the missed configuration and interpolates joints linearly through
   it, so the arm sweeps through itself, the pedestal and the table.

What causes it:
- **ReKep** asks for poses at or just beyond the arm's reach.
- **Our bridge** does the rest:
  - mink is undamped by default;
  - the planner keeps a failed IK result instead of holding the closest reachable pose,
    as a controller would.

These motions are artefacts of the bridge, not ReKep's behaviour, so this file does not
count robot self-contact. ReKep's own self-contact is real but not measured here: its cost
models only the gripper, wrist and held object, so it can fold the gripper into the forearm.
The renders show this in 2 IK-ok plans at ×10, with 83 mm and 16 mm of contact.

The bridge still affects what is reported:
- **Rim exits:** every published-weight rim exit is in an IK-miss plan, so ReKep's path
  violations at ×1 (1/10, 2/9, 1/10) come from the bridge.
- **Table contact:** every table hit in the sweep is in an IK-miss plan.
- **Box contact:** some box contact on path is in IK-miss plans.

ReKep's target poses were not saved, so whether each target was just out of reach could not
be checked directly.

## 4. Re-scoring at a stricter tilt limit (7.5°)

Path violations when the tilt limit is 7.5° instead of 15°, with the rim unchanged:

| | No box | Box on path | Box beside |
|---|---|---|---|
| Ours, 10.5° per axis | 9/10 | 8/10 | 7/10 |
| Ours, 5° per axis | 0/10 | 0/10 | 0/10 |
| ReKep | 2/10 | 3/9 | 2/10 |
| OmniManip | 10/10 | 10/10 | 10/10 |

Ours violates the stricter limit only when it is planned against the looser one. Planned at
5° per axis, it holds 7.5° everywhere. Re-scored at a 5° limit, the 5° plans give 1/10, 7/10
and 5/10, because 5° per axis compounds to up to 7.1° of total tilt.

## 5. ReKep collision-weight sweep (×1, ×3, ×10, ×20, ×30)

This asks whether ReKep can avoid all contact by trading away tilt. Every collision weight
in upstream's loop is scaled by k (`--collision-scale k`, which multiplies the 0.8 subgoal
and 0.5 path weights). The upright cost is left as published.

- **Pre-registered levels:** ×1, ×3, ×10 and ×30.
- **×20 was added after the others had run.** If it appears in the paper, say so.
- **Small samples:** each cell is 10 seeds or fewer, so treat single-plan differences as
  noise.
- **"IK-ok"** means mink tracked every ReKep waypoint (see section 3).
- Plans per level are given in the IK-miss table (section 5.6).

### 5.1 Success rate (out of 10 runs)

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 (published) | 1/10 | 0/10 | 0/10 |
| ×3 | 0/10 | 1/10 | 1/10 |
| ×10 | 0/10 | 2/10 | 0/10 |
| ×20 (added later) | 2/10 | **5/10** | 2/10 |
| ×30 | 2/10 | 1/10 | 0/10 |

Ours is 10/10 in every scene for comparison. ReKep's best is 5/10, beside at ×20. No weight
does well in all three scenes. Success counts environment contact only; counting self-contact
too would drop ×20 on path to 1/10.

### 5.2 Tilt while carrying, median / max (all plans; IK-ok plans in brackets)

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 1.1° / 9.3° (1.1° / 9.3°) | 2.0° / 10.5° (1.6° / 10.5°) | 1.1° / 8.6° (1.1° / 8.6°) |
| ×3 | 1.1° / 1.8° (1.1° / 1.8°) | 1.3° / 11.1° (1.2° / 4.4°) | 2.3° / 13.0° (3.4° / 13.0°) |
| ×10 | 1.4° / 7.2° (1.4° / 7.2°) | 3.9° / 27.2° (4.1° / 27.2°) | 4.3° / 33.8° (3.6° / 9.2°) |
| ×20 | 7.7° / 23.4° (7.0° / 23.4°) | 13.3° / 56.7° (12.5° / 40.7°) | 20.0° / 47.7° (17.9° / 27.5°) |
| ×30 | 16.1° / 37.3° (15.5° / 23.0°) | 20.9° / 71.6° (19.8° / 54.5°) | 37.2° / 175.2° (42.6° / 83.0°) |

The 175.2° is an IK-miss artefact.

Plans tilting past 15° while carrying:

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 0/10 | 0/10 | 0/9 |
| ×3 | 0/10 | 0/8 | 0/10 |
| ×10 | 0/10 | 2/9 | 1/10 |
| ×20 | 2/9 | 3/10 | 8/10 |
| ×30 | 7/10 | 7/8 | 7/9 |

Each ×20 plan in seed order (* marks an IK miss):

| Scene | Tilt per plan |
|---|---|
| No box | 8.7, 10.2, 16.1, 5.4, 3.6, 2.7, 7.7*, 3.6, 23.4 (s8: no plan) |
| Box beside | 14.8, 5.5, 56.7*, 40.7, 8.4, 4.6, 8.3, 18.0, 14.2, 12.5 |
| Box on path | 12.6*, 16.1, 20.6, 27.5, 32.5*, 25.7*, 13.0, 47.7*, 19.3, 16.4 |

### 5.3 Path constraint violations (all plans; IK-ok plans in brackets)

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 1/10 (0/8) | 1/10 (0/9) | 2/9 (0/6) |
| ×3 | 1/10 (1/10) | 2/8 (2/7) | 1/10 (0/5) |
| ×10 | 2/10 (1/9) | 4/9 (3/8) | 4/10 (1/4) |
| ×20 | 7/9 (6/8) | 5/10 (4/9) | 8/10 (5/6) |
| ×30 | 8/10 (6/8) | 7/8 (5/6) | 8/9 (5/5) |

At ×1, every violation is in an IK-miss plan. From ×20 up, most violations are in plans where
IK tracked, so ReKep's own solution breaks the limits. The causes:
- **Spout outside the rim while pouring**, among IK-ok plans:

  | Collision weight | No box | Box beside | Box on path |
  |---|---|---|---|
  | ×1 | 0 | 0 | 0 |
  | ×10 | 1 | 1 | 1 |
  | ×20 | 4 | 2 | 3 |

- **Tilt over 15° while carrying**: the rest.

### 5.4 Contact with the box or mug

Deepest environment contact: the deepest of mug, box and table per plan. Each cell is
median / max in mm, then plans deeper than 1 mm, for all plans with IK-ok plans in brackets:

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 3.1 / 3.2, 9/10 (3.1 / 3.2, 8/8) | 3.2 / 39.9, 10/10 (3.2 / 39.9, 9/9) | 34.9 / 51.6, 9/9 (3.6 / 51.6, 6/6) |
| ×3 | 3.2 / 4.3, 10/10 (3.2 / 4.3, 10/10) | 3.1 / 11.1, 7/8 (3.1 / 11.1, 6/7) | 3.1 / 131.5, 9/10 (3.1 / 11.3, 4/5) |
| ×10 | 3.0 / 12.1, 10/10 (3.1 / 12.1, 9/9) | 2.9 / 42.0, 7/9 (2.8 / 6.7, 6/8) | 3.2 / 146.9, 10/10 (3.0 / 3.2, 4/4) |
| ×20 | 0.0 / 190.3, 2/9 (0.0 / 4.0, 1/8) | 0.0 / 143.8, 1/10 (0.0 / 0.0, 0/9) | 0.0 / 128.6, 1/10 (0.0 / 0.0, 0/6) |
| ×30 | 0.0 / 139.5, 1/10 (0.0 / 0.0, 0/8) | 0.0 / 156.1, 2/8 (0.0 / 0.0, 0/6) | 0.0 / 250.1, 4/9 (0.0 / 0.0, 0/5) |

Every maximum above 100 mm is an IK-miss plan hitting the table (section 3).

- **Up to ×10:** the median sits near 3 mm. That is the mug press at the transport subgoal.
- **On path at ×1:** the median over all plans is 34.9 mm, mostly box contact. In IK-ok
  plans the median drops to 3.6 mm, but the max is still 51.6 mm, so ReKep's own solution
  hits the box too.
- **×20:** four plans go deeper than 1 mm:
  - No box, seed 1: 4.0 mm on the mug, in an IK-ok plan.
  - On path, seed 7: 45.8 mm on the mug and 128.6 mm into the table, in an IK-miss plan.
  - No box and beside: one table hit each, in IK-miss plans.
- **×20 and ×30:** apart from that 4.0 mm, there is no environment contact in any IK-ok plan.
  Every other hit is in an IK-miss plan.

The two tables below split this into box and mug. Table contact only appears in IK-miss
plans:

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 0 | 0 | 0 |
| ×3 | 0 | 0 | 1 |
| ×10 | 0 | 1 | 3 |
| ×20 | 1 | 1 | 1 |
| ×30 | 1 | 1 | 4 |

Box contact (all plans; IK-ok plans in brackets):

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | – | 3/10 (3/9) | 6/9 (3/6) |
| ×3 | – | 0/8 | 2/10 (1/5) |
| ×10 | – | 0/9 | 1/10 (0/4) |
| ×20 | – | **0/10** | **0/10** |
| ×30 | – | 1/8 (0/6) | 3/9 (0/5) |

Mug contact, IK-ok plans, with the deepest in mm:

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 8/8, 3.2 | 9/9, 4.5 | 5/6, 3.1 |
| ×3 | 10/10, 4.3 | 6/7, 11.1 | 4/5, 4.3 |
| ×10 | 9/9, 12.1 | 6/8, 6.7 | 4/4, 3.2 |
| ×20 | 1/8, 4.0 | **0/9** | **0/6** |
| ×30 | 0/8 | 0/6 | 0/5 |

### 5.5 Goal accuracy, IK-ok plans: spout error, median / max in mm

Transport subgoal (10 cm above the opening):

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 0.3 / 1.3 | 0.3 / 3.4 | 0.3 / 67.7 |
| ×3 | 0.3 / 0.3 | 0.3 / 11.3 | 0.3 / 8.2 |
| ×10 | 0.3 / 0.3 | 0.4 / 2.6 | 0.3 / 0.3 |
| ×20 | 0.4 / 4.0 | 0.3 / 7.2 | 2.6 / 24.7 |
| ×30 | 14.8 / 22.7 | 9.4 / 15.6 | 2.0 / 5.7 |

Final pose (5 cm above the opening):

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 0.1 / 6.0 | 0.1 / 0.2 | 0.2 / 0.2 |
| ×3 | 0.1 / 3.3 | 0.1 / 0.8 | 0.1 / 1.1 |
| ×10 | 0.1 / 8.9 | 0.1 / 0.2 | 0.1 / 1.5 |
| ×20 | 0.1 / 1.5 | 0.1 / 22.8 | 0.9 / 4.0 |
| ×30 | 0.1 / 0.5 | 1.2 / 8.2 | 0.7 / 28.1 |

At ×30 the transport subgoal moves 9-15 mm off target in the scenes without a box on the
path. Up to ×20 it stays close.

### 5.6 Plans returned and IK misses

| Collision weight | No box | Box beside | Box on path |
|---|---|---|---|
| ×1 | 10 plans, 2 misses | 10, 1 | 9, 3 |
| ×3 | 10, 0 | 8, 1 | 10, 5 |
| ×10 | 10, 1 | 9, 1 | 10, 6 |
| ×20 | 9, 1 | 10, 1 | 10, 4 |
| ×30 | 10, 2 | 8, 2 | 9, 4 |

**Runs with no plan:**
- **Backtracked to the grasp** (8 runs). Upstream would redo the grasp, but here the grasp is
  given, so the run ends:
  - ×1 on path: s1
  - ×3 beside: s2, s5
  - ×10 beside: s1
  - ×20 no box: s8
  - ×30 beside: s8
  - ×30 on path: s5
- **Killed at the 2400 s limit** (1 run): ×30 beside, s9.

IK misses grow with the weight when the box is on the path (3, 5, 6, 4, 4), so a higher weight
pushes ReKep's poses toward ones mink tracks badly.

### 5.7 What the sweep shows

| Collision weight | What ReKep still does wrong |
|---|---|
| ×1 (published) and ×3 | Upright, but touches the mug in almost every plan, and the box when it is near the path |
| ×10 | Clears the box in IK-ok plans, but mug contact grows to 12 mm |
| ×20 | Clears both in IK-ok plans (one 4 mm mug contact), but tilts past 15° in 8 of 10 plans on path, and the spout leaves the rim in 2-4 IK-ok plans per scene |
| ×30 | No contact in IK-ok plans, but tilts more (median 16-37°), and the transport subgoal moves 9-15 mm off target |

ReKep's soft costs trade collision against tilt and goal accuracy. Raising the collision
weight clears the contact by giving up the upright constraint. The best weight reaches 5/10
beside and only 1-2/10 in the other two scenes. Ours meets all three limits in every scene
and seed without tuning.

## 6. Caveats

- **Carrying tilt can understate the transport stage.** It stops counting when the spout
  first enters the rim. Seed 4 on path at ×1 reports 1.1° but reaches 32.5° during transport.
  That plan is one of the rim exits, so the violation count is unaffected.
- **The IK bridge distorts ReKep's results** (section 3). Self-contact is excluded for that
  reason. But ReKep's ×1 path violations, all its table contact and part of its on-path box
  contact still come from IK-miss plans. Without a fixed bridge and a rerun, those numbers
  belong to ReKep plus our bridge, not to ReKep alone.
- **×20 was not pre-registered.**
- **Some seeds have no ReKep plan.** In each, ReKep backtracked to the grasp, which this setup
  cannot redo because the grasp is given (×1 on path, seed 1; the full list is in section 5.6).

## 7. Files

All paths are under `/home/albusfang/ur5e-rekep/`. Nothing here is committed yet.

- **Main tables:** `outputs/comparison/pour_tea_10seeds/{no_box,box_x003,box_x012}`:
  - `.summary.md`: short table
  - `.md`: constraint table
  - `.full.md`: every metric
  - `.json`: per-plan grades
- **Plans:** `outputs/plans/{pour_compare,pour_obstacle,pour_box12}/`. Each plan has
  `.npz`, `.report.json`, `.run.json` and `.eval.json` files:
  - methods: `ours10g_s*`, `ours5g_s*`, `rekeploop_s*`, `omni_s*`, `omnirrt_s*`
  - sweep: `rekeploopcol{3,10,20,30}_s*`
- **Not committed:** the sweep numbers in section 5 were computed by a scratch script and are
  not yet in the saved tables or the README.

Reproduce one scene's main table:

```
python scripts/eval_pour.py --table outputs/plans/pour_compare --spec tasks/specs/pour_tea.json \
  --methods ours10g ours5g rekeploop omnirrt omni --out outputs/comparison/pour_tea_10seeds/no_box
```
