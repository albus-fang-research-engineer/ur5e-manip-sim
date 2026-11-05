# Pour tea, seeds 0-9: findings with the fixed IK tracker

This file has the results with ReKep's IK tracker fixed (commit `85fdbf9`). The two other
files are the record from before the fix (tag `pour-tea-undamped-ik`):
- `FINDINGS.md` counts every collision.
- `FINDINGS_env_only.md` leaves out robot self-contact.

With the fix, no ReKep plan at the published weight has self-contact or arm-teapot contact.
So the Table 1 below holds under either counting rule.

The setup, planners, scenes and metrics are unchanged; see `FINDINGS_env_only.md`, top
section and "Metrics". In short:
- **Planners:** ours (TSR + CBiRRT, at 10.5° and 5° per axis), ReKep's upstream loop at its
  published cost weights, OmniManip, and OmniManip + RRT-Connect.
- **Scenes:** no box; a box on the path (x = 0.03); a box beside it (x = 0.12).
- **Success:** the plan reaches the goal and holds the path limits (tilt at most 15° while
  carrying; the spout inside the mug's 44.1 mm rim while pouring), and no collision is deeper
  than 1 mm.

## 1. Main results

Only ReKep's column differs from the pre-fix tables. Ours and both OmniManip variants don't
use the tracker; their columns are copied unchanged.

### No box

|  | Ours, 10.5° per axis | Ours, 5° per axis | ReKep | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | **10/10** | **10/10** | 1/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 9.9° | 3.9° / 5.2° | 1.1° / 9.3° | 76.8° / 87.9° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 25.8 / 32.3 mm | 25.1 / 32.3 mm | 20.9 / 51.7 mm | 42.9 / 328.0 mm | 42.7 / 42.8 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 9/10 | 0/10 | 0/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 3.1 / 3.2 mm | 0.0 / 0.3 mm | 0.0 / 0.0 mm |

### Box on path (x = 0.03)

|  | Ours, 10.5° per axis | Ours, 5° per axis | ReKep | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | **10/10** | **10/10** | 1/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 10.7° / 14.9° | 6.2° / 7.2° | 1.1° / 14.7° | 75.2° / 83.5° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/9 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 23.8 / 27.4 mm | 25.3 / 32.0 mm | 32.9 / 70.7 mm | 43.2 / 43.9 mm | 42.7 / 42.7 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 8/9 | 0/10 | 10/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 4.1 / 51.6 mm | 0.0 / 0.0 mm | 73.4 / 73.4 mm |

### Box beside (x = 0.12)

|  | Ours, 10.5° per axis | Ours, 5° per axis | ReKep | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | **10/10** | **10/10** | 0/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 12.7° | 4.9° / 6.5° | 2.0° / 10.5° | 76.8° / 87.9° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 21.7 / 32.4 mm | 21.8 / 32.1 mm | 20.5 / 72.3 mm | 43.5 / 259.9 mm | 42.7 / 42.8 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 10/10 | 0/10 | 0/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 3.2 / 39.9 mm | 0.0 / 0.5 mm | 0.0 / 0.0 mm |

ReKep, before the fix → after:

| | Before | After |
|---|---|---|
| No box: spout off axis, max | 245.8 mm | 51.7 mm |
| No box: deepest collision, max, counting self-contact | 18.3 mm | 3.2 mm |
| On path: success | 0/10 | 1/10 |
| On path: tilt while carrying, max | 8.6° | 14.7° |
| On path: path constraint violations | 2/9 | 1/9 |
| On path: collision rate | 9/9 | 8/9 |
| On path: deepest collision, median | 34.9 mm | 4.1 mm |
| On path: deepest collision, max, counting self-contact | 55.7 mm | 51.6 mm |

The maxima counting only the environment (3.2 mm no box, 51.6 mm on path) are unchanged.
The box-beside column is unchanged.

**In short:**
- Ours succeeds in all 60 runs, across both bounds and all three scenes.
- ReKep stays upright but collides. It presses about 3 mm into the mug in almost every plan,
  and on path it hits the box in 5 of 9 plans.
- OmniManip has no notion of carrying upright. Its target pose is already tilted about 71°,
  so it violates the path limits in every plan, whichever mover reaches that pose.

## 2. The fix

ReKep outputs end-effector poses; a tracker turns them into joint configurations. Upstream
sends the poses to an operational-space controller, which stops short of a pose out of reach.

**Before the fix:**
- The tracker ran mink's IK as configured: undamped, keeping the last iterate.
- On a pose at or past the arm's reach, that iterate could swing the elbow through the
  straight-arm singularity and fold the arm: up to 0.78 m from the target at the published
  weight, and 1.06 m in the sweep.
- The planner kept that configuration and moved on, so the arm swept through itself, the
  table or the box.

**With the fix** (`MinkTracker` in `manip_sim/rekep/adapter.py`):
- **On a hit:** a pose the existing solve reaches within tolerance is left alone.
- **On a miss:** the tracker solves again from the current joints, with damping
  (lm_damping 1, damping 1e-3). It keeps the iterate nearest the target, or stays put if none
  is nearer. It logs the miss and moves on, with no retries, like upstream's controller.
- **New data:** plans now save the pose each waypoint was solved for (`targets` in the
  `.npz`).
- **Test:** `test_mink_tracker_stops_short_of_a_pose_out_of_reach`.
  - For a target 5 m away, the old tracker ended 5.19 m from it, farther than where it
    started.
  - For a target 0.85 m to the side and up, the old tracker folded and stopped 810 mm off;
    the fix stops 45 mm off.

**Second rule: no elbow-branch switch in one step** (commit `958420b`):
- **The case:** after the arm stops short at full stretch (elbow at 0 rad), the next pose can
  be back in reach. The undamped solve then *hits* it, but on the other elbow branch: a
  5.89 rad joint step in ×20 no box s6. Joint interpolation sweeps that step through the
  table. The first fix leaves hits alone, and bisection can't split a branch switch.
- **The rule:** a step that still exceeds the 0.35 rad limit after every bisection, where the
  elbow changes sign, is solved again with damping from the current joints, with every joint
  held within 0.35 rad (`MinkTracker.solve_near`). The arm stops short on its own branch.
- **Checked against upstream:** `environment.py`'s `execute_action` interpolates each action
  into 5 cm / 0.34 rad poses. It drives the OSC toward each one for at most 10 control steps
  (40 for the final, precise one), then prints "OSC pose not reached" and moves on. The arm
  moves continuously, so it can stop short but never jump branches.
- **What it leaves alone:** steps on one branch. That includes the two large wrist steps
  (1.41 and 1.04 rad) in ×10 beside s3 and on path s1. They pass continuously near the wrist
  singularity during the pour, with the gripper moving 7 mm per step, and they're the
  gripper-into-forearm contacts the renders showed. They're ReKep's own motion.
- **On the s6 waypoint itself:** the new rule moves at most 0.19 rad, keeps the elbow at
  0 rad and stops 70 mm short of the target, which was 76 mm away.
- **Table 1 is unaffected:** none of its plans has a step over 0.35 rad.
- **Test:** `test_soft_ik_does_not_switch_elbow_branches_in_one_step`.

## 3. Which runs were rerun

The only runs rerun are the 7 at the published weight whose report has a tracking miss.
Every other run is unchanged.

| Scene | Seed | Worst miss, before → after | What changed |
|---|---|---|---|
| No box | s7 | 276 → 57 mm | The forearm-teapot contact (18.3 mm) is gone. The worst spout excursion drops from 246 to 52 mm, but that still leaves the rim (44.1 mm), so it still fails. |
| No box | s8 | 13 → 9 mm | Spout off axis 35.1 → 32.7 mm; still succeeds. |
| On path | s1 | no plan → no plan | It still backtracks to the grasp, which this setup cannot redo. Worst miss before failing: 278 → 429 mm; the arm now stops short instead of folding. |
| On path | s2 | 32 → 29 mm | Same metrics. Still exits the rim (70.7 mm) and hits the box (46.4 mm). |
| On path | s4 | 777 → 26 mm | The fold is gone, and with it the 51.4 mm box contact and the rim exit. It carries at up to 14.7° and succeeds, with no contact at all. |
| On path | s7 | 11 → 10 mm | Same metrics. |
| Beside | s4 | 14 → 9 mm | Same metrics. Still exits the rim (72.3 mm). |

**Reproducibility check.** Five seeds with no misses were replanned with the fix: no box s0,
on path s0, and beside s0, s2 and s7. All five came out bit for bit identical to the
originals: every array in the `.npz`, and every report field except timing. So the 23
unchanged runs stand as they are.

## 4. What the bridge still affects

- **Remaining misses are small.** The worst is 57 mm, under upstream's 10 cm tolerance for
  intermediate waypoints. So upstream's controller would also have moved on from them.
- **Rim exits.** The three remaining exits (no box s7, on path s2, beside s4) are all in plans
  with small misses (9-57 mm). Whether those misses contribute to the exits isn't measured.
- **Self-contact.** At the published weight, no plan has any self-contact or arm-teapot
  contact.

## 5. Sweep

ReKep at collision weight ×10 and ×20 with the fixed tracker (first rule). The runs rerun are
those whose report has a tracking miss: 9 at ×10 and 7 at ×20. Every other run is unchanged.
The reproducibility check (no box s0, at each weight) came out bit for bit identical.

×3 and ×30, and ×20 no box s6 under the elbow rule, are still running. The pre-fix sweep is in
`FINDINGS.md` section 5.

### 5.1 ×10 and ×20, fixed

Collision counts and depths are environment contact only. The last row adds robot
self-contact.

| | ×10 no box | ×10 on path | ×10 beside | ×20 no box* | ×20 on path | ×20 beside |
|---|---|---|---|---|---|---|
| Success rate | 0/10 | 0/10 | 2/10 | 2/10 | 1/10 | 5/10 |
| Plans returned | 10 | 10 | 9 | 9 | 10 | 10 |
| Tilt while carrying, median / max | 1.4° / 7.2° | 4.3° / 31.9° | 3.9° / 27.2° | 7.7° / 23.4° | 23.1° / 34.0° | 13.3° / 60.7° |
| Path constraint violation rate | 2/10 | 4/10 | 4/9 | 7/9 | 9/10 | 5/10 |
| Spout off axis, median / max | 33.9 / 72.0 mm | 26.0 / 244.1 mm | 26.1 / 159.5 mm | 47.6 / 1645.8 mm | 43.0 / 123.7 mm | 32.2 / 76.3 mm |
| Collision rate, deeper than 1 mm | 10/10 | 9/10 | 7/9 | 2/9 | 1/10 | 1/10 |
| Deepest collision, median / max | 3.0 / 12.1 mm | 3.1 / 22.9 mm | 2.9 / 6.7 mm | 0.0 / 140.3 mm | 0.0 / 58.6 mm | 0.0 / 26.8 mm |
| Deepest contact including self-contact, max | 12.1 mm | 38.2 mm | 83.2 mm | 222.5 mm | 96.8 mm | 58.7 mm |

\* ×20 no box s6 has the branch-switch artefact (section 2); it is being rerun under the elbow
rule. Its 140.3 mm, 222.5 mm and 1645.8 mm are from that artefact.

Success counted with self-contact is the same in every cell.

### 5.2 Before the fix → after

| | Before | After |
|---|---|---|
| ×10 on path: tilt max | 33.8° | 31.9° |
| ×10 on path: collision rate | 10/10 | 9/10 |
| ×10 on path: deepest collision, median / max | 3.2 / 146.9 mm | 3.1 / 22.9 mm |
| ×10 on path: including self-contact, max | 222.9 mm | 38.2 mm |
| ×10 beside: deepest collision, max | 42.0 mm | 6.7 mm |
| ×10 beside: including self-contact, max | 211.3 mm | 83.2 mm |
| ×20 no box: deepest collision, max | 190.3 mm | 140.3 mm* |
| ×20 on path: success | 2/10 | 1/10 |
| ×20 on path: path constraint violations | 8/10 | 9/10 |
| ×20 on path: tilt, median / max | 20.0° / 47.7° | 23.1° / 34.0° |
| ×20 on path: deepest collision, max | 128.6 mm | 58.6 mm |
| ×20 beside: tilt max | 56.7° | 60.7° |
| ×20 beside: deepest collision, max | 143.8 mm | 26.8 mm |

×10 no box is unchanged in every metric.

### 5.3 The reruns

| Run | Worst miss, before → after | What changed |
|---|---|---|
| ×10 no box s5 | 11 → 10 mm | Same metrics. |
| ×10 on path s0 | 930 → 115 mm | The 147 mm contact is gone (no contact now), and the spout's worst excursion drops from 1630 to 39 mm. Tilt rises 13.3° → 31.9°, so it still fails the path limits. |
| ×10 on path s3 | 816 → 76 mm | Contact 125 → 3.2 mm; spout excursion 755 → 70 mm. Still misses the goal. |
| ×10 on path s4, s5, s7 | 7-15 → 6-10 mm | Same metrics. |
| ×10 on path s8 | 960 → 251 mm | Contact 132 → 22.9 mm; tilt 33.8° → 14.6°. The spout now swings 244 mm off axis, so it still fails. |
| ×10 beside s1 | no plan → no plan | Still backtracks to the grasp. |
| ×10 beside s5 | 782 → 62 mm | Contact 42 → 3.0 mm, self-contact 211 → 3.0 mm; spout excursion 739 → 55 mm. Still misses the goal. |
| ×20 no box s6 | 314 → 71 mm | Branch-switch artefact (section 2); being rerun. |
| ×20 no box s8 | no plan → no plan | Still backtracks to the grasp. |
| ×20 on path s0 | 184 → 41 mm | The old plan succeeded counting only the environment (it had 59 mm of self-contact). The new one tilts to 34.0° with the spout 122 mm off axis, so it fails. |
| ×20 on path s4, s5 | 11-14 → 6-10 mm | Same metrics. |
| ×20 on path s7 | 855 → 202 mm | Contact 129 → 58.6 mm; tilt 47.7° → 33.3°. Still fails. |
| ×20 beside s2 | 239 → 32 mm | Contact 144 → 26.8 mm; tilt 56.7° → 60.7°. Still fails. |

### 5.4 What the remaining misses are

Seven fixed plans were checked against their saved target poses: ×1 no box s7, ×10 on path s0
and s8, ×20 no box s6, ×20 on path s0 and s7, and ×20 beside s2.

**Six of them miss at the reach limit, not by folding:** ×1 no box s7, ×10 on path s0 and s8,
×20 on path s0 and s7, and ×20 no box s6 before its jump. At every waypoint they miss by more
than 20 mm:
- **The arm can't extend further:** the elbow is within 0.03 rad of 0, fully straight.
- **The target is past the gripper:** targets are 69-109 cm from the shoulder, and the
  gripper is at 61-102 cm. The misses are 2-25 cm.
- **Steps stay small:** every joint step is within 0.35 rad, and the gripper stays within a
  few mm of the straight line between waypoints.

In these plans the poses ReKep asks for are beyond the arm's reach, and the fixed tracker stops
short, as upstream's controller would.

**The other one is different:** ×20 beside s2 misses by 21-32 mm with the elbow bent
(−1.45 rad) and targets 71-76 cm from the shoulder, so it is not at full stretch. The cause of
those misses isn't determined.

The other fixed plans' misses (up to 76 mm) were not checked one by one.

Self-contact without a jump remains in three fixed plans: ×10 on path s8 (38.2 mm), ×20 on
path s7 (96.8 mm) and ×20 beside s2 (58.7 mm). These have not been rendered yet, so whether
it's ReKep's own motion, as in the two ×10 wrist cases, is not verified.

## 6. Caveats

- **Carrying tilt can understate the transport stage.** It stops counting when the spout
  first enters the rim.
- **Some seeds have no ReKep plan.** On path s1 backtracks to the grasp, which this setup
  cannot redo because the grasp is given.
- **The sweep is partly rerun.** ×3 and ×30 are still pre-fix (in `FINDINGS.md`).
- **Saved tables are pre-fix.** The `*.summary.md` tables, the README and
  `FINDINGS.md`/`FINDINGS_env_only.md` still show the pre-fix numbers. This file's ReKep
  column comes from `table1fix.py`, and the sweep from `sweepfix.py`. Both are scratch scripts that merges the fixed runs over the
  originals. It reproduces the pre-fix ReKep column exactly when given the originals.

## 7. Files

All paths are under `/home/albusfang/ur5e-rekep/`.

- **Fixed reruns:** in `outputs/plans/{pour_compare,pour_obstacle,pour_box12}/`, each with
  `.npz`, `.report.json`, `.run.json` and `.eval.json` files:
  - `rekeploopfix_s*` and `rekeploopcol{10,20}fix_s*`: the first rule;
  - `*fix2_s*`: both rules. Here `pour_compare` is no box,
  `pour_obstacle` is on path and `pour_box12` is beside.
- **Originals:** `rekeploop_s*` in the same directories. Table 1 uses `rekeploopfix_s*`
  where it exists, otherwise `rekeploop_s*`.
- **Scoring script:**
  `/tmp/claude-1020/-home-albusfang-ur5e-manip-sim/fe2c49a4-b063-4dd9-9b75-200e861447d3/scratchpad/pourk/table1fix.py`
  and `/tmp/claude-1020/-home-albusfang-ur5e-manip-sim/fe2c49a4-b063-4dd9-9b75-200e861447d3/scratchpad/pourk/sweepfix.py`
  (scratch, not in the repo).
- **Commits:**
  - `393f920`, tag `pour-tea-undamped-ik`: the study before the fix.
  - `85fdbf9`: the fix and the Table 1 reruns.
  - `ea7722e`: the ×10 and ×20 reruns.
  - `958420b`: the elbow-branch rule.
