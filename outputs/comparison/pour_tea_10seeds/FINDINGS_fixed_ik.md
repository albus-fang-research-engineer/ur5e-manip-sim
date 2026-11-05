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

The sweep in `FINDINGS.md` section 5 is still from before the fix. The ×10 and ×20 runs with
tracking misses are being rerun (9 at ×10, 7 at ×20, plus one reproducibility check per
weight); their results will be added here.

## 6. Caveats

- **Carrying tilt can understate the transport stage.** It stops counting when the spout
  first enters the rim.
- **Some seeds have no ReKep plan.** On path s1 backtracks to the grasp, which this setup
  cannot redo because the grasp is given.
- **Saved tables are pre-fix.** The `*.summary.md` tables, the README and
  `FINDINGS.md`/`FINDINGS_env_only.md` still show the pre-fix numbers. This file's ReKep
  column comes from `table1fix.py`, a scratch script that merges the fixed runs over the
  originals. It reproduces the pre-fix ReKep column exactly when given the originals.

## 7. Files

All paths are under `/home/albusfang/ur5e-rekep/`.

- **Fixed reruns:** `outputs/plans/{pour_compare,pour_obstacle,pour_box12}/rekeploopfix_s*`
  (`.npz`, `.report.json`, `.run.json`, `.eval.json`). Here `pour_compare` is no box,
  `pour_obstacle` is on path and `pour_box12` is beside.
- **Originals:** `rekeploop_s*` in the same directories. Table 1 uses `rekeploopfix_s*`
  where it exists, otherwise `rekeploop_s*`.
- **Scoring script:**
  `/tmp/claude-1020/-home-albusfang-ur5e-manip-sim/fe2c49a4-b063-4dd9-9b75-200e861447d3/scratchpad/pourk/table1fix.py`
  (scratch, not in the repo).
- **Commits:**
  - `393f920`, tag `pour-tea-undamped-ik`: the study before the fix.
  - `85fdbf9`: the fix and the Table 1 reruns.
