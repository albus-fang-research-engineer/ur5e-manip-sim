# Pour tea, seeds 0-9: LangFold's constraints solved by ReKep's loop

This file has one extra row for the pour comparison. It solves LangFold's TSRs with ReKep's
upstream loop, the same solver as Table 1's ReKep column. It is kept separate from Table 1
(`FINDINGS_fixed_ik.md`), whose LangFold and ReKep columns are copied here unchanged.

## What is the same and what changes

| | LangFold (Table 1) | LangFold TSRs + ReKep's loop (new) | ReKep (Table 1) |
|---|---|---|---|
| Constraints | LangFold's TSRs | **LangFold's TSRs** | ReKep's pour program |
| Solver | LangFold's planner (CBiRRT) | **ReKep's upstream loop** | ReKep's upstream loop |
| Run names | `ours10g_s*` | `lfrekep_s*` | `rekeploop*_s*`, newest per run as in Table 1 |

- **Same across all three:**
  - scenes and seeds;
  - the start grasp, `q_grasp` from the task spec, so stage 1 isn't compared;
  - the object models and their keypoints;
  - the scorer, `scripts/eval_pour.py`, and the success rule.
- **New row against the ReKep column: only the constraints change.**
  - **Same solver settings:** ReKep's solver runs exactly as in the ReKep column, with the published cost weights (constraint violations ×200), constraint tolerance 1e-4, backtrack tolerance 0.10, and the fixed IK tracker.
  - **Constraints swapped:** stages 2 and 3 use LangFold's constraints instead of the program's; the grasp constraint stays. That also moves the targets:

    | | ReKep's program | LangFold's TSRs |
    |---|---|---|
    | Transport goal | spout exactly 10 cm above the opening | spout in a box over the rim, 4–10 cm above it |
    | Carrying | pot upright, as a cost with no limit | within 10.5° of upright per axis |
    | Pour goal | spout exactly 5 cm above the opening, pour axis within 10° of straight down | spout pinned within ±5 mm where transport ended, tilted 95 ± 5° |
    | Pour path | spout over the opening's center | pivot ±5 mm, off-axis ±3°, tilt −3° to 110° |
- **New row against the LangFold column: only the solver changes.**
  - **Same constraints:** the TSRs are the ones Table 1's LangFold runs use, with the same parameters.
  - **The pour TSR is built the same way:** from the pot's pose where transport actually ended, as LangFold builds it.
  - **Solver differences:**
    - **Soft constraints:** ReKep's solver minimizes a weighted sum. Constraints and collision avoidance are costs, not hard limits.
    - **Planner features not carried over:** three things in LangFold's planner act on the solve rather than on the constraints:
      - it ranks transport goals by whether the pour works from them;
      - it redraws transport goals up to 4 times;
      - it accepts any configuration in the pour goal region.

How the TSRs become ReKep costs (`manip_sim/rekep/langfold.py`):
- **Cost:** each cost is `TSR.distance` at the teapot's pose. That's the norm of the excess outside the TSR's bounds, 0 inside, which matches ReKep's "≤ 0 is satisfied".
- **Pose:** recovered exactly from keypoints 2–5 (Kabsch).
- **Test:** `tests/test_rekep.py` checks that the cost equals `TSR.distance` on sampled poses inside and outside each TSR.
- **Units:** the distance mixes metres and radians, as LangFold's projection does. ReKep's own program mixes them too.

## Results

Success is Table 1's rule:
- the plan reaches the goal (spout over the rim and within 15° of straight down);
- it holds the path limits (at most 15° of tilt while carrying; spout inside the 44.1 mm rim while pouring);
- no contact with the mug, box or table is deeper than 1 mm.

### Success

| Constraints | Solver | No box | Box on path | Box beside | Total |
|---|---|---|---|---|---|
| ReKep's program | ReKep's loop | 1/10 | 1/10 | 0/10 | 2/30 |
| **LangFold's TSRs** | **ReKep's loop** | **6/10** | **2/10** | **4/10** | **12/30** |
| LangFold's TSRs | LangFold's planner | 10/10 | 10/10 | 10/10 | 30/30 |

### No box

|  | LangFold | LangFold TSRs + ReKep's loop | ReKep |
|---|---|---|---|
| Success rate | **10/10** | 6/10 | 1/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 9.9° | 8.7° / 14.7° | 1.1° / 9.3° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 25.8 / 32.3 mm | 37.3 / 43.1 mm | 20.9 / 51.7 mm |
| Spout tip movement while pouring, median / max | 4.6 / 6.9 mm | 18.8 / 20.9 mm | 32.0 / 105.4 mm |
| Goal failure rate | 0/10 | 0/10 | 1/10 |
| Spout off straight down at the end, median / max (limit 15°) | 10.6° / 11.3° | 8.2° / 11.1° | 9.1° / 10.2° |
| Spout height above the opening at the end, median / max | 89.8 / 95.7 mm | 95.3 / 102.7 mm | 50.0 / 55.7 mm |
| Collision rate, deeper than 1 mm | 0/10 | 4/10 | 9/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 10.8 mm | 3.1 / 3.2 mm |

### Box on path (x = 0.03)

ReKep made 9 plans here; one seed found no plan.

|  | LangFold | LangFold TSRs + ReKep's loop | ReKep |
|---|---|---|---|
| Success rate | **10/10** | 2/10 | 1/10 |
| Tilt while carrying, median / max (limit 15°) | 10.7° / 14.9° | 11.6° / 14.0° | 1.1° / 14.7° |
| Path constraint violation rate | 0/10 | 4/10 | 1/9 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 23.8 / 27.4 mm | 37.7 / 55.5 mm | 32.9 / 70.7 mm |
| Spout tip movement while pouring, median / max | 4.9 / 7.6 mm | 17.9 / 36.1 mm | 31.1 / 69.7 mm |
| Goal failure rate | 0/10 | 2/10 | 0/9 |
| Spout off straight down at the end, median / max (limit 15°) | 8.7° / 13.5° | 11.1° / 34.3° | 10.0° / 10.1° |
| Spout height above the opening at the end, median / max | 62.0 / 100.6 mm | 88.3 / 103.4 mm | 50.0 / 50.4 mm |
| Collision rate, deeper than 1 mm | 0/10 | 6/10 | 8/9 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 2.3 / 37.9 mm | 4.1 / 51.6 mm |

### Box beside (x = 0.12)

|  | LangFold | LangFold TSRs + ReKep's loop | ReKep |
|---|---|---|---|
| Success rate | **10/10** | 4/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 12.7° | 8.3° / 11.5° | 2.0° / 10.5° |
| Path constraint violation rate | 0/10 | 5/10 | 1/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 21.7 / 32.4 mm | 44.1 / 50.6 mm | 20.5 / 72.3 mm |
| Spout tip movement while pouring, median / max | 4.0 / 6.2 mm | 17.5 / 25.2 mm | 28.7 / 128.9 mm |
| Goal failure rate | 0/10 | 1/10 | 0/10 |
| Spout off straight down at the end, median / max (limit 15°) | 9.8° / 13.5° | 8.6° / 33.8° | 9.9° / 11.3° |
| Spout height above the opening at the end, median / max | 89.2 / 100.0 mm | 95.4 / 97.4 mm | 50.0 / 50.2 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 10/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.0 mm | 3.2 / 39.9 mm |

### Failures of the new row, per seed

| Scene | Seed | Failed | Detail |
|---|---|---|---|
| No box | 2, 3, 4, 6 | collision | pot into mug, 5.6, 5.3, 5.7 and 10.8 mm |
| Box on path | 0, 2 | path | spout leaves the rim while pouring |
| Box on path | 4, 9 | path, collision | spout leaves the rim; pot into box, 1.9 and 2.8 mm |
| Box on path | 1 | goal, collision | over-tilts to 111°, spout 21° off straight down at the end; pot into mug, 7.5 mm |
| Box on path | 7 | goal, collision | tilts to only 56°, spout 34° off straight down; pot into mug, 7.5 mm |
| Box on path | 3 | collision | pot into mug, 7.5 mm |
| Box on path | 5 | collision | pot into box, 37.9 mm |
| Box beside | 0, 1, 3, 5, 7 | path | spout leaves the rim while pouring |
| Box beside | 6 | goal | tilts to only 57°, spout 34° off straight down |

All 30 runs produced a plan, with no backtracking. Each took 334–628 s, with up to 24 runs
in parallel.

## Reading

- **The constraints account for part of ReKep's gap: 2/30 → 12/30, with the solver held fixed.**
  - **The transport goal stops colliding:** ReKep's program targets a point 10 cm above the opening, where the upright pot is 3 mm inside the mug. LangFold's region contains collision-free poses, and ReKep's solver finds them. The no-box contact rate falls from 9/10 to 4/10, and the box-beside rate from 10/10 to 0/10.
- **The solver accounts for the rest: 12/30 → 30/30, with the constraints held fixed.**
  - **Bounds are given up as costs:** ReKep's solver trades the TSRs' bounds against its other costs, and the failures are exactly those bounds giving way.
  - **The pivot:** the TSR allows ±5 mm. The spout moves 9–36 mm during the pour here, against 2–8 mm under LangFold's planner. That leaves 9 seeds' spouts outside the rim.
  - **Contact:** collision is a cost, so 10 seeds touch the mug or box, with contact up to 38 mm deep.
  - **Tilt:** 2 seeds tilt only partway (56–57°) and 1 over-tilts (111°).
- **Carrying tilt is not where it fails:** every plan of the new row stays under 15° while carrying, although 10 of the 30 go past the 10.5° the TSR allows (up to 14.7°).

## Caveats

- **Units:** the cost mixes metres and radians. A different weight between them would change how ReKep's solver trades position against angle.
- **Planner features left out:** the three listed above are part of LangFold's planner and aren't in this row. The 12 → 30 step therefore includes them, not just hard against soft constraints.
- **Contact timing:** I haven't located when in the plan the no-box mug contacts happen.
- **Run times:** they're wall-clock times under different machine loads, so they aren't comparable across rows.

## Reproducing

```
scripts/plan_rekep.py --seed $s --upstream-loop --langfold-constraints \
    --program manip_sim/rekep/programs/pour_tea --scene ... --spec ... --out outputs/plans/<scene>/lfrekep_s$s.npz
scripts/eval_pour.py --scene ... --spec ... --plan outputs/plans/<scene>/lfrekep_s*.npz
```

Scenes and specs are as in Table 1:
- no box: `scenes/pour_tea.json` and `tasks/specs/pour_tea.json`;
- box on path and box beside: `outputs/plans/{pour_obstacle,pour_box12}/scene.{scene,spec}.json`.

Plans, reports and scores are in `outputs/plans/{pour_compare,pour_obstacle,pour_box12}/lfrekep_s{0..9}.*`.
