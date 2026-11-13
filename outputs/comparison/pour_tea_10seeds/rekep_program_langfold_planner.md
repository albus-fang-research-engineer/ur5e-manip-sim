# Pour tea, seeds 0-9: ReKep's constraints solved by LangFold's planner

This file adds one row to the pour comparison: ReKep's pour program, unchanged, planned by
LangFold's planner (CBiRRT). It is the mirror of `langfold_tsrs_rekep_solver.md`, and it is
kept separate from Table 1 (`FINDINGS_fixed_ik.md`).

## What is the same and what changes

| | ReKep (Table 1) | ReKep's program + LangFold's planner (new) | LangFold (Table 1) |
|---|---|---|---|
| Constraints | ReKep's pour program | **ReKep's pour program** | LangFold's TSRs |
| Solver | ReKep's upstream loop | **LangFold's planner (CBiRRT)** | LangFold's planner (CBiRRT) |
| Run names | `rekeploop*_s*`, newest per run as in Table 1 | `rekepcb_s*` | `ours10g_s*` |

- **Same across all three:**
  - scenes and seeds;
  - the start grasp, `q_grasp` from the task spec;
  - the object models and their keypoints;
  - the scorer, `scripts/eval_pour.py`, and the success rule.
- **New row against the ReKep column: only the solver changes.**
  - **The constraints are ReKep's program:** `manip_sim/rekep/programs/pour_tea`, the same one the ReKep column runs.
  - **How they're written as TSRs:** each stage-2 and stage-3 constraint is a zero-width TSR on the teapot body, in `manip_sim/rekep/as_tsr.py`.
  - **Test:** `tests/test_rekep.py` checks that each TSR contains a pose exactly where the program's cost is 0.

    | ReKep's program | As a TSR |
    |---|---|
    | Transport goal: spout exactly 10 cm above the opening | spout pinned there, yaw free |
    | Carrying: pot upright | roll = pitch = 0; position and yaw free |
    | Pour goal: spout exactly 5 cm above the opening, pour axis within 10° of straight down | spout pinned there, pour axis in a 10° cone |
    | Pour path: spout over the opening's center | spout x = y = 0; height and rotation free |

  - **No tolerances added:** the program's constraints have none, apart from the 10° cone. Only the planner's own numerical tolerances apply, as they do to LangFold's TSRs: projection 2e-3 and goal containment 5e-3.
- **New row against the LangFold column: only the constraints change.**
  - **Planner settings as in Table 1:** `--eps 0.03 --timeout 60 --pour-goal-region --entry-rounds 4`.
  - **The 10.5° upright limit doesn't apply:** `--upright-deg 10.5` is passed but not used, because the program's carrying constraint replaces it.

## Results

Success is Table 1's rule:
- the plan reaches the goal;
- it holds the path limits;
- no contact with the mug, box or table is deeper than 1 mm.

### Success

| Constraints | Solver | No box | Box on path | Box beside | Total |
|---|---|---|---|---|---|
| ReKep's program | ReKep's loop | 1/10 | 1/10 | 0/10 | 2/30 |
| **ReKep's program** | **LangFold's planner** | **0/10** | **0/10** | **0/10** | **0/30** |
| LangFold's TSRs | ReKep's loop | 6/10 | 2/10 | 4/10 | 12/30 |
| LangFold's TSRs | LangFold's planner | 10/10 | 10/10 | 10/10 | 30/30 |

The new row is in bold. The ReKep's-loop rows and the LangFold row come from `langfold_tsrs_rekep_solver.md` and Table 1, unchanged.

### How the new row fails

None of the 30 runs found a plan. Each one stops at the transport goal with
`[transport] no feasible goal configs.` The planner never reaches the pour stage.

| | No box | Box on path | Box beside |
|---|---|---|---|
| Transport goals sampled, per run | 3,360 | 3,360 | 3,360 |
| Of those, with an IK solution | 1,750–1,811 | 1,749–1,813 | 1,749–1,813 |
| Of those, collision-free | **0** | **0** | **0** |
| First contact found: pot into mug | 80% | 57% | 66% |
| First contact found: arm into box | – | 23% | 14% |
| First contact found: other (arm into table, pedestal or itself) | 21% | 21% | 21% |
| Run time | 91–98 min | 93–101 min | 94–102 min |

- **Sampling:** 3,360 is the planner's whole retry budget. It makes 4 entry rounds, and each round escalates through 120, 240 and 480 samples.
- **Contact counts:** the collision check stops at the first contact it finds, so each IK solution is counted once, under that pair. A solution counted under the table may also touch the mug.
- **Run time:** the runs are slow because they use up the whole budget. They didn't hit the time limit. Zero-width TSRs aren't what makes them slow: a variant with the transport goal raised to 10.5 cm, also zero-width, takes 6–10 minutes per run.

## Reading

- **The success rates are close: 0/30 here against 2/30 for ReKep's loop, with the constraints held fixed.**
  - **The constraints make most of the difference:** with LangFold's planner held fixed, success goes from 0/30 to 30/30 when ReKep's program is replaced by LangFold's TSRs.
- **The two solvers fail differently on the same program.**
  - **ReKep's loop returns plans that collide:** its solver treats collision as a cost, so it returns plans that reach the program's targets through contact. It made 29 plans, and 27 of them touch the mug, box or table deeper than 1 mm, with a median depth of 3.1–4.1 mm per scene.
  - **LangFold's planner returns no plan:** collision is a hard limit, and none of the 3,360 transport goals it samples per run is both reachable and collision-free. So it reports that no plan exists rather than returning one that collides.
- **Why the goal is infeasible:** the program fixes the spout's position exactly, keeps the pot upright, and leaves only yaw free. At 10 cm above the opening, the pot's body reaches into the mug, as in ReKep's own plans. The program has no bound to relax, so no planner can move the goal off the mug.

## Variant: transport goal raised to 10.5 cm

This variant is **not ReKep's program**. It changes one number: the transport goal puts the spout
10.5 cm above the opening instead of 10 cm (`--rekep-transport-height 0.105`). Everything else is the
row above: the program's other constraints, still zero-width, and the same planner settings. At 10 cm an
upright pot is 3 mm into the mug, so 10.5 cm is just above the height where it clears. I chose it without asking first, so it
is a probe of how close the program is to feasible, not a tuned or agreed setting. Run names are
`rekepcb105_s*`.

### Success

| Constraints | Solver | No box | Box on path | Box beside | Total |
|---|---|---|---|---|---|
| ReKep's program (10 cm) | LangFold's planner | 0/10 | 0/10 | 0/10 | 0/30 |
| **ReKep's program, transport at 10.5 cm** | **LangFold's planner** | **10/10** | **10/10** | **9/10** | **29/30** |
| LangFold's TSRs | LangFold's planner | 10/10 | 10/10 | 10/10 | 30/30 |

### Per scene

| | No box | Box on path | Box beside |
|---|---|---|---|
| Success rate | 10/10 | 10/10 | 9/10 |
| Tilt while carrying, median / max (limit 15°) | 1.1° / 1.1° | 1.1° / 1.1° | 1.1° / 1.1° |
| Path constraint violation rate | 0/10 | 0/10 | 0/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 1.7 / 1.9 mm | 1.6 / 2.5 mm | 1.7 / 2.2 mm |
| Spout tip movement while pouring, median / max | 27.6 / 36.2 mm | 87.2 / 160.8 mm | 74.9 / 142.0 mm |
| Goal failure rate | 0/10 | 0/10 | 0/10 |
| Spout off straight down at the end, median / max (limit 15°) | 8.1° / 9.9° | 7.0° / 9.0° | 8.6° / 9.7° |
| Pot tilt at the end | 81–88° | 82–88° | 81–86° |
| Spout height above the opening at the end | 50.0 mm, every run | 50.0 mm, every run | 50.0 mm, every run |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 1/10 |
| Deepest collision, median / max | 0.0 / 0.0 mm | 0.0 / 0.2 mm | 0.0 / 2.4 mm |
| Run time | 6–9 min | 7–10 min | 6–9 min |

- **The one failure, box beside seed 8:** the pot goes 2.4 mm into the box, and a finger 1.1 mm, at 2 of the scorer's densified points. The planner checked the plan for collision, so the contact is probably between two of its waypoints, where the scorer checks more finely.
- **Grazes under the limit:** 3 other box runs touch the box by 0.2–0.7 mm with a finger.
- **Spout movement while pouring is vertical:** ReKep's pour path pins only the spout's x and y and leaves its height free. The spout stays within 2.5 mm of the mug's axis, so the movement is up and down. The movement metric is half the range on the axis that moves most, so the spout's height range while pouring is 31–322 mm, the largest on the box scenes. Under LangFold's TSRs, which pin the spout within ±5 mm, the metric is at most 7.6 mm.

### Reading

- **5 mm separates 0/30 from 29/30.** The program's only blocker under LangFold's planner is its transport goal sitting 3 mm inside the mug; raised just clear of it, the rest of the program plans and scores like LangFold's TSRs.
- **So this does not isolate the representation:** given a hand-corrected goal, ReKep's program reaches LangFold's success rate on LangFold's planner. What LangFold's TSRs give here is a region that contains collision-free goals without that correction, plus a pour that holds the spout still rather than leaving its height free.
- **ReKep's own loop at 10.5 cm was not run,** so this variant has no ReKep's-loop counterpart.

## Caveats

- **Run times:** they're wall-clock times with up to 30 runs in parallel, so they aren't comparable across rows.
- **The pour stage is untested here:** no run reached it, so this row says nothing about how ReKep's pour constraints do under LangFold's planner.
- **Lost runs:** for both the 10 cm row and the 10.5 cm variant, seeds 5–9 were run twice (box beside seed 5 at 10.5 cm survived and was kept). The first attempt was killed by a mistake in my launch script, and its records were deleted. The numbers here are from the relaunch, with the same settings. The time limit for the relaunched 10 cm runs was 3 hours instead of 2; no run came near either limit.

## Reproducing

```
scripts/plan_pour_tea.py --seed $s --rekep-constraints --upright-deg 10.5 --eps 0.03 --timeout 60 \
    --pour-goal-region --entry-rounds 4 --scene ... --spec ...
```

The 10.5 cm variant adds `--rekep-transport-height 0.105` and writes `rekepcb105_s$s.npz`; those plans are scored with
`scripts/eval_pour.py --scene ... --spec ... --plan outputs/plans/<scene>/rekepcb105_s*.npz`.

Scenes and specs are as in Table 1:
- no box: `scenes/pour_tea.json` and `tasks/specs/pour_tea.json`;
- box on path and box beside: `outputs/plans/{pour_obstacle,pour_box12}/scene.{scene,spec}.json`.

Run records are in `outputs/plans/{pour_compare,pour_obstacle,pour_box12}/rekepcb_s{0..9}.run.json`. No plan files exist, because no run produced a plan. The variant's run records and scores are `rekepcb105_s{0..9}.{run,eval}.json` in the same folders; its plans (`.npz`) are git-ignored.
