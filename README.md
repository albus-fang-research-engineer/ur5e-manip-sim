To see candidate points of all objects:
```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa     PYTHONPATH=. python scripts/render_candidates.py 
```
To see candidate points of specified objects:
```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python scripts/render_candidates.py --object mug 
```

# One-shot pipeline: scene picture + language query -> plan -> video

```
export ANTHROPIC_API_KEY=...
MUJOCO_GL=osmesa PYTHONPATH=. python scripts/run_pipeline.py --task "pour tea"
```
Steps (each its own script; `--from STEP` resumes, `--until STEP` stops):
pool, mark, plan (VLM #1), ground (part names -> runtime frames.json,
`outputs/grounding/<scene>`), render, select (VLM #2), preview, emit (VLM #3,
compile-gated), path, video, exec. Arms: `--text-only`, `--no-ground`
(authored sidecars), `--no-emit` (pour_stages hand compilers),
`--ground-provider masks --masks-root DIR` (SAM part masks instead of the
oracle bands). Each run writes `outputs/runs/<stamp>/run.json` with the
commands issued and artifact paths. Any downstream script accepts
`--grounding outputs/grounding/<scene>` to read the runtime symbol tables.

# Representation study: TSRs vs ReKep vs OmniManip

Does the way a task's constraints are written change how well a plan can meet
them? The study runs three representations, each with its own planner, on one
scene and grades every plan with the same code against one written spec. No VLM
or LLM runs at plan time. `manip_sim/rekep/` and `manip_sim/omnimanip/` are
self-contained baselines, and the rest of the repo does not import them, apart
from `plan_task.py --grasp-program`, which reads the shared grasp.

| | Constraints | Planner | Guarantee |
|---|---|---|---|
| Ours (`plan_task.py`) | TSRs: a band on each pose axis, for the subgoal and the path | Goal sampling, then CBiRRT projecting onto the path TSR | Hard: samples stay inside the bands |
| ReKep (`manip_sim/rekep/`) | Python costs on keypoints | Upstream's subgoal and path solvers (annealing + SLSQP) | Soft: a weighted penalty, checked after the solve |
| OmniManip (`manip_sim/omnimanip/`) | Aligned axes and a signed distance along one (Eqs. 2-5) | One SLSQP solve for the target pose, then straight moves (supp. A.4) | Soft: traded against `L_path` by its weights |

The spec is `tasks/specs/insert_marker.json`: tail-first, at most 10° from
vertical at the goal and on the path while inside the rim, tail at least 40 mm
below the rim. The marker has to turn 90° on the way, so the limit cannot apply
to the whole transport. All three start from one grasp and state the same
constraints, so neither is part of the test. Ours uses the hand-authored TSRs in
`tasks/hand_tsr/insert_marker/`, which say what the ReKep program says: upright with
the tail 4 cm above the opening, then lowered to 4.5 cm below it near the opening's
axis. Results are over seeds 0-4 at 0.50 m separation. Physics reuses
`execute_task_plan.py` with no planner-specific code. "Tilt" under Transport is the
marker's resting up axis from vertical; it must reach about 90° by the end, so the
rows show how early it turns.

<!-- comparison:start -->
|  | Ours, 5°, hand TSRs | Ours, 1°, hand TSRs | Ours, 10°, hand TSRs | ReKep | OmniManip |
|---|---|---|---|---|---|
| Plans found | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Meets the goal spec | 5/5 | 5/5 | 3/5 | 5/5 | 5/5 |
| **Goal deviation:** final tilt | 1.6–5.5° | 0.7–1.1° | 6.1–11.1° | 0.6–1.1° | 0.1° |
| Lower end off the cup's axis | 2.9–5.7 mm | 3.5–5.1 mm | 3.5–5.3 mm | 0.2–0.9 mm | 0.5 mm |
| Depth below the rim | 40.5–49.2 mm | 40.3–48.6 mm | 40.3–48.6 mm | 45.0 mm | 45.1 mm |
| **Constraint violation:** goal tilt over limit | 0/5 | 0/5 | 2/5 | 0/5 | 0/5 |
| Worst tilt inside the rim | 2.7–6.9° | 0.7–1.1° | 6.1–11.1° | 0.6–1.4° | 0.1° |
| Meets the path limit inside the rim | 5/5 | 5/5 | 3/5 | 5/5 | 5/5 |
| Plans with a collision (deepest) | 1/5 (2.2 mm) | 0/5 | 0/5 | 0/5 | 0/5 |
| **Transport:** tilt at 25% of travel | 13.2–20.0° | 10.0–19.9° | 13.9–18.8° | 2.1–20.3° | 26.2° |
| Tilt at 50% | 40.6–47.6° | 36.5–49.6° | 38.6–51.3° | 25.6–52.4° | 52.5° |
| Tilt at 75% | 70.3–80.2° | 64.6–80.1° | 67.1–80.5° | 70.4–79.5° | 78.7° |
| Rotation done by half travel | 43–57% | 43–58% | 45–51% | 29–59% | 58% |
| **Path:** end-effector travel | 1.24–1.27 m | 1.23–1.31 m | 1.24–1.30 m | 1.34–1.46 m | 1.15 m |
| Joint travel | 7.8–10.7 rad | 9.3–9.8 rad | 8.2–10.3 rad | 9.0–11.1 rad | 10.2–10.4 rad |
| Planning time | 419.4–444.8 s | 421.7–442.1 s | 393.5–447.2 s | 181.4–255.0 s | 0.6–0.7 s |
| **Physics:** in cup | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Beside the cup | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| Object lost from the hand | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| Tilt after release, in cup | 0.4–36.3° | 0.3–26.0° | 23.4–36.1° | 0.4–1.1° | 0.2–23.6° |
<!-- comparison:end -->

Every run, the sweeps and the raw values are in
`outputs/comparison/insert_marker_hand_tsrs.md` and `.json`. Ours with Opus's TSRs is
in `outputs/comparison/insert_marker_same_grasp.md`, and with sampled grasps as well
in `outputs/comparison/insert_marker.md`.

**Read these results with care:**
- **The constraints explain most of the earlier gap.** From the same grasp, Opus's
  TSRs put 8 of 15 plans in the cup, with 11 colliding and 7 meeting the path limit.
  Hand TSRs give 15, 1 and 13. Opus's `insert` path TSR bounds neither position nor
  shaft tilt, its transport ends with the marker horizontal, and its goal is a
  +-15 mm cube, which is why its tail ends 9-16 mm off the axis.
- **Feasibility over separation, 21 plans per planner.** Ours finds 18, ReKep 21, and
  OmniManip 4 of its 7 (it has one plan per separation). Meeting the spec with no
  collision: ours 17, ReKep 16, as ReKep's penalties let plans through at up to
  38.1 mm into the table. Our 3 failures are not the search running out: 1 at 0.80 m
  is at the grasp step, and 2 at 1.00 m have every lowering goal about 3 rad from the
  start, because stages are planned in turn and transport chooses its end without
  checking the next stage. Those 2 still fail at 80 s per goal. Seeds 3-8 at 0.80 m,
  not in the table: 2 of 6 plan, and the other 4 fail at the grasp step.
- **The marker moves in the hand.** Every planner plans a rigid grip. In physics, the
  tail has shifted about 26 mm by the time it enters the rim, for all three, and the
  clearance is 33.6 mm (rim radius less shaft radius; measured with a one-off trace). Plans that enter well off the
  axis land on the rim with no collision in the plan. One plan, `hand5_s1`, slips
  90-109 mm during `insert`, unexplained so far.
- **Collision checks are spaced.** `plan_constrained` checks every `eps` = 0.12 rad,
  which can step over the cup wall when the path TSR allows that approach, as Opus's
  does. Hand TSRs rule it out with their corridor: at 0.12 rad 1 of 15 plans collides
  (2.2 mm), and at 0.03 rad 2 of 15 do (up to 1.0 mm) with 14 of 15 in the cup. See
  `outputs/comparison/insert_marker_hand_tsrs_eps003.md`.
- **Bands are per axis.** A 10° band allows more than 10° of total tilt, so 2 of 5
  plans at 10° miss the goal spec. For the same reason the transport goal can fall
  outside a lowering corridor of the same width, which the compiler rejects, so
  these runs set the corridor to 1.5 times the band (`--moderate`). IK can also answer
  a full turn away, the same pose out of the planner's reach; `--grasp-program`
  corrects this, while sampled grasps do not.
- **The shared grasp is close, not identical, for ReKep.** It comes from
  `manip_sim/omnimanip/programs/insert_marker.json`, 22 mm along the shaft and 3.9 mm
  above its axis. Ours lands within 0.02 mm and 0.01° of OmniManip's. ReKep solves
  its own grasp stage and lands within 0.6 mm and 3.7°, with the wrist turned 180° in
  4 of 5 seeds, which grips the same way.
- **Grasp mattered before it was fixed.** With sampled grasps, 8 of our 14 plans lost
  the marker, all gripping 7-9 mm above its axis; from the shared grasp only 1 of 60
  does, `hand5_s1` above. With the motion fixed, up to 6 mm holds and 8 mm
  or more loses it, at any position from 8 to 35 mm along the shaft. Closing-axis tilt
  off the table holds up to 5° and fails from 10°. See
  `outputs/comparison/insert_marker_grasp_factors.md`.
- **The lookahead does not apply to a given grasp.** It scores a grasp from 8 goal
  probes and rejects a score of 0, which rejected 6 of the 15 runs from the shared
  grasp although all 6 plan when it is skipped. All 6 failed runs with sampled grasps
  fail on this check too.
- **Residuals are not rankable.** ReKep and OmniManip optimize toward a point and
  TSR sampling lands anywhere inside its band, so compare pass/fail at a shared
  tolerance. Ours at 1° is level with ReKep on final tilt.
- **OmniManip's precision is a choice.** The paper gives no weights, and these are
  set so `L_path` only breaks ties; the weight sweep shows the tilt it can trade.
- **Transport tilt is the program's.** Neither ours nor ReKep's program limits tilt
  during transport, so both turn early like OmniManip, which has no way to state a
  limit. Opus's TSRs held the marker flat for longer; ours could state that too.
- **Planning times are not rankable.** Runs shared the machine under different loads.
  Under similar load in the sweep, ours took 300-410 s and ReKep 200-610 s.
- **One scene, few seeds.** A seed fixes our goals but not our path, because
  `plan_task` does not seed `plan_constrained`. OmniManip fails from 0.70 m apart: near a wrist
  singularity at 0.70 and 0.80 m (at 0.80 m the check may be too strict for this
  waypoint density), and in the final descent at 1.00 m. Tilt after release varies
  widely between near-identical plans, so it needs many runs per condition.
- **Open-loop.** All three plan once. This is fair for undisturbed kinematics, but
  says nothing about ReKep's or OmniManip's reactive loops.

Reproduce:
```
git submodule update --init manip_sim/rekep/upstream   # ReKep's solvers (no license, so not copied here)

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=.
OUT=outputs/plans/compare; SCENE=scenes/insert_marker.json
python scripts/plan_rekep.py     --scene $SCENE --seed 0 --out $OUT/rekep_s0.npz
python scripts/plan_omnimanip.py --scene $SCENE --seed 0 --out $OUT/omni_s0.npz
GRASP="--grasp-program manip_sim/omnimanip/programs/insert_marker.json"
H=tasks/hand_tsr/insert_marker
HAND="--stage-plan $H/stage_plan.json --selections $H/selections.json --emissions $H/emissions.json $GRASP"
python scripts/plan_task_tol.py 5 --moderate 7.5 --scene $SCENE $HAND --seed 0 --out $OUT/hand5_s0.npz
python scripts/plan_task_tol.py 5 --eps 0.03 --moderate 7.5 --scene $SCENE $HAND --timeout 60 --seed 0 \
    --out outputs/plans/compare_eps003/hand5_s0.npz                                          # finer checks
python scripts/plan_task_tol.py 5 --moderate 7.5 --scene outputs/plans/omnimanip_separation_sweep/sep80.scene.json \
    $HAND --seed 0 --out outputs/plans/separation_sweep/hand5_sep80_s0.npz                   # same for plan_rekep
TASK="--scene $SCENE --grounding outputs/grounding/insert_marker_gt_opus1 \
    --stage-plan outputs/runs/gt_opus1/stage_plan.json --selections outputs/runs/gt_opus1/selections.json \
    --emissions outputs/runs/gt_opus1/emissions_call3_opus.json"
python scripts/plan_task_tol.py 5 $TASK $GRASP --seed 0 --out $OUT/same5_s0.npz              # Opus's TSRs
python scripts/plan_task_tol.py 5 $TASK --seed 0 --out $OUT/tsr5_s0.npz                      # sampled grasp
python scripts/plan_task_tol.py 5 --eps 0.03 $TASK $GRASP --timeout 60 --seed 0 \
    --out outputs/plans/compare_eps003/same5_s0.npz                                          # finer checks

python scripts/eval_plan.py --scene $SCENE --spec tasks/specs/insert_marker.json --plan $OUT/*.npz
python scripts/execute_task_plan.py --scene $SCENE --plan $OUT/rekep_s0.npz      # for each plan
COMPARE="python3 scripts/compare_plans.py --spec tasks/specs/insert_marker.json"
$COMPARE --plans $OUT --metrics outputs/metrics/vlm --methods hand5 hand1 hand10 rekep omni \
    --weight-sweep outputs/plans/omnimanip_weight_sweep \
    --separation-sweep outputs/plans/omnimanip_separation_sweep --planner-sweep outputs/plans/separation_sweep \
    --out outputs/comparison/insert_marker_hand_tsrs --readme README.md
$COMPARE --plans outputs/plans/compare_eps003 --metrics outputs/plans/compare_eps003/metrics \
    --methods hand5 hand1 hand10 --out outputs/comparison/insert_marker_hand_tsrs_eps003
$COMPARE --plans $OUT --metrics outputs/metrics/vlm --methods same5 same1 same10 rekep omni \
    --out outputs/comparison/insert_marker_same_grasp
$COMPARE --plans $OUT --metrics outputs/metrics/vlm --seconds $OUT/plan_task_seconds.json \
    --methods tsr5 tsr1 tsr10 tsr5more rekep omni --out outputs/comparison/insert_marker
$COMPARE --plans outputs/plans/compare_eps003 --metrics outputs/plans/compare_eps003/metrics \
    --methods same5 same1 same10 --out outputs/comparison/insert_marker_same_grasp_eps003
```
`outputs/runs/` is not in the repo, so reproducing ours needs a `gt_opus1` run.

## Pour tea: where ReKep's goals come from

ReKep's goal poses are not authored and do not come from a published run. We write
constraints, and ReKep's solver finds the pose that satisfies them. Upstream publishes
no pour-tea run: the only generated program in its repo is `vlm_query/pen/`, so no
pour-tea keypoints, constraint code or solved poses exist to copy or check against.

What upstream does publish is a text example inside the prompt it gives the LLM, in
[`vlm_query/prompt_template.txt`](https://github.com/huangwl18/ReKep/blob/63c43fd/vlm_query/prompt_template.txt#L11-L20)
(lines 11-20) and the paper's appendix A.6 (PDF pp. 22-23). We wrote
`manip_sim/rekep/programs/pour_tea/` from that text:

| Stage | Prompt text | Our cost |
|---|---|---|
| Align, subgoal | "the teapot spout needs to be 10cm above the cup opening" | Distance from `spout_tip` to 10 cm above `opening_center` |
| Align, path | "the robot must still be grasping the teapot handle" | Upstream's grasping cost |
| Align, path | "the teapot must stay upright to avoid spilling" | Cross product of the pot's up axis with z, plus a term against carrying it upside down |
| Pour, subgoal | "the teapot spout needs to be 5cm above the cup opening" | Distance from `spout_tip` to 5 cm above `opening_center` |
| Pour, subgoal | "the teapot spout must be tilted to pour liquid" | Pour axis within 10° of straight down |
| Pour, path | "the robot must still be grasping the teapot handle" | Upstream's grasping cost |
| Pour, path | "the teapot spout is directly above the cup opening" | Horizontal distance from `spout_tip` to `opening_center` |

The 10 cm and 5 cm are ReKep's numbers, but they are what the prompt tells an LLM, not
what a run produced. An LLM could write different values for a given scene. These
choices have no upstream source and are ours:

- The pour angle: the prompt only says "tilted", so 10° from straight down is ours.
- The keypoints: upstream proposes them from vision, and we take them from the sim's
  object symbols.
- The upright cost's scale: it uses a 10 cm virtual axis through the pot. Upstream's
  pen program uses the same cross-product form on real keypoints about 14 cm apart.
- The grasp stage: it is given, as for the other planners, rather than solved.

`scripts/plan_rekep.py --upstream-loop` runs these constraints through the loop in
upstream's `main.py`: solve, move through 5 poses, solve again from where the arm is,
and backtrack at its tolerance, until a path fits in those 5 poses. Nothing is rejected,
and the solvers keep upstream's values, including its 1e-4 tolerance. Unlike the
insert-marker results above, this is not one solve per stage. A single solve is not a
fair stand-in here: upstream scores straight segments between control points but
executes a spline through them, and over seeds 0-4 with no box, one solve per stage left
the rim in 3 of 5 plans, against none with the loop. Over seeds 0-9 the loop leaves it in
1 of 10 (table below).

Some differences remain because this sim is not upstream's: mink's IK moves the arm in
place of the Fetch controller, a damped least-squares IK replaces Lula inside the cost,
and collision uses convex hull distances and hull vertices in place of the exact mesh
and surface samples.

```
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=.
OUT=outputs/plans/pour_compare; SA="--scene scenes/pour_tea.json --spec tasks/specs/pour_tea.json"
python scripts/plan_rekep.py $SA --upstream-loop --seed 0 --out $OUT/rekeploop_s0.npz        # seeds 0-9
python scripts/eval_pour.py $SA --plan $OUT/rekeploop_s*.npz
python scripts/eval_pour.py --table $OUT --spec tasks/specs/pour_tea.json \
    --methods ours10g ours5g rekeploop omnirrt omni --out outputs/comparison/pour_tea_10seeds/no_box
```

### Pour tea results, seeds 0-9

A success reaches the goal, holds the path limits and has no collision deeper than 1 mm,
out of all runs. The other rows are over the plans returned. A path constraint violation
is tilt over the limit while carrying or the spout outside the rim while pouring. Medians
and maxima are across plans, of each plan's worst value.

OmniManip + RRT-Connect keeps OmniManip's target pose and replaces its straight move with
RRT-Connect (`plan_omnimanip.py --mover rrt`): our CBiRRT with every axis free, the held
pot in the collision check, and the same 60 s timeout and 0.03 rad check step as ours. It
separates what the representation does from what the straight-line executor does.
Straight-line OmniManip reaches the same pose from every seed; RRT-Connect's paths differ.

No box:

|  | Ours, 10.5° per axis, goal region | Ours, 5° per axis, goal region | ReKep, upstream's loop | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | 10/10 | 10/10 | 1/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 9.9° | 3.9° / 5.2° | 1.1° / 9.3° | 76.8° / 87.9° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 25.8 mm / 32.3 mm | 25.1 mm / 32.3 mm | 20.9 mm / 245.8 mm | 42.9 mm / 328.0 mm | 42.7 mm / 42.8 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 9/10 | 0/10 | 0/10 |
| Deepest collision, median / max | 0.0 mm / 0.0 mm | 0.0 mm / 0.0 mm | 3.1 mm / 18.3 mm | 0.0 mm / 0.3 mm | 0.0 mm / 0.0 mm |

Box at x = 0.03, near the line between the pot and the mug:

|  | Ours, 10.5° per axis, goal region | Ours, 5° per axis, goal region | ReKep, upstream's loop | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | 10/10 | 10/10 | 0/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 10.7° / 14.9° | 6.2° / 7.2° | 1.1° / 8.6° | 75.2° / 83.5° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 2/9 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 23.8 mm / 27.4 mm | 25.3 mm / 32.0 mm | 32.9 mm / 70.7 mm | 43.2 mm / 43.9 mm | 42.7 mm / 42.7 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 9/9 | 0/10 | 10/10 |
| Deepest collision, median / max | 0.0 mm / 0.0 mm | 0.0 mm / 0.0 mm | 34.9 mm / 55.7 mm | 0.0 mm / 0.0 mm | 73.4 mm / 73.4 mm |

Box at x = 0.12, off that line:

|  | Ours, 10.5° per axis, goal region | Ours, 5° per axis, goal region | ReKep, upstream's loop | OmniManip + RRT-Connect | OmniManip |
|---|---|---|---|---|---|
| Success rate | 10/10 | 10/10 | 0/10 | 0/10 | 0/10 |
| Tilt while carrying, median / max (limit 15°) | 8.4° / 12.7° | 4.9° / 6.5° | 2.0° / 10.5° | 76.8° / 87.9° | 71.2° / 71.2° |
| Path constraint violation rate | 0/10 | 0/10 | 1/10 | 10/10 | 10/10 |
| Spout off the mug's axis while pouring, median / max (rim 44.1 mm) | 21.7 mm / 32.4 mm | 21.8 mm / 32.1 mm | 20.5 mm / 72.3 mm | 43.5 mm / 259.9 mm | 42.7 mm / 42.8 mm |
| Collision rate, deeper than 1 mm | 0/10 | 0/10 | 10/10 | 0/10 | 0/10 |
| Deepest collision, median / max | 0.0 mm / 0.0 mm | 0.0 mm / 0.0 mm | 3.2 mm / 39.9 mm | 0.0 mm / 0.5 mm | 0.0 mm / 0.0 mm |

Read ReKep's success and collision rates with the contact behind them. Almost every ReKep
plan presses about 3 mm into the mug, and that alone fails the 1 mm rule. The contact is at
the transport stage's subgoal, the spout 10 cm above the opening with the pot upright, and
lasts into the start of the pour. It is not on the path between poses. ReKep's subgoal
solver does score collision there, but its cost is soft. Box contact is 6 of 9 plans at
x = 0.03 and 3 of 10 at x = 0.12. ReKep's success if mug contact up to a threshold were
allowed (other contact still at 1 mm):

| Mug contact allowed | No box | Box x = 0.03 | Box x = 0.12 |
|---|---|---|---|
| 1 mm | 1/10 | 0/10 | 0/10 |
| 3 mm | 4/10 | 2/10 | 2/10 |
| 5 mm | 9/10 | 3/10 | 6/10 |

Carrying tilt stops counting when the spout first enters the rim, so it can understate the
transport stage: seed 4 at x = 0.03 reports 1.1° but reaches 32.5° in that stage. That plan
is one of the rim exits. ReKep's seed 1 at x = 0.03 counts as no plan because it backtracked to
the grasp, which upstream would redo and this planner cannot, since the grasp is given.

`outputs/comparison/pour_tea_10seeds/` holds, per scene (`no_box`, `box_x003`, `box_x012`),
this summary (`.summary.md`), the constraint table (`.md`), every metric (`.full.md`) and
the per-plan grades (`.json`). The older `pour_tea_loop` is seeds 0-4 of the no-box scene.
`pour_tea_obstacle` and `pour_tea_box12` predate the loop.

# Perception wing: FoundationPose + PointSO + ROS2 bridge

Three new services following the existing `grasp` sidecar pattern. Everything
is host-networked; fixed ports: grasp `5666` (existing), pose `5667`,
pointso `5668`.

```
ur5e-manip-sim/
├── docker-compose.yml                  # existing (unchanged)
├── docker-compose.perception.yml       # NEW overlay
├── docker/
│   ├── Dockerfile.foundationpose       # NEW
│   ├── Dockerfile.pointso              # NEW
│   └── Dockerfile.ros2                 # NEW
├── pose_server/server.py               # NEW  (mounted into pose)
├── pointso_server/server.py            # NEW  (mounted into pointso)
├── ros2_bridge/
│   ├── pose_bridge_node.py             # NEW  (mounted into ros2-bridge)
│   └── pointso_bridge_node.py          # NEW
├── foundationpose_runtime/
│   ├── weights/                        # refiner 2023-10-28-18-33-37,
│   │                                   # scorer  2024-01-11-20-02-45
│   └── meshes/                         # object CADs (.obj)
└── pointso_runtime/
    └── checkpoints/                    # small.pth / base_finetune.pth
```

## One-time setup

```bash
# PointSO checkpoint (small; use base_finetune.pth for Open6DOR-style tasks)
mkdir -p pointso_runtime/checkpoints
wget -c https://huggingface.co/qizekun/PointSO/resolve/main/small.pth \
     -P pointso_runtime/checkpoints/

# FoundationPose weights: from the repo's Google Drive link (README), place
# both run folders under foundationpose_runtime/weights/
mkdir -p foundationpose_runtime/weights foundationpose_runtime/meshes
```

## Build & run

```bash
docker compose -f docker-compose.yml -f docker-compose.perception.yml \
    --profile perception build pose pointso

docker compose -f docker-compose.yml -f docker-compose.perception.yml \
    --profile perception up -d pose pointso

docker compose -f docker-compose.yml -f docker-compose.perception.yml \
    --profile ros2 up -d ros2-bridge
```

Smoke test from the host or `sim`:

```python
import zmq, msgpack, msgpack_numpy; msgpack_numpy.patch()
s = zmq.Context().socket(zmq.REQ); s.connect("tcp://127.0.0.1:5668")
s.send(msgpack.packb({"cmd": "ping"})); print(msgpack.unpackb(s.recv()))
```

## Notes / sharp edges

- **GPU pinning**: both sidecars are pinned to `device_ids: ["0"]` because
  GPUs 1–2 are saturated by the FSDP VLA training. Change in the overlay if
  that run finishes.
- **Arch**: `TORCH_CUDA_ARCH_LIST=8.6` everywhere (3090). Host `nvcc` 10.1 is
  irrelevant — containers ship their own toolkits.
- **FoundationPose build drift**: `build_all.sh` occasionally breaks against
  upstream kaolin/repo changes. If the image build fails there, set
  `--build-arg FP_COMMIT=<known-good SHA>`.
- **SoFar deps**: the Dockerfile does a full `pip install -e .` (pulls the
  whole SoFar dep tree, including VLM-side extras you don't need for PointSO
  alone). If it fights the base image's torch, retry with `--no-deps` and
  install `easydict pyyaml timm` manually.
- **ROS2 bridge**: host network + `ipc: host` so DDS finds the robot and any
  other Humble machines. Set `ROS_DOMAIN_ID` in your shell env to match.
  The bridge nodes are ZMQ clients only — no CUDA, no model code — so the
  perception environments never see ROS's Python 3.10 pin.
- **Registration masks**: the pose bridge expects an instance mask on
  `/pose_bridge/mask`. In sim, publish the ground-truth instance mask; on
  hardware, front it with your SAM/Florence segmenter (which SoFar already
  bundles if you later widen the pointso image to full SoFar).


  To run frame refinement demo
  ```
  python scripts/demo_refine_frame.py teapot --render outputs/refine_frame/teapot_frame.png
  ```


  VLM in-the-loop planning and execution
  ```
  # 0. preconditions (already done if candidates.json exists)
  PYTHONPATH=. python scripts/propose_interaction_points.py --write

  # 1. VLM-facing renders — one subset per object, all its stage parts
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_candidates.py \
      --object teapot --vlm --parts handle spout
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_candidates.py \
      --object mug --vlm --parts rim
  # eyeball outputs/candidates/vlm/<obj>/*.png: ≤8 marks, 20px labels legible

  # 2. live touchpoint #2 — four calls, writes selections + audit log
  export ANTHROPIC_API_KEY=...
  PYTHONPATH=. python scripts/select_frames.py

  # 3. inspect what the VLM chose, before any planner touches it
  PYTHONPATH=. python scripts/preview_selections.py \
      outputs/selections/pour_tea.json --render outputs/selections/preview.png
  cat outputs/selections/pour_tea.log.json   # attempts/rejections per call

  # 4. plan on the VLM-selected frames, then execute
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/plan_pour_tea.py \
      --selections outputs/selections/pour_tea.json
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/execute_pour_tea.py --arm vlm
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py --arm vlm
  ```

  Emission-ablation arms (hand-authored vs VLM)
  ```
  # --selections decides the arm; plan_pour_tea STAMPS it into the npz
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/plan_pour_tea.py
  #   -> outputs/plans/hand/pour_tea_full_hand.npz          (arm = hand)
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/plan_pour_tea.py \
      --selections outputs/selections/pour_tea.json
  #   -> outputs/plans/vlm/pour_tea_full_vlm.npz            (arm = vlm)

  # render/execute read the stamp: --arm auto (default) needs no flag when
  # only one arm has a plan on disk, and refuses to guess when both do
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py  --arm hand
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/execute_pour_tea.py  --arm hand
  #   -> outputs/videos/hand/pour_tea_full_hand.mp4
  #      outputs/videos/hand/pour_tea_exec_hand.mp4
  #      outputs/metrics/hand/pour_tea_exec_hand.json   ("ablation_arm": "hand")

  # a flag that contradicts the stamp is an ERROR, not an override --
  # the point is that a VLM run can never be filed as ground truth
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_full_plan.py \
      --plan outputs/plans/vlm/pour_tea_full_vlm.npz --arm hand   # SystemExit
  ```


  To verify VLM selection and frames per stage (only example for now, no VLM)
  ```
  # 0. drop the updated files in place
  #    manip_sim/selection.py, tests/test_selection.py,
  #    scripts/preview_selections.py, scripts/plan_pour_tea.py

  # 1. make sure candidates.json was written from the REAL meshes
  PYTHONPATH=. python scripts/propose_interaction_points.py            # dry run: read off constructed IDs
  PYTHONPATH=. python scripts/propose_interaction_points.py --write

  # 2. verify/fix the candidate IDs in your selections file against step 1's print
  #    (my example guessed handle_center=1, spout_tip=3, opening_center=1 from the
  #     smoke pool — your real mug also constructs handle_center and mid_cavity,
  #     which shifts the mug ordering, and the teapot IDs can differ too)

  # 3. per-stage preview, both arms
  PYTHONPATH=. python scripts/preview_selections.py outputs/selections/pour_tea.json \
      --render outputs/selections/preview_stages.png
  PYTHONPATH=. python scripts/preview_selections.py outputs/selections/pour_tea.json \
      --refine --tilt-deg 25 --render outputs/selections/preview_stages_refined.png

  # 4. plan through the seam
  PYTHONPATH=. python scripts/plan_pour_tea.py --selections outputs/selections/pour_tea.json
  ```