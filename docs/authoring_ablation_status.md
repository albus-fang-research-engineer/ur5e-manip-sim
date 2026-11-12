# Authoring ablation: status and handoff (2026-11-11)

Where the call-#3 authoring ablation stands, written to start a new session from.
Branch `authoring-ablation` in the worktree `~/ur5e-authoring` (local, not pushed).
Run everything in the sim image: `scripts/authoring_docker.sh <cmd>` (host Python is 3.8).

## What the ablation tests
Only VLM call #3, constraint emission. Calls #1 and #2 (stage plan, interaction
points) are frozen artifacts, and the frames and points are the same for every
variant. The question is which way of writing the constraints authors a correct TSR
most often.

| Variant | What the model writes | Medium |
|---|---|---|
| A0 framed-chained (default) | relation rows with tolerance words | JSON (the schema) |
| A2 numeric bounds | relation rows with numbers | JSON (**not built**) |
| A3 constrained code | relation rows through helper functions, numbers | Python |
| A4 unconstrained code | raw `Bw` (meters, degrees) in GIVEN frames | Python, re-run at the actual start |
| A5 direct bounds | the same raw `Bw` | JSON, used as written |
| A4-pilot | (old design: code computed `T0_w`/`Tw_e`) | kept only to re-plan the pilot artifacts |

Rules the user set:
- The VLM never writes or chooses a frame or a rotation. Frames are perception's
  canonical frames at call #2's points: `w` on the passive object, `e` on the moving one.
- Name variants in full and flag every prompt-text change.
- ReKep is a methodological precedent, not a baseline.

## Decisions made
- Main ablation in **sim, with ground-truth grounding** (2026-11-11). Call #3 sees
  MuJoCo renders on hardware too (`render_asset.py` + `render_stage_frames.py`), so
  the image type matches. Known gap: the sim teapot and mug are untextured, while
  hardware meshes carry TRELLIS textures. State this in the paper.
- Insert marker: either end may go in. The grader reports the lower end but does not
  fail on it (`scripts/eval_plan.py`, report-only).
- Critic off for this ablation. The critic comparison vs OmniManip is recorded as D9 in
  `~/ur5e-manip-sim-critic/docs/deferred.md`.
- Pour spout limit: a 25 deg tolerance looked fine in renders. The spec
  (`tasks/specs/pour_tea.json`, `max_spout_off_down_deg` = 15) is not changed yet.
- **Approach before scaling:** a few runs per option, compare, then pick ONE setup
  before running ~10 per cell.

## Committed (oldest first, since 0489e19)
- `ed91c9b` call logs keep rejected replies, stop reasons, usage, gate rejections (logging only)
- `c6b211b` insert grader: lower end reported, not graded
- `3c46f64`, `e9120de` scorer `scripts/score_authoring_ablation.py`:
  - 3 planner seeds, 3 jobs at once
  - results reused only at the same clean commit and inputs
  - the emission is the reported unit
- `5751fb4` A4 / A5: given canonical frames, the model writes only `Bw`
- `94b0ffe` antipodal grasp sampler `manip_sim/antipodal.py` (`plan_task.py --grasp-source antipodal`)

## Uncommitted at handoff (reviewed; needs the fixes in the next section before committing)
- Sampler update:
  - closed-finger geometry from the measured stroke (`scripts/measure_gripper_closing.py`,
    `manip_sim/data/sim_2f85_closing.json`)
  - depth range per pose
  - pad-landing check, approach bins
  - grasp seed, grasp source in reports
- Handoff consistency score (authoring metric) and handoff filter (planner keeps goals
  the next stage accepts as its start): `manip_sim/task_plan.py`, `scripts/plan_task.py`.
- Scene format: optional per-object `tilt` and `drop_height`; `scenes/insert_marker_upright.json`.
- B1 repeatable planning (iteration budget, per-stage seeded streams, single-threaded
  math), built by a subagent in a separate worktree, to be merged after review.

## Update, later on 2026-11-11 (all uncommitted, in this worktree)
- **B1 merged** (from a subagent worktree):
  - `plan_constrained` has `max_iters` (`DEFAULT_MAX_ITERS = 200`, user-approved; first
    calibrated at 50 from 15 easy successes using 1-22 iterations) and a 120 s safety cap;
  - per-stage seeded streams from `SeedSequence(seed).spawn`;
  - single-threaded math in the planner scripts;
  - new test `tests/test_plan_repeatable.py`. Verified bitwise-identical plans, alone and
    under load, on both tasks.
- **Handoff filter in both planners:**
  - `plan_task.py` redraws goals (`HANDOFF_ROUNDS = 3`) until the next stage accepts one,
    otherwise "no_admitted_handoff".
  - `plan_pour_tea.py` now FILTERS transport goals the pour cannot start from (it used to
    only rank them and start from a projection), probes 30 goals per draw (was 10),
    redraws up to 3 times, and treats a pour compile error at an entry as a rejection.
  - Pour A3 plans end to end with the merged code.
- **Handoff score** moved out of the planner into `manip_sim/spec_scores.py` (robot-free,
  once per emission, samples goal ∩ path, reports all/none-admitted). The scorer computes
  it before planning. Pilot: insert A0 0/64, A3 34/64, A4-pilot 23/64; pour 0/64 for all three.
- **Scene fixes:** a `face` yaw now accounts for the tilt; `spawn_z_of(name)` uses the
  per-object drop height; scipy rotations; a test that existing scenes spawn as before.
- **Sampler rework done** by a subagent, per both reviews:
  - oriented boxes, a free-air stroke and closings on 24 cylinder diameters;
  - landing judged against the object's tangent plane;
  - closed fingers checked against the object's real collision hulls;
  - `Gripper(approach_state="open")` is the default (user approved the fully open approach).

  Finding: the sim 2F-85's pads stay tilted when closed on an object (21 deg at 19 mm,
  13 deg on a flat face), so the sim gripper may differ from a real 2F-85. Flag this for
  hardware. Measured default state matches `SIM_2F85_BOXES` within 0.05 mm.
- **Consequence:** the lying marker gets 1 grasp (fully open) or 0 (partly open); the
  upright marker gets 12 either way. **Pending user decision:** upright marker for insert
  marker (recommended), relax table clearance / count the fingertip end as contact face
  (at most ~4 grasps), or keep 1 grasp.
- **Still to do:**
  - planner side of the fully open approach: set the free-arm kinematics' gripper joints to
    the fully open measured state for the grasp-phase collision checks, reach and approach,
    in `plan_task.py` and `plan_pour_tea.py`; each kinematics object has its own MjData, so
    the carrying phase is unaffected;
  - re-run the insert check;
  - review chat, then `/code-review`, then commit (B1, handoff, scene, sampler, spec_scores);
  - the free trial;
  - the small paid pilot (needs approval and the call count).
- **Broken scratch scripts:** `outputs/emissions/authoring-pilot-preview/grasp_cases.py`
  and `upright_grasps.py` use the removed sampler API.

## Review findings on the uncommitted work (2026-11-11): fix before committing
The fifth chat review and `/code-review` (high) both ran on the uncommitted sampler,
handoff and scene changes. To do:

**Closed-finger geometry** (`manip_sim/antipodal.py`, `scripts/measure_gripper_closing.py`)
- The pads ROTATE across the stroke: about 3 deg at a 58 mm gap, 16 deg fully open,
  36 deg fully closed. Axis-aligned boxes in the site frame therefore misdescribe the
  contact face:
  - the gap is measured between the innermost corners;
  - `pad_z_at` merges pad and fingertip into a face longer than the pad;
  - pad support and landing assume a flat face perpendicular to the closing axis.

  Fix: record oriented boxes (`geom_xmat` and `geom_size`) and a pad-tilt column; derive
  the gap and the contact face from the pad's actual inner face.
- Free-air closing may differ from closing on an object (the 2F-85 is underactuated).
  Close it on a 19 mm test cylinder and record the pad pose at contact; use whichever is
  real.
- Add a test that the measured row at the planner's default gap reproduces
  `SIM_2F85_BOXES` within about 1 mm.
- The pad-landing check is too strict: p2's face can tilt up to the friction-cone angle,
  so a 2 deg taper over the face drops valid grasps. Measure landing relative to each
  pad's own contact plane and normal.
- Check the closed inner fingers and fingertips against the object, not just the open
  boxes.
- `closed_boxes` and `pad_z_at` ignore the `gripper` argument (always sim 2F-85), and
  `Gripper.pad_z` is unused: make the closing table part of `Gripper`.
- Cache the closed geometry per pair; it is recomputed per pose.
- Restore stricter tests: per-grasp pad support on flat faces, max-vs-min for cylinder
  vs box.

**Fully-open approach (option 2)**
The reviewer agrees, on one condition: the state the planner checks must be the state
execution commands during the approach, in sim AND on hardware.

**Handoff filter** (`manip_sim/task_plan.py`, `scripts/plan_task.py`)
- Port it to `plan_pour_tea.py`; today only insert marker has it. Check that pour writes
  the `.plan.report.json` the scorer reads.
- Sample until enough goals are admitted, with a draw cap, instead of filtering one
  ranked list. Otherwise a false "no_admitted_handoff" appears when admitted goals are
  rare.
- It changes outcomes, so record it as a pipeline delta, ON for every variant and for the
  representation study's planners.
- Catch more than `CompileError` around the recompile.

**Handoff consistency score**
- Use it as a diagnostic, not a headline.
- Sample from subgoal ∩ path (what the planner can actually pick), not the subgoal alone.
- Report "all admitted" / "none admitted" binaries next to the fraction. The fraction
  inherits the Euler-box measure: not comparable between A0 (bounds near 0) and A4/A5
  (near pitch -90).
- Compute it once per emission, in the robot-free (B2) scorer. Today it runs inside every
  planner run (per seed), duplicates the lookahead's compile chain, starts a sandbox
  subprocess per sample for code variants (slow), and drops out of the mean when planning
  fails early, which biases toward the worst emissions.
- Its docstring presupposes the result ("shared words make stages nest"); describe only
  what is measured.

**Scene format** (`manip_sim/scene.py`)
- A `face` yaw is resolved from the untilted body; with a tilt, the front no longer
  faces the target.
- `Scene.spawn_z` ignores the per-object drop height.
- Use scipy `Rotation` instead of a hand-written quaternion product.

## Key findings so far
- Pilot (2026-11-08): 1 run each of A0 / A3 / A4-pilot on insert marker and pour tea.
  All passed the gate; none passed the spec. Causes:
  - **grasp:** the fixed grasp suits only cap-first insertion;
  - **A0 pour transport goal** is object-infeasible;
  - **A0 insert path** leaves one tilt axis free;
  - the **stage-to-stage handoff**.
- Handoff consistency on the pilot emissions: A0 0/64, A3 34/64, A4-pilot 23/64. The
  Hypothesis (not a finding yet: three pilot emissions, Euler-box fractions, A3/A4-pilot
  answers from before the current A4/A5 design): the schema's shared tolerance words make
  stages nest; free numbers do not.
- With the handoff filter and antipodal grasps, A3's tip-first insert passes the full spec.
- Planning was not repeatable: CBiRRT ran on a wall clock with unseeded randomness
  (B1 fixes it). Same spec, same seed gave a path pass and a path fail.
- Lying marker and the 2F-85: the planner models the approach with the fingers partly
  open (fingertips 11 mm past the grip site). That leaves a 0.3 mm depth window, i.e.
  fingertip-end grasps only. Fully open (8.1 mm reach) would allow secure pad grasps.
- Upright marker: 11 grasps (side / oblique / top-down) vs 1 lying.
- With e on the canonical axes, "marker vertical" is pitch -90 deg, the Euler
  singularity: a cone around vertical is expressible, separate tilt bounds are not.
  Expect A4 / A5 errors there.

## Open decisions (for the user)
1. **Option 2:** model the approach with the fingers fully open, in both the sampler and
   the planner's grasp check. A planner change for every variant. Recommended.
2. Include the upright-marker scene in the paid pilot?
3. **B2 robot-free layered scores:** needs one more review of exact definitions. The
   reviewer asked for SO(3)-uniform sampling, feasibility-filtered soundness, hand TSRs as
   a secondary diagnostic only, and "any violation" next to fractions.
   Proposed 2026-11-09, **not decided; nothing built for it.** Score the spec itself, layer
   by layer, leaving the robot out until the last layer:

   | Layer | What it measures | Robot? |
   |---|---|---|
   | 1. Format | parses and passes the feasibility gate on the first try; retries needed | no |
   | 2. Goal precision | share of goal-region samples meeting the task spec (insert: vertical within 10 deg, inside the rim, >= 4 cm deep, either end; pour: spout and tilt limits) | no |
   | 3. Goal coverage | share of acceptable goals the region includes, hand-written TSRs as reference; catches over-tight specs | no |
   | 4. Path precision | share of path-region samples near the motion that break the path rules (e.g. tilt inside the cup) | no |
   | 5. Object feasibility | share of goal poses where the objects don't intersect (e.g. A0 pilot pour: teapot inside the mug) | no |
   | 6. Reachability, then plan success | the current scoring: plan, then grade kinematically (goal, path, no penetration) | yes |

   Review changes asked for (not applied): SO(3)-uniform rotation sampling instead of the
   Euler box; precision on feasible samples only; hand TSRs as a secondary check, not
   ground truth (weakens layer 3); "any violation" next to each fraction. Current scoring
   is one plan, all-or-nothing, seed-dependent, with robot, grasp and planner in every
   step; layered scores whole regions by fixed-seed sampling, one fraction per layer. The
   handoff score (`manip_sim/spec_scores.py`) is meant to join it as a diagnostic.
4. A4-frozen control (no model calls), approved in principle by the reviewer.
5. The lookahead checks reachability with the bare arm (held object ignored), so it is
   optimistic; and 8 vs 24 samples per stage.
6. Stage-to-stage rotation gate (reject an emission when no handoff is admitted): the
   user is thinking.
7. The pour spec's 25 deg spout tolerance.

## Next steps (agreed order)
1. Review and merge B1 (its own chat review and `/code-review`). Fix the review findings
   below on the sampler / handoff / scene work, then commit it (workflow: chat review,
   then `/code-review`, then commit).
2. Free trial (no model calls): re-plan the pilot's insert-marker emissions under each
   grasp option (fixed program; antipodal with the current model; antipodal with
   option 2 if approved).
3. Small paid pilot, 1 run per variant (A0, A3, A4, A5): insert marker lying plus pour
   tea = 18 calls minimum (27 with the upright scene). State the count and get approval
   first. It is the first live run of the new A4 / A5.
4. Pick one scene and grasp setup, then scale.

## Pointers
- Review chat: claude.ai "VLM TSR authoring ablation" (project "VLM evaluation"),
  5 reviews so far.
- Pilot outputs: `outputs/emissions/authoring-pilot/`. Renders, videos and scratch scripts:
  `outputs/emissions/authoring-pilot-preview/`. Re-plan checks:
  `outputs/emissions/authoring-*-check/`.
- Memory notes: authoring-ablation-worktree, authoring-ablation-communication,
  vlm-never-writes-rotations, insert-marker-either-end, workflow-propose-review-commit.

## Update, end of session (2026-11-11)
- User chose the **upright marker** for insert marker. New runner/scorer task
  `insert_marker_upright` (scene `scenes/insert_marker_upright.json`, antipodal grasps,
  frozen gt_opus1 grounding / stage plan / selections). The old `insert_marker` task is
  kept.
- **Fully open approach wired into both planners:** `antipodal.approach_joints()` +
  `set_gripper_joints()` on the free-arm grasp-phase kinematics in `plan_task.py` and
  `plan_pour_tea.py`; test `test_the_planners_grasp_phase_gets_the_fully_open_fingers`.
  `DEFAULT_MAX_ITERS = 200`.
- **Free trial** (tag `trial`, `outputs/emissions/authoring-trial/insert_marker_upright/`,
  3 seeds):
  - A0 (pilot answer): planned 3/3, goal 3/3, path 1/3, collision-free 0/3.
  - A5 (a FAKE cap-first answer, pipeline check only): planned 3/3, goal 3/3, path 3/3,
    collision-free 0/3.
  - A3 (pilot answer): fails to compile at the upright start; it was authored tip-down
    for a lying marker, a half-turn away.
  - A4-pilot: grasp lookahead 0.
- **FIRST THING NEXT SESSION:** both planned answers penetrate the cup ("passive") by
  about 7 mm in every seed, although the planner checks collisions. Find the colliding
  pair in the `.plan.eval.json` (colliding points / segments). Suspect a model mismatch:
  the grader's (and the carrying-phase planner's) gripper finger state while holding
  the marker is the spawn default (pads 64 mm apart), not closed on it, or the grader's
  densified path vs the planner's waypoint checks. Fix so planner and grader agree; then
  re-run the trial.
- Then: review chat, then `/code-review`, then commit everything uncommitted (B1,
  handoff filter both planners, `spec_scores`, scene fixes, sampler rework, fully open
  approach, upright task). Then the small paid pilot (A0/A3/A4/A5 x insert upright +
  pour tea, 1 run each, 18 calls minimum) after the user approves the count.

## Update, next session (2026-11-11, continued; uncommitted)
- **Cause of the ~7 mm cup penetration in the trial:** the colliding pair is the held
  marker against the cup (`cup_main <-> marker_main`), not the gripper (so not the finger
  state). In all 6 plans (A0 and A5, seeds 0-2) the hit lies on a joint-space chord between
  two planner waypoints. CBiRRT checked only its nodes, up to `eps = 0.12` rad apart, so the
  end effector moved up to ~10 cm unchecked. The tracker executes, and the grader checks,
  the straight chord at 0.02 rad. In A5 s1, `plan_constrained`'s final densify also created
  an unchecked midpoint that collides. Depths were 1.2-13.2 mm. Diagnostic and pre-fix
  outputs: `outputs/emissions/authoring-collision-diag/`.
- **PIPELINE CHANGE (planner, every variant, both tasks):** every move between waypoints is
  now collision-checked at 0.02 rad (`TRACK_STEP_RAD`). Before, only the waypoints were
  checked, about 0.12 rad apart. In `manip_sim/planning.py`, `chord_clear()` covers each
  CBiRRT extension, the bridge chord between the two trees, and shortcuts. `densify` and
  `TRACK_STEP_RAD` moved there; `execution.py` and `eval_plan.py` import them. The
  planner's check points are the tracker's on every chord except a rare long one (a
  projection lurch over 0.12 rad), where they differ by under 0.02 rad. If merged, this also changes the
  representation study's planners. List it with the handoff filter and B1 repeatable
  planning. Tests: `test_chord_clear_sees_an_obstacle_between_clear_nodes`,
  `test_planned_path_is_clear_at_the_trackers_step` (both fail with the check disabled).
- **Iteration budget raised:** `DEFAULT_MAX_ITERS` 200 -> 500 (user-approved, relayed by the
  review session), because chord checks reject more extensions. Frozen from here; never
  tune it per variant.
- **Before/after comparison** (`--no-chord-check` vs default, same emissions, seeds 0-2,
  no model calls; `outputs/emissions/authoring-chord-ab/report.md`, scripts `compare.py`,
  `diff_paths.py` there):
  - Insert (upright A0, upright fake A5, representation-study A0 "same5" on the lying
    scene): penetration 1.2-13.2 mm in all 9 plans without the check, 0.0 in all 9 with it.
  - No new planning failure; no budget exhaustion caused by the check; no total path more
    than 20% longer. Iterations rise (max 62 per search, budget 500); search time ~2x.
  - Pour A3: seed 0 identical, seed 1 differs only in the pour stage, seed 2 fails the
    same way in both (budget exhausted on all goals). Goal fails in both (spout limit).
  - **Exposed:** with the check on, A0's insert plans (upright and the representation
    study's) enter the cup tilted 11-38 deg, above the spec's 10 deg path limit. Without
    it they passed tilt only by entering at the rim (38-42 mm off-axis against a 42.3 mm
    rim radius), through the cup wall. The A5 fake answer keeps tilt under 7 deg and
    succeeds 3/3.
- **Trial re-run** (check on, budget 500): A0 collision-free 3/3, path 0/3; A5 (fake)
  success 3/3; A3 and A4-pilot unchanged (compile failure / grasp lookahead 0).
- **User decisions (relayed by the review session):** chord check stays ON by default
  (`--no-chord-check` is the opt-out, for comparisons). **PIPELINE CHANGE, planner:** shortcut
  smoothing in `plan_constrained` no longer stops on the wall clock; its fixed count
  (`n_shortcut = 80`) alone bounds it. Test: `test_shortcut_smoothing_does_not_stop_on_the_wall_clock`
  (a fake clock; fails on the old code). The A0 tilted insertion is undecided: no A0
  result is re-planned or reinterpreted.
- **Next:** review chat, then `/code-review`, then commit (everything uncommitted, with the
  chord check and shortcut change named as pipeline changes in the message). No paid calls yet.
- **Review #6 (chat, 2026-11-11):** the chord check is correct and covers every chord the
  search creates; checking the straight chord is sound (the tracker executes it). Done before
  commit: the insert grader (`eval_plan.py`) now checks the grasp phase with the fingers
  fully open, as the planners check them and execution commands them (it used the 64 mm
  spawn state, so it could miss finger collisions on the approach). This changes grading for
  every method graded by `eval_plan.py`.
- **Open item (not verified): the fully open approach on hardware.** The reviewer's condition:
  the finger state the planner checks must be what execution commands on the approach, in
  sim AND on hardware. Sim holds; this branch has no hardware code.
- **Iteration budget check:** pour A3 seed 2 with 5000 iterations still fails on all 8
  transport goals (each stopped by the 120 s cap at ~3100-3400 iterations), so 500 is not
  deciding that outcome; 500 stays.
- **Deferred (review #6, not MVP; none changes planning or scoring):**
  - handoff score tests: `tests/test_handoff.py` tests the old `task_plan.handoff_consistency`;
    the scorer uses `spec_scores.handoff_scores` (untested). Port the tests, delete the old
    function, report an empty subgoal ∩ path as its own outcome (the scorer drops it now);
  - plan-level check of the approach, lift and seams (built outside `plan_constrained`); the
    grader already checks them and found no hits so far;
  - TSR excess on the densified path (auxiliary only, never scored; the planner reports it at
    waypoints);
  - the pour grader does not check the approach at all;
  - `plan_pour_tea.py` writes no `.plan.report.json` (pour rows lack a grasp source);
  - sampler limits: closings measured on cylinders only (13 deg pad tilt on a flat face vs 21
    on a cylinder); contacts placed anywhere on the tilted face (restrict depth if tilt varies
    across it); `PadFace.normal` takes its sign from the pad center's x, not the side label
    (only matters for objects thin enough to put a pad near the midline);
  - an assertion that the carrying kinematics is unaffected by `set_gripper_joints`.
- **/code-review (high), 2026-11-11.** Fixed:
  - **The grasp lookahead used the open fingers** (bug from the fully open approach): it
    probed later stages' goals near the cup or mug with the fingers fully open, although
    they hold the object there. It now uses its own kinematics with the default finger
    state, as the carrying phase does (`kin_probe` in `plan_task.py`, `plan_pour_tea.py`).
    The trial outputs above predate this fix (they were planned with the open-finger probe).
  - `score_authoring_ablation.last_failure` prefers the FAILED line (a detail line with
    "Error" could misfile no_admitted_handoff as "other planning failure").
  - `plan_pour_tea.py`: an empty transport draw after earlier draws had goals is now
    no_admitted_handoff, not "no feasible goal configs".

  Deferred (not MVP):
  - `handoff_rejection` turns any exception into a rejection, which can hide a code bug
    (the reason is kept in the rejection histogram);
  - `handoff_of` is recomputed on every scoring run, uncached (slow for code variants);
  - the upright task reuses `insert_marker.json`'s `rest_up_axis`, so the transport
    profile's rest-up tilt reads ~90 deg for an upright carry (report-only, wrong);
  - the antipodal `closed_by_w` cache is keyed on exact float widths (never hits);
  - three copies of the handoff rejection loop, two of `HANDOFF_ROUNDS`;
  - a stale "start from a projection" warning branch in `plan_pour_tea.py`.

  Also note: the chord check (default on) applies to every `plan_constrained` caller,
  including `replan.py`, `omnimanip/planner.py` (the OmniManip baseline) and
  `plan_transport.py`. These have wall-clock budgets only, so their searches run slower.
