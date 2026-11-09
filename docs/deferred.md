# Deferred, open and rejected design items

Record of design items that were deliberately left out of the current work,
with enough context to pick them up later without the original chats. Add an
entry whenever something is deferred; move it to "Closed" (with the commit)
when it is done or dropped for good. Rejected ideas stay listed so they are
not re-proposed without new evidence.

Entry format: status, what, where, why deferred, revisit when, plan if
revisited, paper impact, source.

Line references are as of sim commit b133912 (`insert-marker`), paper
commit 2341607 (`~/papers/langfold`) and hardware commit d59f71b
(`vlm-port`). Re-find them by the quoted text if the files have moved.

## Context: the critic/repair work (branch `critic-repair`)

Started 2026-11-07 for the UR5e pour-tea port (`ur5e-manip-hardware`, branch
`vlm-port`). On the 11/05 hardware run (`outputs/runs/20261105_035449`),
call #2 anchored the carry stage (`transport_active`) on
`teapot.handle_center` ("Lifting the teapot is anchored at the handle"),
picked independently of the grasp's pick (call #2 runs once per role; see
`call2/selections.json`). Call #3 cannot offset from the selected point: the
end-effector transform pins it (`manip_sim/compile_tsr.py:784`), translation
terms anchor only on passive (mug) points, and the teapot has no grounded
quantities. A replay with a sim stand-in emission put the handle over the
mug: spout 14.2 cm from the rim center, teapot base 2.9 cm below the rim.

The chosen fix is the paper's own mechanism rather than a prompt change:
wire VLM call 4 (`critique_preview`) and call 5 (`repair`), which exist in
`manip_sim/vlm.py` only as prompt builders, parsers and client methods
(commit e4c75b6, 2026-08-15), and let a critic re-pick the point
(`reselect_point`). Decisions so far (Claude Code session with the user,
reviewed in the claude.ai chat "Langfold manipulation planning system",
2026-11-07):

- Calls 4 and 5 share the edit grammar, the vocabulary-checked parser and the
  code that applies edits; they keep separate prompts so each can be ablated.
- Call 4 runs once per task, not per stage. Stages depend on each other in
  two directions. Forward: stage k+1 starts where k ends; recompiling k..K
  after an edit (deterministic, no VLM) handles that. Backward: stage k's goal
  is right only if it serves k+1 (the carry must set up the pour), which a
  stage-by-stage critic cannot judge because k+1 is not authored yet. Our
  failure is the backward kind.
- Critic and repair get different action sets (`manip_sim/critic.py`,
  `ACTION_TABLE`). The critic only sees members of the goal set, so its
  evidence can justify removing poses but never adding them; repair answers
  planner failures, where the specification is too tight, so it may loosen.
  - Critic: tighten_tolerance, tighten_slack, reselect_point,
    reselect_axis, reemit_stage.
  - Repair: relax_tolerance, free_dof, widen_slack, widen_clearance,
    reselect_point, reselect_axis, reemit_stage.
  - `widen_clearance` is repair-only, although it was first planned for the
    critic: in the compiler (`compile_tsr.py:793-796`) a clearance shifts the
    above/below band to [off+clr, off+clr+band] without narrowing it, so it
    moves poses rather than only removing them.
- Edit targets use the emission's stage name, the same slot names
  `CompileError` reports (e.g. `carry teapot above mug.subgoal.rot[0]`);
  `<stage name>.point.<object>` targets a stage's point choice. Only the
  active object's point can be re-picked: the passive object's point roots
  the frame w for several stages.
- Call 5 gets the measurement report and images (images behind a flag). Its
  output stays a symbolic edit.
- The VLM never emits a number; magnitude changes are one enum step.
- A verdict may carry several edits (one per target), applied all-or-none,
  recompiled once, then always previewed again. One round budget of 2 rounds
  spans calls 4 and 5 together. Neither call may reverse the other's edit on
  the same target (a row, or a stage's point choice): e.g. repair may not
  re-pick the handle after the critic moved the anchor to the spout. A
  reject when the budget is spent returns the typed failure
  `preview_rejected` and stops; it is not handed to call 5.
- `reselect_point` names a grounded point, mapped to call #2's candidate mark
  and applied through call #2's own binding code (one selection path). A
  named point with no candidate mark in that stage's pool, or on another
  object, is a typed refusal handled like a parse error (re-query within the
  parse-retry budget), never a silent no-op.
- Swapping the point changes only the position of the end-effector
  transform; its rotation comes from call #3's rows on canonical directions
  (`compile_tsr.py:784`), so call #3 is not re-run at first.
- All new behavior is behind opt-in flags, off by default; prompts are
  byte-identical when off.
- Sim changes: the rule against modifying the sim during hardware work means
  no unverified changes on a stable branch (`insert-marker`, which hardware
  mounts read-only at /opt/manip-sim and imports through its `*_compat.py`
  modules, including `vlm.py`). Work happens on this branch in the worktree
  `~/ur5e-manip-sim-critic`; merge to `insert-marker` only after the sim suite
  passes and the hardware suite passes with `SIM_DIR` pointed at this branch
  (a past unverified sim patch broke hardware call #2 through the compat
  import).
- The critic checks the specification, both the goal sets and the path
  sets, not individual configurations. Before planning there is no path to
  render, but CBiRRT keeps every path inside P_k, so "is the path right?"
  reduces to "is P_k right?".
- Evidence per task:
  - One image per state: the initial scene, then each stage's goal, iso and
    top panels side by side, grounded points labeled. Each goal is drawn at
    its subgoal nominal, because the compiler builds stage k+1 against stage
    k's `subgoal.nominal()`; drawing the top-ranked IK goal instead would
    show stage k+1 starting somewhere its constraints were not compiled
    from. The report gives the top-ranked IK goal's offset from the nominal
    where goal ranking exists (see O3).
  - The path rows as text, with their tolerance tokens.
  - Path-boundary renders for each existing path row: vary one row at a time
    to its bound, keep the other rows at nominal, label the row and its
    token. Never joint corners, which would bring back R4's bias toward
    tightening loose-but-fine rows. For paths the boundary is the honest
    evidence, because the planner may use any of P_k.
  - Free path rows are rendered too, at sample angles (decided 2026-11-08,
    formerly open item O1). A free row is what a missing constraint looks
    like (e.g. the teapot tipped over mid-carry). The critic's edit for it is
    `reemit_stage`: re-run call #3 for that stage with its previous emission
    replayed and the critic's diagnosis appended, through the existing
    compile-retry path, so call #3's parser and the compiler validate the new
    row. The same re-run is the fallback when a re-pick alone does not
    satisfy the next preview. Cost: 1 call #3 call per re-emitted stage.
  - The spread of IK-feasible goals as text (range of each labeled point's
    offset across feasible goals), until D5 adds renders.
- The grasp-goal state draws the Robotiq 2F-85 (the gripper on the UR5e),
  placed with the same TCP transform IK uses (the pad_offset chain,
  `manip_sim/task_plan.py:41`); a wrong offset would draw the grasp 15-20 cm
  off and mislead the critic. One gripper source per repo: the sim uses
  robosuite's Robotiq85 (`manip_sim/grasping.py:75`); hardware uses its STL
  meshes (`ur5e-manip-hardware/test/curobo_incontainer/robotiq/`,
  `robotiq_2f85_frozen.urdf`), since MuJoCo cannot load the DAE visuals. The
  sim must not depend on hardware assets.
- The grasp is context, not editable: its TSR is hand-written in the sim and
  taught on hardware. The critic prompt says so, so no edits are spent on it.
- No separate calibration step (see Rejected). The first live run on 11/05 is
  judged on two verdicts: reject on the handle-anchored carry, accept after
  the re-pick.

## Deferred

### D1. `constrain_row`: critic action that adds an omitted constraint row

- **Status:** deferred (not in the MVP).
- **What:** a critic edit that adds a rotation (or translation) row call #3
  left out, e.g. no path row keeps the teapot upright during the carry.
- **Where:** would extend the critic action set (today `REPAIR_ACTIONS`,
  `manip_sim/vlm.py:170`) and `parse_critic` (`vlm.py:959`).
- **Why deferred:** it is not a one-token edit. A row needs a relation, two
  signed directions and a tolerance, which duplicates call #3's row grammar
  inside the critic (a second parser and validation path to keep in sync).
  Our actual failure is a wrong point, which `reselect_point` covers, and the
  missing-row case is covered in the MVP by `reemit_stage` (see the
  decisions above), which reuses call #3's own grammar and validation.
- **What `reemit_stage` lacks, i.e. why `constrain_row` may come back:**
  re-emitting regenerates the whole stage, so call #3 may also change rows
  that were fine (or drop the selected point's intent), and each re-emit is
  a full call #3 call. `constrain_row` would be surgical: one added row,
  everything else untouched, no call #3 call.
- **Revisit when:** `reemit_stage` repairs of missing rows fail to add the
  row, or change rows the critic did not flag, in the runs logged under D6;
  or their cost matters.
- **Plan if revisited:** a structured `constrain_row` edit whose row is
  validated by call #3's own row parser (no second grammar), applied to the
  emission and recompiled like any other edit. The free-row renders and the
  critic's diagnosis already in place are its inputs, so nothing built for
  `reemit_stage` is thrown away.
- **Paper impact:** the methodology's example of a critic edit is exactly this
  case ("such as a rotation row the specification omitted",
  `~/papers/langfold/src/methodology.tex:140`, `ea/methodology.tex:129`).
  Replace it with the re-picked point (also the sharper contrast with
  OmniManip, whose RRC never re-picks the interaction point). See D7.
- **Source:** Claude Code session 2026-11-07; review in "Langfold manipulation
  planning system".

### D2. `insert_stage` and `reorder_stages`

- **Status:** deferred; out of both action sets for now.
- **What:** repair actions that change call #1's stage plan.
- **Where:** `REPAIR_ACTIONS`, `manip_sim/vlm.py:170-171`.
- **Why deferred:** they restructure the task plan (call #1's job), which the
  critic prompt already puts out of scope ("Edits must target the
  constraint-authoring layer only"). Nothing so far needs them, and applying
  them means re-running calls 2 and 3 for the affected stages.
- **Revisit when:** a failure is clearly a missing or misordered stage (e.g. a
  carry that cannot reach the pour goal without an intermediate lift).
- **Plan if revisited:** route to a call #1 re-query with the failure record,
  not a token edit; then re-run calls 2 and 3 for the changed stages.
- **Paper impact:** none unless used.
- **Source:** review in "Langfold manipulation planning system", 2026-11-07.

### D3. Finer tolerance levels

- **Status:** deferred.
- **What:** more levels between the current enum steps.
- **Where:** `manip_sim/compile_tsr.py:147-171` (rotation 5 / 15 / 30 deg;
  clearance 0 / 1 / 3 cm / large; slack and centered tolerances) and the
  matching token tuples in `manip_sim/vlm.py:107-109`.
- **Why deferred:** steps are coarse (each rotation step is 2-3x), but no
  failure has needed an in-between value yet.
- **Revisit when:** a repair has to jump a full step and the result is
  clearly too loose (or a tighten makes the goal set empty).
- **Plan if revisited:** add levels to the shared vocabulary so call #3 can
  author them too. Never add repair-only intervals, and never let code
  compute a "minimal relaxation" (see Rejected R3).
- **Paper impact:** none beyond the token table.
- **Source:** Claude Code session 2026-11-07; review in "Langfold manipulation
  planning system".

### D4. Fallback: reword call #2's question for stages with a passive object

- **Status:** deferred; kept as a flagged fallback.
- **What:** ask call #2 "which point on the teapot must end at the goal
  relative to the mug?" instead of "select the interaction point" for stages
  with a passive object. One sentence, no schema change.
- **Where:** call #2 prompt builder in `manip_sim/vlm.py`
  (`select_point_axis`).
- **Why deferred:** it changes an existing prompt for every task, and call 4
  with `reselect_point` fixes the failure without touching it. Call #1 already
  listed the spout first in the carry's parts and call #2 still chose the
  handle, so extra information alone may not help.
- **Revisit when:** the critic re-picks the carry point in at least half of
  the runs logged under D6. The repair loop must not become the normal path
  (a paid step repairing a systematic error); then the error belongs in
  call #2.
- **Plan if revisited:** add it behind its own flag with a new
  `PROMPT_DELTAS` entry; re-run call #2 on 11/05 (4 calls) and a sim
  regression (3-4 calls), with approval.
- **Paper impact:** none if kept behind a flag for an ablation.
- **Source:** Claude Code session 2026-11-06/07; handoff in "UR5e teapot pour
  carry handle issue".

### D5. Showing the spread of feasible goals in the preview

- **Status:** deferred (MVP shows the top-ranked goal only).
- **What:** render several IK-feasible goals, not just the one that will
  execute, so the critic sees how much the goal region allows.
- **Where:** preview renderer (to be built on `render_base`,
  `scripts/render_stage_frames.py:594`).
- **Why deferred:** overlapping translucent copies may be hard to read, and
  extra images cost tokens. Samples drawn from the TSR box before IK were
  rejected (R4).
- **Revisit when:** the critic accepts a region whose spread is wrong (e.g. a
  free yaw that lets the spout point away from the mug) because it saw only
  the nominal goal.
- **Plan if revisited:** farthest point sampling among IK-feasible goals,
  distance = world displacement of the labeled points (spout, handle, body
  center); show up to 3 besides the top-ranked goal; skip if the labeled
  points spread less than about 1 cm. Compare translucent overlay vs. a small
  montage offline before any paid call.
- **Paper impact:** none.
- **Source:** Claude Code session 2026-11-07; review in "Langfold manipulation
  planning system".

### D6. Labeled evaluation set for the critic

- **Status:** deferred to after the MVP.
- **What:** preview images from real runs with a human accept/reject label,
  used to score the critic.
- **Why deferred:** per-scene calibration is infeasible on unseen scenes
  (R1), so the only way to know how often the critic is right is to measure
  it across many runs. That needs runs first.
- **Revisit when:** the hardware loop runs end to end.
- **Plan if revisited:** save each run's preview images, report, verdict and
  edits; the user labels them; re-score on prompt or model changes (1 paid
  call per case). Also log per run: rounds to accept, edits applied, and the
  labeled points' offsets before and after (spout-to-rim distance for pour
  tea; keep task-specific metrics out of the pipeline report).
- **Paper impact:** gives a number for the critic's reliability and for the
  claim that typed edits beat label-only verdicts (OmniManip's RRC returns
  only success / failure / refinement; we saw label-only recovery fail in our
  own OmniManip testing).
- **Source:** Claude Code session 2026-11-07/08.

### D7. Paper wording changes

- **Status:** pending the user's approval; do not edit or push
  `~/papers/langfold` (Overleaf clone) without asking.
- **What and where:**
  - Algorithm 1 runs call 4 inside the per-stage loop
    (`src/algorithm.tex:25`, also `ea/algorithm.tex`); the design now runs it
    once per task, before planning.
  - "returns as a token edit to the stage's emission"
    (`src/methodology.tex:139-140`, `ea/methodology.tex:128-129`):
    `reselect_point` edits call #2's selection, so this becomes "a token edit
    to the specification" (selections plus emission).
  - Example "a rotation row the specification omitted" -> the re-picked point
    (D1).
  - "every VLM call returns only selections and tokens" (`src/algorithm.tex:6`,
    `ea/algorithm.tex:6`): the critic's diagnosis is free text and feeds a
    call #3 re-run, so narrow it to "everything that reaches the compiler is a
    selection or a token".
  - Call 5 as "a symbolic repair request" (`src/methodology.tex:151`,
    introduction) stays true with images: the output is symbolic.
  - Optionally state the principle behind the split action sets: the critic's
    evidence can only justify removing poses.
  - F_c's orientation: the methodology says call #2's anchor, with the axes
    from the part fits, defines frame F_c. In the compiler the end-effector
    transform takes only its position from the selected point; its rotation
    comes from call #3's rows on canonical directions
    (`manip_sim/compile_tsr.py:784`). Check whether the part-fit axes enter
    anywhere else; if not, the paper overstates their role.
- **Revisit when:** the critic loop works on hardware, before the next paper
  revision.
- **Source:** review in "Langfold manipulation planning system", 2026-11-07.

### D8. A true spout-tip point on hardware

- **Status:** deferred.
- **What:** hardware grounding has `spout_center`, not a spout tip. The sim's
  `spout_tip` was hand-authored.
- **Where:** hardware part grounding (`ur5e-manip-hardware`, grounding_vlm
  outputs); 11/05 teapot points: base_center, body_center, handle_center,
  lid_center, spout_center.
- **Why deferred:** `spout_center` is close enough to test the critic loop;
  a tip needs a grounding change (an extreme point along the spout's fitted
  axis).
- **Revisit when:** pours miss the mug because the stream leaves from the tip,
  not the spout's center.
- **Plan if revisited:** derive the tip as the spout part's farthest point
  along its fitted axis, away from the body; add it to the candidate pool so
  call #2 and the critic can name it.
- **Paper impact:** none.
- **Source:** Claude Code session 2026-11-06.

## Open design items (need a decision before or during the MVP)

### O2. Pour compile when the mug faces away from the teapot

- **What:** on 11/05 the mug's front (its handle direction) is about 18 deg off
  the teapot's facing direction, so the pour's tilt-axis row does not land on
  a w basis vector and fails the alignment gate (`ALIGN_TOL_RAD` = 10 deg,
  `manip_sim/compile_tsr.py:174`). Sim scenes always align them.
- **Why open, not deferred:** random placement on real scenes will rarely put
  the mug square to the teapot, so the next hardware run will likely hit it,
  and every misaligned scene spends paid compile retries. Placing the mug
  square on captures is a setup guard, not a general fix.
- **Candidate fix:** let the passive frame w take its yaw from the teapot's
  approach direction instead of the mug's front when the mug's front is
  arbitrary (rotationally symmetric body), or treat the mug as front-less
  (hardware already reports front as None for such objects).
- **Paper impact:** possibly the canonicalization description.
- **Source:** Claude Code session 2026-11-07; doc review 2026-11-08.

### O3. Goal ranking on the hardware path

- **What:** the design previews and plans to goals sampled from G_k within
  P_k, IK-checked and ranked by next-stage feasibility. On hardware,
  `outputs/runs/plan_stages_hw.py` (header, lines 17-18) runs hand-written
  regions only: the cuRobo sidecar walks its own ranked goals with no pour
  lookahead, and a pour failure re-plans the carry with the next seed
  instead. There is no VLM mode yet.
- **Needed for:** the critic report's "top-ranked IK goal" offset and the
  call 5 failure records on hardware.
- **Progress:** hardware commit 952f64e adds `plan_stages_hw.py --method
  vlm` (compile at the actual entry, sidecar goal sampling, typed failure
  records in report.json). Live check on 11/05 with the sim stand-in
  emissions: the handle-anchored carry gets 858 IK solutions and 0
  collision-free goals, recorded as `no_feasible_ik`. Next-stage ranking is
  still missing. The record does not name the colliding body; the sidecar
  funnel gives counts only.
- **Plan:** step 3 of the order (VLM mode in `plan_stages_hw`) adds goal
  sampling on VLM-authored regions; next-stage ranking is part of it or is
  recorded here as still missing.
- **Source:** doc review 2026-11-08; checked in code 2026-11-08.

### O4. 11/05 `teapot.lid_center` grounded at the base's height

- **What:** in hardware run 20261105_035449 the grounded `teapot.lid_center`
  sits at about the same height as `teapot.base_center`, not at the top of
  the teapot. In the teapot's body frame (preview inputs `labels`): lid z
  -0.0318 m, base z -0.0308 m, body_center z +0.0043 m. The critic report
  shows the same at the carry goal: round 1 points_at_goal height lid
  -0.0321 m, base -0.0317 m; round 2 lid -0.0043 m, base -0.0039 m.
- **Why it matters:** it looks like a grounding error in that run's points
  (the lid's candidate resolved to a bottom point), unrelated to the critic.
  A critic or call #2 decision that names `lid_center` would act on a wrong
  location, and the label on the preview images points at the base.
- **Trigger to revisit:** any critic edit or call #2 selection on a
  hardware run that uses `lid_center`; at that point check the grounding
  candidates for the lid (`grounding_vlm/canon/lib_teapot/candidates.json`)
  before trusting it.
- **Source:** coordinator note 2026-11-08, from the 11/05 preview renders;
  numbers checked against the round 1/2 `report.json` and
  `preview_inputs.json`.

## Proposals awaiting critic evidence (P-items)

Decided 2026-11-08: stop changing the critic's input on guesses about how the
critic will behave, and run the real critic first. These proposals are logged
here, not built. Each is a hypothesis; the trigger says what in the critic's
logged output (its "expected" / "observed" text, verdicts and edits) would
justify building it. Source: the step-5 review in the "UR5e hardware
deployment review" chat (2026-11-08) unless noted.

- **P1. Collision paragraph is too permissive.** "That is the planner's job"
  and "only 'no collision-free pose' is evidence" may make the critic accept
  a spec whose nominal puts one part inside another, once the nearest free
  pose hides it. Proposed rewrite: a collision alone is not evidence; judge
  whether the nominal expresses the task; if fixing "no collision-free pose"
  needs loosening, accept and state the diagnosis (call 5 loosens). Trigger:
  the critic accepts a hand-built negative case (D-case: spout inside the
  mug) or cites "planner's job" for a wrong anchor.
- **P2. Draw the nominal faded beside the substituted pose,** captioned with
  the shift and the share of each row's band it used. Trigger: the critic
  misjudges a goal whose drawn pose was substituted.
- **P3. Second path center near the goal for translation rows** (rotation
  rows keep the midpoint); midpoint by translation interpolation + slerp.
  Trigger: the critic's path observations show it can't judge translation
  bounds relative to the passive object, or misreads the pour tilt panels.
- **P4. Free goal heading.** With yaw free, the drawn heading is arbitrary.
  Caption "heading about <passive> up is free" plus 2-3 sample headings, or
  render the goal cuRobo's funnel picks. Trigger: the critic judges alignment
  or reselects an axis from the drawn heading.
- **P5. Exempt resting table contact of objects at their start pose**
  (registration tilt can read 3-5 mm into the table). Trigger: an initial or
  grasp image carries a table-collision note.
- **P6. Mesh face-orientation check** for the collision test: one known
  colliding and one known free pose per library mesh. Trigger: any collision
  note that disagrees with the image or with cuRobo.
- **P7. The nearest-free search samples the TSR box before IK.** Related to
  R4 (rejected for spread previews). Here it only picks one substitute pose
  near the nominal, but it is still pre-IK, unlike the earlier agreement
  ("render the top-ranked IK-feasible goal"). Trigger: a substituted pose
  that cuRobo finds unreachable, or the critic reacting to one.
- **P8. O2 fix location.** Reviewer: fix planner-side (rank carry goals by
  pour feasibility, as sim plan_pour_tea's lookahead). Ours: compiler-side
  (take w's heading from the active object when the tilt axis misses w), so
  the preview compiles the pour too. Trigger: the critic spends a round on
  the pour's "compile rejected" note.
- **P9. Write down loop rules:** the final check only accepts or rejects
  (edits logged, not applied); parse retries capped; stages chain from
  subgoal.nominal(), not the drawn pose; critic_compat is temporary until the
  merge; paper (D7): call 4 acts as a collision gate through the VLM. Docs
  only; do with the commit after the first real critic run.

## Rejected (do not re-propose without new evidence)

- **R1. Per-scene calibration of the critic.** Infeasible on a new random
  scene: nothing known to compare against. Replaced by D6.
- **R2. One shared prompt for calls 4 and 5.** Inputs and action sets differ,
  so the prompt would be shared in name only, and separate prompts keep the
  call 4 / call 5 ablations clean. The grammar, parser and applier are shared.
- **R3. Code-computed "minimal relaxation" from the failure record.** Produces
  an interval no token describes, which breaks the closed-vocabulary claim as
  much as the VLM emitting a number. Use enum steps (D3).
- **R4. Preview samples drawn from the TSR box before IK.** Shows poses that
  never execute, and box corners look least like the intended pose, which
  biases the critic toward rejecting correct specifications and feeds
  tighten/loosen ping-pong. Preview IK-feasible goals instead.
- **R5. OmniManip-style label-only verdicts.** A bare success / failure /
  refinement leaves recovery blind ("try the next candidate"); we observed
  this in our OmniManip testing. Every reject carries typed edits and a
  diagnosis.

## Closed

- **O1. Free path rows in the path-boundary renders** -- decided 2026-11-08:
  render them, with `reemit_stage` as the critic's edit (see the decisions
  above and D1). The alternative, rendering only bounded rows, would leave a
  missing path constraint invisible until execution, since the planner
  satisfies a specification with no such row.
