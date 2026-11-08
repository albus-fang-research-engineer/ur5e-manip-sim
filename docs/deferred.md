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
- Critic and repair get different action sets. The critic only sees members
  of the goal set, so its evidence can justify removing poses (tighten,
  re-pick, widen clearance) but never adding them; repair answers planner
  failures, where the specification is too tight, so it may loosen.
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
    evidence, because the planner may use any of P_k. Free rows: see O1.
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
  inside the critic. Our actual failure is a wrong point, which
  `reselect_point` covers.
- **Revisit when:** a preview or execution shows a missing row (the motion
  violates something the task obviously needs and no row bounds it). The
  path-boundary renders are where this shows first, if free rows are
  rendered (see O1).
- **Plan if revisited:** do not add a second row grammar. Re-run call #3 for
  that stage with its previous emission replayed and the critic's diagnosis
  appended, using the existing compile-retry path and its validation. Only if
  that proves unreliable, consider a structured `constrain_row` edit
  validated by call #3's own parser.
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

### O1. Free path rows in the path-boundary renders

- **What:** rendering a free path row at sample angles shows exactly the
  missing-row case (e.g. the teapot tipped over during the carry), but with
  D1 deferred the critic has no valid edit for it.
- **Options:** (a) pull D1's plan into the MVP: the critic may request a
  call #3 re-run for the stage with its diagnosis appended, through the
  existing compile-retry path (cheap; the loop already re-runs call #3 when a
  re-pick alone does not satisfy the next preview); or (b) render only rows
  that exist and defer free-row renders together with D1.
- **Leaning:** (a), since it reuses existing machinery and path checking is
  the point of these renders. Needs the user's decision.
- **Source:** doc review in "Langfold manipulation planning system",
  2026-11-08.

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
- **Plan:** step 3 of the order (VLM mode in `plan_stages_hw`) adds goal
  sampling on VLM-authored regions; next-stage ranking is part of it or is
  recorded here as still missing.
- **Source:** doc review 2026-11-08; checked in code 2026-11-08.

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

(none yet)
