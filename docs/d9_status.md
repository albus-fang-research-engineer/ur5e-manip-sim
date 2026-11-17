# D9 repair study: status and handoff (2026-11-15)

Typed repair (Langfold) vs OmniManip-style render-and-check, in sim. Design record:
docs/deferred.md D9 (its text predates the decisions below; update it when the study
is written up). Branch `worktree-d9-omnimanip` (worktree
`.claude/worktrees/d9-omnimanip`), pushed. Do not edit `~/ur5e-manip-sim-critic`
(hardware session) or `~/ur5e-authoring` (authoring ablation).

## Decisions so far

- Conditions: `typed` (Langfold: critic call #4 on the preview with its typed facts;
  planner failures to repair call #5 as a typed record) and `render_check`
  (OmniManip-style, Sec. 3.3 / Alg. 1: label per stage; refine rewrites near the
  previous answer once per stage, failure rewrites from the prompt; a planner failure
  is a bare "failure" and every editable stage is rewritten) are required; `untyped`
  (Langfold with every failure detail withheld) is strongly recommended.
- Framing: "Langfold's typed repair vs OmniManip's recovery strategy, transplanted
  into Langfold", never "vs OmniManip". Port deviations: reconstructed prompt (the
  published RRC.json is 404), our renders (no arrow), one model for all conditions,
  fresh call #3 instead of a ranked direction list, interaction_value dropped,
  "Think Step By Step" renamed "Assessment" (the API refuses it: stop_reason refusal,
  category reasoning_extraction).
- Same for every condition: claude-opus-5, temperature omitted (API default 1.0),
  round budget 2 (rounds that change the spec), hard call cap 8 requests per case
  (a call starts only if all its parse retries fit), chord-checked planner.
- Fixed = the final spec passes the grader in a majority of planner seeds 0-2;
  stopping early is not fixed. Calls per fix = all calls / runs fixed.
- Starting specs must plan and pass the grader AND be accepted by a control critic
  call (pilot 1: the critic spent its rounds on the starting spec's own flaws).
  tasks/repair_study/start/{insert_marker,pour_tea}: both qualified, both controls
  accept.
- Faults: one edit (or one conceptual edit across stages, `ctx: both` / `also`),
  kept only if the faulty spec plans on 0 of 3 seeds. Kinds: compile (fails at
  compile time: fixes about equally everywhere), magnitude, plausible (compiles and
  renders fine, only the planner sees it). Representative task: pour tea.
- Unreachable goals and "no path after goals were found" are not in the study:
  they are reported to the user, not repaired.
- A goal region with 0 collision-free goals after the planner's resampling (up to
  480) is treated as in collision and re-proposed (no penetration analysis).
- Upstream-first rule (merged 3d83bee): a stage that failed to compile only because
  an earlier stage was wrong keeps its original rows when it compiles after the
  earlier fix, instead of being rewritten by call #3.
- call #3 prompt: authoring-ablation's word ranges + no "loosest word" rule
  cherry-picked (0d9e899, 742fc7b). The compiler's 5 deg entry-attitude allowance and
  the grader's 25 deg pour spout limit (06b9991 etc. on authoring-ablation) are NOT
  in D9 yet: a separate step, then re-qualify specs and faults (free).
- Paid runs need the user's approval with a call count and a USD estimate
  (~$0.10 per request, ~$0.35-0.45 per pour case at claude-opus-5 list price).

## Results

### Scale-up sets 1-2 (2026-11-15): current loop for every condition

Pour tea, upstream-first rule and the new call #3 prompt for every condition.
Set 1: 5 shared faults x 3 conditions (outputs/repair_study/scale/set1, $7.99).
Set 2a: the 2 Langfold-only faults x 3 conditions (also scale/set1). Set 2b: a
second run of the 5 shared faults (scale/set2). Sets 2a + 2b: 21 runs, $11.06
(first reported as $11.71: that sum counted re-emit calls twice).
Fixed = runs fixed / runs; calls = VLM calls per run, on average.

| Fault | Type | Langfold fixed | Langfold calls | Withheld fixed | Withheld calls | OmniManip-style fixed | OmniManip-style calls |
|---|---|---|---|---|---|---|---|
| wrong anchor, both stages | shared | 2/2 | 2 | 2/2 | 2 | 0/2 | 5.5 |
| wrong anchor, carry only | shared | 2/2 | 2 | 2/2 | 2 | 0/2 | 5 |
| height contact | shared | 1/2 | 4.5 | 0/2 | 5 | 0/2 | 6 |
| height small | shared | 2/2 | 5 | 0/2 | 6 | 0/2 | 6 |
| clearance contact | shared | 1/2 | 3.5 | 0/2 | 4 | 0/2 | 6.5 |
| **Shared total** | | **8/10** | **3.4** | **4/10** | **3.8** | **0/10** | **5.8** |
| handoff | Langfold-only | 0/1 | 5 | 0/1 | 4 | 0/1 | 5 |
| contradictory rows | Langfold-only | 0/1 | 4 | 0/1 | 4 | 0/1 | 3 |
| **All faults** | | **8/12** | **3.6** | **4/12** | **3.8** | **0/12** | **5.5** |

Calls per fixed run: Langfold 5.4, withheld 11.5, OmniManip-style none fixed.

Langfold's losses share one pattern: a round went to re-tightening the pour after
call #3 rewrote it, not to the fault (round budget 2).
- Height contact (set 1): the fault spans the carry and the pour (one height for
  both), so both stages needed rewriting, one per round; the pour rewrite came out
  looser and the last round went to tightening it.
- Contradictory rows: round 1 fixed the fault; round 2 tightened the pour path
  unprompted; accepted, but passed the grader on seed 0 only (seed 1 goal check
  failed, seed 2 no path).
- Handoff: round 1 fixed the carry, round 2 tightened the pour; round 3 saw that the
  pour still started from the high carry height, with the budget spent.
Possibly a side effect of the new call #3 prompt (word ranges); the pilot (old
prompt) fixed handoff 2/3 and contradictory rows 1/1. One run each: a signal.

### Critic memory check (2026-11-16, branch d9-critic-memory)

Langfold (typed) with `--critic-memory` (vlm.critic_review_text from the hardware
session's critic-memory branch: round n of 3, all edits at once, edit only rows
with a nameable failing pose, earlier verdicts and what changed; sim loop: rounds
from the shared ledger, no planner text in the history). 7 pour faults x 1 run
(outputs/repair_study/memory/faults), $3.32; controls on the unfaulted start spec,
one critic call each, 3 with and 3 without memory (memory/control), $1.59.

| Fault | No memory fixed | No memory calls | Memory fixed | Memory calls |
|---|---|---|---|---|
| wrong anchor, both stages | 2/2 | 2 | 1/1 | 2 |
| wrong anchor, carry only | 2/2 | 2 | 1/1 | 2 |
| height contact | 1/2 | 4.5 | 1/1 | 5 |
| height small | 2/2 | 5 | 1/1 | 5 |
| clearance contact | 1/2 | 3.5 | 1/1 | 4 |
| handoff | 0/1 | 5 | 1/1 | 5 |
| contradictory rows | 0/1 | 4 | 0/1 | 3 |
| **All** | **8/12** | **3.6** | **6/7** | **3.7** |

- False accepts (the critic accepted, then the spec failed the planner or grader):
  no memory 1 of 9 accepts (contradictory rows, last round); memory 1 of 7 (the same
  fault, round 2). Last-round accepts with memory: 3 of 3 then passed.
- Contradictory rows fails the same way with and without memory (seed 1 goal check,
  seed 2 no path on the unchanged carry): call #3's rewritten pour, not the critic.
- False rejects on the correct spec: 0 of 3 with memory, 0 of 3 without (all 6
  accepted and passed).
- Memory turned the round-3 rejects that spent the budget (handoff, height contact)
  into round-3 accepts that passed. One run per fault: a signal, not a rate.

### Earlier results (mixed versions)

Pour tea, final results so far (pilot 3 + the upstream-first runs; mostly single
runs, temperature 1.0: signals, not rates). Fixed = runs fixed / runs; calls = VLM
calls per run, on average. "Shared" = a mistake either pipeline can make (OmniManip
also picks a point and predicts a distance); "Langfold-only" = exists only because
Langfold's specifications are richer (path regions, several rows).

| Fault | Type | Langfold fixed | Langfold calls | Withheld fixed | Withheld calls | OmniManip-style fixed | OmniManip-style calls |
|---|---|---|---|---|---|---|---|
| wrong anchor, both stages | shared | 1/1 | 2 | 1/1 | 2 | 0/1 | 5 |
| wrong anchor, carry only | shared | 3/3 | 2 | 1/1 | 3 | 0/1 | 5 |
| height contact | shared | 1/1 | 5 | 0/1 | 6 | 0/1 | 6 |
| height small | shared | 1/1 | 5 | 0/1 | 6 | 0/1 | 6 |
| clearance contact | shared | 2/3 | 4.3 | 1/1 | 4 | 0/1 | 7 |
| **Shared total** | | **8/9** | **3.6** | **3/5** | **4.2** | **0/5** | **5.8** |
| handoff | Langfold-only | 2/3 | 4.7 | 0/1 | 5 | 0/1 | 7 |
| contradictory rows | Langfold-only | 1/1 | 4 | 1/1 | 3 | 0/1 | 6 |
| **All faults** | | **11/13** | **3.8** | **4/7** | **4.1** | **0/7** | **6.0** |

Calls per fixed case: Langfold ~4.5, withheld ~7.3, OmniManip-style none fixed.
Langfold cells use the loop with the upstream-first rule; withheld and
OmniManip-style ran on the earlier loop (the rule cannot affect OmniManip-style; it
could help withheld on clearance contact and handoff). All of these used the old
call #3 prompt (before the word-ranges cherry-pick). Records: outputs/repair_study/
pilot3, pilot3_upstream, rule_check.

Insert marker, pilot 2 (outputs/repair_study/pilot2): wrong anchor typed fixed (2),
untyped fixed (3), render_check not (6, cannot by construction); contradictory rows
and handoff (compile-time) fixed by all three conditions in 3 calls.

Spend so far ~$57 (pilot 1 $4.06, pilot 2 $2.89, controls ~$0.70, probes ~$0.10,
pilot 3 $11.25, rule reruns $5.43, scale-up set 1 $7.99, sets 2a + 2b $11.06,
critic memory check $4.91, etc.).

## How to run

- Docker image `manip-sim:latest`; copy the git-ignored meshes into a new worktree
  (assets/objects/<obj>/meshes from ~/ur5e-manip-sim).
- One case: `scripts/run_repair_case.py --task pour_tea --selections S --emissions E
  --fault F --condition typed|render_check|untyped --out OUT --mode dry_run|replay|live`.
- A batch: `PILOT_TASK=pour_tea PILOT_SEL=.. PILOT_EM=.. PILOT_FAULTS=".." PILOT_CONDITIONS=".."
  PILOT_JOBS=6 scripts/run_repair_pilot.sh OUT` (resumable; finished cases skipped).
- Score: `manip_sim.repair_study.summarize/table_md` over the record.json files.
- The API key is loaded from ~/ur5e-manip-hardware/.env into the environment only.

## Open

1. Grader step: pour spout limit 25 deg (and the compiler's 5 deg allowance?) into D9,
   then re-qualify the starting specs and faults.
2. Scale-up bit by bit on the current loop (upstream-first + new call #3 prompt), so
   every condition runs on one version. Done: set 1 (5 shared faults x 3 conditions)
   and sets 2a + 2b (2 Langfold-only faults; second run of the shared faults).
   Open: whether call #3's pour rewrites cause Langfold's extra tightening rounds.
3. docs/deferred.md D9 entry to update with the decisions above.
4. Paper (extended abstract): H5 as a hypothesis only; not edited (user on hold).
