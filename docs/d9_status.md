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

## Results (1 run per cell; old loop and old call #3 prompt unless noted)

Pour tea, pilot 3 (outputs/repair_study/pilot3), typed clearance/handoff cells
replaced by the upstream-first runs (pilot3_upstream, rule_check):

| Kind | Fault | typed | untyped | render_check |
|---|---|---|---|---|
| plausible | height contact | fixed (5) | not (6) | not (6) |
| plausible | height small | fixed (5) | not (6) | not (6) |
| plausible | wrong anchor, both stages | fixed (2) | fixed (2) | not (5), cannot by construction |
| cause upstream | wrong anchor, carry only | fixed 3/3 (2 each) | fixed (3) | not (5), cannot by construction |
| cause upstream | clearance contact | 2/3 with rule (0/1 without) | fixed (4) | not (7) |
| cause upstream | handoff | 2/3 with rule (0/1 without) | not (5) | not (7) |
| control | contradictory rows | fixed (4) | fixed (3) | not (6) |

Insert marker, pilot 2 (outputs/repair_study/pilot2): wrong anchor typed fixed (2),
untyped fixed (3), render_check not (6, cannot by construction); contradictory rows
and handoff (compile-time) fixed by all three conditions in 3 calls.

Spend so far ~$33 (pilot 1 $4.06, pilot 2 $2.89, controls ~$0.70, probes ~$0.10,
pilot 3 $11.25, rule reruns $5.43, etc.).

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
2. Scale-up (3 repeats of the qualified pour faults x 3 conditions) on hold until the
   user decides; it would use the current loop (upstream-first + new call #3 prompt).
3. docs/deferred.md D9 entry to update with the decisions above.
4. Paper (extended abstract): H5 as a hypothesis only; not edited (user on hold).
