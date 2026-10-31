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
to the whole transport. All three start from one grasp, so grasp
selection is not part of the test. Results are over seeds 0-4 at 0.50 m
separation. Physics reuses `execute_task_plan.py` with no planner-specific code.
"Tilt" under Transport is the marker's resting up axis from vertical; it must reach
about 90° by the end, so the rows show how early it turns.

<!-- comparison:start -->
|  | Ours, 5°, same grasp | Ours, 1°, same grasp | Ours, 10°, same grasp | ReKep | OmniManip |
|---|---|---|---|---|---|
| Plans found | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Meets the goal spec | 5/5 | 5/5 | 1/5 | 5/5 | 5/5 |
| **Goal deviation:** final tilt | 3.4–6.3° | 0.5–1.1° | 6.6–12.6° | 0.6–1.1° | 0.1° |
| Lower end off the cup's axis | 8.8–15.8 mm | 11.9–14.6 mm | 10.2–16.2 mm | 0.2–0.9 mm | 0.5 mm |
| Depth below the rim | 49.2–59.9 mm | 49.2–65.3 mm | 53.1–58.4 mm | 45.0 mm | 45.1 mm |
| **Constraint violation:** goal tilt over limit | 0/5 | 0/5 | 4/5 | 0/5 | 0/5 |
| Worst tilt inside the rim | 6.2–42.3° | 3.1–63.8° | 10.4–13.9° | 0.6–1.4° | 0.1° |
| Meets the path limit inside the rim | 3/5 | 4/5 | 0/5 | 5/5 | 5/5 |
| Plans with a collision (deepest) | 3/5 (12.6 mm) | 3/5 (8.6 mm) | 5/5 (13.8 mm) | 0/5 | 0/5 |
| **Transport:** tilt at 25% of travel | 1.0–5.9° | 2.4–10.7° | 2.2–4.6° | 2.1–20.3° | 26.2° |
| Tilt at 50% | 2.4–9.3° | 3.6–7.8° | 4.8–8.1° | 25.6–52.4° | 52.5° |
| Tilt at 75% | 5.5–40.1° | 15.0–76.4° | 13.6–53.1° | 70.4–79.5° | 78.7° |
| Rotation done by half travel | 26–38% | 29–40% | 30–52% | 29–59% | 58% |
| **Path:** end-effector travel | 1.35–1.53 m | 1.42–1.72 m | 1.36–1.67 m | 1.34–1.46 m | 1.15 m |
| Joint travel | 9.3–10.2 rad | 8.9–11.3 rad | 8.6–10.5 rad | 9.0–11.1 rad | 10.2–10.4 rad |
| Planning time | 163.3–189.8 s | 168.2–186.3 s | 159.5–178.2 s | 181.4–255.0 s | 0.6–0.7 s |
| **Physics:** in cup | 3/5 | 4/5 | 1/5 | 5/5 | 5/5 |
| Beside the cup | 2/5 | 1/5 | 4/5 | 0/5 | 0/5 |
| Object lost from the hand | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| Tilt after release, in cup | 1.5–36.9° | 0.5–39.8° | 16.6° | 0.4–1.1° | 0.2–23.6° |
<!-- comparison:end -->

Every run, the OmniManip sweeps and the raw values are in
`outputs/comparison/insert_marker_same_grasp.md` and `.json`. Ours with its own
sampled grasps is in `outputs/comparison/insert_marker.md`.

**Read these results with care:**
- **Not yet planner and representation only.** Ours uses the TSRs Opus emitted in
  `gt_opus1`, while the other two use hand-written programs. Hand-authored TSRs for
  `insert_marker` do not exist yet.
- **Our collision check can step over the cup wall.** `plan_constrained` checks
  collisions every `eps` = 0.12 rad, so 11 of our 15 plans pass through the wall
  between checks, and all 7 that end beside the cup are among them. Physically the
  marker stays outside. At `eps` = 0.03 no plan collides and 13 of 15 end in the cup,
  but only 2 of 15 meet the path limit, down from 7: the emitted `insert` path TSR
  bounds neither position nor shaft tilt, and the plans have to go over the rim now.
  2 plans still end beside the cup with no collision found, unexplained so far. See
  `outputs/comparison/insert_marker_same_grasp_eps003.md`.
- **The shared grasp is close, not identical, for ReKep.** It comes from
  `manip_sim/omnimanip/programs/insert_marker.json`, 22 mm along the shaft and 3.9 mm
  above its axis. Ours lands within 0.02 mm and 0.01° of OmniManip's. ReKep solves
  its own grasp stage and lands within 0.6 mm and 3.7°, with the wrist turned 180° in
  4 of 5 seeds, which grips the same way.
- **Grasp mattered before it was fixed.** With sampled grasps, 8 of our 14 plans lost
  the marker, all gripping 7-9 mm above its axis; from the shared grasp none of 15 do.
  With the motion fixed, up to 6 mm holds and 8 mm or more loses it, at any position
  from 8 to 35 mm along the shaft. Closing-axis tilt off the table holds up to 5° and
  fails from 10°. See `outputs/comparison/insert_marker_grasp_factors.md`.
- **The lookahead does not apply to a given grasp.** It scores a grasp from 8 goal
  probes and rejects a score of 0, which rejected 6 of the 15 runs from the shared
  grasp although all 6 plan when it is skipped. All 6 failed runs with sampled grasps
  fail on this check too.
- **Residuals are not rankable.** ReKep and OmniManip optimize toward a point and
  TSR sampling lands anywhere inside its band, so compare pass/fail at a shared
  tolerance. Ours at 1° is level with ReKep on final tilt.
- **OmniManip's precision is a choice.** The paper gives no weights, and these are
  set so `L_path` only breaks ties; the weight sweep shows the tilt it can trade.
- **Transport tilt is partly the program's.** ReKep can state a path limit, but
  the program here sets none during transport. OmniManip has no way to state one.
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
TASK="--scene $SCENE --grounding outputs/grounding/insert_marker_gt_opus1 \
    --stage-plan outputs/runs/gt_opus1/stage_plan.json --selections outputs/runs/gt_opus1/selections.json \
    --emissions outputs/runs/gt_opus1/emissions_call3_opus.json"
GRASP="--grasp-program manip_sim/omnimanip/programs/insert_marker.json"
python scripts/plan_task_tol.py 5 $TASK $GRASP --seed 0 --out $OUT/same5_s0.npz
python scripts/plan_task_tol.py 5 $TASK --seed 0 --out $OUT/tsr5_s0.npz                      # sampled grasp
python scripts/plan_task_tol.py 5 --eps 0.03 $TASK $GRASP --timeout 60 --seed 0 \
    --out outputs/plans/compare_eps003/same5_s0.npz                                          # finer checks

python scripts/eval_plan.py --scene $SCENE --spec tasks/specs/insert_marker.json --plan $OUT/*.npz
python scripts/execute_task_plan.py --scene $SCENE --plan $OUT/rekep_s0.npz      # for each plan
COMPARE="python3 scripts/compare_plans.py --spec tasks/specs/insert_marker.json"
$COMPARE --plans $OUT --metrics outputs/metrics/vlm --methods same5 same1 same10 rekep omni \
    --weight-sweep outputs/plans/omnimanip_weight_sweep \
    --separation-sweep outputs/plans/omnimanip_separation_sweep \
    --out outputs/comparison/insert_marker_same_grasp --readme README.md
$COMPARE --plans $OUT --metrics outputs/metrics/vlm --seconds $OUT/plan_task_seconds.json \
    --methods tsr5 tsr1 tsr10 tsr5more rekep omni --out outputs/comparison/insert_marker
$COMPARE --plans outputs/plans/compare_eps003 --metrics outputs/plans/compare_eps003/metrics \
    --methods same5 same1 same10 --out outputs/comparison/insert_marker_same_grasp_eps003
```
`outputs/runs/` is not in the repo, so reproducing ours needs a `gt_opus1` run.

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