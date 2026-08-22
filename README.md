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