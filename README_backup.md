# manip-sim

Simulation platform for VLM-authored TSR constraints + constrained planning.
Two isolated wings sharing one repo, per the engine/benchmark alignment plan:

| Wing | Image | Stack | Purpose |
|---|---|---|---|
| `sim` | `manip-sim` | MuJoCo **3.3.7** + robosuite **1.5.2** | Day-to-day dev; purpose-built constrained scenes (pour, insertion); path-manifold experiments |
| `libero` | `manip-sim-libero` | MuJoCo 2.3.7 + robosuite **1.4.0** + LIBERO | Open6DOR V2 benchmark (Phase 3). Built on demand, `--profile phase3` |

They are separate images because LIBERO pins `robosuite==1.4.0`, which is
API-incompatible with 1.5 (composite controllers, mujoco 3.x). Isolation
means a dependency change in one wing can never silently shift results in
the other.

## Quick start (Docker)

```bash
# GPU host (needs nvidia-container-toolkit):
docker compose build sim
docker compose run --rm sim python scripts/smoke_test.py
docker compose run --rm sim python scripts/ik_test.py

# CPU-only host:
docker compose run --rm sim-cpu python scripts/smoke_test.py

# Interactive shell:
docker compose run --rm sim
```

The entrypoint auto-selects the rendering backend: EGL if an NVIDIA GPU is
visible, OSMesa (software) otherwise. Both paths are headless; no X server
needed.

## Quick start (bare Ubuntu, no Docker)

```bash
sudo apt install libosmesa6 libegl1 libgl1 libglfw3 ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GPU:        MUJOCO_GL=egl    python scripts/smoke_test.py
# CPU-only:   MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/smoke_test.py
```

## Version pins that matter (verified the hard way)

1. **`mujoco==3.3.7`, not latest.** robosuite 1.5.2 declares `mujoco>=3.3.0`
   but crashes on `mujoco>=3.10`: the sparse mass-matrix API was reworked and
   `mj_fullM` now takes `(model, data, dst)`, while robosuite's OSC controller
   calls the old `(model, dst, qM)` form. Symptom: `TypeError: mj_fullM():
   incompatible function arguments` on the first `env.reset()`.
2. **`numpy<2`** — numba and the LIBERO-era tooling still expect 1.x.
3. **OSMesa needs both env vars**: `MUJOCO_GL=osmesa` *and*
   `PYOPENGL_PLATFORM=osmesa`. With only the first, PyOpenGL grabs the wrong
   platform and dies with `'NoneType' object has no attribute 'glGetError'`.
4. **IK controller config**: don't mutate the BASIC composite config's part
   `type` to `IK_POSE` in place — leftover OSC keys (`input_max`, ...) collide
   with the IK controller's kwargs in 1.5.2. Replace the whole part dict
   (see `scripts/ik_test.py`).
5. **The "mink-based whole-body IK" warning at import is harmless** for
   arm-only work: the pip wheel simply doesn't ship `robosuite.examples`
   (source-repo-only), and the loader only needs it for the GR1 humanoid's
   default controller. `mink` itself is installed and importable — usable
   standalone for differential-IK / TSR-projection experiments. The per-arm
   `IK_POSE` controller works (tested).

## What the smoke test asserts

- import + version guard (mujoco 3.3.x)
- Panda `Lift` env construction
- 256×256 RGB-D offscreen render under the active backend
- ground-truth object pose read from `env.sim` — the critical-path pose
  source in sim; estimated poses (GenPose++/FoundationPose) belong in the
  perception-ablation arm, not the main loop
- 20-step random rollout
- robot selectable via `ROBOT` env var (default `UR5e`)

## Layout

```
manip-sim/
├── docker/
│   ├── Dockerfile           # main wing (tested end-to-end)
│   ├── Dockerfile.libero    # Open6DOR V2 wing (Phase 3; build on demand)
│   └── entrypoint.sh        # EGL/OSMesa auto-selection
├── docker-compose.yml
├── requirements.txt
└── scripts/
    ├── smoke_test.py
    └── ik_test.py
```

## Robot choice: UR5e (default)

`ROBOT=UR5e` is the default in both test scripts; any stock robosuite robot
name drops in (`ROBOT=Panda python scripts/smoke_test.py`). Everything —
scene, cameras, ground-truth state, OSC control — is robot-agnostic in
robosuite; the robot is one string.

The single asterisk: robosuite 1.5.2's built-in `IK_POSE` controller
hard-asserts a whitelist `{Panda, Sawyer, Baxter, GR1FixedLowerBody}` and
**UR5e is not on it**. This costs nothing in practice:

- `OSC_POSE` (the default and the standard choice) works on UR5e.
- `mink` differential IK works on any MJCF, UR5e included — verified to
  ~1e-16 residual in `scripts/ik_test.py`. Since mink is also the natural
  vehicle for TSR-projection / constrained-IK work, it was going to be the
  IK path regardless.

## AMD notes

- **CPU (amd64/x86)**: that is what these images are — `nvidia/cuda` base
  is amd64. Nothing to change.
- **AMD GPU (no NVIDIA)**: swap the base image in `docker/Dockerfile` to
  `ubuntu:22.04`, delete the `deploy.resources` GPU reservation in
  `docker-compose.yml`, and add `devices: ["/dev/dri:/dev/dri"]` to the
  service. Mesa's EGL then renders on the AMD GPU with the packages already
  installed (`libegl1`/`libgles2`); the entrypoint's fallback logic still
  covers pure-CPU hosts via OSMesa.

## Next steps (matching the plan)

- [ ] MJCF asset-conversion script (one script, reused for every object)
- [ ] Task-class skeleton for the purpose-built constrained scenes
      (pour with sphere-liquid proxy, peg/socket insertion) as robosuite
      `ManipulationEnv` subclasses — same house as Open6DOR V2
- [ ] Noise-injected pose/tracking ablation harness as a first-class module
      (per the caveat: sim ground truth hides exactly the failure modes the
      architecture targets)
- [ ] Phase 3: build `libero` wing, clone Open6DOR V2 assets from the SOFAR
      release, pin commit in `Dockerfile.libero`
- [ ] Phase 3: SIMPLER (SAPIEN backend) as real-camera visual-gap check
# ur5e-manip-sim
