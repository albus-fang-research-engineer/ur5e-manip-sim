#!/usr/bin/env bash
# Pick the best available headless rendering backend at container start.
# EGL (GPU) if an NVIDIA device is visible; otherwise OSMesa (CPU).
set -e

if [ -e /dev/nvidia0 ] || command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    echo "[manip-sim] NVIDIA GPU detected -> MUJOCO_GL=${MUJOCO_GL}"
else
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    echo "[manip-sim] No GPU detected -> MUJOCO_GL=osmesa (software rendering)"
fi

exec "$@"
