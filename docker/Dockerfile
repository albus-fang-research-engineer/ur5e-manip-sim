# =============================================================================
# manip-sim : MuJoCo 3.3 + robosuite 1.5 (the "house")
#
# CUDA runtime base so the same image later hosts GenPose++ / FoundationPose /
# nvblox-adjacent GPU work without a rebuild of the sim layer. On a machine
# without a GPU the image still runs fine — rendering falls back to OSMesa
# (software) via the entrypoint.
#
# Rendering backends:
#   EGL    -> headless GPU rendering (needs nvidia-container-toolkit and
#             NVIDIA_DRIVER_CAPABILITIES to include "graphics")
#   OSMesa -> headless CPU rendering (always works, slower)
#   GLFW   -> on-screen viewer (X11 forwarding; optional)
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip python3-dev \
        build-essential git curl ca-certificates \
        # --- OpenGL / rendering backends ---
        libegl1 libgles2 libopengl0 \
        libosmesa6 \
        libglfw3 libglew-dev \
        libgl1-mesa-glx libglib2.0-0 \
        # --- misc ---
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# EGL on NVIDIA inside containers: expose the GL vendor ICD.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : {\n    "library_path" : "libEGL_nvidia.so.0"\n  }\n}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace

# Virtualenv keeps the stack isolated from system python.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Default to EGL; entrypoint downgrades to OSMesa if no NVIDIA device found.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
