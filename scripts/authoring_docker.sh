#!/usr/bin/env bash
# Run a command for the authoring ablation in the sim image (Python 3.10;
# the host's is 3.8). The worktree is mounted at its own path and the
# shared .git read-only, so artifacts can record the commit; the API key is
# passed through from the caller's environment, never written anywhere.
#   scripts/authoring_docker.sh python scripts/run_authoring_ablation.py --tag v1
#   scripts/authoring_docker.sh python -m pytest -q tests/test_code_authoring.py
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
common="$(git -C "$here" rev-parse --git-common-dir)"
common="$(cd "$here" && cd "$common" && pwd)"
exec docker run --rm -u "$(id -u):$(id -g)" \
  -v "$here:$here" -v "$common:$common:ro" -w "$here" \
  -e HOME=/tmp -e PYTHONPATH="$here" -e NUMBA_CACHE_DIR=/tmp/numba \
  -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  -e ANTHROPIC_API_KEY -e MANIP_VLM_MODEL -e MANIP_VLM_TEMPERATURE -e MANIP_VLM_MAX_TOKENS \
  manip-sim:latest "$@"
