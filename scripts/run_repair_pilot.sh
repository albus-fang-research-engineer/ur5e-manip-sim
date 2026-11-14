#!/usr/bin/env bash
# D9 pilot (PAID): the qualified faults x 3 conditions x 1 repeat, run_repair_case.py
# --mode live, PILOT_JOBS cases at a time (default 3) in the sim image.
# Hard bound: 8 requests per case (call cap 8; a call starts only if its retries fit).
# Resumable: a case with a record.json is skipped; a case folder without one (stopped
# mid-run) is left alone and reported, move it aside to rerun that case.
#   ANTHROPIC_API_KEY=... [PILOT_TASK=pour_tea PILOT_SEL=.. PILOT_EM=.. PILOT_FAULTS=".."]
#     scripts/run_repair_pilot.sh OUT_DIR
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$(realpath -m "${1:?usage: run_repair_pilot.sh OUT_DIR}")"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
read -r -a FAULTS <<< "${PILOT_FAULTS:-insert_wrong_anchor insert_contradictory_perpendicular insert_handoff_path_relation}"
CONDITIONS=(typed render_check untyped)
TASK=${PILOT_TASK:-insert_marker_upright}
SEL=${PILOT_SEL:-tasks/repair_study/start/insert_marker/selections.json}
EM=${PILOT_EM:-tasks/repair_study/start/insert_marker/emissions.json}
mkdir -p "$out"
run_one() {
  local f=$1 c=$2 d="$out/$1/$2"
  if [ -f "$d/record.json" ]; then        # resume: a finished case is never rerun (or paid again)
    echo "[pilot] $f / $c: already finished, skipped"; return 0
  fi
  if [ -d "$d" ]; then                    # a case stopped mid-run is rerun from the start
    echo "[pilot] $f / $c: no record (stopped mid-run?); move $d aside to rerun it"; return 0
  fi
  mkdir -p "$d"
  docker run --rm -u "$(id -u):$(id -g)" -v "$here:$here" -v "$out:$out" -w "$here" \
    -e HOME=/tmp -e PYTHONPATH="$here" -e NUMBA_CACHE_DIR=/tmp/numba \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    -e ANTHROPIC_API_KEY -e MANIP_VLM_MODEL=claude-opus-5 -e MANIP_VLM_TEMPERATURE=none \
    -e MANIP_VLM_MAX_TOKENS=4000 \
    manip-sim:latest python scripts/run_repair_case.py --task "$TASK" \
    --selections "$SEL" --emissions "$EM" --fault "$f" --condition "$c" \
    --out "$d" --mode live > "$d/run.log" 2>&1
  echo "[pilot] $f / $c: $(grep '^\[case\] '"$c"':' "$d/run.log" | tail -1)"
}
export -f run_one; export out here SEL EM TASK
jobs=()
for f in "${FAULTS[@]}"; do for c in "${CONDITIONS[@]}"; do jobs+=("$f $c"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P "${PILOT_JOBS:-3}" -L 1 bash -c 'run_one "$0" "$1"'
echo "[pilot] done -> $out"
