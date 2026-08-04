#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <spatial-autopilot-pid>" >&2
  exit 2
fi

spatial_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
python_bin="/home/honglab/leijiaxin/anaconda3/bin/python"
runner="scripts/run_topic4_fcxr_lc3_x_lifecycle.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$result_root"
printf '%s\n' "$$" > "$result_root/x_lifecycle_autopilot.pid"

# Wall guard sized from the measured cost of a 40k second, not by eye.  The E4
# reconnaissance measured 2026-08-04: the 32 s main run took ~4 h of wall clock,
# i.e. 320-450 s of wall per simulated second once spikes are stored for the whole
# trajectory.  The guard exists to break a hang, not to schedule; rows and cells
# resume from valid DONE files, so an over-long guard costs nothing while an
# under-sized one destroys hours of completed work.  The previous value could not
# cover the registered work of this stage and terminated it at 12 h.
on_error() {
  code="$?"
  "$python_bin" - "$result_root" "$code" <<'PY'
import datetime, json, os, sys
root, code = sys.argv[1], int(sys.argv[2])
path = os.path.join(root, "X_LIFECYCLE_AUTOPILOT_FAILED.json")
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "FAILED", "exit_code": code,
               "failed": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
  exit "$code"
}
trap on_error ERR

run_guarded() {
  local label="$1"
  local limit_s="$2"
  shift 2
  local swap_base_kib last_swap_kib current_swap_kib delta_kib start_epoch child_pid
  swap_base_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
  "$@" &
  child_pid="$!"
  printf '%s\n' "$child_pid" > "$result_root/${label}_simulation.pid"
  last_swap_kib="$swap_base_kib"
  start_epoch="$(date +%s)"
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 30
    current_swap_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
    delta_kib="$((current_swap_kib - swap_base_kib))"
    if (( delta_kib >= 524288 && current_swap_kib > last_swap_kib )); then
      kill -TERM "$child_pid"
      wait "$child_pid"
    fi
    if (( $(date +%s) - start_epoch > limit_s )); then
      kill -TERM "$child_pid"
      wait "$child_pid"
    fi
    last_swap_kib="$current_swap_kib"
  done
  wait "$child_pid"
}

while kill -0 "$spatial_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" - "$result_root" <<'PY'
import json, os, sys
path = os.path.join(sys.argv[1], "SPATIAL_PROBE_DONE.json")
if not os.path.isfile(path) or json.load(open(path)).get("status") != "DONE":
    raise SystemExit("spatial autopilot ended without SPATIAL_PROBE_DONE")
PY

"$python_bin" "$runner" lock
# <=10 candidates x (8 s low + 4 s high) ~= 15 h at the measured rate.
run_guarded x_calibration 86400 "$python_bin" "$runner" calibrate --confirm-run
"$python_bin" "$runner" lifecycle-manifest
# <=6 nominal runs x up to 45 s ~= 34 h at the measured rate.
run_guarded lifecycle 180000 "$python_bin" "$runner" lifecycle --confirm-run

"$python_bin" - "$result_root" <<'PY'
import datetime, json, os, sys
root = sys.argv[1]
path = os.path.join(root, "X_LIFECYCLE_AUTOPILOT_DONE.json")
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "DONE",
               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
