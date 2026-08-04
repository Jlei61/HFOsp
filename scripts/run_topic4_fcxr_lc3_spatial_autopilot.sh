#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <recon-autopilot-pid>" >&2
  exit 2
fi

recon_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
python_bin="/home/honglab/leijiaxin/anaconda3/bin/python"
runner="scripts/run_topic4_fcxr_lc3_spatial.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$result_root"
printf '%s\n' "$$" > "$result_root/spatial_autopilot.pid"

# Wall guard sized from the measured cost of a 40k second, not by eye.  The E4
# reconnaissance measured 2026-08-04: the 32 s main run took ~4 h of wall clock,
# i.e. 320-450 s of wall per simulated second once spikes are stored for the whole
# trajectory.  The guard exists to break a hang, not to schedule; rows and cells
# resume from valid DONE files, so an over-long guard costs nothing while an
# under-sized one destroys hours of completed work.  The previous value could not
# cover the registered work of this stage and terminated it at 8 h.
on_error() {
  code="$?"
  "$python_bin" - "$result_root" "$code" <<'PY'
import datetime, json, os, sys
root, code = sys.argv[1], int(sys.argv[2])
path = os.path.join(root, "SPATIAL_AUTOPILOT_FAILED.json")
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

while kill -0 "$recon_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" - "$result_root" <<'PY'
import json, os, sys
path = os.path.join(sys.argv[1], "dynamic_reconnaissance", "aggregate.json")
if not os.path.isfile(path) or json.load(open(path)).get("status") != "COMPLETE":
    raise SystemExit("recon autopilot ended without a COMPLETE aggregate")
PY

"$python_bin" "$runner" lock
"$python_bin" "$runner" manifest

swap_base_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
"$python_bin" "$runner" all --confirm-run &
sim_pid="$!"
printf '%s\n' "$sim_pid" > "$result_root/spatial_simulation.pid"
last_swap_kib="$swap_base_kib"
start_epoch="$(date +%s)"
while kill -0 "$sim_pid" 2>/dev/null; do
  sleep 30
  current_swap_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
  delta_kib="$((current_swap_kib - swap_base_kib))"
  if (( delta_kib >= 524288 && current_swap_kib > last_swap_kib )); then
    kill -TERM "$sim_pid"
    wait "$sim_pid"
  fi
  # many 500 ms arms across states, amplitudes and bases; per-cell resume exists.
  if (( $(date +%s) - start_epoch > 108000 )); then
    kill -TERM "$sim_pid"
    wait "$sim_pid"
  fi
  last_swap_kib="$current_swap_kib"
done
wait "$sim_pid"

"$python_bin" - "$result_root" <<'PY'
import datetime, json, os, sys
root = sys.argv[1]
path = os.path.join(root, "SPATIAL_AUTOPILOT_DONE.json")
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "DONE",
               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
