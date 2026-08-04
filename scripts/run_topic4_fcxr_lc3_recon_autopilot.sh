#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <geometry-autopilot-pid>" >&2
  exit 2
fi

geometry_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/dynamic_reconnaissance"
python_bin="/home/honglab/leijiaxin/anaconda3/bin/python"
runner="scripts/run_topic4_fcxr_lc3_recon.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$result_root"
printf '%s\n' "$$" > "$result_root/autopilot.pid"

# Wall guard sized from the measured cost of a 40k second, not by eye.  The E4
# reconnaissance measured 2026-08-04: the 32 s main run took ~4 h of wall clock,
# i.e. 320-450 s of wall per simulated second once spikes are stored for the whole
# trajectory.  The guard exists to break a hang, not to schedule; rows and cells
# resume from valid DONE files, so an over-long guard costs nothing while an
# under-sized one destroys hours of completed work.  The previous value could not
# cover the registered work of this stage and terminated it at 5 h with zero completed rows.
on_error() {
  code="$?"
  "$python_bin" - "$result_root" "$code" <<'PY'
import datetime, json, os, sys
root, code = sys.argv[1], int(sys.argv[2])
path = os.path.join(root, "AUTOPILOT_FAILED.json")
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

while kill -0 "$geometry_pid" 2>/dev/null; do
  sleep 30
done

# Scientific geometry failure does not block reconnaissance.  The geometry
# execution lock is the engineering prerequisite and is written before prep/map.
"$python_bin" "$runner" lock
"$python_bin" "$runner" manifest

swap_base_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
"$python_bin" "$runner" all --confirm-run &
sim_pid="$!"
printf '%s\n' "$sim_pid" > "$result_root/simulation.pid"
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
  # 3 rows x up to 45 s + reduce + the primary-seed landmark replay ~= 20 h.
  if (( $(date +%s) - start_epoch > 108000 )); then
    kill -TERM "$sim_pid"
    wait "$sim_pid"
  fi
  last_swap_kib="$current_swap_kib"
done
wait "$sim_pid"
