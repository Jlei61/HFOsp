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
  if (( $(date +%s) - start_epoch > 28800 )); then
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
