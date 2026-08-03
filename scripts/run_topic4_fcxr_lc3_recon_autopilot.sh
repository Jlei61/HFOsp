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

while kill -0 "$geometry_pid" 2>/dev/null; do
  sleep 30
done

# Scientific geometry failure does not block reconnaissance.  The geometry
# execution lock is the engineering prerequisite and is written before prep/map.
"$python_bin" "$runner" lock
"$python_bin" "$runner" manifest
"$python_bin" "$runner" all --confirm-run
