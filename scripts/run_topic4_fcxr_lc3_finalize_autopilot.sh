#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <x-lifecycle-autopilot-pid>" >&2
  exit 2
fi

upstream_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
python_bin="/home/honglab/leijiaxin/anaconda3/bin/python"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$result_root"
printf '%s\n' "$$" > "$result_root/finalize_autopilot.pid"

while kill -0 "$upstream_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" scripts/finalize_topic4_fcxr_lc3.py
