#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT/results/topic5_minimal_sequence_kernel_closeout/cross_dataset_v0_2}"
MAX_WORKERS="${MAX_WORKERS:-3}"
mkdir -p "$OUT_ROOT/logs"

run_cell() {
  local dataset="$1"
  local seed="$2"
  local run_dir="$OUT_ROOT/${dataset}_to_other/seed_${seed}"
  local log="$OUT_ROOT/logs/${dataset}_${seed}.log"
  if [[ -f "$run_dir/summary.json" ]]; then
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    mv "$run_dir" "${run_dir}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  conda run -n cuda_env python \
    "$ROOT/scripts/train_topic5_cross_dataset_confirmation_v0_2.py" \
    --source-dataset "$dataset" \
    --seed "$seed" \
    --run-dir "$run_dir" \
    --device cuda \
    --cpu-threads 4 \
    --gpu-memory-fraction 0.22 \
    >"$log" 2>&1
}
export ROOT OUT_ROOT
export -f run_cell

for dataset in epilepsiae yuquan; do
  for seed in 20260725 20260726 20260727; do
    printf '%s %s\n' "$dataset" "$seed"
  done
done | xargs -P "$MAX_WORKERS" -n 2 bash -c 'run_cell "$0" "$1"'

complete="$(find "$OUT_ROOT" -name summary.json -type f | wc -l)"
if [[ "$complete" -ne 6 ]]; then
  echo "expected 6 complete cross-dataset cells, found $complete" >&2
  exit 1
fi
printf '{"status":"COMPLETE","expected_cells":6,"complete_cells":%d}\n' \
  "$complete" >"$OUT_ROOT/LAUNCHER_DONE.json"
