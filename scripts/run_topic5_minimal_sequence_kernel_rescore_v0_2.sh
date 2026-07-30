#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT/results/topic5_minimal_sequence_kernel_closeout/formal_v0_2}"
MAX_WORKERS="${MAX_WORKERS:-3}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
mkdir -p "$OUT_ROOT/logs"

mapfile -t SUBJECTS < <(
  find "$ROOT/results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject" \
    -maxdepth 1 -type f -name '*.npz' -printf '%f\n' \
    | sed 's/\.npz$//' | sort
)
SEEDS=(20260725 20260726 20260727)

run_cell() {
  local subject="$1"
  local seed="$2"
  local output="$OUT_ROOT/seed_${seed}/${subject}"
  local log="$OUT_ROOT/logs/seed_${seed}_${subject}.log"
  if [[ -f "$output/summary.json" ]]; then
    return 0
  fi
  if [[ -d "$output" ]]; then
    mv "$output" "${output}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  conda run -n cuda_env python \
    "$ROOT/scripts/analyze_topic5_minimal_sequence_kernel_cell_v0_2.py" \
    --subject "$subject" \
    --seed "$seed" \
    --output-dir "$output" \
    --device "$GPU_DEVICE" \
    --batch-size 512 \
    --cpu-threads 4 \
    >"$log" 2>&1
}
export ROOT OUT_ROOT GPU_DEVICE
export -f run_cell

tasks=()
for seed in "${SEEDS[@]}"; do
  for subject in "${SUBJECTS[@]}"; do
    tasks+=("$subject $seed")
  done
done

printf '%s\n' "${tasks[@]}" \
  | xargs -P "$MAX_WORKERS" -n 2 bash -c 'run_cell "$0" "$1"'

expected=$(( ${#SEEDS[@]} * ${#SUBJECTS[@]} ))
complete="$(find "$OUT_ROOT" -name summary.json -type f | wc -l)"
if [[ "$complete" -ne "$expected" ]]; then
  echo "expected $expected complete cells, found $complete" >&2
  exit 1
fi
printf '{"status":"COMPLETE","expected_cells":%d,"complete_cells":%d}\n' \
  "$expected" "$complete" >"$OUT_ROOT/LAUNCHER_DONE.json"
