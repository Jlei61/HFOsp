#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-2}"
OUTPUT_ROOT="results/topic5_history_rnn_early_ictal_field/g1_sequential_capacity_matched_development_v0_1"
mkdir -p "${OUTPUT_ROOT}/logs"

subjects=(epilepsiae_1073 epilepsiae_1146 yuquan_chenziyang)
# name|history_dim|half_life_hours|learning_rate
configs=(
  h32_half2_lr3e4\|32\|2.0\|3e-4
  h32_half0p5_lr3e4\|32\|0.5\|3e-4
  h32_half2_lr1e3\|32\|2.0\|1e-3
)

run_one() {
  local subject="$1"
  local config="$2"
  IFS='|' read -r name dimension half_life learning_rate <<<"${config}"
  local output="${OUTPUT_ROOT}/${name}/${subject}"
  local log="${OUTPUT_ROOT}/logs/${name}__${subject}.log"
  if [[ -f "${output}/DONE.json" ]]; then
    echo "[skip] ${name} ${subject}"
    return 0
  fi
  echo "[start] ${name} ${subject}"
  conda run -n cuda_env python \
    scripts/run_topic5_history_rnn_gate1_sequential_fold_v0_1.py \
    --heldout-subject "${subject}" \
    --seed 20260725 \
    --artifact-root "${ARTIFACT_ROOT}" \
    --output-dir "${output}" \
    --device cuda:0 \
    --embedding-batch-size 8192 \
    --history-dim "${dimension}" \
    --initial-half-life-hours "${half_life}" \
    --matched-cycles 3 \
    --history-cycles 3 \
    --segment-batch-size 16 \
    --bptt-chunk 256 \
    --learning-rate "${learning_rate}" \
    >"${log}" 2>&1
  echo "[done] ${name} ${subject}"
}

active=0
for config in "${configs[@]}"; do
  for subject in "${subjects[@]}"; do
    run_one "${subject}" "${config}" &
    active=$((active + 1))
    if (( active >= WORKERS )); then
      wait -n
      active=$((active - 1))
    fi
  done
done
wait

conda run -n cuda_env python \
  scripts/summarize_topic5_history_rnn_gate1_sequential_development_v0_1.py \
  --input-dir "${OUTPUT_ROOT}"
