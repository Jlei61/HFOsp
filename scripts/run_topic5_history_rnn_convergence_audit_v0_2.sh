#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-2}"
OUTPUT_ROOT="results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_diagnostics/real_convergence"
SEED=20260725
PATIENTS=(epilepsiae_1073 epilepsiae_1146 yuquan_chenziyang)
CYCLES=(10 30)
mkdir -p "${OUTPUT_ROOT}/logs"

run_one() {
  local cycles="$1"
  local subject="$2"
  local output="${OUTPUT_ROOT}/c${cycles}/${subject}"
  local log="${OUTPUT_ROOT}/logs/c${cycles}_${subject}.log"
  if [[ -f "${output}/DONE.json" ]]; then
    echo "[skip] c${cycles} ${subject}"
    return 0
  fi
  if [[ -d "${output}" ]]; then
    mv "${output}" "${output}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  mkdir -p "$(dirname "${output}")"
  echo "[start] c${cycles} ${subject}"
  conda run -n cuda_env python \
    scripts/run_topic5_history_rnn_gate1_sequential_fold_v0_1.py \
    --heldout-subject "${subject}" \
    --seed "${SEED}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --output-dir "${output}" \
    --device cuda:0 \
    --history-dim 16 \
    --initial-half-life-hours 2 \
    --matched-cycles 3 \
    --history-cycles "${cycles}" \
    --segment-batch-size 16 \
    --bptt-chunk 256 \
    --learning-rate 3e-4 \
    --rank-weight 0.2 \
    >"${log}" 2>&1
  echo "[done] c${cycles} ${subject}"
}

active=0
failed=0
for cycles in "${CYCLES[@]}"; do
  for subject in "${PATIENTS[@]}"; do
    run_one "${cycles}" "${subject}" &
    active=$((active + 1))
    if (( active >= WORKERS )); then
      if ! wait -n; then failed=1; fi
      active=$((active - 1))
    fi
  done
done
while (( active > 0 )); do
  if ! wait -n; then failed=1; fi
  active=$((active - 1))
done
exit "${failed}"
