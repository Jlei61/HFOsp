#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-8}"
OUTPUT_ROOT="${3:-results/topic5_history_rnn_data_aligned_static_transfer_v0_3}"
G0_ROOT="results/topic5_history_rnn_early_ictal_field/g0_causal_prefix"
G1_ROOT="results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_refit_c30"
CONTRACT="config/topic5_history_rnn_direct_early_ictal_transfer_v0_2.json"
FEATURE_ROOT="${OUTPUT_ROOT}/feature_cache"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${FEATURE_ROOT}" "${LOG_ROOT}"

readarray -t subjects < <(python - "${G0_ROOT}" <<'PY'
import pandas as pd, pathlib, sys
frame = pd.read_csv(pathlib.Path(sys.argv[1]) / "subject_causal_history_inventory.csv")
for subject in sorted(frame.loc[frame.g2_patient_eligible, "subject"]):
    print(subject)
PY
)

run_one() {
  local subject="$1"
  local feature_dir="${FEATURE_ROOT}/${subject}"
  local output="${OUTPUT_ROOT}/${subject}"
  local log="${LOG_ROOT}/${subject}.log"
  local failed="${OUTPUT_ROOT}/${subject}.FAILED.json"
  if [[ -f "${output}/DONE.json" ]]; then
    echo "[skip] ${subject}"
    return 0
  fi
  if [[ ! -f "${feature_dir}/all_subject_readout_features.csv.gz" ]]; then
    if [[ -d "${feature_dir}" ]]; then
      mv "${feature_dir}" "${feature_dir}.incomplete.$(date +%Y%m%dT%H%M%S)"
    fi
    echo "[features] ${subject}"
    conda run -n cuda_env python \
      scripts/run_topic5_history_rnn_early_ictal_fold_v0_1.py \
      --heldout-subject "${subject}" \
      --artifact-root "${ARTIFACT_ROOT}" \
      --g1-root "${G1_ROOT}" \
      --g0-root "${G0_ROOT}" \
      --direct-transfer-contract "${CONTRACT}" \
      --output-dir "${feature_dir}" \
      --device cuda:0 \
      --export-feature-table \
      >>"${log}" 2>&1
  fi
  if [[ -d "${output}" ]]; then
    mv "${output}" "${output}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  echo "[train] ${subject}"
  conda run -n cuda_env python \
    scripts/run_topic5_history_rnn_data_aligned_fold_v0_3.py \
    --heldout-subject "${subject}" \
    --feature-table "${feature_dir}/all_subject_readout_features.csv.gz" \
    --output-dir "${output}" \
    --steps 600 \
    --learning-rate 0.03 \
    --weight-decay 0.01 \
    --seeds 11 29 47 \
    --n-perm 5000 \
    >>"${log}" 2>&1
  rm -f "${failed}"
  echo "[done] ${subject}"
}

active=0
failed=0
for subject in "${subjects[@]}"; do
  (
    if run_one "${subject}"; then
      :
    else
      status=$?
      python - "${OUTPUT_ROOT}/${subject}.FAILED.json" "${subject}" "${status}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "FAILED", "stage": "data_aligned_static_transfer_v0_3",
    "subject": sys.argv[2], "exit_code": int(sys.argv[3]), "time_epoch": time.time(),
}, indent=2) + "\n")
PY
      exit "${status}"
    fi
  ) &
  active=$((active + 1))
  if (( active >= WORKERS )); then
    if ! wait -n; then failed=1; fi
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  if ! wait -n; then failed=1; fi
  active=$((active - 1))
done
if (( failed != 0 )); then
  echo "one or more data-aligned folds failed" >&2
  exit 1
fi

conda run -n cuda_env python \
  scripts/summarize_topic5_history_rnn_data_aligned_v0_3.py \
  --input-dir "${OUTPUT_ROOT}" \
  >"${LOG_ROOT}/summary.log" 2>&1
echo "[complete] ${OUTPUT_ROOT}"
