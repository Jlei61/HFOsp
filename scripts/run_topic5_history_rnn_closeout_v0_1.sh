#!/usr/bin/env bash
set -euo pipefail

ROOT="results/topic5_history_rnn_early_ictal_field"
G1="${ROOT}/g1_sequential_formal_v0_1/G1_MULTI_SEED_SUMMARY.json"
G2="${ROOT}/g2_early_ictal_loso_v0_1/G2_G3_SUMMARY.json"

if [[ ! -f "${G1}" ]]; then
  echo "G1 multi-seed inference is incomplete" >&2
  exit 2
fi

status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${G1}")"
if [[ "${status}" == "G1_MULTI_SEED_PASS_OPEN_G2" && ! -f "${G2}" ]]; then
  echo "G1 passed but G2/G3 inference is incomplete" >&2
  exit 2
fi
if [[ "${status}" != "G1_MULTI_SEED_PASS_OPEN_G2" && -f "${G2}" ]]; then
  echo "illegal G2/G3 summary exists after a failed G1 gate" >&2
  exit 2
fi

conda run -n cuda_env python \
  scripts/closeout_topic5_history_rnn_early_ictal_field_v0_1.py \
  --root "${ROOT}"
conda run -n cuda_env python \
  scripts/plot_topic5_history_rnn_early_ictal_field_v0_1.py \
  --root "${ROOT}"
conda run -n cuda_env python \
  scripts/audit_topic5_history_rnn_reproducibility_v0_1.py \
  --repo . --result-root "${ROOT}"

echo "[complete] ${status}"
