#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-2}"
G1_ROOT="${3:-results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1}"
G0_ROOT="results/topic5_history_rnn_early_ictal_field/g0_causal_prefix"
OUTPUT_ROOT="${4:-results/topic5_history_rnn_direct_early_ictal_transfer_v0_2}"
CONTRACT="config/topic5_history_rnn_direct_early_ictal_transfer_v0_2.json"

mkdir -p "${OUTPUT_ROOT}/logs"
readarray -t subjects < <(python - "${G0_ROOT}" <<'PY'
import pandas as pd, pathlib, sys
frame = pd.read_csv(pathlib.Path(sys.argv[1]) / "subject_causal_history_inventory.csv")
for subject in sorted(frame.loc[frame.g2_patient_eligible, "subject"]):
    print(subject)
PY
)

run_one() {
  local subject="$1"
  local output="${OUTPUT_ROOT}/${subject}"
  local log="${OUTPUT_ROOT}/logs/${subject}.log"
  local failed="${OUTPUT_ROOT}/${subject}.FAILED.json"
  if [[ -f "${output}/DONE.json" ]]; then
    echo "[skip] ${subject}"
    return 0
  fi
  if [[ -d "${output}" ]]; then
    mv "${output}" "${output}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  echo "[start] ${subject}"
  if conda run -n cuda_env python \
      scripts/run_topic5_history_rnn_early_ictal_fold_v0_1.py \
      --heldout-subject "${subject}" \
      --artifact-root "${ARTIFACT_ROOT}" \
      --g1-root "${G1_ROOT}" \
      --g0-root "${G0_ROOT}" \
      --direct-transfer-contract "${CONTRACT}" \
      --output-dir "${output}" \
      --device cuda:0 \
      >"${log}" 2>&1; then
    rm -f "${failed}"
  else
    status=$?
    python - "${failed}" "${subject}" "${status}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "FAILED", "stage": "direct_transfer_v0_2",
    "subject": sys.argv[2], "exit_code": int(sys.argv[3]),
    "time_epoch": time.time(),
}, indent=2) + "\n")
PY
    echo "[failed] ${subject} exit=${status}" >&2
    return "${status}"
  fi
  echo "[done] ${subject}"
}

active=0
failed=0
for subject in "${subjects[@]}"; do
  run_one "${subject}" &
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
  echo "one or more direct-transfer folds failed; see *.FAILED.json" >&2
  exit 1
fi

conda run -n cuda_env python \
  scripts/summarize_topic5_history_rnn_direct_early_ictal_transfer_v0_2.py \
  --input-dir "${OUTPUT_ROOT}" \
  --g0-root "${G0_ROOT}"
