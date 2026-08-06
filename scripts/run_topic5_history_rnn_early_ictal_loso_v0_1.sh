#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-2}"
G1_ROOT="results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1"
G0_ROOT="results/topic5_history_rnn_early_ictal_field/g0_causal_prefix"
OUTPUT_ROOT="results/topic5_history_rnn_early_ictal_field/g2_early_ictal_loso_v0_1"
GATE="${G1_ROOT}/G1_MULTI_SEED_SUMMARY.json"

if [[ ! -f "${GATE}" ]]; then
  echo "[locked] G1 multi-seed summary absent"
  exit 0
fi
status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${GATE}")"
if [[ "${status}" != "G1_MULTI_SEED_PASS_OPEN_G2" ]]; then
  echo "[locked] ${status}"
  exit 0
fi

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
      --output-dir "${output}" \
      --device cuda:0 \
      >"${log}" 2>&1; then
    rm -f "${failed}"
  else
    status=$?
    python - "${failed}" "${subject}" "${status}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "FAILED",
    "stage": "g2_loso",
    "subject": sys.argv[2],
    "exit_code": int(sys.argv[3]),
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
  echo "one or more G2 folds failed; see *.FAILED.json" >&2
  exit 1
fi

conda run -n cuda_env python \
  scripts/summarize_topic5_history_rnn_early_ictal_loso_v0_1.py \
  --input-dir "${OUTPUT_ROOT}" \
  --g0-root "${G0_ROOT}"
