#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-8}"
CYCLES="${3:-10}"
G0_ROOT="results/topic5_history_rnn_early_ictal_field/g0_causal_prefix"
OUTPUT_ROOT="results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_refit_c${CYCLES}/seed_20260725"
LOG_ROOT="results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_refit_c${CYCLES}/logs"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

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
  local log="${LOG_ROOT}/${subject}.log"
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
      scripts/run_topic5_history_rnn_gate1_sequential_fold_v0_1.py \
      --heldout-subject "${subject}" \
      --seed 20260725 \
      --artifact-root "${ARTIFACT_ROOT}" \
      --output-dir "${output}" \
      --device cuda:0 \
      --history-dim 16 \
      --initial-half-life-hours 2 \
      --matched-cycles 3 \
      --history-cycles "${CYCLES}" \
      --segment-batch-size 16 \
      --bptt-chunk 256 \
      --learning-rate 3e-4 \
      --rank-weight 0.2 \
      >"${log}" 2>&1; then
    rm -f "${failed}"
    echo "[done] ${subject}"
  else
    status=$?
    python - "${failed}" "${subject}" "${status}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "FAILED", "stage": "g1_direct_refit_v0_2",
    "subject": sys.argv[2], "exit_code": int(sys.argv[3]),
    "time_epoch": time.time(),
}, indent=2) + "\n")
PY
    echo "[failed] ${subject} exit=${status}" >&2
    return "${status}"
  fi
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
  echo "one or more direct-refit folds failed" >&2
  exit 1
fi

python - "${OUTPUT_ROOT}" "${CYCLES}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
done = list(root.glob("*/DONE.json"))
failed = list(root.glob("*.FAILED.json"))
if len(done) != 16 or failed:
    raise SystemExit(f"refit incomplete: done={len(done)} failed={len(failed)}")
(root.parent / "REFIT_SUMMARY.json").write_text(json.dumps({
    "status": "COMPLETE",
    "contract": "topic5_history_rnn_direct_checkpoint_refit_v0_2",
    "history_cycles": int(sys.argv[2]),
    "n_completed_folds": len(done),
    "n_failed_folds": len(failed),
    "target_values_read": False,
}, indent=2) + "\n")
PY
