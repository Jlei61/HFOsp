#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-selected_history_interventions_20260729}"
formal_root="${repo_root}/results/topic5_ordered_history_architecture_audit/formal/architecture_controls_formal_20260729"
shuffle_root="${repo_root}/results/topic5_ordered_history_architecture_audit/rank_shuffle/selected_architecture_rank_shuffle_20260729"
selection="${repo_root}/results/topic5_ordered_history_architecture_audit/analysis/ARCHITECTURE_SUMMARY.json"
run_root="${repo_root}/results/topic5_ordered_history_architecture_audit/interventions/${tag}"
mkdir -p "${run_root}/logs"
mapfile -t subjects < <(
  python - "${repo_root}" <<'PY'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
frame = pd.read_csv(
    root / "results/topic5_interictal_rank_distribution/dataset_v0_4/subject_audit.csv"
)
for subject in sorted(frame.loc[frame.status.eq("ok"), "subject"].astype(str)):
    print(subject)
PY
)
seeds=(20260725 20260726 20260727)
workers=3
jobs=()
for seed in "${seeds[@]}"; do
  for subject in "${subjects[@]}"; do
    jobs+=("${seed}|${subject}")
  done
done
expected="${#jobs[@]}"

python "${repo_root}/scripts/monitor_topic5_rank_distribution_resources.py" \
  --pid "$$" \
  --output "${run_root}/resource_log.csv" \
  --interval-seconds 20 &
resource_pid=$!

run_worker() {
  local worker_id="$1"
  local index=0
  for job in "${jobs[@]}"; do
    if (( index % workers != worker_id )); then
      index=$((index + 1))
      continue
    fi
    IFS='|' read -r seed subject <<< "${job}"
    local cell="${run_root}/seed_${seed}/${subject}"
    local log="${run_root}/logs/seed${seed}_${subject}.log"
    if [[ -f "${cell}/DONE.json" ]]; then
      index=$((index + 1))
      continue
    fi
    if [[ -e "${cell}" ]]; then
      echo "incomplete existing cell blocks safe resume: ${cell}" >&2
      return 3
    fi
    mkdir -p "$(dirname "${cell}")"
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    conda run --no-capture-output -n cuda_env \
      python "${repo_root}/scripts/run_topic5_selected_history_interventions_v0_1.py" \
        --subject "${subject}" \
        --seed "${seed}" \
        --selection-summary "${selection}" \
        --formal-root "${formal_root}" \
        --shuffle-root "${shuffle_root}" \
        --output-dir "${cell}" \
        --device cuda \
        --batch-size 256 \
      > "${log}" 2>&1
    index=$((index + 1))
  done
}

worker_pids=()
for worker_id in 0 1 2; do
  run_worker "${worker_id}" &
  worker_pids+=("$!")
done
exit_code=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done
kill "${resource_pid}" 2>/dev/null || true
wait "${resource_pid}" 2>/dev/null || true
completed="$(find "${run_root}" -name DONE.json -type f | wc -l)"
python - "${run_root}" "${expected}" "${completed}" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected, completed, exit_code = map(int, sys.argv[2:])
(root / "LAUNCHER_DONE.json").write_text(
    json.dumps(
        {
            "status": "COMPLETE" if exit_code == 0 and completed == expected else "INCOMPLETE",
            "expected_cells": expected,
            "completed_cells": completed,
            "exit_code": exit_code,
        },
        indent=2,
    )
    + "\n"
)
PY
exit "${exit_code}"
