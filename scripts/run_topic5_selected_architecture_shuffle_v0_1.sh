#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
architecture="${1:?selected architecture is required}"
tag="${2:-selected_rank_shuffle_20260729}"
case "${architecture}" in
  linear_state|vanilla_rnn|low_rank_r0|low_rank_r1|low_rank_r2|low_rank_r4)
    ;;
  *)
    echo "unsupported selected architecture: ${architecture}" >&2
    exit 2
    ;;
esac
run_root="${repo_root}/results/topic5_ordered_history_architecture_audit/rank_shuffle/${tag}"
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
    local cell="${run_root}/${architecture}/seed_${seed}/${subject}"
    local log="${run_root}/logs/${architecture}_seed${seed}_${subject}.log"
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
      python "${repo_root}/scripts/train_topic5_architecture_control_v0_1.py" \
        --run-dir "${cell}" \
        --heldout-subject "${subject}" \
        --architecture "${architecture}" \
        --seed "${seed}" \
        --rank-shuffle \
        --batch-size 1024 \
        --shared-cycles 1 \
        --calibration-cycles 4 \
        --updates-per-patient 8 \
        --rollouts 2000 \
        --cpu-threads 4 \
        --gpu-memory-fraction 0.24 \
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
python - "${run_root}" "${architecture}" "${expected}" "${completed}" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
architecture = sys.argv[2]
expected, completed, exit_code = map(int, sys.argv[3:])
value = {
    "status": "COMPLETE" if exit_code == 0 and completed == expected else "INCOMPLETE",
    "selected_architecture": architecture,
    "expected_cells": expected,
    "completed_cells": completed,
    "exit_code": exit_code,
}
(root / "LAUNCHER_DONE.json").write_text(json.dumps(value, indent=2) + "\n")
PY
exit "${exit_code}"
