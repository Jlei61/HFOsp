#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${repo_root}/config/topic5_static_scaffold_reliability_history_necessity_v0_1.yaml"
run_root="${repo_root}/results/topic5_interictal_scaffold_reliability_history_necessity/history_runs_v0_1"
frozen_root="${repo_root}/results/topic5_interictal_rank_distribution/runs/formal_multiseed_20260725_v1"
mkdir -p "${run_root}"

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
if (( ${#subjects[@]} != 34 )); then
  echo "Expected 34 subjects, found ${#subjects[@]}" >&2
  exit 3
fi

mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( mem_available_kb < 32 * 1024 * 1024 )); then
  echo "Refusing launch: MemAvailable is below 32 GiB" >&2
  exit 4
fi

python "${repo_root}/scripts/monitor_topic5_history_necessity_v0_1.py" \
  --launcher-pid "$$" \
  --run-root "${run_root}" \
  --expected-folds 102 \
  --expected-models 306 \
  --interval-seconds 30 &
monitor_pid=$!
cleanup() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT

run_shard() {
  local seed="$1"
  local shard="$2"
  local seed_root="${run_root}/seed_${seed}"
  local log_root="${seed_root}/logs"
  mkdir -p "${log_root}"
  local index
  for index in "${!subjects[@]}"; do
    if (( index % 2 != shard )); then
      continue
    fi
    local subject="${subjects[$index]}"
    local fold_root="${seed_root}/${subject}"
    if [[ -f "${fold_root}/DONE.json" ]] && \
       python - "${fold_root}/DONE.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)
PY
    then
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    conda run --no-capture-output -n cuda_env \
      python "${repo_root}/scripts/train_topic5_history_necessity_v0_1.py" \
        --config "${config}" \
        --run-dir "${fold_root}" \
        --heldout-subject "${subject}" \
        --seed "${seed}" \
      > "${log_root}/${subject}.log" 2>&1
  done
}

seeds=(20260725 20260726 20260727)
worker_pids=()
for seed in "${seeds[@]}"; do
  for shard in 0 1; do
    run_shard "${seed}" "${shard}" &
    worker_pids+=("$!")
  done
done

exit_code=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

if (( exit_code == 0 )); then
  python "${repo_root}/scripts/summarize_topic5_history_necessity_v0_1.py" \
    --history-root "${run_root}" \
    --frozen-formal-root "${frozen_root}" \
    --seeds "${seeds[@]}" \
    > "${run_root}/summary.log" 2>&1 || exit_code=1
fi

exit "${exit_code}"
