#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tuning_result="${1:?first argument must be selected_hyperparameters.json}"
run_tag="${2:-formal_multiseed_20260725}"
formal_root="${repo_root}/results/topic5_interictal_rank_distribution/runs/${run_tag}"
mkdir -p "${formal_root}"

if [[ ! -f "${tuning_result}" ]]; then
  echo "Missing tuning result: ${tuning_result}" >&2
  exit 2
fi
read -r hidden_size learning_rate offset_dim < <(
  python - "${tuning_result}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))["selected_hyperparameters"]
print(value["hidden_size"], value["learning_rate"], value["local_offset_dim"])
PY
)

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

python "${repo_root}/scripts/monitor_topic5_rank_distribution_resources.py" \
  --pid "$$" \
  --output "${formal_root}/resource_log.csv" \
  --interval-seconds 30 &
monitor_pid=$!

run_seed() {
  local seed="$1"
  local seed_root="${formal_root}/seed_${seed}"
  local log_root="${seed_root}/logs"
  mkdir -p "${log_root}"
  for subject in "${subjects[@]}"; do
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    conda run --no-capture-output -n cuda_env \
      python "${repo_root}/scripts/train_topic5_interictal_rank_distribution.py" \
        --run-dir "${seed_root}/${subject}" \
        --heldout-subject "${subject}" \
        --hidden-size "${hidden_size}" \
        --learning-rate "${learning_rate}" \
        --local-offset-dim "${offset_dim}" \
        --seed "${seed}" \
        --batch-size 1024 \
        --formal-coverage \
        --coverage-shared-cycles 1 \
        --coverage-calibration-cycles 4 \
        --coverage-updates-per-patient 8 \
      > "${log_root}/${subject}.log" 2>&1
  done
  python "${repo_root}/scripts/summarize_topic5_rank_distribution_runs.py" \
    --runs-root "${seed_root}" \
    --expected-folds 34 \
    --output-prefix formal \
    > "${log_root}/summary.log" 2>&1
}

seeds=(20260725 20260726 20260727)
seed_pids=()
for seed in "${seeds[@]}"; do
  run_seed "${seed}" &
  seed_pids+=("$!")
done

exit_code=0
for pid in "${seed_pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

if (( exit_code == 0 )); then
  python "${repo_root}/scripts/summarize_topic5_rank_distribution_multiseed.py" \
    --formal-root "${formal_root}" \
    --seeds "${seeds[@]}" \
    > "${formal_root}/summary.log" 2>&1 || exit_code=1
fi

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${exit_code}"
