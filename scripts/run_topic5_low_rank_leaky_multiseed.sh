#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:?first argument must be selected_hyperparameters.json}"
run_tag="${2:-low_rank_leaky_multiseed_20260725}"
run_root="${repo_root}/results/topic5_low_rank_dynamics/runs/${run_tag}"
mkdir -p "${run_root}"

read -r hidden_size learning_rate offset_dim < <(
  python - "${selection}" <<'PY'
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
  exit 2
fi

python "${repo_root}/scripts/monitor_topic5_rank_distribution_resources.py" \
  --pid "$$" \
  --output "${run_root}/resource_log.csv" \
  --interval-seconds 30 &
resource_pid=$!

run_seed() {
  local seed="$1"
  for rank in 0 1 2 3 4; do
    local rank_root="${run_root}/seed_${seed}/rank_${rank}"
    local log_root="${rank_root}/logs"
    mkdir -p "${log_root}"
    for subject in "${subjects[@]}"; do
      CUDA_VISIBLE_DEVICES=0 \
      OMP_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      conda run --no-capture-output -n cuda_env \
        python "${repo_root}/scripts/train_topic5_low_rank_leaky_rnn.py" \
          --run-dir "${rank_root}/${subject}" \
          --heldout-subject "${subject}" \
          --recurrent-rank "${rank}" \
          --hidden-size "${hidden_size}" \
          --learning-rate "${learning_rate}" \
          --local-offset-dim "${offset_dim}" \
          --seed "${seed}" \
          --batch-size 1024 \
          --shared-cycles 1 \
          --calibration-cycles 4 \
          --updates-per-patient 8 \
          --rollouts 5000 \
          --trajectory-events 500 \
        > "${log_root}/${subject}.log" 2>&1
    done
  done
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
  python "${repo_root}/scripts/summarize_topic5_low_rank_leaky_rnn.py" \
    --root "${run_root}" \
    --seeds "${seeds[@]}" \
    --ranks 0 1 2 3 4 \
    > "${run_root}/summary.log" 2>&1 || exit_code=1
fi
kill "${resource_pid}" 2>/dev/null || true
wait "${resource_pid}" 2>/dev/null || true
exit "${exit_code}"
