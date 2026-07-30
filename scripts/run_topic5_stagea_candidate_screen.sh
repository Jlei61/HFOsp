#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

seed="${TOPIC5_STAGEA_SEED:-20260724}"
hidden_size="${TOPIC5_STAGEA_HIDDEN_SIZE:-32}"
max_parallel="${TOPIC5_STAGEA_MAX_PARALLEL:-3}"
run_prefix="${TOPIC5_STAGEA_RUN_PREFIX:-stagea_screen}"
run_root="results/topic5_interictal_operator_static_readout/runs"

if [[ "$#" -gt 0 ]]; then
  subjects=("$@")
else
  subjects=(1077 1084 1096 1125 1150 139 253 384 548 620 635 922 958)
fi

monitor_log="${run_root}/${run_prefix}_h${hidden_size}_seed${seed}_gpu_resource.log"
nvidia-smi dmon -s pucmt -d 1 >"$monitor_log" 2>&1 &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

trainer_pids=()
status=0
for subject in "${subjects[@]}"; do
  run="${run_root}/${run_prefix}_e${subject}_h${hidden_size}_seed${seed}"
  if [[ -e "$run" ]]; then
    echo "Refusing to overwrite existing run: $run" >&2
    exit 2
  fi
  /usr/bin/time -v conda run --no-capture-output -n cuda_env \
    python scripts/train_topic5_interictal_operator_stage_a.py \
    --run-dir "$run" \
    --heldout-subject "epilepsiae_${subject}" \
    --hidden-size "$hidden_size" \
    --smoke \
    --epochs 24 \
    --calibration-epochs 12 \
    --steps-per-epoch 48 \
    --calibration-steps-per-epoch 24 \
    --batch-size 256 \
    --include-rank-shuffle-control \
    --include-static-neural-controls \
    >"${run}_stdout.log" 2>"${run}_time_resource.log" &
  trainer_pids+=("$!")

  if [[ "${#trainer_pids[@]}" -ge "$max_parallel" ]]; then
    for trainer_pid in "${trainer_pids[@]}"; do
      if ! wait "$trainer_pid"; then
        status=1
      fi
    done
    trainer_pids=()
    if [[ "$status" -ne 0 ]]; then
      exit "$status"
    fi
  fi
done

for trainer_pid in "${trainer_pids[@]}"; do
  if ! wait "$trainer_pid"; then
    status=1
  fi
done
exit "$status"
