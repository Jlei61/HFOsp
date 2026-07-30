#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="results/topic5_interictal_operator_static_readout/runs"
monitor_log="${run_root}/stagea_convergence3_seed20260724_gpu_resource.log"

nvidia-smi dmon -s pucmt -d 1 >"$monitor_log" 2>&1 &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

trainer_pids=()
for subject in e1077 e1084 e139; do
  run="${run_root}/stagea_convergence_${subject}_seed20260724"
  heldout="epilepsiae_${subject#e}"
  /usr/bin/time -v conda run --no-capture-output -n cuda_env \
    python scripts/train_topic5_interictal_operator_stage_a.py \
    --run-dir "$run" \
    --heldout-subject "$heldout" \
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
done

status=0
for trainer_pid in "${trainer_pids[@]}"; do
  if ! wait "$trainer_pid"; then
    status=1
  fi
done
exit "$status"
