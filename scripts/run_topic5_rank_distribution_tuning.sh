#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_tag="${1:-tuning_20260725}"
run_root="${repo_root}/results/topic5_interictal_rank_distribution/runs/${run_tag}"
mkdir -p "${run_root}"

mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( mem_available_kb < 32 * 1024 * 1024 )); then
  echo "Refusing launch: MemAvailable is below 32 GiB" >&2
  exit 2
fi

python "${repo_root}/scripts/monitor_topic5_rank_distribution_resources.py" \
  --pid "$$" \
  --output "${run_root}/resource_log.csv" \
  --interval-seconds 15 &
monitor_pid=$!

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/tune_topic5_interictal_rank_distribution.py" \
    --run-dir "${run_root}/selection" \
  > "${run_root}/tuning.log" 2>&1

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
