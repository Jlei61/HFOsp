#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_tag="${1:-pilot_20260725_seed20260725_h32}"
run_root="${repo_root}/results/topic5_interictal_rank_distribution/runs/${run_tag}"
log_root="${run_root}/logs"
mkdir -p "${log_root}"

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

subjects=(
  "yuquan_songzishuo"
  "epilepsiae_548"
  "epilepsiae_1073"
)
pids=()
for subject in "${subjects[@]}"; do
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  conda run --no-capture-output -n cuda_env \
    python "${repo_root}/scripts/train_topic5_interictal_rank_distribution.py" \
      --run-dir "${run_root}/${subject}" \
      --heldout-subject "${subject}" \
      --hidden-size 32 \
      --seed 20260725 \
      --pilot \
    > "${log_root}/${subject}.log" 2>&1 &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

python "${repo_root}/scripts/summarize_topic5_rank_distribution_runs.py" \
  --runs-root "${run_root}" \
  --expected-folds 3 \
  --output-prefix pilot \
  > "${log_root}/summary.log" 2>&1 || exit_code=1

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${exit_code}"
