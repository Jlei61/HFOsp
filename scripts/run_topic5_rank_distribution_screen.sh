#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_tag="${1:-screen_20260725_seed20260725_h32}"
run_root="${repo_root}/results/topic5_interictal_rank_distribution/runs/${run_tag}"
log_root="${run_root}/logs"
mkdir -p "${log_root}"

mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( mem_available_kb < 32 * 1024 * 1024 )); then
  echo "Refusing launch: MemAvailable is below 32 GiB" >&2
  exit 2
fi

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
  echo "Refusing launch: expected 34 subjects, found ${#subjects[@]}" >&2
  exit 3
fi

python "${repo_root}/scripts/monitor_topic5_rank_distribution_resources.py" \
  --pid "$$" \
  --output "${run_root}/resource_log.csv" \
  --interval-seconds 30 &
monitor_pid=$!

exit_code=0
active_pids=()
active_subjects=()
reap_one() {
  local finished_index=""
  while [[ -z "${finished_index}" ]]; do
    for index in "${!active_pids[@]}"; do
      if ! kill -0 "${active_pids[$index]}" 2>/dev/null; then
        finished_index="${index}"
        break
      fi
    done
    [[ -n "${finished_index}" ]] || sleep 1
  done
  if ! wait "${active_pids[$finished_index]}"; then
    exit_code=1
  fi
  unset 'active_pids[finished_index]'
  unset 'active_subjects[finished_index]'
  active_pids=("${active_pids[@]}")
  active_subjects=("${active_subjects[@]}")
}

for subject in "${subjects[@]}"; do
  while (( ${#active_pids[@]} >= 3 )); do
    reap_one
  done
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  conda run --no-capture-output -n cuda_env \
    python "${repo_root}/scripts/train_topic5_interictal_rank_distribution.py" \
      --run-dir "${run_root}/${subject}" \
      --heldout-subject "${subject}" \
      --hidden-size 32 \
      --seed 20260725 \
    > "${log_root}/${subject}.log" 2>&1 &
  active_pids+=("$!")
  active_subjects+=("${subject}")
done
while (( ${#active_pids[@]} )); do
  reap_one
done

python "${repo_root}/scripts/summarize_topic5_rank_distribution_runs.py" \
  --runs-root "${run_root}" \
  --expected-folds 34 \
  --output-prefix screen \
  > "${log_root}/summary.log" 2>&1 || exit_code=1

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${exit_code}"
