#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${1:-${repo_root}/results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0}"
config="${2:-${repo_root}/config/topic5_persistent_path_mode_rnn_v1_0.yaml}"
max_parallel="${3:-8}"
mkdir -p "${run_root}/logs"

mapfile -t subjects < <(
  conda run -n cuda_env python -c "
import pandas as pd
frame = pd.read_csv('${repo_root}/results/topic5_interictal_rank_distribution/dataset_v0_4/subject_audit.csv')
subjects = sorted(frame.loc[frame.status.eq('ok'), 'subject'].astype(str))
assert len(subjects) == 34, len(subjects)
print('\\n'.join(subjects))
"
)
mapfile -t subjects < <(printf '%s\n' "${subjects[@]}" | sed '/^[[:space:]]*$/d')
if [[ "${#subjects[@]}" -ne 34 ]]; then
  echo "Refusing launch: expected 34 non-empty subjects, got ${#subjects[@]}" >&2
  exit 5
fi
seeds=(20260726 20260727 20260728)
specifications=(
  "0 no_history"
  "1 merged_path"
  "2 intact"
  "2 weight_shuffle"
  "2 mode_shuffle"
)

mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( mem_available_kb < 64 * 1024 * 1024 )); then
  echo "Refusing launch: MemAvailable is below 64 GiB" >&2
  exit 4
fi

python - "${run_root}" "${config}" "${max_parallel}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, config, parallel = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
digest = hashlib.sha256(config.read_bytes()).hexdigest()
manifest = {
    "status": "RUNNING",
    "contract": "topic5_persistent_path_mode_rnn_v1_0",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "n_subjects": 34,
    "n_seeds": 3,
    "conditions": [
        "no_history",
        "merged_path",
        "k2_intact",
        "k2_weight_shuffle",
        "k2_mode_shuffle",
    ],
    "expected_runs": 510,
    "formal_shared_coverage_cycles": 2,
    "formal_calibration_coverage_cycles": 4,
    "updates_per_patient": 8,
    "batch_size": 1024,
    "rollouts": 5000,
    "max_parallel": parallel,
    "config": str(config.resolve()),
    "config_sha256": digest,
    "ictal_target_read": False,
}
(root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
PY

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/monitor_topic5_persistent_path_formal.py" \
  --root "${run_root}" \
  --launcher-pid "$$" \
  --watch \
  --interval-seconds 30 \
  > "${run_root}/monitor.log" 2>&1 &
monitor_pid=$!

run_one() {
  local subject="$1"
  local seed="$2"
  local mode_count="$3"
  local control="$4"
  local run_dir="${run_root}/seed_${seed}/k_${mode_count}/${control}/${subject}"
  local state="${run_dir}/run_state.json"
  local log="${run_root}/logs/${subject}_seed${seed}_k${mode_count}_${control}.log"
  local primary_flag=()

  if [[ -z "${subject//[[:space:]]/}" ]]; then
    echo "Refusing empty held-out subject" >&2
    return 6
  fi
  if [[ -f "${state}" ]] && rg -q '"status": "COMPLETE"' "${state}"; then
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    local archived="${run_dir}.interrupted_$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${run_dir}" "${archived}"
  fi
  if [[ "${control}" != "intact" ]]; then
    primary_flag=(--primary-only)
  fi
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=6 \
  MKL_NUM_THREADS=6 \
  conda run --no-capture-output -n cuda_env \
    python "${repo_root}/scripts/train_topic5_persistent_path_rnn.py" \
      --config "${config}" \
      --run-dir "${run_dir}" \
      --heldout-subject "${subject}" \
      --mode-count "${mode_count}" \
      --control "${control}" \
      --seed "${seed}" \
      --device cuda:0 \
      --formal-coverage \
      --coverage-shared-cycles 2 \
      --coverage-calibration-cycles 4 \
      --coverage-updates-per-patient 8 \
      --batch-size 1024 \
      --rollouts 5000 \
      "${primary_flag[@]}" \
      > "${log}" 2>&1
}

running=0
failure=0
for seed in "${seeds[@]}"; do
  for subject in "${subjects[@]}"; do
    for specification in "${specifications[@]}"; do
      read -r mode_count control <<< "${specification}"
      state="${run_root}/seed_${seed}/k_${mode_count}/${control}/${subject}/run_state.json"
      if [[ -f "${state}" ]] && rg -q '"status": "COMPLETE"' "${state}"; then
        continue
      fi
      run_one "${subject}" "${seed}" "${mode_count}" "${control}" &
      running=$((running + 1))
      if [[ "${running}" -ge "${max_parallel}" ]]; then
        if ! wait -n; then
          failure=1
        fi
        running=$((running - 1))
      fi
    done
  done
done

while [[ "${running}" -gt 0 ]]; do
  if ! wait -n; then
    failure=1
  fi
  running=$((running - 1))
done

wait "${monitor_pid}" 2>/dev/null || true
if [[ "${failure}" -eq 0 ]]; then
  conda run --no-capture-output -n cuda_env \
    python "${repo_root}/scripts/analyze_topic5_persistent_path_formal.py" \
      --root "${run_root}" \
      > "${run_root}/analysis.log" 2>&1 || failure=1
fi
exit "${failure}"
