#!/usr/bin/env bash
set -uo pipefail

cd /home/honglab/leijiaxin/HFOsp

run_root="${1:-results/topic5_structured_axis_graph/screen_persistent_path_mode_v0_9}"
config="${2:-config/topic5_persistent_path_mode_rnn_v0_9.yaml}"
max_parallel="${3:-6}"
running=0
failure=0

subjects=(
  epilepsiae_1073
  epilepsiae_1146
  yuquan_chenziyang
)
seeds=(20260726 20260727 20260728)
specifications=(
  "0 no_history"
  "1 merged_path"
  "1 intact"
  "1 weight_shuffle"
  "2 intact"
  "2 weight_shuffle"
  "2 mode_shuffle"
  "3 intact"
  "3 weight_shuffle"
  "3 mode_shuffle"
  "4 intact"
  "4 weight_shuffle"
  "4 mode_shuffle"
)

mkdir -p "${run_root}/logs"

run_one() {
  local subject="$1"
  local seed="$2"
  local mode_count="$3"
  local control="$4"
  local run_dir="${run_root}/seed_${seed}/k_${mode_count}/${control}/${subject}"
  local state="${run_dir}/run_state.json"
  local log="${run_root}/logs/${subject}_seed${seed}_k${mode_count}_${control}.log"
  local primary_flag=()

  if [[ -f "${state}" ]] && rg -q '"status": "COMPLETE"' "${state}"; then
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    echo "Non-complete run directory already exists: ${run_dir}" >&2
    return 2
  fi
  if [[ "${control}" != "intact" ]]; then
    primary_flag=(--primary-only)
  fi
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n cuda_env \
    python scripts/train_topic5_persistent_path_rnn.py \
    --config "${config}" \
    --run-dir "${run_dir}" \
    --heldout-subject "${subject}" \
    --mode-count "${mode_count}" \
    --control "${control}" \
    --seed "${seed}" \
    --device cuda:0 \
    "${primary_flag[@]}" \
    > "${log}" 2>&1
}

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

exit "${failure}"
