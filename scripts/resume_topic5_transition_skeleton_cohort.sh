#!/usr/bin/env bash
set -uo pipefail

cd /home/honglab/leijiaxin/HFOsp

run_root="${1:-results/topic5_structured_axis_graph/cohort_screen_transition_skeleton_v0_7_1}"
config="${2:-config/topic5_transition_skeleton_graph_rnn_v0_7.yaml}"
subject_csv="results/topic5_structured_axis_graph/axis_prior_v1_fast/axis_prior_audit.csv"
max_parallel=9
running=0
failure=0

mkdir -p "${run_root}/logs"

run_one() {
  local subject="$1"
  local seed="$2"
  local rank="$3"
  local control="$4"
  local run_dir="${run_root}/seed_${seed}/rank_${rank}/${control}/${subject}"
  local state="${run_dir}/run_state.json"
  local log="${run_root}/logs/resume_${subject}_seed${seed}_rank${rank}_${control}.log"

  if [[ -f "${state}" ]] && rg -q '"status": "COMPLETE"' "${state}"; then
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    echo "non-complete run directory already exists: ${run_dir}" >&2
    return 2
  fi
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n cuda_env \
    python scripts/train_topic5_axis_graph_rnn.py \
    --config "${config}" \
    --run-dir "${run_dir}" \
    --heldout-subject "${subject}" \
    --structured-rank "${rank}" \
    --seed "${seed}" \
    --prior-control "${control}" \
    --primary-only \
    --device cuda:0 \
    > "${log}" 2>&1
}

for seed in 20260726 20260727 20260728; do
  while IFS=, read -r subject dataset rest; do
    if [[ "${subject}" == "subject" ]]; then
      continue
    fi
    for specification in "0 intact" "1 intact" "1 weight_shuffle"; do
      read -r rank control <<< "${specification}"
      state="${run_root}/seed_${seed}/rank_${rank}/${control}/${subject}/run_state.json"
      if [[ -f "${state}" ]] && rg -q '"status": "COMPLETE"' "${state}"; then
        continue
      fi
      run_one "${subject}" "${seed}" "${rank}" "${control}" &
      running=$((running + 1))
      if [[ "${running}" -ge "${max_parallel}" ]]; then
        if ! wait -n; then
          failure=1
        fi
        running=$((running - 1))
      fi
    done
  done < "${subject_csv}"
done

while [[ "${running}" -gt 0 ]]; do
  if ! wait -n; then
    failure=1
  fi
  running=$((running - 1))
done

exit "${failure}"
