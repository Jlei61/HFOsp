#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/honglab/leijiaxin/HFOsp"
ENV_NAME="cuda_env"
JOBS=6

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
LOG_ROOT="results/topic5_symmetric_axis_propagation_state_v2_2/formal/launcher_logs"
mkdir -p "$LOG_ROOT"
mapfile -t SUBJECTS < <(
  python - <<'PY'
import json
path = "results/topic5_symmetric_axis_propagation_state_v2_2/input_audit/physical_axis_formal_cohort.json"
for subject in json.load(open(path, encoding="utf-8"))["subjects"]:
    print(subject)
PY
)
SEEDS=(17 29 43)
TASK_COUNT=$((${#SUBJECTS[@]} * ${#SEEDS[@]}))
if [[ "${#SUBJECTS[@]}" -ne 22 || "$TASK_COUNT" -ne 66 ]]; then
  echo "formal Claim-2 task count drift: subjects=${#SUBJECTS[@]} tasks=$TASK_COUNT" >&2
  exit 3
fi

echo "launching $TASK_COUNT formal Claim-2 tasks with concurrency $JOBS"
for heldout in "${SUBJECTS[@]}"; do
  if [[ -z "$heldout" ]]; then
    echo "empty heldout subject" >&2
    exit 4
  fi
  for seed in "${SEEDS[@]}"; do
    running="$(jobs -rp | wc -l || true)"
    while [[ "$running" -ge "$JOBS" ]]; do
      sleep 5
      running="$(jobs -rp | wc -l || true)"
    done
    log="$LOG_ROOT/claim2_${heldout}_seed${seed}.log"
    (
      export OMP_NUM_THREADS=4
      export MKL_NUM_THREADS=4
      export CUBLAS_WORKSPACE_CONFIG=:4096:8
      conda run --no-capture-output -n "$ENV_NAME" \
        python scripts/train_topic5_symmetric_axis_formal_claim2_v2_2.py \
        --heldout-subject "$heldout" \
        --seed "$seed"
    ) >"$log" 2>&1 &
  done
done
wait
echo "formal Claim-2 launcher finished"

