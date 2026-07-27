#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/honglab/leijiaxin/HFOsp"
ENV_NAME="cuda_env"
JOBS=4
MODE="formal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --smoke)
      MODE="smoke"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be positive" >&2
  exit 2
fi

cd "$ROOT"
mkdir -p results/topic5_symmetric_axis_propagation_state_v2_2/development/launcher_logs

SUBJECTS=(epilepsiae_1077 epilepsiae_1146 yuquan_chengshuai)
OBJECTIVES=(next_only next_plus_rollout_h3 next_plus_rollout_h5)
SEEDS=(17 29 43)
if [[ "$MODE" == "smoke" ]]; then
  SEEDS=(17)
fi

TASK_COUNT=$((${#SUBJECTS[@]} * ${#OBJECTIVES[@]} * ${#SEEDS[@]}))
if [[ "$MODE" == "formal" && "$TASK_COUNT" -ne 27 ]]; then
  echo "formal task count drift: $TASK_COUNT" >&2
  exit 3
fi
if [[ "$MODE" == "smoke" && "$TASK_COUNT" -ne 9 ]]; then
  echo "smoke task count drift: $TASK_COUNT" >&2
  exit 3
fi

echo "launching $TASK_COUNT development tasks with concurrency $JOBS ($MODE)"
for subject in "${SUBJECTS[@]}"; do
  for objective in "${OBJECTIVES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      running="$(jobs -rp | wc -l || true)"
      while [[ "$running" -ge "$JOBS" ]]; do
        sleep 2
        running="$(jobs -rp | wc -l || true)"
      done
      log="results/topic5_symmetric_axis_propagation_state_v2_2/development/launcher_logs/${MODE}_${subject}_${objective}_seed${seed}.log"
      extra=()
      if [[ "$MODE" == "smoke" ]]; then
        extra=(--smoke)
      fi
      (
        export OMP_NUM_THREADS=4
        export MKL_NUM_THREADS=4
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        conda run --no-capture-output -n "$ENV_NAME" \
          python scripts/train_topic5_symmetric_axis_propagation_state_v2_2.py \
          --subject "$subject" \
          --objective "$objective" \
          --seed "$seed" \
          "${extra[@]}"
      ) >"$log" 2>&1 &
    done
  done
done

wait
echo "development launcher finished ($MODE)"
