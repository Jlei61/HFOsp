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
path = "results/topic5_symmetric_axis_propagation_state_v2_2/formal/PHYSICAL_AXIS_FORMAL_LOCK.json"
for subject in json.load(open(path, encoding="utf-8"))["subjects"]:
    print(subject)
PY
)
SEEDS=(17 29 43)
TASK_COUNT=$((${#SUBJECTS[@]} * ${#SEEDS[@]}))
if [[ "${#SUBJECTS[@]}" -ne 22 || "$TASK_COUNT" -ne 66 ]]; then
  echo "formal Claim-4 task count drift: subjects=${#SUBJECTS[@]} tasks=$TASK_COUNT" >&2
  exit 3
fi

resource_ready() {
  local available_kb gpu_used gpu_free gpu_total
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  [[ "$available_kb" -ge $((32 * 1024 * 1024)) ]] || return 1
  IFS=',' read -r gpu_used gpu_free < <(
    nvidia-smi \
      --query-gpu=memory.used,memory.free \
      --format=csv,noheader,nounits | head -1
  )
  gpu_used="${gpu_used// /}"
  gpu_free="${gpu_free// /}"
  gpu_total=$((gpu_used + gpu_free))
  [[ "$gpu_free" -ge $((gpu_total / 5)) ]]
}

echo "launching $TASK_COUNT formal Claim-4 tasks with concurrency $JOBS"
PIDS=()
LABELS=()
for subject in "${SUBJECTS[@]}"; do
  [[ -n "$subject" ]] || { echo "empty subject" >&2; exit 4; }
  for seed in "${SEEDS[@]}"; do
    while [[ "$(jobs -rp | wc -l || true)" -ge "$JOBS" ]]; do
      sleep 5
    done
    while ! resource_ready; do
      echo "resource guard paused dispatch at $(date --iso-8601=seconds)"
      sleep 30
    done
    log="$LOG_ROOT/claim4_${subject}_seed${seed}.log"
    (
      export OMP_NUM_THREADS=4
      export MKL_NUM_THREADS=4
      export CUBLAS_WORKSPACE_CONFIG=:4096:8
      conda run --no-capture-output -n "$ENV_NAME" \
        python scripts/train_topic5_symmetric_axis_formal_claim4_v2_2.py \
          --subject "$subject" \
          --seed "$seed"
    ) >"$log" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${subject}/seed_${seed}")
  done
done
FAILED=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "Claim-4 task failed: ${LABELS[$index]}" >&2
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "formal Claim-4 launcher stopped because one or more tasks failed" >&2
  exit 5
fi
conda run --no-capture-output -n "$ENV_NAME" \
  python scripts/analyze_topic5_symmetric_axis_formal_claim4_v2_2.py
python - <<'PY'
import json
from pathlib import Path
path = Path("results/topic5_symmetric_axis_propagation_state_v2_2/formal/analysis/CLAIM4_STATUS.json")
status = json.loads(path.read_text(encoding="utf-8"))
if status.get("status") not in {"complete", "not_estimable"}:
    raise SystemExit(f"Claim-4 analyzer did not finish: {status.get('status')}")
PY
echo "formal Claim-4 launcher and analysis finished"
