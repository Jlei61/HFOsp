#!/usr/bin/env bash
set -uo pipefail

REPO=/home/honglab/leijiaxin/HFOsp
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
ROOT="$REPO/results/epi_prssm/continuous_marked_state/r1/r1_2"
LOG="$ROOT/logs"
SUBJECTS=(
  epilepsiae_620 epilepsiae_958 epilepsiae_139
  yuquan_huanghanwen yuquan_zhangjiaqi yuquan_hanyuxuan
)

cd "$REPO" || exit 2
mkdir -p "$LOG"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:${LD_LIBRARY_PATH:-}"

status() {
  "$PY" - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from src.topic5_continuous_marked_state_r1 import contract
root = contract.RESULT_ROOT / "r1_2"
contract.atomic_json(root / "RUN_STATUS.json", {
    "status": sys.argv[1], "stage": sys.argv[2],
    "contract": contract.REVISION, "sealed_opened": False,
})
PY
}

run_logged() {
  local stage="$1"
  shift
  status RUNNING "$stage"
  if ! "$@" >"$LOG/${stage}.log" 2>&1; then
    status FAILED "$stage"
    return 1
  fi
}

for subject in "${SUBJECTS[@]}"; do
  run_logged "coverage_${subject}" "$PY" \
    scripts/topic5_continuous_marked_state_r1/build_r1_2_coverage.py \
    --subject "$subject" || exit 11
done
run_logged denominators "$PY" \
  scripts/topic5_continuous_marked_state_r1/build_r1_2_denominators.py || exit 12

for subject in "${SUBJECTS[@]}"; do
  run_logged "baseline_${subject}" "$PY" \
    scripts/topic5_continuous_marked_state_r1/run_r1_2_baseline.py \
    --subject "$subject" --device cuda --mark-epochs 30 \
    --mark-batch-size 1024 || exit 21
done

for subject in "${SUBJECTS[@]}"; do
  run_logged "bridge_${subject}" "$PY" \
    scripts/topic5_continuous_marked_state_r1/run_r1_2_bridge.py \
    --subject "$subject" --device cuda --epochs 6 \
    --max-train-anchors 256 --max-validation-anchors 128 \
    --anchor-batch-size 4 || exit 31
done

for subject in "${SUBJECTS[@]}"; do
  run_logged "cache_${subject}" "$PY" \
    scripts/topic5_continuous_marked_state_r1/run_r1_2_cache.py \
    --subject "$subject" --device cuda --anchor-batch-size 8 || exit 41
done

for subject in "${SUBJECTS[@]}"; do
  for arm in explicit explicit_raw; do
    run_logged "t1_${subject}_${arm}" "$PY" \
      scripts/topic5_continuous_marked_state_r1/run_r1_2_t1.py \
      --subject "$subject" --arm "$arm" --device cuda \
      --epochs 4 --chunk-anchors 256 || exit 51
  done
done

run_logged aggregate "$PY" \
  scripts/topic5_continuous_marked_state_r1/aggregate_r1_2.py || exit 61
run_logged audit "$PY" \
  scripts/topic5_continuous_marked_state_r1/audit_r1_2_package.py || exit 62
status COMPLETE complete
