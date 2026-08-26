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
from src.topic5_continuous_marked_state_r1 import contract
root = contract.RESULT_ROOT / "r1_2"
contract.atomic_json(root / "RUN_STATUS.json", {
    "status": sys.argv[1], "stage": sys.argv[2],
    "contract": contract.REVISION, "sealed_opened": False,
})
PY
}

# Refuse to start any fitting until every independently generated cache has an
# atomic COMPLETE manifest.  This is a readiness check, not a scientific gate.
for subject in "${SUBJECTS[@]}"; do
  "$PY" - "$subject" <<'PY' || exit 42
import json, sys
from src.topic5_continuous_marked_state_r1 import contract
p = contract.RESULT_ROOT / "r1_2/cache" / sys.argv[1] / "manifest.json"
d = json.loads(p.read_text())
assert d["status"] == "COMPLETE" and d["sealed_opened"] is False
assert d["n_unreadable_anchors"] == 0
PY
done

status RUNNING t1_parallel_two_workers
for subject in "${SUBJECTS[@]}"; do
  pids=()
  stages=()
  for arm in explicit explicit_raw; do
    stage="t1_${subject}_${arm}"
    result="$ROOT/t1_full/$subject/${arm}_d8_seed_0/result.json"
    if "$PY" - "$result" "$ROOT/cache/$subject/manifest.json" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
from src.topic5_continuous_marked_state_r1 import contract
result_path, cache_path = map(Path, sys.argv[1:])
r = json.loads(result_path.read_text())
assert r["status"] == "COMPLETE" and r["sealed_opened"] is False
assert r["cache_manifest_sha256"] == contract.sha256_file(cache_path)
PY
    then
      continue
    fi
    stages+=("$stage")
    "$PY" scripts/topic5_continuous_marked_state_r1/run_r1_2_t1.py \
      --subject "$subject" --arm "$arm" --device cuda \
      --epochs 4 --chunk-anchors 256 >"$LOG/${stage}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      status FAILED "${stages[$index]}"
      failed=1
    fi
  done
  (( failed == 0 )) || exit 51
done

status RUNNING denominators_final
"$PY" scripts/topic5_continuous_marked_state_r1/build_r1_2_denominators.py \
  >"$LOG/denominators_final.log" 2>&1 || {
    status FAILED denominators_final
    exit 52
  }
status RUNNING aggregate
"$PY" scripts/topic5_continuous_marked_state_r1/aggregate_r1_2.py \
  >"$LOG/aggregate.log" 2>&1 || {
    status FAILED aggregate
    exit 61
  }
status RUNNING audit
"$PY" scripts/topic5_continuous_marked_state_r1/audit_r1_2_package.py \
  >"$LOG/audit.log" 2>&1 || {
    status FAILED audit
    exit 62
  }
status COMPLETE complete
