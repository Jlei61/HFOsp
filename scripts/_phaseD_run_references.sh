#!/usr/bin/env bash
# Phase D: re-detect 3 reference subjects with legacy_align=True.
# Sequential (single GPU).  Logs go to logs/phaseD/<subject>.log.
set -e
cd /home/honglab/leijiaxin/HFOsp

PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
OUT=results/hfo_detection
LOG=logs/phaseD
mkdir -p "$LOG"

for s in gaolan dongyiming wangyiyang; do
  echo "=== [$(date '+%F %T')] starting $s ==="
  "$PY" scripts/run_hfo_detection.py \
    --dataset yuquan \
    --subject "$s" \
    --gpu \
    --output-dir "$OUT" \
    --skip-existing \
    --output-summary "$OUT/_phaseD_${s}_summary.json" \
    > "$LOG/${s}.log" 2>&1
  echo "=== [$(date '+%F %T')] finished $s ==="
done
echo "=== ALL DONE ==="
