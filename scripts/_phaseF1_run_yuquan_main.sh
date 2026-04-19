#!/usr/bin/env bash
# Phase F-1: re-detect the 10 Yuquan main-cohort subjects with legacy_align=True.
#
# Cohort = 13 yuquan subjects with existing legacy lagPat
#        - { gaolan, dongyiming, wangyiyang }            (Phase D, already done)
#        = 10 main-cohort subjects covered here.
#
# Pre-Phase-D detections were moved to
#   results/hfo_detection/<subject>.pre_phaseD_backup/
# before this script.
#
# Sequential single-GPU run (this script does NOT parallelize: a second concurrent
# GPU detection on the same RTX 3090 would contend SM time and risk OOM with
# cuFFT cache + 3.8 GiB peak per worker).
#
# Logs: logs/phaseF1/<subject>.log
# Per-subject summaries: results/hfo_detection/_phaseF1_<subject>_summary.json
set -e
cd /home/honglab/leijiaxin/HFOsp

PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
OUT=results/hfo_detection
LOG=logs/phaseF1
mkdir -p "$LOG"

SUBJECTS=(
  chengshuai
  chenziyang
  hanyuxuan
  huanghanwen
  huangwanling
  litengsheng
  liyouran
  sunyuanxin
  xuxinyi
  zhangjinhan
)

echo "=== [$(date '+%F %T')] Phase F-1 start: ${#SUBJECTS[@]} subjects ==="
for s in "${SUBJECTS[@]}"; do
  echo "=== [$(date '+%F %T')] starting $s ==="
  "$PY" scripts/run_hfo_detection.py \
    --dataset yuquan \
    --subject "$s" \
    --gpu \
    --output-dir "$OUT" \
    --skip-existing \
    --output-summary "$OUT/_phaseF1_${s}_summary.json" \
    > "$LOG/${s}.log" 2>&1
  echo "=== [$(date '+%F %T')] finished $s ==="
done
echo "=== [$(date '+%F %T')] Phase F-1 ALL DONE ==="
