#!/usr/bin/env bash
# Phase E §9.3 (second stage): batch lagPat / packedTimes backfill for all 11
# yuquan subjects (3 reference + 8 backfill-only). Writes into the raw EDF
# directory; the 3 reference subjects' existing legacy lagPat/packedTimes
# files are first moved into `<subject>/.legacy_backup/`.
#
# This is a CPU/IO job (no GPU); safe to run in parallel with the GPU-bound
# Phase F-1 detection.
set -e
cd /home/honglab/leijiaxin/HFOsp

PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
LOG=logs/phaseE2_packing
mkdir -p "$LOG"

# Order: references first (so their backups happen early and we can spot-check
# vs legacy), then 8 backfill-only.
SUBJECTS=(
  gaolan
  dongyiming
  wangyiyang
  pengzihang
  songzishuo
  zhangbichen
  zhangjiaqi
  zhangkexuan
  zhaochenxi
  zhaojinrui
  zhourongxuan
)

echo "=== [$(date '+%F %T')] Phase E2 packing start: ${#SUBJECTS[@]} subjects ==="
for s in "${SUBJECTS[@]}"; do
  echo "=== [$(date '+%F %T')] starting $s ==="
  "$PY" scripts/run_yuquan_lagpat_backfill.py \
    --subject "$s" \
    > "$LOG/${s}.log" 2>&1
  echo "=== [$(date '+%F %T')] finished $s ==="
done
echo "=== [$(date '+%F %T')] Phase E2 ALL DONE ==="
