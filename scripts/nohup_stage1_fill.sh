#!/usr/bin/env bash
# Tail of the stage-1 grid at an explicit width. The primary pass keeps working
# forward from index 0 with its own pool; this takes index 4 onward, which it
# would not reach for hours. Cells already on disk are skipped by _run_cell.
set -u
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
PY=/home/honglab/leijiaxin/anaconda3/bin/python
DIR=results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability
LOG=$DIR/nohup_stage1_fill.log
echo "########## stage-1 fill start $(date '+%F %T')" >>"$LOG"
$PY scripts/run_topic4_fcxr_lc3_stage1_fill.py --confirm-run --workers 5 --from-index 4 >>"$LOG" 2>&1
echo "########## stage-1 fill exit=$? $(date '+%F %T')" >>"$LOG"
echo "STAGE1_FILL_WRAPPER_DONE" >>"$LOG"
