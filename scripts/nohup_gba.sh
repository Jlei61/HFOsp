#!/usr/bin/env bash
# Four 90 s trajectories: brake off, sensor-only, and acting at two strengths.
set -u
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
PY=/home/honglab/leijiaxin/anaconda3/bin/python
DIR=results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability
LOG=$DIR/nohup_gba.log
echo "########## global-burst adaptation start $(date '+%F %T')" >>"$LOG"
$PY scripts/run_topic4_fcxr_lc3_gba.py --confirm-run --workers 4 >>"$LOG" 2>&1
echo "########## global-burst adaptation exit=$? $(date '+%F %T')" >>"$LOG"
echo "GBA_WRAPPER_DONE" >>"$LOG"
