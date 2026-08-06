#!/usr/bin/env bash
set -u
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
PY=/home/honglab/leijiaxin/anaconda3/bin/python
DIR=results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability
LOG=$DIR/nohup_subject_runaway.log
echo "########## subject runaway start $(date '+%F %T')" >>"$LOG"
$PY scripts/run_topic4_fcxr_lc3_subject_runaway.py --confirm-run --workers 2 >>"$LOG" 2>&1
echo "########## subject runaway exit=$? $(date '+%F %T')" >>"$LOG"
echo "SUBJECT_RUNAWAY_WRAPPER_DONE" >>"$LOG"
