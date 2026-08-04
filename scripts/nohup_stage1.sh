#!/usr/bin/env bash
# Watch the frozen quiet state for 12 s -- twice the slowest observed ignition.
# Queued behind the entry ledgers rather than launched beside them: the runner
# sizes its own pool per round, but the machine is already carrying three 20 s
# trajectories and the lifecycle-closing sweep, so the whole stage waits for real
# headroom before the first worker starts.
set -u
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
PY=/home/honglab/leijiaxin/anaconda3/bin/python
DIR=results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability
LOG=$DIR/nohup_stage1.log
NEED_GIB=150
SWAP0=$(awk '/SwapFree/{f=$2}/SwapTotal/{t=$2}END{printf "%d",(t-f)/1024}' /proc/meminfo)

avail_gib() { awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
swap_delta() { awk -v b="$SWAP0" '/SwapFree/{f=$2}/SwapTotal/{t=$2}END{printf "%d",(t-f)/1024-b}' /proc/meminfo; }

echo "########## stage-1 entry window queued $(date '+%F %T')  need=${NEED_GIB}GiB" >>"$LOG"
waited=0
while [ "$(avail_gib)" -lt "$NEED_GIB" ] || [ "$(swap_delta)" -ge 256 ]; do
  [ $((waited % 1800)) -eq 0 ] && \
    echo "[stage1] waiting: avail=$(avail_gib)GiB swap_delta=$(swap_delta)MiB" >>"$LOG"
  waited=$((waited + 60)); sleep 60
  if [ "$waited" -ge 43200 ]; then
    echo "[stage1] gave up waiting after 12 h" >>"$LOG"; exit 1
  fi
done

echo "########## stage-1 entry window start $(date '+%F %T') at avail=$(avail_gib)GiB" >>"$LOG"
$PY scripts/run_topic4_fcxr_lc3_stage1.py --confirm-run >>"$LOG" 2>&1
echo "########## stage-1 entry window exit=$? $(date '+%F %T')" >>"$LOG"
echo "STAGE1_WRAPPER_DONE" >>"$LOG"
