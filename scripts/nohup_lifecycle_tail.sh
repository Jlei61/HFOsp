#!/usr/bin/env bash
# Re-run the free-wear arm to 70 s with the event ledger, so the registered
# return test can decide instead of a 2 s workpoint label. Queued behind the
# stage-1 grid: a 70 s arm peaks near 48 GiB.
set -u
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
PY=/home/honglab/leijiaxin/anaconda3/bin/python
DIR=results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability
LOG=$DIR/nohup_lifecycle_tail.log
NEED_GIB=110
SWAP0=$(awk '/SwapFree/{f=$2}/SwapTotal/{t=$2}END{printf "%d",(t-f)/1024}' /proc/meminfo)
avail() { awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
swapd() { awk -v b="$SWAP0" '/SwapFree/{f=$2}/SwapTotal/{t=$2}END{printf "%d",(t-f)/1024-b}' /proc/meminfo; }

echo "########## lifecycle tail queued $(date '+%F %T') need=${NEED_GIB}GiB" >>"$LOG"
w=0
while [ "$(avail)" -lt "$NEED_GIB" ] || [ "$(swapd)" -ge 256 ]; do
  [ $((w % 1800)) -eq 0 ] && echo "[tail] waiting: avail=$(avail)GiB" >>"$LOG"
  w=$((w+60)); sleep 60
  [ "$w" -ge 43200 ] && { echo "[tail] gave up after 12 h" >>"$LOG"; exit 1; }
done
echo "########## lifecycle tail start $(date '+%F %T') at avail=$(avail)GiB" >>"$LOG"
$PY scripts/run_topic4_fcxr_lc3_lifecycle_tail.py --confirm-run >>"$LOG" 2>&1
echo "########## lifecycle tail exit=$? $(date '+%F %T')" >>"$LOG"
echo "LIFECYCLE_TAIL_WRAPPER_DONE" >>"$LOG"
