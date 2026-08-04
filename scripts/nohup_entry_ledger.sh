#!/usr/bin/env bash
# Entry ledger for the three no-kick seeds.  Concurrency is decided by free
# memory at each launch rather than fixed: a 20 s run holds ~12 GiB of E spike
# bools plus the substrate, and the lifecycle-closing sweep is still resident and
# still growing, so a fixed width would either waste the machine or overcommit it.
set -u
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
PY=/home/honglab/leijiaxin/anaconda3/bin/python
DIR=results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability
LOG=$DIR/nohup_entry_ledger.log
NEED_GIB=110
SWAP0=$(awk '/SwapFree/{f=$2}/SwapTotal/{t=$2}END{printf "%d",(t-f)/1024}' /proc/meminfo)

avail_gib() { awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
swap_delta() { awk -v b="$SWAP0" '/SwapFree/{f=$2}/SwapTotal/{t=$2}END{printf "%d",(t-f)/1024-b}' /proc/meminfo; }

echo "########## entry ledger start $(date '+%F %T')  swap_baseline=${SWAP0}MiB" >>"$LOG"

pids=()
for seed in 406 401 405; do
  # Wait for room.  Also back off if swap starts moving: that is the signal that
  # something already resident is being pushed out, whoever owns it.
  waited=0
  while [ "$(avail_gib)" -lt "$NEED_GIB" ] || [ "$(swap_delta)" -ge 256 ]; do
    if [ "$waited" -eq 0 ]; then
      echo "[entry] seed $seed waiting: avail=$(avail_gib)GiB swap_delta=$(swap_delta)MiB" >>"$LOG"
    fi
    waited=$((waited + 60)); sleep 60
    if [ "$waited" -ge 21600 ]; then
      echo "[entry] seed $seed gave up waiting after 6 h" >>"$LOG"; continue 2
    fi
  done
  echo "[entry] launching seed $seed at avail=$(avail_gib)GiB $(date '+%T')" >>"$LOG"
  $PY scripts/run_topic4_fcxr_lc3_entry.py --noise "$seed" --confirm-run \
      >>"$DIR/entry_seed${seed}.log" 2>&1 &
  pids+=($!)
  sleep 180   # let the substrate and spike buffer land before sizing the next
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "########## entry ledger exit=$fail $(date '+%F %T')" >>"$LOG"
echo "ENTRY_LEDGER_WRAPPER_DONE" >>"$LOG"
