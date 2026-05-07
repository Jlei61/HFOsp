#!/usr/bin/env bash
# Watchdog: wait for the Phase E backfill runner to finish, then immediately
# launch Phase F-1 (yuquan main-cohort re-detection) on the same GPU.
#
# Usage:
#   nohup bash scripts/_phaseE_to_F1_chain.sh > logs/phaseF1/_chain.log 2>&1 &
#
# Behavior:
#   - Polls /proc/<pid> every 60s (cheap, no signals).
#   - When pid is gone, runs scripts/_phaseF1_run_yuquan_main.sh.
#   - Aborts (does NOT launch F-1) if Phase E final log line does not contain
#     "ALL DONE" - this catches partial / failed Phase E runs.
set -e
cd /home/honglab/leijiaxin/HFOsp
mkdir -p logs/phaseF1

PHASE_E_PID="${PHASE_E_PID:-358227}"
RUNNER_LOG="logs/phaseE/_runner.log"
F1_SCRIPT="scripts/_phaseF1_run_yuquan_main.sh"

echo "[$(date '+%F %T')] watchdog started, waiting for pid=$PHASE_E_PID"
while kill -0 "$PHASE_E_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] pid=$PHASE_E_PID gone, validating Phase E completion"

if ! grep -q "Phase E ALL DONE" "$RUNNER_LOG" 2>/dev/null; then
  echo "[ABORT] Phase E runner exited but '$RUNNER_LOG' lacks 'Phase E ALL DONE'."
  echo "[ABORT] Refusing to chain into Phase F-1 - investigate Phase E first."
  exit 2
fi

echo "[$(date '+%F %T')] Phase E completed cleanly. Launching Phase F-1."
exec bash "$F1_SCRIPT"
