#!/usr/bin/env bash
# Managed launcher for the patient-side (zero-simulation) analyses.
#
# Discipline: every long run is nohup-detached and writes PID / log / status.
# Numeric threads are pinned to 1 per process so these never compete with the
# formal cohort workers, which run at five processes x one thread.
#
# SAFETY: nothing here touches src/ or scripts/ (the formal worker hashes its
# imported modules against commit 96618174; a modified module aborts every
# newly launched worker).
#
# usage: launch_patient_jobs.sh <tag> <status_dir> -- <cmd...>
set -euo pipefail

TAG="$1"; shift
STATUS_DIR="$1"; shift
[ "$1" = "--" ] && shift

mkdir -p "$STATUS_DIR"
LOG="$STATUS_DIR/$TAG.log"
STATUS="$STATUS_DIR/$TAG.status"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

nohup bash -c '
  echo "RUNNING pid=$$ started_at=$(date -Is)" > "'"$STATUS"'"
  if "$@" >> "'"$LOG"'" 2>&1; then
    echo "SUCCESS exit_code=0 finished_at=$(date -Is)" > "'"$STATUS"'"
  else
    rc=$?
    echo "FAILED exit_code=$rc finished_at=$(date -Is)" > "'"$STATUS"'"
  fi
' _ "$@" >/dev/null 2>&1 &

echo "launched $TAG pid=$! log=$LOG status=$STATUS"
