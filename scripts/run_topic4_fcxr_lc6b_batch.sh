#!/usr/bin/env bash
# Bounded-parallel launcher for FCXR-LC6B batches (registered extensions, natural-path atlas).
#
# Reads task lines "TAG<TAB>command..." from a file and runs them with at most MAX_WORKERS at a time.
#
# Why not the pool script: that one checks the WHOLE pool's requirement before every launch, which is
# right for a fixed four-way run and wrong for a larger fleet -- it would refuse to start worker 5 on
# a machine that comfortably fits 16.  Here the guard is incremental: before each launch it asks
# whether ONE more worker still leaves the floor intact, re-reading MemAvailable live, so a shared
# machine throttles the fleet instead of overcommitting it.
#
# Measured on this substrate: peak RSS per worker is 6.89 GiB, flat across arm types (it is dominated
# by the substrate, not by spike volume).  PER_WORKER_GIB carries ~30% headroom on top of that.
set -u

ROOT="/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2"
OUT="${ROOT}/results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas"
TASKFILE="${1:?usage: run_topic4_fcxr_lc6b_batch.sh <taskfile> [batch_name]}"
BATCH="${2:-batch}"
LOGS="${OUT}/logs/${BATCH}"
MAX_WORKERS="${MAX_WORKERS:-16}"
PER_WORKER_GIB="${PER_WORKER_GIB:-9}"
FLOOR_GIB="${FLOOR_GIB:-40}"
SWAP_GROWTH_MIB="${SWAP_GROWTH_MIB:-2048}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${LOGS}"
cd "${ROOT}" || exit 1
LOG="${LOGS}/batch.log"

mem_available_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
swap_used_mib()     { awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {printf "%d", (t-f)/1024}' /proc/meminfo; }
say() { echo "$(date -Is) $*" | tee -a "${LOG}"; }

BASELINE_SWAP=$(swap_used_mib)
say "batch ${BATCH} start: max_workers=${MAX_WORKERS} per_worker=${PER_WORKER_GIB}GiB floor=${FLOOR_GIB}GiB mem=$(mem_available_gib)GiB swap=${BASELINE_SWAP}MiB"

pids=()
while IFS=$'\t' read -r tag cmd; do
  [ -z "${tag:-}" ] && continue
  case "${tag}" in \#*) continue ;; esac

  while [ "$(jobs -rp | wc -l)" -ge "${MAX_WORKERS}" ]; do sleep 15; done
  # Incremental guard: does ONE more worker still leave the floor intact, right now?
  held=0
  while [ "$(mem_available_gib)" -lt $(( PER_WORKER_GIB + FLOOR_GIB )) ] || \
        [ "$(swap_used_mib)" -gt $(( BASELINE_SWAP + SWAP_GROWTH_MIB )) ]; do
    if [ "$(jobs -rp | wc -l)" -eq 0 ]; then
      say "ABORT ${tag}: cannot fit even one worker (mem=$(mem_available_gib)GiB swap=$(swap_used_mib)MiB)"
      break 2
    fi
    held=1
    say "resource hold before ${tag}: mem=$(mem_available_gib)GiB swap=$(swap_used_mib)MiB running=$(jobs -rp | wc -l)"
    sleep 60
  done
  [ "${held}" -eq 1 ] && say "resource hold cleared before ${tag}"

  say "launch ${tag}  running=$(jobs -rp | wc -l) mem=$(mem_available_gib)GiB"
  bash -c "${cmd}" > "${LOGS}/${tag}.log" 2>&1 &
  echo "$!" > "${LOGS}/${tag}.pid"
  pids+=("$!:${tag}")
  sleep 4
done < "${TASKFILE}"

fail=0
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"; tag="${entry##*:}"
  if wait "${pid}"; then say "done ${tag}"; else say "FAILED ${tag}"; fail=1; fi
done

say "batch ${BATCH} settled: n=${#pids[@]} fail=${fail} peak_swap=$(swap_used_mib)MiB"
if [ "${fail}" -eq 0 ]; then
  printf '{"status":"DONE","batch":"%s","n":%d}\n' "${BATCH}" "${#pids[@]}" > "${OUT}/DONE_LC6B_${BATCH}.json"
else
  printf '{"status":"FAILED","batch":"%s","n":%d}\n' "${BATCH}" "${#pids[@]}" > "${OUT}/FAILED_LC6B_${BATCH}.json"
fi
say "batch ${BATCH} exit fail=${fail}"
exit "${fail}"
