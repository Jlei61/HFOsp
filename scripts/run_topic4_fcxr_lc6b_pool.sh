#!/usr/bin/env bash
# FCXR-LC6B: run the eight registered clamp arms with a bounded worker pool, then finalize.
#
# Resource contract: each arm is a single-threaded process whose measured peak RSS on this substrate
# is ~6.9 GiB (LC6A resource_log).  Before starting a worker the pool re-reads MemAvailable and refuses
# to add one that would not fit; it never removes a finished product.
set -u

ROOT="/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2"
OUT="${ROOT}/results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas"
LOGS="${OUT}/logs"
RUNNER="${ROOT}/scripts/run_topic4_fcxr_lc6b_clamp_forks.py"
MAX_WORKERS=4
PER_WORKER_GIB=8
MARGIN_GIB=24

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${LOGS}"
cd "${ROOT}" || exit 1

mem_available_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
swap_used_mib()     { awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {printf "%d", (t-f)/1024}' /proc/meminfo; }

BASELINE_SWAP=$(swap_used_mib)
echo "$(date -Is) pool start  mem_available=$(mem_available_gib)GiB swap=${BASELINE_SWAP}MiB" \
  | tee -a "${LOGS}/pool.log"

pids=()
for snapshot in S2 S4; do
  for arm in NAT H_CLAMP D_CLAMP DH_CLAMP; do
    tag="${snapshot}_${arm}"
    if [ -f "${OUT}/forks/${tag}/summary.json" ]; then
      echo "$(date -Is) skip ${tag} (already complete)" | tee -a "${LOGS}/pool.log"
      continue
    fi
    # wait for a free slot
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_WORKERS}" ]; do sleep 20; done
    need=$(( MAX_WORKERS * PER_WORKER_GIB + MARGIN_GIB ))
    while [ "$(mem_available_gib)" -lt "${need}" ] || \
          [ "$(swap_used_mib)" -gt $(( BASELINE_SWAP + 2048 )) ]; do
      echo "$(date -Is) resource hold before ${tag}: mem=$(mem_available_gib)GiB swap=$(swap_used_mib)MiB" \
        | tee -a "${LOGS}/pool.log"
      sleep 60
    done
    echo "$(date -Is) launch ${tag}  mem=$(mem_available_gib)GiB" | tee -a "${LOGS}/pool.log"
    python "${RUNNER}" run --snapshot "${snapshot}" --arm "${arm}" --confirm-run \
      > "${LOGS}/${tag}.log" 2>&1 &
    pids+=("$!:${tag}")
    sleep 5
  done
done

fail=0
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"; tag="${entry##*:}"
  if wait "${pid}"; then
    echo "$(date -Is) done ${tag}" | tee -a "${LOGS}/pool.log"
  else
    echo "$(date -Is) FAILED ${tag}" | tee -a "${LOGS}/pool.log"
    fail=1
  fi
done

echo "$(date -Is) all arms settled (fail=${fail}); finalizing" | tee -a "${LOGS}/pool.log"
if [ "${fail}" -eq 0 ]; then
  python "${RUNNER}" finalize --confirm-run > "${LOGS}/finalize.log" 2>&1
  status=$?
else
  status=2
fi

if [ "${status}" -eq 0 ]; then
  printf '{"status":"DONE","stage":"POOL","fail":%d}\n' "${fail}" > "${OUT}/DONE_LC6B_POOL.json"
else
  printf '{"status":"FAILED","stage":"POOL","fail":%d,"finalize_status":%d}\n' \
    "${fail}" "${status}" > "${OUT}/FAILED_LC6B_POOL.json"
fi
echo "$(date -Is) pool exit status=${status}" | tee -a "${LOGS}/pool.log"
exit "${status}"
