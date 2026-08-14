#!/usr/bin/env bash
# Detached LC6A T4/T5 dispatcher.  Launch this script itself with setsid+nohup.
set -euo pipefail

ROOT=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"
PY=/home/honglab/leijiaxin/anaconda3/bin/python
MANIFEST="$ROOT/config/topic4_fcxr_lc6a_patient_axis_surround.json"
MAX_SLOTS=4

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
mkdir -p "$OUT/pids" "$OUT/logs" "$OUT/functional_probes" "$OUT/trajectories"

wait_for_graph_stage() {
  while true; do
    if [[ -f "$OUT/FAILED_LC6A_GRAPH_RECALIBRATION.json" \
       || -f "$OUT/FAILED_LC6A_GRAPH_FAMILY_FINALIZE.json" \
       || -f "$OUT/FAILED_LC6A_TWO_HOP_AUDIT.json" ]]; then
      echo "LC6A graph-only hard gate failed" >&2
      return 1
    fi
    if [[ -f "$OUT/DONE_LC6A_GRAPH_FAMILY.json" \
       && -f "$OUT/DONE_LC6A_TWO_HOP_AUDIT.json" \
       && -f "$OUT/lc5_to_lc6a_authorization.json" ]]; then
      return 0
    fi
    sleep 15
  done
}

check_headroom() {
  local available_kib
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available_kib < 100663296 )); then
    echo "MemAvailable below 96 GiB; refusing a new LC6A arm" >&2
    return 1
  fi
}

run_pool() {
  local stage=$1
  shift
  local -a queue=("$@")
  local -a pids=()
  local -A name_by_pid=()
  local condition pid finished status
  while (( ${#queue[@]} > 0 || ${#pids[@]} > 0 )); do
    while (( ${#queue[@]} > 0 && ${#pids[@]} < MAX_SLOTS )); do
      check_headroom
      condition=${queue[0]}
      queue=("${queue[@]:1}")
      if [[ "$stage" == "functional" ]]; then
        "$PY" scripts/run_topic4_fcxr_lc6a_functional_probe.py run \
          --condition "$condition" --execution-manifest "$MANIFEST" --confirm-run \
          > "$OUT/logs/functional_${condition}.log" 2>&1 &
      elif [[ "$stage" == "natural" ]]; then
        "$PY" scripts/run_topic4_fcxr_lc6a_natural_trajectory.py \
          --condition "$condition" --execution-manifest "$MANIFEST" --confirm-run \
          > "$OUT/logs/natural_${condition}.log" 2>&1 &
      else
        echo "unknown pool stage: $stage" >&2
        return 1
      fi
      pid=$!
      echo "$pid" > "$OUT/pids/${stage}_${condition}.pid"
      pids+=("$pid")
      name_by_pid[$pid]=$condition
      echo "started $stage $condition pid=$pid"
    done
    finished=
    if wait -n -p finished "${pids[@]}"; then
      status=0
    else
      status=$?
    fi
    condition=${name_by_pid[$finished]:-unknown}
    if (( status != 0 )); then
      echo "$stage $condition failed with exit=$status" >&2
      return "$status"
    fi
    echo "finished $stage $condition pid=$finished"
    unset 'name_by_pid[$finished]'
    local -a remaining=()
    for pid in "${pids[@]}"; do
      [[ "$pid" == "$finished" ]] || remaining+=("$pid")
    done
    pids=("${remaining[@]}")
  done
}

wait_for_graph_stage

"$PY" scripts/run_topic4_fcxr_lc6a_functional_probe.py lock \
  --execution-manifest "$MANIFEST" --confirm-run \
  > "$OUT/logs/functional_lock.log" 2>&1

run_pool functional C0 C1 Q1 Q2 Q3

# C0 alone establishes the frozen IED-exposure reference required by all four comparisons.
run_pool natural C0

# Freeze the C0-derived local companion classifier before any Q-arm result exists.
"$PY" scripts/lock_topic4_fcxr_lc6a_local_classifier.py \
  --execution-manifest "$MANIFEST" --confirm-run \
  > "$OUT/logs/local_classifier_lock.log" 2>&1

run_pool natural C1 Q1 Q2 Q3

echo "LC6A fixed T4/T5 experiment block complete"
