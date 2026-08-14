#!/usr/bin/env bash
# Detached LC6A T4/T5 dispatcher.  Launch this script itself with setsid+nohup.
set -euo pipefail

ROOT=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"
PY=/home/honglab/leijiaxin/anaconda3/bin/python
MANIFEST="$ROOT/config/topic4_fcxr_lc6a_patient_axis_surround.json"
MAX_SLOTS=4
RSS_BUDGET_GIB=8.0
STAGE_SWAP_BASELINE_MIB=0
SUBMISSION_SLOT_CAP=1

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

swap_used_mib() {
  awk '
    /^SwapTotal:/ {total=$2}
    /^SwapFree:/  {free=$2}
    END {printf "%.3f", (total-free)/1024.0}
  ' /proc/meminfo
}

check_headroom() {
  local available_kib swap_now_mib swap_delta_mib required_kib slots_by_memory
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  swap_now_mib=$(swap_used_mib)
  swap_delta_mib=$(awk -v now="$swap_now_mib" -v base="$STAGE_SWAP_BASELINE_MIB" \
    'BEGIN {printf "%.3f", now-base}')
  if awk -v delta="$swap_delta_mib" 'BEGIN {exit !(delta >= 256.0)}'; then
    echo "stage swap delta is +${swap_delta_mib} MiB; pausing new LC6A submissions" >&2
    return 1
  fi
  required_kib=$(awk -v rss="$RSS_BUDGET_GIB" \
    'BEGIN {printf "%.0f", 3.0*rss*1024.0*1024.0}')
  if (( available_kib < 100663296 )); then
    echo "MemAvailable below 96 GiB; pausing new LC6A submissions" >&2
    return 1
  fi
  if (( available_kib < required_kib )); then
    echo "MemAvailable below 3x measured per-arm RSS budget; pausing submission" >&2
    return 1
  fi
  slots_by_memory=$(awk -v kib="$available_kib" -v rss="$RSS_BUDGET_GIB" \
    'BEGIN {n=int(kib/(3.0*rss*1024.0*1024.0)); if (n<1) n=1; print n}')
  SUBMISSION_SLOT_CAP=$slots_by_memory
  (( SUBMISSION_SLOT_CAP > MAX_SLOTS )) && SUBMISSION_SLOT_CAP=$MAX_SLOTS
  return 0
}

remeasure_rss_budget() {
  "$PY" - <<'PY'
import json
from pathlib import Path

path = Path("results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/resource_log.jsonl")
rows = []
if path.is_file():
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if str(row.get("stage", "")).startswith("LC6A_C0_CHUNK"):
            rows.append(float(row["self_peak_rss_gib"]))
if not rows:
    raise SystemExit("C0 natural arm did not publish a measured RSS budget")
print(f"{max(rows):.6f}")
PY
}

run_pool() {
  local stage=$1
  shift
  local -a queue=("$@")
  local -a pids=()
  local -A name_by_pid=()
  local condition pid finished status
  STAGE_SWAP_BASELINE_MIB=$(swap_used_mib)
  while (( ${#queue[@]} > 0 || ${#pids[@]} > 0 )); do
    while (( ${#queue[@]} > 0 )); do
      if ! check_headroom; then
        break
      fi
      if (( ${#pids[@]} >= SUBMISSION_SLOT_CAP )); then
        break
      fi
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
      elif [[ "$stage" == "gain" ]]; then
        "$PY" scripts/run_topic4_fcxr_lc6a_gain_forks.py run \
          --condition "$condition" --execution-manifest "$MANIFEST" --confirm-run \
          > "$OUT/logs/gain_${condition}.log" 2>&1 &
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
    if (( ${#pids[@]} == 0 )); then
      # A temporary machine-level resource condition is not a scientific
      # failure.  Keep the detached dispatcher alive and retry without
      # consuming or dropping a registered arm.
      sleep 30
      continue
    fi
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

# C0 is the first full natural arm under the current engine path.  Replace the
# conservative inherited 8 GiB estimate with its measured peak before filling
# the four-arm natural pool.
RSS_BUDGET_GIB=$(remeasure_rss_budget)
echo "measured LC6A natural-arm RSS budget: ${RSS_BUDGET_GIB} GiB"

# Freeze the C0-derived local companion classifier before any Q-arm result exists.
"$PY" scripts/lock_topic4_fcxr_lc6a_local_classifier.py \
  --execution-manifest "$MANIFEST" --confirm-run \
  > "$OUT/logs/local_classifier_lock.log" 2>&1

run_pool natural C1 Q1 Q2 Q3

"$PY" scripts/aggregate_topic4_fcxr_lc6a_phenotypes.py --confirm-run \
  > "$OUT/logs/phenotype_map.log" 2>&1

"$PY" scripts/run_topic4_fcxr_lc6a_gain_forks.py lock --confirm-run \
  > "$OUT/logs/gain_lock.log" 2>&1

mapfile -t gain_conditions < <(
  "$PY" - <<'PY'
import json
from pathlib import Path
path = Path("results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround/gain_fork_lock.json")
for row in json.loads(path.read_text())["selected"]:
    print(row["condition"])
PY
)
run_pool gain "${gain_conditions[@]}"

"$PY" scripts/run_topic4_fcxr_lc6a_gain_forks.py finalize --confirm-run \
  > "$OUT/logs/gain_finalize.log" 2>&1

"$PY" scripts/run_topic4_fcxr_lc6a_confirmation.py build \
  --execution-manifest "$MANIFEST" --confirm-run \
  > "$OUT/logs/confirmation_build.log" 2>&1

confirmation_authorized=$(
  "$PY" - <<'PY'
import json
from pathlib import Path
path = Path("results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround/confirmation_lock.json")
print("1" if json.loads(path.read_text()).get("authorized") else "0")
PY
)
if [[ "$confirmation_authorized" == "1" ]]; then
  mapfile -t confirmation_args < <(
    "$PY" - <<'PY'
import json
from pathlib import Path
path = Path("results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround/confirmation_lock.json")
row = json.loads(path.read_text())
print(row["parent_condition"])
print(row["output_condition"])
print(row["graph_artifact"])
PY
  )
  check_headroom
  "$PY" scripts/run_topic4_fcxr_lc6a_natural_trajectory.py \
    --condition "${confirmation_args[0]}" \
    --output-condition "${confirmation_args[1]}" \
    --graph-artifact "${confirmation_args[2]}" \
    --confirmation-lock "$OUT/confirmation_lock.json" \
    --execution-manifest "$MANIFEST" --confirm-run \
    > "$OUT/logs/confirmation_natural.log" 2>&1
fi

"$PY" scripts/run_topic4_fcxr_lc6a_confirmation.py finalize --confirm-run \
  > "$OUT/logs/confirmation_finalize.log" 2>&1

echo "LC6A fixed T4/T5, conditional T7, and conditional confirmation blocks complete"
