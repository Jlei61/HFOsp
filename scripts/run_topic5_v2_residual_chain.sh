#!/bin/bash
# Topic5 V2 Phase-1-v2 — residual alignment + cohort-perm nulls chain (committed launcher).
#
# Runs the {common_resid, aperiodic_resid} x {narrow, broad} residual-survival groups end to end:
#   1. residual alignment (fast, sequential per group)
#   2. cohort-perm nulls (parallel disjoint-subject workers -> _partial_{feature}/, resumable)
#   3. full-list aggregation per group (resumes all subjects -> the combined output)
# and writes a run manifest so a later audit can tell which combined artifact a run produced.
#
# WHY THIS IS A COMMITTED SCRIPT (not scratchpad memory):
#   * OMP/BLAS single-threading is baked in below. Each null worker is single-process Python that
#     permutes in a loop; without this every worker grabs all cores for BLAS and oversubscribes
#     (26 workers x 80 cores) -> crawl (~7h vs ~40min). This MUST NOT live only in a report.
#   * The full-list aggregation (step 3) relies on the runner's overwrite guard: a subset worker
#     writes only to _partial_{feature}/, and the combined output is written by the full-list
#     invocation. A stray single-subject run against the combined is now refused by the runner.
#
# Usage:  scripts/run_topic5_v2_residual_chain.sh [N_PERM=1000] [BATCH=4] [SEL=all|one:<feat>:<sub>]
# Example (single dev group):  scripts/run_topic5_v2_residual_chain.sh 50 4 one:common_resid:narrow
set -uo pipefail

# --- BLAS/OpenMP single-threading (the load-bearing engineering lesson; see header) -------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

# --- locked cohort membership (Phase-1-v2; narrow n=20, broad n=17) -----------------------------
# Source: classify_subject_contacts axis-set assignment for the V2 band-scan cohort. Locked for this
# phase; edit here (not in a scratchpad file) if the cohort is intentionally re-cut.
NARROW="epilepsiae_1077 epilepsiae_1084 epilepsiae_1096 epilepsiae_1125 epilepsiae_1146 epilepsiae_1150 epilepsiae_139 epilepsiae_253 epilepsiae_384 epilepsiae_442 epilepsiae_548 epilepsiae_583 epilepsiae_590 epilepsiae_620 epilepsiae_635 epilepsiae_916 epilepsiae_922 epilepsiae_958 yuquan_xuxinyi yuquan_zhangkexuan"
BROAD="epilepsiae_1077 epilepsiae_1084 epilepsiae_1096 epilepsiae_1125 epilepsiae_1146 epilepsiae_1150 epilepsiae_139 epilepsiae_253 epilepsiae_384 epilepsiae_583 epilepsiae_590 epilepsiae_620 epilepsiae_635 epilepsiae_916 epilepsiae_922 yuquan_xuxinyi yuquan_zhangkexuan"
list_for(){ [ "$1" = narrow ] && echo "$NARROW" || echo "$BROAD"; }

NPERM=${1:-1000}
BATCH=${2:-4}
SEL=${3:-all}

if [ "$SEL" = all ]; then
  WGRPS=("common_resid narrow" "common_resid broad" "aperiodic_resid narrow" "aperiodic_resid broad")
else                      # one:<feat>:<sub>  (single-group dev validation)
  WGRPS=("$(echo "$SEL" | sed 's/^one://; s/:/ /')")
fi

RUN_ID="resid_$(date -u +%Y%m%dT%H%M%SZ)_$$"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
LOGDIR="$ROOT/results/run_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/${RUN_ID}.log"
MANIFEST="$LOGDIR/${RUN_ID}.manifest.json"

# --- run manifest (3.3: institutionalize run provenance instead of memory) ----------------------
{
  echo "{"
  echo "  \"run_id\": \"$RUN_ID\","
  echo "  \"launcher\": \"scripts/run_topic5_v2_residual_chain.sh\","
  echo "  \"commit\": \"$COMMIT\","
  echo "  \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"pid\": $$,"
  echo "  \"n_perm\": $NPERM,"
  echo "  \"batch\": $BATCH,"
  echo "  \"groups\": \"${WGRPS[*]}\","
  echo "  \"omp_num_threads\": \"$OMP_NUM_THREADS\","
  echo "  \"narrow_n\": $(echo "$NARROW" | wc -w),"
  echo "  \"broad_n\": $(echo "$BROAD" | wc -w),"
  echo "  \"log\": \"$LOG\""
  echo "}"
} > "$MANIFEST"

echo "=== V2 residual chain: run_id=$RUN_ID commit=$COMMIT n_perm=$NPERM batch=$BATCH ===" | tee -a "$LOG"
echo "manifest: $MANIFEST" | tee -a "$LOG"

# ---- 1. residual alignments (fast; per group, sequential) -------------------------------------
for g in "${WGRPS[@]}"; do
  set -- $g; feat=$1; sub=$2; list=$(list_for "$sub")
  echo "[align $feat/$sub] $(date -u +%H:%M:%S)" | tee -a "$LOG"
  python scripts/run_topic5_v2_alignment.py --feature "$feat" --substrate "$sub" --subjects $list \
    >>"$LOG" 2>&1 && echo "  align $feat/$sub OK" | tee -a "$LOG" \
    || echo "  align $feat/$sub FAILED (see $LOG)" | tee -a "$LOG"
done

# ---- 2. nulls workers (all groups, batched, parallel; write to _partial only) ------------------
pids=()
for g in "${WGRPS[@]}"; do
  set -- $g; feat=$1; sub=$2; list=$(list_for "$sub"); arr=($list)
  for ((i=0; i<${#arr[@]}; i+=BATCH)); do
    batch="${arr[@]:i:BATCH}"
    python scripts/run_topic5_v2_nulls.py --feature "$feat" --substrate "$sub" --subjects $batch \
      --n-perm "$NPERM" >>"$LOGDIR/${RUN_ID}_worker_${feat}_${sub}_${i}.log" 2>&1 &
    pids+=($!)
  done
done
echo "launched ${#pids[@]} nulls workers (n_perm=$NPERM), waiting... $(date -u +%H:%M:%S)" | tee -a "$LOG"
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "workers done ($fail non-zero exits — stragglers resumed by aggregation) $(date -u +%H:%M:%S)" | tee -a "$LOG"

# ---- 3. aggregate each group (full-list -> combined; overwrite guard protects it) --------------
for g in "${WGRPS[@]}"; do
  set -- $g; feat=$1; sub=$2; list=$(list_for "$sub")
  echo "[aggregate $feat/$sub] $(date -u +%H:%M:%S)" | tee -a "$LOG"
  python scripts/run_topic5_v2_nulls.py --feature "$feat" --substrate "$sub" --subjects $list \
    --n-perm "$NPERM" >>"$LOG" 2>&1 && echo "  agg $feat/$sub OK" | tee -a "$LOG" \
    || echo "  agg $feat/$sub FAILED (see $LOG)" | tee -a "$LOG"
done

echo "=== V2 residual chain done: run_id=$RUN_ID $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
echo "manifest -> $MANIFEST"
