#!/usr/bin/env bash
# Everything Agent A runs once the producer training queue has finished.
# Idempotent: each stage skips work whose configuration hash already matches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:${LD_LIBRARY_PATH:-}

ROOT=/data/hfosp_group_event_state_v0_2/agent_a
PROD=$ROOT/producers/main
S=$PROD/states
LOGS=$ROOT/logs
mkdir -p "$LOGS"
cd "$HERE"

WORKERS=${WORKERS:-12}
SEEDS=${SEEDS:-"1 2 3"}

echo "== 1. registry =="
$PY scripts/topic5_group_event_state/v02_write_registry.py --seeds $SEEDS \
    2>&1 | tee "$LOGS/registry.log"

echo "== 2. future-block evaluation with the frozen states =="
STATE_DIRS=()
for p in P_local P_slow; do for s in $SEEDS; do
  [ -d "$S/${p}_seed${s}" ] && STATE_DIRS+=("$S/${p}_seed${s}")
done; done
# Secondary arms are limited on purpose: the load-bearing increment B vs B+S runs
# for every (producer, seed); the state-only arm and the second shift offset are
# diagnostics and running them for all six would nearly double the wall clock.
$PY scripts/topic5_group_event_state/v02_run_future_block.py \
    --cohort all --workers "$WORKERS" --tag with_state \
    --shift-extra-steps 1 4 \
    --state-only-for P_slow_seed1 P_local_seed1 \
    --shift-for P_slow_seed1 P_slow_seed2 P_slow_seed3 P_local_seed1 \
    --extra-shift-for P_slow_seed1 \
    --state-dir "${STATE_DIRS[@]}" 2>&1 | tee "$LOGS/future_block_with_state.log"

echo "== 3. H2a same-prefix continuation =="
$PY scripts/topic5_group_event_state/v02_run_prefix.py \
    --workers "$WORKERS" --tag main --producer-root "$PROD" \
    --producers P_local P_slow --seeds $SEEDS 2>&1 | tee "$LOGS/prefix.log"

echo "== 4. A4 memory-truncation replays (P_slow seed 1) =="
$PY scripts/topic5_group_event_state/v02_run_state_diagnostics.py \
    --producer P_slow --seed 1 --gpus 0 1 --jobs-per-gpu 2 \
    --producer-root "$PROD" 2>&1 | tee "$LOGS/state_diagnostics.log"

echo "== 5. latent read-outs and the coarse matched wrong-time donor =="
D_FAST=$($PY - <<'EOF'
import glob, json
p = sorted(glob.glob('/data/hfosp_group_event_state_v0_2/agent_a/producers/main/runs/*/P_slow/seed1/result.json'))[0]
print(json.load(open(p))['state_extraction']['d_fast'])
EOF
)
echo "d_fast=$D_FAST"
$PY scripts/topic5_group_event_state/v02_make_state_variants.py \
    --state-dir "$S/P_slow_seed1" --d-fast "$D_FAST" 2>&1 | tee "$LOGS/state_variants.log"

echo "== 6. evaluation of the A4 arms =="
DIAG_DIRS=()
for d in "$PROD"/states_diag/P_slow_seed1_*; do [ -d "$d" ] && DIAG_DIRS+=("$d"); done
for v in fast_only slow_only matched_donor; do
  [ -d "$S/P_slow_seed1_$v" ] && DIAG_DIRS+=("$S/P_slow_seed1_$v")
done
$PY scripts/topic5_group_event_state/v02_run_future_block.py \
    --cohort all --workers "$WORKERS" --tag diagnostics --no-mlp \
    --shift-extra-steps 1 --state-only-for --shift-for \
    --state-dir "${DIAG_DIRS[@]}" 2>&1 | tee "$LOGS/future_block_diagnostics.log"

echo "== 7. cohort tables + the load-bearing figure =="
$PY scripts/topic5_group_event_state/v02_summarize.py \
    --future-root "$ROOT/future_block/with_state" \
    --prefix-root "$ROOT/prefix/main" \
    --diagnostics-root "$ROOT/future_block/diagnostics" \
    --tag main 2>&1 | tee "$LOGS/summarize.log"

echo "== 8. registry refresh (now complete) =="
$PY scripts/topic5_group_event_state/v02_write_registry.py --seeds $SEEDS \
    2>&1 | tee -a "$LOGS/registry.log"

echo "ALL STAGES DONE"
