#!/usr/bin/env bash
# Launch the six target-blind pilot patients x three seeds of the SPF-RNN
# ladder as independent CPU processes.
#
# CPU, not GPU: the ladder models are small enough that CUDA kernel-launch
# overhead dominates (measured 2026-07-30 on this machine, one M4 training
# epoch over 8000 events: CPU 1.43 s vs CUDA 3.58 s).  Parallelism across
# (subject, seed) is where the throughput is.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${1:-results/topic5_shared_propagation_field/development/ladder_pilot_v0_4}"
THREADS="${SPF_THREADS:-4}"
PILOT_CSV="results/topic5_shared_propagation_field/phase0/pilot_subjects_target_blind.csv"

mkdir -p "$OUT_ROOT/logs"
SUBJECTS=$(tail -n +2 "$PILOT_CSV" | cut -d, -f1)
SEEDS="20260730 20260731 20260801"

echo "launching pilot -> $OUT_ROOT (threads/proc=$THREADS)"
PIDS=()
for subject in $SUBJECTS; do
  for seed in $SEEDS; do
    OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" \
      conda run --no-capture-output -n cuda_env python \
        scripts/run_topic5_spf_model_ladder.py \
        --subject "$subject" --seeds "$seed" --device cpu \
        --output-root "$OUT_ROOT" \
        > "$OUT_ROOT/logs/${subject}_seed${seed}.log" 2>&1 &
    PIDS+=("$!")
  done
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$((FAILED + 1))
  fi
done
if (( FAILED > 0 )); then
  echo "$FAILED pilot shards failed; aggregation refused" >&2
  exit 1
fi
conda run --no-capture-output -n cuda_env python \
  scripts/aggregate_topic5_spf_ladder.py --root "$OUT_ROOT"
echo "pilot processes finished"
