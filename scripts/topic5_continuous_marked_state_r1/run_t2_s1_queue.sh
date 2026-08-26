#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
ROOT=results/epi_prssm/continuous_marked_state/r1
R13="$ROOT/r1_3"
OUT="$ROOT/t2_s1_long_scale"
mkdir -p "$OUT/logs"

while [[ ! -f "$R13/reports/r1_3_summary.json" ]]; do
  sleep 30
done

jobs=()
for subject in epilepsiae_620 epilepsiae_958; do
  for seed in 0 1 2; do
    for scale in 100 1000; do
      target="$OUT/human/$subject/seed_${seed}_n_${scale}/result.json"
      [[ -f "$target" ]] && continue
      while (( $(jobs -pr | wc -l) >= 2 )); do
        wait -n
      done
      PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" \
        scripts/topic5_continuous_marked_state_r1/run_t2_s1_human.py \
        --subject "$subject" --seed "$seed" --scale-events "$scale" \
        > "$OUT/logs/${subject}_seed_${seed}_n_${scale}.log" 2>&1 &
      jobs+=("$!")
    done
  done
done
wait
PYTHONPATH=. "$PYTHON" scripts/topic5_continuous_marked_state_r1/aggregate_t2_s1_human.py \
  > "$OUT/logs/aggregate.log" 2>&1
