#!/usr/bin/env bash
set -euo pipefail

repo=/home/honglab/leijiaxin/HFOsp
python=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
root="$repo/results/epi_prssm/continuous_marked_state/r1/t2_long_total_effect"
t1="$root/t1_r1_3"
logs="$root/logs"
mkdir -p "$logs"
cd "$repo"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for seed in 0 1 2; do
  result="$t1/fits/yuquan_zhangjiaqi/explicit_seed_${seed}/result.json"
  if [[ -f "$result" ]] && grep -q '"status": "COMPLETE"' "$result"; then
    continue
  fi
  "$python" scripts/topic5_continuous_marked_state_r1/run_r1_3_target_observer.py \
    --subject yuquan_zhangjiaqi \
    --arm explicit \
    --seed "$seed" \
    --device cuda \
    --observer-epochs 2 \
    --joint-epochs 2 \
    --chunk-anchors 16 \
    --output-root "$t1" \
    --observation-cache-root "$t1/cache" \
    > "$logs/t1_zhangjiaqi_seed_${seed}.log" 2>&1
done

for seed in 0 1 2; do
  for window in event_count_10000 physical_6h; do
    result="$root/human/yuquan_zhangjiaqi/$window/seed_${seed}/result.json"
    if [[ "${FORCE_HUMAN:-0}" != "1" ]] && [[ -f "$result" ]] \
      && grep -q '"status": "COMPLETE"' "$result"; then
      continue
    fi
    "$python" scripts/topic5_continuous_marked_state_r1/run_t2_long_total_human.py \
      --subject yuquan_zhangjiaqi \
      --seed "$seed" \
      --window "$window" \
      --device cuda \
      --t1-root "$t1" \
      --output-root "$root/human" \
      > "$logs/human_zhangjiaqi_${window}_seed_${seed}.log" 2>&1
  done
done
