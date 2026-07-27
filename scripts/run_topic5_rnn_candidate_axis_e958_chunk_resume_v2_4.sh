#!/usr/bin/env bash
set -euo pipefail

cd /home/honglab/leijiaxin/HFOsp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python_bin=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
trainer=scripts/train_topic5_rnn_candidate_axis_v2_4.py
log_root=results/topic5_rnn_axis_positive_static_transfer_v2_4/formal/chunk_logs
mkdir -p "${log_root}"

seeds=(17 29 43)
chunks=(15,16,17 18,19,20 21,22,23 24,25,26 27,28,29 30,31)
pids=()

for seed in "${seeds[@]}"; do
  for chunk in "${chunks[@]}"; do
    "${python_bin}" "${trainer}" \
      --subject epilepsiae_958 \
      --seed "${seed}" \
      --device cpu \
      --candidate-indices "${chunk}" \
      --candidates-only \
      >"${log_root}/epilepsiae_958_seed${seed}_axes_${chunk//,/-}.log" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

pids=()
for seed in "${seeds[@]}"; do
  "${python_bin}" "${trainer}" \
    --subject epilepsiae_958 \
    --seed "${seed}" \
    --device cpu \
    >>"results/topic5_rnn_axis_positive_static_transfer_v2_4/formal/launcher_logs/epilepsiae_958_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
