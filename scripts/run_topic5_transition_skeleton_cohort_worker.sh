#!/usr/bin/env bash
set -euo pipefail

subject="${1:?subject is required}"
seed="${2:?seed is required}"
screen_root="${3:?screen root is required}"
config="${4:-config/topic5_transition_skeleton_graph_rnn_v0_7.yaml}"

cd /home/honglab/leijiaxin/HFOsp

PYTHONUNBUFFERED=1 conda run --no-capture-output -n cuda_env \
  python scripts/train_topic5_axis_graph_rnn.py \
  --config "${config}" \
  --run-dir "${screen_root}/seed_${seed}/rank_0/intact/${subject}" \
  --heldout-subject "${subject}" \
  --structured-rank 0 \
  --seed "${seed}" \
  --prior-control intact \
  --primary-only \
  --device cuda:0

PYTHONUNBUFFERED=1 conda run --no-capture-output -n cuda_env \
  python scripts/train_topic5_axis_graph_rnn.py \
  --config "${config}" \
  --run-dir "${screen_root}/seed_${seed}/rank_1/intact/${subject}" \
  --heldout-subject "${subject}" \
  --structured-rank 1 \
  --seed "${seed}" \
  --prior-control intact \
  --primary-only \
  --device cuda:0

PYTHONUNBUFFERED=1 conda run --no-capture-output -n cuda_env \
  python scripts/train_topic5_axis_graph_rnn.py \
  --config "${config}" \
  --run-dir "${screen_root}/seed_${seed}/rank_1/weight_shuffle/${subject}" \
  --heldout-subject "${subject}" \
  --structured-rank 1 \
  --seed "${seed}" \
  --prior-control weight_shuffle \
  --primary-only \
  --device cuda:0
