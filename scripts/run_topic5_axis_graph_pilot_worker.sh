#!/usr/bin/env bash
set -euo pipefail

subject="${1:?subject is required}"
seed="${2:-20260726}"
pilot_root="${3:-results/topic5_structured_axis_graph/pilot_rank012_v0_6}"

cd /home/honglab/leijiaxin/HFOsp
for rank in 0 1 2; do
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n cuda_env \
    python scripts/train_topic5_axis_graph_rnn.py \
    --run-dir "${pilot_root}/seed_${seed}/rank_${rank}/${subject}" \
    --heldout-subject "${subject}" \
    --structured-rank "${rank}" \
    --seed "${seed}" \
    --device cuda:0
done
