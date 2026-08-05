#!/usr/bin/env bash
# Three unit families for v0.4: 102 main + 34 rank-shuffle + 68 split-half.
set -u -o pipefail
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic5_interictal_ictal_shared_axis_rnn_v0_4.yaml
OUT=results/topic5_interictal_ictal_shared_axis_rnn_v0_4
cd "$W" || exit 1
stamp(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
fail(){ printf '{"status":"FAILED","stage":"%s","utc":"%s"}\n' "$1" "$(stamp)" > "$OUT/TRAIN_FAILED.json"; exit 1; }
echo "[$(stamp)] main 102 units (3 seeds)"
$PY scripts/launch_topic5_shared_axis_rnn_v0_4.py --config $CFG --models shared_axis \
    --workers 12 --resume || fail main
echo "[$(stamp)] rank-shuffle 34 units (seed 11)"
$PY scripts/launch_topic5_shared_axis_rnn_v0_4.py --config $CFG --models shared_axis_rank_shuffle \
    --seeds 11 --workers 12 --resume || fail rank_shuffle
echo "[$(stamp)] split-half 68 units (seed 11)"
$PY scripts/launch_topic5_shared_axis_rnn_v0_4.py --config $CFG --models shared_axis \
    --seeds 11 --fit-halves first second --workers 12 --resume || fail split_half
printf '{"status":"COMPLETE","utc":"%s"}\n' "$(stamp)" > "$OUT/TRAIN_COMPLETE.json"
echo "[$(stamp)] TRAINING COMPLETE"
