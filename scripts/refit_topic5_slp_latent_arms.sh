#!/bin/bash
# The budget probe showed the tissue-field arms were truncated while their
# comparators were not: static and recurrent converge in under 180 epochs and
# gain nothing from a longer budget, while the learned latent arm runs 157-316
# and gains up to 0.08.  Its cohort numbers are therefore contaminated and the
# "indistinguishable from static" reading is an artefact of the epoch ceiling.
#
# Re-run the two latent arms with a budget large enough for them to converge.
# The stopping rule and patience are unchanged, so arms are still compared on
# equal terms -- each simply trains until it stops improving.
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-slp-rnn
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
R=$W/results/topic5_spatial_latent_propagation_rnn_v0_1

# Stop the orchestrator before it spends hours on further seeds at the old budget.
for p in $(pgrep -f "topic5_slp"); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f "topic5_slp"); do kill -9 "$p" 2>/dev/null; done
sleep 3

cd "$W" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Preserve the truncated fits as evidence rather than deleting them.
if [ ! -d "$R/per_subject_budget95" ]; then
  cp -r "$R/per_subject" "$R/per_subject_budget95"
fi
rm -rf "$R"/per_subject/*/LATENT_LEARNED_SPATIAL_RNN "$R"/per_subject/*/LATENT_FIXED_LOCAL_RNN

$PY - <<'PY'
import json, pathlib
cfg = json.loads(pathlib.Path(
    "results/topic5_spatial_latent_propagation_rnn_v0_1/development/FROZEN_CONFIG.json"
).read_text())
cfg.update({"epochs_freeze": 400, "patience": 12})
pathlib.Path(
    "results/topic5_spatial_latent_propagation_rnn_v0_1/development/FROZEN_CONFIG_LONG.json"
).write_text(json.dumps(cfg, indent=1))
print("long-budget config:", json.dumps(cfg))
PY

setsid nohup "$PY" -u scripts/launch_topic5_slp_cohort.py \
  --config "$R/development/FROZEN_CONFIG_LONG.json" \
  --arms LATENT_FIXED_LOCAL_RNN LATENT_LEARNED_SPATIAL_RNN \
  --seeds 1 --workers 5 \
  > "$R/refit_latent_arms.log" 2>&1 < /dev/null &
sleep 12
echo "trainers: $(pgrep -fc 'train_topic5_slp_unit')"
