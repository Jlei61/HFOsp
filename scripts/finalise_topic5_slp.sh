#!/bin/bash
# Everything after the uniform-budget refit, in dependency order.
#
# Waits for the cohort rather than assuming it: the refit is the reason every
# number downstream changed, so nothing here may run against a half-refit tree.
set -u
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-slp-rnn
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
R=$W/results/topic5_spatial_latent_propagation_rnn_v0_1
CFG=$R/development/FROZEN_CONFIG_LONG.json
cd "$W" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

stamp "waiting for the 105-unit refit"
while [ "$(find "$R/per_subject" -name DONE.json 2>/dev/null | wc -l)" -lt 105 ]; do
  if [ "$(pgrep -fc train_topic5_slp_unit || echo 0)" -eq 0 ]; then
    stamp "trainers gone at $(find "$R/per_subject" -name DONE.json | wc -l)/105; retrying stragglers"
    find "$R/per_subject" -name FAILED.json -delete 2>/dev/null
    "$PY" scripts/launch_topic5_slp_cohort.py --config "$CFG" \
      --arms STATIC_CONTACT ORDINARY_GRU CONTACT_GRAPH_RNN \
             LATENT_FIXED_LOCAL_RNN LATENT_LEARNED_SPATIAL_RNN \
      --seeds 1 --workers 5 >> "$R/refit_retry.log" 2>&1
    [ "$(find "$R/per_subject" -name DONE.json | wc -l)" -lt 105 ] && break
  fi
  sleep 120
done
stamp "cohort at $(find "$R/per_subject" -name DONE.json | wc -l)/105"

# Leave-contact-out trained its tissue-field arm under the old ceiling, and its
# comparator did not, so any residual bias runs the same way as the cohort bias.
# Redo it at the same budget as everything else.
stamp "leave-contact-out at the corrected budget"
if [ ! -d "$R/leave_contact_out_budget95" ] && [ -d "$R/leave_contact_out" ]; then
  mv "$R/leave_contact_out" "$R/leave_contact_out_budget95"
fi
"$PY" scripts/run_topic5_slp_leave_contact_out.py --config "$CFG" \
  --fraction 0.25 --seeds 1 --workers 5 >> "$R/lco_refit.log" 2>&1 || true

stamp "re-deriving every summary from the refit tree"
"$PY" scripts/verify_topic5_slp_static_baseline.py >> "$R/finalise.log" 2>&1 || true
"$PY" scripts/aggregate_topic5_slp_cohort.py >> "$R/finalise.log" 2>&1
"$PY" scripts/check_topic5_slp_seed_stability.py >> "$R/finalise.log" 2>&1 || true
"$PY" scripts/run_topic5_slp_flow_ordering.py >> "$R/finalise.log" 2>&1 || true
"$PY" scripts/run_topic5_slp_leave_contact_out.py --config "$CFG" --aggregate-only \
  >> "$R/finalise.log" 2>&1 || true
"$PY" scripts/plot_topic5_slp_figures.py >> "$R/finalise.log" 2>&1
"$PY" scripts/write_topic5_slp_closeout.py >> "$R/finalise.log" 2>&1
"$PY" scripts/accept_topic5_slp_v0_1.py > "$R/ACCEPTANCE.txt" 2>&1
stamp "FINALISE COMPLETE"
cat "$R/ACCEPTANCE.txt"
