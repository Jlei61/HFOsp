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

count_core() {
  "$PY" - <<'COUNT'
import json, os
R = "results/topic5_spatial_latent_propagation_rnn_v0_1"
subs = json.load(open(f"{R}/INPUT_MANIFEST.json"))["frozen_cohort"]["primary"]
arms = ["STATIC_CONTACT", "ORDINARY_GRU", "CONTACT_GRAPH_RNN",
        "LATENT_FIXED_LOCAL_RNN", "LATENT_LEARNED_SPATIAL_RNN"]
print(sum(1 for s in subs for a in arms
          if os.path.exists(f"{R}/per_subject/{s}/{a}/seed1/DONE.json")))
COUNT
}

# The tail of the cohort is its largest patients, and running them at the
# concurrency the small ones tolerated exhausted the GPU: fourteen units died of
# out-of-memory in one pass, none of them for a reason that has anything to do
# with the models. Every stage below therefore queues at WORKERS, not at the
# rate the early cohort ran at.
#
# The retry list carries the dense ceiling arm as well. It is a pre-registered
# control rather than a core arm, so count_core ignores it -- but four of its
# patients died in the same out-of-memory pass, and a control that silently
# drops a fifth of the cohort answers a weaker question than the one registered.
WORKERS=2

# pgrep -c already prints 0 when nothing matches, and it also exits 1.  Adding
# "|| echo 0" therefore appends a SECOND zero, and the two-line result makes the
# arithmetic test error out rather than compare -- so the retry below could
# never fire and the run sat at 98/105 with nothing running.
count_trainers() { pgrep -fc "train_topic5_slp_unit\.py"; }

stamp "waiting for the 105-unit refit"
attempt=0
while [ "$(count_core)" -lt 105 ]; do
  if [ "$(count_trainers)" -eq 0 ]; then
    attempt=$((attempt + 1))
    # An earlier version left this loop as soon as one retry ended short, which
    # sent the summaries off against a 91-unit tree. Retry until the cohort is
    # whole or the retries stop achieving anything.
    if [ "$attempt" -gt 15 ]; then
      stamp "giving up at $(count_core)/105 after $attempt retries; the gate records the gap"
      break
    fi
    stamp "trainers gone at $(count_core)/105; retry $attempt at $WORKERS workers"
    find "$R/per_subject" -name FAILED.json -delete 2>/dev/null
    "$PY" scripts/launch_topic5_slp_cohort.py --config "$CFG" \
      --arms STATIC_CONTACT ORDINARY_GRU CONTACT_GRAPH_RNN \
             LATENT_FIXED_LOCAL_RNN LATENT_LEARNED_SPATIAL_RNN \
             LATENT_DENSE_RNN \
      --seeds 1 --workers "$WORKERS" >> "$R/refit_retry.log" 2>&1
  fi
  sleep 120
done
stamp "cohort at $(count_core)/105"

# Leave-contact-out trained its tissue-field arm under the old ceiling, and its
# comparator did not, so any residual bias runs the same way as the cohort bias.
# Redo it at the same budget as everything else.
# The flow-ordering readout compares a patient against itself refitted from a
# different start, so it needs a second seed. Cancelling seeds 2-3 removed every
# within-patient pair and left the one structural question the recovery gate
# licenses unanswerable.
stamp "second seed for the learned arm, for the within-patient comparison"
"$PY" scripts/launch_topic5_slp_cohort.py --config "$CFG" \
  --arms LATENT_LEARNED_SPATIAL_RNN --seeds 2 --workers "$WORKERS" \
  >> "$R/seed2_latent.log" 2>&1 || true

stamp "leave-contact-out at the corrected budget"
if [ ! -d "$R/leave_contact_out_budget95" ] && [ -d "$R/leave_contact_out" ]; then
  mv "$R/leave_contact_out" "$R/leave_contact_out_budget95"
fi
"$PY" scripts/run_topic5_slp_leave_contact_out.py --config "$CFG" \
  --fraction 0.25 --seeds 1 --workers "$WORKERS" >> "$R/lco_refit.log" 2>&1 || true

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
