#!/bin/bash
# One detached run that carries the study to its acceptance gate.
#
# Ordered by what the deliverables depend on, not by what is interesting: the
# first seed of every arm has to exist before any cohort statistic means
# anything, leave-contact-out is the one question the recovery gate leaves open,
# and the two controls only qualify results that already exist.
#
# Concurrency is deliberately low. Seventeen simultaneous fits exhausted the card
# and cost seventeen units to out-of-memory errors; the batch size is not the
# place to fix that, because changing it per patient would make a patient's arms
# incomparable with each other.
set -u
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-slp-rnn
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
R=$W/results/topic5_spatial_latent_propagation_rnn_v0_1
SP=/tmp/claude-1002/-home-honglab-leijiaxin-HFOsp/7f31b88a-f5f4-4d4e-ab87-b94f0fef6a17/scratchpad
CFG=$R/development/FROZEN_CONFIG.json
ARMS="STATIC_CONTACT ORDINARY_GRU CONTACT_GRAPH_RNN LATENT_FIXED_LOCAL_RNN LATENT_LEARNED_SPATIAL_RNN"

cd "$W" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stamp() { echo "[$(date +%H:%M:%S)] $*"; }

# A unit that died on a transient resource error must be retried, not left out of
# the cohort. Resume skips whatever is already done, so this only costs the
# stragglers.
retry_failures() {
  local n
  n=$(find "$R/per_subject" -name FAILED.json 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    stamp "clearing $n failed units for retry"
    find "$R/per_subject" -name FAILED.json -delete
  fi
}

stamp "STAGE 1  first seed, every arm, every patient"
for attempt in 1 2 3; do
  retry_failures
  "$PY" scripts/launch_topic5_slp_cohort.py --config "$CFG" \
    --arms $ARMS --seeds 1 --workers 5 >> "$R/orchestrator_cohort.log" 2>&1
  missing=$("$PY" - <<'PY'
import json, os
R='results/topic5_spatial_latent_propagation_rnn_v0_1'
subs=json.load(open(f'{R}/INPUT_MANIFEST.json'))['frozen_cohort']['primary']
arms=["STATIC_CONTACT","ORDINARY_GRU","CONTACT_GRAPH_RNN","LATENT_FIXED_LOCAL_RNN","LATENT_LEARNED_SPATIAL_RNN"]
print(sum(1 for s in subs for a in arms
          if not os.path.exists(f'{R}/per_subject/{s}/{a}/seed1/DONE.json')))
PY
)
  stamp "  attempt $attempt: $missing units still missing at seed 1"
  [ "$missing" -eq 0 ] && break
done

stamp "STAGE 2  leave-contact-out"
for attempt in 1 2; do
  "$PY" scripts/run_topic5_slp_leave_contact_out.py --config "$CFG" \
    --fraction 0.25 --seeds 1 --workers 4 >> "$R/orchestrator_lco.log" 2>&1
  done_n=$(find "$R/leave_contact_out" -name DONE.json 2>/dev/null | wc -l)
  stamp "  attempt $attempt: $done_n/84 leave-contact-out units"
  [ "$done_n" -ge 84 ] && break
done

stamp "STAGE 3  controls that qualify the results"
"$PY" scripts/run_topic5_slp_geometry_shuffle_control.py --config "$CFG" --workers 4 \
  --subjects epilepsiae_1146 epilepsiae_1084 epilepsiae_1150 epilepsiae_620 \
             yuquan_pengzihang epilepsiae_922 epilepsiae_384 epilepsiae_548 \
  >> "$R/orchestrator_controls.log" 2>&1 || true
"$PY" scripts/probe_topic5_slp_convergence_bias.py --work "$SP/convprobe2" --workers 4 \
  >> "$R/orchestrator_controls.log" 2>&1 || true
retry_failures
"$PY" scripts/launch_topic5_slp_cohort.py --config "$CFG" \
  --arms LATENT_DENSE_RNN --seeds 1 --workers 5 >> "$R/orchestrator_controls.log" 2>&1 || true

stamp "STAGE 4  further seeds, best effort"
for seed in 2 3; do
  retry_failures
  "$PY" scripts/launch_topic5_slp_cohort.py --config "$CFG" \
    --arms $ARMS --seeds "$seed" --workers 5 >> "$R/orchestrator_cohort.log" 2>&1 || true
  stamp "  seed $seed pass finished"
done

stamp "STAGE 5  aggregate, figures, closeout, acceptance"
# Deliberately NOT clearing failures here. Every earlier stage retries them; a
# marker that survives to this point is a unit that failed repeatedly, and the
# acceptance gate has to see it. Deleting them first would make the gate pass by
# destroying its own evidence.
stamp "  failures surviving every retry: $(find "$R/per_subject" -name FAILED.json 2>/dev/null | wc -l)"
"$PY" scripts/verify_topic5_slp_static_baseline.py >> "$R/orchestrator_final.log" 2>&1 || true
"$PY" scripts/run_topic5_slp_flow_ordering.py >> "$R/orchestrator_final.log" 2>&1 || true
"$PY" scripts/aggregate_topic5_slp_cohort.py >> "$R/orchestrator_final.log" 2>&1
"$PY" scripts/run_topic5_slp_leave_contact_out.py --config "$CFG" --aggregate-only \
  >> "$R/orchestrator_final.log" 2>&1 || true
"$PY" scripts/plot_topic5_slp_figures.py >> "$R/orchestrator_final.log" 2>&1
"$PY" scripts/write_topic5_slp_closeout.py >> "$R/orchestrator_final.log" 2>&1
"$PY" scripts/accept_topic5_slp_v0_1.py > "$R/ACCEPTANCE.txt" 2>&1
stamp "ORCHESTRATOR COMPLETE"
tail -14 "$R/ACCEPTANCE.txt"
