#!/usr/bin/env bash
# Stage 2: Joint-only state and perturbation analysis.
#
# Scope is fixed by the round plan and is NOT widened here: Joint arm only, the
# 3 canary networks, the 6 representative sites. The 7x7 grid is escalated to
# only if a state difference is first established -- it costs ~9 h and answers a
# spatial question that does not arise until there is a difference to localise.
#
# The two states are `low_activity` and `pre_ictal`. `low_activity` earns its
# name only if scripts/verify_topic4_zm_baseline_window.py finds a window whose
# rate AND h-weighted z AND h-weighted m all sit inside the Z/M-off support. If
# it does not, the chain continues but the states are reported as
# `early transition vs pre-ictal` and the word baseline is not used.
set -u
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-data-driven-zm-ictal-transition
R="$W/results/topic4_sef_hfo/data_driven_zm_ictal_transition"
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic4_data_driven_zm_ictal_transition_v1.json
JOINT=joint_04_control
SEEDS="1801 1802 1803"
MAXJOBS=8
cd "$W" || exit 1
COMMIT=$(git -C "$W" rev-parse HEAD)
mkdir -p "$R/chain_logs" "$R/dose" "$R/perturbation"
say() { echo "{\"step\": \"$1\", \"status\": \"$2\", \"t\": \"$(date '+%m-%d %H:%M:%S')\"}" | tee -a "$R/stage2.log"; }

# The chain's OWN unit matches 'topic4-zmitx-*', so counting it would make
# wait_all() wait for itself forever. Same self-match trap as `pgrep -f`.
SELF=topic4-zmitx-stage2-chain.service
n_active() { systemctl --user list-units --no-legend --plain --state=active 'topic4-zmitx-*' 2>/dev/null \
             | awk -v self="$SELF" '$1 ~ /\.service$/ && $1 != self' | wc -l ; }
wait_slot() { while [ "$(n_active)" -ge "$MAXJOBS" ]; do sleep 30; done; }
wait_all()  { while [ "$(n_active)" -gt 0 ]; do sleep 30; done; }

launch() {   # launch <unit> <logfile> <args...>
  local unit="$1" log="$2"; shift 2
  systemctl --user reset-failed "$unit" 2>/dev/null
  systemd-run --user --unit="$unit" --quiet \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=MKL_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --working-directory="$W" \
    --property=StandardOutput=append:"$log" \
    --property=StandardError=append:"$log" \
    /usr/bin/nohup "$@" || say "launch_${unit}" FAILED
}

# ---- 1. wait for the checkpoint and shadow runs ----
say wait_inputs start
wait_all
say wait_inputs done

# ---- 2. does the low-activity state earn the name baseline? ----
say baseline_verdict start
$PY scripts/verify_topic4_zm_baseline_window.py \
    --seeds $SEEDS --candidate-id "$JOINT" > "$R/chain_logs/baseline_verdict.log" 2>&1 \
  && say baseline_verdict done || say baseline_verdict FAILED
LABEL=$($PY -c "
import json
try:
    print(json.load(open('$R/baseline_window_verdict.json'))['label_to_use'])
except Exception:
    print('early transition vs pre-ictal')" 2>/dev/null)
say "state_label" "$LABEL"

# ---- 3. dose calibration, on the LOW-ACTIVITY state only ----
# Calibrating on the pre-ictal state would pick a dose tuned to the state whose
# response is the thing being measured.
say dose start
for rung in 16 32 64 128 256; do
  for s in $SEEDS; do
    CK="$R/checkpoints/${JOINT}_seed_${s}_low_activity.npz"
    [ -f "$CK" ] || { say "dose_s${s}_n${rung}" NO_CHECKPOINT; continue; }
    wait_slot
    launch "topic4-zmitx-dose-s${s}-n${rung}" "$R/chain_logs/dose_s${s}_n${rung}.log" \
      $PY scripts/run_topic4_zm_perturbation_worker.py \
        --config "$CFG" --candidate-id "$JOINT" --seed "$s" \
        --checkpoint "$CK" --label low_activity --sites representative \
        --dose-cells "$rung" --expected-commit "$COMMIT" --allow-uncommitted-config \
        --out-json "$R/dose/${JOINT}_seed_${s}_low_activity_n${rung}.json" \
        --out-npz "$R/dose/${JOINT}_seed_${s}_low_activity_n${rung}.npz"
  done
done
wait_all
say dose done

# ---- 4. pick the dose ----
$PY scripts/audit_topic4_zm_ictal_transition.py --config "$CFG" --gate dose \
    > "$R/chain_logs/gate_dose.log" 2>&1
DOSE=$($PY -c "
import json
v=json.load(open('$R/dose_freeze.json'))
print(v['selected_dose_cells'] if v['selected_dose_cells'] else 0)" 2>/dev/null || echo 0)
say dose_selected "$DOSE"
if [ "$DOSE" = "0" ] || [ -z "$DOSE" ]; then
  say chain STOPPED_NO_SUBEVENT_PROBE_REGIME
  $PY -c "
import json
json.dump({'status':'STOPPED','stopped_at':'dose_gate',
           'reason':'no ladder rung is simultaneously measurable, low-activity-safe and linear',
           'consequence':'the state comparison cannot be run with a probe that is '
                         'both detectable and free of probe-attributable ignition'},
          open('$R/STAGE2_DONE.json','w'), indent=2)"
  exit 0
fi

# ---- 5. the paired state comparison, same dose and same sites at both states ----
say pairs start
for s in $SEEDS; do
  for label in low_activity pre_ictal; do
    CK="$R/checkpoints/${JOINT}_seed_${s}_${label}.npz"
    [ -f "$CK" ] || { say "pair_s${s}_${label}" NO_CHECKPOINT; continue; }
    wait_slot
    launch "topic4-zmitx-pair-s${s}-${label}" "$R/chain_logs/pair_s${s}_${label}.log" \
      $PY scripts/run_topic4_zm_perturbation_worker.py \
        --config "$CFG" --candidate-id "$JOINT" --seed "$s" \
        --checkpoint "$CK" --label "$label" --sites representative \
        --dose-cells "$DOSE" --expected-commit "$COMMIT" --allow-uncommitted-config \
        --out-json "$R/perturbation/${JOINT}_seed_${s}_${label}_n${DOSE}.json"
  done
done
wait_all
say pairs done

# ---- 6. paired analysis ----
say analysis start
$PY scripts/analyze_topic4_zm_state_susceptibility.py \
    --config "$CFG" --dose "$DOSE" --seeds $SEEDS --state-label "$LABEL" \
    > "$R/chain_logs/state_susceptibility.log" 2>&1 \
  && say analysis done || say analysis FAILED
say chain STAGE2_COMPLETE
