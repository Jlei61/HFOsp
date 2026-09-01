#!/usr/bin/env bash
# Autonomous chain to the Figure 5 candidate.
#
# Priority order is deliberate: everything here serves the figure. The Node grid,
# the four-arm factorial and the spatial re-registration control -- which answer
# the connectivity question -- are NOT in this chain.
#
# No step aborts the chain silently. A gate that fails records its verdict and
# the chain stops at that point with a reason, because "the work point has no
# interpretable interictal residence segment" is a finding, not an error.
set -u
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-data-driven-zm-ictal-transition
R="$W/results/topic4_sef_hfo/data_driven_zm_ictal_transition"
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic4_data_driven_zm_ictal_transition_v1.json
COMMIT=$(git -C "$W" rev-parse HEAD)
cd "$W" || exit 1
mkdir -p "$R/chain_logs"
say() { echo "{\"step\": \"$1\", \"status\": \"$2\", \"t\": \"$(date +%H:%M:%S)\"}" | tee -a "$R/chain.log"; }

# ---- 1. wait for Phase 1A ----
say wait_phase1a start
while pgrep -f "launch_topic4_zm_ictal_transition.py --config .* --phase canary" >/dev/null; do sleep 60; done
while systemctl --user list-units --no-legend --plain --state=active 'topic4-zmitx-canary*' | grep -q service; do sleep 60; done
say wait_phase1a done

# ---- 2. gates that only read artifacts ----
for gate in interictal-baseline cost-projection repertoire recruitment; do
  if $PY scripts/audit_topic4_zm_ictal_transition.py --config "$CFG" --gate "$gate" \
       > "$R/chain_logs/gate_${gate}.log" 2>&1; then say "gate_${gate}" pass
  else say "gate_${gate}" fail; fi
done

# The interictal-baseline gate is the round's one science blocker. If it failed,
# stop here: the finding is that this work point has no interpretable interictal
# residence segment, and the baseline checkpoint is NOT moved earlier to rescue it.
if ! $PY -c "
import json,sys
v=json.load(open('$R/interictal_baseline_gate.json'))
sys.exit(0 if v['status']=='PASS' else 1)" 2>/dev/null; then
  say chain STOPPED_AT_INTERICTAL_BASELINE_GATE
  $PY -c "
import json
json.dump({'status':'STOPPED','stopped_at':'interictal_baseline_gate',
           'reason':'fewer than the required number of canary networks have an interpretable interictal residence segment',
           'figure5_possible':'panels A/B/C only, from the interictal trajectory'},
          open('$R/DONE.json','w'), indent=2)"
  notify-send "ZM-ITX" "chain stopped at the interictal-baseline gate" 2>/dev/null
  exit 0
fi

# ---- 3. freeze the dose (baseline checkpoints only) ----
say dose_runs start
$PY scripts/launch_topic4_zm_ictal_transition.py --config "$CFG" --phase dose \
    --expected-commit "$COMMIT" > "$R/chain_logs/phase_dose.log" 2>&1
say dose_runs done
$PY scripts/audit_topic4_zm_ictal_transition.py --config "$CFG" --gate dose \
    > "$R/chain_logs/gate_dose.log" 2>&1 && say gate_dose pass || say gate_dose fail

if ! $PY -c "
import json,sys
sys.exit(0 if json.load(open('$R/dose_freeze.json'))['status']=='PASS' else 1)" 2>/dev/null; then
  say chain STOPPED_NO_SUBEVENT_PROBE_REGIME
  $PY -c "
import json
json.dump({'status':'STOPPED','stopped_at':'dose_freeze',
           'reason':'NO_SUBEVENT_PROBE_REGIME: no ladder rung is measurable, baseline-safe and linear',
           'boundary':'not worked around by loosening the ignition criterion',
           'figure5_possible':'panels A/B/C only'}, open('$R/DONE.json','w'), indent=2)"
  notify-send "ZM-ITX" "chain stopped: no sub-ignition probe regime" 2>/dev/null
  exit 0
fi

# ---- 4. the perturbation Figure 5 D/E/F needs ----
say fig5_perturbation start
$PY scripts/launch_topic4_zm_ictal_transition.py --config "$CFG" --phase fig5 \
    --expected-commit "$COMMIT" > "$R/chain_logs/phase_fig5.log" 2>&1
say fig5_perturbation done

# ---- 5. counterfactual attribution (cheap, and it decides what the figure may claim) ----
say counterfactual start
$PY scripts/launch_topic4_zm_ictal_transition.py --config "$CFG" --phase counterfactual \
    --expected-commit "$COMMIT" > "$R/chain_logs/phase_counterfactual.log" 2>&1
say counterfactual done

$PY -c "
import json, glob, os
root='$R'
json.dump({'status':'READY_FOR_FIGURE5',
           'n_worker_runs': len(glob.glob(os.path.join(root,'workers','*_seed_180*.json'))),
           'n_grid_perturbation': len(glob.glob(os.path.join(root,'perturbation','*_grid.json'))),
           'n_counterfactual': len(glob.glob(os.path.join(root,'counterfactual','*.json'))),
           'deprioritised':['Node grid','four-arm factorial','r180 re-registration control']},
          open(os.path.join(root,'DONE.json'),'w'), indent=2)"
say chain READY_FOR_FIGURE5
notify-send "ZM-ITX" "chain complete: ready to build Figure 5" 2>/dev/null
