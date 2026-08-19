#!/usr/bin/env bash
# Stage 3: the 2x2 connectivity factorial, sequenced AFTER stage 2.
#
# Waits for stage 2 rather than running beside it, because the user's priority
# order is figure first, joint perturbation second, connectivity last, and both
# would otherwise contend for the same worker pool.
#
# This unit is deliberately NOT named topic4-zmitx-*: the stage 2 chain waits on
# every unit matching that prefix, so sharing it would deadlock stage 2 against
# a chain that is itself waiting for stage 2.
set -u
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-data-driven-zm-ictal-transition
R="$W/results/topic4_sef_hfo/data_driven_zm_ictal_transition"
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic4_data_driven_zm_ictal_transition_v1.json
cd "$W" || exit 1
COMMIT=$(git -C "$W" rev-parse HEAD)
say() { echo "{\"step\": \"$1\", \"status\": \"$2\", \"t\": \"$(date '+%m-%d %H:%M:%S')\"}" | tee -a "$R/stage3.log"; }

say wait_stage2 start
while true; do
  [ -f "$R/STAGE2_DONE.json" ] && { say wait_stage2 "stage2_stopped_early"; break; }
  grep -q "STAGE2_COMPLETE" "$R/stage2.log" 2>/dev/null && { say wait_stage2 done; break; }
  systemctl --user is-active --quiet topic4-zmitx-stage2-chain || {
    say wait_stage2 "stage2_chain_gone"; break; }
  sleep 120
done
# let any straggler unit from stage 2 finish before taking the pool
while systemctl --user list-units --no-legend --plain --state=active 'topic4-zmitx-*' \
      2>/dev/null | grep -q service; do sleep 60; done

say formal_launch start
$PY scripts/launch_topic4_zm_ictal_transition.py --config "$CFG" --phase formal \
    --expected-commit "$COMMIT" --allow-uncommitted-config \
    > "$R/chain_logs/formal_launch.log" 2>&1 \
  && say formal_launch done || { say formal_launch FAILED; exit 1; }

say factorial start
$PY scripts/analyze_topic4_zm_connectivity_factorial.py --config "$CFG" \
    > "$R/chain_logs/factorial.log" 2>&1 \
  && say factorial done || say factorial FAILED
say chain STAGE3_COMPLETE
