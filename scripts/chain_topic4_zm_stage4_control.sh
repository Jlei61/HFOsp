#!/usr/bin/env bash
# Stage 4 first leg: the matched spatial re-registration control.
#
# This is the sensitivity gate for the stage 3 factorial. Until it has run, the
# factorial numbers stay in the archive: without it we cannot say the speed-up
# needs the registration the patient's data produced rather than any equivalent
# structure.
set -u
W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-data-driven-zm-ictal-transition
R="$W/results/topic4_sef_hfo/data_driven_zm_ictal_transition"
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic4_data_driven_zm_ictal_transition_v1.json
cd "$W" || exit 1
COMMIT=$(git -C "$W" rev-parse HEAD)
say() { echo "{\"step\": \"$1\", \"status\": \"$2\", \"t\": \"$(date '+%m-%d %H:%M:%S')\"}" | tee -a "$R/stage4.log"; }

# never take the pool while stage 3 stragglers are alive
while systemctl --user list-units --no-legend --plain --state=active 'topic4-zmitx-formal-*' \
      2>/dev/null | grep -q service; do sleep 60; done

say control_launch start
$PY scripts/launch_topic4_zm_ictal_transition.py --config "$CFG" --phase control \
    --expected-commit "$COMMIT" --allow-uncommitted-config \
    > "$R/chain_logs/control_launch.log" 2>&1 \
  && say control_launch done || { say control_launch FAILED; exit 1; }

say control_analysis start
$PY scripts/analyze_topic4_zm_reregistration_control.py --config "$CFG" \
    > "$R/chain_logs/control_analysis.log" 2>&1 \
  && say control_analysis done || say control_analysis FAILED
say chain STAGE4_CONTROL_COMPLETE
