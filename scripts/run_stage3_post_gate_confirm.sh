#!/usr/bin/env bash
# Stage 3 POST-GATE confirmation (2026-06-15). ONLY the single advance-gate-PASS cell.
# Bounded, conservative (user 2026-06-15): start T=30000 (10x scout), 3 seeds — do NOT go wide/huge
# first. If pos-end clean global < ~50 or a seed stays weak, re-invoke with T=60000 (not more parallel).
# Goal: accumulate enough to test the TEMPLATE-READINESS (second-tier) gate, NOT yet the template test.
# RAM-safe: ~13GB/sim, cap + free-RAM gate (OOM lesson). Engine guard aborts if the engine drifts.
set -u
cd /home/honglab/leijiaxin/HFOsp
OUT=results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/post_gate
mkdir -p "$OUT/logs"
MEAN="${MEAN:-17.5}"; SEP="${SEP:-0.7}"; STD="${STD:-1.0}"
T="${T:-30000}"
SEEDS="${SEEDS:-1 2 3}"
MAXJOBS="${MAXJOBS:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

free_gb(){ free -g | awk '/^Mem:/{print $7}'; }
running(){ pgrep -f "run_sef_hfo_snn_cm_spontaneous_readout.py.*post_gate" | grep -vc grep; }
done_ok(){ python3 - "$OUT/sidecar_$1.json" "$OUT/readout_$1.json" <<'PY' 2>/dev/null
import sys, json
for p in sys.argv[1:]:
    json.load(open(p))
PY
}

echo "POST-GATE start $(date +%H:%M:%S): cell m$MEAN/sep$SEP/std$STD T=$T seeds=[$SEEDS] maxjobs=$MAXJOBS"
for s in $SEEDS; do
  tag="pg_m${MEAN}_sep${SEP}_T${T}_s${s}"
  if done_ok "$tag"; then echo "[skip] $tag"; continue; fi
  while [ "$(running)" -ge "$MAXJOBS" ] || [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 30; done
  echo "[$(date +%H:%M:%S)] launch $tag (running=$(running) free=$(free_gb)G)"
  nohup python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
    --lesion twoend_equal --core-mean "$MEAN" --sep-frac "$SEP" --core-std "$STD" --seed "$s" \
    --T "$T" --dump-fullfield --tag "$tag" --out "$OUT" > "$OUT/logs/$tag.log" 2>&1 &
  sleep 5
done
wait
echo "POST-GATE DONE $(date +%H:%M:%S)"
