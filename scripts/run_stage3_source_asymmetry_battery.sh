#!/usr/bin/env bash
# Stage 3 source-asymmetry re-run battery (2026-06-15).
# Tests WHY two parameter-equal foci still split by source. Three conditions on the candidate cell
# (sep0.7/std1.0/m17.5), per seed, with --dump-fullfield:
#   base   : re-establish per-seed winner (hidden neg_clean vs pos_clean) + full-field local-vs-global
#   swap   : swap the two cores' threshold RNG draws -> does the winner flip END? (=> threshold-driven)
#   mirror : identical threshold profile on both cores -> read-out still biased? (=> geom/readout)
#
# RAM-SAFE + CONFLICT-SAFE (OOM lesson 2026-06-14): ~13GB/sim; cap concurrency AND gate on free RAM so
# it auto-throttles around any other user's running sims. Launch in background; idempotent-ish (skips a
# tag whose readout_*.json already exists).
set -u
cd /home/honglab/leijiaxin/HFOsp
OUT=results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/asym_reruns
mkdir -p "$OUT/logs"
SEEDS="${SEEDS:-1 2 3 4 5 6}"
MAXJOBS="${MAXJOBS:-5}"            # 5 x ~13GB = 65GB
MIN_FREE_GB="${MIN_FREE_GB:-40}"  # don't launch a new sim if 'available' RAM is below this
T="${T:-3000}"
CELL="--sep-frac 0.7 --core-std 1.0 --core-mean 17.5"

free_gb(){ free -g | awk '/^Mem:/{print $7}'; }                                  # 'available'
running(){ pgrep -f "run_sef_hfo_snn_cm_spontaneous_readout.py.*asym_reruns" | grep -vc grep; }

launch(){  # $1=extra_flags  $2=tag  $3=seed
  if [ -f "$OUT/readout_$2.json" ]; then echo "[skip] $2 (exists)"; return; fi
  while [ "$(running)" -ge "$MAXJOBS" ] || [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 30; done
  echo "[$(date +%H:%M:%S)] launch $2 (running=$(running) free=$(free_gb)G)"
  nohup python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
    --lesion twoend_equal $CELL --seed "$3" --T "$T" --dump-fullfield $1 \
    --tag "$2" --out "$OUT" > "$OUT/logs/$2.log" 2>&1 &
  sleep 5   # stagger connectivity-build RAM spikes
}

echo "BATTERY start $(date +%H:%M:%S): seeds=[$SEEDS] T=$T maxjobs=$MAXJOBS min_free=${MIN_FREE_GB}G"
for s in $SEEDS; do
  launch ""            "base_s$s"   "$s"
  launch "--swap-vth"  "swap_s$s"   "$s"
  launch "--mirror-vth" "mirror_s$s" "$s"
done
wait
echo "BATTERY DONE $(date +%H:%M:%S)"
