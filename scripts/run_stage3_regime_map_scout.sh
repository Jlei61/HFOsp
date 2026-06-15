#!/usr/bin/env bash
# Stage 3 local-global REGIME-MAP scout (2026-06-15). Coarse core_mean x sep_frac grid, SHORT T,
# few seeds -> maps local / relay / collision regimes. NOT a long formal run (long sims are
# post-gate confirmation only). RAM-safe: ~13GB/sim, cap concurrency + free-RAM gate (OOM lesson).
set -u
cd /home/honglab/leijiaxin/HFOsp
OUT=results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map
mkdir -p "$OUT/logs"
MEANS="${MEANS:-16.5 17.0 17.5}"
SEPS="${SEPS:-0.6 0.7 0.8}"
SEEDS="${SEEDS:-1 2 3}"
MAXJOBS="${MAXJOBS:-5}"            # 5 x ~13GB = 65GB
MIN_FREE_GB="${MIN_FREE_GB:-40}"
T="${T:-3000}"

free_gb(){ free -g | awk '/^Mem:/{print $7}'; }
running(){ pgrep -f "run_sef_hfo_snn_cm_spontaneous_readout.py.*regime_map" | grep -vc grep; }
# IMPROVE-8: skip only if BOTH sidecar AND readout exist and are loadable JSON (no false skip on half-write)
done_ok(){ python3 - "$OUT/sidecar_$1.json" "$OUT/readout_$1.json" <<'PY' 2>/dev/null
import sys, json
for p in sys.argv[1:]:
    json.load(open(p))
PY
}

launch(){  # $1=mean $2=sep $3=seed
  tag="rm_m${1}_sep${2}_s${3}"
  if done_ok "$tag"; then echo "[skip] $tag (complete)"; return; fi
  while [ "$(running)" -ge "$MAXJOBS" ] || [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 30; done
  echo "[$(date +%H:%M:%S)] launch $tag (running=$(running) free=$(free_gb)G)"
  nohup python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
    --lesion twoend_equal --core-mean "$1" --sep-frac "$2" --core-std 1.0 --seed "$3" \
    --T "$T" --dump-fullfield --tag "$tag" --out "$OUT" > "$OUT/logs/$tag.log" 2>&1 &
  sleep 5
}

echo "SCOUT start $(date +%H:%M:%S): means=[$MEANS] seps=[$SEPS] seeds=[$SEEDS] T=$T maxjobs=$MAXJOBS"
for m in $MEANS; do for sp in $SEPS; do for s in $SEEDS; do launch "$m" "$sp" "$s"; done; done; done
wait
echo "SCOUT DONE $(date +%H:%M:%S)"
