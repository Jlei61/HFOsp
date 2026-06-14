#!/usr/bin/env bash
# Stage 3 regime screen — 8h autonomous parameter exploration (2026-06-14, user-confirmed).
#
# Goal: find a twoend operating point / geometry giving a BALANCED bidirectional regime
# (collision_rate < 30% AND both ends >=3 clean source events AND both directions present),
# OR establish that none exists in this parameter box (a real conclusion, not a failure).
#
# Pre-registered SOURCE-based gate is applied later by the agent on the raw CSV (kept out of this
# unattended script). Sims are independent + parallel; one crash does not stop the rest.
#
# Grid: Phase 0 sign sanity (oneend known-source) + Phase 1 twoend_equal sep x std x mean x seed
#       + Phase 2 twoend_deph fallback arm.  61 runs, T=3000, ~25min/run, 12-way parallel ~ 3-4h.
set -u
cd "$(dirname "$0")/.." || exit 1
RUNNER=scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
OUT=results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_screen
LOG="$OUT/logs"
mkdir -p "$LOG"
T=${T:-3000}
NPAR=${NPAR:-4}   # conservative default: a science screen must not let resource contention manufacture false negatives (OOM 2026-06-14)
CMDS="$LOG/_cmds.txt"
: > "$CMDS"

emit() {  # emit <tag> <runner-args...>
  local tag="$1"; shift
  echo "python $RUNNER $* --T $T --out $OUT --tag $tag > $LOG/$tag.log 2>&1" >> "$CMDS"
}

# Phase 0 — known-source sign sanity (oneend_neg -> expect mostly +1, oneend_pos -> expect mostly -1)
for s in 1 2; do
  emit "gs_sign_oneendneg_s$s" --lesion oneend_neg --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --seed "$s"
  emit "gs_sign_oneendpos_s$s" --lesion oneend_pos --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --seed "$s"
done
# Phase 1 — twoend_equal grid: separation x spread x operating-point x seed
for sep in 0.6 0.7 0.8; do
  for std in 0.5 1.0 1.5; do
    for m in 17.0 17.5; do
      for s in 1 2 3; do
        emit "gs_te_sep${sep}_std${std}_m${m}_s${s}" \
          --lesion twoend_equal --core-mean "$m" --core-std "$std" --sep-frac "$sep" --seed "$s"
      done
    done
  done
done
# Phase 2 — twoend_deph fallback arm (the known-working separation, for comparison only)
for s in 1 2 3; do
  emit "gs_deph_s$s" --lesion twoend_deph --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --dephase 0.3 --seed "$s"
done

# --- preflight: never OOM. cap NPAR to available RAM (~14GB/run, keep 20GB headroom) + warn on
#     concurrent sims (a co-running sweep + this grid is exactly what blew the machine 2026-06-14). ---
PER_RUN_GB=14
AVAIL_GB=$(free -g | awk '/Mem/{print $7}')
ACTIVE=$(pgrep -cf "run_sef_hfo_snn_cm_spontaneous_readou[t]" || true)
SAFE_NPAR=$(( (AVAIL_GB - 20) / PER_RUN_GB )); [ "$SAFE_NPAR" -lt 1 ] && SAFE_NPAR=1
if [ "$NPAR" -gt "$SAFE_NPAR" ]; then
  echo "[regime_screen][preflight] capping NPAR $NPAR -> $SAFE_NPAR (avail ${AVAIL_GB}G, ~${PER_RUN_GB}G/run)"
  NPAR=$SAFE_NPAR
fi
[ "${ACTIVE:-0}" -gt 0 ] && echo "[regime_screen][preflight] WARNING: $ACTIVE other sim(s) already running — may still contend at NPAR=$NPAR"

N=$(wc -l < "$CMDS")
echo "[regime_screen] $N runs x T=$T, ${NPAR}-way parallel -> $OUT  (start $(date '+%F %H:%M'))"
xargs -P "$NPAR" -I CMD bash -c CMD < "$CMDS"
echo "[regime_screen] sims done ($(date '+%F %H:%M')); aggregating..."
python scripts/aggregate_stage3_regime_screen.py --dir "$OUT"
echo "[regime_screen] DONE ($(date '+%F %H:%M')). raw CSV: $OUT/regime_screen_raw.csv"
