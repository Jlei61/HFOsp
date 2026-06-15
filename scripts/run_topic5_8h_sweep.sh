#!/usr/bin/env bash
# Topic5 unattended 8h sweep (2026-06-14, user-confirmed: 3 nulls + all activation metrics + B-line EI).
# Robust ordering: early 2-null results land BEFORE the slow baseline-augment, so an augment
# failure can never lose the primary results; the 3-null pass then overwrites with the fuller form.
set -u
cd /home/honglab/leijiaxin/HFOsp || exit 1
ROOT=results/topic5_ictal_recruitment
CACHE_LOG=$ROOT/t0_feature_cache_build.log
CACHE_PID=4068300
PY=python
METRICS="broadband hfa ramp ei"
BS="1000 2000"

echo "[sweep] $(date) START — waiting for cache build (CACHE BUILD DONE or pid $CACHE_PID gone)"
for i in $(seq 1 300); do                       # up to 5h safety wait
  grep -q 'CACHE BUILD DONE' "$CACHE_LOG" 2>/dev/null && { echo "[sweep] cache build DONE"; break; }
  kill -0 "$CACHE_PID" 2>/dev/null || { echo "[sweep] cache pid gone; proceeding with cached subjects"; break; }
  sleep 60
done

echo "[sweep] $(date) STEP add-ei (fast, from cached HFA trace, no reload)"
$PY scripts/build_topic5_t0_feature_cache.py --add-ei

echo "[sweep] $(date) STEP A-line 2-null sweep (early results; anchor-matched = NA until augment)"
for m in $METRICS; do for B in $BS; do
  echo "[sweep] $(date) A-line(2null) activation=$m B=$B"
  $PY scripts/run_topic5_axis_alignment.py --activation "$m" --B "$B" || echo "[sweep] FAILED $m B=$B"
done; done

echo "[sweep] $(date) STEP augment baseline-activity (reloads EDFs, ~1.5h) -> enables anchor-matched null"
$PY scripts/build_topic5_t0_feature_cache.py --augment-baseline

echo "[sweep] $(date) STEP A-line 3-null sweep (now with anchor-matched; overwrites the 2-null JSONs)"
for m in $METRICS; do for B in $BS; do
  echo "[sweep] $(date) A-line(3null) activation=$m B=$B"
  $PY scripts/run_topic5_axis_alignment.py --activation "$m" --B "$B" || echo "[sweep] FAILED(3null) $m B=$B"
done; done

echo "[sweep] $(date) STEP figures (best-effort)"
$PY scripts/plot_topic5_axis_alignment.py 1000 || echo "[sweep] fig B1000 failed"
$PY scripts/plot_topic5_axis_alignment.py 2000 || echo "[sweep] fig B2000 failed"

echo "SWEEP_ALL_DONE $(date)"
