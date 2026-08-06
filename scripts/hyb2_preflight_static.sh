#!/usr/bin/env bash
# Pre-launch static gate for HYB2 (added after TWO 41-minute runs died in the post-simulation
# reduction: a duplicate dict keyword, then a stale local left behind by an extraction).
# pyflakes F821 catches undefined names in milliseconds; there is no excuse for finding them
# after 41 minutes of compute.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
for f in scripts/run_topic4_fcxr_hyb2.py scripts/hyb2_resource_watchdog.py \
         src/topic4_fcxr_hyb2.py src/snn_engine/event_limited_recruitment.py; do
  out=$(python -m pyflakes "$f" 2>&1 | grep -E "undefined name" || true)
  if [ -n "$out" ]; then echo "UNDEFINED NAME in $f:"; echo "$out"; fail=1; fi
done
python -m pytest -q tests/test_topic4_fcxr_hyb2.py tests/test_event_limited_recruitment.py \
  >/dev/null 2>&1 || { echo "HYB2 unit tests FAILED"; fail=1; }
[ "$fail" -eq 0 ] && echo "PRE-LAUNCH STATIC GATE: PASS" || { echo "PRE-LAUNCH STATIC GATE: FAIL"; exit 1; }
