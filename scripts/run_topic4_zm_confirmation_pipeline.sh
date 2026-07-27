#!/usr/bin/env bash
# Crash-visible, two-stage numerical confirmation for one positive Z/M carrier seed.
#
# Usage:
#   scripts/run_topic4_zm_confirmation_pipeline.sh SEED [WAIT_PID]
#
# WAIT_PID is the discovery worker that currently owns this seed's resource
# slot.  The pipeline waits for it to exit, then runs exactly one long worker
# at a time.  All scientific outputs are written atomically by the Python
# runner; this wrapper only supplies durable logs and phase ordering.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 SEED [WAIT_PID]" >&2
    exit 2
fi

seed="$1"
wait_pid="${2:-}"
case "$seed" in
    1|3|4) ;;
    *) echo "seed must be one of 1,3,4" >&2; exit 2 ;;
esac
if [[ -n "$wait_pid" && ! "$wait_pid" =~ ^[0-9]+$ ]]; then
    echo "WAIT_PID must be an integer" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

out="results/topic4_sef_hfo/zm_branch_decision"
mkdir -p "$out/logs"
log="$out/logs/confirmation_seed${seed}.log"

adjudicate_on_exit() {
    rc=$?
    trap - EXIT
    {
        echo "[pipeline] $(date -Is) seed=$seed exit=$rc; recomputing verdict"
        python scripts/adjudicate_topic4_zm_branch_decision.py
    } >>"$log" 2>&1 || true
    exit "$rc"
}
trap adjudicate_on_exit EXIT

if [[ -n "$wait_pid" ]]; then
    echo "[pipeline] $(date -Is) seed=$seed waiting for discovery pid=$wait_pid" >>"$log"
    while kill -0 "$wait_pid" 2>/dev/null; do
        sleep 30
    done
fi

echo "[pipeline] $(date -Is) seed=$seed starting native dt/2 anchor" >>"$log"
anchor="$out/anchors_dt2/seed${seed}/anchor.json"
if python - "$anchor" <<'PY'
import json
import os
import sys

p = sys.argv[1]
if not os.path.exists(p):
    raise SystemExit(1)
d = json.load(open(p))
tags = {
    f"{x.get('bin_name')}__{x.get('fast_phase')}"
    for x in d.get("captured_states", [])
}
ok = (
    d.get("resolution") == "dt2"
    and float(d.get("dt", 0.0)) == 0.05
    and bool((d.get("selection") or {}).get("eligibility", {}).get("eligible"))
    and "bounded_mid__peak" in tags
)
raise SystemExit(0 if ok else 1)
PY
then
    echo "[pipeline] $(date -Is) seed=$seed reusing verified dt/2 anchor" >>"$log"
else
    python scripts/run_topic4_zm_branch_decision.py \
        --phase anchor --seed "$seed" --resolution dt2 --T 15000 --confirm-run \
        >>"$log" 2>&1
fi

echo "[pipeline] $(date -Is) seed=$seed starting native dt/2 fork" >>"$log"
python scripts/run_topic4_zm_branch_decision.py \
    --phase fork --seed "$seed" --resolution dt2 \
    --evidence-tier dt2_confirmation \
    --states bounded_mid__peak --arms freeze_all --replicates noise_replay \
    --T-cont 8000 --confirm-run --dump-traces \
    >>"$log" 2>&1

echo "[pipeline] $(date -Is) seed=$seed starting 20 s original-dt fork" >>"$log"
python scripts/run_topic4_zm_branch_decision.py \
    --phase fork --seed "$seed" --resolution dt \
    --evidence-tier long_confirmation \
    --states bounded_mid__peak --arms freeze_all --replicates noise_replay \
    --T-cont 20000 --confirm-run --dump-traces \
    >>"$log" 2>&1

echo "[pipeline] $(date -Is) seed=$seed complete" >>"$log"
