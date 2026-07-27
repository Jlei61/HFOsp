#!/usr/bin/env bash
# Durable post-confirmation coordinator for Rev3.1 Tasks 9A/9B routing.
#
# It never decides that a carrier exists.  It waits for the fail-closed
# branch_verdict.json, starts exactly two seed workers only after the registered
# two-seed confirmation passes, and stops on a failed confirmation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

out="results/topic4_sef_hfo/zm_branch_decision"
mkdir -p "$out/logs"
coordinator_log="$out/logs/postcarrier_coordinator.log"

verdict_status() {
    python - "$out/branch_verdict.json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    print("pending")
    raise SystemExit(0)
d = json.load(open(path))
confirmation = d.get("confirmation") or {}
if (
    d.get("verdict") == "carrier_at_visited_states"
    and confirmation.get("status") == "passed"
):
    print("passed")
elif confirmation.get("status") == "failed":
    print("failed")
else:
    print("pending")
PY
}

echo "[postcarrier] $(date -Is) waiting for registered confirmation" >>"$coordinator_log"
while true; do
    status="$(verdict_status)"
    case "$status" in
        passed)
            echo "[postcarrier] $(date -Is) confirmation passed" >>"$coordinator_log"
            break
            ;;
        failed)
            echo "[postcarrier] $(date -Is) confirmation failed; stop" >>"$coordinator_log"
            exit 3
            ;;
        pending)
            sleep 60
            ;;
        *)
            echo "[postcarrier] invalid verdict status: $status" >&2
            exit 2
            ;;
    esac
done

source_complete() {
    local seed="$1"
    python - "$out/source_rhythm/dt/seed${seed}/source_rhythm.json" "$seed" <<'PY'
import json
import os
import sys

path, seed = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path):
    raise SystemExit(1)
d = json.load(open(path))
ok = (
    int(d.get("seed", -1)) == seed
    and d.get("resolution") == "dt"
    and (d.get("source_rhythm") or {}).get("source_temporal_class") is not None
    and os.path.exists(d.get("fields_path", ""))
)
raise SystemExit(0 if ok else 1)
PY
}

run_source() {
    local seed="$1"
    local log="$out/logs/postcarrier_seed${seed}.log"
    echo "[postcarrier] $(date -Is) seed=$seed source start" >>"$log"
    if source_complete "$seed"; then
        echo "[postcarrier] $(date -Is) seed=$seed reuse source-rhythm audit" >>"$log"
    else
        python scripts/run_topic4_zm_branch_decision.py \
            --phase source_rhythm --seed "$seed" --resolution dt --confirm-run \
            >>"$log" 2>&1
    fi
    echo "[postcarrier] $(date -Is) seed=$seed source complete" >>"$log"
}

run_rank() {
    local seed="$1"
    local log="$out/logs/postcarrier_seed${seed}.log"
    echo "[postcarrier] $(date -Is) seed=$seed rank start" >>"$log"
    python scripts/run_topic4_zm_branch_decision.py \
        --phase effective_rank --seed "$seed" --resolution dt --confirm-run \
        >>"$log" 2>&1
    echo "[postcarrier] $(date -Is) seed=$seed rank complete" >>"$log"
}

run_source 1 &
pid1=$!
run_source 3 &
pid3=$!

rc=0
wait "$pid1" || rc=$?
wait "$pid3" || rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "[postcarrier] $(date -Is) worker failure rc=$rc" >>"$coordinator_log"
    exit "$rc"
fi

python scripts/analyze_topic4_zm_source_rhythm.py >>"$coordinator_log" 2>&1

run_rank 1 &
pid1=$!
run_rank 3 &
pid3=$!

rc=0
wait "$pid1" || rc=$?
wait "$pid3" || rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "[postcarrier] $(date -Is) rank worker failure rc=$rc" >>"$coordinator_log"
    exit "$rc"
fi

python scripts/analyze_topic4_zm_effective_rank.py >>"$coordinator_log" 2>&1
python scripts/plot_topic4_zm_branch_decision.py >>"$coordinator_log" 2>&1
echo "[postcarrier] $(date -Is) complete" >>"$coordinator_log"
