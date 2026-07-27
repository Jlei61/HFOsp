#!/usr/bin/env bash
# Wait-only finalizer for the Rev3.1 post-carrier phases.
#
# This script never launches an SNN simulation.  Independent tmux workers own
# one seed/phase manifest each.  The finalizer waits for those atomic manifests,
# fails closed if a required worker disappears before completion, then runs the
# registered aggregators, adjudicator and diagnostic plotter.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

out="results/topic4_sef_hfo/zm_branch_decision"
log="$out/logs/finalize_when_ready.log"
mkdir -p "$(dirname "$log")"

json_true() {
    local path="$1"
    local field="$2"
    python - "$path" "$field" <<'PY'
import json
import os
import sys

path, field = sys.argv[1:]
if not os.path.exists(path):
    raise SystemExit(1)
payload = json.load(open(path))
raise SystemExit(0 if payload.get(field) is True else 1)
PY
}

worker_alive() {
    local phase="$1"
    local seed="$2"
    pgrep -f \
        "python scripts/run_topic4_zm_branch_decision.py --phase ${phase} --seed ${seed} " \
        >/dev/null
}

assert_phase_workers() {
    local phase="$1"
    local subdir="$2"
    local filename="$3"
    local complete_field="$4"
    local seed
    for seed in 1 3 4; do
        local path="$out/$subdir/seed${seed}/$filename"
        if json_true "$path" "$complete_field"; then
            continue
        fi
        if ! worker_alive "$phase" "$seed"; then
            echo "[finalizer] $(date -Is) P0: ${phase} seed=${seed} " \
                 "worker absent before ${complete_field}=true" >>"$log"
            exit 4
        fi
    done
}

assert_all_workers() {
    assert_phase_workers \
        effective_rank effective_rank rank_probes.json probe_matrix_complete
    assert_phase_workers \
        entry_boundary boundaries/entry entry_probes.json complete
    assert_phase_workers \
        offset_boundary boundaries/offset offset_probes.json complete
}

wait_phase() {
    local phase="$1"
    local subdir="$2"
    local filename="$3"
    local complete_field="$4"
    local seed
    while true; do
        assert_all_workers
        local pending=0
        for seed in 1 3 4; do
            local path="$out/$subdir/seed${seed}/$filename"
            if json_true "$path" "$complete_field"; then
                continue
            fi
            pending=$((pending + 1))
        done
        echo "[finalizer] $(date -Is) ${phase} pending=${pending}" >>"$log"
        if [[ "$pending" -eq 0 ]]; then
            return 0
        fi
        sleep 120
    done
}

echo "[finalizer] $(date -Is) wait-only finalizer started" >>"$log"

wait_phase effective_rank effective_rank rank_probes.json probe_matrix_complete
python scripts/analyze_topic4_zm_effective_rank.py >>"$log" 2>&1

# Source class is already a registered three-seed disagreement.  The finalizer
# records that the explanatory modal/Floquet branch remains closed.
python - "$out/source_rhythm/source_rhythm_summary.json" <<'PY' >>"$log" 2>&1
import json
import sys

payload = json.load(open(sys.argv[1]))
print(
    "[finalizer] source-rhythm status="
    f"{payload.get('status')} carrier_type={payload.get('carrier_type')}; "
    "modal operator not launched"
)
PY

wait_phase entry_boundary boundaries/entry entry_probes.json complete
python scripts/analyze_topic4_zm_entry_boundary.py >>"$log" 2>&1

wait_phase offset_boundary boundaries/offset offset_probes.json complete
python scripts/analyze_topic4_zm_offset_boundary.py >>"$log" 2>&1

python scripts/adjudicate_topic4_zm_branch_decision.py >>"$log" 2>&1
python scripts/plot_topic4_zm_branch_decision.py >>"$log" 2>&1
echo "[finalizer] $(date -Is) complete" >>"$log"
