#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/honglab/leijiaxin/HFOsp
STAGE="$ROOT/results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"
FIT_STATUS="$STAGE/fit/run_logs/workers"
DECOMP_STATUS="$STAGE/decomposition/run_logs/workers"
DECOMP_CONTROLLER="$STAGE/decomposition/status/controller.json"
STATE="$STAGE/postfit_training_chain.status"
LOG="$STAGE/postfit_training_chain.log"
PYTHON=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
EXPECTED_HEAD="$(git rev-parse HEAD)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec >>"$LOG" 2>&1

count_status() {
    local directory=$1
    local state=$2
    local count=0
    local file
    shopt -s nullglob
    for file in "$directory"/*.status; do
        if grep -q "^${state} " "$file"; then
            count=$((count + 1))
        fi
    done
    shopt -u nullglob
    printf '%s\n' "$count"
}

write_state() {
    printf '%s checked_at=%s\n' "$1" "$(date -Is)" >"$STATE"
}

while true; do
    fit_ok=$(count_status "$FIT_STATUS" SUCCESS)
    fit_bad=$(count_status "$FIT_STATUS" FAILED)
    decomp_ok=$(count_status "$DECOMP_STATUS" SUCCESS)
    decomp_bad=$(count_status "$DECOMP_STATUS" FAILED)
    write_state "WAITING fit_success=$fit_ok fit_failed=$fit_bad decomposition_success=$decomp_ok decomposition_failed=$decomp_bad"
    if (( fit_bad > 0 || decomp_bad > 0 )); then
        write_state "STOPPED_UPSTREAM_FAILURE fit_success=$fit_ok fit_failed=$fit_bad decomposition_success=$decomp_ok decomposition_failed=$decomp_bad"
        notify-send "Topic 4 rev22-DCI" "Training postfit stopped: upstream failure" || true
        exit 2
    fi
    if (( fit_ok == 384 && decomp_ok == 16 )) && \
       [[ -f "$DECOMP_CONTROLLER" ]] && \
       grep -q '"status": "COMPLETE"' "$DECOMP_CONTROLLER"; then
        break
    fi
    sleep 600
done

if [[ -n "$(git status --porcelain=v1)" ]] || [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
    write_state "STOPPED_POSTFIT_PROVENANCE_DRIFT expected_head=$EXPECTED_HEAD actual_head=$(git rev-parse HEAD)"
    notify-send "Topic 4 rev22-DCI" "Training postfit stopped: worktree drift" || true
    exit 3
fi

write_state "RUNNING_AGGREGATE fit_success=384 decomposition_success=16"
"$PYTHON" scripts/aggregate_topic4_rev22_fit.py

write_state "RUNNING_RESPONSE_FIT fit_success=384 decomposition_success=16"
"$PYTHON" scripts/freeze_topic4_rev22_proposals.py

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
            "data_driven_dual_core_interictal_identifiability")
aggregate = json.loads((root / "fit/aggregate/fit_aggregate.json").read_text())
frozen = json.loads((root / "response_fit/frozen_candidates.json").read_text())
if aggregate.get("status") != "FIT_AGGREGATE_COMPLETE":
    raise SystemExit("fit aggregate did not complete")
if frozen.get("status") != "PENDING_EXECUTION_MANIFEST_BINDING":
    raise SystemExit("candidate freeze crossed the expected handoff boundary")
PY

write_state "COMPLETE_PENDING_EXECUTION_MANIFEST_BINDING fit_success=384 decomposition_success=16"
notify-send "Topic 4 rev22-DCI" "Training response fit complete; candidates await manifest binding" || true
