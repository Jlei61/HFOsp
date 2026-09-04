#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/honglab/leijiaxin/HFOsp
WT=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-rev22-postfit
STAGE="$ROOT/results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"
PYTHON=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
STATE="$STAGE/full_postfit_chain.status"
LOG="$STAGE/full_postfit_chain.log"
TRAINING_STATE="$STAGE/postfit_training_chain.status"
EXECUTION_CONFIG="$WT/config/topic4_rev22_dci_response_execution.json"
STRUCTURAL_CONFIG="$WT/config/topic4_rev22_dci_structural_null_execution.json"
FINAL_MANIFEST="$STAGE/response_fit/final_execution_candidate_manifest.json"
FROZEN="$STAGE/response_fit/frozen_candidates.json"
STRUCTURAL_MANIFEST="$STAGE/structural_nulls/candidate_manifest.json"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$STAGE"
exec >>"$LOG" 2>&1
cd "$WT"

write_state() {
    printf '%s checked_at=%s\n' "$1" "$(date -Is)" >"$STATE"
}

fail() {
    write_state "FAILED stage=$1"
    notify-send "Topic 4 rev22-DCI" "Full postfit stopped at $1" || true
    exit 2
}

assert_controller_complete() {
    local phase=$1
    local path="$STAGE/$phase/status/controller.json"
    "$PYTHON" - "$path" "$phase" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
phase = sys.argv[2]
payload = json.loads(path.read_text())
if payload.get("status") != "COMPLETE":
    raise SystemExit(f"{phase} controller is not complete: {payload.get('status')}")
counts = payload.get("state_counts") or {}
if int(counts.get("complete", -1)) != int(payload.get("job_count", -2)):
    raise SystemExit(f"{phase} controller has an incomplete inventory: {counts}")
PY
}

run_controller() {
    local phase=$1
    local config=$2
    local manifest=$3
    local commit=$4
    write_state "RUNNING_CONTROLLER phase=$phase commit=$commit"
    "$PYTHON" scripts/run_topic4_rev22_dci_controller.py \
        --config "$config" \
        --candidate-manifest "$manifest" \
        --phase "$phase" \
        --expected-commit "$commit" \
        --artifact-root "$ROOT" \
        --maximum-workers-override 16 || fail "${phase}_controller"
    assert_controller_complete "$phase" || fail "${phase}_inventory"
}

while true; do
    status=$(cat "$TRAINING_STATE" 2>/dev/null || true)
    write_state "WAITING_TRAINING state=${status// /_}"
    if [[ "$status" == COMPLETE_PENDING_EXECUTION_MANIFEST_BINDING* ]]; then
        break
    fi
    if [[ "$status" == STOPPED_* || "$status" == FAILED* ]]; then
        fail training_postfit
    fi
    sleep 600
done

if [[ -n "$(git status --porcelain=v1)" ]]; then
    fail dirty_worktree_before_freeze
fi

write_state "FREEZING_EXECUTION_CONTRACT"
"$PYTHON" scripts/freeze_topic4_rev22_execution_contract.py \
    --extra-candidates "$STAGE/response_fit/proposal_candidates.json" \
    --execution-config-out "$EXECUTION_CONFIG" \
    --candidate-manifest-out "$FINAL_MANIFEST" || fail execution_freeze

"$PYTHON" scripts/freeze_topic4_rev22_proposals.py \
    --bind-execution-manifest "$FINAL_MANIFEST" || fail candidate_binding

"$PYTHON" scripts/freeze_topic4_rev22_structural_nulls.py \
    --base-execution-config "$EXECUTION_CONFIG" \
    --execution-manifest "$FINAL_MANIFEST" \
    --frozen-candidates "$FROZEN" \
    --out-config "$STRUCTURAL_CONFIG" \
    --out-manifest "$STRUCTURAL_MANIFEST" || fail structural_freeze

unexpected=$(git status --porcelain=v1 | awk '{print $2}' | \
    grep -v -E '^config/topic4_rev22_dci_(response_execution|structural_null_execution)\.json$' || true)
if [[ -n "$unexpected" ]]; then
    fail unexpected_worktree_changes_after_freeze
fi
git add config/topic4_rev22_dci_response_execution.json \
        config/topic4_rev22_dci_structural_null_execution.json
if ! git diff --cached --quiet; then
    git commit -m "Freeze rev22 postfit execution contracts" || fail execution_contract_commit
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
    fail dirty_worktree_after_freeze
fi
EXECUTION_COMMIT=$(git rev-parse HEAD)
write_state "EXECUTION_CONTRACT_FROZEN commit=$EXECUTION_COMMIT"

run_controller qualification "$EXECUTION_CONFIG" "$FINAL_MANIFEST" "$EXECUTION_COMMIT"
run_controller confirmation "$EXECUTION_CONFIG" "$FINAL_MANIFEST" "$EXECUTION_COMMIT"
run_controller structural_nulls "$STRUCTURAL_CONFIG" "$STRUCTURAL_MANIFEST" "$EXECUTION_COMMIT"

write_state "AGGREGATING_TRAINING_ONLY_FROZEN_STAGES"
"$PYTHON" scripts/aggregate_topic4_rev22_frozen_stages.py \
    --candidate-manifest "$FINAL_MANIFEST" || fail frozen_stage_aggregate

write_state "OPENING_SELECTION_BLIND_VALIDATION"
"$PYTHON" scripts/aggregate_topic4_rev22_validation.py \
    --candidate-manifest "$FINAL_MANIFEST" || fail validation_aggregate

write_state "FITTING_DESCRIPTIVE_VALIDATION_SURFACES"
"$PYTHON" scripts/fit_topic4_rev22_validation_response.py || fail validation_response

write_state "AGGREGATING_STRUCTURAL_CONTROLS"
"$PYTHON" scripts/aggregate_topic4_rev22_structural_nulls.py || fail structural_aggregate

FIGURES="$STAGE/figures"
write_state "RENDERING_RESPONSE_AND_FAMILY_FIGURES"
"$PYTHON" scripts/plot_topic4_rev22_dci_results.py \
    --validation-aggregate "$STAGE/validation/validation_aggregate.json" \
    --validation-response-surfaces "$STAGE/validation/validation_response_surface.json" \
    --fit-aggregate "$STAGE/fit/aggregate/fit_aggregate.json" \
    --response-fit "$STAGE/response_fit/response_fit.json" \
    --frozen-candidates "$FROZEN" \
    --output-dir "$FIGURES" || fail main_figures
"$PYTHON" scripts/plot_topic4_rev22_structural_nulls.py \
    --aggregate "$STAGE/structural_nulls/structural_null_aggregate.json" \
    --out "$FIGURES/structural_nulls" || fail structural_figures

write_state "RENDERING_FINAL_CANDIDATE_ACCEPTANCE"
FINAL_ID_LIST="$STAGE/response_fit/final_acceptance_candidate_ids.txt"
"$PYTHON" - "$FROZEN" >"$FINAL_ID_LIST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
mapping = payload.get("mask_to_candidates") or {}
ids = mapping.get("M1111") or mapping.get("M1100") or []
if not ids:
    raise SystemExit("no frozen full-model candidate")
for candidate_id in ids:
    print(candidate_id)
PY
mapfile -t final_ids <"$FINAL_ID_LIST"
if (( ${#final_ids[@]} == 0 )); then
    fail no_final_acceptance_candidate
fi
for candidate_id in "${final_ids[@]}"; do
    "$PYTHON" scripts/paper_figures/plot_topic4_rev22_final_candidate_acceptance.py \
        --config "$WT/config/topic4_rev22_dci_dual_core_interictal_identifiability.json" \
        --validation "$STAGE/validation/validation_aggregate.json" \
        --frozen-candidates "$FROZEN" \
        --candidate-manifest "$FINAL_MANIFEST" \
        --confirmation-workers "$STAGE/confirmation/workers" \
        --candidate-id "$candidate_id" \
        --output-dir "$FIGURES/final_candidate/$candidate_id" \
        --artifact-root "$ROOT" || fail "final_acceptance_${candidate_id}"
done

write_state "RUNNING_COMPLETION_AUDIT"
"$PYTHON" scripts/audit_topic4_rev22_completion.py \
    --stage "$STAGE" --worktree "$WT" || fail completion_audit

write_state "COMPLETE commit=$EXECUTION_COMMIT final_candidates=${final_ids[*]}"
notify-send "Topic 4 rev22-DCI" "Full postfit and figures complete" || true
