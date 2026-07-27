#!/usr/bin/env python3
"""Freeze the v2.2.1 closeout and transition-decomposition manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V22 = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
DECOMPOSITION = ROOT / "results/topic5_interictal_transition_decomposition_v0_1"
CLINICAL = ROOT / "results/topic5_clinical_onset_source_annotation_v0_1"
OUTPUT = DECOMPOSITION / "FINAL_REPRODUCIBILITY_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    required = [
        V22 / "formal/TRAINER_EPOCH_AUDIT.json",
        V22 / "formal/analysis/INTERICTAL_CLAIM_SUMMARY.json",
        V22 / "target_audit/TARGET_METADATA_GATE.json",
        V22 / "closeout_v2_2_1/CLOSEOUT_STATUS.json",
        V22 / "closeout_v2_2_1/SCORING_CONTRACT_AUDIT.json",
        V22 / "closeout_v2_2_1/model_class_comparison.csv",
        V22 / "closeout_v2_2_1/calibration_step_summary.csv",
        V22 / "closeout_v2_2_1/operator_identifiability_patient_metrics.csv",
        V22
        / "closeout_v2_2_1/figures/v2_2_1_closeout_diagnostics.png",
        V22 / "closeout_v2_2_1/figures/README.md",
        DECOMPOSITION / "DECOMPOSITION_STATUS.json",
        DECOMPOSITION / "SCORING_CONTRACT_AUDIT.json",
        DECOMPOSITION / "CROSS_SHAFT_STATUS.json",
        DECOMPOSITION / "patient_model_metrics.csv",
        DECOMPOSITION / "cohort_comparisons.csv",
        DECOMPOSITION / "operator_component_metrics.csv",
        DECOMPOSITION / "history_depth_metrics.csv",
        DECOMPOSITION / "cross_shaft_positive_metrics.csv",
        DECOMPOSITION / "cross_shaft_prefix_metrics.csv",
        DECOMPOSITION / "cross_shaft_eligibility.csv",
        DECOMPOSITION / "figures/transition_signal_decomposition.png",
        DECOMPOSITION
        / "figures/transition_signal_decomposition_paper_ready.png",
        DECOMPOSITION
        / "figures/transition_signal_decomposition_paper_ready.pdf",
        DECOMPOSITION / "figures/README.md",
        DECOMPOSITION / "paper_ready_plot.log",
        CLINICAL / "READINESS_STATUS.json",
        CLINICAL / "BLINDING_CONTRACT.json",
        CLINICAL / "annotation_registry.csv",
        ROOT
        / "docs/superpowers/specs/"
        "2026-07-27-topic5-interictal-transition-signal-decomposition-v0_1.md",
        ROOT
        / "docs/superpowers/plans/"
        "2026-07-27-topic5-interictal-transition-signal-decomposition-v0_1.md",
        ROOT / "src/topic5_transition_decomposition_v0_1.py",
        ROOT / "scripts/run_topic5_transition_decomposition_v0_1.py",
        ROOT / "scripts/plot_topic5_transition_decomposition_v0_1.py",
        ROOT
        / "docs/archive/topic5/"
        "interictal_transition_signal_decomposition_v0_1_result_2026-07-27.md",
        ROOT
        / "docs/paper-draft/"
        "interictal_transition_signal_decomposition.md",
        ROOT
        / "results/run_logs/"
        "topic5_v2_2_1_closeout_transition_decomposition_pytest_2026-07-27.log",
        ROOT
        / "results/run_logs/"
        "topic5_transition_decomposition_source_only_patch_pytest_2026-07-27.log",
        ROOT
        / "results/run_logs/"
        "topic5_transition_decomposition_final_contract_pytest_2026-07-27.log",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    epoch = load_json(V22 / "formal/TRAINER_EPOCH_AUDIT.json")
    claims = load_json(V22 / "formal/analysis/INTERICTAL_CLAIM_SUMMARY.json")
    target = load_json(V22 / "target_audit/TARGET_METADATA_GATE.json")
    closeout = load_json(V22 / "closeout_v2_2_1/CLOSEOUT_STATUS.json")
    closeout_scoring = load_json(
        V22 / "closeout_v2_2_1/SCORING_CONTRACT_AUDIT.json"
    )
    decomposition = load_json(DECOMPOSITION / "DECOMPOSITION_STATUS.json")
    decomposition_scoring = load_json(
        DECOMPOSITION / "SCORING_CONTRACT_AUDIT.json"
    )
    clinical = load_json(CLINICAL / "READINESS_STATUS.json")

    checks = {
        "epoch_audit_pass": epoch.get("status") == "PASS",
        "formal_configs_66_of_66": (
            epoch.get("formal_resolved_configs_audited") == 66
        ),
        "claim1_fail": claims.get("claim1_predictive_adequacy") == "FAIL",
        "claim2_next_fail": claims.get("claim_statuses", {}).get("claim2_next")
        == "FAIL",
        "claim2_future_fail": claims.get("claim_statuses", {}).get("claim2_future")
        == "FAIL",
        "claim3_locked": claims.get("claim_statuses", {}).get(
            "claim3_random_axis"
        )
        == "LOCKED_NOT_RUN",
        "claim4_locked": claims.get("claim_statuses", {}).get(
            "claim4_shared_scaffold"
        )
        == "LOCKED_NOT_RUN",
        "closeout_complete": closeout.get("status") == "COMPLETE",
        "closeout_scoring_pass": closeout_scoring.get("status") == "PASS",
        "decomposition_complete": decomposition.get("status") == "COMPLETE",
        "decomposition_scoring_pass": (
            decomposition_scoring.get("status") == "PASS"
        ),
        "clinical_source_still_blinded_pending": (
            clinical.get("status") == "AWAITING_BLINDED_MANUAL_ANNOTATION"
            and clinical.get("consensus_exact_seizures") == 0
        ),
        "target_values_never_read": not any(
            [
                target.get("energy_values_read", False),
                target.get("recruitment_values_read", False),
                closeout.get("target_values_read", False),
                decomposition.get("target_values_read", False),
                clinical.get("target_values_read", False),
            ]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"final reproducibility checks failed: {failed}")

    payload = {
        "contract": "topic5_v2_2_1_closeout_and_transition_decomposition",
        "status": "COMPLETE",
        "scientific_decision": decomposition["decision"],
        "checks": checks,
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in required
        },
        "target_values_read": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
