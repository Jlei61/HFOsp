#!/usr/bin/env python3
"""Audit the locked exploratory Topic 5.2 C7 scoring outputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402
from scripts.score_topic5_latent_early_ictal_v0_2 import (  # noqa: E402
    AUTHORIZATION, EARLY, SCORER_REVISION,
)


def main() -> None:
    authorization = json.loads(AUTHORIZATION.read_text())
    unlock_path = EARLY / "TARGET_UNLOCK_RECORD.json"
    summary_path = EARLY / "EARLY_ICTAL_SUMMARY.json"
    seizure_path = EARLY / "EARLY_ICTAL_PER_SEIZURE.csv"
    patient_path = EARLY / "EARLY_ICTAL_PER_PATIENT.csv"
    identity_path = EARLY / "EARLY_ICTAL_IDENTITY.csv"
    required = [unlock_path, summary_path, seizure_path, patient_path, identity_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"C7 outputs missing: {missing}")

    unlock = json.loads(unlock_path.read_text())
    summary = json.loads(summary_path.read_text())
    seizure = pd.read_csv(seizure_path)
    patient = pd.read_csv(patient_path)
    identity = pd.read_csv(identity_path)
    checks = {
        "authorization_true": authorization.get("authorized") is True,
        "scorer_hash_still_authorized": authorization.get("scorer_sha256") == sha256_file(
            ROOT / "scripts/score_topic5_latent_early_ictal_v0_2.py"
        ),
        "target_free_hashes_unchanged": all(
            sha256_file(ROOT / relative) == digest
            for relative, digest in authorization.get("target_free_hashes", {}).items()
        ),
        "unlock_bound_to_authorization": unlock.get("authorization_sha256") == sha256_file(AUTHORIZATION),
        "unlock_records_target_access": unlock.get("target_values_read") is True,
        "no_training_or_selection_after_unlock": (
            unlock.get("training_or_model_selection_after_unlock") is False
            and summary.get("training_or_model_selection_after_unlock") is False
        ),
        "scorer_revision_locked": summary.get("scorer_revision") == SCORER_REVISION,
        "prediction_contract_locked": summary.get("prediction")
        == "FIXED_MEAN_PHASES_TAU1_TO3_MEDIAN_SEED_REAL_ARM_NO_ORACLE",
        "progress_orientation_frozen_before_target": authorization.get("progress_orientation_sensitivity")
        == "TARGET_FREE_FROZEN_LATERNESS_EQUALS_NEGATIVE_PREREGISTERED_EARLYNESS; PRIMARY_UNCHANGED",
        "progress_laterness_reported_as_sensitivity": summary.get("axes", {}).get("PROGRESS", {}).get(
            "laterness_orientation_sensitivity", {}
        ).get("role") == "TARGET_FREE_FROZEN_SIGN_SEMANTICS_SENSITIVITY_PRIMARY_UNCHANGED",
        "patient_denominator_17": patient["subject"].nunique() == 17,
        "seizure_denominator_167": seizure[["subject", "seizure_idx"]].drop_duplicates().shape[0] == 167,
        "two_axis_seizure_rows_334": len(seizure) == 334 and set(seizure["axis"]) == {"PROGRESS", "FIELD"},
        "two_axis_patient_rows_34": len(patient) == 34 and set(patient["axis"]) == {"PROGRESS", "FIELD"},
        "one_row_per_subject_axis": not patient.duplicated(["subject", "axis"]).any(),
        "identity_contains_no_phase_or_oracle_selection": not {
            "phase", "best_phase", "best_axis", "oracle_mode"
        }.intersection(identity.columns),
        "claim_boundary_exploratory": summary.get("claim_boundary")
        == "LOCKED_INTERNAL_EXPLORATORY; TARGET PREVIOUSLY VIEWED; NOT CONFIRMATORY",
    }
    payload = {
        "contract": "topic5_latent_early_ictal_C7_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "patients": int(patient["subject"].nunique()),
            "seizures": int(seizure[["subject", "seizure_idx"]].drop_duplicates().shape[0]),
            "seizure_axis_rows": int(len(seizure)),
            "patient_axis_rows": int(len(patient)),
            "identity_rows": int(len(identity)),
        },
        "output_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in required
        },
        "target_values_read": True,
    }
    atomic_write_json(EARLY / "EARLY_ICTAL_AUDIT.json", payload)
    if payload["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"C7 audit failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
