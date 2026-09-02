#!/usr/bin/env python3
"""Freeze all target-free Topic 5.2 artifacts before C7 target access."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import MAPPING, SPATIAL  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import REFERENCE  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.run_topic5_spatial_patch_response_v0_2 import RESPONSE  # noqa: E402


def main() -> None:
    paths = [
        REFERENCE / "REFERENCE_STATE_MANIFEST.csv",
        SPATIAL / "data_alignment/PERTURBATION_RESPONSE_MATRIX.json",
        SPATIAL / "data_alignment/FINITE_TIME_RESPONSE_FIELDS.npz",
        SPATIAL / "data_alignment/DATA_ALIGNMENT_SUMMARY.json",
        SPATIAL / "data_alignment/DATA_ALIGNMENT_AUDIT.json",
        OUT / "axis_perturbation/responses/PROGRESS_SIGN_SEMANTICS_AUDIT.json",
        MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv",
        MAPPING / "GEOMETRY_REGISTRATION_AUDIT.json",
        RESPONSE / "SPATIAL_PATCH_CONTROL_FIELDS.npz",
        RESPONSE / "SPATIAL_PATCH_CONTROL_SUMMARY.json",
        RESPONSE / "SPATIAL_PATCH_FREEZE_SEAL.json",
        OUT / "SNN_INPUT_ELIGIBILITY.json",
        SPATIAL / "SNN_ALIGNMENT_SUMMARY.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise RuntimeError(f"target-free C7 prerequisites missing: {missing}")
    if json.loads((SPATIAL / "data_alignment/DATA_ALIGNMENT_AUDIT.json").read_text()).get("status") != "PASS":
        raise RuntimeError("C5 audit did not pass")
    if json.loads((RESPONSE / "SPATIAL_PATCH_CONTROL_SUMMARY.json").read_text()).get("status") != "SPATIAL_CONTROL_FIELD_COMPLETE":
        raise RuntimeError("patch response not complete")
    scorer = ROOT / "scripts/score_topic5_latent_early_ictal_v0_2.py"
    target_manifest = PARENT / "early_ictal/EARLY_ICTAL_TARGET_MANIFEST.csv"
    null_manifest = PARENT / "NULL_INDEX_MAP_MANIFEST.csv"
    routing = PARENT / "EARLY_ICTAL_ROUTING_METADATA.csv"
    payload = {
        "contract": "topic5_latent_early_ictal_unlock_authorization_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "authorized": True,
        "target_free_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "scorer_sha256": sha256_file(scorer), "target_manifest_sha256": sha256_file(target_manifest),
        "null_manifest_sha256": sha256_file(null_manifest), "routing_sha256": sha256_file(routing),
        "prediction_contract": "FIXED_MEAN_PHASES_TAU1_TO3_MEDIAN_SEED_REAL_ARM_NO_BEST_AXIS_OR_PHASE",
        "progress_orientation_sensitivity": "TARGET_FREE_FROZEN_LATERNESS_EQUALS_NEGATIVE_PREREGISTERED_EARLYNESS; PRIMARY_UNCHANGED",
        "spatial_null_contract": "EXISTING_FROZEN_ALL_CONTACT_PRIMARY_AND_GEOMETRY_SENSITIVITIES",
        "project_history_target_previously_viewed": True,
        "training_or_model_selection_after_unlock_allowed": False, "target_values_read_during_authorization": False,
    }
    atomic_write_json(OUT / "EARLY_ICTAL_UNLOCK_AUTHORIZATION.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
