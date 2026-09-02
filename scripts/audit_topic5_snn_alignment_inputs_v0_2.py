#!/usr/bin/env python3
"""Eligibility-only audit for SNN cross-model convergence; never reads SNN fields."""
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
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402
from scripts.run_topic5_spatial_patch_response_v0_2 import RESPONSE  # noqa: E402


CANONICAL = Path("/home/honglab/leijiaxin/HFOsp")
REGISTRY = CANONICAL / "docs/paper_figure_registry.md"
D52 = CANONICAL / "results/topic4_sef_hfo/data_driven_core_field_rev10_d/spatial_ou_accessibility_d5_2_confirmation"
D63 = CANONICAL / "results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_kmeans_d6_3_fresh_replication"
MIN_LONG_RUN_MS = 20000.0


def candidate(root: Path, asset_id: str) -> dict[str, object]:
    manifest_path = root / "candidate_manifest.json"; verdict_path = root / "confirmation_verdict.json"
    manifest = json.loads(manifest_path.read_text()); verdict = json.loads(verdict_path.read_text())
    config_ref = manifest.get("config", {}); config_path = CANONICAL / str(config_ref.get("path", ""))
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    duration = float(
        manifest.get("fixed_contract", {}).get("duration_ms", config.get("search", {}).get("simulation", {}).get("duration_ms", float("nan")))
    )
    if asset_id == "D5_2":
        replication = "DEVELOPMENT_CONFIRMATION_ONLY"
        natural = bool(verdict.get("kmeans_reaches_patient_matched_q05", False))
        n_networks = len(manifest.get("fixed_contract", {}).get("confirmation_network_seeds", []))
        late_runaway = "NOT_ASSESSED_BEYOND_8S"
    else:
        replication = "REPLICATED" if bool(verdict.get("replication_pass", False)) else "NOT_REPLICATED"
        # The producer uses string-valued adjudications such as
        # ``DIAGNOSTIC_ONLY``.  Python truthiness would incorrectly turn any
        # non-empty rejection/status string into a scientific PASS.
        fig4_acceptance = verdict.get("fig4_acceptance", False)
        natural = fig4_acceptance is True or str(fig4_acceptance).upper() in {
            "PASS", "ACCEPTED", "CONFIRMED",
        }
        n_networks = int(verdict.get("candidate_metrics", {}).get("n_networks", 0))
        late_runaway = (
            "NO_RUNAWAY_WITHIN_WINDOW" if int(verdict.get("candidate_metrics", {}).get("n_runaway_networks", -1)) == 0
            else "RUNAWAY_PRESENT_OR_UNKNOWN"
        )
    runtime_mode = config.get("execution", {}).get("runtime_mode")
    reasons = []
    if runtime_mode is None: reasons.append("RUNTIME_MODE_NOT_EXPLICIT")
    if not duration >= MIN_LONG_RUN_MS: reasons.append("SIMULATION_SHORTER_THAN_20S_LONG_RUN_CONTRACT")
    if replication != "REPLICATED": reasons.append("FRESH_NETWORK_REPLICATION_NOT_CLOSED")
    if not natural: reasons.append("NATURAL_MODE_PATIENT_BENCHMARK_NOT_PASSED")
    reasons += ["SINGLE_PATIENT_E1146", "NO_LOCKED_RNN_TO_SNN_FIELD_MAPPING_OR_CORE_DEFINITION"]
    return {
        "asset_id": asset_id, "patient": "epilepsiae_1146", "geometry": "E1146_TOPIC4_MODEL_GRID",
        "producer_manifest": str(manifest_path), "producer_manifest_sha256": sha256_file(manifest_path),
        "verdict_path": str(verdict_path), "verdict_sha256": sha256_file(verdict_path),
        "config_path": str(config_path), "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "engine_baseline_id": manifest.get("selection_freeze", {}).get("primary_candidate_id", manifest.get("selection_freeze", {}).get("selected_nonzero_candidate_id")),
        "runtime_mode": runtime_mode, "simulation_duration_ms": duration,
        "late_runaway_status": late_runaway, "network_replication_status": replication,
        "natural_mode_validation": natural, "n_networks": n_networks,
        "field_definitions": "DEVELOPMENT_NODE_FIELD_AND_EVENT_READOUT; NOT_LOCKED_FOR_TOPIC5_C6",
        "rnn_snn_mapping": "MISSING_LOCKED_GEOMETRY_AND_SIGN_CONTRACT",
        "source_status": "DIAGNOSTIC_ONLY", "ineligibility_reasons": reasons,
        "field_values_read": False,
    }


def main() -> None:
    required = {
        "REFERENCE_STATE_MANIFEST.csv": REFERENCE / "REFERENCE_STATE_MANIFEST.csv",
        "PERTURBATION_RESPONSE_MATRIX.json": SPATIAL / "data_alignment/PERTURBATION_RESPONSE_MATRIX.json",
        "FINITE_TIME_RESPONSE_FIELDS.npz": SPATIAL / "data_alignment/FINITE_TIME_RESPONSE_FIELDS.npz",
        "SPATIAL_PATCH_CONTROL_FIELDS.npz": RESPONSE / "SPATIAL_PATCH_CONTROL_FIELDS.npz",
        "DATA_ALIGNMENT_SUMMARY.json": SPATIAL / "data_alignment/DATA_ALIGNMENT_SUMMARY.json",
        "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv": MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing: raise RuntimeError(f"SNN value-access prerequisites missing: {missing}")
    sources = [candidate(D52, "D5_2"), candidate(D63, "D6_3")]
    status = "SNN_ALIGNMENT_NOT_IDENTIFIABLE"
    payload = {
        "contract": "topic5_snn_input_eligibility_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "C6_status": "NOT_IDENTIFIABLE",
        "registry_path": str(REGISTRY), "registry_sha256": sha256_file(REGISTRY),
        "rnn_prerequisite_hashes": {name: sha256_file(path) for name, path in required.items()},
        "candidates": sources,
        "adjudication": (
            "Both available data-driven SNN sources remain diagnostic: one is an 8-s development confirmation; "
            "the 16-s fresh-network arm failed replication and neither has explicit runtime_mode, a >=20-s late-runaway audit, "
            "an adequate patient denominator, or a pre-locked Topic5 RNN-SNN field mapping."
        ),
        "value_access_decision": "NOT_OPENED_BECAUSE_NO_SOURCE_IS_CASE_SERIES_OR_COHORT_ELIGIBLE",
        "field_values_read": False, "target_values_read": False,
    }
    atomic_write_json(OUT / "SNN_INPUT_ELIGIBILITY.json", payload)
    atomic_write_json(SPATIAL / "SNN_ALIGNMENT_SUMMARY.json", {
        "contract": "topic5_snn_alignment_C6_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "C6_status": "NOT_IDENTIFIABLE", "n_patients": 0,
        "reason": payload["adjudication"], "field_values_read": False, "target_values_read": False,
    })
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
