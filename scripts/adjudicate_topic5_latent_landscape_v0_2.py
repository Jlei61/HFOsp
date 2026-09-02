#!/usr/bin/env python3
"""Build the Topic 5.2 C1--C7 claim ladder and canonical result indexes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402


SYSTEM = OUT / "system_identification"
TRANSPORT = OUT / "dynamical_transport"
RESPONSES = OUT / "axis_perturbation" / "responses"
REFERENCE = OUT / "axis_perturbation" / "reference_freeze"
SPATIAL = OUT / "spatial_control_field"
DATA = SPATIAL / "data_alignment"
PATCH = SPATIAL / "patch_response"
EARLY = OUT / "early_ictal_exploratory"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def canonical_copy(source: Path, destination: Path) -> None:
    """Create an atomic canonical copy while keeping producer outputs immutable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def prefixed_generic(path: Path, prefix: str) -> pd.DataFrame:
    values = pd.read_csv(path)
    values = values[values["tier"].eq("generic_all_identifiable")].copy()
    values = values.drop(columns=["tier"])
    return values.rename(columns={column: f"{prefix}{column}" for column in values.columns if column != "patient"})


def build_patient_table() -> pd.DataFrame:
    tables = [
        prefixed_generic(SYSTEM / "C1_PATIENT_EFFECTS.csv", "c1_"),
        prefixed_generic(TRANSPORT / "C2_PATIENT_EFFECTS.csv", "c2_"),
        prefixed_generic(RESPONSES / "C3_PATIENT_EFFECTS.csv", "c3_"),
        prefixed_generic(DATA / "C5_PATIENT_EFFECTS.csv", "c5_"),
    ]
    result = tables[0]
    for table in tables[1:]:
        result = result.merge(table, on="patient", how="outer", validate="one_to_one")
    sign = pd.read_csv(RESPONSES / "PROGRESS_SIGN_SEMANTICS_PATIENT_EFFECTS.csv")
    sign = sign[sign["tier"].eq("generic_all_identifiable")].drop(
        columns=["tier", "target_values_read", "analysis_role"]
    )
    sign = sign.rename(columns={column: f"sign_{column}" for column in sign.columns if column != "patient"})
    result = result.merge(sign, on="patient", how="left", validate="one_to_one")

    patch_path = PATCH / "PATCH_TOPOLOGY_CONSISTENCY.csv"
    if patch_path.is_file():
        patch = pd.read_csv(patch_path).groupby(["patient", "axis"], as_index=False)[
            ["real_arm_pair_cosine", "topology_margin", "median_sign_agreement_fraction"]
        ].median()
        patch = patch.pivot(index="patient", columns="axis").reset_index()
        patch.columns = [
            "patient" if column[0] == "patient" else f"patch_{column[1].lower()}_{column[0]}"
            for column in patch.columns
        ]
        result = result.merge(patch, on="patient", how="left", validate="one_to_one")

    early_path = EARLY / "EARLY_ICTAL_PER_PATIENT.csv"
    if early_path.is_file():
        early = pd.read_csv(early_path).pivot(index="subject", columns="axis").reset_index()
        early.columns = [
            "patient" if column[0] == "subject" else f"c7_{column[1].lower()}_{column[0]}"
            for column in early.columns
        ]
        result = result.merge(early, on="patient", how="left", validate="one_to_one")
    result["C6_snn_status"] = "NOT_IDENTIFIABLE"
    return result.sort_values("patient").reset_index(drop=True)


def main() -> None:
    c1 = load(SYSTEM / "LATENT_GEOMETRY_SUMMARY.json")
    c2 = load(TRANSPORT / "DYNAMICAL_TRANSPORT_SUMMARY.json")
    c3c4 = load(RESPONSES / "PERTURBATION_CLAIM_SUMMARY.json")
    c5 = load(DATA / "DATA_ALIGNMENT_SUMMARY.json")
    patch = load(PATCH / "SPATIAL_PATCH_CONTROL_SUMMARY.json")
    c6 = load(SPATIAL / "SNN_ALIGNMENT_SUMMARY.json")
    c7 = load(EARLY / "EARLY_ICTAL_SUMMARY.json")
    sign = load(RESPONSES / "PROGRESS_SIGN_SEMANTICS_AUDIT.json")
    addendum = load(OUT / "CONTROL_REFERENCED_ADDENDUM.json")
    operator = load(SPATIAL / "patch_operator" / "PATCH_OPERATOR_SUMMARY.json")
    operator_convergence = operator["topology_convergence"]["endpoints"]
    operator_link = operator["data_link"]["endpoints"]
    convergence_supported = all(
        operator_convergence[key]["status"] == "SUPPORTED"
        for key in ("reliability_corrected_margin", "leave_one_topology_out_margin")
    )
    # The two legs of the data link are adjudicated separately: does the consensus
    # operator match held-out propagation beyond coarse spatial structure, and is that
    # match specific to the patient once the identity null's smoothing is matched?
    link_alignment_supported = operator_link["within_shaft_margin"].get("status") == "SUPPORTED"
    link_identity_supported = (
        operator_link["smoothing_matched_identity_margin"].get("status") == "SUPPORTED"
    )
    link_supported = link_alignment_supported and link_identity_supported

    canonical_sources = {
        "LATENT_GEOMETRY_SUMMARY.json": SYSTEM / "LATENT_GEOMETRY_SUMMARY.json",
        "DYNAMICAL_TRANSPORT_SUMMARY.json": TRANSPORT / "DYNAMICAL_TRANSPORT_SUMMARY.json",
        "REFERENCE_STATE_MANIFEST.csv": REFERENCE / "REFERENCE_STATE_MANIFEST.csv",
        "PERTURBATION_RESPONSE_MATRIX.json": DATA / "PERTURBATION_RESPONSE_MATRIX.json",
        "FINITE_TIME_RESPONSE_FIELDS.npz": DATA / "FINITE_TIME_RESPONSE_FIELDS.npz",
        "SPATIAL_PATCH_CONTROL_FIELDS.npz": PATCH / "SPATIAL_PATCH_CONTROL_FIELDS.npz",
        "DATA_ALIGNMENT_SUMMARY.json": DATA / "DATA_ALIGNMENT_SUMMARY.json",
        "SNN_ALIGNMENT_SUMMARY.json": SPATIAL / "SNN_ALIGNMENT_SUMMARY.json",
        "EARLY_ICTAL_EXPLORATORY_SUMMARY.json": EARLY / "EARLY_ICTAL_SUMMARY.json",
        "PATCH_OPERATOR_SUMMARY.json": SPATIAL / "patch_operator" / "PATCH_OPERATOR_SUMMARY.json",
    }
    for name, source in canonical_sources.items():
        canonical_copy(source, OUT / name)

    pass1_sources = [
        OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv",
        OUT / "PASS1_EVENT_SAMPLE_AUDIT.json",
        SYSTEM / "PASS1_AUDIT.json",
        SYSTEM / "PASS1_CELL_GEOMETRY.csv",
        SYSTEM / "PASS1_FUTURE_FIELD_EMERGENCE.csv",
    ]
    atomic_write_json(OUT / "PASS1_STREAMING_MANIFEST.json", {
        "contract": "topic5_pass1_streaming_manifest_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LATENT_GEOMETRY_COMPLETE",
        "artifacts": {str(path.relative_to(ROOT)): sha256_file(path) for path in pass1_sources},
        "target_values_read": False,
    })
    pass2_sources = [
        REFERENCE / "REFERENCE_FREEZE_AUDIT.json",
        REFERENCE / "REFERENCE_FREEZE_SEAL.json",
        RESPONSES / "PERTURBATION_AUDIT.json",
        RESPONSES / "PERTURBATION_CLAIM_SUMMARY.json",
        PATCH / "PATCH_RESPONSE_AUDIT.json",
        PATCH / "SPATIAL_PATCH_CONTROL_SUMMARY.json",
    ]
    atomic_write_json(OUT / "PASS2_PERTURBATION_MANIFEST.json", {
        "contract": "topic5_pass2_perturbation_manifest_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PERTURBATION_RESPONSE_COMPLETE",
        "artifacts": {str(path.relative_to(ROOT)): sha256_file(path) for path in pass2_sources},
        "target_values_read": False,
    })

    patients = build_patient_table()
    atomic_write_csv(OUT / "COHORT_PATIENT_TABLE.csv", patients)
    generic_c1 = c1["tiers"]["generic_all_identifiable"]
    generic_c2 = c2["tiers"]["generic_all_identifiable"]
    generic_c3 = c3c4["C3_tiers"]["generic_all_identifiable"]
    generic_c5 = c5["tiers"]["generic_all_identifiable"]
    claims = {
        "C1_geometry": {
            "status": generic_c1["C1_status"],
            "supported_subclaims": [
                name for name, value in generic_c1["endpoints"].items()
                if value.get("status") == "SUPPORTED"
            ],
            "limiting_endpoint": "early_emergence_real_minus_C_suffix",
        },
        "C2_dynamics": {
            "status": generic_c2["C2_status"],
            "supported_subclaims": [
                name for name, value in generic_c2["endpoints"].items()
                if value.get("status") == "SUPPORTED"
            ],
            "supported_subclaim_reference": "POSITIVITY_VS_ZERO_ONLY_NOT_VS_ORDER_SHUFFLED_CONTROL",
            "order_specificity": generic_c2["order_specificity"],
            "vs_order_shuffled_control": {
                name: {
                    "median": value["vs_order_shuffled_control"]["median"],
                    "positive": value["vs_order_shuffled_control"]["positive"],
                    "n_patients": value["vs_order_shuffled_control"]["n_patients"],
                    "p_holm_control_family": value["vs_order_shuffled_control"]["p_holm_control_family"],
                    "status": value["vs_order_shuffled_control"]["status_vs_order_shuffled_control"],
                }
                for name, value in generic_c2["endpoints"].items()
                if "vs_order_shuffled_control" in value
            },
            "limiting_endpoint": "event_to_PF_manifold_convergence",
        },
        "C3_axis_specific_perturbation": {
            "status": generic_c3["C3_joint_status"],
            "progress_status": generic_c3["C3_progress_status"],
            "field_status": generic_c3["C3_field_status"],
            "empirical_chord_role": "POSITIVE_SECONDARY_MODEL_INTERNAL_STATE_TRANSPLANTATION",
            "progress_sign_semantics": sign["status"],
            "primary_changed_by_sign_audit": False,
        },
        "C4_topology_convergence": {
            "status": c3c4["C4_topology"]["C4_status"],
            "median_real_arm_pair_cosine": c3c4["C4_topology"]["patient_median_real_arm_pair_cosine"],
            "median_real_arm_to_order_shuffled_cosine": c3c4["C4_topology"][
                "patient_median_real_arm_to_C_suffix_cosine"
            ],
            "per_axis": addendum["C4_topology_per_axis"],
            "axis_boundary": (
                "The future-field response axis carries the claim; the progress axis margin has a "
                "median CI that includes zero and its order-shuffled arm already reaches most of "
                "the real-arm similarity."
            ),
        },
        "C5_patient_specific_data_alignment": {
            "status": generic_c5["C5_status"],
            "n_patients": generic_c5["n_patients"],
            "registration_boundary": c5["claim_boundary"],
            "progress_orientation_issue": "PREREGISTERED_EARLYNESS_SIGN_UNSUPPORTED",
            "laterness_posthoc_target_free": sign["C5_progress_orientation_sensitivity"]["tiers"]["generic_all_identifiable"],
            "spatial_null_families": addendum["C5_spatial_null_families"]["tiers"]["generic_all_identifiable"],
            "identity_smoothing_match": addendum["C5_identity_smoothing_match"]["tiers"]["generic_all_identifiable"],
            "primary_changed_by_sign_audit": False,
        },
        "C6_cross_model_convergence": {
            "status": c6["C6_status"],
            "source_status": c6["status"],
            "field_values_read": c6["field_values_read"],
        },
        "C7_cross_state": {
            "status": "EXPLORATORY_COMPLETE",
            "source_status": c7["status"],
            "axes": c7["axes"],
            "confirmatory": False,
        },
        # C8 repeats the C4 question with a perturbation that uses no fitted hidden
        # axis, so the shared future-field label can no longer explain convergence.
        "C8_axis_free_operator": {
            "status": "SUPPORTED" if convergence_supported else "UNSUPPORTED",
            "perturbation": "GEOMETRY_ONLY_GAUSSIAN_TISSUE_PATCH",
            "n_patients": operator["topology_convergence"]["n_patients"],
            "median_real_pair_similarity": operator["topology_convergence"]["median_real_pair_similarity"],
            "median_real_to_shuffled_similarity": operator["topology_convergence"]["median_real_to_shuffled_similarity"],
            "endpoints": operator_convergence,
            "phase_invariance": operator["phase_invariance"],
            "residual_axis_dependence": operator["operator_definition"]["residual_axis_dependence"],
        },
        "C9_operator_to_data_link": {
            "status": "SUPPORTED" if link_supported else "PARTIAL",
            "alignment_beyond_coarse_geometry": "SUPPORTED" if link_alignment_supported else "UNSUPPORTED",
            "patient_specificity_after_smoothing_match": "SUPPORTED" if link_identity_supported else "UNSUPPORTED",
            "n_patients": operator["data_link"]["n_patients"],
            "transition_operator": operator["data_link"]["transition_operator"],
            "endpoints": operator_link,
            "claim_boundary": (
                "A match to held-out interictal transition statistics, not to anatomical "
                "connectivity."
            ),
        },
    }
    payload = {
        "contract": "topic5_latent_landscape_claim_ladder_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SCIENTIFIC_CLOSEOUT_COMPLETE",
        "engineering_status": "PASS",
        "scientific_verdict": "TOPOLOGY_CONVERGENT_COMPUTATION_WITH_UNRESOLVED_PROGRESS_SIGN_NOT_CONFIRMATORY_PATIENT_SPECIFIC_LANDSCAPE",
        "claims": claims,
        "spatial_patch": {
            "status": patch["status"],
            "claim_boundary": patch["claim_boundary"],
            "axis_summaries": patch["axis_summaries"],
        },
        "strongest_safe_claim": (
            "Frozen recurrent models encode event phase and future-field information, and distinct real-order "
            "topologies converge on similar finite-time future-field responses. The complete patient-specific "
            "state-control landscape is not confirmatorily identified because true order did not yield earlier "
            "future-field commitment, trajectories did not approach the conditional manifold, and the preregistered "
            "axis-selective double dissociation failed. The preregistered earlyness-signed progress field failed C5, "
            "whereas a target-free post-hoc laterness reorientation produced positive spatial and identity margins; "
            "that deterministic sign sensitivity requires a new frozen confirmation and does not rescue C5."
        ),
        "allowed_claims": [
            "Raw frozen hidden states contain held-out progress and continuous future-field information beyond matched baselines.",
            "Teacher-forced local dynamics transport fitted progress and field directions and keep the transverse gain below one, but the order-shuffled control arm reproduces every one of those quantities, so they are architecture-level rather than order-specific, and no attracting propagation channel is established.",
            "Matched empirical high-u state chords shift future-field output more than small-u control chords as a secondary model-internal result.",
            "Distinct real-order recurrent topologies converge on similar finite-time future-field responses relative to the order-shuffled arm; on the progress response axis the same margin has a median CI that includes zero.",
            "A target-free sign-semantics audit identifies a preregistered progress-output orientation mismatch and a positive laterness sensitivity that is hypothesis-generating only, and that sensitivity is roughly halved by shaft- and distance-preserving spatial nulls.",
            "C7 is a locked internal exploratory analysis and cannot provide confirmation.",
        ],
        "forbidden_claims": [
            "The RNN identifies a patient-specific biological attractor or propagation channel.",
            "Progress and future-field axes are causally dissociated under the preregistered primary.",
            "The functional response field is confirmatorily validated as patient-specific by held-out interictal data.",
            "RNN and SNN provide cohort-level cross-model convergence.",
            "Interictal fields predict or confirm early-ictal recruitment.",
            "Any post-hoc progress sign reorientation rescues C3 or C5.",
            "Tangent transport, transverse contraction, or progress-axis topology convergence is order-specific or learned rather than architectural.",
            "The synchronized all-contact laterness margin is the spatially controlled effect size.",
        ],
        "E3_decision": {
            "state": [
                "GENERIC_PROGRESS_GEOMETRY_CLOSED",
                "FUTURE_FIELD_FUNCTIONAL_CONVERGENCE_SUPPORTED" if convergence_supported
                else "FUTURE_FIELD_FUNCTIONAL_CONVERGENCE_UNSUPPORTED",
                "DIRECT_OPERATOR_TO_DATA_LINK_SUPPORTED" if link_supported
                else "DIRECT_OPERATOR_TO_DATA_LINK_ALIGNMENT_SUPPORTED_PATIENT_SPECIFICITY_PENDING"
                if link_alignment_supported
                else "DIRECT_OPERATOR_TO_DATA_LINK_PENDING",
            ],
            "gate": (
                "E3 is no longer gated on re-confirming the ordinal-phase output orientation. "
                "It is gated on whether the topology-consensus operator matches held-out patient "
                "propagation under spatial and cross-patient nulls."
            ),
            "allowed_next_role": (
                "low-parameter compression of the consensus operator into a smooth susceptibility "
                "field, once the operator-to-data link closes"
            ),
        },
        "canonical_artifacts": {
            name: {"source": str(source.relative_to(ROOT)), "sha256": sha256_file(OUT / name)}
            for name, source in canonical_sources.items()
        },
        "control_referenced_addendum": {
            "source": str((OUT / "CONTROL_REFERENCED_ADDENDUM.json").relative_to(ROOT)),
            "sha256": sha256_file(OUT / "CONTROL_REFERENCED_ADDENDUM.json"),
            "role": addendum["role"],
        },
        "target_access": {
            "C1_to_C6_target_values_read": False,
            "C7_target_values_read_after_authorized_freeze": True,
            "training_or_model_selection_after_unlock": False,
        },
    }
    atomic_write_json(OUT / "CLAIM_LADDER_ADJUDICATION.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
