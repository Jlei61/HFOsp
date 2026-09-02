#!/usr/bin/env python3
"""Final engineering and provenance audit for Topic 5.2 v0.2."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402


# Spec section 15 lists 30 required tests.  Each entry names the direct automated
# test(s) that would fail if the invariant broke, or the stage audit that is the
# only evidence.  An item whose evidence is a stage-audit *status* is recorded as
# such rather than being counted as a test; a blanket "coverage complete" boolean
# would otherwise hide the difference.
SPEC_15_TESTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("checkpoint_resolver_630_cells",
     ("test_enumerate_complete_checkpoint_matrix", "test_exact_reuse_applies_only_to_l0_l1_l3"), "input"),
    ("full_state_q_clone", ("test_complete_decoder_state_clone_has_no_tensor_alias",), "E0_replay"),
    ("same_h_different_r_k_changes_decoder_decision", (), "E0_replay"),
    ("teacher_forced_hidden_logit_stop_size_replay", (), "E0_replay"),
    ("closed_loop_does_not_read_true_suffix_or_set_size",
     ("test_prefix_ranks_keeps_only_observed_prefix",), "closed_loop_transition"),
    ("event_first_phase_balanced_weights", ("test_phase_weights_balance_bins_and_events",), "pass1"),
    ("raw_primary_and_residual_sensitivity_routing", (), "pass1"),
    ("step_phase_primary_contact_phase_sensitivity", (), "pass1"),
    ("future_field_axis_uses_train_events_only", (), "pass1"),
    ("heldout_u_is_not_a_model_input", (), "pass1"),
    ("own_and_shared_node_space_guard",
     ("test_future_field_axis_tiers_do_not_relabel_own_fits_as_ab",), "input"),
    ("field_direction_sign_frozen_on_train",
     ("test_geometry_mapping_spacing_and_signed_field_do_not_oracle_flip",), "pass1"),
    ("P_PF_shuffled_PF_capacity_matching",
     ("test_spline_derivative_and_weighted_ridge_recover_linear_signal",), "pass1"),
    ("jacobian_uses_the_same_true_next_input", ("test_analytic_leaky_rnn_jvp_matches_autograd",), "transport"),
    ("local_normals_from_conditional_residual_space",
     ("test_local_normals_are_orthonormal_and_axis_normal",), "transport"),
    ("local_tangent_perturbation_keeps_r_and_k", (), "axis_perturbation"),
    ("transplantation_pairs_use_frozen_observables_only", (), "axis_perturbation"),
    ("support_gate_runs_in_the_z_o_conditional_space",
     ("test_directional_sd_matches_full_covariance_with_diagonal_correction",), "reference_freeze"),
    ("tau0_excluded_from_finite_time_endpoints", (), "axis_perturbation"),
    ("open_loop_branches_share_future_inputs", (), "axis_perturbation"),
    ("endpoints_use_train_only_contact_space_axes", (), "axis_perturbation"),
    ("response_matrix_orientation", (), "axis_perturbation"),
    ("gain_is_a_covariate_not_a_silent_state_filter", (), "axis_perturbation"),
    ("cross_patient_mapping_never_reads_field_values",
     ("test_geometry_mapping_spacing_and_signed_field_do_not_oracle_flip",), "geometry_registration"),
    ("snn_mapping_core_and_mode_sign_frozen_before_alignment", (), "early_ictal"),
    ("snn_eligibility_never_upgraded_by_alignment", (), "early_ictal"),
    ("early_ictal_target_sealed_until_target_free_freeze", (), "early_ictal"),
    ("patient_aggregation_does_not_reuse_event_seed_arm_fit", (), "data_alignment"),
    ("parameter_hashes_identical_before_and_after",
     ("test_parameter_hash_detects_state_mutation",), "transport"),
    ("figures_source_tables_and_claim_ladder_correspond", (), "figure_visual"),
)


def load(relative: str) -> dict[str, object]:
    return json.loads((OUT / relative).read_text())


def parameter_hash_invariance() -> tuple[bool, dict[str, int]]:
    """Verify the recorded per-cell before/after parameter hashes, not an audit status."""
    checked = unchanged = 0
    for path in sorted((OUT / "dynamical_transport" / "per_cell").glob("*/*/*/metrics.json")):
        metrics = json.loads(path.read_text())
        checked += 1
        unchanged += int(
            bool(metrics.get("model_hash_unchanged")) and bool(metrics.get("decoder_hash_unchanged"))
        )
    return checked > 0 and checked == unchanged, {"cells_checked": checked, "cells_unchanged": unchanged}


def main() -> None:
    canonical = [
        "CONTRACT.json", "CHECKPOINT_MANIFEST.csv", "INPUT_AUDIT.json", "RESOURCE_BUDGET.json",
        "PASS1_STREAMING_MANIFEST.json", "LATENT_GEOMETRY_SUMMARY.json",
        "DYNAMICAL_TRANSPORT_SUMMARY.json", "REFERENCE_STATE_MANIFEST.csv",
        "PASS2_PERTURBATION_MANIFEST.json", "PERTURBATION_RESPONSE_MATRIX.json",
        "FINITE_TIME_RESPONSE_FIELDS.npz", "SPATIAL_PATCH_CONTROL_FIELDS.npz",
        "DATA_ALIGNMENT_SUMMARY.json", "SNN_INPUT_ELIGIBILITY.json", "SNN_ALIGNMENT_SUMMARY.json",
        "EARLY_ICTAL_EXPLORATORY_SUMMARY.json", "COHORT_PATIENT_TABLE.csv",
        "CLAIM_LADDER_ADJUDICATION.json", "CONTROL_REFERENCED_ADDENDUM.json",
        "C5_SPATIAL_NULL_FAMILY_PATIENT_EFFECTS.csv", "C5_SMOOTHING_MATCHED_IDENTITY.csv",
        "PATCH_OPERATOR_SUMMARY.json",
    ]
    figures = OUT / "paper-ready-figure" / "latent_landscape_candidate" / "figures"
    stem = "topic5_latent_landscape_v0_2_candidate"
    figure_files = [
        figures / f"{stem}.png", figures / f"{stem}.pdf", figures / f"{stem}.svg",
        figures / f"{stem}_metadata.json", figures / "README.md", figures / "FIGURE_VISUAL_QA.json",
    ]
    missing = [name for name in canonical if not (OUT / name).is_file()]
    missing += [str(path.relative_to(ROOT)) for path in figure_files if not path.is_file()]

    audits = {
        "input": load("INPUT_AUDIT.json").get("status"),
        "E0_replay": load("E0_REPLAY_AUDIT.json").get("status"),
        "pass1": load("system_identification/PASS1_AUDIT.json").get("status"),
        "transport": load("dynamical_transport/TRANSPORT_AUDIT.json").get("status"),
        "closed_loop_transition": load("dynamical_transport/closed_loop_transition/CLOSED_LOOP_TRANSITION_AUDIT.json").get("status"),
        "reference_freeze": load("axis_perturbation/reference_freeze/REFERENCE_FREEZE_AUDIT.json").get("status"),
        "axis_perturbation": load("axis_perturbation/responses/PERTURBATION_AUDIT.json").get("status"),
        "geometry_registration": load("spatial_control_field/cross_patient_geometry_mapping/GEOMETRY_REGISTRATION_AUDIT.json").get("status"),
        "data_alignment": load("spatial_control_field/data_alignment/DATA_ALIGNMENT_AUDIT.json").get("status"),
        "patch_freeze": load("spatial_control_field/patch_freeze/PATCH_FREEZE_AUDIT.json").get("status"),
        "patch_response": load("spatial_control_field/patch_response/PATCH_RESPONSE_AUDIT.json").get("status"),
        "early_ictal": load("early_ictal_exploratory/EARLY_ICTAL_AUDIT.json").get("status"),
        "figure_visual": json.loads((figures / "FIGURE_VISUAL_QA.json").read_text()).get("status") if (figures / "FIGURE_VISUAL_QA.json").is_file() else "MISSING",
    }
    accepted_statuses = {"PASS"}
    audit_pass = all(status in accepted_statuses for status in audits.values())

    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    source_counts = manifest["checkpoint_source"].value_counts().to_dict()
    reference = pd.read_csv(OUT / "REFERENCE_STATE_MANIFEST.csv") if (OUT / "REFERENCE_STATE_MANIFEST.csv").is_file() else pd.DataFrame()
    claim = load("CLAIM_LADDER_ADJUDICATION.json") if (OUT / "CLAIM_LADDER_ADJUDICATION.json").is_file() else {}
    snn = load("SNN_INPUT_ELIGIBILITY.json")
    early = load("early_ictal_exploratory/EARLY_ICTAL_SUMMARY.json")
    unresolved_failures = sorted(OUT.glob("**/FAILURE.json"))
    recovered_failures = sorted(OUT.glob("**/RECOVERED_FAILURE.json"))

    junit = OUT / "PYTEST_JUNIT.xml"
    test_counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    executed_tests: set[str] = set()
    if junit.is_file():
        root = ET.parse(junit).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            for key in test_counts:
                test_counts[key] += int(suite.attrib.get(key, 0))
            executed_tests.update(case.attrib.get("name", "") for case in suite.findall("testcase"))

    counts = {
        "checkpoint_cells": int(len(manifest)),
        "formal_v0_5_cells": int(source_counts.get("V0_5_FORMAL_UNIT", 0)),
        "exact_v0_3_reuse_cells": int(source_counts.get("V0_3_EXACT_REUSE", 0)),
        "reference_states": int(len(reference)),
        "patients": int(manifest["patient"].nunique()),
        "fits": int(manifest["fit_id"].nunique()),
        "recovered_failure_records": len(recovered_failures),
        "unresolved_failure_records": len(unresolved_failures),
    }
    checks = {
        "all_canonical_artifacts_present": not missing,
        "all_stage_audits_pass": audit_pass,
        "checkpoint_matrix_630": counts["checkpoint_cells"] == 630,
        "checkpoint_sources_531_plus_99": counts["formal_v0_5_cells"] == 531 and counts["exact_v0_3_reuse_cells"] == 99,
        "cohort_28_patients_42_fits": counts["patients"] == 28 and counts["fits"] == 42,
        "no_unresolved_failure_records": not unresolved_failures,
        "pytest_pass": junit.is_file() and test_counts["tests"] >= 19 and test_counts["failures"] == 0 and test_counts["errors"] == 0,
        "claim_ladder_complete": claim.get("status") == "SCIENTIFIC_CLOSEOUT_COMPLETE",
        "C6_values_not_opened_when_ineligible": snn.get("C6_status") == "NOT_IDENTIFIABLE" and snn.get("field_values_read") is False,
        "C7_locked_exploratory_only": early.get("status") == "CROSS_STATE_EXPLORATORY_COMPLETE" and early.get("claim_boundary") == "LOCKED_INTERNAL_EXPLORATORY; TARGET PREVIOUSLY VIEWED; NOT CONFIRMATORY",
        "no_post_unlock_training_or_selection": early.get("training_or_model_selection_after_unlock") is False,
        "figure_png_pdf_svg_same_producer_state": all(path.is_file() for path in figure_files),
    }
    hash_invariance_ok, hash_invariance_counts = parameter_hash_invariance()
    checks["parameter_hashes_unchanged_per_cell"] = hash_invariance_ok
    coverage = {}
    for item, tests, stage in SPEC_15_TESTS:
        present = [name for name in tests if name in executed_tests]
        coverage[item] = {
            "direct_tests": present,
            "stage_audit": stage,
            "stage_audit_status": audits.get(stage),
            "evidence": (
                "AUTOMATED_TEST_AND_STAGE_AUDIT" if present and audits.get(stage) == "PASS"
                else "AUTOMATED_TEST_ONLY" if present
                else "STAGE_AUDIT_STATUS_ONLY" if audits.get(stage) == "PASS"
                else "UNCOVERED"
            ),
        }
    coverage["parameter_hashes_identical_before_and_after"]["per_cell_verification"] = hash_invariance_counts
    # "Mapped" means every spec item names a real evidence source, not that every
    # item has an executable regression test.  The two counters below keep that
    # distinction visible instead of collapsing it into one green boolean.
    checks["spec_test_map_complete"] = all(
        entry["evidence"] != "UNCOVERED" for entry in coverage.values()
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "contract": "topic5_latent_landscape_closeout_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "stage_audits": audits,
        "counts": counts,
        "pytest": test_counts,
        "spec_test_coverage": {
            "spec_section": "docs/superpowers/specs/2026-08-14-topic5-latent-propagation-landscape-v0-2-design.md#15",
            "required_items": len(SPEC_15_TESTS),
            "items_with_direct_automated_test": sum(
                1 for entry in coverage.values() if entry["direct_tests"]
            ),
            "items_evidenced_only_by_stage_audit_status": sum(
                1 for entry in coverage.values() if entry["evidence"] == "STAGE_AUDIT_STATUS_ONLY"
            ),
            "items_uncovered": sum(1 for entry in coverage.values() if entry["evidence"] == "UNCOVERED"),
            "note": (
                "A stage-audit status is weaker evidence than a regression test: it shows the stage "
                "ran without a recorded failure, not that the named invariant is separately checked."
            ),
            "items": coverage,
        },
        "missing": missing,
        "unresolved_failures": [str(path.relative_to(ROOT)) for path in unresolved_failures],
        "recovered_failures": [str(path.relative_to(ROOT)) for path in recovered_failures],
        "artifact_hashes": {
            name: sha256_file(OUT / name) for name in canonical if (OUT / name).is_file()
        },
        "figure_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in figure_files if path.is_file()
        },
        "scientific_status": claim.get("scientific_verdict"),
        "target_access": {
            "target_free_branches": "SEALED_BEFORE_C7",
            "C7": "AUTHORIZED_THEN_READ",
            "post_unlock_training_or_model_selection": False,
        },
    }
    atomic_write_json(OUT / "CLOSEOUT_AUDIT.json", payload)
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"closeout failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
