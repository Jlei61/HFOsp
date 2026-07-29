"""Fail-closed Phase-C adjudication and route separation."""
from copy import deepcopy

import pytest

from src.topic4_zm_phasec_verdict import (
    FINE_VERDICTS,
    NEXT_ROUTES,
    adjudicate_phasec,
    artifact_sha256,
    build_provenance,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _inputs(
    *,
    c0_verdict="replicated_balanced_asynchronous_tonic_candidate",
    primary_verdict="no_local_maturation_window",
    shell_verdict="no_local_maturation_window",
    c0_status="complete",
    primary_status="complete",
    shell_status="complete",
    modal_status="complete",
):
    c0 = {"verdict": c0_verdict, "eligible_seeds": [1, 3, 4]}
    primary = {"verdict": primary_verdict, "complete_seeds": [1, 3, 4]}
    shell = {"verdict": shell_verdict, "complete_seeds": [1, 3, 4]}
    modal = {"verdict": "seed_specific_validated_operators", "status": modal_status}
    coverage = {
        "c0": {"status": c0_status},
        "c1_primary": {"status": primary_status},
        "c1_shell": {"status": shell_status},
        "modal": {"status": modal_status},
    }
    artifacts = {
        "c0": c0,
        "c1_primary": primary,
        "c1_shell": shell,
        "modal": modal,
        "coverage": coverage,
    }
    provenance = build_provenance(
        artifacts,
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
    )
    return {**artifacts, "provenance": provenance}


def _run(**kwargs):
    return adjudicate_phasec(**_inputs(**kwargs))


def _assert_lifecycle_locked(out):
    assert out["entry"] == "not_tested"
    assert out["offset"] == "not_tested"
    assert out["recovery_lifecycle"] == "not_established"
    assert out["phase_c2_authorized"] is False
    assert out["actuator_authorized"] is False
    assert out["observation_match"] == "blocked"
    assert out["observation_layer"] == "blocked_reference_artifacts"
    assert out["layers"]["entry"] == "not_tested"
    assert out["layers"]["offset"] == "not_tested"
    assert out["layers"]["recovery_lifecycle"] == "not_established"


def test_vocabulary_is_closed_and_top_level_uses_fine_verdict():
    assert len(FINE_VERDICTS) == len(set(FINE_VERDICTS))
    assert len(NEXT_ROUTES) == len(set(NEXT_ROUTES))
    out = _run()
    assert out["verdict"] == out["fine_verdict"]
    assert out["verdict"] in FINE_VERDICTS
    assert out["next_route"] in NEXT_ROUTES
    _assert_lifecycle_locked(out)


@pytest.mark.parametrize("mutator,reason_fragment", [
    (lambda x: x.pop("provenance"), "provenance"),
    (lambda x: x["provenance"]["artifact_sha256"].pop("c0"), "provenance"),
    (lambda x: x["c0"].update({"verdict": "mutated_after_hash"}), "provenance"),
    (lambda x: x["provenance"].update({"manifest_sha256": "short"}), "provenance"),
    (lambda x: x["coverage"].pop("modal"), "provenance"),
])
def test_missing_or_sha_mismatched_inputs_fail_closed(mutator, reason_fragment):
    inputs = _inputs()
    mutator(inputs)
    if "provenance" not in inputs:
        inputs["provenance"] = {}
    out = adjudicate_phasec(**inputs)
    assert out["verdict"] == "no_evidence"
    assert out["next_route"] == "no_evidence"
    assert reason_fragment in out["reason"]
    _assert_lifecycle_locked(out)


def test_artifact_hash_rejects_nonfinite_or_nonjson_payload():
    with pytest.raises(TypeError):
        artifact_sha256({"x": float("nan")})
    with pytest.raises(TypeError):
        artifact_sha256({"x": object()})


@pytest.mark.parametrize(
    "c0_status,expected",
    [
        ("blocked_observables", "C0_blocked_observables"),
        ("insufficient", "C0_insufficient_coverage"),
        ("resolution_sensitive", "resolution_sensitive_identity"),
    ],
)
def test_c0_coverage_gates_are_explicit(c0_status, expected):
    out = _run(c0_status=c0_status)
    assert out["verdict"] == expected
    assert out["next_route"] == (
        "mixed_identity_requires_refinement"
        if expected == "resolution_sensitive_identity"
        else "no_evidence"
    )
    _assert_lifecycle_locked(out)


def test_c0_opposite_seed_identities_are_not_majority_voted():
    inputs = _inputs(c0_verdict="heterogeneous_or_unresolved")
    inputs["c0"]["seed_classes"] = {
        "1": "balanced_asynchronous_tonic_candidate",
        "3": "balanced_asynchronous_tonic_candidate",
        "4": "refractory_limited_plateau",
    }
    inputs["provenance"] = build_provenance(
        {k: inputs[k] for k in ("c0", "c1_primary", "c1_shell", "modal", "coverage")},
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
    )
    out = adjudicate_phasec(**inputs)
    assert out["verdict"] == "seed_heterogeneous_identity"
    assert out["next_route"] == "mixed_identity_requires_refinement"


def test_mixed_c0_identity_requires_refinement_not_a_default_route():
    out = _run(c0_verdict="heterogeneous_or_unresolved")
    assert out["verdict"] == "mixed_or_indeterminate_tonic_branch"
    assert out["next_route"] == "mixed_identity_requires_refinement"


def test_registered_primary_window_is_not_hidden_by_mixed_c0_identity():
    out = _run(
        c0_verdict="heterogeneous_or_unresolved",
        primary_verdict="local_maturation_window",
        shell_verdict="not_tested",
        shell_status="not_tested",
    )
    assert out["verdict"] == "maturation_window_at_primary_convex_states"
    assert out["next_route"] == (
        "local_maturation_window_found_slow_path_audit_required"
    )
    assert out["layers"]["source_identity"]["verdict"] == (
        "heterogeneous_or_unresolved"
    )
    assert out["reason"] == (
        "contiguous_maturation_window_at_primary_convex_states_"
        "with_unresolved_source_identity"
    )
    _assert_lifecycle_locked(out)


def test_primary_window_is_the_only_slow_path_authorization():
    out = _run(primary_verdict="local_maturation_window")
    assert out["verdict"] == "maturation_window_at_primary_convex_states"
    assert out["next_route"] == "local_maturation_window_found_slow_path_audit_required"
    assert out["reason"] == (
        "contiguous_maturation_window_at_primary_convex_states"
    )
    _assert_lifecycle_locked(out)


def test_shell_positive_never_promotes_primary_reachability():
    out = _run(
        primary_verdict="no_local_maturation_window",
        shell_verdict="local_maturation_window",
    )
    assert out["verdict"] == "maturation_candidate_in_secondary_shell"
    assert out["next_route"] != "local_maturation_window_found_slow_path_audit_required"
    assert out["next_route"] == "mixed_identity_requires_refinement"
    assert out["layers"]["primary_neighbourhood"]["verdict"] == (
        "no_local_maturation_window"
    )
    _assert_lifecycle_locked(out)


@pytest.mark.parametrize(
    "shell_verdict,expected",
    [
        ("local_maturation_window", "maturation_candidate_in_secondary_shell"),
        ("isolated_maturation_candidate", "isolated_maturation_candidate"),
        ("seed_heterogeneous_maturation", "seed_heterogeneous_maturation"),
    ],
)
def test_mixed_c0_does_not_hide_closed_secondary_shell_structure(
    shell_verdict, expected,
):
    out = _run(
        c0_verdict="heterogeneous_or_unresolved",
        primary_verdict="no_local_maturation_window",
        shell_verdict=shell_verdict,
    )
    assert out["verdict"] == expected
    assert out["next_route"] == "mixed_identity_requires_refinement"
    assert out["layers"]["source_identity"]["verdict"] == (
        "heterogeneous_or_unresolved"
    )
    _assert_lifecycle_locked(out)


def test_balanced_tonic_plus_complete_negative_routes_to_independent_mechanism():
    out = _run()
    assert out["verdict"] == "no_maturation_in_tested_secondary_shell"
    assert out["next_route"] == (
        "tonic_substrate_valid_independent_maturation_exit_required"
    )
    _assert_lifecycle_locked(out)


def test_saturated_branch_plus_complete_negative_routes_to_fast_carrier_repair():
    out = _run(c0_verdict="replicated_refractory_limited_plateau")
    assert out["verdict"] == "no_maturation_in_tested_secondary_shell"
    assert out["layers"]["source_identity"]["verdict"] == (
        "replicated_refractory_limited_plateau"
    )
    assert out["next_route"] == "fast_carrier_repair_required"
    _assert_lifecycle_locked(out)


@pytest.mark.parametrize(
    "primary_verdict,expected",
    [
        ("isolated_maturation_candidate", "isolated_maturation_candidate"),
        ("representation_sensitive", "representation_sensitive_maturation"),
        ("seed_heterogeneous_maturation", "seed_heterogeneous_maturation"),
    ],
)
def test_primary_ambiguous_classes_cannot_authorize_slow_path(
    primary_verdict, expected
):
    out = _run(primary_verdict=primary_verdict)
    assert out["verdict"] == expected
    assert out["next_route"] == "mixed_identity_requires_refinement"
    _assert_lifecycle_locked(out)


def test_incomplete_shell_is_not_a_negative():
    out = _run(
        shell_verdict="insufficient_coverage",
        shell_status="incomplete",
    )
    assert out["verdict"] == "secondary_shell_incomplete"
    assert out["verdict"] != "no_maturation_in_tested_secondary_shell"
    _assert_lifecycle_locked(out)


def test_C1_manifest_block_has_no_scientific_fallthrough():
    out = _run(primary_status="blocked_manifest")
    assert out["verdict"] == "C1_blocked_manifest"
    assert out["next_route"] == "no_evidence"
    _assert_lifecycle_locked(out)


def test_explicit_C0_only_result_is_not_lifecycle_or_maturation():
    out = _run(
        primary_verdict="not_tested",
        shell_verdict="not_tested",
        primary_status="not_tested",
        shell_status="not_tested",
        modal_status="not_tested",
    )
    assert out["verdict"] == "balanced_AI_tonic_candidate_supported"
    assert out["next_route"] == (
        "tonic_substrate_valid_independent_maturation_exit_required"
    )
    _assert_lifecycle_locked(out)


def test_engineering_complete_unknown_labels_do_not_default_to_success_or_negative():
    out = _run(
        c0_verdict="engineering_complete",
        primary_verdict="engineering_complete",
        shell_verdict="engineering_complete",
    )
    assert out["verdict"] == "no_evidence"
    assert out["next_route"] == "no_evidence"
    _assert_lifecycle_locked(out)


def test_modal_label_cannot_override_C0_C1():
    inputs = _inputs()
    inputs["modal"] = {
        "verdict": "maturation_window_at_primary_convex_states",
        "status": "complete",
    }
    inputs["provenance"] = build_provenance(
        {k: inputs[k] for k in ("c0", "c1_primary", "c1_shell", "modal", "coverage")},
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
    )
    out = adjudicate_phasec(**inputs)
    assert out["verdict"] == "no_maturation_in_tested_secondary_shell"
    assert out["next_route"] == (
        "tonic_substrate_valid_independent_maturation_exit_required"
    )
    _assert_lifecycle_locked(out)
