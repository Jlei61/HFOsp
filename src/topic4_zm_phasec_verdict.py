"""Pure, fail-closed adjudication for the Z/M Phase-C audit.

This module only combines already-produced C0/C1/modal summaries.  It never
derives a simulator metric and it deliberately keeps three statements apart:

* ``fine_verdict``: the evidence label from Phase C (spec section 9);
* ``next_route``: the next *diagnostic* branch suggested by that evidence;
* lifecycle fields: permanently negative/not-tested in Phase C.

In particular, a secondary-shell candidate is not a physically reachable
primary-path window, and engineering completion is never lifecycle evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping


PHASEC_VERDICT_VERSION = "zm_phasec_verdict_v1_2026-07-28"

FINE_VERDICTS = (
    "C0_blocked_observables",
    "C0_insufficient_coverage",
    "resolution_sensitive_identity",
    "refractory_saturated_branch_supported",
    "balanced_AI_tonic_candidate_supported",
    "mixed_or_indeterminate_tonic_branch",
    "seed_heterogeneous_identity",
    "C1_blocked_manifest",
    "maturation_window_at_primary_convex_states",
    "maturation_candidate_in_secondary_shell",
    "isolated_maturation_candidate",
    "no_maturation_in_tested_primary_neighbourhood",
    "no_maturation_in_tested_secondary_shell",
    "secondary_shell_incomplete",
    "representation_sensitive_maturation",
    "seed_heterogeneous_maturation",
    "no_evidence",
)

NEXT_ROUTES = (
    "fast_carrier_repair_required",
    "tonic_substrate_valid_independent_maturation_exit_required",
    "local_maturation_window_found_slow_path_audit_required",
    "mixed_identity_requires_refinement",
    "no_evidence",
)

_HEX64 = frozenset("0123456789abcdef")
_REQUIRED_ARTIFACTS = ("c0", "c1_primary", "c1_shell", "modal", "coverage")

_AI_LABELS = {
    "balanced_AI_tonic_candidate_supported",
    "replicated_balanced_asynchronous_tonic_candidate",
    "balanced_asynchronous_tonic_candidate",
}
_SATURATION_LABELS = {
    "refractory_saturated_branch_supported",
    "replicated_refractory_limited_plateau",
    "refractory_limited_plateau",
}
_MIXED_LABELS = {
    "mixed_or_indeterminate_tonic_branch",
    "heterogeneous_or_unresolved",
    "mixed_or_unresolved",
}

_PRIMARY_LABELS = {
    "maturation_window_at_primary_convex_states": "positive",
    "local_maturation_window": "positive",
    "isolated_maturation_candidate": "isolated",
    "no_maturation_in_tested_primary_neighbourhood": "negative",
    "no_local_maturation_window": "negative",
    "representation_sensitive_maturation": "representation_sensitive",
    "representation_sensitive": "representation_sensitive",
    "seed_heterogeneous_maturation": "seed_heterogeneous",
    "insufficient_coverage": "incomplete",
    "C1_blocked_manifest": "blocked",
    "not_tested": "not_tested",
}

_SHELL_LABELS = {
    "maturation_candidate_in_secondary_shell": "positive",
    "local_maturation_window": "positive",
    "isolated_maturation_candidate": "isolated",
    "no_maturation_in_tested_secondary_shell": "negative",
    "no_local_maturation_window": "negative",
    "secondary_shell_incomplete": "incomplete",
    "insufficient_coverage": "incomplete",
    "representation_sensitive_maturation": "representation_sensitive",
    "representation_sensitive": "representation_sensitive",
    "seed_heterogeneous_maturation": "seed_heterogeneous",
    "not_tested": "not_tested",
}


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy or raise ``TypeError``."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite floats cannot be provenance-hashed")
        return value
    # Support numpy scalar-like values without importing numpy.
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """Canonical SHA-256 for one summary artifact.

    The summary is hashed exactly as supplied.  SHA declarations therefore
    live in the separate provenance manifest and cannot self-authenticate.
    """
    payload = json.dumps(
        _jsonable(artifact),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Build the explicit provenance object consumed by ``adjudicate_phasec``."""
    return {
        "status": "complete",
        "manifest_sha256": str(manifest_sha256),
        "config_sha256": str(config_sha256),
        "artifact_sha256": {
            name: artifact_sha256(artifacts[name])
            for name in _REQUIRED_ARTIFACTS
            if name in artifacts
        },
    }


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX64)
    )


def _verify_provenance(artifacts: Mapping[str, Any], provenance: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(provenance, Mapping):
        return {"status": "invalid", "reasons": ["provenance_missing"]}
    if provenance.get("status") != "complete":
        reasons.append("provenance_not_complete")
    for key in ("manifest_sha256", "config_sha256"):
        if not _valid_sha(provenance.get(key)):
            reasons.append(f"{key}_missing_or_invalid")
    declared = provenance.get("artifact_sha256")
    if not isinstance(declared, Mapping):
        reasons.append("artifact_sha256_missing")
        declared = {}
    for name in _REQUIRED_ARTIFACTS:
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            reasons.append(f"{name}_artifact_missing")
            continue
        expected = declared.get(name)
        if not _valid_sha(expected):
            reasons.append(f"{name}_sha_missing_or_invalid")
            continue
        try:
            observed = artifact_sha256(artifact)
        except TypeError:
            reasons.append(f"{name}_not_canonical_json")
            continue
        if observed != expected:
            reasons.append(f"{name}_sha_mismatch")
    return {
        "status": "ok" if not reasons else "invalid",
        "reasons": reasons,
        "manifest_sha256": provenance.get("manifest_sha256"),
        "config_sha256": provenance.get("config_sha256"),
    }


def _status(coverage: Mapping[str, Any], key: str) -> str | None:
    row = coverage.get(key)
    return row.get("status") if isinstance(row, Mapping) else None


def _c0_identity(c0: Mapping[str, Any]) -> str:
    label = c0.get("verdict")
    if label in _AI_LABELS:
        return "balanced_AI_tonic_candidate_supported"
    if label in _SATURATION_LABELS:
        return "refractory_saturated_branch_supported"
    if label == "resolution_sensitive_identity":
        return label
    if label in {"C0_blocked_observables", "C0_insufficient_coverage"}:
        return str(label)
    if label == "seed_heterogeneous_identity":
        return label
    if label in _MIXED_LABELS:
        classes = c0.get("seed_classes")
        if isinstance(classes, Mapping):
            values = set(classes.values())
            if values & _AI_LABELS and values & _SATURATION_LABELS:
                return "seed_heterogeneous_identity"
        return "mixed_or_indeterminate_tonic_branch"
    return "no_evidence"


def _layer_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Keep small adjudication-relevant fields, not a duplicate artifact."""
    out = {"verdict": artifact.get("verdict")}
    for key in (
        "status",
        "seed_classes",
        "eligible_seeds",
        "complete_seeds",
        "representation_results",
        "reason",
        "reasons",
    ):
        if key in artifact:
            out[key] = deepcopy(artifact[key])
    return out


def _locked_result(
    fine_verdict: str,
    next_route: str,
    *,
    c0: Mapping[str, Any],
    primary: Mapping[str, Any],
    shell: Mapping[str, Any],
    modal: Mapping[str, Any],
    provenance_check: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    if fine_verdict not in FINE_VERDICTS:
        raise ValueError(f"unregistered fine verdict: {fine_verdict}")
    if next_route not in NEXT_ROUTES:
        raise ValueError(f"unregistered next route: {next_route}")
    layers = {
        "source_identity": _layer_summary(c0),
        "primary_neighbourhood": _layer_summary(primary),
        "secondary_shell": _layer_summary(shell),
        "seed_specific_modal": _layer_summary(modal),
        "observation_layer": "blocked_reference_artifacts",
        "observation_match": "blocked",
        "entry": "not_tested",
        "offset": "not_tested",
        "recovery_lifecycle": "not_established",
    }
    return {
        "version": PHASEC_VERDICT_VERSION,
        "verdict": fine_verdict,
        "fine_verdict": fine_verdict,
        "next_route": next_route,
        "reason": reason,
        "layers": layers,
        "provenance_check": deepcopy(dict(provenance_check)),
        "observation_layer": "blocked_reference_artifacts",
        "observation_match": "blocked",
        "entry": "not_tested",
        "offset": "not_tested",
        "recovery_lifecycle": "not_established",
        "phase_c2_authorized": False,
        "actuator_authorized": False,
    }


def adjudicate_phasec(
    *,
    c0: Mapping[str, Any],
    c1_primary: Mapping[str, Any],
    c1_shell: Mapping[str, Any],
    modal: Mapping[str, Any],
    coverage: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the single fail-closed Phase-C decision.

    Required coverage schema::

        {
          "c0": {"status": "complete|blocked_observables|insufficient|resolution_sensitive"},
          "c1_primary": {"status": "complete|blocked_manifest|not_tested"},
          "c1_shell": {"status": "complete|incomplete|not_tested"},
          "modal": {"status": "complete|partial|not_tested"}
        }

    Modal completion never changes the phenotype verdict.  It is nevertheless
    required as an explicit layer so that missing work is not silently hidden.
    """
    artifacts = {
        "c0": c0,
        "c1_primary": c1_primary,
        "c1_shell": c1_shell,
        "modal": modal,
        "coverage": coverage,
    }
    pcheck = _verify_provenance(artifacts, provenance)

    def result(fine: str, route: str, reason: str) -> dict[str, Any]:
        return _locked_result(
            fine,
            route,
            c0=c0 if isinstance(c0, Mapping) else {},
            primary=c1_primary if isinstance(c1_primary, Mapping) else {},
            shell=c1_shell if isinstance(c1_shell, Mapping) else {},
            modal=modal if isinstance(modal, Mapping) else {},
            provenance_check=pcheck,
            reason=reason,
        )

    if pcheck["status"] != "ok":
        return result("no_evidence", "no_evidence", "provenance_missing_or_mismatched")
    if not isinstance(coverage, Mapping):
        return result("no_evidence", "no_evidence", "coverage_missing")
    statuses = {
        key: _status(coverage, key)
        for key in ("c0", "c1_primary", "c1_shell", "modal")
    }
    if any(value is None for value in statuses.values()):
        return result("no_evidence", "no_evidence", "coverage_layer_missing")

    c0_status = statuses["c0"]
    if c0_status == "blocked_observables":
        return result("C0_blocked_observables", "no_evidence", "C0_observables_blocked")
    if c0_status == "insufficient":
        return result("C0_insufficient_coverage", "no_evidence", "C0_coverage_incomplete")
    if c0_status == "resolution_sensitive":
        return result(
            "resolution_sensitive_identity",
            "mixed_identity_requires_refinement",
            "C0_identity_changes_with_resolution",
        )
    if c0_status != "complete":
        return result("no_evidence", "no_evidence", "unknown_C0_coverage_status")

    identity = _c0_identity(c0)
    if identity in {
        "C0_blocked_observables",
        "C0_insufficient_coverage",
        "resolution_sensitive_identity",
    }:
        route = (
            "mixed_identity_requires_refinement"
            if identity == "resolution_sensitive_identity"
            else "no_evidence"
        )
        return result(identity, route, "C0_summary_reports_block_or_sensitivity")
    if identity == "no_evidence":
        return result("no_evidence", "no_evidence", "C0_identity_unrecognized_or_missing")
    identity_unresolved = identity in {
        "mixed_or_indeterminate_tonic_branch",
        "seed_heterogeneous_identity",
    }

    primary_status = statuses["c1_primary"]
    if primary_status == "blocked_manifest":
        return result("C1_blocked_manifest", "no_evidence", "C1_manifest_blocked")
    if primary_status not in {"complete", "not_tested"}:
        return result("C1_blocked_manifest", "no_evidence", "C1_primary_coverage_incomplete")

    primary_kind = _PRIMARY_LABELS.get(c1_primary.get("verdict"))
    shell_kind = _SHELL_LABELS.get(c1_shell.get("verdict"))
    if primary_kind is None or shell_kind is None:
        return result("no_evidence", "no_evidence", "unregistered_C1_verdict")

    # A locked, explicit C0-only stopping point may report the identity label.
    # It is not a completed Phase C1 decision.
    if primary_status == "not_tested":
        if primary_kind != "not_tested" or statuses["c1_shell"] != "not_tested":
            return result("no_evidence", "no_evidence", "C1_status_summary_conflict")
        if shell_kind != "not_tested":
            return result("no_evidence", "no_evidence", "C1_shell_status_summary_conflict")
        if identity_unresolved:
            return result(
                identity,
                "mixed_identity_requires_refinement",
                "C0_identity_not_mechanistically_resolved_C1_not_tested",
            )
        route = (
            "fast_carrier_repair_required"
            if identity == "refractory_saturated_branch_supported"
            else "tonic_substrate_valid_independent_maturation_exit_required"
        )
        return result(identity, route, "C0_identity_supported_C1_explicitly_not_tested")

    if primary_kind == "blocked":
        return result("C1_blocked_manifest", "no_evidence", "C1_summary_reports_block")
    if primary_kind == "incomplete":
        return result("C1_blocked_manifest", "no_evidence", "C1_primary_incomplete")
    if primary_kind == "positive":
        return result(
            "maturation_window_at_primary_convex_states",
            "local_maturation_window_found_slow_path_audit_required",
            (
                "contiguous_maturation_window_on_primary_reachable_path_"
                "with_unresolved_source_identity"
                if identity_unresolved
                else "contiguous_maturation_window_on_primary_reachable_path"
            ),
        )
    if identity_unresolved:
        return result(
            identity,
            "mixed_identity_requires_refinement",
            "C0_identity_not_mechanistically_resolved",
        )
    if primary_kind == "isolated":
        return result(
            "isolated_maturation_candidate",
            "mixed_identity_requires_refinement",
            "isolated_primary_candidate_is_not_a_window",
        )
    if primary_kind == "representation_sensitive":
        return result(
            "representation_sensitive_maturation",
            "mixed_identity_requires_refinement",
            "primary_maturation_depends_on_representation",
        )
    if primary_kind == "seed_heterogeneous":
        return result(
            "seed_heterogeneous_maturation",
            "mixed_identity_requires_refinement",
            "primary_maturation_disagrees_across_seeds",
        )
    if primary_kind != "negative":
        return result("no_evidence", "no_evidence", "primary_C1_not_adjudicated")

    shell_status = statuses["c1_shell"]
    if shell_status == "incomplete":
        return result(
            "secondary_shell_incomplete",
            (
                "fast_carrier_repair_required"
                if identity == "refractory_saturated_branch_supported"
                else "mixed_identity_requires_refinement"
            ),
            "primary_negative_but_secondary_shell_incomplete",
        )
    if shell_status == "not_tested":
        if shell_kind != "not_tested":
            return result("no_evidence", "no_evidence", "C1_shell_status_summary_conflict")
        route = (
            "fast_carrier_repair_required"
            if identity == "refractory_saturated_branch_supported"
            else "tonic_substrate_valid_independent_maturation_exit_required"
        )
        return result(
            "no_maturation_in_tested_primary_neighbourhood",
            route,
            "complete_primary_negative_secondary_shell_explicitly_not_tested",
        )
    if shell_status != "complete":
        return result("secondary_shell_incomplete", "no_evidence", "unknown_shell_status")
    if shell_kind == "positive":
        # Sensitivity evidence only.  It must never authorize the primary
        # slow-path route.
        route = (
            "fast_carrier_repair_required"
            if identity == "refractory_saturated_branch_supported"
            else "mixed_identity_requires_refinement"
        )
        return result(
            "maturation_candidate_in_secondary_shell",
            route,
            "secondary_shell_candidate_does_not_establish_primary_reachability",
        )
    if shell_kind == "isolated":
        return result(
            "isolated_maturation_candidate",
            "mixed_identity_requires_refinement",
            "isolated_secondary_candidate_is_not_a_window",
        )
    if shell_kind == "representation_sensitive":
        return result(
            "representation_sensitive_maturation",
            "mixed_identity_requires_refinement",
            "secondary_maturation_depends_on_representation",
        )
    if shell_kind == "seed_heterogeneous":
        return result(
            "seed_heterogeneous_maturation",
            "mixed_identity_requires_refinement",
            "secondary_maturation_disagrees_across_seeds",
        )
    if shell_kind == "incomplete":
        return result(
            "secondary_shell_incomplete",
            "mixed_identity_requires_refinement",
            "secondary_shell_summary_incomplete",
        )
    if shell_kind == "negative":
        route = (
            "fast_carrier_repair_required"
            if identity == "refractory_saturated_branch_supported"
            else "tonic_substrate_valid_independent_maturation_exit_required"
        )
        return result(
            "no_maturation_in_tested_secondary_shell",
            route,
            "complete_primary_and_secondary_bounded_negative",
        )
    return result("no_evidence", "no_evidence", "no_registered_fallthrough")
