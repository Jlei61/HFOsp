#!/usr/bin/env python3
"""Fail-closed machine audit for H2b cross-task transfer v0.2.

The audit has two admissible outcomes.  ``PASS_WAITING_FOR_DATA_MOUNTS`` means
the frozen cohort census and durable queue are internally consistent but raw
inference caches are not mounted.  ``PASS_COMPLETE`` additionally requires the
entire state-cache, patient-separate probe, phenotype and patient-first result
chain.  Waiting is an operational state, never a biological result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch as _torch  # noqa: F401; load the compatible native runtime first
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    H2B_V0_2_REVISION,
    V0_2_RESULT_ROOT,
    atomic_json,
    sha256_file,
    support_tier,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.risk_probe import (  # noqa: E402
    validate_risk_table,
)
from src.topic5_continuous_marked_state_r1.coverage import (  # noqa: E402
    CoverageTable,
)


PROBE_PRODUCTS = (
    "per_seed_probe_metrics.csv",
    "patient_median_probe_metrics.csv",
    "lead_curve.csv",
    "time_label_permutation.json",
    "positive_synthetic.json",
    "risk_probe_machine_audit.json",
)
# A permutation null that could not be built must carry explicit nulls rather
# than a number that could be mistaken for a biological negative.
NULL_PERMUTATION_SCALARS = (
    "observed_state_minus_observation",
    "null_median", "null_mean", "null_q025", "null_q975",
)
FALSE_BOUNDARY_KEYS = (
    "formal_test_partition_opened", "sealed_opened", "h3_or_t2_run",
    "paper_ready_figures_modified",
)
RISK_BOOLEAN_COLUMNS = (
    "is_case", "horizon_seizure_free", "in_ictal_or_postictal",
    "observation_available", "wrong_time_donor_valid",
    "wrong_time_same_segment", "wrong_time_exclusion_clear",
)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _hash_bound(path: Path, expected: str | None, label: str) -> None:
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(bool(expected), f"{label} has no frozen SHA256")
    observed = sha256_file(path)
    _require(observed == str(expected), f"{label} SHA256 drift: {observed}")


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().astype(int).tolist())
        if values.issubset({0, 1}):
            return series.astype(int).astype(bool)
    lower = series.astype(str).str.strip().str.lower()
    _require(set(lower.unique()).issubset({"true", "false"}),
             f"{name} is not strict boolean")
    return lower.map({"true": True, "false": False}).astype(bool)


def _risk_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={
        "patient_id": str, "seizure_id": str, "risk_set_id": str,
        "anchor_id": str, "split": str, "evaluation_tier": str,
        "segment_id": str, "observation_signature": str,
    })
    for name in RISK_BOOLEAN_COLUMNS:
        frame[name] = _strict_bool(frame[name], name)
    for name in ("anchor_time", "seizure_onset", "segment_start", "segment_end"):
        frame[name] = frame[name].astype(np.float64)
    return frame


def _boundary_false(payload: dict[str, Any], label: str) -> None:
    boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
    for key in FALSE_BOUNDARY_KEYS:
        if key in payload:
            _require(payload[key] is False, f"{label}: {key} is not false")
        if key in boundary:
            _require(boundary[key] is False, f"{label}.boundary: {key} is not false")
    for key in ("seizure_loss_updates_state", "state_source_uses_seizure_labels",
                "seizure_gradient_path"):
        if key in payload:
            _require(payload[key] is False, f"{label}: {key} is not false")
        if key in boundary:
            _require(boundary[key] is False, f"{label}.boundary: {key} is not false")


def _audit_probe_permutation_semantics(output: Path, label: str) -> str:
    """Require inferential status to agree with finite primary-lead support."""
    audit = _json(output / "risk_probe_machine_audit.json")
    patient = pd.read_csv(output / "patient_median_probe_metrics.csv")
    permutation = _json(output / "time_label_permutation.json")
    metric = "state_minus_observation_conditional_log_loss"
    primary = patient[patient["lead_minutes"].astype(int) == 30]
    values = (
        primary[metric].to_numpy(dtype=float)
        if metric in primary.columns else np.asarray([], dtype=float)
    )
    values = values[np.isfinite(values)]
    summary = audit.get("time_label_permutation") or {}
    _require(summary.get("status") == permutation.get("status"),
             f"{label}: compact/full permutation status drift")
    if len(values):
        observed = float(np.median(values))
        _require(permutation.get("status") == "COMPLETE",
                 f"{label}: finite primary effect lacks COMPLETE permutation")
        _require(np.isfinite(float(permutation.get(
            "observed_state_minus_observation", np.nan
        ))), f"{label}: observed permutation effect is non-finite")
        _require(np.isclose(
            float(permutation["observed_state_minus_observation"]), observed,
            rtol=1e-10, atol=1e-12,
        ), f"{label}: permutation observed effect differs from patient median")
        null = np.asarray(permutation.get("null_values") or [], dtype=float)
        requested = int(permutation.get("n_permutations", -1))
        _require(len(null) == requested and np.isfinite(null).all(),
                 f"{label}: COMPLETE permutation has missing/non-finite nulls")
        if "n_finite_permutations" in permutation:
            _require(int(permutation["n_finite_permutations"]) == requested,
                     f"{label}: finite permutation denominator drift")
        return "COMPLETE"
    _require(permutation.get("status") == "NOT_ESTIMABLE_AT_PRIMARY_LEAD",
             f"{label}: non-finite primary effect was presented as inferential")
    _require(permutation.get("observed_state_minus_observation") is None,
             f"{label}: non-estimable permutation retained observed effect")
    _require(int(permutation.get("n_finite_permutations", -1)) == 0,
             f"{label}: non-estimable permutation has a finite denominator")
    _require((permutation.get("null_values") or []) == [],
             f"{label}: non-estimable permutation retained null draws")
    return "NOT_ESTIMABLE_AT_PRIMARY_LEAD"


def _expected_probe_tasks(root: Path, support: dict[str, dict]) -> list[str]:
    """Every patient/arm the queue is obliged to produce, derived from the inputs.

    Deriving the denominator from the risk tables rather than from whatever
    happens to exist under ``fits/`` is what turns "the queue stopped" into a
    detectable state instead of a silently short cohort.
    """
    tasks = []
    for subject, manifest in sorted(support.items()):
        if int(manifest["n_primary_eligible_seizures"]) < 2:
            continue
        tasks.append(f"{subject}/primary")
        if (root / "risk_sets" / subject
                / "matched_wrong_time_risk_sets.csv").is_file():
            tasks.append(f"{subject}/matched_wrong_time")
    return tasks


def _audit_probe_task(root: Path, label: str) -> dict[str, Any]:
    """Accept COMPLETE or structured primary-lead non-estimability."""
    subject, analysis = label.split("/", maxsplit=1)
    output = root / "fits/by_subject" / subject / analysis
    for name in PROBE_PRODUCTS:
        _require((output / name).is_file() and (output / name).stat().st_size > 0,
                 f"{label}: probe product {name} is missing or empty")
    audit = _json(output / "risk_probe_machine_audit.json")
    _boundary_false(audit, f"{label}: probe")
    _require(audit.get("status") == "COMPLETE", f"{label}: probe did not complete")
    _require(audit.get("execution_status", "COMPLETE") == "COMPLETE",
             f"{label}: probe execution did not complete")
    estimability = audit.get("scientific_estimability")
    _require(estimability in {"ESTIMABLE", "NOT_ESTIMABLE", None},
             f"{label}: unknown scientific estimability {estimability!r}")
    _require(audit.get("positive_synthetic", {}).get("status") == "PASS",
             f"{label}: positive synthetic recovery failed")
    _require(audit.get("seed_aggregation")
             == "median_within_patient_before_cohort_inference",
             f"{label}: optimizer seeds were treated as patients")
    permutation = _json(output / "time_label_permutation.json")
    permutation_status = _audit_probe_permutation_semantics(output, label)
    if permutation_status == "NOT_ESTIMABLE_AT_PRIMARY_LEAD":
        _require(int(permutation.get("n_permutations_run", -1)) == 0,
                 f"{label}: an unestimable permutation reports permutations run")
        _require(int(permutation.get("n_finite_permutations", -1)) == 0,
                 f"{label}: an unestimable permutation has finite null draws")
        for key in NULL_PERMUTATION_SCALARS:
            _require(permutation.get(key) is None,
                     f"{label}: unestimable permutation fabricated {key}")
        _require((permutation.get("null_values") or []) == [],
                 f"{label}: unestimable permutation retained null draws")
        _require(estimability == "NOT_ESTIMABLE",
                 f"{label}: permutation is unestimable but the audit says {estimability!r}")
    else:
        _require(estimability == "ESTIMABLE",
                 f"{label}: COMPLETE permutation lacks explicit ESTIMABLE status")
        _require(int(permutation.get("n_permutations_run", 0)) > 0,
                 f"{label}: a completed permutation ran no permutations")
    receipt_path = output / ".task_complete.json"
    receipt: dict[str, Any] = {}
    if receipt_path.is_file():
        receipt = _json(receipt_path)
        for name, digest in (receipt.get("output_sha256") or {}).items():
            observed = sha256_file(output / name)
            _require(observed == digest,
                     f"{label}: {name} changed after completion ({observed})")
    return {
        "task": label,
        "execution_status": audit.get("execution_status", audit.get("status")),
        "scientific_estimability": estimability,
        "permutation_status": permutation_status,
        "hash_baseline_adopted_from_existing_output": bool(
            receipt.get("hash_baseline_adopted_from_existing_output")
        ),
    }


def _audit_inventory(root: Path) -> tuple[dict, dict[tuple[str, int], dict]]:
    path = root / "manifests/r1_7_checkpoint_inventory.json"
    inventory = _json(path)
    _require(inventory.get("status") == "COMPLETE", "checkpoint inventory not COMPLETE")
    _require(inventory.get("revision") == H2B_V0_2_REVISION,
             "checkpoint inventory revision drift")
    _require(inventory.get("h1_is_stratification_not_h2b_gate") is True,
             "H1 stability was turned into an H2b gate")
    _boundary_false(inventory, "checkpoint inventory")
    entries = inventory.get("entries") or []
    subjects = list(map(str, inventory.get("subjects") or []))
    _require(len(entries) == int(inventory.get("n_cells", -1)),
             "checkpoint inventory cell count drift")
    _require(len(set(subjects)) == int(inventory.get("n_subjects", -1)),
             "checkpoint inventory subject count drift")
    by_key: dict[tuple[str, int], dict] = {}
    for entry in entries:
        key = (str(entry["subject"]), int(entry["seed"]))
        _require(key not in by_key, f"duplicate checkpoint cell {key}")
        by_key[key] = entry
        _boundary_false(entry, f"checkpoint {key}")
        _require(entry.get("state_source_uses_seizure_labels") is False,
                 f"checkpoint {key} used seizure labels")
        if entry.get("checkpoint_available"):
            _hash_bound(Path(entry["checkpoint_path"]), entry.get("checkpoint_sha256"),
                        f"checkpoint {key}")
            _hash_bound(Path(entry["result_path"]), entry.get("result_sha256"),
                        f"checkpoint result {key}")
    _require(sum(bool(row.get("checkpoint_available")) for row in entries)
             == int(inventory.get("n_checkpoint_available_cells", -1)),
             "available checkpoint count drift")
    return inventory, by_key


def _audit_census(root: Path, inventory_path: Path) -> tuple[dict, pd.DataFrame]:
    manifest_root = root / "manifests"
    census = _json(manifest_root / "support_census.json")
    _require(census.get("status") == "COMPLETE", "support census not COMPLETE")
    _boundary_false(census, "support census")
    _require(census.get("post_development_seizure_identifiers_persisted") is False,
             "support census persisted post-development seizure identifiers")
    _hash_bound(inventory_path, census.get("checkpoint_inventory_sha256"),
                "support census checkpoint inventory")
    output_paths = {
        "patient_support_census": manifest_root / "patient_support_census.csv",
        "support_by_lead_provisional": manifest_root / "support_by_lead_provisional.csv",
        "seizure_crosswalk": manifest_root / "seizure_crosswalk.csv",
    }
    for name, path in output_paths.items():
        _hash_bound(path, (census.get("output_sha256") or {}).get(name), name)
    patients = pd.read_csv(output_paths["patient_support_census"])
    _require(patients["subject"].astype(str).is_unique,
             "patient support census contains duplicate subject")
    _require(len(patients) == int(census.get("n_subjects", -1)),
             "patient support census denominator drift")
    crosswalk = pd.read_csv(output_paths["seizure_crosswalk"], dtype={
        "subject": str, "seizure_id": str,
    })
    _require(not crosswalk.duplicated(["subject", "seizure_id"]).any(),
             "development seizure crosswalk IDs are not unique")
    for row in patients.itertuples(index=False):
        subject = str(row.subject)
        coverage_path = Path(str(row.coverage_path))
        if not bool(row.coverage_available):
            _require(not (crosswalk.subject == subject).any(),
                     f"{subject}: seizures persisted without a frozen coverage split")
            continue
        coverage = CoverageTable.load(coverage_path)
        selected = crosswalk[crosswalk.subject == subject]
        _require(bool((selected.onset_epoch.astype(float) < coverage.dev_end_epoch).all()),
                 f"{subject}: post-development seizure leaked into crosswalk")
        _require(len(selected) == int(row.n_seizures_in_frozen_inventory),
                 f"{subject}: development seizure denominator drift")
    return census, patients


def _audit_complete(root: Path, inventory: dict,
                    inventory_by_key: dict[tuple[str, int], dict], *,
                    allow_precompletion: bool = False) -> dict:
    queue = _json(root / "QUEUE_STATUS.json")
    if allow_precompletion:
        _require(queue.get("status") == "RUNNING" and queue.get("stage") == "AUDIT_RESULTS",
                 "pre-completion audit requires the explicit AUDIT_RESULTS stage")
    else:
        complete = _json(root / "COHORT_RUN_COMPLETE.json")
        _require(complete.get("status") == "COMPLETE", "cohort completion marker not COMPLETE")
        _require(queue.get("status") == "COMPLETE", "queue status not COMPLETE")
        _require(queue.get("stage") == "COHORT_RUN_COMPLETE", "queue completion stage drift")
        _boundary_false(complete, "cohort completion")
    _boundary_false(queue, "queue completion")

    upstream_audit_path = root / "reports/r1_7b_consumer_acceptance_audit.json"
    upstream_audit = _json(upstream_audit_path)
    _boundary_false(upstream_audit, "R1.7B consumer acceptance")
    _require(
        upstream_audit.get("status") == "PASS_EXPLORATORY_DEVELOPMENT_SOURCE",
        "R1.7B source was not accepted by the consumer-side audit",
    )
    upstream_gate = upstream_audit.get("v0_1_stage2_release_gate") or {}
    _require(upstream_gate.get("gate_met_by_consumed_release") is False,
             "R1.7B was silently promoted to the formal v0.1 release gate")
    _require(upstream_gate.get("weakening_is_declared_not_silent") is True,
             "R1.7B exploratory release weakening was not declared")
    _require(upstream_audit.get("development_only") is True,
             "R1.7B consumer audit did not preserve development-only scope")

    input_manifests = sorted((root / "risk_sets").glob("*/input_manifest.json"))
    _require(bool(input_manifests), "no completed subject input manifest")
    expected_cache_keys: set[tuple[str, int]] = set()
    support_by_subject: dict[str, dict] = {}
    for path in input_manifests:
        value = _json(path)
        subject = str(value["subject"])
        _require(value.get("status") == "COMPLETE", f"{subject}: input manifest incomplete")
        _boundary_false(value, f"{subject}: input manifest")
        _require(value.get("post_development_seizure_identifiers_persisted") is False,
                 f"{subject}: post-development seizure identifiers persisted")
        for key in ("query", "seizure", "support", "global_exclusion"):
            _hash_bound(Path(value[f"{key}_path"]), value.get(f"{key}_sha256"),
                        f"{subject}: {key}")
        for key in ("coverage", "design"):
            _hash_bound(Path(value[f"{key}_path"]), value.get(f"{key}_sha256"),
                        f"{subject}: {key}")
        _require(value.get("state_source_uses_seizure_labels") is False,
                 f"{subject}: seizure label entered state source")
        primary = int(value["n_primary_eligible_seizures"])
        _require(value.get("support_tier") == support_tier(primary),
                 f"{subject}: support tier/count drift")
        support_by_subject[subject] = value
        if int(value["n_primary_eligible_seizures"]) > 0:
            expected_cache_keys.update(
                key for key, entry in inventory_by_key.items()
                if key[0] == subject and entry.get("checkpoint_available")
            )

    expected_probe_tasks = _expected_probe_tasks(root, support_by_subject)
    probe_task_audits = [
        _audit_probe_task(root, label) for label in expected_probe_tasks
    ]

    observed_cache_keys: set[tuple[str, int]] = set()
    for cache in sorted((root / "state_cache").glob("*/seed_*/states.npz")):
        subject = cache.parents[1].name
        seed = int(cache.parent.name.split("_")[-1])
        key = (subject, seed)
        observed_cache_keys.add(key)
        manifest = _json(cache.with_suffix(".manifest.json"))
        _hash_bound(cache, manifest.get("cache_sha256"), f"state cache {key}")
        _require(manifest.get("status") == "COMPLETE", f"state cache {key} incomplete")
        _require(manifest.get("h2b_revision") == H2B_V0_2_REVISION,
                 f"state cache {key} revision drift")
        _require(manifest.get("state_frozen") is True
                 and manifest.get("all_parameters_frozen") is True,
                 f"state cache {key} was not frozen")
        _require(manifest.get("seizure_gradient_path") is False
                 and manifest.get("state_update_uses_seizure_label") is False,
                 f"state cache {key} has a seizure-label training path")
        _require(manifest.get("max_source_time_le_anchor") is True
                 and manifest.get("all_current_observations_fresh") is True
                 and manifest.get("gap_reset") is True,
                 f"state cache {key} failed causal/fresh/gap contract")
        checkpoint = inventory_by_key.get(key)
        _require(checkpoint is not None and checkpoint.get("checkpoint_available"),
                 f"state cache {key} has no frozen inventory checkpoint")
        _require(manifest.get("checkpoint_sha256") == checkpoint.get("checkpoint_sha256"),
                 f"state cache {key} checkpoint digest drift")
        with np.load(cache, allow_pickle=False) as data:
            _require(data["anchor_time_epoch"].dtype == np.float64,
                     f"state cache {key} time is not float64")
            _require(data["coverage_segment_index"].dtype == np.int64,
                     f"state cache {key} segment is not int64")
            _require(np.all(data["max_source_time_epoch"] <=
                            data["anchor_time_epoch"] + 1e-9),
                     f"state cache {key} uses a future observation")
            _require(bool(data["observation_available"].all()),
                     f"state cache {key} contains unavailable observations")
    _require(observed_cache_keys == expected_cache_keys,
             f"state cache cells differ from expected: missing={sorted(expected_cache_keys-observed_cache_keys)}, extra={sorted(observed_cache_keys-expected_cache_keys)}")

    probe_subjects = []
    n_probe_analyses = 0
    permutation_status_counts: dict[str, int] = {}
    for subject, input_manifest in support_by_subject.items():
        summary_path = root / "risk_sets" / subject / "risk_table_summary.json"
        if int(input_manifest["n_primary_eligible_seizures"]) < 1:
            _require(not summary_path.exists(),
                     f"{subject}: risk table exists without an eligible primary seizure")
            continue
        summary = _json(summary_path)
        _boundary_false(summary, f"{subject}: risk table summary")
        primary_path = Path(summary["primary"]["path"])
        _hash_bound(primary_path, summary["primary"].get("sha256"),
                    f"{subject}: primary risk table")
        manifest = _json(primary_path.with_suffix(".manifest.json"))
        _hash_bound(primary_path, manifest.get("risk_table_sha256"),
                    f"{subject}: primary risk-table manifest")
        frame = _risk_table(primary_path)
        validate_risk_table(
            frame, arms=("B_history", "B_observation", "B_state", "memoryless"),
            require_wrong_time=False,
        )
        _require(set(frame.patient_id.astype(str)) == {subject},
                 f"{subject}: patient table pooled heterogeneous patients")
        primary_ids = set(
            pd.read_csv(input_manifest["seizure_path"], dtype={"seizure_id": str})
            .loc[lambda data: _strict_bool(data.primary_30min_supported,
                                           "primary_30min_supported")]
            .seizure_id.astype(str)
        )
        _require(set(frame.seizure_id.astype(str)).issubset(primary_ids),
                 f"{subject}: sensitivity lead recruited non-primary seizure")
        if int(input_manifest["n_primary_eligible_seizures"]) >= 2:
            probe_subjects.append(subject)
            audit_path = root / "fits/by_subject" / subject / "primary/risk_probe_machine_audit.json"
            audit = _json(audit_path)
            _boundary_false(audit, f"{subject}: primary probe")
            _require(audit.get("positive_synthetic", {}).get("status") == "PASS",
                     f"{subject}: positive synthetic recovery failed")
            _hash_bound(primary_path, audit.get("input", {}).get("risk_table_sha256"),
                        f"{subject}: primary probe input")
            _require(audit.get("seed_aggregation") ==
                     "median_within_patient_before_cohort_inference",
                     f"{subject}: optimizer seeds treated as patients")
            status = _audit_probe_permutation_semantics(
                audit_path.parent, f"{subject}/primary",
            )
            permutation_status_counts[status] = (
                permutation_status_counts.get(status, 0) + 1
            )
            n_probe_analyses += 1
            wrong_output = root / "fits/by_subject" / subject / "matched_wrong_time"
            if (wrong_output / "risk_probe_machine_audit.json").is_file():
                status = _audit_probe_permutation_semantics(
                    wrong_output, f"{subject}/matched_wrong_time",
                )
                permutation_status_counts[status] = (
                    permutation_status_counts.get(status, 0) + 1
                )
                n_probe_analyses += 1
    _require(n_probe_analyses == len(expected_probe_tasks),
             "completed probe denominator differs from the risk-table task list")

    primary_index = _json(root / "fits/primary/cohort_probe_index.json")
    _require(primary_index.get("heterogeneous_patient_feature_dimensions_never_pooled") is True,
             "patient feature matrices were pooled")
    _require(int(primary_index.get("n_patients", -1)) == len(probe_subjects),
             "primary probe patient denominator drift")
    aggregate = _json(root / "reports/cohort_patient_first_summary.json")
    _boundary_false(aggregate, "patient-first aggregate")
    _require(aggregate.get("patient_first") is True
             and aggregate.get("h1_stability_used_as_gate") is False,
             "patient-first or H1-stratification contract drift")
    per_patient = pd.read_csv(root / "reports/per_patient_lead_results.csv")
    _require(not per_patient.duplicated(
        ["patient_id", "lead_minutes", "evaluation_tier"]
    ).any(), "patient-first output contains duplicate patient/lead rows")

    phenotype = _json(root / "reports/phenotype_target_availability.json")
    _require(phenotype.get("target_reclustered") is False
             and phenotype.get("replacement_target_invented") is False
             and phenotype.get("early_recruitment_scalar_derived_here") is False,
             "secondary phenotype target was redefined after seeing H2b state")
    for subject, table in (phenotype.get("subject_tables") or {}).items():
        _hash_bound(Path(table["path"]), table.get("sha256"),
                    f"{subject}: frozen phenotype table")
    available_phenotype_subjects = {
        str(subject): table
        for subject, table in (phenotype.get("subject_tables") or {}).items()
        if int(table.get("n_available_target_rows", 0)) > 0
    }
    n_estimable_phenotype_cells = 0
    if available_phenotype_subjects:
        phenotype_index = _json(root / "fits/phenotype/cohort_phenotype_index.json")
        _boundary_false(phenotype_index, "phenotype cohort index")
        _require(phenotype_index.get("status") == "COMPLETE",
                 "phenotype cohort index is not COMPLETE")
        _require(phenotype_index.get("target_reclustered") is False,
                 "phenotype cohort index reports target reclustering")
        _require(
            phenotype_index.get("heterogeneous_patient_feature_dimensions_never_pooled")
            is True,
            "phenotype features were pooled across heterogeneous patients",
        )
        patient_audits = phenotype_index.get("patient_audits") or {}
        _require(set(patient_audits) == set(available_phenotype_subjects),
                 "phenotype execution denominator differs from frozen target availability")
        _require(int(phenotype_index.get("n_patients_run", -1)) ==
                 len(available_phenotype_subjects),
                 "phenotype patient execution count drift")
        for subject, entry in patient_audits.items():
            audit_path = Path(entry["path"])
            _hash_bound(audit_path, entry.get("sha256"),
                        f"{subject}: phenotype audit")
            patient_audit = _json(audit_path)
            _boundary_false(patient_audit, f"{subject}: phenotype probe")
            target_path = Path(available_phenotype_subjects[subject]["path"])
            _require(Path(patient_audit["input"]["input_path"]).resolve() ==
                     target_path.resolve(),
                     f"{subject}: phenotype probe points to another target table")
            _hash_bound(target_path, patient_audit["input"].get("input_sha256"),
                        f"{subject}: phenotype probe input")
            _require(patient_audit.get("target_reclustered") is False,
                     f"{subject}: phenotype target was reclustered")
            _require(patient_audit.get("target_frozen_before_probe") is True,
                     f"{subject}: phenotype target was not frozen before fitting")
            _require(patient_audit.get("positive_synthetic", {}).get("status") == "PASS",
                     f"{subject}: phenotype positive synthetic failed")
            probe_audit = patient_audit.get("probe_audit") or {}
            _require(probe_audit.get("matched_wrong_time_is_not_a_phenotype_gate")
                     is True,
                     f"{subject}: wrong-time donor availability gated phenotype fitting")
            _require(probe_audit.get("phenotype_arms") ==
                     ["baseline", "observation", "state"],
                     f"{subject}: unexpected phenotype comparison arms")
        for label, output in (phenotype_index.get("outputs") or {}).items():
            _hash_bound(Path(output["path"]), output.get("sha256"),
                        f"phenotype cohort {label} output")
        per_seed_path = Path(phenotype_index["outputs"]["per_seed"]["path"])
        per_seed_phenotype = pd.read_csv(per_seed_path)
        _require("status" in per_seed_phenotype,
                 "phenotype per-seed output has no estimability status")
        status_counts = per_seed_phenotype.groupby(
            ["patient_id", "target_name"], dropna=False,
        )["status"].nunique()
        _require(bool((status_counts == 1).all()),
                 "phenotype estimability changes across optimizer seeds")
        n_estimable_phenotype_cells = int(
            per_seed_phenotype.loc[
                per_seed_phenotype["status"].astype(str) == "ok",
                ["patient_id", "target_name"],
            ].drop_duplicates().shape[0]
        )
        _require("correct_minus_wrong_time_loss" not in per_seed_phenotype,
                 "secondary phenotype silently reintroduced wrong-time as a gate")
    return {
        "n_subjects_with_input_manifest": len(input_manifests),
        "r1_7b_consumer_acceptance_audit": {
            "path": str(upstream_audit_path),
            "sha256": sha256_file(upstream_audit_path),
            "status": upstream_audit.get("status"),
            "formal_v0_1_release_gate_met": False,
        },
        "n_state_cache_cells": len(observed_cache_keys),
        "n_probe_subjects": len(probe_subjects),
        "n_probe_analyses": n_probe_analyses,
        "n_expected_probe_tasks": len(expected_probe_tasks),
        "probe_task_audits": probe_task_audits,
        "permutation_status_counts": permutation_status_counts,
        "n_patient_first_rows": len(per_patient),
        "n_phenotype_subject_tables": len(phenotype.get("subject_tables") or {}),
        "n_phenotype_patients_run": len(available_phenotype_subjects),
        "n_estimable_phenotype_patient_targets": n_estimable_phenotype_cells,
    }


def run(root: Path, *, allow_precompletion: bool = False) -> dict:
    root = root.resolve()
    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    inventory, by_key = _audit_inventory(root)
    census, patients = _audit_census(root, inventory_path)
    queue = _json(root / "QUEUE_STATUS.json")
    _boundary_false(queue, "queue status")
    stage = str(queue.get("stage"))
    if queue.get("status") == "WAITING":
        _require(stage in {
            "WAITING_FOR_DATA_MOUNTS", "WAITING_FOR_REQUIRED_RAW_CACHES",
        }, "unknown waiting stage")
        mounts = census.get("raw_mounts_present") or {}
        if stage == "WAITING_FOR_DATA_MOUNTS":
            _require(mounts and not all(bool(value) for value in mounts.values()),
                     "queue claims missing mounts although all are present")
        else:
            _require(int(census.get("n_required_subjects_with_raw_cache", 0))
                     < int(census.get("n_subjects_requiring_raw_for_primary_h2b", 0)),
                     "queue claims incomplete required raw caches although all are present")
        detail = {
            "n_subjects": int(inventory["n_subjects"]),
            "n_checkpoint_available_cells": int(
                inventory["n_checkpoint_available_cells"]
            ),
            "n_subjects_runnable_now": int(census["n_subjects_runnable_now"]),
            "n_subjects_requiring_raw_for_primary_h2b": int(
                census.get("n_subjects_requiring_raw_for_primary_h2b", 0)
            ),
            "n_required_subjects_with_raw_cache": int(
                census.get("n_required_subjects_with_raw_cache", 0)
            ),
            "raw_mounts_present": mounts,
            "waiting_is_not_scientific_result": True,
        }
        status = "PASS_WAITING_FOR_DATA_MOUNTS"
    else:
        detail = _audit_complete(
            root, inventory, by_key, allow_precompletion=allow_precompletion,
        )
        status = "PASS_PRECOMPLETION" if allow_precompletion else "PASS_COMPLETE"
    payload = {
        "status": status,
        "revision": "h2b_cross_task_v0_2_machine_audit_v1",
        "created_utc": utc_now(),
        "result_root": str(root),
        "details": detail,
        "patient_first": True,
        "h1_stability_used_as_gate": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "paper_ready_figures_modified": False,
    }
    atomic_json(root / "reports/machine_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    parser.add_argument("--allow-precompletion", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args.result_root, allow_precompletion=args.allow_precompletion)
    except Exception as exc:
        root = args.result_root.resolve()
        payload = {
            "status": "FAIL", "revision": "h2b_cross_task_v0_2_machine_audit_v1",
            "created_utc": utc_now(), "result_root": str(root),
            "error": repr(exc), "formal_test_partition_opened": False,
            "sealed_opened": False, "h3_or_t2_run": False,
            "paper_ready_figures_modified": False,
        }
        atomic_json(root / "reports/machine_audit.json", payload)
        raise
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
