"""Fail-closed machine audit for H2b cross-task transfer.

The audit distinguishes an engineering-complete instrument from a scientifically
estimable cohort result.  In particular, an unavailable R1.7 release is recorded
as an upstream boundary; it is never silently replaced by partial fits.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np

from .contract import (
    H2B_REVISION,
    RESULT_ROOT,
    RunBoundary,
    sha256_file,
    utc_now,
)


AUDIT_REVISION = "h2b_cross_task_machine_audit_v0_1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, passed: bool, *, evidence: Any = None,
           status_if_false: str = "FAIL") -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if bool(passed) else status_if_false,
        "evidence": evidence,
    }


def split_contract_audit(split_ids: Mapping[str, Mapping[str, Iterable[str]]]
                         ) -> dict[str, Any]:
    overlaps: dict[str, list[str]] = {}
    for patient, by_split in split_ids.items():
        sets = {str(label): set(map(str, ids)) for label, ids in by_split.items()}
        labels = sorted(sets)
        duplicated: set[str] = set()
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1:]:
                duplicated.update(sets[left].intersection(sets[right]))
        if duplicated:
            overlaps[str(patient)] = sorted(duplicated)
    return {
        "pass": not overlaps,
        "overlapping_seizure_ids": overlaps,
        "split_seizure_ids": {
            str(patient): {
                str(split): sorted(map(str, ids))
                for split, ids in by_split.items()
            }
            for patient, by_split in split_ids.items()
        },
    }


def _git_changed_paths(repo_root: Path, base: str = "b7eaf0a1") -> list[str]:
    commands = (
        ["git", "diff", "--name-only", base],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    paths: set[str] = set()
    for command in commands:
        completed = subprocess.run(
            command, cwd=repo_root, check=False, capture_output=True, text=True,
        )
        if completed.returncode == 0:
            paths.update(line for line in completed.stdout.splitlines() if line)
    return sorted(paths)


def _state_manifests(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (root / "state_cache").glob("*/seed_*/states.manifest.json")
    ):
        payload = _json(path)
        cache = Path(str(payload.get("cache", "")))
        cache_ok = cache.is_file() and sha256_file(cache) == payload.get("cache_sha256")
        source_hashes = payload.get("source_hashes") or {}
        state_source = Path(__file__).resolve().with_name("state_extraction.py")
        runner_source = (
            Path(__file__).resolve().parents[2]
            / "scripts/topic5_continuous_marked_state_h2b/extract_states.py"
        )
        rows.append({
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "subject": payload.get("checkpoint_subject"),
            "seed": payload.get("checkpoint_seed"),
            "cache": str(cache),
            "cache_hash_match": cache_ok,
            "checkpoint_hash_match": (
                Path(str(payload.get("checkpoint", ""))).is_file()
                and sha256_file(Path(str(payload["checkpoint"])))
                == payload.get("checkpoint_sha256")
            ) if payload.get("checkpoint") else False,
            "state_frozen": payload.get("state_frozen") is True,
            "all_parameters_frozen": payload.get("all_parameters_frozen") is True,
            "seizure_gradient_path": payload.get("seizure_gradient_path"),
            "source_task": payload.get("source_task"),
            "max_source_time_le_anchor": payload.get("max_source_time_le_anchor"),
            "gap_policy": payload.get("gap_policy"),
            "absolute_time_dtype": payload.get("absolute_time_dtype"),
            "current_observation_fresh": payload.get(
                "all_current_observations_fresh"
            ),
            "formal": payload.get("formal"),
            "sealed": payload.get("sealed"),
            "state_extraction_source_hash_match": (
                source_hashes.get("state_extraction.py") == sha256_file(state_source)
            ),
            "extract_states_source_hash_match": (
                source_hashes.get("extract_states.py") == sha256_file(runner_source)
            ),
        })
    return rows


def build_machine_audit(
    *, result_root: Path = RESULT_ROOT,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(result_root).resolve()
    repo = Path(repo_root or root.parents[4]).resolve()
    manifest_root = root / "manifests"
    checkpoint_path = manifest_root / "state_checkpoint_inventory.json"
    watch_path = manifest_root / "r1_7_watch.json"
    instrument_path = root / "reports/instrument_validation.json"
    risk_path = root / "fits/e384_instrument/risk_probe_machine_audit.json"

    checkpoints = _json(checkpoint_path) if checkpoint_path.exists() else {}
    checkpoint_rows = checkpoints.get("entries") or []
    checkpoint_hashes_ok = bool(checkpoint_rows) and all(
        Path(str(row.get("checkpoint_path", ""))).is_file()
        and sha256_file(Path(str(row["checkpoint_path"])))
        == row.get("checkpoint_sha256_expected")
        and row.get("state_frozen_before_seizure_task") is True
        and row.get("state_source_uses_seizure_labels") is False
        for row in checkpoint_rows
    )
    states = _state_manifests(root)
    state_ok = bool(states) and all(
        row["cache_hash_match"]
        and row["checkpoint_hash_match"]
        and row["state_frozen"]
        and row["all_parameters_frozen"]
        and row["seizure_gradient_path"] is False
        and row["source_task"] == "continuous_background_and_ied_timing_mark"
        and row["max_source_time_le_anchor"] is True
        and row["gap_policy"] == "reset_at_recorded_coverage_segment_start"
        and row["absolute_time_dtype"] == "float64"
        and row["formal"] is False
        and row["sealed"] is False
        and row["state_extraction_source_hash_match"]
        and row["extract_states_source_hash_match"]
        for row in states
    )
    freshness_values = [row["current_observation_fresh"] for row in states]
    freshness_ok = bool(freshness_values) and all(value is True for value in freshness_values)

    risk = _json(risk_path) if risk_path.exists() else {}
    risk_manifest_path = root / "risk_sets/e384_risk_sets.manifest.json"
    risk_manifest = _json(risk_manifest_path) if risk_manifest_path.exists() else {}
    wrong_risk_path = root / (
        "fits/e384_wrong_time_instrument/risk_probe_machine_audit.json"
    )
    wrong_risk = _json(wrong_risk_path) if wrong_risk_path.exists() else {}
    split_ids = risk.get("train_select_test_seizure_ids") or {}
    split_audit = split_contract_audit(split_ids)
    risk_ok = bool(risk) and (
        risk.get("identical_risk_sets_across_arms") is True
        and risk.get("regularization_selected_only_on_train_select") is True
        and risk.get("seed_is_patient_replicate") is False
        and risk.get("lead_to_split_consistency") is True
        and split_audit["pass"]
        and risk_manifest.get("risk_set_hash") == risk.get("risk_set_hash")
        and risk_manifest.get("risk_table_sha256") == sha256_file(
            root / "risk_sets/e384_risk_sets.csv"
        )
    )
    wrong_risk_ok = bool(wrong_risk) and (
        wrong_risk.get("identical_risk_sets_across_arms") is True
        and wrong_risk.get(
            "wrong_time_donors_same_patient_segment_and_exclusion_clear"
        ) is True
        and "wrong_time" in (wrong_risk.get("arms") or [])
    )

    phenotype_path = root / (
        "fits/e384_phenotype_not_estimable/phenotype_transfer_machine_audit.json"
    )
    phenotype = _json(phenotype_path) if phenotype_path.exists() else {}
    phenotype_availability_path = root / "reports/e384_phenotype_availability.json"
    phenotype_availability = (
        _json(phenotype_availability_path)
        if phenotype_availability_path.exists() else {}
    )
    phenotype_boundary_ok = (
        phenotype.get("status") == "NOT_ESTIMABLE_NO_USABLE_FROZEN_TARGET"
        and phenotype.get("target_reclustered") is False
        and phenotype_availability.get("status")
        == "NOT_ESTIMABLE_FROZEN_TARGET_UNAVAILABLE"
        and phenotype_availability.get("replacement_target_invented") is False
    )

    instrument = _json(instrument_path) if instrument_path.exists() else {}
    positive_ok = (instrument.get("positive_synthetic") or {}).get("status") == "PASS"
    permutation = instrument.get("time_label_permutation") or {}
    permutation_ok = (
        permutation.get("status") == "PASS"
        and np.isfinite(float(permutation.get("null_median", np.nan)))
    )
    causality_ok = (instrument.get("causality_perturbation") or {}).get("status") == "PASS"

    watcher = _json(watch_path) if watch_path.exists() else {}
    r1_7_ready = watcher.get("r1_7_outputs_authorized_for_h2b") is True
    r1_7_use = _json(manifest_root / "r1_7_import.json") \
        if (manifest_root / "r1_7_import.json").exists() else None
    incomplete_r1_7_not_used = r1_7_ready or r1_7_use is None
    r1_7_availability_path = root / "reports/r1_7_availability.json"
    r1_7_availability = (
        _json(r1_7_availability_path) if r1_7_availability_path.exists() else {}
    )
    r1_7_boundary_ok = (
        (r1_7_ready and r1_7_availability.get("status") == "READY_FOR_IMPORT")
        or (
            not r1_7_ready
            and r1_7_availability.get("status") == "UNAVAILABLE_NOT_USED"
            and r1_7_availability.get("r1_7_outputs_referenced_by_h2b") is False
        )
    )

    changed = _git_changed_paths(repo)
    paper_paths = [path for path in changed if (
        "paper-ready-figure" in path or path.startswith("scripts/paper_figures/")
    )]
    h3_t2_paths = [path for path in changed if (
        "/t2" in path.lower() or "h3" in Path(path).name.lower()
    )]
    boundary = asdict(RunBoundary())

    checks = [
        _check("checkpoint_frozen_before_seizure_task_and_hash_recomputable",
               checkpoint_hashes_ok, evidence=checkpoint_rows),
        _check("state_source_only_interictal_and_no_seizure_gradient", state_ok,
               evidence=states),
        _check("anchor_after_data_excluded_and_gap_reset", state_ok,
               evidence={"state_manifests": len(states)}),
        _check("current_observation_is_fresh_not_stale_cache", freshness_ok,
               evidence=freshness_values),
        _check("crosswalk_unique_and_exact", (
            (manifest_root / "seizure_crosswalk.csv").exists()
            and (_json(manifest_root / "exclusion_funnel.json")
                 .get("seizure_crosswalk", {}).get("n_unmatched") == 0)
            and (_json(manifest_root / "exclusion_funnel.json")
                 .get("seizure_crosswalk", {}).get("n_ambiguous") == 0)
        ) if (manifest_root / "exclusion_funnel.json").exists() else False),
        _check("train_select_test_seizures_disjoint_and_all_leads_same_split",
               split_audit["pass"] and bool(split_ids), evidence=split_audit),
        _check("same_risk_sets_all_arms_and_patient_first_seed_aggregation",
               risk_ok, evidence=risk.get("risk_set_hash")),
        _check("wrong_time_direct_comparison_uses_strict_shared_subset",
               wrong_risk_ok, evidence={
                   "arms": wrong_risk.get("arms"),
                   "risk_set_hash": wrong_risk.get("risk_set_hash"),
               }),
        _check("phenotype_target_boundary_fails_closed_without_reclustering",
               phenotype_boundary_ok, evidence=phenotype_availability),
        _check("positive_synthetic_recovers_state_increment", positive_ok,
               evidence=instrument.get("positive_synthetic")),
        _check("time_label_permutation_returns_increment_near_zero", permutation_ok,
               evidence=permutation),
        _check("future_data_perturbation_leaves_past_state_bitwise_unchanged",
               causality_ok, evidence=instrument.get("causality_perturbation")),
        _check("formal_and_sealed_false", (
            boundary["formal_test_partition_opened"] is False
            and boundary["sealed_opened"] is False
        ), evidence=boundary),
        _check("incomplete_r1_7_outputs_not_imported",
               incomplete_r1_7_not_used and r1_7_boundary_ok,
               evidence={"watch_status": watcher.get("status"),
                         "fit_result_count": watcher.get("fit_result_count"),
                         "import_manifest": r1_7_use}),
        _check("t2_h3_and_physical_clock_not_run", (
            boundary["h3_or_t2_run"] is False
            and boundary["physical_clock_run"] is False
            and not h3_t2_paths
        ), evidence=h3_t2_paths),
        _check("paper_ready_figures_not_modified", (
            boundary["paper_ready_figures_modified"] is False and not paper_paths
        ), evidence=paper_paths),
    ]
    failed = [row["name"] for row in checks if row["status"] == "FAIL"]
    status = "COMPLETE" if not failed else "INCOMPLETE"
    support_path = manifest_root / "seizure_support_by_lead.csv"
    return {
        "status": status,
        "audit_revision": AUDIT_REVISION,
        "contract": H2B_REVISION,
        "created_utc": utc_now(),
        "scientific_scope": (
            "development E384 instrument only" if not r1_7_ready
            else "development checkpoint-available cohort"
        ),
        "scientific_claim_eligible": bool(status == "COMPLETE" and r1_7_ready),
        "r1_7_integration_status": "READY" if r1_7_ready else "UNAVAILABLE_NOT_USED",
        "boundary": boundary,
        "checks": checks,
        "failed_checks": failed,
        "counts": {
            "checkpoint_inventory_rows": len(checkpoint_rows),
            "state_cache_manifests": len(states),
        },
        "source_artifacts": {
            "checkpoint_inventory": str(checkpoint_path),
            "checkpoint_inventory_sha256": (
                sha256_file(checkpoint_path) if checkpoint_path.exists() else None
            ),
            "seizure_support_by_lead": str(support_path),
            "seizure_support_by_lead_sha256": (
                sha256_file(support_path) if support_path.exists() else None
            ),
            "risk_probe_audit": str(risk_path),
            "risk_probe_audit_sha256": sha256_file(risk_path) if risk_path.exists() else None,
            "wrong_time_risk_probe_audit": str(wrong_risk_path),
            "wrong_time_risk_probe_audit_sha256": (
                sha256_file(wrong_risk_path) if wrong_risk_path.exists() else None
            ),
            "phenotype_availability": str(phenotype_availability_path),
            "phenotype_availability_sha256": (
                sha256_file(phenotype_availability_path)
                if phenotype_availability_path.exists() else None
            ),
            "instrument_validation": str(instrument_path),
            "instrument_validation_sha256": (
                sha256_file(instrument_path) if instrument_path.exists() else None
            ),
            "r1_7_watch": str(watch_path),
            "r1_7_watch_sha256": sha256_file(watch_path) if watch_path.exists() else None,
            "r1_7_availability": str(r1_7_availability_path),
            "r1_7_availability_sha256": (
                sha256_file(r1_7_availability_path)
                if r1_7_availability_path.exists() else None
            ),
        },
        "changed_paths_from_base": changed,
    }
