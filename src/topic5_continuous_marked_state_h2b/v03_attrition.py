"""Outcome-value-blind attrition census for H2b v0.3.

The census reuses v0.2 eligibility and execution metadata, but deliberately
does not read probe coefficients, losses, ranks, labels, or p-values.  It is a
denominator audit, not a re-analysis of the v0.2 outcome.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .contract import sha256_file, utc_now


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _cell_reason(entry: dict[str, Any], support: dict[str, Any], cache: bool) -> str:
    if not bool(entry.get("checkpoint_available")):
        return f"checkpoint_unavailable:{entry.get('analysis_status', 'unknown')}"
    if int(support.get("n_seizures_in_frozen_inventory", 0)) == 0:
        return "no_frozen_seizure_support"
    if int(support.get("primary_complete_coverage_seizures", 0)) == 0:
        return "no_primary_horizon_complete_coverage"
    if not bool(support.get("coverage_available")):
        return "coverage_unavailable"
    if not bool(support.get("upstream_design_available")):
        return "upstream_design_unavailable"
    if not bool(support.get("raw_inference_cache_available")):
        return "raw_inference_cache_unavailable"
    if not cache:
        return "eligible_cell_without_state_cache"
    return "state_cache_available"


def build_attrition_payload(v02_root: Path | str) -> dict[str, Any]:
    root = Path(v02_root).resolve()
    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    support_path = root / "manifests/support_census.json"
    machine_path = root / "reports/machine_audit.json"
    inventory = _json(inventory_path)
    support_payload = _json(support_path)
    machine = _json(machine_path)
    _require(inventory.get("status") == "COMPLETE", "v0.2 inventory is incomplete")
    _require(support_payload.get("status") == "COMPLETE",
             "v0.2 support census is incomplete")
    _require(machine.get("status") == "PASS_COMPLETE",
             "v0.2 machine audit is not PASS_COMPLETE")
    _require(machine.get("formal_test_partition_opened") is False,
             "v0.2 formal partition was opened")
    _require(machine.get("sealed_opened") is False,
             "v0.2 sealed partition was opened")

    support_by_subject = {
        str(row["subject"]): dict(row)
        for row in support_payload.get("patient_rows") or []
    }
    entries = list(inventory.get("entries") or [])
    _require(len(entries) == int(inventory.get("n_cells", -1)),
             "v0.2 cell denominator drift")
    cell_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    by_h1: dict[str, Counter[str]] = defaultdict(Counter)
    cached_subjects: set[str] = set()

    for entry in entries:
        subject = str(entry["subject"])
        seed = int(entry["seed"])
        _require(subject in support_by_subject,
                 f"support census lacks inventory subject {subject}")
        support = support_by_subject[subject]
        cache_path = root / "state_cache" / subject / f"seed_{seed}" / "states.npz"
        manifest_path = cache_path.with_suffix(".manifest.json")
        cache_present = cache_path.is_file() and manifest_path.is_file()
        cache_sha256 = None
        manifest_sha256 = None
        if cache_present:
            manifest = _json(manifest_path)
            _require(manifest.get("status") == "COMPLETE",
                     f"state cache manifest incomplete: {manifest_path}")
            observed_cache_hash = sha256_file(cache_path)
            _require(manifest.get("cache_sha256") == observed_cache_hash,
                     f"state cache hash drift: {cache_path}")
            _require(manifest.get("checkpoint_sha256") == entry.get("checkpoint_sha256"),
                     f"checkpoint lineage drift: {manifest_path}")
            _require(manifest.get("checkpoint_subject") == subject,
                     f"cache subject drift: {manifest_path}")
            _require(int(manifest.get("checkpoint_seed", -1)) == seed,
                     f"cache seed drift: {manifest_path}")
            cache_sha256 = observed_cache_hash
            manifest_sha256 = sha256_file(manifest_path)
            cached_subjects.add(subject)
        reason = _cell_reason(entry, support, cache_present)
        reasons[reason] += 1
        h1_group = "h1_stable" if bool(entry.get("h1_stable_subject")) else "h1_unstable"
        by_h1[h1_group]["total_cells"] += 1
        by_h1[h1_group]["checkpoint_available_cells"] += int(
            bool(entry.get("checkpoint_available"))
        )
        by_h1[h1_group]["state_cache_cells"] += int(cache_present)
        cell_rows.append({
            "subject": subject,
            "dataset": str(entry.get("dataset", "")),
            "seed": seed,
            "h1_stable_subject": bool(entry.get("h1_stable_subject")),
            "checkpoint_available": bool(entry.get("checkpoint_available")),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "state_cache_available": cache_present,
            "state_cache_path": str(cache_path) if cache_present else None,
            "state_cache_sha256": cache_sha256,
            "state_manifest_sha256": manifest_sha256,
            "n_frozen_seizures": int(support.get("n_seizures_in_frozen_inventory", 0)),
            "n_primary_complete_coverage_seizures": int(
                support.get("primary_complete_coverage_seizures", 0)
            ),
            "attrition_reason": reason,
        })

    contrast_rows: list[dict[str, Any]] = []
    contrast_counts: Counter[str] = Counter()
    fits_root = root / "fits/by_subject"
    if fits_root.is_dir():
        for path in sorted(fits_root.glob("*/*/risk_probe_machine_audit.json")):
            audit = _json(path)
            subject, analysis = path.relative_to(fits_root).parts[:2]
            execution = str(audit.get("execution_status", audit.get("status", "UNKNOWN")))
            estimability = str(audit.get("scientific_estimability", "UNSPECIFIED"))
            permutation = str((audit.get("time_label_permutation") or {}).get(
                "status", "UNSPECIFIED"
            ))
            contrast_counts[f"{analysis}:{estimability}"] += 1
            contrast_rows.append({
                "subject": subject,
                "analysis": analysis,
                "execution_status": execution,
                "scientific_estimability": estimability,
                "permutation_status": permutation,
                "audit_path": str(path),
                "audit_sha256": sha256_file(path),
            })

    n_checkpoint = sum(bool(row["checkpoint_available"]) for row in cell_rows)
    n_cache = sum(bool(row["state_cache_available"]) for row in cell_rows)
    _require(n_checkpoint == int(inventory.get("n_checkpoint_available_cells", -1)),
             "checkpoint-available denominator drift")
    _require(n_cache == int(machine.get("details", {}).get("n_state_cache_cells", -1)),
             "state-cache denominator drift")
    return {
        "status": "COMPLETE",
        "revision": "h2b_v0_3_attrition_audit_v1",
        "created_utc": utc_now(),
        "source_v0_2_root": str(root),
        "source_sha256": {
            "checkpoint_inventory": sha256_file(inventory_path),
            "support_census": sha256_file(support_path),
            "machine_audit": sha256_file(machine_path),
        },
        "outcome_values_read": False,
        "selection_assessment": (
            "denominator and estimability metadata only; no probe loss, coefficient, "
            "rank, p-value, or phenotype value was read"
        ),
        "funnel": {
            "total_r1_7b_cells": len(cell_rows),
            "checkpoint_available_cells": n_checkpoint,
            "state_cache_cells": n_cache,
            "state_cache_subjects": len(cached_subjects),
            "subjects_with_input_manifest": int(
                machine.get("details", {}).get("n_subjects_with_input_manifest", -1)
            ),
            "probe_tasks": len(contrast_rows),
        },
        "attrition_reason_counts": dict(sorted(reasons.items())),
        "by_preexisting_h1_stratum": {
            key: dict(value) for key, value in sorted(by_h1.items())
        },
        "contrast_estimability_counts": dict(sorted(contrast_counts.items())),
        "cell_rows": cell_rows,
        "contrast_rows": contrast_rows,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
