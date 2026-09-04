"""Endpoint-specific structural support; never infer missing measurement fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


SPATIAL_ARRAYS = ("participation", "relative_delay", "tied_group_id", "contact_ok")
FREQUENCY_WAVEFORM_ARRAYS = (
    "participation", "band_features", "band_envelope", "cross_band_lag", "waveform", "contact_ok"
)


def audit_array_capabilities(
    dataset_dir: Path,
    index: Mapping[str, Any],
    *,
    h2b_target_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check declared files and shapes without reading any event value."""

    arrays = index.get("arrays", {})
    n_events = int(index.get("n_events", -1))

    def _family(required: Sequence[str]) -> dict[str, Any]:
        missing, malformed = [], []
        for name in required:
            spec = arrays.get(name)
            if not isinstance(spec, Mapping):
                missing.append(name)
                continue
            path = Path(dataset_dir) / str(spec.get("file", ""))
            shape = spec.get("shape")
            if not path.is_file():
                missing.append(name)
            elif not isinstance(shape, list) or not shape or int(shape[0]) != n_events:
                malformed.append(name)
        return {
            "available": not missing and not malformed,
            "required_arrays": list(required),
            "missing_arrays": missing,
            "malformed_arrays": malformed,
            "measurement_scope": "interictal_group_events_only",
        }

    early_field: dict[str, Any]
    if h2b_target_record is None:
        early_field = {
            "available": False,
            "status": "not_yet_measurable",
            "reason": "no versioned v0.3.4 ictal target registry/crosswalk was supplied; interictal waveform.npy is not an ictal target",
        }
    else:
        target_path = h2b_target_record.get("target_file") or h2b_target_record.get("early_ictal_field_file")
        n_eligible = h2b_target_record.get("n_eligible_seizures")
        present = bool(target_path) and Path(str(target_path)).is_file()
        valid_count = isinstance(n_eligible, int) and n_eligible >= 0
        early_field = {
            "available": bool(present and valid_count),
            "status": "measurement_available" if present and valid_count else "not_yet_measurable",
            "target_file": str(target_path) if target_path else None,
            "n_eligible_seizures": n_eligible if valid_count else None,
            "reason": None if present and valid_count else "target registry entry lacks an existing target file or a non-negative n_eligible_seizures",
        }

    return {
        "count": {"available": True, "measurement_scope": "coverage_and_event_timestamps"},
        "conditional_spatial_grammar_participation": _family(SPATIAL_ARRAYS),
        "multiband_waveform": _family(FREQUENCY_WAVEFORM_ARRAYS),
        "early_ictal_field": early_field,
    }


def endpoint_rows(
    *,
    subject: str,
    blocks_by_horizon: Mapping[int, Mapping[str, int]],
    prior_support: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    count_requirement_30m: int | None,
    grammar_requirement_30m: int | None,
) -> list[dict[str, Any]]:
    """Build support rows without converting unknown power into a made-up gate."""

    rows: list[dict[str, Any]] = []
    for horizon in (300, 1800, 7200, 21600):
        exploratory = horizon >= 7200
        blocks = int(blocks_by_horizon[horizon]["dev_test"])
        requirement = count_requirement_30m if horizon == 1800 else None
        status = "support_described_only"
        estimable = None
        if requirement is not None:
            estimable = blocks >= requirement
            status = "estimable" if estimable else "not_estimable"
        rows.append({
            "subject": subject,
            "endpoint": "future_event_count",
            "horizon_seconds": horizon,
            "core": not exploratory,
            "exploratory": exploratory,
            "available_independent_blocks": blocks,
            "required_independent_blocks": requirement,
            "estimable": estimable,
            "status": status,
            "reason": None if requirement is not None else "no calibrated endpoint-specific requirement; support only",
        })

        spatial_cap = bool(capabilities["conditional_spatial_grammar_participation"]["available"])
        requirement = grammar_requirement_30m if horizon == 1800 else None
        if not spatial_cap:
            spatial_status, spatial_estimable = "not_yet_measurable", False
        elif requirement is None:
            spatial_status, spatial_estimable = "support_described_only", None
        else:
            spatial_estimable = blocks >= requirement
            spatial_status = "estimable" if spatial_estimable else "not_estimable"
        rows.append({
            "subject": subject,
            "endpoint": "conditional_spatial_grammar_participation",
            "horizon_seconds": horizon,
            "core": horizon in (300, 1800),
            "exploratory": exploratory,
            "available_independent_blocks": blocks,
            "required_independent_blocks": requirement,
            "available_positive_anchors_prior_audit": prior_support.get("grammar_positive_anchors", {}).get(str(horizon), {}).get("development_evaluation"),
            "estimable": spatial_estimable,
            "status": spatial_status,
            "measurement_available": spatial_cap,
        })

        fw_cap = bool(capabilities["multiband_waveform"]["available"])
        rows.append({
            "subject": subject,
            "endpoint": "conditional_multiband_waveform",
            "horizon_seconds": horizon,
            "core": False,
            "exploratory": True,
            "available_independent_blocks": blocks,
            "required_independent_blocks": None,
            "estimable": None if fw_cap else False,
            "status": "support_described_only" if fw_cap else "not_yet_measurable",
            "measurement_available": fw_cap,
            "reason": "measurement fields exist, but no endpoint-specific power/assay requirement is calibrated" if fw_cap else "required interictal arrays missing",
        })

    n_seizures = int(prior_support.get("seizures", {}).get("development_evaluation", 0))
    rows.append({
        "subject": subject,
        "endpoint": "h2b_seizure_risk",
        "horizon_seconds": None,
        "core": True,
        "exploratory": False,
        "available_held_out_seizures": n_seizures,
        "required_held_out_seizures": None,
        "estimable": False if n_seizures == 0 else None,
        "status": "not_estimable" if n_seizures == 0 else "support_described_only",
        "reason": "zero seizures in the registered evaluation phase" if n_seizures == 0 else "seizure count known; risk-set assay requirement not yet calibrated",
    })
    early_cap = bool(capabilities["early_ictal_field"]["available"])
    rows.append({
        "subject": subject,
        "endpoint": "h2b_early_ictal_field",
        "horizon_seconds": None,
        "core": True,
        "exploratory": False,
        "available_held_out_seizures": n_seizures,
        "required_held_out_seizures": None,
        "estimable": None if early_cap else False,
        "status": "support_described_only" if early_cap else "not_yet_measurable",
        "reason": capabilities["early_ictal_field"].get("reason"),
    })
    return rows
