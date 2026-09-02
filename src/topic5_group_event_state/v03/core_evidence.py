"""Stable figure payload for the group-event-state scientific story.

The plotting contract deliberately separates current pilot measurements from
future H2b/H3 slots.  Empty future slots are represented explicitly rather
than filled with simulated values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math


FORMAT = "group_event_state_core_evidence_v1"
HORIZON_KEYS = ("300s", "1800s", "7200s")
HORIZON_MINUTES = (5, 30, 120)
MARK_ENDPOINTS = ("continue", "positive_size", "subset")


@dataclass(frozen=True)
class FigurePaths:
    payload: Path
    metadata: Path
    figure_dir: Path


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _skill(delta_state_minus_control: Any, scale: Any = 1.0) -> float | None:
    """Convert loss contrast to a paper-facing gain (positive is favourable)."""
    delta = _finite(delta_state_minus_control)
    denominator = _finite(scale)
    if delta is None or denominator is None or denominator <= 0:
        return None
    return -delta / denominator


def build_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("format") != "group_event_state_v0_3_pilot_summary":
        raise ValueError("not a group-event-state v0.3 pilot summary")
    subjects = list(summary["subjects"])
    aliases = {subject: f"P{i + 1}" for i, subject in enumerate(subjects)}
    h1_rows: list[dict[str, Any]] = []
    h2a_rows: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for subject in subjects:
        subject_row = summary["per_subject"][subject]
        training.append(
            {
                "subject": subject,
                "alias": aliases[subject],
                "optimization_status": subject_row["optimization_status"],
                "selected_epochs": subject_row["selected_epochs"],
                "n_seeds": subject_row["n_seeds"],
            }
        )
        for key, minutes in zip(HORIZON_KEYS, HORIZON_MINUTES):
            row = subject_row["horizons"][key]
            intercept = row.get("intercept_poisson_nll")
            h1_rows.append(
                {
                    "subject": subject,
                    "alias": aliases[subject],
                    "horizon_minutes": minutes,
                    "count_gain_over_multiscale": _skill(
                        row.get("count_correct_minus_multiscale"), intercept
                    ),
                    "correct_time_gain_over_shifted": _skill(
                        row.get("count_correct_minus_shifted"), intercept
                    ),
                    "count_pair_seeds_multiscale": row["count_pair_estimable_seeds"][
                        "correct_vs_multiscale"
                    ],
                    "count_pair_seeds_shifted": row["count_pair_estimable_seeds"][
                        "correct_vs_shifted"
                    ],
                    "n_anchors": row["n_development_test_anchors"],
                    "coverage_status": (
                        "insufficient_coverage"
                        if row["n_insufficient_coverage_seeds"]
                        else "ok"
                    ),
                }
            )
            for endpoint in MARK_ENDPOINTS:
                h2a_rows.append(
                    {
                        "subject": subject,
                        "alias": aliases[subject],
                        "horizon_minutes": minutes,
                        "endpoint": endpoint,
                        "gain_over_shifted": _skill(
                            row.get(f"{endpoint}_correct_minus_shifted")
                        ),
                        "n_seeds": row["n_seeds"],
                        "coverage_status": (
                            "insufficient_coverage"
                            if row["n_insufficient_coverage_seeds"]
                            else "ok"
                        ),
                    }
                )
    payload = {
        "format": FORMAT,
        "status": "development_pilot",
        "source": {
            "summary_format": summary["format"],
            "source_commit": summary["source_commit"],
            "sealed_partition_opened": False,
            "model_layer_nested": bool(
                summary["nested_source_audit"]["model_layer_nested_contract"]
            ),
            "measurement_layer_nested": bool(
                summary["nested_source_audit"]["measurement_layer_nested_contract"]
            ),
        },
        "horizons_minutes": list(HORIZON_MINUTES),
        "training": training,
        "h1_future_block": {
            "status": "measured",
            "rows": h1_rows,
            "gain_definition": (
                "(control Poisson NLL - correct-state Poisson NLL) / "
                "fitted-intercept Poisson NLL; positive favours state"
            ),
        },
        "h2a_repertoire": {
            "status": "measured",
            "rows": h2a_rows,
            "gain_definition": (
                "wrong-time mark NLL - correct-time mark NLL; "
                "positive favours correct-time state"
            ),
            "same_prefix": {
                "status": "not_yet_run",
                "rows": [],
                "required_fields": [
                    "subject",
                    "horizon_minutes",
                    "prefix_definition",
                    "state_gain_nll",
                    "n_prefix_pairs",
                ],
            },
        },
        "h2b_transfer": {
            "status": "not_yet_run",
            "risk_rows": [],
            "field_rows": [],
            "risk_required_fields": [
                "subject",
                "lead_minutes",
                "brier_skill_state_over_history",
                "log_score_gain",
                "n_heldout_seizures",
            ],
            "field_required_fields": [
                "subject",
                "lead_minutes",
                "early_ictal_field_gain",
                "path_gain",
                "n_heldout_seizures",
            ],
        },
        "h3_feedback": {
            "status": "not_yet_run",
            "model_rows": [],
            "impulse_rows": [],
            "model_required_fields": [
                "subject",
                "feedback_family",
                "future_block_log_score_gain_over_no_feedback",
            ],
            "impulse_required_fields": [
                "event_type",
                "lag_minutes",
                "functional_state_change",
                "lower",
                "upper",
            ],
        },
        "claim_boundary": {
            "h1_h2a": (
                "current three-patient development data do not establish a "
                "replicable slow predictive state"
            ),
            "h2b_h3": "not yet run; figure slots contain no synthetic observations",
            "upstream_measurement": (
                "legacy full-record contact selection remains transductive"
            ),
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != FORMAT:
        raise ValueError(f"payload format must be {FORMAT!r}")
    if tuple(payload.get("horizons_minutes", [])) != HORIZON_MINUTES:
        raise ValueError("primary physical horizons must remain 5/30/120 min")
    if payload["source"].get("sealed_partition_opened"):
        raise ValueError("sealed partition must remain closed for this figure package")
    for section in ("h1_future_block", "h2a_repertoire", "h2b_transfer", "h3_feedback"):
        if section not in payload:
            raise ValueError(f"missing figure section {section}")
    for row in payload["h1_future_block"]["rows"]:
        if row["horizon_minutes"] not in HORIZON_MINUTES:
            raise ValueError("unexpected H1 horizon")
    for row in payload["h2a_repertoire"]["rows"]:
        if row["endpoint"] not in MARK_ENDPOINTS:
            raise ValueError("unexpected H2a endpoint")


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    validate_payload(payload)
    return payload


def write_payload(summary_path: Path, output_path: Path) -> dict[str, Any]:
    summary = json.loads(Path(summary_path).read_text())
    payload = build_payload(summary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    tmp.replace(output_path)
    return payload
