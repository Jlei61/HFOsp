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


FORMAT = "group_event_state_core_evidence_v2"
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


def build_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("format") != "group_event_state_v0_3_1_closeout_summary":
        raise ValueError("not a group-event-state v0.3.1 closeout summary")
    subjects = list(summary["subjects"])
    aliases = {subject: f"P{i + 1}" for i, subject in enumerate(subjects)}
    diagnostic_count_rows: list[dict[str, Any]] = []
    diagnostic_mark_rows: list[dict[str, Any]] = []
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
            diagnostic_count_rows.append(
                {
                    "subject": subject,
                    "alias": aliases[subject],
                    "horizon_minutes": minutes,
                    "state_alone_minus_history_nll": _finite(
                        row.get("count_correct_minus_multiscale")
                    ),
                    "correct_minus_shifted_nll": _finite(
                        row.get("count_correct_minus_shifted")
                    ),
                    "count_pair_scored_seeds_multiscale": row[
                        "count_pair_scored_seeds"
                    ][
                        "correct_vs_multiscale"
                    ],
                    "count_pair_scored_seeds_shifted": row[
                        "count_pair_scored_seeds"
                    ][
                        "correct_vs_shifted"
                    ],
                    "posthoc_flagged_seeds_multiscale": row[
                        "count_pair_posthoc_flagged_seeds"
                    ]["correct_vs_multiscale"],
                    "posthoc_flagged_seeds_shifted": row[
                        "count_pair_posthoc_flagged_seeds"
                    ]["correct_vs_shifted"],
                    "n_anchors": row["n_development_test_anchors"],
                    "coverage_status": (
                        "insufficient_coverage"
                        if row["n_insufficient_coverage_seeds"]
                        else "ok"
                    ),
                }
            )
            for endpoint in MARK_ENDPOINTS:
                diagnostic_mark_rows.append(
                    {
                        "subject": subject,
                        "alias": aliases[subject],
                        "horizon_minutes": minutes,
                        "endpoint": endpoint,
                        "correct_minus_shifted_nll": _finite(
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
        "status": "v0_3_1_closed_major_revision",
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
        "v0_3_1_diagnostics": {
            "status": "archival_not_primary_estimand",
            "count_rows": diagnostic_count_rows,
            "mark_rows": diagnostic_mark_rows,
            "reason": (
                "v0.3.1 measured S alone versus H and S versus shifted S; "
                "it did not measure residual H+S contrasts"
            ),
        },
        "h1_future_block": {
            "status": "not_yet_run",
            "rows": [],
            "gain_definition": (
                "control negative-binomial NLL minus H+S_correct NLL; "
                "positive favours residual state"
            ),
            "required_fields": [
                "subject",
                "horizon_minutes",
                "residual_gain_over_history",
                "correct_time_gain_over_shifted",
                "dynamic_gain_over_mean",
                "n_score_blocks",
            ],
        },
        "h2a_repertoire": {
            "status": "not_yet_run",
            "rows": [],
            "gain_definition": (
                "best of H, H+S_shifted and H+S_mean NLL minus "
                "H+S_correct NLL; positive favours dynamic correct-time state"
            ),
            "required_fields": [
                "subject",
                "horizon_minutes",
                "endpoint",
                "gain_over_best_control",
                "gain_over_history",
                "gain_over_shifted",
                "gain_over_mean",
                "n_score_blocks",
            ],
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
                "v0.3.1 did not measure the residual H+S estimand; state "
                "learning is unresolved, not negative"
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
    if payload["status"] == "v0_3_1_closed_major_revision":
        if payload["h1_future_block"]["rows"] or payload["h2a_repertoire"]["rows"]:
            raise ValueError("v0.3.1 diagnostics must not populate residual-state panels")


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
