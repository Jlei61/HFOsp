#!/usr/bin/env python3
"""Audit human rank events for Stable Interaction Graph identifiability.

The audit is target blind.  It reads only the frozen interictal rank-event
dataset and never touches geometry, clinical labels, ictal targets, or SNN
artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import (  # noqa: E402
    SubjectRankEvents,
    load_frozen_cohort,
    sha256_file,
)


CONTRACT = "topic5_stable_interaction_graph_identifiability_v2"
MIN_START_EVENTS = 20
MIN_INTERMEDIATE_SUPPORT = 20
MIN_SENDER_EXPOSURES = 20
MIN_EVAL_SUFFIX_DECISIONS = 100
MIN_REPEATED_START_GROUPS = 4


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _first_key(row: np.ndarray) -> str:
    packed = np.packbits(np.asarray(row, dtype=np.uint8))
    return hashlib.sha1(packed.tobytes()).hexdigest()[:16]


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if not counts.size:
        return float("nan")
    probability = counts / counts.sum()
    return float(-np.sum(probability * np.log2(probability)))


def _suffix_signature(event: np.ndarray) -> bytes:
    value = np.asarray(event, dtype=np.int16).copy()
    value[value == 0] = -1
    value[value > 0] -= 1
    return value.tobytes()


def _normalized_event_vector(event: np.ndarray, count: int) -> np.ndarray:
    value = np.full(event.shape, 1.25, dtype=np.float32)
    participant = event >= 0
    denominator = max(int(count) - 1, 1)
    value[participant] = event[participant].astype(np.float32) / denominator
    return value


def _sampled_within_start_distance(
    groups: np.ndarray,
    counts: np.ndarray,
    event_indices: np.ndarray,
    *,
    seed: int,
    max_events: int = 256,
    max_pairs: int = 512,
) -> float:
    indices = np.asarray(event_indices, dtype=int)
    if len(indices) < 2:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if len(indices) > max_events:
        indices = np.sort(rng.choice(indices, size=max_events, replace=False))
    vectors = np.stack(
        [_normalized_event_vector(groups[index], counts[index]) for index in indices]
    )
    left = rng.integers(0, len(indices), size=max_pairs)
    right = rng.integers(0, len(indices), size=max_pairs)
    valid = left != right
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(vectors[left[valid]] - vectors[right[valid]])))


def _timing_audit(record: SubjectRankEvents) -> dict[str, Any]:
    with np.load(record.path, allow_pickle=False) as artifact:
        if "event_lag_raw" not in artifact.files:
            return {
                "lag_raw_available": False,
                "lag_raw_shape_aligned": False,
                "lag_raw_participant_finite_fraction": float("nan"),
                "lag_raw_nonparticipant_nan_fraction": float("nan"),
                "lag_raw_within_event_span_median": float("nan"),
            }
        lag = np.asarray(artifact["event_lag_raw"], dtype=float)
        participation = record.group_ids >= 0
        aligned = lag.shape == record.group_ids.shape
        if not aligned:
            return {
                "lag_raw_available": True,
                "lag_raw_shape_aligned": False,
                "lag_raw_participant_finite_fraction": float("nan"),
                "lag_raw_nonparticipant_nan_fraction": float("nan"),
                "lag_raw_within_event_span_median": float("nan"),
            }
        spans = []
        for event_lag, mask in zip(lag, participation):
            values = event_lag[mask & np.isfinite(event_lag)]
            if values.size >= 2:
                spans.append(float(np.max(values) - np.min(values)))
        return {
            "lag_raw_available": True,
            "lag_raw_shape_aligned": True,
            "lag_raw_participant_finite_fraction": float(
                np.mean(np.isfinite(lag[participation]))
            ),
            "lag_raw_nonparticipant_nan_fraction": float(
                np.mean(~np.isfinite(lag[~participation]))
                if np.any(~participation)
                else 1.0
            ),
            "lag_raw_within_event_span_median": float(
                np.median(spans) if spans else np.nan
            ),
        }


def audit_subject(record: SubjectRankEvents) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train, validation, test = record.development_split(0.15, 0.15)
    groups = record.group_ids
    counts = record.group_count
    first = groups == 0
    train_groups = groups[train]
    train_counts = counts[train]
    train_first = first[train]
    keys = np.asarray([_first_key(row) for row in train_first])

    start_rows: list[dict[str, Any]] = []
    unseen_candidates = []
    for ordinal, key in enumerate(sorted(set(keys))):
        local = np.flatnonzero(keys == key)
        absolute = train[local]
        start_contacts = np.flatnonzero(train_first[local[0]])
        signatures = [_suffix_signature(groups[index]) for index in absolute]
        _, signature_counts = np.unique(signatures, return_counts=True)
        remaining = np.setdiff1d(train, absolute, assume_unique=True)
        remaining_intermediate = np.sum(groups[remaining] > 0, axis=0)
        intermediate_ok = bool(
            np.all(
                remaining_intermediate[start_contacts]
                >= MIN_INTERMEDIATE_SUPPORT
            )
        )
        qualifies = bool(
            len(absolute) >= MIN_START_EVENTS and intermediate_ok
        )
        if qualifies:
            unseen_candidates.append(key)
        start_rows.append(
            {
                "subject": record.subject,
                "dataset": record.dataset,
                "start_key": key,
                "start_contacts": "|".join(
                    map(str, record.contact_names[start_contacts])
                ),
                "n_events_inner_train": int(len(absolute)),
                "n_unique_suffixes": int(len(signature_counts)),
                "suffix_unique_fraction": float(
                    len(signature_counts) / max(len(absolute), 1)
                ),
                "suffix_entropy_bits": _entropy_from_counts(signature_counts),
                "sampled_suffix_distance": _sampled_within_start_distance(
                    groups,
                    counts,
                    absolute,
                    seed=20260731 + ordinal,
                ),
                "start_contacts_intermediate_support_min_after_holdout": int(
                    np.min(remaining_intermediate[start_contacts])
                ),
                "unseen_start_candidate": qualifies,
            }
        )

    sender = (train_groups >= 0) & (
        train_groups < (train_counts[:, None] - 1)
    )
    sender_exposure = np.sum(sender, axis=0)
    intermediate = np.sum(train_groups > 0, axis=0)
    start_exposure = np.sum(train_first, axis=0)
    half = len(train) // 2
    early_decisions = int(np.sum(counts[train[:half]] - 1))
    late_decisions = int(np.sum(counts[train[half:]] - 1))
    validation_decisions = int(np.sum(counts[validation] - 1))
    test_decisions = int(np.sum(counts[test] - 1))
    repeated_groups = int(
        sum(row["n_events_inner_train"] >= MIN_START_EVENTS for row in start_rows)
    )
    eval_support = bool(
        validation_decisions >= MIN_EVAL_SUFFIX_DECISIONS
        and test_decisions >= MIN_EVAL_SUFFIX_DECISIONS
    )
    sender_support = bool(
        np.mean(sender_exposure >= MIN_SENDER_EXPOSURES) >= 0.80
    )
    unseen_eligible = bool(
        repeated_groups >= MIN_REPEATED_START_GROUPS
        and len(unseen_candidates) >= 1
        and sender_support
        and eval_support
    )
    timing = _timing_audit(record)
    eligible_start_rows = [
        row for row in start_rows if row["n_events_inner_train"] >= MIN_START_EVENTS
    ]
    weights = np.asarray(
        [row["n_events_inner_train"] for row in eligible_start_rows], dtype=float
    )

    def weighted(name: str) -> float:
        values = np.asarray([row[name] for row in eligible_start_rows], dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            return float("nan")
        return float(np.average(values[valid], weights=weights[valid]))

    row = {
        "contract": CONTRACT,
        "subject": record.subject,
        "dataset": record.dataset,
        "input_sha256": record.input_sha256,
        "n_contacts": int(groups.shape[1]),
        "n_events": int(len(groups)),
        "n_inner_train_events": int(len(train)),
        "n_inner_validation_events": int(len(validation)),
        "n_inner_test_events": int(len(test)),
        "n_inner_train_suffix_decisions": int(np.sum(train_counts - 1)),
        "n_validation_suffix_decisions": validation_decisions,
        "n_test_suffix_decisions": test_decisions,
        "n_unique_first_rank_groups_inner_train": int(len(start_rows)),
        "n_repeated_start_groups_ge20": repeated_groups,
        "max_events_same_start_inner_train": int(
            max((row["n_events_inner_train"] for row in start_rows), default=0)
        ),
        "within_start_suffix_unique_fraction_weighted": weighted(
            "suffix_unique_fraction"
        ),
        "within_start_suffix_entropy_bits_weighted": weighted(
            "suffix_entropy_bits"
        ),
        "within_start_suffix_distance_weighted": weighted(
            "sampled_suffix_distance"
        ),
        "start_contact_coverage_fraction": float(np.mean(start_exposure > 0)),
        "intermediate_contact_coverage_fraction": float(np.mean(intermediate > 0)),
        "sender_contact_coverage_ge20_fraction": float(
            np.mean(sender_exposure >= MIN_SENDER_EXPOSURES)
        ),
        "sender_exposure_min": int(np.min(sender_exposure)),
        "sender_exposure_median": float(np.median(sender_exposure)),
        "intermediate_exposure_min": int(np.min(intermediate)),
        "early_half_suffix_decisions": early_decisions,
        "late_half_suffix_decisions": late_decisions,
        "n_unseen_start_candidates": int(len(unseen_candidates)),
        "unseen_start_eligible": unseen_eligible,
        "generation_adequacy_eligible": eval_support,
        "rank_count_median": float(np.median(counts)),
        "rank_count_q90": float(np.quantile(counts, 0.90)),
        "rank_cardinality_median": float(
            np.median(
                [
                    np.sum(event == step)
                    for event, count in zip(groups, counts)
                    for step in range(int(count))
                ]
            )
        ),
        **timing,
        "lag_raw_semantics": (
            "stored continuous within-event lagPatRaw in seconds; legacy "
            "spectrogram-centroid timing, not certified contact peak time"
        ),
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
    }
    return row, start_rows


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {}
    q = np.quantile(array, [0, 0.25, 0.5, 0.75, 1])
    return {
        key: float(value)
        for key, value in zip(("min", "q25", "median", "q75", "max"), q)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_stable_interaction_graph/development"
        / "identifiability_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_frozen_cohort(args.dataset_dir)
    rows = []
    starts = []
    for subject in sorted(records):
        row, subject_starts = audit_subject(records[subject])
        rows.append(row)
        starts.extend(subject_starts)

    summary = {
        "contract": CONTRACT,
        "status": "COMPLETE",
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest_sha256": sha256_file(
            args.dataset_dir / "dataset_manifest.json"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "n_subjects": len(rows),
        "n_generation_adequacy_eligible": int(
            sum(row["generation_adequacy_eligible"] for row in rows)
        ),
        "n_unseen_start_eligible": int(
            sum(row["unseen_start_eligible"] for row in rows)
        ),
        "eligible_by_dataset": {
            dataset: {
                "n_subjects": int(sum(row["dataset"] == dataset for row in rows)),
                "generation": int(
                    sum(
                        row["dataset"] == dataset
                        and row["generation_adequacy_eligible"]
                        for row in rows
                    )
                ),
                "unseen_start": int(
                    sum(
                        row["dataset"] == dataset
                        and row["unseen_start_eligible"]
                        for row in rows
                    )
                ),
            }
            for dataset in sorted({row["dataset"] for row in rows})
        },
        "events": _quantiles(row["n_events"] for row in rows),
        "contacts": _quantiles(row["n_contacts"] for row in rows),
        "inner_train_suffix_decisions": _quantiles(
            row["n_inner_train_suffix_decisions"] for row in rows
        ),
        "within_start_suffix_distance": _quantiles(
            row["within_start_suffix_distance_weighted"] for row in rows
        ),
        "timing_contract": {
            "subjects_with_aligned_event_lag_raw": int(
                sum(row["lag_raw_shape_aligned"] for row in rows)
            ),
            "primary_use": "rank step only",
            "sensitivity_status": (
                "continuous lag is available but is legacy spectrogram-centroid "
                "timing, not certified contact peak time"
            ),
        },
        "eligibility_thresholds": {
            "min_start_events": MIN_START_EVENTS,
            "min_repeated_start_groups": MIN_REPEATED_START_GROUPS,
            "min_intermediate_support": MIN_INTERMEDIATE_SUPPORT,
            "min_sender_exposures": MIN_SENDER_EXPOSURES,
            "min_evaluation_suffix_decisions": MIN_EVAL_SUFFIX_DECISIONS,
        },
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
    }
    state = {
        "contract": CONTRACT,
        "state": (
            "PASS_DATA_AUDIT_FOR_SYNTHETIC_CALIBRATION_AND_PILOT_PLANNING"
            if summary["n_generation_adequacy_eligible"] >= 6
            else "BLOCK_PILOT_INSUFFICIENT_EVENT_SUPPORT"
        ),
        "human_pilot_authorized": False,
        "authorization_condition": (
            "generic synthetic G0-A must pass before any human SIG fit"
        ),
        "claim": (
            "This audit establishes support and timing availability only; it "
            "does not establish a stable interaction graph."
        ),
    }
    output = args.output_dir
    _write_csv(output / "per_subject_identifiability.csv", rows)
    _write_csv(output / "per_start_group.csv", starts)
    _write_json(output / "cohort_identifiability.json", summary)
    _write_json(output / "AUDIT_STATE.json", state)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
