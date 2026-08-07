#!/usr/bin/env python3
"""Phase-0 audit for the Topic-5 Shared Propagation Field RNN."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import (  # noqa: E402
    CONTRACT_NAME,
    audit_legacy_snn_lagpat,
    audit_subject,
    load_frozen_cohort,
    sha256_file,
)


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
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"min": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan}
    quantile = np.quantile(array, [0.0, 0.25, 0.50, 0.75, 1.0])
    return {
        key: float(value)
        for key, value in zip(["min", "q25", "median", "q75", "max"], quantile)
    }


def _select_target_blind_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select six pilots without A/B, axis, SOZ, or heldout outcomes.

    Each dataset contributes low-, middle-, and high-precedence-entropy
    subjects among those above its median event count.  This gives a
    target-blind repertoire spread while avoiding tiny engineering cases.
    """
    selected = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        pool = [row for row in rows if row["dataset"] == dataset]
        event_floor = float(np.median([row["n_events"] for row in pool]))
        eligible = [
            row
            for row in pool
            if row["n_events"] >= event_floor
            and np.isfinite(float(row["precedence_entropy_train80"]))
        ]
        eligible.sort(
            key=lambda row: (
                float(row["precedence_entropy_train80"]),
                -int(row["n_events"]),
                str(row["subject"]),
            )
        )
        positions = sorted({0, len(eligible) // 2, len(eligible) - 1})
        labels = ["low", "middle", "high"]
        for label, position in zip(labels, positions):
            row = eligible[position]
            selected.append(
                {
                    "subject": row["subject"],
                    "dataset": dataset,
                    "target_blind_stratum": label,
                    "n_events": row["n_events"],
                    "n_contacts": row["n_contacts"],
                    "precedence_entropy_train80": row[
                        "precedence_entropy_train80"
                    ],
                    "selection_inputs": "dataset,n_events,train80_precedence_entropy",
                }
            )
    return selected


def _audit_snn(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(root.rglob("*_lagPat_withFreqCent.npz"))
    rows = []
    failures = []
    for path in paths:
        try:
            rows.append(audit_legacy_snn_lagpat(path))
        except Exception as exc:  # fail is recorded, never silently dropped
            failures.append({"path": str(path), "error": repr(exc)})
    event_counts = [int(row["n_events"]) for row in rows]
    summary = {
        "root": str(root),
        "n_artifacts_found": len(paths),
        "n_artifacts_valid": len(rows),
        "n_artifacts_failed": len(failures),
        "n_events_total_not_poolable_across_conditions": int(sum(event_counts)),
        "max_events_in_one_artifact": int(max(event_counts, default=0)),
        "artifacts_with_at_least_100_events": int(
            sum(count >= 100 for count in event_counts)
        ),
        "g0_dataset_status": "NOT_READY_AS_STANDARDIZED_POSITIVE_CONTROL",
        "event_count_threshold_status": (
            "NOT_DEFINED_PENDING_STANDARDIZED_SNN_LEARNING_CURVE"
        ),
        "reason": (
            "Existing files span heterogeneous SNN conditions and seeds; they "
            "must not be pooled into a positive-control cohort without a frozen "
            "condition/lesion/observation manifest. No event-count threshold is "
            "used as a substitute for that missing scientific contract."
        ),
        "failures": failures,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SPF-RNN human inputs and optional SNN rank artifacts"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/topic5_shared_propagation_field/phase0",
    )
    parser.add_argument(
        "--snn-root",
        type=Path,
        default=ROOT / "results/topic4_sef_hfo",
    )
    parser.add_argument(
        "--skip-snn",
        action="store_true",
        help="Skip read-only inventory of existing virtual-SEEG lagPat artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_frozen_cohort(args.dataset_dir)
    rows = [audit_subject(records[subject]) for subject in sorted(records)]
    pilot = _select_target_blind_pilot(rows)

    summary: dict[str, Any] = {
        "contract": CONTRACT_NAME,
        "status": "PHASE0_HUMAN_INPUT_PASS_STRUCTURAL_IDENTIFIABILITY_PENDING_G0",
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest_sha256": sha256_file(
            args.dataset_dir / "dataset_manifest.json"
        ),
        "n_subjects": len(rows),
        "n_epilepsiae": int(sum(row["dataset"] == "epilepsiae" for row in rows)),
        "n_yuquan": int(sum(row["dataset"] == "yuquan" for row in rows)),
        "n_events": int(sum(int(row["n_events"]) for row in rows)),
        "n_contacts": _finite_quantiles(row["n_contacts"] for row in rows),
        "events_per_subject": _finite_quantiles(row["n_events"] for row in rows),
        "rank_count": _finite_quantiles(row["rank_count_median"] for row in rows),
        "rank_set_size": _finite_quantiles(
            row["rank_set_size_median"] for row in rows
        ),
        "tied_rank_set_fraction": _finite_quantiles(
            row["tied_rank_set_fraction"] for row in rows
        ),
        "precedence_entropy_train80": _finite_quantiles(
            row["precedence_entropy_train80"] for row in rows
        ),
        "subjects_contract_pass": int(
            sum(row["rank_event_contract"] == "PASS" for row in rows)
        ),
        "subjects_with_repeated_first_rank_condition": int(
            sum(row["max_events_same_first_rank_set"] >= 2 for row in rows)
        ),
        "subjects_with_zero_support_inner_train_contacts": int(
            sum(row["n_zero_support_contacts_inner_train"] > 0 for row in rows)
        ),
        "subjects_with_full_support_inner_train_contacts": int(
            sum(row["n_full_support_contacts_inner_train"] > 0 for row in rows)
        ),
        "contact_repeat_contract": (
            "representation assigns at most one rank-group id per contact/event"
        ),
        "latency_contract": (
            "event_lag_raw is available as within-window spectrogram centroid "
            "time; exact peak time is not certified"
        ),
        "old_heldout20_status": "PREVIOUSLY_READ_NOT_CONFIRMATORY_FOR_RNNV2",
        "recommended_development_split": (
            "chronological inner train/validation inside old train80 only"
        ),
        "structural_n_min_status": "PENDING_STANDARDIZED_SNN_CALIBRATION",
        "pilot_selection": (
            "target-blind dataset-balanced train80 precedence-entropy strata"
        ),
    }

    output = args.output_dir
    _write_csv(output / "human_subject_audit.csv", rows)
    _write_csv(output / "pilot_subjects_target_blind.csv", pilot)
    _write_json(output / "human_input_audit.json", summary)

    if not args.skip_snn:
        snn_rows, snn_summary = _audit_snn(args.snn_root)
        _write_csv(output / "snn_legacy_artifact_audit.csv", snn_rows)
        _write_json(output / "snn_positive_control_readiness.json", snn_summary)
        summary["snn_positive_control"] = snn_summary
        _write_json(output / "human_input_audit.json", summary)

    _write_json(
        output / "PHASE0_STATE.json",
        {
            "contract": CONTRACT_NAME,
            "state": summary["status"],
            "human_input_pass": True,
            "g0_pass": False,
            "g1_pass": False,
            "g2_pass": False,
            "g3_pass": False,
            "claim": (
                "Data feasibility is established; no propagation-field or "
                "mechanism claim has been tested."
            ),
        },
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
