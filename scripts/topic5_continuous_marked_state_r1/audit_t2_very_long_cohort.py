#!/usr/bin/env python3
"""Parallel 34-subject audit of long contiguous IED support and T1 readiness."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import (
    load_full_admissible_event_stream,
    write_full_admissible_coverage,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable


SCALES = (1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 50000)
DELAY_EVENTS = 1000
REVISION = "t2_very_long_full_cohort_train_validation_contiguous_support_v2"


def _t1_status(subject: str) -> dict:
    formal = contract.RESULT_ROOT / "r1_3/fits" / subject
    extension = contract.RESULT_ROOT / "t2_long_total_effect/t1_r1_3/fits" / subject
    root = formal if formal.exists() else extension
    results = []
    if root.exists():
        for seed in range(10):
            path = root / f"explicit_seed_{seed}/result.json"
            if not path.exists():
                continue
            value = json.loads(path.read_text())
            results.append(value)
    selected = [
        int(value["fit_trace"]["selected_total_epoch"]) for value in results
        if value.get("status") == "COMPLETE" and value.get("sealed_opened") is False
    ]
    return {
        "t1_source": (
            "formal_r1_3" if formal.exists() else
            "long_extension" if extension.exists() else "none"
        ),
        "t1_completed_seeds": int(len(selected)),
        "t1_selected_epoch_positive_seeds": int(sum(value > 0 for value in selected)),
        "t1_ready": bool(selected and all(value > 0 for value in selected)),
    }


def _subject(subject: str) -> dict:
    coverage_path = contract.RESULT_ROOT / "r1_2/coverage" / f"{subject}.npz"
    manifest_path = coverage_path.with_suffix(".manifest.json")
    if not coverage_path.exists() or not manifest_path.exists():
        coverage_meta = write_full_admissible_coverage(subject)
    else:
        coverage_meta = json.loads(manifest_path.read_text())
    coverage = CoverageTable.load(coverage_path)
    stream = load_full_admissible_event_stream(subject, coverage)
    time = np.asarray(stream.event_time, dtype=np.float64)
    split = np.asarray(stream.split, dtype=np.int8)
    segment = np.searchsorted(coverage.stop, time, side="right")
    if np.any(segment >= len(coverage.start)):
        raise ValueError(f"{subject}: event after final coverage")
    inside = (time >= coverage.start[segment]) & (time < coverage.stop[segment])
    if not bool(inside.all()):
        raise ValueError(f"{subject}: event outside coverage")
    validation = split == 1
    support = {}
    max_events = 0
    six_hour_counts = {0: [], 1: []}
    for label in np.unique(segment):
        index = np.flatnonzero(segment == label)
        local_time = time[index]
        max_events = max(max_events, len(index))
        local_split = split[index]
        for scale in SCALES:
            row = support.setdefault(scale, {
                0: {"eligible": 0, "elapsed": []},
                1: {"eligible": 0, "elapsed": []},
            })
            for code in (0, 1):
                endpoints = np.flatnonzero(local_split == code)
                eligible_local = endpoints[
                    endpoints >= int(scale) + DELAY_EVENTS
                ]
                elapsed = (
                    local_time[eligible_local] - local_time[eligible_local - scale]
                    if len(eligible_local) else np.empty(0)
                )
                row[code]["eligible"] += int(len(eligible_local))
                row[code]["elapsed"].extend(elapsed.tolist())
        for end_local in range(len(index)):
            code = int(local_split[end_local])
            if code not in (0, 1):
                continue
            requested = local_time[end_local] - 6.0 * 3600.0
            if requested < coverage.start[int(label)] - 1e-6:
                continue
            start_local = int(np.searchsorted(local_time, requested, side="left"))
            count = int(end_local - start_local)
            if start_local >= DELAY_EVENTS:
                six_hour_counts[code].append(count)
    row = {
        "subject": subject,
        "dataset": stream.dataset,
        "development_recorded_hours": float(
            coverage_meta["train_recorded_seconds"]
            + coverage_meta["validation_recorded_seconds"]
        ) / 3600.0,
        "train_recorded_hours": float(coverage_meta["train_recorded_seconds"]) / 3600.0,
        "validation_recorded_hours": float(
            coverage_meta["validation_recorded_seconds"]
        ) / 3600.0,
        "development_events": int(len(time)),
        "train_events": int((split == 0).sum()),
        "validation_events": int(validation.sum()),
        "max_events_one_recorded_segment": int(max_events),
        "max_recorded_segment_hours": float(np.max(coverage.stop - coverage.start)) / 3600.0,
        "physical_6h_train_windows": int(len(six_hour_counts[0])),
        "physical_6h_validation_windows": int(len(six_hour_counts[1])),
        "physical_6h_delay1000_windows": int(len(six_hour_counts[1])),
        "physical_6h_events_median": (
            float(np.median(six_hour_counts[1])) if six_hour_counts[1] else None
        ),
        "physical_6h_events_p10": (
            float(np.quantile(six_hour_counts[1], 0.1)) if six_hour_counts[1] else None
        ),
        "physical_6h_events_p90": (
            float(np.quantile(six_hour_counts[1], 0.9)) if six_hour_counts[1] else None
        ),
        "sealed_opened": False,
        **_t1_status(subject),
    }
    for scale in SCALES:
        train_elapsed = np.asarray(support[scale][0]["elapsed"], dtype=np.float64)
        elapsed = np.asarray(support[scale][1]["elapsed"], dtype=np.float64)
        row[f"n{scale}_train_windows"] = int(support[scale][0]["eligible"])
        row[f"n{scale}_validation_windows"] = int(support[scale][1]["eligible"])
        row[f"n{scale}_train_hours_median"] = (
            float(np.median(train_elapsed) / 3600.0) if len(train_elapsed) else None
        )
        row[f"n{scale}_hours_median"] = (
            float(np.median(elapsed) / 3600.0) if len(elapsed) else None
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "t2_long_total_effect/cohort_support",
    )
    args = parser.parse_args()
    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    workers = max(1, min(int(args.workers), len(subjects)))
    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future = {pool.submit(_subject, subject): subject for subject in subjects}
        for item in as_completed(future):
            subject = future[item]
            try:
                rows.append(item.result())
            except Exception as error:
                failures.append({"subject": subject, "error": repr(error)})
    rows.sort(key=lambda value: (
        -value["n10000_validation_windows"],
        -value["physical_6h_delay1000_windows"],
        -value["development_events"],
    ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "per_subject_support.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    candidate = [
        value for value in rows
        if (
            value["n3000_train_windows"] > 0
            and value["n3000_validation_windows"] > 0
        ) or (
            value["physical_6h_train_windows"] > 0
            and value["physical_6h_validation_windows"] > 0
        )
    ]
    payload = {
        "status": "COMPLETE" if not failures and len(rows) == len(subjects) else "FAIL",
        "revision": REVISION,
        "workers": workers,
        "subjects_expected": len(subjects),
        "subjects_complete": len(rows),
        "failures": failures,
        "scales_events": list(SCALES),
        "candidate_count": len(candidate),
        "candidates": candidate,
        "all_subjects": rows,
        "per_subject_csv": str(csv_path),
        "per_subject_csv_sha256": contract.sha256_file(csv_path),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "interpretation": (
            "event support and T1 readiness are separate; long support without a "
            "non-degenerate T1 is observable but not scientifically scoreable"
        ),
    }
    contract.atomic_json(args.output_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
