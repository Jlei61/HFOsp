#!/usr/bin/env python3
"""Corrected-segment support audit for exact N and previous-N H3 windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.h3_long import (
    H3_LONG_SUPPORT_REVISION,
    SCALES,
)
from src.topic5_continuous_marked_state_r1.r1_2 import (
    load_full_admissible_event_stream,
)


REVISION = H3_LONG_SUPPORT_REVISION


def greedy_disjoint_count(position: np.ndarray, scale: int) -> int:
    count = 0
    last = -10**18
    for value in np.sort(np.asarray(position, dtype=np.int64)):
        if int(value) - last >= int(scale):
            count += 1
            last = int(value)
    return count


def subject_support(subject: str) -> dict:
    root = contract.RESULT_ROOT / "r1_2"
    coverage = CoverageTable.load(root / "coverage" / f"{subject}.npz")
    stream = load_full_admissible_event_stream(subject, coverage)
    time = np.asarray(stream.event_time, dtype=np.float64)
    split = np.asarray(stream.split, dtype=np.int8)
    contract.assert_development_times(subject, time[split == 0], "train")
    contract.assert_development_times(subject, time[split == 1], "validation")
    segment = np.searchsorted(coverage.stop, time, side="right")
    if np.any(segment >= len(coverage.start)):
        raise ValueError(f"{subject}: event after final coverage")
    inside = (time >= coverage.start[segment]) & (time < coverage.stop[segment])
    if not bool(inside.all()):
        raise ValueError(f"{subject}: event outside recorded coverage")
    row = {
        "subject": subject, "dataset": stream.dataset,
        "events": int(len(time)),
        "train_events": int((split == 0).sum()),
        "validation_events": int((split == 1).sum()),
        "recorded_segments": int(len(np.unique(segment))),
        "max_events_one_recorded_segment": int(max(
            np.sum(segment == label) for label in np.unique(segment)
        )),
    }
    for n in SCALES:
        minimal = {0: 0, 1: 0}
        full = {0: 0, 1: 0}
        minimal_positions = {0: [], 1: []}
        full_positions = {0: [], 1: []}
        duration = {0: [], 1: []}
        for label in np.unique(segment):
            index = np.flatnonzero(segment == label)
            local_split = split[index]
            local_time = time[index]
            for code in (0, 1):
                endpoint = np.flatnonzero(local_split == code)
                # Leave room for the next event scored by the one-step design.
                endpoint = endpoint[endpoint < len(index) - 1]
                endpoint = endpoint[local_split[endpoint + 1] == code]
                real = endpoint[endpoint >= n - 1]
                causal = endpoint[endpoint >= 2 * n - 1]
                minimal[code] += int(len(real))
                full[code] += int(len(causal))
                minimal_positions[code].extend(
                    [(int(label), int(value)) for value in real]
                )
                full_positions[code].extend(
                    [(int(label), int(value)) for value in causal]
                )
                if len(real):
                    duration[code].extend(
                        (local_time[real] - local_time[real - n + 1]).tolist()
                    )
        row[f"n{n}_train_minimal_pairs"] = minimal[0]
        row[f"n{n}_validation_minimal_pairs"] = minimal[1]
        row[f"n{n}_train_full_control_pairs"] = full[0]
        row[f"n{n}_validation_full_control_pairs"] = full[1]
        for code, label in ((0, "train"), (1, "validation")):
            by_segment_minimal: dict[int, list[int]] = {}
            by_segment_full: dict[int, list[int]] = {}
            for segment_label, position in minimal_positions[code]:
                by_segment_minimal.setdefault(segment_label, []).append(position)
            for segment_label, position in full_positions[code]:
                by_segment_full.setdefault(segment_label, []).append(position)
            row[f"n{n}_{label}_minimal_independent_blocks"] = int(sum(
                greedy_disjoint_count(np.asarray(values), n)
                for values in by_segment_minimal.values()
            ))
            row[f"n{n}_{label}_full_control_independent_units"] = int(sum(
                greedy_disjoint_count(np.asarray(values), 2 * n)
                for values in by_segment_full.values()
            ))
        seconds = np.asarray(duration[1], dtype=np.float64)
        row[f"n{n}_validation_hours_median"] = (
            float(np.median(seconds) / 3600.0) if len(seconds) else None
        )
        row[f"n{n}_validation_hours_q25"] = (
            float(np.quantile(seconds, .25) / 3600.0) if len(seconds) else None
        )
        row[f"n{n}_validation_hours_q75"] = (
            float(np.quantile(seconds, .75) / 3600.0) if len(seconds) else None
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long/support",
    )
    args = parser.parse_args()
    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(subjects))) as pool:
        future = {pool.submit(subject_support, subject): subject for subject in subjects}
        for item in as_completed(future):
            subject = future[item]
            try:
                rows.append(item.result())
            except Exception as error:
                failures.append({"subject": subject, "error": repr(error)})
    rows.sort(key=lambda value: value["subject"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "per_subject_support.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    selected = set(contract.R1_5_EXTENSION_SUBJECTS)
    cells = []
    for row in rows:
        if row["subject"] not in selected:
            continue
        for n in SCALES:
            minimal = (
                row[f"n{n}_train_minimal_pairs"] >= 100
                and row[f"n{n}_validation_minimal_pairs"] >= 100
            )
            full = (
                row[f"n{n}_train_full_control_pairs"] >= 100
                and row[f"n{n}_validation_full_control_pairs"] >= 100
            )
            cells.append({
                "subject": row["subject"], "scale_events": int(n),
                "minimal_support": bool(minimal),
                "full_causal_control_support": bool(full),
                "role": (
                    "full_control" if full else
                    "boundary_incomplete_control" if minimal else
                    "not_applicable_support"
                ),
                "train_minimal_pairs": row[f"n{n}_train_minimal_pairs"],
                "validation_minimal_pairs": row[
                    f"n{n}_validation_minimal_pairs"
                ],
                "train_full_control_pairs": row[
                    f"n{n}_train_full_control_pairs"
                ],
                "validation_full_control_pairs": row[
                    f"n{n}_validation_full_control_pairs"
                ],
                "train_independent_blocks": row[
                    f"n{n}_train_full_control_independent_units"
                    if full else f"n{n}_train_minimal_independent_blocks"
                ],
                "validation_independent_blocks": row[
                    f"n{n}_validation_full_control_independent_units"
                    if full else f"n{n}_validation_minimal_independent_blocks"
                ],
                "independent_unit_width_events": int(2 * n if full else n),
                "validation_hours_median": row[
                    f"n{n}_validation_hours_median"
                ],
            })
    payload = {
        "status": (
            "COMPLETE" if len(rows) == len(subjects) and not failures else "FAIL"
        ),
        "revision": REVISION,
        "subjects_expected": len(subjects), "subjects_complete": len(rows),
        "failures": failures, "scales_events": list(SCALES),
        "r1_5_subjects": list(contract.R1_5_EXTENSION_SUBJECTS),
        "scheduled_cells": [cell for cell in cells if cell["minimal_support"]],
        "not_applicable_cells": [
            cell for cell in cells if not cell["minimal_support"]
        ],
        "per_subject_csv": str(csv_path),
        "per_subject_csv_sha256": contract.sha256_file(csv_path),
        "development_time_contract_verified": True,
        "source_hashes": {
            "producer": contract.sha256_file(Path(__file__)),
            "h3_long": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/h3_long.py"
            ),
            "contract": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/contract.py"
            ),
            "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
        },
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": (
            "support only; sliding pairs are not independent biological units, "
            "and fewer than three final validation units is descriptive"
        ),
    }
    contract.atomic_json(args.output_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
