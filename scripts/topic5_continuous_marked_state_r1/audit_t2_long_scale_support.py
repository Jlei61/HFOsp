#!/usr/bin/env python3
"""Audit observable 10^2/10^3/10^4-event histories before T2-S1."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design


SCALES = (100, 1000, 10000)
REVISION = "t2_long_event_history_contiguous_coverage_audit_v1"


def main() -> None:
    output_root = contract.RESULT_ROOT / "t2_s1_long_scale"
    rows = []
    subjects = contract.PILOT_SUBJECTS
    for subject in subjects:
        cache = contract.RESULT_ROOT / "r1_2" / "cache" / subject
        design = load_full_design(cache / "full_design.npz")
        coverage = CoverageTable.load(
            contract.RESULT_ROOT / "r1_2" / "coverage" / f"{subject}.npz"
        )
        event_time = np.asarray(design.event_time, dtype=np.float64)
        event_split = np.asarray(design.event_split, dtype=np.int8)
        # Reset at every actually unrecorded positive gap, even if upstream
        # metadata gave the adjoining blocks the same broader session label.
        segment = np.searchsorted(coverage.stop, event_time, side="right")
        if np.any(segment >= len(coverage.start)):
            raise ValueError(f"{subject}: event after final coverage segment")
        inside = (
            (event_time >= coverage.start[segment])
            & (event_time < coverage.stop[segment])
        )
        if not bool(inside.all()):
            raise ValueError(f"{subject}: event outside recorded segment")
        prior = np.zeros(len(event_time), dtype=np.int64)
        position_by_segment = {}
        for label in np.unique(segment):
            index = np.flatnonzero(segment == label)
            index = index[np.argsort(event_time[index], kind="stable")]
            prior[index] = np.arange(len(index), dtype=np.int64)
            position_by_segment[int(label)] = index
        validation = event_split == 1
        for scale in SCALES:
            eligible = validation & (prior >= int(scale))
            elapsed = []
            for label, index in position_by_segment.items():
                selected = index[eligible[index]]
                if not len(selected):
                    continue
                local_position = prior[selected]
                elapsed.extend(
                    (event_time[selected] - event_time[index[local_position - scale]])
                    .astype(float).tolist()
                )
            elapsed = np.asarray(elapsed, dtype=np.float64)
            rows.append({
                "subject": subject,
                "scale_events": int(scale),
                "validation_events": int(validation.sum()),
                "eligible_validation_events": int(eligible.sum()),
                "eligible_fraction": float(eligible.sum() / max(validation.sum(), 1)),
                "eligible_contiguous_segments": int(len(np.unique(segment[eligible]))),
                "max_prior_events_in_contiguous_segment": int(
                    prior[validation].max() if bool(validation.any()) else 0
                ),
                "history_elapsed_hours_median": (
                    float(np.median(elapsed) / 3600.0) if len(elapsed) else None
                ),
                "history_elapsed_hours_p10": (
                    float(np.quantile(elapsed, 0.1) / 3600.0) if len(elapsed) else None
                ),
                "history_elapsed_hours_p90": (
                    float(np.quantile(elapsed, 0.9) / 3600.0) if len(elapsed) else None
                ),
                "history_crosses_unrecorded_gap": False,
                "sealed_opened": False,
            })
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "long_scale_observability.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "COMPLETE",
        "revision": REVISION,
        "scales_events": list(SCALES),
        "subjects": list(subjects),
        "rows": rows,
        "selection_rule": (
            "T2-S1 eligibility is determined only by complete prior-event history "
            "within one recorded coverage segment; ordinary lack of support is "
            "unobservable, not a negative biological result"
        ),
        "sealed_opened": False,
    }
    target = output_root / "long_scale_observability.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(temporary, target)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
