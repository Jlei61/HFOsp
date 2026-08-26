#!/usr/bin/env python3
"""Freeze all admissible R1.2 raw-anchor and event denominators."""
from __future__ import annotations

import json

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import (
    FULL_COVERAGE_REVISION, FULL_STREAM_REVISION, R1_2_REVISION,
    _candidate_anchors, load_full_admissible_event_stream,
)
from src.topic5_continuous_marked_state_r1.raw_observation import RawAnchorReader


def main() -> None:
    root = contract.RESULT_ROOT / "r1_2"
    prior = json.loads(
        (contract.RESULT_ROOT / "manifests/R1_2_ANCHOR_DENOMINATORS.json").read_text()
    )
    prior_by_subject = {row["subject"]: row for row in prior["rows"]}
    rows = []
    for subject in contract.PILOT_SUBJECTS:
        coverage = CoverageTable.load(root / "coverage" / f"{subject}.npz")
        stream = load_full_admissible_event_stream(subject, coverage)
        reader = RawAnchorReader(subject, stream.event_time)
        raw_time, _raw_split, _raw_session = reader.anchor_times()
        admissible_before_readability = int(sum(
            np.sum((coverage.start <= value) & (value < coverage.stop)) == 1
            for value in raw_time
        ))
        _time, split, _session = _candidate_anchors(reader, coverage)
        old = prior_by_subject[subject]
        train = int(np.sum(split == 0))
        validation = int(np.sum(split == 1))
        rows.append({
            "subject": subject,
            "train_anchors": train,
            "validation_anchors": validation,
            "total_anchors": train + validation,
            "raw_cache_train_anchors_before_r1_2_admissibility": int(old["train_anchors"]),
            "raw_cache_validation_anchors_before_r1_2_admissibility": int(old["validation_anchors"]),
            "anchors_removed_by_ictal_or_2h_postictal_coverage": int(
                old["total_anchors"] - admissible_before_readability
            ),
            "anchors_removed_by_unreadable_background": int(
                admissible_before_readability - train - validation
            ),
            "train_events": int(np.sum(stream.split == 0)),
            "validation_events": int(np.sum(stream.split == 1)),
        })
    output = {
        "status": "COMPLETE", "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "full_stream_revision": FULL_STREAM_REVISION,
        "full_coverage_revision": FULL_COVERAGE_REVISION,
        "rows": rows,
        "interpretation": (
            "Every readable R0 raw anchor inside the R1.2 admissible time axis is retained. "
            "Ictal and 2 h postictal anchors are excluded because they cannot causally "
            "correct a state later scored as interictal; ordinary preictal events remain "
            "in the event stream, while raw availability follows the frozen cache."
        ),
        "sealed_opened": False,
    }
    path = root / "manifests/R1_2_ADMISSIBLE_DENOMINATORS.json"
    contract.atomic_json(path, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
