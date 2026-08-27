#!/usr/bin/env python3
"""Freeze the prospective R1.7A cohort without reading any model outcome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import (
    load_full_admissible_event_stream,
)
from src.topic5_continuous_marked_state_r1.r1_7 import R1_7A_REVISION


EXCLUSION_PROVENANCE = {
    "epilepsiae_139": "R1 pilot architecture and threshold selection",
    "epilepsiae_384": "R1.5 and R1.6 optimiser confirmation",
    "epilepsiae_620": "R0/R1 architecture pilot",
    "epilepsiae_922": "long-scale support discovery",
    "epilepsiae_958": "R0/R1 architecture pilot",
    "epilepsiae_1096": "R1.5 and R1.6 optimiser confirmation",
    "yuquan_chengshuai": "long-scale discovery and R1.6 confirmation",
    "yuquan_chenziyang": "long-scale discovery and R1.6 confirmation",
    "yuquan_hanyuxuan": "R1 pilot and long-scale discovery",
    "yuquan_huanghanwen": "R0/R1 architecture pilot",
    "yuquan_pengzihang": "long-scale support discovery",
    "yuquan_zhangjiaqi": "R0/R1 pilot and R1.6 confirmation",
    "yuquan_zhangkexuan": "R1.5 and R1.6 optimiser confirmation",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_7a"
    )
    args = parser.parse_args()
    rows = []
    for manifest_path in sorted((args.r1_2_root / "coverage").glob("*.manifest.json")):
        subject = manifest_path.name.removesuffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        coverage_path = args.r1_2_root / "coverage" / f"{subject}.npz"
        coverage = CoverageTable.load(coverage_path)
        stream = load_full_admissible_event_stream(subject, coverage)
        row = {
            "subject": subject,
            "dataset": subject.split("_", 1)[0],
            "n_contacts": int(stream.n_contacts),
            "train_events": int(np.sum(stream.split == 0)),
            "validation_events": int(np.sum(stream.split == 1)),
            "train_recorded_seconds": float(manifest["train_recorded_seconds"]),
            "validation_recorded_seconds": float(manifest["validation_recorded_seconds"]),
            "n_coverage_segments": int(manifest["n_coverage_segments"]),
            "n_continuity_sessions": int(manifest["n_continuity_sessions"]),
            "coverage_sha256": contract.sha256_file(coverage_path),
            "historically_excluded": subject in EXCLUSION_PROVENANCE,
            "exclusion_reason": EXCLUSION_PROVENANCE.get(subject),
        }
        row["support_eligible"] = bool(
            not row["historically_excluded"]
            and row["n_contacts"] >= 6
            and row["train_events"] >= 1000
            and row["validation_events"] >= 300
            and row["train_recorded_seconds"] >= 21600.0
            and row["validation_recorded_seconds"] >= 5400.0
        )
        rows.append(row)
    selected = []
    for dataset in ("epilepsiae", "yuquan"):
        eligible = [
            row for row in rows
            if row["dataset"] == dataset and row["support_eligible"]
        ]
        eligible.sort(key=lambda row: (
            -row["validation_events"],
            -row["validation_recorded_seconds"],
            row["subject"],
        ))
        selected.extend(row["subject"] for row in eligible[:5])
    if tuple(sorted(selected)) != tuple(sorted(contract.R1_7A_SUBJECTS)):
        raise RuntimeError(
            "R1.7A frozen tuple disagrees with outcome-blind selection rule: "
            f"{selected}"
        )
    for row in rows:
        row["selected"] = row["subject"] in selected
    payload = {
        "status": "FROZEN",
        "revision": R1_7A_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_uses_model_outcomes": False,
        "selection_rule": {
            "historical_decision_subjects_excluded": True,
            "minimum_contacts": 6,
            "minimum_train_events": 1000,
            "minimum_validation_events": 300,
            "minimum_train_recorded_seconds": 21600.0,
            "minimum_validation_recorded_seconds": 5400.0,
            "within_dataset_ranking": [
                "validation_events descending",
                "validation_recorded_seconds descending",
                "subject lexical ascending",
            ],
            "take_per_dataset": 5,
        },
        "selected_subjects": selected,
        "n_selected": len(selected),
        "dataset_counts": {
            dataset: sum(value.startswith(dataset + "_") for value in selected)
            for dataset in ("epilepsiae", "yuquan")
        },
        "rows": rows,
        "development_only": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    contract.atomic_json(args.output_root / "manifests/cohort_inventory.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
