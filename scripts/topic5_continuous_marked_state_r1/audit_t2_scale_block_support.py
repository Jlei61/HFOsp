#!/usr/bin/env python3
"""Census of independent non-overlapping N-event blocks, by timescale and layer.

The long-scale H3 route was retired partly because per-event sliding windows
overlap heavily, so row counts vastly exceed independent information.  This
answers the successor question quantitatively: at which N can a block-level test
be run at all, and on how many patients?

A block is a complete run of N consecutive admissible events inside one recorded
coverage segment.  Blocks never span a recording gap and a segment's remainder
is dropped rather than forming a short block, matching the T2 support rule.

Layers reported:
  D_mechanism  - the contract's T2 layer (validation, last 40% of recorded time)
  validation   - the whole development validation layer (reusing H1 data, so not
                 admissible as T2 evidence; shown only as an upper bound)
  TRAIN        - where the model is fitted, so never held-out evidence; shown
                 only to demonstrate the ceiling is a property of the dataset
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design
from src.topic5_continuous_marked_state_r1.r1_7 import (
    complete_event_blocks_by_segment, coverage_segment_for_times,
    split_validation_by_recorded_time,
)


DEFAULT_SCALES = (100, 200, 500, 1000, 2000, 5000, 10000)
COHORT_ELIGIBLE_BLOCKS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=contract.RESULT_ROOT / "r1_7b_cohort_extension")
    parser.add_argument("--scales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    inventory = json.loads(
        (args.root / "manifests/cohort_inventory.json").read_text()
    )
    subjects = inventory["selected_subjects"]
    per_subject: dict[str, dict] = {}
    for subject in subjects:
        coverage = CoverageTable.load(
            args.root / "upstream_r1_2" / "coverage" / f"{subject}.npz"
        )
        layer = split_validation_by_recorded_time(
            coverage, validation_start=coverage.train_end_epoch,
            validation_stop=coverage.dev_end_epoch,
        )
        manifest = json.loads((args.root / "cache" / subject / "manifest.json").read_text())
        design = load_full_design(Path(manifest["design"]))
        segment = coverage_segment_for_times(coverage, design.event_time)
        masks = {
            "d_mechanism": (design.event_split == 1)
            & (design.event_time >= layer.mechanism_start),
            "validation": design.event_split == 1,
            "train": design.event_split == 0,
        }
        row = {"n_events": {k: int(v.sum()) for k, v in masks.items()}, "blocks": {}}
        for name, mask in masks.items():
            row["blocks"][name] = {
                str(scale): int(complete_event_blocks_by_segment(
                    segment, mask, block_events=scale)[0])
                for scale in args.scales
            }
        per_subject[subject] = row
    totals = {
        layer: {
            str(scale): {
                "total_blocks": int(sum(
                    per_subject[s]["blocks"][layer][str(scale)] for s in subjects)),
                "patients_cohort_eligible": int(sum(
                    per_subject[s]["blocks"][layer][str(scale)] >= COHORT_ELIGIBLE_BLOCKS
                    for s in subjects)),
                "patients_with_any_block": int(sum(
                    per_subject[s]["blocks"][layer][str(scale)] >= 1 for s in subjects)),
            } for scale in args.scales
        } for layer in ("d_mechanism", "validation", "train")
    }
    largest = None
    for scale in sorted(args.scales):
        if totals["d_mechanism"][str(scale)]["patients_cohort_eligible"] >= 2:
            largest = scale
    payload = {
        "status": "COMPLETE",
        "revision": "t2_scale_block_support_census_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_subjects": len(subjects), "subjects": subjects,
        "scales": list(args.scales),
        "cohort_eligible_block_threshold": COHORT_ELIGIBLE_BLOCKS,
        "per_subject": per_subject, "totals": totals,
        "largest_scale_with_two_or_more_cohort_eligible_patients_in_d_mechanism": largest,
        "note": (
            "Blocks never span a recorded gap.  d_mechanism is the only layer "
            "admissible as T2 evidence; validation reuses H1 data and train is "
            "where the model is fitted."
        ),
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    output = args.output or (args.root / "reports/t2_scale_block_support.json")
    contract.atomic_json(output, payload)
    header = f"{'layer':14s} " + " ".join(f"N={s:<11d}" for s in args.scales)
    print("independent non-overlapping N-event blocks -- total / patients with >=5\n")
    print(header); print("-" * len(header))
    for layer in ("d_mechanism", "validation", "train"):
        print(f"{layer:14s} " + " ".join(
            f"{totals[layer][str(s)]['total_blocks']:>4d} /{totals[layer][str(s)]['patients_cohort_eligible']:<7d}"
            for s in args.scales))
    print(f"\nlargest D_mechanism scale with >=2 cohort-eligible patients: {largest}")
    print(f"written: {output}")


if __name__ == "__main__":
    main()
