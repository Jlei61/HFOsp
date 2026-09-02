#!/usr/bin/env python3
"""C0: how much recorded time each patient really supports, per horizon.

Prints and stores, per patient and per split, the *recorded hours* left after the
gap, seizure-onset, postictal and split cuts, and the number of **non-overlapping**
target blocks each fixed-time horizon can carry.  A horizon with too few disjoint
blocks is marked ``not_estimable`` here, before any model is trained, so a thin
denominator can never be discovered after the fact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.support import (  # noqa: E402
    EXPLORATORY_HORIZON_MINUTES,
    MAIN_HORIZONS_MINUTES,
    POSTICTAL_EXCLUSION_SECONDS,
    SPLIT_NAMES,
    subject_support,
)
from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402

# v0.1 audit artefacts live in the main checkout and are read-only for agent C.
V0_1_RESULTS = Path("/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1")
DEFAULT_INVENTORY = V0_1_RESULTS / "block_inventory.csv"
DEFAULT_DATASET = Path("/data/hfosp_group_event_state_v0_1/dataset")
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3/support"

# A horizon needs enough disjoint development-test blocks for a patient-level
# effect to mean anything.  Pre-registered here, before any score exists.
MIN_TEST_BLOCKS = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--postictal-exclusion-s", type=float, default=POSTICTAL_EXCLUSION_SECONDS
    )
    parser.add_argument("--tag", default="primary")
    args = parser.parse_args()

    horizons = list(MAIN_HORIZONS_MINUTES) + [EXPLORATORY_HORIZON_MINUTES]
    subjects = sorted(p.name for p in args.dataset_root.iterdir() if (p / "index.json").exists())

    records = []
    for subject in subjects:
        rec = subject_support(
            subject,
            args.inventory,
            args.dataset_root / subject / "index.json",
            postictal_exclusion_s=args.postictal_exclusion_s,
            horizons=horizons,
        )
        rec["eligible_horizons"] = [
            h
            for h in horizons
            if rec["horizon_support"][str(h)]["n_independent_target_blocks"][
                "development_test"
            ]
            >= MIN_TEST_BLOCKS
        ]
        records.append(rec)

    header = f"{'subject':26s} {'cov_h':>7s} {'usable_h':>9s} {'test_h':>7s}" + "".join(
        f" {str(h) + 'm':>7s}" for h in horizons
    )
    print(header)
    print("-" * len(header))
    for rec in sorted(records, key=lambda r: -r["usable_hours_after_seizure_cuts"]):
        counts = "".join(
            f" {rec['horizon_support'][str(h)]['n_independent_target_blocks']['development_test']:7d}"
            for h in horizons
        )
        print(
            f"{rec['subject']:26s} {rec['coverage_hours']:7.1f} "
            f"{rec['usable_hours_after_seizure_cuts']:9.1f} "
            f"{rec['split_hours']['development_test']:7.1f}{counts}"
        )

    summary = {
        "tag": args.tag,
        "postictal_exclusion_s": args.postictal_exclusion_s,
        "horizons_minutes": horizons,
        "min_development_test_blocks": MIN_TEST_BLOCKS,
        "split_names": list(SPLIT_NAMES),
        "n_subjects": len(records),
        "n_subjects_eligible_per_horizon": {
            str(h): sum(1 for r in records if h in r["eligible_horizons"]) for h in horizons
        },
        "subjects": records,
    }
    out = Path(args.out_dir) / f"coverage_support_{args.tag}.json"
    write_json_atomic(summary, out)
    print(f"\nn eligible per horizon: {summary['n_subjects_eligible_per_horizon']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
