#!/usr/bin/env python3
"""Build exact metadata coverage for R1 development likelihoods."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import write_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(contract.PILOT_SUBJECTS))
    parser.add_argument("--output-root", type=Path,
                        default=contract.RESULT_ROOT / "coverage")
    args = parser.parse_args()
    summaries = []
    for subject in args.subjects:
        output = args.output_root / f"{subject}.npz"
        manifest = write_coverage(subject, output)
        summaries.append(manifest)
        parity = manifest.get("duration_parity") or {}
        print(
            f"{subject:24s} segments={manifest['n_merged_segments']:4d} "
            f"events={manifest['n_events']:7d} "
            f"parity_max={parity.get('max_abs_seconds', float('nan')):.6g}s",
            flush=True,
        )
    contract.atomic_json(args.output_root / "COVERAGE_MANIFEST.json", {
        "contract": contract.REVISION,
        "subjects": summaries,
        "n_subjects": len(summaries),
        "all_duration_parity_exact": all(
            row.get("duration_parity", {}).get("max_abs_seconds") == 0.0
            for row in summaries
        ),
        "sealed_opened": False,
    })


if __name__ == "__main__":
    main()
