"""PR6 driver: run interictal synchrony analysis and generate Figures A–E.

Usage examples:
    # Epilepsiae only (default; event rows)
    python scripts/pr6_interictal_sync_figures.py

    # Both datasets (event rows)
    python scripts/pr6_interictal_sync_figures.py \
        --yuquan-events results/interictal_synchrony/yuquan_soz/aggregated/yuquan_sync_event_annotations.csv

    # Custom output directory
    python scripts/pr6_interictal_sync_figures.py \
        --output-dir results/interictal_synchrony/analysis/yuquan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony_analysis import run_pr6_analysis


RESULTS_DIR = Path("results")
DEFAULT_EPILEPSIAE_EVENTS = (
    RESULTS_DIR
    / "interictal_synchrony"
    / "epilepsiae_ready_full_artifacts"
    / "aggregated"
    / "epilepsiae_sync_event_annotations.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "interictal_synchrony" / "analysis" / "combined"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PR6: interictal synchrony statistics and Figures A–E"
    )
    ap.add_argument(
        "--epilepsiae-events",
        default=str(DEFAULT_EPILEPSIAE_EVENTS),
        help="Path to Epilepsiae event rows CSV (preferred input)",
    )
    ap.add_argument(
        "--yuquan-events",
        default="",
        help="Path to Yuquan annotated event rows CSV (from PR5 aggregation)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    ap.add_argument(
        "--min-interval-sec",
        type=float,
        default=10800.0,
        help="Minimum clean interval duration for fixed-window analysis (default 3h)",
    )
    ap.add_argument(
        "--window-sec",
        type=float,
        default=3600.0,
        help="Duration of each fixed window in seconds (default 1h)",
    )
    ap.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for normalized trajectory",
    )
    ap.add_argument(
        "--primary-metric",
        default="sync_legacy_global",
        help="Primary synchrony metric for single-metric figures",
    )
    ap.add_argument(
        "--example-subjects",
        default="",
        help="Comma-separated subjects for per-subject Figure A folders (default: all assigned subjects)",
    )
    args = ap.parse_args()

    epi_events = args.epilepsiae_events if args.epilepsiae_events and Path(args.epilepsiae_events).exists() else None
    yq_events = args.yuquan_events if args.yuquan_events and Path(args.yuquan_events).exists() else None
    example_subjects = [s.strip() for s in args.example_subjects.split(",") if s.strip()] or None

    summary = run_pr6_analysis(
        epilepsiae_events_csv=epi_events,
        yuquan_events_csv=yq_events,
        output_dir=args.output_dir,
        min_interval_sec=args.min_interval_sec,
        window_sec=args.window_sec,
        n_bins=args.n_bins,
        primary_metric=args.primary_metric,
        example_subjects=example_subjects,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
