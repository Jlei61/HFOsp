"""Aggregate Yuquan event rows into seizure-interval tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony_aggregation import run_yuquan_sync_aggregation


RESULTS_DIR = Path("results")
SYNC_DIR = RESULTS_DIR / "interictal_synchrony" / "yuquan_blocks"
SYNC_EVENT_CSV = SYNC_DIR / "yuquan_interictal_sync_events.csv"
OUTPUT_DIR = RESULTS_DIR / "interictal_synchrony" / "yuquan"


def _parse_subjects(raw: str) -> list[str] | None:
    subjects = [x.strip() for x in str(raw).split(",") if x.strip()]
    return subjects or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate Yuquan synchrony rows into seizure-interval tables."
    )
    ap.add_argument("--sync-event-csv", default=str(SYNC_EVENT_CSV))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--data-root", default="/mnt/yuquan_data/yuquan_24h_edf")
    ap.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject list. Default: infer from event CSV.",
    )
    ap.add_argument(
        "--seizure-inventory-path",
        default="",
        help="Optional CSV/JSON seizure inventory artifact to reuse instead of reparsing EDFs.",
    )
    args = ap.parse_args()

    summary = run_yuquan_sync_aggregation(
        sync_event_csv=args.sync_event_csv,
        output_dir=args.output_dir,
        data_root=args.data_root,
        subjects=_parse_subjects(args.subjects),
        seizure_inventory_path=args.seizure_inventory_path or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
