"""Aggregate Yuquan event-level synchrony into interval/window tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony_aggregation import run_yuquan_sync_aggregation


RESULTS_DIR = Path("results")
SYNC_DIR = RESULTS_DIR / "interictal_synchrony" / "yuquan_blocks"
SYNC_EVENT_CSV = SYNC_DIR / "yuquan_interictal_sync_events.csv"
OUTPUT_DIR = RESULTS_DIR / "interictal_synchrony" / "yuquan"


def main() -> None:
    summary = run_yuquan_sync_aggregation(
        sync_event_csv=SYNC_EVENT_CSV,
        output_dir=OUTPUT_DIR,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
