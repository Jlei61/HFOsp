"""Run PR4 interictal synchrony on Yuquan legacy lagPat assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony import (
    run_yuquan_interictal_synchrony,
    save_interictal_synchrony_summary,
)


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "interictal_synchrony" / "yuquan_blocks"
SUMMARY_CSV = OUTPUT_DIR / "yuquan_interictal_sync_summary.csv"
EVENT_CSV = OUTPUT_DIR / "yuquan_interictal_sync_events.csv"


def main() -> None:
    rows = run_yuquan_interictal_synchrony(
        str(OUTPUT_DIR),
        event_rows_csv_path=str(EVENT_CSV),
    )
    summary_path = save_interictal_synchrony_summary(rows, str(SUMMARY_CSV))
    print(
        json.dumps(
            {
                "n_source_blocks": len(rows),
                "summary_csv": summary_path,
                "event_rows_csv": str(EVENT_CSV),
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
