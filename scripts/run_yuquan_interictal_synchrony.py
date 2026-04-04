"""Run PR4 interictal synchrony on Yuquan legacy lagPat assets."""

from __future__ import annotations

import argparse
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


def _parse_subjects(raw: str) -> list[str] | None:
    subjects = [x.strip() for x in str(raw).split(",") if x.strip()]
    return subjects or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Yuquan PR4 synchrony export from legacy lagPat assets."
    )
    ap.add_argument("--artifact-root", default="/mnt/yuquan_data/yuquan_24h_edf")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--event-csv", default=str(EVENT_CSV))
    ap.add_argument("--summary-csv", default=str(SUMMARY_CSV))
    ap.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject list. Default: all subjects under artifact root.",
    )
    ap.add_argument(
        "--soz-core-json",
        default="",
        help="JSON mapping {subject: [SOZ channel names]} for core/penumbra split.",
    )
    args = ap.parse_args()

    core_channels_by_subject = None
    if args.soz_core_json and Path(args.soz_core_json).exists():
        with open(args.soz_core_json, "r", encoding="utf-8") as fh:
            core_channels_by_subject = json.load(fh)
        print(f"[INFO] SOZ core channels loaded for {len(core_channels_by_subject)} subjects")

    rows = run_yuquan_interictal_synchrony(
        str(args.output_dir),
        artifact_root=args.artifact_root,
        subjects=_parse_subjects(args.subjects),
        core_channels_by_subject=core_channels_by_subject,
        event_rows_csv_path=str(args.event_csv),
    )
    summary_path = save_interictal_synchrony_summary(rows, str(args.summary_csv))
    print(
        json.dumps(
            {
                "n_source_blocks": len(rows),
                "summary_csv": summary_path,
                "event_rows_csv": str(args.event_csv),
                "output_dir": str(args.output_dir),
                "soz_core_json": args.soz_core_json or None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
