"""Run PR4 interictal synchrony on Epilepsiae ready_full_artifacts subjects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony import (
    run_epilepsiae_interictal_synchrony_from_manifest,
    save_interictal_synchrony_summary,
)


RESULTS_DIR = Path("results")
MANIFEST_CSV = RESULTS_DIR / "epilepsiae_sync_subject_manifest.csv"
OUTPUT_DIR = RESULTS_DIR / "interictal_synchrony" / "epilepsiae_ready_full_artifacts"
SUMMARY_CSV = OUTPUT_DIR / "epilepsiae_ready_full_artifacts_interictal_sync_summary.csv"
EVENT_CSV = OUTPUT_DIR / "epilepsiae_ready_full_artifacts_interictal_sync_events.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="PR4 Epilepsiae interictal synchrony")
    ap.add_argument(
        "--soz-core-json",
        default="",
        help="JSON mapping {subject: [i-channel names]} for clinical SOZ core",
    )
    args = ap.parse_args()

    core_channels_by_subject = None
    if args.soz_core_json and Path(args.soz_core_json).exists():
        with open(args.soz_core_json, "r", encoding="utf-8") as fh:
            core_channels_by_subject = json.load(fh)
        print(f"[INFO] SOZ core channels loaded for {len(core_channels_by_subject)} subjects")

    rows = run_epilepsiae_interictal_synchrony_from_manifest(
        str(MANIFEST_CSV),
        str(OUTPUT_DIR),
        tier="ready_full_artifacts",
        event_rows_csv_path=str(EVENT_CSV),
        core_channels_by_subject=core_channels_by_subject,
    )
    summary_path = save_interictal_synchrony_summary(rows, str(SUMMARY_CSV))
    print(
        json.dumps(
            {
                "n_source_blocks": len(rows),
                "summary_csv": summary_path,
                "event_rows_csv": str(EVENT_CSV),
                "output_dir": str(OUTPUT_DIR),
                "soz_core_json": args.soz_core_json or None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
