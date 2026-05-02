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
    ap.add_argument(
        "--lagpat-root",
        default="",
        help=(
            "Override Epilepsiae lagPat artifact root. Default = canonical legacy "
            "tree /mnt/epilepsia_data/interilca_inter_results/all_data_lns/. Pass "
            "results/epilepsiae_lagpat_backfill to consume the new-pipeline "
            "lagPat in Stage D sensitivity audits. The function probes "
            "<root>/<subject>/all_recs/ first then falls back to flat "
            "<root>/<subject>/ — works for both layouts."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Override OUTPUT_DIR. Default = the canonical "
             "results/interictal_synchrony/epilepsiae_ready_full_artifacts/.",
    )
    ap.add_argument(
        "--manifest",
        default="",
        help="Override the synchrony subject manifest CSV. Default = the "
             "canonical results/epilepsiae_sync_subject_manifest.csv. Pass "
             "a filtered manifest to limit the run to a subject subset "
             "(e.g. Stage D smoke on the 12 stable subjects).",
    )
    args = ap.parse_args()

    core_channels_by_subject = None
    if args.soz_core_json and Path(args.soz_core_json).exists():
        with open(args.soz_core_json, "r", encoding="utf-8") as fh:
            core_channels_by_subject = json.load(fh)
        print(f"[INFO] SOZ core channels loaded for {len(core_channels_by_subject)} subjects")

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    manifest_csv = Path(args.manifest) if args.manifest else MANIFEST_CSV
    summary_csv = output_dir / SUMMARY_CSV.name
    event_csv = output_dir / EVENT_CSV.name

    rows = run_epilepsiae_interictal_synchrony_from_manifest(
        str(manifest_csv),
        str(output_dir),
        tier="ready_full_artifacts",
        artifact_root=args.lagpat_root or None,
        event_rows_csv_path=str(event_csv),
        core_channels_by_subject=core_channels_by_subject,
    )
    summary_path = save_interictal_synchrony_summary(rows, str(summary_csv))
    print(
        json.dumps(
            {
                "n_source_blocks": len(rows),
                "summary_csv": summary_path,
                "event_rows_csv": str(event_csv),
                "output_dir": str(output_dir),
                "manifest_csv": str(manifest_csv),
                "soz_core_json": args.soz_core_json or None,
                "lagpat_root": args.lagpat_root or None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
