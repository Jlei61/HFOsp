"""Unified PR4 entrypoint for event-level interictal synchrony outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony import (
    build_block_summary_from_event_rows,
    build_event_rows_from_result,
    build_interictal_synchrony,
    build_interictal_synchrony_from_legacy_lagpat,
    save_interictal_synchrony_result,
    save_interictal_synchrony_rows_csv,
    save_interictal_synchrony_summary,
)


def _parse_csv_arg(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _metadata_from_args(args: argparse.Namespace) -> dict[str, object]:
    metadata: dict[str, object] = {}
    scalar_fields = (
        "subject",
        "patient_code",
        "recording_id",
        "block_stem",
        "block_start_day_night",
        "block_end_day_night",
        "timezone_name",
        "day_night_rule",
        "tier",
    )
    for name in scalar_fields:
        value = getattr(args, name)
        if value not in (None, ""):
            metadata[name] = value
    for name in ("block_start_epoch", "block_end_epoch"):
        value = getattr(args, name)
        if value is not None:
            metadata[name] = float(value)
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute PR4 event-level interictal synchrony.")
    ap.add_argument("--group-analysis-npz")
    ap.add_argument("--lagpat-npz")
    ap.add_argument("--packed-times-npy")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--events-csv")
    ap.add_argument("--summary-csv")
    ap.add_argument("--core-channels", default="")
    ap.add_argument("--subject", default="")
    ap.add_argument("--patient-code", default="")
    ap.add_argument("--recording-id", default="")
    ap.add_argument("--block-stem", default="")
    ap.add_argument("--block-start-epoch", type=float)
    ap.add_argument("--block-end-epoch", type=float)
    ap.add_argument("--block-start-day-night", default="")
    ap.add_argument("--block-end-day-night", default="")
    ap.add_argument("--timezone-name", default="")
    ap.add_argument("--day-night-rule", default="")
    ap.add_argument("--tier", default="")
    args = ap.parse_args()

    if bool(args.group_analysis_npz) == bool(args.lagpat_npz):
        raise ValueError("Specify exactly one input mode: groupAnalysis or legacy lagPat.")
    if args.lagpat_npz and not args.packed_times_npy:
        raise ValueError("--packed-times-npy is required with --lagpat-npz.")

    core_channels = _parse_csv_arg(args.core_channels) or None
    metadata = _metadata_from_args(args)
    if args.group_analysis_npz:
        result = build_interictal_synchrony(
            args.group_analysis_npz,
            core_channels=core_channels,
            metadata=metadata,
        )
    else:
        result = build_interictal_synchrony_from_legacy_lagpat(
            args.lagpat_npz,
            args.packed_times_npy,
            core_channels=core_channels,
            metadata=metadata,
        )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_interictal_synchrony_result(result, str(out_npz))
    event_rows = build_event_rows_from_result(result)

    summary_row = {
        **result.metadata,
        "output_npz_path": str(out_npz),
        **build_block_summary_from_event_rows(event_rows),
    }
    if args.events_csv:
        save_interictal_synchrony_rows_csv(event_rows, args.events_csv)
    if args.summary_csv:
        save_interictal_synchrony_summary([summary_row], args.summary_csv)

    print(json.dumps(summary_row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
