"""
Augment an existing event CSV with per-region synchrony columns.

Epilepsiae: adds sync_phase_i, sync_phase_l, sync_phase_e (and legacy/span variants)
            based on electrode focus_rel labels from SQL.

Usage:
    python scripts/compute_region_stratified_synchrony.py \
        --event-csv results/.../events.csv \
        --focus-rel-json results/epilepsiae_electrode_focus_rel.json \
        --artifact-root /mnt/epilepsia_data/interilca_inter_results/all_data_lns \
        --output-csv results/.../events_region.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony import (
    compute_event_synchrony_metrics,
    load_legacy_lagpat_group_analysis,
)


def _build_region_masks(
    ch_names: list[str],
    region_channels: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    masks = {}
    name_set = {str(x) for x in ch_names}
    for region, channels in region_channels.items():
        ch_set = set(channels)
        mask = np.array([name in ch_set for name in ch_names], dtype=bool)
        if np.any(mask):
            masks[region] = mask
    return masks


def augment_events_with_regions(
    event_df: pd.DataFrame,
    focus_rel: dict[str, dict[str, list[str]]],
    artifact_root: Path,
) -> pd.DataFrame:
    metrics = ["phase", "legacy", "span"]
    regions = ["i", "l", "e"]
    new_cols = {}
    for region in regions:
        for metric in metrics:
            col = f"sync_{metric}_{region}"
            new_cols[col] = np.full(len(event_df), np.nan, dtype=np.float64)
        new_cols[f"n_{region}"] = np.zeros(len(event_df), dtype=np.int64)

    grouped = event_df.groupby(["subject", "block_stem"])
    n_blocks = 0
    for (subject, block_stem), idx in grouped.groups.items():
        subj_str = str(subject)
        if subj_str not in focus_rel:
            continue

        subject_dir = artifact_root / subj_str / "all_recs"
        lagpat_path = subject_dir / f"{block_stem}_lagPat.npz"
        packed_path = subject_dir / f"{block_stem}_packedTimes.npy"
        if not lagpat_path.exists() or not packed_path.exists():
            continue

        ga = load_legacy_lagpat_group_analysis(str(lagpat_path), str(packed_path))
        ch_names = ga["ch_names"]
        lag_raw = ga["lag_raw"]
        events_bool = ga["events_bool"]
        event_windows = ga["event_windows"]

        region_masks = _build_region_masks(ch_names, focus_rel[subj_str])

        block_rows = event_df.loc[idx].sort_values("event_index")
        event_indices = block_rows["event_index"].values

        for region, mask in region_masks.items():
            if region not in regions:
                continue
            result = compute_event_synchrony_metrics(
                lag_raw, events_bool, event_windows, channel_mask=mask
            )
            for ei, df_idx in zip(event_indices, block_rows.index):
                if ei < len(result["phase"]):
                    for metric in metrics:
                        new_cols[f"sync_{metric}_{region}"][df_idx] = result[metric][ei]
                    new_cols[f"n_{region}"][df_idx] = result["n_participating"][ei]

        n_blocks += 1
        if n_blocks % 50 == 0:
            print(f"  processed {n_blocks} blocks...")

    for col, values in new_cols.items():
        event_df[col] = values

    print(f"  total: {n_blocks} blocks augmented with region metrics")
    return event_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment event CSV with per-region synchrony")
    ap.add_argument("--event-csv", required=True)
    ap.add_argument("--focus-rel-json", required=True)
    ap.add_argument(
        "--artifact-root",
        default="/mnt/epilepsia_data/interilca_inter_results/all_data_lns",
    )
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    with open(args.focus_rel_json, "r", encoding="utf-8") as f:
        focus_rel = json.load(f)
    print(f"[INFO] Loaded focus_rel for {len(focus_rel)} subjects")

    df = pd.read_csv(args.event_csv)
    print(f"[INFO] Loaded {len(df)} event rows from {args.event_csv}")

    df = augment_events_with_regions(df, focus_rel, Path(args.artifact_root))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Wrote augmented CSV to {args.output_csv}")

    for region in ["i", "l", "e"]:
        col = f"sync_phase_{region}"
        valid = df[col].notna().sum()
        print(f"  {col}: {valid}/{len(df)} valid ({100*valid/len(df):.1f}%)")


if __name__ == "__main__":
    main()
