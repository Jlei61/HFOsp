#!/usr/bin/env python3
"""
Quick-look utility for SEEG/EEG recordings.

Examples
--------
Preview a 15 s segment from the 5th `.data` file inside a recording folder::

    python preview_seeg_segment.py \
        --record-dir "/Volumes/My Passport/interilca_inter_results/all_data_lns/139/all_recs" \
        --file-index 4 --start 30 --duration 15

Target a specific pair of files directly::

    python preview_seeg_segment.py \
        --data-file "/Volumes/My Passport/inv2/pat_13902/adm_139102/rec_13902102/13902102_0003.data" \
        --head-file "/Volumes/My Passport/inv2/pat_13902/adm_139102/rec_13902102/13902102_0003.head"

Defaults: 10 s segment,仅显示颅内通道（自动过滤 scalp EEG），可选平均参考。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from epilepsiae_utils import eeg_chns, epilepsiae_block


DATA_ROOT = Path("/Volumes/My Passport")
CANDIDATE_RECORD_DIRS = [
    DATA_ROOT / "inv2/pat_13902/adm_139102/rec_13901102",
    DATA_ROOT / "inv2/pat_13902/adm_139102/rec_13902102",
    DATA_ROOT / "inv2/pat_13902/adm_139102/rec_13900102",
    DATA_ROOT / "interilca_inter_results/all_data_lns/139/all_recs",
]
DEFAULT_RECORD_DIR = next((path for path in CANDIDATE_RECORD_DIRS if path.is_dir()), None)


def _resolve_alias_path(path: Path) -> Path:
    if not path.exists():
        return path

    try:
        text = path.read_text(errors="ignore")
    except (OSError, UnicodeDecodeError):
        text = None

    if text and "start_ts=" in text:
        return path

    try:
        raw = path.read_bytes()
    except OSError:
        return path

    alias_text = raw.decode("utf-16le", errors="ignore")
    marker = "epilepsiae/"
    idx = alias_text.find(marker)
    if idx == -1:
        return path

    relative_str = alias_text[idx + len(marker):].split("\x00", 1)[0].strip()
    if not relative_str:
        return path

    relative_path = Path(relative_str)
    candidate = DATA_ROOT / relative_path
    if candidate.exists():
        return candidate

    if candidate.suffix != path.suffix:
        alt_candidate = candidate.with_suffix(path.suffix)
        if alt_candidate.exists():
            return alt_candidate

    if path.suffix == ".data":
        alt = candidate.with_suffix(".data")
        if alt.exists():
            return alt

    if path.suffix == ".head":
        alt = candidate.with_suffix(".head")
        if alt.exists():
            return alt

    return path


def _resolve_files(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.data_file and args.head_file:
        return (
            _resolve_alias_path(Path(args.head_file)),
            _resolve_alias_path(Path(args.data_file)),
        )

    if not args.record_dir:
        raise SystemExit("Either provide --record-dir or both --data-file and --head-file.")

    record_dir = Path(args.record_dir)
    if not record_dir.is_dir():
        raise SystemExit(f"Recording directory not found: {record_dir}")

    data_files = sorted(record_dir.glob("*.data"))
    if not data_files:
        raise SystemExit(f"No .data files found in {record_dir}")

    index = args.file_index
    if index < 0 or index >= len(data_files):
        raise SystemExit(f"file-index {index} out of range (found {len(data_files)} files)")

    data_file = _resolve_alias_path(data_files[index])
    head_file = _resolve_alias_path(data_file.with_suffix(".head"))

    if not data_file.exists():
        raise SystemExit(f"Data file not found: {data_file}")

    if not head_file.exists():
        raise SystemExit(f"Missing head file for {data_file}")

    return head_file, data_file


def _select_channels(channel_names: Iterable[str], exclude_eeg: bool,
                     limit: Iterable[str], max_channels: int | None) -> np.ndarray:
    names = np.array([name.strip() for name in channel_names])
    keep = np.ones(names.shape[0], dtype=bool)

    if exclude_eeg:
        keep &= ~np.isin(names, eeg_chns)

    if limit:
        limit_set = {name.strip() for name in limit}
        keep &= np.array([name in limit_set for name in names])

    selected_indices = np.flatnonzero(keep)
    if selected_indices.size == 0:
        raise SystemExit("No channels left after applying filters; adjust --include-eeg or --channels.")

    if max_channels is not None and selected_indices.size > max_channels:
        selected_indices = selected_indices[:max_channels]

    return selected_indices


def plot_segment(block: epilepsiae_block, start: float, duration: float,
                 indices: np.ndarray, avg_ref: bool) -> None:
    start = max(0.0, start)
    available = block.headInfo["duration_in_sec"]
    if start >= available:
        raise SystemExit(f"Start time {start}s exceeds recording length {available:.2f}s")

    end = min(start + duration, available)
    if end <= start:
        raise SystemExit("Duration too short or start beyond data range.")

    segment = block.fetch_data(start, end)
    segment = segment[indices]

    if avg_ref:
        segment = segment - segment.mean(axis=0, keepdims=True)

    times = np.arange(segment.shape[1]) / block.fs
    scale = np.median(np.abs(segment))
    if scale == 0:
        scale = 1.0
    gap = 6 * scale

    plt.figure(figsize=(12, 6))
    for row, ch_idx in enumerate(indices):
        plt.plot(times, segment[row] + row * gap, linewidth=0.7)

    plt.yticks(np.arange(len(indices)) * gap, [block.chn_names[i].strip() for i in indices])
    plt.xlabel("Time (s)")
    plt.title(f"{Path(block.datafile).name} | {end - start:.1f}s segment")
    plt.tight_layout()
    plt.show()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a short SEEG/EEG segment with simple stacking plot.")
    parser.add_argument("--record-dir", type=Path,
                        default=DEFAULT_RECORD_DIR,
                        help=("Directory containing .data/.head pairs (e.g. .../all_recs). "
                              "Defaults to detected dataset under /Volumes/My Passport if present."))
    parser.add_argument("--data-file", type=Path, help="Direct path to a .data file")
    parser.add_argument("--head-file", type=Path, help="Direct path to the matching .head file")
    parser.add_argument("--file-index", type=int, default=0,
                        help="Index of .data file within --record-dir (default: 0)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Segment duration in seconds (default: 10)")
    parser.add_argument("--channels", nargs="*",
                        help="Explicit channel names to include (after stripping)")
    parser.add_argument("--max-channels", type=int,
                        help="Limit number of channels shown (after filters)")
    parser.add_argument("--include-eeg", action="store_true",
                        help="Include canonical scalp EEG channels in the plot")
    parser.add_argument("--avg-ref", action="store_true",
                        help="Apply common average reference after channel selection")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    head_path, data_path = _resolve_files(args)
    block = epilepsiae_block(str(data_path), str(head_path))

    indices = _select_channels(
        block.chn_names,
        exclude_eeg=not args.include_eeg,
        limit=args.channels,
        max_channels=args.max_channels,
    )

    print(f"Loaded {len(block.chn_names)} channels @ {block.fs} Hz from {data_path}")
    print(f"Showing {len(indices)} channels; recording length {block.headInfo['duration_in_sec']:.1f}s")

    plot_segment(block, args.start, args.duration, indices, args.avg_ref)


if __name__ == "__main__":
    main()