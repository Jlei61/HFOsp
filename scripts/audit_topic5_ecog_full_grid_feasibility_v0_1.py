#!/usr/bin/env python3
"""Target-free feasibility audit for the full-grid ECoG RNN experiment.

This stage reads only v2 single-contact detections and canonical block
metadata. It freezes gap-aware block splits and train-only packing contacts,
then measures how many dense-grid events and spatial patches are available.
It does not read raw suffix ranks, RNN outputs, seizure fields, or early-ictal
targets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.group_event_analysis import (  # noqa: E402
    build_windows_from_detections,
    refine_packed_windows_by_all_bool,
)


GRID_ROWS = "ABCDEFGH"
EXPECTED_GRID = tuple(f"G{row}{col}" for row in GRID_ROWS for col in range(1, 9))
BAD_GRID = {"1084": {"GC1"}, "958": set()}
PACK_WIN_SEC = {"958": 0.25, "1084": 0.18}


def stable_fold(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: Path, subject: str) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "subject", "recording_id", "block_no", "block_stem",
            "block_start_epoch", "gap_to_prev_sec", "sample_rate_sql", "head_exists",
            "data_exists", "head_duration_sec",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"block inventory lacks required fields: {sorted(missing)}")
        rows = [row for row in reader if row["subject"] == str(subject)]
    rows.sort(key=lambda row: (int(row["recording_id"]), int(row["block_no"])))
    return rows


def load_complete_seizure_intervals(path: Path, subject: str) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject", "eeg_onset_epoch", "eeg_offset_epoch", "has_complete_eeg_interval"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"seizure inventory lacks required fields: {sorted(missing)}")
        intervals = [
            (float(row["eeg_onset_epoch"]), float(row["eeg_offset_epoch"]))
            for row in reader
            if row["subject"] == str(subject) and row["has_complete_eeg_interval"] == "True"
        ]
    return np.asarray(intervals, dtype=float).reshape((-1, 2))


def exclude_ictal_windows(windows: np.ndarray, block_start_epoch: float, seizures: np.ndarray) -> np.ndarray:
    if windows.size == 0 or seizures.size == 0:
        return np.ones(windows.shape[0], dtype=bool)
    absolute_start = windows[:, 0] + float(block_start_epoch)
    absolute_end = windows[:, 1] + float(block_start_epoch)
    overlap = np.zeros(windows.shape[0], dtype=bool)
    for seizure_start, seizure_end in seizures:
        overlap |= (absolute_end > seizure_start) & (absolute_start < seizure_end)
    return ~overlap


def eligible_row(row: dict[str, str], gpu_root: Path) -> bool:
    stem = row["block_stem"]
    return (
        float(row["sample_rate_sql"]) >= 500.0
        and row["data_exists"] == "True"
        and row["head_exists"] == "True"
        and (gpu_root / row["subject"] / f"{stem}_gpu.npz").exists()
        and float(row["head_duration_sec"]) >= 300.0
    )


def assign_split(rows: list[dict[str, str]], subject: str) -> dict[str, str]:
    """Group contiguous blocks, then hash-order and ratio-balance whole groups."""
    assignments: dict[str, str] = {}
    by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_recording[row["recording_id"]].append(row)
    groups: list[tuple[str, list[str]]] = []
    for recording, rec_rows in sorted(by_recording.items()):
        group = -1
        previous_index: int | None = None
        group_size = 0
        current: list[str] = []
        for row in sorted(rec_rows, key=lambda item: int(item["block_no"])):
            index = int(row["block_no"])
            gap = float(row["gap_to_prev_sec"])
            new_group = previous_index is None or index != previous_index + 1 or gap > 5.0 or group_size >= 6
            if new_group:
                if current:
                    groups.append((f"{recording}|{group}", current))
                group += 1
                group_size = 0
                current = []
            current.append(row["block_stem"])
            previous_index = index
            group_size += 1
        if current:
            groups.append((f"{recording}|{group}", current))

    targets = {"train": 0.70 * len(rows), "validation": 0.15 * len(rows), "test": 0.15 * len(rows)}
    assigned = {name: 0 for name in targets}
    ordered = sorted(groups, key=lambda item: stable_fold(f"ecog-v0.1|{subject}|{item[0]}"))
    for _, stems in ordered:
        split_name = min(
            targets,
            key=lambda name: (assigned[name] / max(targets[name], 1.0), name),
        )
        for stem in stems:
            assignments[stem] = split_name
        assigned[split_name] += len(stems)
    return assignments


def load_grid_detections(path: Path, subject: str) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as artifact:
        names = [str(value) for value in artifact["chns_names"].tolist()]
        index = {name: idx for idx, name in enumerate(names)}
        grid = [name for name in EXPECTED_GRID if name not in BAD_GRID.get(subject, set()) and name in index]
        detections: dict[str, np.ndarray] = {}
        for name in grid:
            arr = np.asarray(artifact["whole_dets"][index[name]], dtype=float)
            detections[name] = arr.reshape((-1, 2)) if arr.size else np.empty((0, 2), dtype=float)
    return grid, detections


def overlaps(windows: np.ndarray, detections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return overlap mask and first overlapping detection midpoint per window."""
    present = np.zeros(windows.shape[0], dtype=bool)
    first_mid = np.full(windows.shape[0], np.nan, dtype=float)
    if windows.size == 0 or detections.size == 0:
        return present, first_mid
    det = detections[np.argsort(detections[:, 0], kind="stable")]
    for event_index, (start, end) in enumerate(windows):
        left = int(np.searchsorted(det[:, 1], start, side="right"))
        right = int(np.searchsorted(det[:, 0], end, side="left"))
        if right > left:
            present[event_index] = True
            first_mid[event_index] = float(np.min(np.mean(det[left:right], axis=1)))
    return present, first_mid


def grid_xy(name: str) -> tuple[int, int]:
    return GRID_ROWS.index(name[1]), int(name[2:]) - 1


def patch_rows(subject: str, names: list[str], event_bool: np.ndarray, event_time: np.ndarray) -> list[dict[str, Any]]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    rows: list[dict[str, Any]] = []
    for size in (2, 3):
        for row0 in range(9 - size):
            for col0 in range(9 - size):
                patch = [f"G{GRID_ROWS[row]}{col + 1}" for row in range(row0, row0 + size) for col in range(col0, col0 + size)]
                available = [name for name in patch if name in name_to_index]
                if len(available) != len(patch):
                    continue
                idx = np.asarray([name_to_index[name] for name in available], dtype=int)
                patch_bool = event_bool[:, idx]
                n_any = int(np.sum(np.any(patch_bool, axis=1)))
                n_ge2 = int(np.sum(np.sum(patch_bool, axis=1) >= 2))
                global_first = np.nanmin(event_time, axis=1)
                patch_values = event_time[:, idx]
                patch_first = np.full(event_time.shape[0], np.nan, dtype=float)
                patch_has_value = np.any(np.isfinite(patch_values), axis=1)
                patch_first[patch_has_value] = np.nanmin(patch_values[patch_has_value], axis=1)
                enters_later = np.isfinite(patch_first) & np.isfinite(global_first) & (patch_first > global_first + 0.005)
                rows.append({
                    "subject": subject,
                    "patch_size": size,
                    "row0": row0,
                    "col0": col0,
                    "contacts": "|".join(available),
                    "n_events_any": n_any,
                    "n_events_two_or_more": n_ge2,
                    "n_events_enters_after_first_5ms": int(np.sum(enters_later)),
                })
    return rows


def audit_subject(
    subject: str,
    inventory: Path,
    seizure_inventory: Path,
    gpu_root: Path,
    window_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = load_inventory(inventory, subject)
    rows = [row for row in raw_rows if eligible_row(row, gpu_root)]
    seizures = load_complete_seizure_intervals(seizure_inventory, subject)
    split = assign_split(rows, subject)
    expected = [name for name in EXPECTED_GRID if name not in BAD_GRID.get(subject, set())]
    train_counts = {name: 0 for name in expected}
    block_grid: dict[str, list[str]] = {}
    block_detections: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        stem = row["block_stem"]
        names, dets = load_grid_detections(gpu_root / subject / f"{stem}_gpu.npz", subject)
        block_grid[stem] = names
        block_detections[stem] = dets
        if split[stem] == "train":
            for name in expected:
                train_counts[name] += int(dets.get(name, np.empty((0, 2))).shape[0])
    values = np.asarray([train_counts[name] for name in expected], dtype=float)
    threshold = float(values.mean() + values.std())
    packing = [name for name in expected if train_counts[name] > threshold]
    fallback = False
    if len(packing) < 4:
        packing = sorted(expected, key=lambda name: (-train_counts[name], name))[:4]
        fallback = True

    block_rows: list[dict[str, Any]] = []
    all_bool: dict[str, list[np.ndarray]] = defaultdict(list)
    all_time: dict[str, list[np.ndarray]] = defaultdict(list)
    names_ref = expected
    for row in rows:
        stem = row["block_stem"]
        dets = block_detections[stem]
        pack_dets = {name: dets.get(name, np.empty((0, 2))) for name in packing}
        windows = build_windows_from_detections(
            pack_dets,
            window_sec=PACK_WIN_SEC[subject],
            ext_ms=30.0,
            chns_thr=0.5,
            time_axis_hz=500.0,
        )
        windows = refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.7)
        win = np.asarray([(item.start, item.end) for item in windows], dtype=float).reshape((-1, 2))
        n_windows_before_ictal_exclusion = int(win.shape[0])
        keep_interictal = exclude_ictal_windows(win, float(row["block_start_epoch"]), seizures)
        win = win[keep_interictal]
        windows = [window for window, keep in zip(windows, keep_interictal) if bool(keep)]
        event_bool = np.zeros((len(windows), len(names_ref)), dtype=bool)
        event_time = np.full((len(windows), len(names_ref)), np.nan, dtype=float)
        for contact_index, name in enumerate(names_ref):
            event_bool[:, contact_index], event_time[:, contact_index] = overlaps(
                win, dets.get(name, np.empty((0, 2), dtype=float))
            )
        cache_dir = window_root / subject
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{stem}.npz"
        temporary = cache_dir / f"{stem}.tmp.npz"
        np.savez_compressed(
            temporary,
            schema_version=np.asarray("topic5_ecog_window_cache_v0.1"),
            subject=np.asarray(subject),
            block_stem=np.asarray(stem),
            split=np.asarray(split[stem]),
            channel_names=np.asarray(names_ref),
            packing_contacts=np.asarray(packing),
            windows=win.astype(np.float64),
            events_bool=event_bool.astype(np.uint8),
            first_detection_midpoint=event_time.astype(np.float32),
        )
        temporary.replace(cache_path)
        counts = event_bool.sum(axis=1)
        which = split[stem]
        all_bool[which].append(event_bool)
        all_time[which].append(event_time)
        block_rows.append({
            "subject": subject,
            "recording_id": row["recording_id"],
            "block_index": int(row["block_no"]),
            "block_stem": stem,
            "split": which,
            "gap_to_prev_sec": float(row["gap_to_prev_sec"]),
            "duration_sec": float(row["head_duration_sec"]),
            "n_grid_contacts": len(block_grid[stem]),
            "n_windows": int(len(windows)),
            "n_windows_excluded_ictal": int(n_windows_before_ictal_exclusion - len(windows)),
            "n_events_ge2": int(np.sum(counts >= 2)),
            "n_events_ge3": int(np.sum(counts >= 3)),
            "window_cache_path": str(cache_path),
            "window_cache_sha256": sha256_file(cache_path),
        })

    split_summary: dict[str, Any] = {}
    patch_output: list[dict[str, Any]] = []
    for which in ("train", "validation", "test"):
        eb = np.concatenate(all_bool[which], axis=0) if all_bool[which] else np.zeros((0, len(names_ref)), bool)
        et = np.concatenate(all_time[which], axis=0) if all_time[which] else np.full((0, len(names_ref)), np.nan)
        counts = eb.sum(axis=1)
        split_summary[which] = {
            "n_blocks": int(sum(row["split"] == which for row in block_rows)),
            "n_events": int(eb.shape[0]),
            "n_events_ge2": int(np.sum(counts >= 2)),
            "n_events_ge3": int(np.sum(counts >= 3)),
            "n_contact_decisions_lower_bound": int(np.sum(np.maximum(counts - 1, 0))),
            "n_contacts_observed": int(np.sum(eb.any(axis=0))) if eb.shape[0] else 0,
            "participant_count_median": float(np.median(counts)) if counts.size else None,
            "participant_count_p90": float(np.quantile(counts, 0.9)) if counts.size else None,
        }
        if which == "train":
            patch_output = patch_rows(subject, names_ref, eb, et)

    channel_rows = [{
        "subject": subject,
        "contact": name,
        "row": grid_xy(name)[0],
        "col": grid_xy(name)[1],
        "is_bad_config": name in BAD_GRID.get(subject, set()),
        "train_detection_count": int(train_counts.get(name, 0)),
        "packing_contact": name in packing,
    } for name in EXPECTED_GRID]
    summary = {
        "subject": subject,
        "expected_grid_contacts": len(expected),
        "bad_grid_contacts": sorted(BAD_GRID.get(subject, set())),
        "n_complete_seizure_intervals_used_for_exclusion": int(seizures.shape[0]),
        "eligible_blocks": len(rows),
        "packing_threshold_mean_plus_sd": threshold,
        "packing_contacts": packing,
        "packing_fallback_top4": fallback,
        "split_summary": split_summary,
        "spec_gate": {
            "train_contacts_ge48": split_summary["train"]["n_contacts_observed"] >= min(48, len(expected)),
            "train_events_ge2_ge5000": split_summary["train"]["n_events_ge2"] >= 5000,
            "test_decisions_ge1000": split_summary["test"]["n_contact_decisions_lower_bound"] >= 1000,
        },
    }
    summary["spec_gate"]["all_pass"] = bool(all(summary["spec_gate"].values()))
    return summary, block_rows, channel_rows, patch_output


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["958", "1084"])
    parser.add_argument("--inventory", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp/results/epilepsiae_block_inventory.csv"))
    parser.add_argument("--seizure-inventory", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp/results/epilepsiae_seizure_inventory.csv"))
    parser.add_argument("--gpu-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp/results/hfo_detection"))
    parser.add_argument("--output", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/feasibility"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    patch_output: list[dict[str, Any]] = []
    for subject in args.subjects:
        summary, blocks, channels, patches = audit_subject(
            subject,
            args.inventory,
            args.seizure_inventory,
            args.gpu_root,
            args.output / "windows",
        )
        summaries.append(summary)
        block_rows.extend(blocks)
        channel_rows.extend(channels)
        patch_output.extend(patches)
        print(json.dumps(summary, indent=2, sort_keys=True))
    write_csv(args.output / "BLOCK_SPLIT.csv", block_rows)
    write_csv(args.output / "GRID_CHANNELS.csv", channel_rows)
    write_csv(args.output / "PATCH_FEASIBILITY.csv", patch_output)
    payload = {
        "schema": "topic5_ecog_full_grid_feasibility_v0.1",
        "target_free": True,
        "subjects": summaries,
        "inventory": str(args.inventory),
        "seizure_inventory": str(args.seizure_inventory),
        "gpu_root": str(args.gpu_root),
    }
    (args.output / "EVENT_FEASIBILITY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
