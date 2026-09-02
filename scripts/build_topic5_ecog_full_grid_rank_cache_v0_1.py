#!/usr/bin/env python3
"""Build full-grid ECoG rank-set caches from frozen window caches.

The worker is resumable and shardable. It reads one raw Epilepsiae block at a
time, computes CAR ripple-band spectrogram centroids for every available grid
contact, verifies participation against the target-free window cache, and
writes event-by-contact dense ranks with non-participants fixed to -1.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.group_event_analysis import (  # noqa: E402
    EventWindow,
    compute_centroid_matrix_spectrogram,
    lag_rank_from_centroids,
)
from src.preprocessing import load_epilepsiae_block  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def densify_rank_sets(rank: np.ndarray) -> np.ndarray:
    """Map tie-aware competition ranks to consecutive rank-set labels."""
    values = np.asarray(rank)
    output = np.full(values.shape, -1, dtype=np.int16)
    for event in range(values.shape[1]):
        observed = np.unique(values[:, event][values[:, event] >= 0])
        for dense, old in enumerate(observed):
            output[values[:, event] == old, event] = dense
    return output


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_block_lookup(path: Path, subject: str) -> dict[str, dict[str, str]]:
    rows = [row for row in load_csv(path) if row["subject"] == str(subject)]
    return {row["block_stem"]: row for row in rows}


def load_detection_dict(path: Path, channel_names: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as artifact:
        names = [str(value) for value in artifact["chns_names"].tolist()]
        index = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in channel_names if name not in index]
        if missing:
            raise ValueError(f"GPU artifact lacks grid contacts: {missing}")
        output: dict[str, np.ndarray] = {}
        for name in channel_names:
            arr = np.asarray(artifact["whole_dets"][index[name]], dtype=float)
            output[name] = arr.reshape((-1, 2)) if arr.size else np.empty((0, 2), dtype=float)
    return output


def sparse_window_centroids(
    *,
    inventory_row: dict[str, str],
    channel_names: list[str],
    windows: list[EventWindow],
    detections: dict[str, np.ndarray],
    bad_channels: list[str],
    pad_sec: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Read and filter only padded unions of event windows.

    A one-second default pad keeps every scored window far from the IIR and
    band-pass boundaries. This mode is intended for sparse ECoG blocks and is
    validated against full-block ranks before formal aggregation.
    """
    from scipy.signal import butter, filtfilt

    if not windows:
        return (
            np.empty((len(channel_names), 0), dtype=float),
            np.empty((len(channel_names), 0), dtype=bool),
            float(inventory_row["head_sample_rate"]),
            float(inventory_row["block_start_epoch"]),
        )
    duration = float(inventory_row["head_duration_sec"])
    expanded = sorted([
        (max(0.0, float(window.start) - float(pad_sec)),
         min(duration, float(window.end) + float(pad_sec)), index)
        for index, window in enumerate(windows)
    ])
    chunks: list[tuple[float, float, list[int]]] = []
    for start, end, index in expanded:
        if chunks and start <= chunks[-1][1]:
            old_start, old_end, indices = chunks[-1]
            chunks[-1] = (old_start, max(old_end, end), [*indices, index])
        else:
            chunks.append((start, end, [index]))

    centroids = np.full((len(channel_names), len(windows)), np.nan, dtype=float)
    participation = np.zeros((len(channel_names), len(windows)), dtype=bool)
    sfreq_ref: float | None = None
    block_start_ref: float | None = None
    for start, end, indices in chunks:
        raw = load_epilepsiae_block(
            data_path=inventory_row["data_path"],
            head_path=inventory_row["head_path"],
            reference="car",
            drop_channels=bad_channels,
            crop_start_sec=start,
            crop_duration_sec=end - start,
        )
        sfreq = float(raw.sfreq)
        actual_start = round(float(start) * sfreq) / sfreq
        if sfreq_ref is None:
            sfreq_ref = sfreq
            # SQL inventory is the canonical Epilepsiae time source; .head is
            # known to carry an eight-hour timezone offset in some records.
            block_start_ref = float(inventory_row["block_start_epoch"])
        elif sfreq != sfreq_ref:
            raise RuntimeError("sample rate changed between sparse chunks")
        raw_index = {str(name): idx for idx, name in enumerate(raw.ch_names)}
        missing = [name for name in channel_names if name not in raw_index]
        if missing:
            raise ValueError(f"sparse raw chunk lacks grid contacts: {missing}")
        signal = np.asarray(raw.data[[raw_index[name] for name in channel_names]], dtype=np.float64)
        del raw
        nyquist = 0.5 * sfreq
        b, a = butter(4, [80.0 / nyquist, 250.0 / nyquist], btype="band")
        signal_band = filtfilt(b, a, signal, axis=-1)
        del signal
        local_windows = [windows[index] for index in indices]
        # The loader rounds crop_start_sec to an integer sample. Use that
        # actual rounded origin, not the requested floating-point start.
        local_centroids, local_bool = compute_centroid_matrix_spectrogram(
            windows=local_windows,
            detections=detections,
            ch_names=channel_names,
            x_band=signal_band,
            sfreq=sfreq,
            start_sec=actual_start,
            spec_freq_range=(80.0, 250.0),
            centroid_power=3.0,
        )
        centroids[:, indices] = local_centroids
        participation[:, indices] = local_bool
        del signal_band
        gc.collect()
    assert sfreq_ref is not None and block_start_ref is not None
    return centroids, participation, sfreq_ref, block_start_ref


def cache_is_current(
    path: Path,
    window_sha256: str,
    tie_tol_ms: float,
    read_mode: str,
    sparse_pad_sec: float,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            return (
                str(cache["schema_version"].item()) == "topic5_ecog_rank_cache_v0.1"
                and str(cache["window_cache_sha256"].item()) == window_sha256
                and float(cache["tie_tol_ms"].item()) == float(tie_tol_ms)
                and bool(cache["bad_channels_removed_before_car"].item())
                and bool(cache["dense_rank_sets"].item())
                and str(cache["raw_read_mode"].item()) == str(read_mode)
                and float(cache["sparse_pad_sec"].item()) == float(sparse_pad_sec)
                and cache["ranks"].ndim == 2
                and np.isfinite(cache["lag_sec"][cache["participation"].astype(bool)]).all()
            )
    except Exception:
        return False


def process_block(
    subject: str,
    row: dict[str, str],
    inventory_row: dict[str, str],
    gpu_root: Path,
    output_root: Path,
    tie_tol_ms: float,
    read_mode: str,
    sparse_pad_sec: float,
    force: bool,
) -> dict[str, Any]:
    from scipy.signal import butter, filtfilt

    stem = row["block_stem"]
    window_path = Path(row["window_cache_path"])
    if not window_path.is_absolute():
        window_path = ROOT / window_path
    if not window_path.exists():
        raise FileNotFoundError(f"missing window cache: {window_path}")
    window_sha = sha256_file(window_path)
    if window_sha != row["window_cache_sha256"]:
        raise RuntimeError(f"window cache hash mismatch for {stem}")

    out_dir = output_root / subject / "per_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.npz"
    if not force and cache_is_current(out_path, window_sha, tie_tol_ms, read_mode, sparse_pad_sec):
        return {"block_stem": stem, "status": "skipped_current", "output_path": str(out_path)}

    started = time.time()
    with np.load(window_path, allow_pickle=False) as cache:
        windows = np.asarray(cache["windows"], dtype=float)
        cached_bool = np.asarray(cache["events_bool"], dtype=bool)
        midpoint = np.asarray(cache["first_detection_midpoint"], dtype=float)
        channel_names = [str(value) for value in cache["channel_names"].tolist()]
        split = str(cache["split"].item())

    gpu_path = gpu_root / subject / f"{stem}_gpu.npz"
    if len(windows) == 0:
        # A target-free empty window cache cannot acquire a rank after reading
        # the signal. Avoid loading and bandpassing a full one-hour ECoG block.
        n_channels = len(channel_names)
        temporary = out_dir / f"{stem}.tmp.npz"
        np.savez_compressed(
            temporary,
            schema_version=np.asarray("topic5_ecog_rank_cache_v0.1"),
            subject=np.asarray(subject),
            block_stem=np.asarray(stem),
            split=np.asarray(split),
            channel_names=np.asarray(channel_names),
            windows=np.empty((0, 2), dtype=np.float64),
            event_epoch=np.empty(0, dtype=np.float64),
            participation=np.empty((0, n_channels), dtype=np.uint8),
            lag_sec=np.empty((0, n_channels), dtype=np.float32),
            ranks=np.empty((0, n_channels), dtype=np.int16),
            midpoint_lag_sec=np.empty((0, n_channels), dtype=np.float32),
            midpoint_ranks=np.empty((0, n_channels), dtype=np.int16),
            source_event_index=np.empty(0, dtype=np.int32),
            tie_tol_ms=np.asarray(float(tie_tol_ms)),
            sfreq=np.asarray(float(inventory_row["head_sample_rate"])),
            block_start_epoch=np.asarray(float(inventory_row["block_start_epoch"])),
            window_cache_sha256=np.asarray(window_sha),
            gpu_sha256=np.asarray(sha256_file(gpu_path)),
            data_path=np.asarray(inventory_row["data_path"]),
            head_path=np.asarray(inventory_row["head_path"]),
            bad_channels_removed_before_car=np.asarray(True),
            dense_rank_sets=np.asarray(True),
            raw_read_mode=np.asarray(read_mode),
            sparse_pad_sec=np.asarray(float(sparse_pad_sec)),
        )
        temporary.replace(out_path)
        return {
            "block_stem": stem,
            "status": "completed_empty_without_raw_read",
            "split": split,
            "n_events_input": 0,
            "n_events_output": 0,
            "n_events_dropped_nonfinite": 0,
            "n_channels": n_channels,
            "n_participations": 0,
            "runtime_sec": float(time.time() - started),
            "output_path": str(out_path),
            "output_sha256": sha256_file(out_path),
        }

    detections = load_detection_dict(gpu_path, channel_names)
    event_windows = [
        EventWindow(start=float(start), end=float(end), event_id=index)
        for index, (start, end) in enumerate(windows)
    ]
    bad_channels = ["GC1"] if subject == "1084" else []
    if read_mode == "sparse":
        centroids, computed_bool, sfreq, block_start_epoch = sparse_window_centroids(
            inventory_row=inventory_row,
            channel_names=channel_names,
            windows=event_windows,
            detections=detections,
            bad_channels=bad_channels,
            pad_sec=sparse_pad_sec,
        )
    else:
        raw = load_epilepsiae_block(
            data_path=inventory_row["data_path"],
            head_path=inventory_row["head_path"],
            reference="car",
            drop_channels=bad_channels,
        )
        sfreq = float(raw.sfreq)
        raw_index = {str(name): idx for idx, name in enumerate(raw.ch_names)}
        missing_raw = [name for name in channel_names if name not in raw_index]
        if missing_raw:
            raise ValueError(f"raw block lacks grid contacts: {missing_raw}")
        signal = np.asarray(raw.data[[raw_index[name] for name in channel_names]], dtype=np.float64)
        block_start_epoch = float(inventory_row["block_start_epoch"])
        del raw
        gc.collect()
        nyquist = 0.5 * sfreq
        b, a = butter(4, [80.0 / nyquist, 250.0 / nyquist], btype="band")
        signal_band = filtfilt(b, a, signal, axis=-1)
        del signal
        gc.collect()
        centroids, computed_bool = compute_centroid_matrix_spectrogram(
            windows=event_windows,
            detections=detections,
            ch_names=channel_names,
            x_band=signal_band,
            sfreq=sfreq,
            start_sec=0.0,
            spec_freq_range=(80.0, 250.0),
            centroid_power=3.0,
        )
        del signal_band
        gc.collect()
    if sfreq < 500.0:
        raise ValueError(f"Nyquist-invalid sfreq={sfreq} for {stem}")

    computed_event_contact = computed_bool.T
    if not np.array_equal(computed_event_contact, cached_bool):
        mismatch = int(np.sum(computed_event_contact != cached_bool))
        raise RuntimeError(f"participation mismatch for {stem}: {mismatch} cells")

    finite_bool = computed_bool & np.isfinite(centroids)
    keep_event = finite_bool.sum(axis=0) >= 2
    dropped_nonfinite = int(np.sum(~keep_event))
    centroids = centroids[:, keep_event]
    finite_bool = finite_bool[:, keep_event]
    windows_kept = windows[keep_event]
    midpoint_kept = midpoint[keep_event].T
    lag_sec, rank = lag_rank_from_centroids(
        centroids,
        finite_bool,
        align="first_centroid",
        tie_tol_ms=float(tie_tol_ms),
    )
    midpoint_lag, midpoint_rank = lag_rank_from_centroids(
        midpoint_kept,
        finite_bool,
        align="first_centroid",
        tie_tol_ms=float(tie_tol_ms),
    )
    rank = densify_rank_sets(rank)
    midpoint_rank = densify_rank_sets(midpoint_rank)
    ranks_event = rank.T.astype(np.int16)
    participation_event = finite_bool.T.astype(np.uint8)
    if not np.array_equal(ranks_event < 0, participation_event == 0):
        raise RuntimeError(f"rank/participation mask mismatch for {stem}")

    temporary = out_dir / f"{stem}.tmp.npz"
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("topic5_ecog_rank_cache_v0.1"),
        subject=np.asarray(subject),
        block_stem=np.asarray(stem),
        split=np.asarray(split),
        channel_names=np.asarray(channel_names),
        windows=windows_kept.astype(np.float64),
        event_epoch=(block_start_epoch + windows_kept.mean(axis=1)).astype(np.float64),
        participation=participation_event,
        lag_sec=lag_sec.T.astype(np.float32),
        ranks=ranks_event,
        midpoint_lag_sec=midpoint_lag.T.astype(np.float32),
        midpoint_ranks=midpoint_rank.T.astype(np.int16),
        source_event_index=np.flatnonzero(keep_event).astype(np.int32),
        tie_tol_ms=np.asarray(float(tie_tol_ms)),
        sfreq=np.asarray(sfreq),
        block_start_epoch=np.asarray(block_start_epoch),
        window_cache_sha256=np.asarray(window_sha),
        gpu_sha256=np.asarray(sha256_file(gpu_path)),
        data_path=np.asarray(inventory_row["data_path"]),
        head_path=np.asarray(inventory_row["head_path"]),
        bad_channels_removed_before_car=np.asarray(True),
        dense_rank_sets=np.asarray(True),
        raw_read_mode=np.asarray(read_mode),
        sparse_pad_sec=np.asarray(float(sparse_pad_sec)),
    )
    temporary.replace(out_path)
    return {
        "block_stem": stem,
        "status": "completed",
        "split": split,
        "n_events_input": int(windows.shape[0]),
        "n_events_output": int(ranks_event.shape[0]),
        "n_events_dropped_nonfinite": dropped_nonfinite,
        "n_channels": len(channel_names),
        "n_participations": int(participation_event.sum()),
        "runtime_sec": float(time.time() - started),
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=("958", "1084"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tie-tol-ms", type=float, default=5.0)
    parser.add_argument("--read-mode", choices=("full", "sparse"), default="full")
    parser.add_argument("--sparse-pad-sec", type=float, default=1.0)
    parser.add_argument("--feasibility-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/feasibility"))
    parser.add_argument("--inventory", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp/results/epilepsiae_block_inventory.csv"))
    parser.add_argument("--gpu-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp/results/hfo_detection"))
    parser.add_argument("--output-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"))
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard_id < args.n_shards:
        raise ValueError("require 0 <= shard-id < n-shards")

    split_rows = [
        row for row in load_csv(args.feasibility_root / "BLOCK_SPLIT.csv")
        if row["subject"] == args.subject
    ]
    split_rows.sort(key=lambda row: (int(row["recording_id"]), int(row["block_index"])))
    selected = split_rows[args.shard_id::args.n_shards]
    if args.smoke:
        selected = selected[:1]
    inventory = load_block_lookup(args.inventory, args.subject)
    log_dir = args.output_root / args.subject
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard_{args.shard_id:02d}_of_{args.n_shards:02d}.json"
    log: dict[str, Any] = {
        "schema": "topic5_ecog_rank_cache_worker_v0.1",
        "subject": args.subject,
        "shard_id": args.shard_id,
        "n_shards": args.n_shards,
        "pid": os.getpid(),
        "records": [],
        "failures": [],
    }
    atomic_json(log_path, log)
    for row in selected:
        stem = row["block_stem"]
        try:
            result = process_block(
                args.subject,
                row,
                inventory[stem],
                args.gpu_root,
                args.output_root,
                args.tie_tol_ms,
                args.read_mode,
                args.sparse_pad_sec,
                args.force,
            )
            log["records"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"block_stem": stem, "type": type(exc).__name__, "error": str(exc)}
            log["failures"].append(failure)
            print(json.dumps(failure, sort_keys=True), flush=True)
        atomic_json(log_path, log)
    log["complete"] = True
    atomic_json(log_path, log)
    if log["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
