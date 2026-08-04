"""Metadata-driven source interval resolution, shared across Topic 5 analyses.

Extracted verbatim from ``scripts/audit_topic5_event_innovation_v3_0_phase0.py``
(frozen v3.0 audit) so later Topic 5 work can reuse the same metadata-inventory
resolver instead of reimplementing it from event timestamps. ``SourceSegment``
is not defined here: it is a frozen dataclass owned by
``src.topic5_event_innovation_data`` (also used there by
``assign_continuity_units``) and is only re-exported below for callers'
convenience.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.topic5_event_innovation_data import SourceSegment


ROOT = Path(__file__).resolve().parents[1]


def _lagpat_path(subject: str, record_name: str) -> Path:
    dataset, short = subject.split("_", 1)
    directory = (
        Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns") / short / "all_recs"
        if dataset == "epilepsiae"
        else Path("/mnt/yuquan_data/yuquan_24h_edf") / short
    )
    candidates = (
        directory / f"{record_name}_lagPat_withFreqCent.npz",
        directory / f"{record_name}_lagPat.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{subject}: lagPat source missing for {record_name}")


def _montage_hash(subject: str, record_name: str) -> tuple[str, str]:
    path = _lagpat_path(subject, record_name)
    with np.load(path, allow_pickle=True) as artifact:
        names = np.asarray(artifact["chnNames"]).astype(str)
    digest = hashlib.sha256("\0".join(names.tolist()).encode()).hexdigest()
    return digest, str(path)


def _inventory_rows(subject: str, config: Mapping[str, Any]) -> pd.DataFrame:
    dataset, short = subject.split("_", 1)
    inventory_path = ROOT / str(
        config[
            "epilepsiae_block_inventory"
            if dataset == "epilepsiae"
            else "yuquan_block_inventory"
        ]
    )
    frame = pd.read_csv(inventory_path)
    return frame[frame["subject"].astype(str) == short].copy()


def _yuquan_inventory_fallback(subject: str, record_name: str) -> dict[str, Any]:
    short = subject.split("_", 1)[1]
    edf = Path("/mnt/yuquan_data/yuquan_24h_edf") / short / f"{record_name}.edf"
    if not edf.exists():
        raise FileNotFoundError(f"{subject}: EDF metadata source missing: {edf}")
    with edf.open("rb") as stream:
        fixed = stream.read(256)
    if len(fixed) < 256:
        raise ValueError(f"{subject}: EDF header is truncated: {edf}")
    date_string = fixed[168:176].decode("ascii", errors="strict").strip()
    time_string = fixed[176:184].decode("ascii", errors="strict").strip()
    day, month, short_year = map(int, date_string.split("."))
    hour, minute, second = map(int, time_string.split("."))
    year = 2000 + short_year if short_year < 85 else 1900 + short_year
    start = datetime(
        year, month, day, hour, minute, second, tzinfo=timezone.utc
    ).timestamp()
    n_records = int(float(fixed[236:244].decode("ascii").strip()))
    record_duration = float(fixed[244:252].decode("ascii").strip() or "1")
    if n_records <= 0 or record_duration <= 0:
        raise ValueError(f"{subject}: EDF duration metadata is invalid: {edf}")
    duration = float(n_records * record_duration)
    return {
        "subject": short,
        "recording_id": record_name,
        "block_id": record_name,
        "block_stem": record_name,
        "block_start_epoch": float(start),
        "block_end_epoch": float(start + duration),
        "duration_sec": duration,
        "edf_path": str(edf),
    }


def build_source_segments(
    subject: str,
    source_ids: np.ndarray,
    record_names: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[tuple[SourceSegment, ...], list[dict[str, Any]]]:
    """Resolve source intervals from inventories/EDF headers, never event density."""

    dataset, short = subject.split("_", 1)
    inventory = _inventory_rows(subject, config)
    lookup = {str(row.block_stem): row._asdict() for row in inventory.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    segments: list[SourceSegment] = []
    seen: set[str] = set()
    for source_id, record_name in zip(source_ids.astype(str), record_names.astype(str)):
        if source_id in seen:
            continue
        seen.add(source_id)
        row = lookup.get(record_name)
        provenance = "frozen_block_inventory"
        if row is None and dataset == "yuquan":
            row = _yuquan_inventory_fallback(subject, record_name)
            provenance = "selected_edf_header_probe"
        if row is None:
            raise RuntimeError(f"{subject}: no metadata inventory row for {record_name}")
        start = float(row["block_start_epoch"])
        stop = float(row["block_end_epoch"])
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise RuntimeError(f"{subject}: invalid metadata interval for {record_name}")
        montage_hash, lagpat_path = _montage_hash(subject, record_name)
        group = str(row["recording_id"]) if dataset == "epilepsiae" else short
        segment = SourceSegment(
            source_id=str(source_id),
            start_time=start,
            stop_time=stop,
            continuity_group=group,
            montage_hash=montage_hash,
            continuity_verified=True,
        )
        segments.append(segment)
        records.append(
            {
                "subject": subject,
                "dataset": dataset,
                "source_id": str(source_id),
                "record_name": record_name,
                "start_time": start,
                "stop_time": stop,
                "duration_seconds": stop - start,
                "continuity_group": group,
                "montage_hash": montage_hash,
                "metadata_provenance": provenance,
                "lagpat_path": lagpat_path,
            }
        )
    if not segments:
        raise RuntimeError(f"{subject}: no source metadata segments")
    return tuple(segments), records
