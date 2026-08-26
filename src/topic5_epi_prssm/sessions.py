"""Source blocks, recorded intervals and sessions.

Recorded coverage is resolved from the frozen block inventories, never from
event density: a stretch of normally recorded EEG in which no HFO event was
detected is *recorded time with no events*, not missing data.  Two quantities
are kept apart at every session boundary and never conflated:

``metadata_gap_seconds``  unrecorded wall time between two recorded intervals
``event_silence_seconds`` time from the last event of one block to the first of
                          the next, which is always at least as large.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..topic5_source_intervals import _yuquan_inventory_fallback
from .contracts import (
    EPILEPSIAE_BLOCK_INVENTORY,
    FROZEN,
    YUQUAN_BLOCK_INVENTORY,
)

#: Sub-second negative metadata gaps occur where two recorded intervals abut and
#: the inventory rounds start/end differently.  They are clamped to zero and
#: counted; anything below this bound is a real overlap and fails closed.
MAX_ABUTTING_OVERLAP_SECONDS = 2.0


@dataclass(frozen=True)
class BlockInterval:
    block_stem: str
    recording_id: str
    start_epoch: float
    stop_epoch: float
    n_events: int
    first_event_time: float
    last_event_time: float


@dataclass(frozen=True)
class SessionTable:
    """Per-event session assignment plus the boundary bookkeeping behind it."""

    session_index: np.ndarray          # (n_events,) int32, 0-based, chronological
    block_index: np.ndarray            # (n_events,) int32, 0-based over ordered blocks
    blocks: tuple[BlockInterval, ...]
    block_session: np.ndarray          # (n_blocks,) int32
    metadata_gap_seconds: np.ndarray   # (n_blocks,) float64, nan for the first block
    event_silence_seconds: np.ndarray  # (n_blocks,) float64, nan for the first block
    join_seconds: float
    interval_provenance: tuple[str, ...] = ()   # per block
    n_clamped_abutting_gaps: int = 0
    observed_events_during_gap: bool = False

    @property
    def n_sessions(self) -> int:
        return int(self.block_session.max()) + 1 if len(self.block_session) else 0


@lru_cache(maxsize=4)
def _inventory(dataset: str) -> pd.DataFrame:
    path = EPILEPSIAE_BLOCK_INVENTORY if dataset == "epilepsiae" else YUQUAN_BLOCK_INVENTORY
    frame = pd.read_csv(path, low_memory=False)
    frame["subject"] = frame["subject"].astype(str)
    frame["block_stem"] = frame["block_stem"].astype(str)
    frame["recording_id"] = frame["recording_id"].astype(str)
    return frame


def build_sessions(
    subject: str,
    event_times: np.ndarray,
    record_names: np.ndarray,
    *,
    join_seconds: float | None = None,
) -> SessionTable:
    """Assign every event to a session using metadata-resolved recorded intervals.

    A session is a maximal run of recorded intervals whose consecutive
    ``metadata_gap_seconds`` is at most ``join_seconds`` and whose
    ``recording_id`` continuity group agrees.
    """
    join = FROZEN["session_join_seconds"] if join_seconds is None else float(join_seconds)
    dataset, short = subject.split("_", 1)
    inventory = _inventory(dataset)
    rows = inventory[inventory["subject"] == short]
    # Continuity group: the recording for Epilepsiae, the subject for Yuquan.
    # This matches src.topic5_source_intervals.build_source_segments, where the
    # Yuquan inventory's recording_id equals the block stem and would otherwise
    # make every block its own continuity group -- and therefore its own session.
    lookup = {
        str(r.block_stem): (
            float(r.block_start_epoch),
            float(r.block_end_epoch),
            str(r.recording_id) if dataset == "epilepsiae" else short,
        )
        for r in rows.itertuples(index=False)
    }

    names = np.asarray(record_names).astype(str)
    times = np.asarray(event_times, dtype=np.float64)
    if names.shape != times.shape:
        raise ValueError(f"{subject}: record_names and event_times disagree in shape")
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{subject}: event times are not chronological")

    # ordered unique blocks, in event order (verified monotone upstream)
    order: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            order.append(name)

    blocks: list[BlockInterval] = []
    provenance: list[str] = []
    block_index = np.empty(len(times), dtype=np.int32)
    for position, name in enumerate(order):
        mask = names == name
        block_index[mask] = position
        if name in lookup:
            start, stop, recording = lookup[name]
            provenance.append("frozen_block_inventory")
        elif dataset == "yuquan":
            # Nine Yuquan subjects have no inventory row at all; the recorded
            # interval is read from the EDF fixed header by the upstream frozen
            # resolver.  Still metadata, still never event density.
            row = _yuquan_inventory_fallback(subject, name)
            start = float(row["block_start_epoch"])
            stop = float(row["block_end_epoch"])
            recording = short
            provenance.append("selected_edf_header_probe")
        else:
            raise RuntimeError(f"{subject}: no metadata inventory row for block {name}")
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise RuntimeError(f"{subject}: invalid metadata interval for block {name}")
        block_times = times[mask]
        blocks.append(
            BlockInterval(
                block_stem=name,
                recording_id=recording,
                start_epoch=start,
                stop_epoch=stop,
                n_events=int(mask.sum()),
                first_event_time=float(block_times[0]),
                last_event_time=float(block_times[-1]),
            )
        )

    n_blocks = len(blocks)
    metadata_gap = np.full(n_blocks, np.nan)
    event_silence = np.full(n_blocks, np.nan)
    block_session = np.zeros(n_blocks, dtype=np.int32)
    n_clamped = 0
    for i in range(1, n_blocks):
        raw_gap = blocks[i].start_epoch - blocks[i - 1].stop_epoch
        if raw_gap < 0.0:
            if raw_gap < -MAX_ABUTTING_OVERLAP_SECONDS:
                raise RuntimeError(
                    f"{subject}: recorded intervals overlap by {-raw_gap:.3f}s at block "
                    f"{blocks[i].block_stem}; this is not a rounding artefact"
                )
            n_clamped += 1
            raw_gap = 0.0
        metadata_gap[i] = raw_gap
        event_silence[i] = blocks[i].first_event_time - blocks[i - 1].last_event_time
        same_group = blocks[i].recording_id == blocks[i - 1].recording_id
        joined = same_group and (metadata_gap[i] <= join)
        block_session[i] = block_session[i - 1] if joined else block_session[i - 1] + 1

    return SessionTable(
        session_index=block_session[block_index].astype(np.int32),
        block_index=block_index,
        blocks=tuple(blocks),
        block_session=block_session,
        metadata_gap_seconds=metadata_gap,
        event_silence_seconds=event_silence,
        join_seconds=join,
        interval_provenance=tuple(provenance),
        n_clamped_abutting_gaps=n_clamped,
    )


def session_boundaries(session_index: np.ndarray) -> np.ndarray:
    """Boolean per event: True when this event opens a new session."""
    flags = np.zeros(len(session_index), dtype=bool)
    if len(flags):
        flags[0] = True
        flags[1:] = np.diff(np.asarray(session_index)) != 0
    return flags
