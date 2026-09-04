"""Fixed physical-time anchors with explicit containment and embargo rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AnchorRecord:
    """One causal scoring anchor for one future horizon.

    ``embargo_stop`` can be later than ``target_stop``.  This permits a 5 min
    endpoint to share a split with a model selected for a 30 min maximum target
    without letting its training anchors approach the split boundary more
    closely than the registered maximum horizon.
    """

    epoch: float
    target_stop: float
    embargo_stop: float
    horizon_seconds: float
    segment_id: int
    session_id: int
    phase: str


def _phase_names(partition: Any) -> tuple[str, ...]:
    names = getattr(partition, "phase_names", None)
    if names is not None:
        return tuple(str(v) for v in names)
    recorded = getattr(partition, "recorded_seconds", None)
    if isinstance(recorded, dict):
        return tuple(k for k in recorded if k != "base_refit")
    # v0.3.2 EvalPartition has a stable four-phase public contract.
    return ("base_fit", "inner_val", "dev_val", "dev_test")


def build_fixed_time_anchors(
    segments: Sequence[Any],
    partition: Any,
    *,
    horizons_seconds: Iterable[float],
    grid_seconds: float = 300.0,
    warmup_seconds: float = 300.0,
    embargo_seconds: float | None = None,
) -> list[AnchorRecord]:
    """Build anchors using geometry only; no target value is read.

    An anchor is emitted only when its whole target and embargo intervals remain
    inside one real coverage segment and one chronological phase.  Consequently
    it cannot cross a recording gap, seizure/postictal cut or split boundary.
    """

    horizons = tuple(float(h) for h in horizons_seconds)
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("horizons_seconds must contain finite positive values")
    if not math.isfinite(grid_seconds) or grid_seconds <= 0:
        raise ValueError("grid_seconds must be finite and positive")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be non-negative")
    global_embargo = max(horizons) if embargo_seconds is None else float(embargo_seconds)
    if global_embargo < max(horizons):
        raise ValueError("embargo_seconds cannot be shorter than the largest target horizon")

    records: list[AnchorRecord] = []
    for seg in segments:
        first = math.ceil((float(seg.start_epoch) + warmup_seconds) / grid_seconds) * grid_seconds
        t = first
        while t < float(seg.stop_epoch):
            phase = str(partition.phase_of(t))
            _phase_lo, phase_hi = partition.bounds(phase)
            for horizon in horizons:
                target_stop = t + horizon
                embargo_stop = t + global_embargo
                last_target_instant = np.nextafter(target_stop, -np.inf)
                last_embargo_instant = np.nextafter(embargo_stop, -np.inf)
                if target_stop > float(seg.stop_epoch) or embargo_stop > float(seg.stop_epoch):
                    continue
                if target_stop > phase_hi or embargo_stop > phase_hi:
                    continue
                if partition.phase_of(float(last_target_instant)) != phase:
                    continue
                if partition.phase_of(float(last_embargo_instant)) != phase:
                    continue
                records.append(
                    AnchorRecord(
                        epoch=float(t),
                        target_stop=float(target_stop),
                        embargo_stop=float(embargo_stop),
                        horizon_seconds=float(horizon),
                        segment_id=int(seg.segment_id),
                        session_id=int(seg.session_id),
                        phase=phase,
                    )
                )
            t += grid_seconds
    validate_anchor_records(records, segments, partition)
    return records


def validate_anchor_records(records: Sequence[AnchorRecord], segments: Sequence[Any], partition: Any) -> None:
    """Fail closed if a stored anchor crosses any scientific boundary."""

    by_id = {int(s.segment_id): s for s in segments}
    for row in records:
        if row.segment_id not in by_id:
            raise ValueError(f"unknown segment_id={row.segment_id}")
        seg = by_id[row.segment_id]
        if row.session_id != int(seg.session_id):
            raise ValueError("anchor session does not match its coverage segment")
        if not (float(seg.start_epoch) <= row.epoch < row.target_stop <= float(seg.stop_epoch)):
            raise ValueError("target is not contained in its real coverage segment")
        if not (row.target_stop <= row.embargo_stop <= float(seg.stop_epoch)):
            raise ValueError("embargo is not contained in its real coverage segment")
        for stop in (row.target_stop, row.embargo_stop):
            if partition.phase_of(float(np.nextafter(stop, -np.inf))) != row.phase:
                raise ValueError("target or embargo crosses a chronological phase boundary")
        if partition.phase_of(row.epoch) != row.phase:
            raise ValueError("anchor phase label is inconsistent")


def independent_window_count(
    segments: Sequence[Any], partition: Any, *, phase: str, horizon_seconds: float
) -> int:
    """Maximum number of disjoint windows inside real covered phase pieces."""

    horizon = float(horizon_seconds)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon_seconds must be finite and positive")
    lo, hi = partition.bounds(phase)
    total = 0
    for seg in segments:
        a = max(float(seg.start_epoch), float(lo))
        b = min(float(seg.stop_epoch), float(hi))
        if b > a:
            total += int(math.floor((b - a) / horizon))
    return total
