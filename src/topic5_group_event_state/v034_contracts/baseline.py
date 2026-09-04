"""Causal multiscale event-history baseline on fixed physical-time anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TAU_SECONDS: tuple[float, ...] = (300.0, 1800.0, 7200.0, 10800.0, 21600.0, 43200.0)


@dataclass(frozen=True)
class BaselineMatrix:
    values: np.ndarray
    names: tuple[str, ...]
    provenance: dict[str, Any]


def build_multiscale_history(
    *,
    anchor_times: np.ndarray,
    anchor_segment: np.ndarray,
    event_times: np.ndarray,
    event_segment: np.ndarray,
    event_features: Mapping[str, np.ndarray] | None = None,
    segment_bounds: Mapping[int, tuple[float, float]] | None = None,
    tau_seconds: Sequence[float] = TAU_SECONDS,
) -> BaselineMatrix:
    """Build a strictly prefix-only ``B_multiscale`` matrix.

    Events exactly at an anchor are excluded (``event_time < anchor_time``).
    Accumulators reset across coverage segments.  No future-seizure quantity is
    accepted or constructed here.
    """

    at = np.asarray(anchor_times, dtype=np.float64)
    aseg = np.asarray(anchor_segment, dtype=np.int64)
    et = np.asarray(event_times, dtype=np.float64)
    eseg = np.asarray(event_segment, dtype=np.int64)
    if at.ndim != 1 or aseg.shape != at.shape or et.ndim != 1 or eseg.shape != et.shape:
        raise ValueError("time and segment arrays must be aligned one-dimensional arrays")
    if not (np.isfinite(at).all() and np.isfinite(et).all()):
        raise ValueError("times must be finite")
    taus = tuple(float(v) for v in tau_seconds)
    if not taus or any(not np.isfinite(v) or v <= 0 for v in taus):
        raise ValueError("tau_seconds must be finite and positive")

    features: dict[str, np.ndarray] = {}
    for name, raw in (event_features or {}).items():
        if "seizure" in name.lower() and "previous" not in name.lower():
            raise ValueError(f"future/concurrent seizure feature is forbidden in interictal baseline: {name}")
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != et.size or not np.isfinite(x).all():
            raise ValueError(f"invalid event feature {name!r}")
        features[str(name)] = x

    columns: list[np.ndarray] = []
    names: list[str] = []
    rate_block = np.zeros((at.size, len(taus)), dtype=np.float64)
    feature_blocks = {
        name: np.zeros((at.size, len(taus) * x.shape[1]), dtype=np.float64)
        for name, x in features.items()
    }
    since = np.full(at.size, np.inf, dtype=np.float64)

    for seg in np.unique(aseg):
        qi = np.flatnonzero(aseg == seg)
        ei = np.flatnonzero(eseg == seg)
        qi = qi[np.argsort(at[qi], kind="stable")]
        ei = ei[np.argsort(et[ei], kind="stable")]
        for out_pos in qi:
            t = float(at[out_pos])
            past = ei[et[ei] < t]
            if past.size:
                age = t - et[past]
                since[out_pos] = float(age[-1])
                for j, tau in enumerate(taus):
                    weight = np.exp(-age / tau)
                    rate_block[out_pos, j] = weight.sum() / tau
                    denom = float(weight.sum())
                    for name, x in features.items():
                        width = x.shape[1]
                        feature_blocks[name][out_pos, j * width:(j + 1) * width] = (
                            (weight[:, None] * x[past]).sum(axis=0) / max(denom, 1e-12)
                        )

    columns.append(np.log1p(rate_block))
    names.extend(f"log_rate_tau{int(tau)}" for tau in taus)
    finite_since = np.isfinite(since)
    columns.append(np.log1p(np.where(finite_since, since, 7 * 86400.0))[:, None])
    names.append("log_time_since_last_event")
    columns.append(finite_since.astype(np.float64)[:, None])
    names.append("has_previous_event")
    for feature_name, block in feature_blocks.items():
        columns.append(block)
        width = features[feature_name].shape[1]
        for tau in taus:
            names.extend(f"{feature_name}[{j}]_tau{int(tau)}" for j in range(width))

    day_phase = 2.0 * np.pi * (at % 86400.0) / 86400.0
    columns.extend((np.sin(day_phase)[:, None], np.cos(day_phase)[:, None]))
    names.extend(("clock_sin_day", "clock_cos_day"))
    if segment_bounds is not None:
        into = np.asarray([t - segment_bounds[int(seg)][0] for t, seg in zip(at, aseg)], dtype=np.float64)
        duration = np.asarray([
            segment_bounds[int(seg)][1] - segment_bounds[int(seg)][0] for seg in aseg
        ], dtype=np.float64)
        columns.extend((np.log1p(np.clip(into, 0, None))[:, None], (into / np.maximum(duration, 1.0))[:, None]))
        names.extend(("log_seconds_into_segment", "fraction_through_segment"))

    matrix = np.concatenate(columns, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("baseline matrix contains non-finite values")
    return BaselineMatrix(
        values=matrix,
        names=tuple(names),
        provenance={
            "name": "B_multiscale",
            "causal": True,
            "event_inclusion": "event_time < anchor_time",
            "tau_seconds": list(taus),
            "resets_at": "coverage_segment",
            "contains_future_seizure_information": False,
            "normalization_contract": "fit on TRAIN only; frozen thereafter",
        },
    )
