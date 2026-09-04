"""Causal slow-level controls and exposure-overlap audits for v0.3.4 H3."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ControlDefinition:
    name: str
    causal_at_anchor: bool
    allowed_primary_comparator: bool
    uses_future_inputs_for_earlier_anchors: bool
    description: str

    def as_dict(self) -> dict:
        return asdict(self)


def rolling_prefix_slow_level(
    event_times: np.ndarray,
    event_segments: np.ndarray,
    anchor_times: np.ndarray,
    anchor_segments: np.ndarray,
    *,
    half_life_seconds: float,
    event_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Strictly causal EWMA event-rate/mark level at each anchor.

    Events at exactly the anchor time are excluded.  The state resets at every
    real coverage segment.  Column 0 is an exponentially weighted event rate;
    remaining columns are weighted event-value means when supplied.
    """

    et = np.asarray(event_times, dtype=np.float64)
    es = np.asarray(event_segments, dtype=np.int64)
    at = np.asarray(anchor_times, dtype=np.float64)
    ass = np.asarray(anchor_segments, dtype=np.int64)
    if et.shape != es.shape or at.shape != ass.shape or et.ndim != 1 or at.ndim != 1:
        raise ValueError("times and segment arrays must be aligned 1-D arrays")
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    values = None if event_values is None else np.asarray(event_values, dtype=np.float64)
    if values is not None:
        if values.shape[0] != et.size:
            raise ValueError("event_values first dimension must align to events")
        if values.ndim == 1:
            values = values[:, None]
    out_dim = 1 + (0 if values is None else values.shape[1])
    out = np.zeros((at.size, out_dim), dtype=np.float64)
    latest = np.full(at.size, np.nan, dtype=np.float64)
    tau = float(half_life_seconds) / np.log(2.0)

    for segment in np.unique(ass):
        ai = np.flatnonzero(ass == segment)
        ei = np.flatnonzero(es == segment)
        ai = ai[np.argsort(at[ai], kind="stable")]
        ei = ei[np.argsort(et[ei], kind="stable")]
        mass = 0.0
        weighted = np.zeros(out_dim - 1, dtype=np.float64)
        cursor = 0
        last_time = float(et[ei[0]]) if ei.size else (float(at[ai[0]]) if ai.size else 0.0)
        last_event = np.nan
        for aidx in ai:
            anchor = float(at[aidx])
            while cursor < ei.size and float(et[ei[cursor]]) < anchor:
                eidx = int(ei[cursor])
                now = float(et[eidx])
                decay = float(np.exp(-(now - last_time) / tau))
                mass *= decay
                weighted *= decay
                mass += 1.0
                if values is not None:
                    weighted += values[eidx]
                last_time = now
                last_event = now
                cursor += 1
            decay = float(np.exp(-(anchor - last_time) / tau))
            anchor_mass = mass * decay
            anchor_weighted = weighted * decay
            out[aidx, 0] = anchor_mass / tau
            if values is not None and anchor_mass > 0:
                out[aidx, 1:] = anchor_weighted / anchor_mass
            latest[aidx] = last_event
    finite = np.isfinite(latest)
    strict = bool(np.all(latest[finite] < at[finite]))
    return out, {
        "definition": "strictly prior event EWMA, reset at true coverage segment",
        "half_life_seconds": float(half_life_seconds),
        "causal_at_anchor": strict,
        "latest_source_time_before_anchor": strict,
        "n_anchors": int(at.size),
        "n_anchors_with_prior_event": int(finite.sum()),
    }


def selection_period_mean_oracle(values: np.ndarray) -> tuple[np.ndarray, ControlDefinition]:
    """Repeat an all-selection-period mean, explicitly labelled noncausal.

    It uses no future target, but later *inputs* contribute to early anchors.
    It is an informative retrospective period-level upper bound, never a
    primary prospective comparator.
    """

    x = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(x, axis=0, keepdims=True)
    repeated = np.repeat(mean, x.shape[0], axis=0)
    return repeated, ControlDefinition(
        name="selection_period_mean_oracle",
        causal_at_anchor=False,
        allowed_primary_comparator=False,
        uses_future_inputs_for_earlier_anchors=True,
        description="retrospective period-level input oracle; no target labels, but later inputs inform earlier anchors",
    )


def interval_overlap_fraction(real_start: float, real_stop: float, donor_start: float, donor_stop: float) -> float:
    """Intersection length divided by the real exposure length."""

    if real_stop <= real_start or donor_stop <= donor_start:
        raise ValueError("intervals must have positive length")
    overlap = max(0.0, min(float(real_stop), float(donor_stop)) - max(float(real_start), float(donor_start)))
    return float(overlap / (float(real_stop) - float(real_start)))


def event_window_overlap_fraction(real_lo: int, real_hi: int, donor_lo: int, donor_hi: int) -> float:
    """Shared event rows divided by the real N-event exposure length."""

    if real_hi <= real_lo or donor_hi <= donor_lo:
        raise ValueError("event windows must have positive length")
    overlap = max(0, min(int(real_hi), int(donor_hi)) - max(int(real_lo), int(donor_lo)))
    return float(overlap / (int(real_hi) - int(real_lo)))


def audit_replacement_event_overlap(
    real_lo: np.ndarray,
    real_hi: np.ndarray,
    real_segment: np.ndarray,
    donor_lo: np.ndarray,
    donor_hi: np.ndarray,
    donor_segment: np.ndarray,
    *,
    max_allowed_fraction: float = 0.0,
) -> dict:
    """Exact row-overlap audit for every real/replacement exposure pair.

    Event indices can overlap only inside the same coverage segment.  The
    returned per-pair values must be persisted by a human runner; a summary
    alone is insufficient to detect one copied target window.
    """

    arrays = [np.asarray(v, dtype=np.int64) for v in (
        real_lo, real_hi, real_segment, donor_lo, donor_hi, donor_segment,
    )]
    if len({v.shape for v in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("real and donor window arrays must be aligned 1-D arrays")
    rlo, rhi, rseg, dlo, dhi, dseg = arrays
    fractions = np.zeros(rlo.size, dtype=np.float64)
    for i in range(rlo.size):
        if rseg[i] == dseg[i]:
            fractions[i] = event_window_overlap_fraction(rlo[i], rhi[i], dlo[i], dhi[i])
    threshold = float(max_allowed_fraction)
    return {
        "n_pairs": int(fractions.size),
        "n_pairs_with_overlap": int(np.sum(fractions > 0.0)),
        "median_overlap_fraction": float(np.median(fractions)) if fractions.size else None,
        "max_overlap_fraction": float(np.max(fractions)) if fractions.size else None,
        "max_allowed_fraction": threshold,
        "passed": bool(np.all(fractions <= threshold + 1e-12)),
        "per_pair_overlap_fraction": fractions.tolist(),
    }
