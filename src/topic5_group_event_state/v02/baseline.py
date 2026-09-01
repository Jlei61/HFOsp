"""``B_multiscale``: the interpretable multiscale baseline every arm is judged against.

If this baseline matches a recurrent producer, that is a result in its own right
(CC 3.1) -- it would say the useful part of "history" is a handful of exponential
moving averages, not a learned state.  So it is built to be genuinely strong:
four physical timescales, every mark family, the clock, the session geometry and
the seizure bookkeeping.

Two design rules that keep the comparison honest:

* every accumulator is reset at the start of a coverage segment and decayed by
  the *real* elapsed seconds, so no feature can see across a gap, a seizure or a
  postictal exclusion (clauses C2, C3, C8);
* seizure-derived columns live here and only here.  ``B_multiscale`` is a
  nuisance model and is allowed to know when the last seizure was (CC 3.1); the
  representation producers are not (DC 11).  Handing the baseline that knowledge
  can only make the increment ``B -> B+S`` harder to obtain, which is the
  conservative direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .marks import EventMarks
from .timeline import AnchorGrid, CoverageSegment

# Physical timescales named by the plan (SP 3): 1, 5, 30, 120 minutes.
EWMA_TAU_SECONDS: tuple[float, ...] = (60.0, 300.0, 1800.0, 7200.0)

# The per-contact participation field is carried at two scales only.  All four
# would add 4C columns for a 52-contact patient; the rule is fixed for every
# patient rather than chosen per patient.
FIELD_TAU_SECONDS: tuple[float, ...] = (300.0, 7200.0)

# Shrinkage of an EWMA-weighted mean towards the TRAIN mean (= 0 after
# standardisation) when almost no weight has accumulated yet.
MEAN_PRIOR_WEIGHT = 1e-3

SECONDS_PER_DAY = 86400.0
SEIZURE_TIME_CAP_SECONDS = 7 * SECONDS_PER_DAY
POSTICTAL_INDICATOR_SECONDS = 6 * 3600.0


# SP A1 lists a "current clock / event baseline" *before* the multiscale one:
# what can be said from the calendar and from the fact that an event just
# happened, with no history summary of any kind.  Seizure bookkeeping is
# deliberately excluded from it -- that belongs to the nuisance model, not to
# "the clock".
CLOCK_ONLY_FEATURES = (
    "log_time_since_last_event",
    "has_previous_event",
    "log_events_so_far_in_segment",
    "clock_sin_day",
    "clock_cos_day",
    "clock_sin_half_day",
    "clock_cos_half_day",
    "log_seconds_into_segment",
    "log_seconds_left_in_segment",
    "fraction_through_segment",
    "log_segment_duration",
    "days_since_recording_start",
)


@dataclass(frozen=True)
class BaselineFeatures:
    x: np.ndarray                     # (A, F) float64
    names: tuple[str, ...]
    notes: dict[str, Any]

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    def clock_only_columns(self) -> np.ndarray:
        lookup = {name: i for i, name in enumerate(self.names)}
        missing = [n for n in CLOCK_ONLY_FEATURES if n not in lookup]
        if missing:
            raise KeyError(f"clock-only baseline is missing columns: {missing}")
        return np.array([lookup[n] for n in CLOCK_ONLY_FEATURES], dtype=np.int64)


def _ewma_after_each_event(
    event_times: np.ndarray,
    event_segment: np.ndarray,
    values: np.ndarray | None,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal EWMA numerator/denominator immediately after each event.

    Reset whenever the segment changes: an accumulator that survives a recording
    gap would carry invented history across it (clause C2).
    """

    n = event_times.size
    dim = 1 if values is None else values.shape[1]
    num = np.zeros((n, dim), dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    acc_num = np.zeros(dim, dtype=np.float64)
    acc_den = 0.0
    prev_t = None
    prev_seg = -1
    for i in range(n):
        seg = int(event_segment[i])
        t = float(event_times[i])
        if seg != prev_seg or prev_t is None:
            acc_num = np.zeros(dim, dtype=np.float64)
            acc_den = 0.0
        else:
            decay = float(np.exp(-(t - prev_t) / tau))
            acc_num = acc_num * decay
            acc_den = acc_den * decay
        acc_num = acc_num + (1.0 if values is None else values[i])
        acc_den = acc_den + 1.0
        num[i] = acc_num
        den[i] = acc_den
        prev_t = t
        prev_seg = seg
    return num, den


def _decay_to_anchor(
    value: np.ndarray, event_times: np.ndarray, grid: AnchorGrid, tau: float
) -> np.ndarray:
    """Decay the post-event accumulator forward to the anchor by the real dt."""

    pos = grid.last_event_pos
    out = np.zeros((grid.n_anchors,) + value.shape[1:], dtype=np.float64)
    has = pos >= 0
    if not np.any(has):
        return out
    idx = pos[has]
    dt = grid.t_anchor[has] - event_times[idx]
    decay = np.exp(-dt / tau)
    shape = (decay.size,) + (1,) * (value.ndim - 1)
    out[has] = value[idx] * decay.reshape(shape)
    return out


def build_baseline_features(
    grid: AnchorGrid,
    segments: Sequence[CoverageSegment],
    event_times: np.ndarray,
    event_segment: np.ndarray,
    marks: EventMarks,
    *,
    seizure_onsets: np.ndarray,
    seizure_offsets: np.ndarray,
) -> BaselineFeatures:
    """Assemble ``B_multiscale`` at every anchor of the fixed grid."""

    event_times = np.asarray(event_times, dtype=np.float64)
    event_segment = np.asarray(event_segment, dtype=np.int64)
    columns: list[np.ndarray] = []
    names: list[str] = []

    def _add(block: np.ndarray, labels: Sequence[str]) -> None:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] != len(labels):
            raise ValueError(f"{len(labels)} labels for {block.shape[1]} columns")
        columns.append(block)
        names.extend(labels)

    # --- rate at four timescales -------------------------------------------------
    for tau in EWMA_TAU_SECONDS:
        num, _den = _ewma_after_each_event(event_times, event_segment, None, tau)
        rate = _decay_to_anchor(num, event_times, grid, tau)[:, 0] / tau
        _add(np.log1p(np.clip(rate, 0.0, None)), [f"rate_tau{int(tau)}"])

    # --- time since last event ---------------------------------------------------
    since = grid.seconds_since_last_event
    finite = np.isfinite(since)
    _add(np.where(finite, np.log1p(np.clip(since, 0.0, None)),
                  np.log1p(SEIZURE_TIME_CAP_SECONDS)), ["log_time_since_last_event"])
    _add(finite.astype(np.float64), ["has_previous_event"])
    _add(np.log1p(np.clip(grid.n_events_before, 0, None).astype(np.float64)),
         ["log_events_so_far_in_segment"])

    # --- size / STOP and the mark families, as EWMA-weighted means ---------------
    scalar_blocks: dict[str, np.ndarray] = {}
    for block_name, sl in marks.block_slices.items():
        scalar_blocks[block_name] = marks.continuous[:, sl]

    for tau in EWMA_TAU_SECONDS:
        for block_name, values in scalar_blocks.items():
            num, den = _ewma_after_each_event(event_times, event_segment, values, tau)
            num_a = _decay_to_anchor(num, event_times, grid, tau)
            den_a = _decay_to_anchor(den[:, None], event_times, grid, tau)[:, 0]
            mean = num_a / (den_a + MEAN_PRIOR_WEIGHT)[:, None]
            labels = [
                f"{block_name}[{i}]_tau{int(tau)}" for i in range(values.shape[1])
            ]
            _add(mean, labels)

    # --- per-contact participation field at two timescales ------------------------
    part = marks.participation.astype(np.float64)
    for tau in FIELD_TAU_SECONDS:
        num, den = _ewma_after_each_event(event_times, event_segment, part, tau)
        num_a = _decay_to_anchor(num, event_times, grid, tau)
        den_a = _decay_to_anchor(den[:, None], event_times, grid, tau)[:, 0]
        field = num_a / (den_a + MEAN_PRIOR_WEIGHT)[:, None]
        _add(field, [f"participation[{c}]_tau{int(tau)}" for c in range(part.shape[1])])

    # --- clock -------------------------------------------------------------------
    # Absolute time-of-day phase is unknown without a per-cohort timezone, but the
    # (sin, cos) pair spans every rotation of it, so a linear readout absorbs the
    # offset.  No timezone assumption enters the features.
    phase = 2 * np.pi * (grid.t_anchor % SECONDS_PER_DAY) / SECONDS_PER_DAY
    _add(np.sin(phase), ["clock_sin_day"])
    _add(np.cos(phase), ["clock_cos_day"])
    phase12 = 2 * np.pi * (grid.t_anchor % (SECONDS_PER_DAY / 2)) / (SECONDS_PER_DAY / 2)
    _add(np.sin(phase12), ["clock_sin_half_day"])
    _add(np.cos(phase12), ["clock_cos_half_day"])

    # --- session position and coverage -------------------------------------------
    seg_start = np.array([segments[i].start_epoch for i in grid.segment_index])
    seg_stop = np.array([segments[i].stop_epoch for i in grid.segment_index])
    into = grid.t_anchor - seg_start
    left = seg_stop - grid.t_anchor
    _add(np.log1p(np.clip(into, 0.0, None)), ["log_seconds_into_segment"])
    _add(np.log1p(np.clip(left, 0.0, None)), ["log_seconds_left_in_segment"])
    _add(into / np.maximum(seg_stop - seg_start, 1.0), ["fraction_through_segment"])
    _add(np.log1p(np.maximum(seg_stop - seg_start, 0.0)), ["log_segment_duration"])
    _add((grid.t_anchor - float(grid.t_anchor.min() if grid.n_anchors else 0.0))
         / SECONDS_PER_DAY, ["days_since_recording_start"])

    # --- seizure bookkeeping (baseline only; never an input to a producer) --------
    onsets = np.sort(np.asarray(seizure_onsets, dtype=np.float64))
    offsets = np.sort(np.asarray(seizure_offsets, dtype=np.float64))
    since_prev = np.full(grid.n_anchors, SEIZURE_TIME_CAP_SECONDS, dtype=np.float64)
    to_next = np.full(grid.n_anchors, SEIZURE_TIME_CAP_SECONDS, dtype=np.float64)
    if offsets.size:
        j = np.searchsorted(offsets, grid.t_anchor, side="right") - 1
        ok = j >= 0
        since_prev[ok] = np.clip(grid.t_anchor[ok] - offsets[j[ok]], 0.0,
                                 SEIZURE_TIME_CAP_SECONDS)
    if onsets.size:
        j = np.searchsorted(onsets, grid.t_anchor, side="left")
        ok = j < onsets.size
        to_next[ok] = np.clip(onsets[j[ok]] - grid.t_anchor[ok], 0.0,
                              SEIZURE_TIME_CAP_SECONDS)
    _add(np.log1p(since_prev), ["log_time_since_prev_seizure"])
    _add((since_prev < POSTICTAL_INDICATOR_SECONDS).astype(np.float64),
         ["recent_seizure_indicator"])
    _add(np.log1p(np.minimum(since_prev, to_next)), ["log_time_to_nearest_seizure"])

    x = np.concatenate(columns, axis=1) if columns else np.zeros((grid.n_anchors, 0))
    if not np.isfinite(x).all():
        bad = [names[i] for i in np.flatnonzero(~np.isfinite(x).all(axis=0))]
        raise ValueError(f"non-finite baseline features: {bad[:8]}")
    return BaselineFeatures(
        x=x,
        names=tuple(names),
        notes={
            "ewma_tau_seconds": list(EWMA_TAU_SECONDS),
            "field_tau_seconds": list(FIELD_TAU_SECONDS),
            "sleep_wake": "not_available",
            "asm_intervention": "not_available",
            "uses_seizure_times": True,
            "timezone_assumption": "none (sin/cos day and half-day basis)",
            "n_seizures": int(onsets.size),
        },
    )
