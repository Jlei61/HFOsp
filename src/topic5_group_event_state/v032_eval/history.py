"""Explicit multiscale history ``H`` evaluated at arbitrary causal query times.

``H_rate`` and ``H_strong`` are nested.  Every column is a function of events
strictly before the query time and inside the same coverage segment, decayed by
real elapsed seconds.  Nothing here reads the segment end: segments end at
seizure onset, so "seconds left in segment" would leak future seizure timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

SECONDS_PER_DAY = 86400.0
NO_EVENT_CAP_SECONDS = 7 * SECONDS_PER_DAY
MEAN_PRIOR_WEIGHT = 1e-3


@dataclass(frozen=True)
class HistoryInputs:
    """Plain arrays so the builder can be tested without the dataset on disk."""

    event_times: np.ndarray          # (N,) sorted within segment
    event_segment: np.ndarray        # (N,)
    event_session: np.ndarray        # (N,)
    segment_start: Mapping[int, float]
    session_start: Mapping[int, float]
    recording_start: float
    participation: np.ndarray        # (N, C) bool
    vocab_mask: np.ndarray           # (C,) bool
    shaft_of_contact: Sequence[str]  # (C,)
    group_count: np.ndarray          # (N,)
    relative_delay: np.ndarray       # (N, C)
    mark_continuous: np.ndarray      # (N, D) TRAIN-standardised marks (0 where invalid)
    mark_names: Sequence[str]


def _last_event_before(
    inputs: HistoryInputs, query_times: np.ndarray, query_segment: np.ndarray
) -> np.ndarray:
    """Index of the last event strictly before each query time in its segment."""

    t = np.asarray(inputs.event_times, dtype=np.float64)
    pos = np.searchsorted(t, np.asarray(query_times, dtype=np.float64), side="left") - 1
    out = np.full(query_times.shape, -1, dtype=np.int64)
    ok = pos >= 0
    same = np.zeros_like(ok)
    same[ok] = inputs.event_segment[pos[ok]] == np.asarray(query_segment)[ok]
    out[ok & same] = pos[ok & same]
    return out


def _ewma_after_each_event(inputs: HistoryInputs, values: np.ndarray | None, tau: float):
    """Causal EWMA numerator/denominator right after each event, reset per segment."""

    n = inputs.event_times.size
    dim = 1 if values is None else values.shape[1]
    num = np.zeros((n, dim), dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    acc_num = np.zeros(dim)
    acc_den = 0.0
    prev_t = None
    prev_seg = -1
    for i in range(n):
        seg = int(inputs.event_segment[i])
        t = float(inputs.event_times[i])
        if seg != prev_seg or prev_t is None:
            acc_num = np.zeros(dim)
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


def _decay_to_query(value: np.ndarray, event_times: np.ndarray, last: np.ndarray,
                    query_times: np.ndarray, tau: float) -> np.ndarray:
    out = np.zeros((query_times.size,) + value.shape[1:], dtype=np.float64)
    has = last >= 0
    if not has.any():
        return out
    dt = query_times[has] - event_times[last[has]]
    decay = np.exp(-dt / tau).reshape((-1,) + (1,) * (value.ndim - 1))
    out[has] = value[last[has]] * decay
    return out


class HistoryFeatureBuilder:
    """Builds ``H_rate`` / ``H_strong`` columns at anchors or event pre-times."""

    def __init__(self, inputs: HistoryInputs, *, lookback_seconds: Sequence[float],
                 ewma_tau_seconds: Sequence[float], field_tau_seconds: Sequence[float]):
        self.inputs = inputs
        self.lookback = tuple(float(v) for v in lookback_seconds)
        self.taus = tuple(float(v) for v in ewma_tau_seconds)
        self.field_taus = tuple(float(v) for v in field_tau_seconds)
        part = np.asarray(inputs.participation, dtype=bool)
        vocab = np.asarray(inputs.vocab_mask, dtype=bool)
        self.vocab_index = np.flatnonzero(vocab)
        part_vocab = part[:, vocab]
        size = part_vocab.sum(axis=1).astype(np.float64)
        shafts = np.asarray(list(inputs.shaft_of_contact))[vocab]
        unique_shafts = sorted(set(shafts.tolist()))
        shaft_matrix = np.stack([shafts == s for s in unique_shafts], axis=1) if unique_shafts else np.zeros((vocab.sum(), 0), bool)
        n_shafts = (part_vocab.astype(np.float64) @ shaft_matrix.astype(np.float64) > 0).sum(axis=1).astype(np.float64)
        delay = np.asarray(inputs.relative_delay, dtype=np.float64)[:, vocab]
        masked = np.where(part_vocab, delay, -np.inf)
        span = masked.max(axis=1) if masked.shape[1] else np.zeros(masked.shape[0])
        span = np.where(np.isfinite(span), span, 0.0)
        self.extent = np.column_stack([
            size,
            np.asarray(inputs.group_count, dtype=np.float64),
            n_shafts,
            size / max(int(vocab.sum()), 1),
            span,
        ])
        self.extent_names = ("size", "n_groups", "n_shafts", "vocab_coverage", "delay_span")
        self.part_vocab = part_vocab.astype(np.float64)
        self._cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    # -- accumulators ------------------------------------------------------------
    def _ewma(self, key: str, values: np.ndarray | None, tau: float):
        cache_key = (key, tau)
        if cache_key not in self._cache:
            self._cache[cache_key] = _ewma_after_each_event(self.inputs, values, tau)
        return self._cache[cache_key]

    def features(self, query_times: np.ndarray, query_segment: np.ndarray,
                 *, variant: str) -> tuple[np.ndarray, list[str]]:
        if variant not in ("H_rate", "H_strong"):
            raise ValueError(f"unknown history variant {variant!r}")
        q = np.asarray(query_times, dtype=np.float64)
        seg = np.asarray(query_segment, dtype=np.int64)
        inputs = self.inputs
        t = np.asarray(inputs.event_times, dtype=np.float64)
        last = _last_event_before(inputs, q, seg)
        columns: list[np.ndarray] = []
        names: list[str] = []

        def add(block: np.ndarray, labels: Sequence[str]) -> None:
            block = np.asarray(block, dtype=np.float64)
            if block.ndim == 1:
                block = block[:, None]
            if block.shape[1] != len(labels):
                raise ValueError("label/column mismatch")
            columns.append(block)
            names.extend(labels)

        seg_start = np.asarray([inputs.segment_start[int(s)] for s in seg], dtype=np.float64)
        sess_of_seg = {}
        for i in range(inputs.event_times.size):
            sess_of_seg.setdefault(int(inputs.event_segment[i]), int(inputs.event_session[i]))
        # --- last IEI -----------------------------------------------------------
        has = last >= 0
        since = np.where(has, q - t[np.clip(last, 0, max(t.size - 1, 0))], NO_EVENT_CAP_SECONDS)
        add(np.log1p(np.clip(since, 0.0, None)), ["log_seconds_since_last_event"])
        add(has.astype(float), ["has_previous_event"])
        prev = last - 1
        two = has & (prev >= 0)
        two[two] &= inputs.event_segment[prev[two]] == seg[two]
        last_iei = np.full(q.size, NO_EVENT_CAP_SECONDS)
        last_iei[two] = t[last[two]] - t[prev[two]]
        add(np.log1p(np.clip(last_iei, 0.0, None)), ["log_last_iei"])
        add(two.astype(float), ["has_two_events"])
        # --- lookback counts / rates / coverage ---------------------------------
        first_in_seg = np.searchsorted(t, seg_start, side="left")
        n_before = np.where(has, last + 1 - first_in_seg, 0).astype(np.float64)
        for L in self.lookback:
            lo = np.searchsorted(t, q - L, side="left")
            hi = np.searchsorted(t, q, side="left")
            lo = np.maximum(lo, first_in_seg)
            count = np.clip(hi - lo, 0, None).astype(np.float64)
            covered = np.clip(np.minimum(L, q - seg_start), 0.0, L)
            rate = count / np.maximum(covered, 1.0)
            add(np.log1p(count), [f"log_count_{int(L)}s"])
            add(np.log1p(rate * 3600.0), [f"log_rate_per_hour_{int(L)}s"])
            add(covered / L, [f"covered_fraction_{int(L)}s"])
        # --- clock --------------------------------------------------------------
        phase = 2 * np.pi * (q % SECONDS_PER_DAY) / SECONDS_PER_DAY
        add(np.sin(phase), ["clock_sin_day"])
        add(np.cos(phase), ["clock_cos_day"])
        phase12 = 2 * np.pi * (q % (SECONDS_PER_DAY / 2)) / (SECONDS_PER_DAY / 2)
        add(np.sin(phase12), ["clock_sin_half_day"])
        add(np.cos(phase12), ["clock_cos_half_day"])
        # --- session position (no segment end) -----------------------------------
        into_seg = q - seg_start
        sess_start = np.asarray([inputs.session_start[sess_of_seg.get(int(s), int(s))] if sess_of_seg.get(int(s)) in inputs.session_start else inputs.segment_start[int(s)] for s in seg], dtype=np.float64)
        add(np.log1p(np.clip(into_seg, 0.0, None)), ["log_seconds_into_segment"])
        add(np.log1p(np.clip(q - sess_start, 0.0, None)), ["log_seconds_into_session"])
        # This is a real, interpretable drift covariate.  The GLM's TRAIN-range
        # winsorisation prevents unbounded extrapolation without deleting the
        # scientific control from H.
        add((q - float(inputs.recording_start)) / SECONDS_PER_DAY, ["days_since_recording_start"])
        add(np.log1p(n_before), ["log_events_so_far_in_segment"])
        if variant == "H_rate":
            x = np.concatenate(columns, axis=1)
            return x, names
        # --- H_strong: extent / dispersion / multiband / repertoire EWMAs -------------
        for tau in self.taus:
            num, den = self._ewma("extent", self.extent, tau)
            num_q = _decay_to_query(num, t, last, q, tau)
            den_q = _decay_to_query(den[:, None], t, last, q, tau)[:, 0]
            add(num_q / (den_q + MEAN_PRIOR_WEIGHT)[:, None],
                [f"{n}_ewma{int(tau)}" for n in self.extent_names])
            num, den = self._ewma("marks", self.inputs.mark_continuous, tau)
            num_q = _decay_to_query(num, t, last, q, tau)
            add(num_q / (den_q + MEAN_PRIOR_WEIGHT)[:, None],
                [f"mark:{n}_ewma{int(tau)}" for n in self.inputs.mark_names])
        for tau in self.field_taus:
            num, den = self._ewma("participation", self.part_vocab, tau)
            num_q = _decay_to_query(num, t, last, q, tau)
            den_q = _decay_to_query(den[:, None], t, last, q, tau)[:, 0]
            add(num_q / (den_q + MEAN_PRIOR_WEIGHT)[:, None],
                [f"participation[{int(c)}]_ewma{int(tau)}" for c in self.vocab_index])
        x = np.concatenate(columns, axis=1)
        if not np.isfinite(x).all():
            bad = [names[i] for i in np.flatnonzero(~np.isfinite(x).all(axis=0))]
            raise ValueError(f"non-finite history features: {bad[:8]}")
        return x, names


@dataclass(frozen=True)
class Standardiser:
    mean: np.ndarray
    scale: np.ndarray
    n_rows: int
    phase: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    @classmethod
    def fit(cls, x: np.ndarray, phase: str) -> "Standardiser":
        x = np.asarray(x, dtype=np.float64)
        scale = x.std(axis=0)
        return cls(mean=x.mean(axis=0), scale=np.where(scale > 1e-9, scale, 1.0),
                   n_rows=int(x.shape[0]), phase=phase)

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "n_rows": self.n_rows, "phase": self.phase}


def history_inputs_from_timeline(timeline) -> HistoryInputs:
    seg_start = timeline.segment_start_map()
    return HistoryInputs(
        event_times=timeline.event_times,
        event_segment=timeline.event_segment,
        event_session=timeline.event_session,
        segment_start=seg_start,
        session_start=timeline.session_start_map(),
        recording_start=float(min(seg_start.values())),
        participation=timeline.participation,
        vocab_mask=timeline.vocab_mask,
        shaft_of_contact=timeline.contact_shafts,
        group_count=timeline.group_count,
        relative_delay=timeline.relative_delay,
        mark_continuous=timeline.marks.continuous,
        mark_names=timeline.marks.continuous_names,
    )
