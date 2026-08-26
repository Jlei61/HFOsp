"""Deterministic causal event-history features for R1."""
from __future__ import annotations

from dataclasses import dataclass
import math
from zoneinfo import ZoneInfo

import numpy as np

from .data import R1EventStream


BASE_HISTORY_NAMES = (
    "has_previous_event",
    "log_time_since_previous_event",
    "count_trace_30s",
    "count_trace_2m",
    "count_trace_10m",
    "last_load",
    "recent_mean_load_5",
    "last_group_fraction",
    "tod_sin",
    "tod_cos",
    "log_session_elapsed_minutes",
)


def history_names(n_contacts: int) -> tuple[str, ...]:
    return BASE_HISTORY_NAMES + tuple(
        f"previous_participation_c{i}" for i in range(n_contacts)
    ) + tuple(
        f"previous_group_rank_c{i}" for i in range(n_contacts)
    ) + tuple(
        f"participation_trace_2m_c{i}" for i in range(n_contacts)
    )


@dataclass(frozen=True)
class HistoryScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, value: np.ndarray) -> "HistoryScaler":
        mean = np.mean(value, axis=0, dtype=np.float64)
        scale = np.std(value, axis=0, dtype=np.float64)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean) / self.scale).astype(np.float32)


class DeterministicHistory:
    """Evaluate strictly pre-query history at events or quadrature nodes."""

    def __init__(self, stream: R1EventStream, session_start: dict[int, float]):
        self.stream = stream
        self.session_start = {int(k): float(v) for k, v in session_start.items()}
        event_time = stream.event_time
        session = stream.session
        n, contacts = stream.n_events, stream.n_contacts
        self.count_after = np.zeros((n, 3), dtype=np.float64)
        self.participation_after = np.zeros((n, contacts), dtype=np.float64)
        self.recent_mean_load = np.zeros(n, dtype=np.float64)
        recent: list[float] = []
        count = np.zeros(3, dtype=np.float64)
        part_trace = np.zeros(contacts, dtype=np.float64)
        tau = np.asarray([30.0, 120.0, 600.0])
        for index in range(n):
            opening = index == 0 or session[index] != session[index - 1]
            if opening:
                count.fill(0.0)
                part_trace.fill(0.0)
                recent = []
            else:
                dt = max(float(event_time[index] - event_time[index - 1]), 0.0)
                count *= np.exp(-dt / tau)
                part_trace *= math.exp(-dt / 120.0)
            count += 1.0
            part_trace += stream.participation[index].astype(np.float64)
            recent.append(float(stream.load[index]))
            recent = recent[-5:]
            self.count_after[index] = count
            self.participation_after[index] = part_trace
            self.recent_mean_load[index] = float(np.mean(recent))

    def evaluate(self, query_time: np.ndarray,
                 query_session: np.ndarray) -> np.ndarray:
        query_time = np.asarray(query_time, dtype=np.float64)
        query_session = np.asarray(query_session, dtype=np.int64)
        if query_time.shape != query_session.shape:
            raise ValueError("query times and sessions disagree")
        flat_time = query_time.reshape(-1)
        flat_session = query_session.reshape(-1)
        index = np.searchsorted(self.stream.event_time, flat_time, side="left") - 1
        has = index >= 0
        safe = np.clip(index, 0, max(self.stream.n_events - 1, 0))
        has &= self.stream.session[safe] == flat_session
        previous_time = np.where(has, self.stream.event_time[safe], flat_time)
        age = np.maximum(flat_time - previous_time, 0.0)

        count = np.zeros((len(flat_time), 3), dtype=np.float64)
        part_trace = np.zeros((len(flat_time), self.stream.n_contacts), dtype=np.float64)
        count[has] = self.count_after[safe[has]] * np.exp(
            -age[has, None] / np.asarray([30.0, 120.0, 600.0])[None, :]
        )
        part_trace[has] = self.participation_after[safe[has]] * np.exp(
            -age[has, None] / 120.0
        )
        previous_participation = np.zeros_like(part_trace)
        previous_rank = np.zeros_like(part_trace)
        previous_participation[has] = self.stream.participation[safe[has]]
        selected_gid = self.stream.group_ids[safe].astype(np.float64)
        denom = np.maximum(self.stream.group_count[safe] - 1, 1).astype(np.float64)
        rank_value = np.where(
            self.stream.participation[safe], selected_gid / denom[:, None], 0.0
        )
        previous_rank[has] = rank_value[has]
        last_load = np.zeros(len(flat_time), dtype=np.float64)
        recent_load = np.zeros(len(flat_time), dtype=np.float64)
        last_group_fraction = np.zeros(len(flat_time), dtype=np.float64)
        last_load[has] = self.stream.load[safe[has]]
        recent_load[has] = self.recent_mean_load[safe[has]]
        last_group_fraction[has] = (
            self.stream.group_count[safe[has]] / float(self.stream.n_contacts)
        )

        zone = ZoneInfo("Europe/Berlin" if self.stream.dataset == "epilepsiae" else "Asia/Shanghai")
        # Converting every timestamp through datetime is expensive at quadrature
        # scale.  Dataset time zones in the current records are fixed offsets
        # over each patient span; compute the offset once at the median time.
        import datetime as _datetime
        mid = float(np.median(flat_time))
        offset = _datetime.datetime.fromtimestamp(mid, tz=zone).utcoffset()
        offset_seconds = float(offset.total_seconds()) if offset is not None else 0.0
        day_seconds = np.mod(flat_time + offset_seconds, 86400.0)
        phase = 2.0 * np.pi * day_seconds / 86400.0
        start = np.asarray([
            self.session_start.get(int(value), float(time))
            for value, time in zip(flat_session, flat_time)
        ])
        session_elapsed = np.maximum(flat_time - start, 0.0)
        base = np.column_stack([
            has.astype(np.float64),
            np.log1p(age),
            count,
            last_load,
            recent_load,
            last_group_fraction,
            np.sin(phase),
            np.cos(phase),
            np.log1p(session_elapsed / 60.0),
        ])
        value = np.concatenate(
            [base, previous_participation, previous_rank, part_trace], axis=1
        ).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("non-finite deterministic history")
        return value.reshape(*query_time.shape, value.shape[-1])


def session_start_map(stream: R1EventStream,
                      coverage_session: np.ndarray,
                      coverage_start: np.ndarray) -> dict[int, float]:
    result: dict[int, float] = {}
    for label, start in zip(coverage_session, coverage_start):
        result[int(label)] = min(result.get(int(label), float("inf")), float(start))
    missing = set(np.unique(stream.session).tolist()) - set(result)
    if missing:
        raise ValueError(f"coverage lacks event sessions {sorted(missing)}")
    return result
