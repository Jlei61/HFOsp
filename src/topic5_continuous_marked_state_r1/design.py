"""Precomputed event and quadrature designs for exact R1 baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import CoverageTable
from .data import R1EventStream
from .history import DeterministicHistory, HistoryScaler, session_start_map


@dataclass(frozen=True)
class SplitDesign:
    split: str
    event_index: np.ndarray
    event_history: np.ndarray
    quadrature_history: np.ndarray
    quadrature_weight_seconds: np.ndarray
    recorded_seconds: float

    def validate(self) -> None:
        if self.event_history.shape[0] != len(self.event_index):
            raise ValueError("event design rows disagree")
        if self.quadrature_history.shape[0] != len(self.quadrature_weight_seconds):
            raise ValueError("quadrature design rows disagree")
        if np.any(self.quadrature_weight_seconds <= 0):
            raise ValueError("non-positive quadrature weight")
        if not np.isfinite(self.event_history).all():
            raise ValueError("non-finite event history")
        if not np.isfinite(self.quadrature_history).all():
            raise ValueError("non-finite quadrature history")
        if not np.isclose(self.quadrature_weight_seconds.sum(), self.recorded_seconds,
                          rtol=1e-10, atol=1e-6):
            raise ValueError("quadrature weights do not sum to recorded duration")


@dataclass(frozen=True)
class SubjectDesign:
    stream: R1EventStream
    coverage: CoverageTable
    scaler: HistoryScaler
    train: SplitDesign
    validation: SplitDesign


def _quadrature_grid(stream: R1EventStream, coverage: CoverageTable,
                     split: str, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split coverage at every event so causal history is smooth within a cell."""
    node, weight = np.polynomial.legendre.leggauss(int(order))
    segment_start, segment_stop, segment_session = coverage.split_segments_with_session(split)
    times: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    for left, right, label in zip(segment_start, segment_stop, segment_session):
        event = stream.event_time[
            (stream.session == label)
            & (stream.event_time > left)
            & (stream.event_time < right)
        ]
        boundary = np.concatenate([[left], event, [right]])
        width = np.diff(boundary)
        keep = width > 0
        a = boundary[:-1][keep]
        b = boundary[1:][keep]
        midpoint = 0.5 * (a + b)
        half = 0.5 * (b - a)
        times.append((midpoint[:, None] + half[:, None] * node[None, :]).reshape(-1))
        weights.append((half[:, None] * weight[None, :]).reshape(-1))
        sessions.append(np.full(len(a) * int(order), int(label), dtype=np.int64))
    if not times:
        raise ValueError(f"{stream.subject}: empty {split} quadrature grid")
    return np.concatenate(times), np.concatenate(weights), np.concatenate(sessions)


def _build_unscaled(stream: R1EventStream, coverage: CoverageTable,
                    history: DeterministicHistory, split: str,
                    quadrature_order: int) -> SplitDesign:
    event_index = np.flatnonzero(stream.mask(split))
    event_history = history.evaluate(
        stream.event_time[event_index], stream.session[event_index]
    )
    q_time, q_weight, q_session = _quadrature_grid(
        stream, coverage, split, quadrature_order
    )
    quadrature_history = history.evaluate(q_time, q_session)
    segment_start, segment_stop = coverage.split_segments(split)
    result = SplitDesign(
        split=split,
        event_index=event_index.astype(np.int64),
        event_history=event_history,
        quadrature_history=quadrature_history,
        quadrature_weight_seconds=q_weight.astype(np.float64),
        recorded_seconds=float(np.sum(segment_stop - segment_start)),
    )
    result.validate()
    return result


def build_subject_design(stream: R1EventStream, coverage: CoverageTable,
                         *, quadrature_order: int = 4) -> SubjectDesign:
    if stream.subject != coverage.subject:
        raise ValueError("stream/coverage subject mismatch")
    starts = session_start_map(stream, coverage.session, coverage.start)
    history = DeterministicHistory(stream, starts)
    train_raw = _build_unscaled(
        stream, coverage, history, "train", quadrature_order
    )
    validation_raw = _build_unscaled(
        stream, coverage, history, "validation", quadrature_order
    )
    # The scale is a property of recorded TRAIN time, not only event times.
    scaler = HistoryScaler.fit(np.concatenate([
        train_raw.event_history,
        train_raw.quadrature_history,
    ], axis=0))

    def scaled(value: SplitDesign) -> SplitDesign:
        result = SplitDesign(
            split=value.split,
            event_index=value.event_index,
            event_history=scaler.transform(value.event_history),
            quadrature_history=scaler.transform(value.quadrature_history),
            quadrature_weight_seconds=value.quadrature_weight_seconds,
            recorded_seconds=value.recorded_seconds,
        )
        result.validate()
        return result

    return SubjectDesign(
        stream=stream, coverage=coverage, scaler=scaler,
        train=scaled(train_raw), validation=scaled(validation_raw),
    )
