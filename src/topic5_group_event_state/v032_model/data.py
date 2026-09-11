"""One patient's complete v0.3.2 model-side view: events, tokens, anchors, H."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.topic5_group_event_state.v02.subject import load_subject_timeline
from src.topic5_group_event_state.v03.partition import PHASE_NAMES, nested_time_partition

from .features import TrainStandardizer, build_subject_features, feature_fingerprint
from .history_baseline import (
    HistoryBaseline,
    fit_provisional_history_baseline,
    load_agent2_history_baseline,
    load_endpoint_eligibility,
)
from .paths import (
    DATASET_ROOT,
    ENDPOINT_ELIGIBILITY,
    HISTORY_BASELINE_REGISTRY,
    MODEL_ROOT,
    SHARED_ROOT,
)

DEFAULT_HORIZONS = (300.0, 1800.0, 7200.0)
STATE_TRAIN_PHASE = PHASE_NAMES.index("state_train")


@dataclass
class SubjectBundle:
    subject: str
    event_times: np.ndarray
    event_segment: np.ndarray
    event_session: np.ndarray
    event_phase: np.ndarray
    stream_positions: np.ndarray
    x_raw: np.ndarray
    x_std: np.ndarray
    feature_names: tuple[str, ...]
    standardizer: TrainStandardizer
    t_anchor: np.ndarray
    anchor_segment: np.ndarray
    anchor_session: np.ndarray
    anchor_phase: np.ndarray
    last_event_pos: np.ndarray
    seconds_since_last_event: np.ndarray
    eligible: np.ndarray
    counts: np.ndarray
    horizons: tuple[float, ...]
    phase_lower: np.ndarray
    phase_upper: np.ndarray
    segment_bounds: np.ndarray
    history: HistoryBaseline
    eligibility: dict[str, Any] | None
    fingerprint: dict[str, Any]
    baseline_x: np.ndarray | None = None
    baseline_names: tuple[str, ...] | None = None
    timeline: Any = field(default=None, repr=False)
    partition: Any = field(default=None, repr=False)

    # ----------------------------------------------------------------- helpers
    @property
    def n_events(self) -> int:
        return int(self.event_times.size)

    @property
    def n_anchors(self) -> int:
        return int(self.t_anchor.size)

    def horizon_index(self, horizon: float) -> int:
        return list(self.horizons).index(float(horizon))

    def anchor_mask(self, phase: str, horizon: float) -> np.ndarray:
        """Anchors of one nested phase whose whole window stays inside that phase."""

        i = PHASE_NAMES.index(phase)
        h_i = self.horizon_index(horizon)
        return (
            (self.anchor_phase == i)
            & self.eligible[:, h_i]
            & (self.t_anchor + float(horizon) <= self.phase_upper[i] + 1e-6)
        )

    def train_event_mask(self) -> np.ndarray:
        return self.event_phase == STATE_TRAIN_PHASE

    def effective_independent_windows(self, phase: str, horizon: float) -> int:
        i = PHASE_NAMES.index(phase)
        lo, hi = float(self.phase_lower[i]), float(self.phase_upper[i])
        total = 0
        for start, stop in self.segment_bounds:
            a, b = max(float(start), lo), min(float(stop), hi)
            if b > a:
                total += int(math.floor((b - a) / float(horizon)))
        return total

    def log_mu_h(self, horizon: float) -> np.ndarray:
        return np.asarray(self.history.log_mu[int(horizon)], dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "n_events": self.n_events,
            "n_anchors": self.n_anchors,
            "n_features": int(self.x_std.shape[1]),
            "n_segments": int(self.segment_bounds.shape[0]),
            "events_by_phase": {n: int((self.event_phase == i).sum()) for i, n in enumerate(PHASE_NAMES)},
            "anchors_by_phase_and_horizon": {
                f"{int(h)}s": {n: int(self.anchor_mask(n, h).sum()) for n in PHASE_NAMES}
                for h in self.horizons
            },
            "effective_independent_windows": {
                f"{int(h)}s": {n: self.effective_independent_windows(n, h) for n in PHASE_NAMES}
                for h in self.horizons
            },
            "h_source": self.history.source,
            "fingerprint": self.fingerprint,
        }


def bundle_from_arrays(
    timeline,
    partition,
    *,
    x_raw: np.ndarray,
    feature_names: Sequence[str],
    history: HistoryBaseline,
    eligibility: dict[str, Any] | None,
    fingerprint: dict[str, Any],
    subject: str | None = None,
) -> SubjectBundle:
    """Assemble a bundle from a timeline-like object (real or toy) and raw tokens."""

    grid = timeline.grid
    horizons = tuple(float(h) for h in timeline.config.horizons_seconds)
    event_times = np.asarray(timeline.event_times, dtype=np.float64)
    event_segment = np.asarray(timeline.event_segment, dtype=np.int64)
    seg_session = np.asarray([s.session_id for s in timeline.segments], dtype=np.int64)
    event_phase = partition.labels_of(event_times).astype(np.int64)
    train_mask = event_phase == STATE_TRAIN_PHASE
    standardizer = TrainStandardizer.fit(x_raw, train_mask)
    counts = (np.asarray(grid.window_hi) - np.asarray(grid.window_lo)).astype(np.int64)
    lower = np.array([partition.bounds(n)[0] for n in PHASE_NAMES], dtype=np.float64)
    upper = np.array([partition.bounds(n)[1] for n in PHASE_NAMES], dtype=np.float64)
    baseline = getattr(timeline, "baseline", None)
    return SubjectBundle(
        subject=subject or str(getattr(timeline, "subject", "unknown")),
        event_times=event_times,
        event_segment=event_segment,
        event_session=seg_session[event_segment],
        event_phase=event_phase,
        stream_positions=np.asarray(timeline.stream_positions, dtype=np.int64),
        x_raw=np.asarray(x_raw, dtype=np.float32),
        x_std=standardizer.transform(x_raw),
        feature_names=tuple(feature_names),
        standardizer=standardizer,
        t_anchor=np.asarray(grid.t_anchor, dtype=np.float64),
        anchor_segment=np.asarray(grid.segment_index, dtype=np.int64),
        anchor_session=np.asarray(grid.session_id, dtype=np.int64),
        anchor_phase=partition.labels_of(np.asarray(grid.t_anchor)).astype(np.int64),
        last_event_pos=np.asarray(grid.last_event_pos, dtype=np.int64),
        seconds_since_last_event=np.asarray(grid.seconds_since_last_event, dtype=np.float64),
        eligible=np.asarray(grid.eligible, dtype=bool),
        counts=counts,
        horizons=horizons,
        phase_lower=lower,
        phase_upper=upper,
        segment_bounds=np.array([[s.start_epoch, s.stop_epoch] for s in timeline.segments], dtype=np.float64),
        history=history,
        eligibility=eligibility,
        fingerprint=dict(fingerprint),
        baseline_x=None if baseline is None else np.asarray(baseline.x, dtype=np.float64),
        baseline_names=None if baseline is None else tuple(baseline.names),
        timeline=timeline,
        partition=partition,
    )


def load_subject_bundle(
    subject: str,
    *,
    features_root: Path = MODEL_ROOT / "features",
    shared_root: Path = SHARED_ROOT,
    dataset_root: Path = DATASET_ROOT,
    horizons: Sequence[float] = DEFAULT_HORIZONS,
    allow_provisional_h: bool = True,
) -> SubjectBundle:
    """Real-data bundle.  ``H`` comes from Agent 2 when present, else provisional."""

    npz_path = build_subject_features(subject, dataset_root=dataset_root, out_root=features_root)
    with np.load(npz_path, allow_pickle=True) as data:
        x_raw = np.asarray(data["x_raw"], dtype=np.float32)
        names = tuple(str(v) for v in data["names"].tolist())
    timeline = load_subject_timeline(subject, dataset_root=dataset_root)
    partition = nested_time_partition(timeline.segments)
    if x_raw.shape[0] != timeline.event_times.size:
        raise ValueError(f"{subject}: cached features ({x_raw.shape[0]}) do not match timeline events "
                         f"({timeline.event_times.size}); rebuild features")
    shared_root = Path(shared_root)
    history, reason = load_agent2_history_baseline(
        shared_root / HISTORY_BASELINE_REGISTRY.name, subject, timeline.grid.t_anchor, horizons
    )
    if history is None:
        if not allow_provisional_h:
            raise FileNotFoundError(f"{subject}: Agent 2 history baseline unavailable ({reason})")
        history = fit_provisional_history_baseline(timeline, partition, horizons)
        history.meta["agent2_registry_reason"] = reason
    eligibility = load_endpoint_eligibility(shared_root / ENDPOINT_ELIGIBILITY.name, subject)
    fingerprint = feature_fingerprint(subject, dataset_root=dataset_root)
    return bundle_from_arrays(
        timeline, partition, x_raw=x_raw, feature_names=names, history=history,
        eligibility=eligibility, fingerprint=fingerprint, subject=subject,
    )
