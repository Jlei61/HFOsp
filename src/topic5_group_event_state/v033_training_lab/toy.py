"""Production toy bundle for smoke / harness jobs (same structure as the v0.3.2 test toy).

Two coverage segments separated by a gap, a 5-minute anchor grid, the
20/50/10/20 nested recorded-time partition and three horizons -- small enough
to train in seconds, structured enough to exercise every boundary rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.topic5_group_event_state.v02.timeline import CoverageSegment
from src.topic5_group_event_state.v03.partition import nested_time_partition
from src.topic5_group_event_state.v032_model.data import SubjectBundle, bundle_from_arrays
from src.topic5_group_event_state.v032_model.history_baseline import fit_provisional_history_baseline

HORIZONS = (300.0, 1800.0, 7200.0)


def toy_timeline(seed: int = 0, *, seg_len: float = 40_000.0, gap: float = 10_000.0,
                 rate_per_second: float = 0.05, n_baseline: int = 6):
    rng = np.random.default_rng(seed)
    starts = (0.0, seg_len + gap)
    segments = [CoverageSegment(i, i, s, s + seg_len) for i, s in enumerate(starts)]
    events = np.sort(np.concatenate([rng.uniform(s, s + seg_len, int(seg_len * rate_per_second)) for s in starts]))
    event_segment = np.searchsorted(np.asarray(starts), events, side="right") - 1
    t_anchor = np.concatenate([np.arange(s + 300.0, s + seg_len, 300.0) for s in starts])
    seg_of = np.searchsorted(np.asarray(starts), t_anchor, side="right") - 1
    seg_start = np.asarray(starts)[seg_of]
    seg_stop = seg_start + seg_len
    a, h = t_anchor.size, len(HORIZONS)
    eligible = np.zeros((a, h), bool)
    lo = np.zeros((a, h), np.int64)
    hi = np.zeros((a, h), np.int64)
    for i, horizon in enumerate(HORIZONS):
        eligible[:, i] = t_anchor + horizon <= seg_stop
        lo[:, i] = np.searchsorted(events, t_anchor, side="left")
        hi[:, i] = np.searchsorted(events, t_anchor + horizon, side="left")
    pos = np.searchsorted(events, t_anchor, side="left") - 1
    in_seg = (pos >= 0) & (events[np.clip(pos, 0, None)] >= seg_start)
    last = np.where(in_seg, pos, -1)
    x = rng.normal(size=(a, n_baseline))
    names = tuple(["rate_tau60", "rate_tau1800", "clock_sin_day", "log_time_since_prev_seizure"]
                  + [f"f{i}" for i in range(4, n_baseline)])
    grid = SimpleNamespace(
        t_anchor=t_anchor, segment_index=seg_of, session_id=seg_of, last_event_pos=last, eligible=eligible,
        window_lo=lo, window_hi=hi, horizons_seconds=HORIZONS,
        seconds_since_last_event=np.where(last >= 0, t_anchor - events[np.clip(last, 0, None)], np.inf), n_anchors=a,
    )
    timeline = SimpleNamespace(
        subject="toy", segments=segments, event_times=events, event_segment=event_segment,
        stream_positions=np.arange(events.size), grid=grid, baseline=SimpleNamespace(x=x, names=names),
        config=SimpleNamespace(horizons_seconds=HORIZONS),
    )
    return timeline, nested_time_partition(segments)


def toy_bundle(seed: int = 0, *, n_features: int = 6, **timeline_kwargs) -> SubjectBundle:
    timeline, partition = toy_timeline(seed, **timeline_kwargs)
    rng = np.random.default_rng(seed + 1)
    x_raw = rng.normal(size=(timeline.event_times.size, n_features)).astype(np.float32)
    history = fit_provisional_history_baseline(timeline, partition, HORIZONS)
    return bundle_from_arrays(timeline, partition, x_raw=x_raw, feature_names=tuple(f"x{i}" for i in range(n_features)),
                              history=history, eligibility=None, fingerprint={"toy": True, "seed": int(seed)})
