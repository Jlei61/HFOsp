"""Frozen per-event marks and the sparse future-block targets they aggregate into.

Two vocabularies, kept apart on purpose because H3's two estimands live on them:

``count features``   what an event is *as an occurrence*: that it happened, how
                     large it was, how long since the last one.  These are the
                     only event channels ``M1`` may feed back into the state.
``mark features``    what an event *was*: which contacts, how far the recruitment
                     spread, which bands carried it, what the waveform looked
                     like.  These are what ``M2`` adds on top, and what the
                     content perturbation replaces while leaving count and time
                     untouched.

Block targets are stored sparsely -- per anchor, per horizon, a handful of
summaries -- rather than as an ``events x horizon x contact`` tensor, which for
the longest patient would be tens of gigabytes of mostly zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Which frozen mark columns belong to which reported endpoint.  The decomposition
# is a hierarchy, not a duplication: ``conditional_mark`` is the joint score and
# these are its named parts.
MARK_GROUPS = ("participation", "extent", "multiband", "waveform_crossband")


@dataclass
class EventFeatures:
    """One patient's frozen event vocabulary, aligned to the interictal stream."""

    t_abs: np.ndarray            # (n,) float64 absolute seconds
    count_features: np.ndarray   # (n, 4) float32  -- M1's channel
    mark_features: np.ndarray    # (n, F) float32  -- M2's channel
    mark_group_slices: dict[str, tuple[int, int]]
    count_feature_names: list[str]
    mark_feature_names: list[str]
    participation: np.ndarray    # (n, C) bool, kept for the participation target
    size: np.ndarray             # (n,) float32
    band_available: np.ndarray   # (B,) bool

    @property
    def n_events(self) -> int:
        return int(self.t_abs.size)


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None))


def build_event_features(
    seq_arrays: dict[str, np.ndarray],
    t_abs: np.ndarray,
    dt_prev: np.ndarray,
    band_available: Sequence[bool],
) -> EventFeatures:
    """Compute the frozen count/mark vocabularies for a patient's whole stream.

    Every pooled quantity is masked by ``participation``: a non-participating
    contact carries a finite phantom value in the legacy files, and pooling it
    would mix producer noise into the mark that ``M2`` is supposed to test.
    """

    part = np.asarray(seq_arrays["participation"], dtype=bool)          # (n, C)
    delay = np.asarray(seq_arrays["relative_delay"], dtype=np.float32)  # (n, C)
    tied = np.asarray(seq_arrays["tied_group_id"], dtype=np.int16)      # (n, C)
    band_feat = np.asarray(seq_arrays["band_features"], dtype=np.float32)  # (n, C, B, 5)
    xband = np.asarray(seq_arrays["cross_band_lag"], dtype=np.float32)  # (n, C, P)
    wave_rms = np.asarray(seq_arrays["waveform_rms"], dtype=np.float32)  # (n, V)

    n, n_contacts = part.shape
    n_bands = band_feat.shape[2]
    n_pairs = xband.shape[2]
    n_views = wave_rms.shape[1]
    avail = np.asarray(band_available, dtype=bool)

    size = part.sum(1).astype(np.float32)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        span = np.where(part & np.isfinite(delay), delay, np.nan)
        span = np.nan_to_num(np.nanmax(span, axis=1) - np.nanmin(span, axis=1), nan=0.0)
    n_tied = np.array(
        [len(np.unique(tied[i][part[i] & (tied[i] >= 0)])) for i in range(n)], dtype=np.float32
    )

    dt = np.asarray(dt_prev, dtype=np.float64)
    dt = np.where(np.isfinite(dt), dt, np.nan)
    # A session's first event has no interval; a zero would claim it arrived
    # instantly after nothing.  The channel carries its own missing flag instead.
    dt_missing = ~np.isfinite(dt)
    dt_filled = np.where(dt_missing, 0.0, dt)

    count = np.stack(
        [
            np.ones(n, dtype=np.float32),                      # occurrence impulse
            _safe_log1p(size).astype(np.float32),              # burden magnitude
            (size / max(n_contacts, 1)).astype(np.float32),    # burden fraction
            _safe_log1p(dt_filled).astype(np.float32),         # local rate
        ],
        axis=1,
    )
    count_names = [
        "occurrence",
        "log1p_size",
        "size_fraction",
        "log1p_dt_prev",
    ]

    mask3 = part[:, :, None]
    # An all-masked slice is the legitimate answer for a band that this patient's
    # sampling rate cannot represent; it becomes an explicit zero under the
    # availability mask below rather than an imputed value.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        energy = np.where(mask3, band_feat[:, :, :, 2], np.nan)     # log integrated energy
        peak_t = np.where(mask3, band_feat[:, :, :, 0], np.nan)     # peak time
        energy_mean = np.nan_to_num(np.nanmean(energy, axis=1), nan=0.0)
        energy_spread = np.nan_to_num(np.nanstd(energy, axis=1), nan=0.0)
        peak_mean = np.nan_to_num(np.nanmean(peak_t, axis=1), nan=0.0)
        lag_pooled = np.nan_to_num(
            np.nanmean(np.where(part[:, :, None], xband, np.nan), axis=1), nan=0.0
        )
    # A band the sampling rate cannot represent is missing, not silent.
    energy_mean *= avail
    energy_spread *= avail
    peak_mean *= avail

    blocks: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, tuple[int, int]] = {}

    start = 0
    blocks.append(part.astype(np.float32))
    names += [f"participation_{i}" for i in range(n_contacts)]
    groups["participation"] = (start, start + n_contacts)
    start += n_contacts

    extent = np.stack(
        [
            _safe_log1p(1000.0 * span).astype(np.float32),
            (n_tied / np.maximum(size, 1.0)).astype(np.float32),
            np.nan_to_num(n_tied).astype(np.float32),
        ],
        axis=1,
    )
    blocks.append(extent)
    names += ["log1p_span_ms", "tied_groups_per_contact", "n_tied_groups"]
    groups["extent"] = (start, start + extent.shape[1])
    start += extent.shape[1]

    multiband = np.concatenate([energy_mean, energy_spread, peak_mean], axis=1).astype(np.float32)
    blocks.append(multiband)
    names += (
        [f"band{b}_log_energy_mean" for b in range(n_bands)]
        + [f"band{b}_log_energy_spread" for b in range(n_bands)]
        + [f"band{b}_peak_time_mean" for b in range(n_bands)]
    )
    groups["multiband"] = (start, start + multiband.shape[1])
    start += multiband.shape[1]

    wave_cross = np.concatenate([lag_pooled, wave_rms], axis=1).astype(np.float32)
    blocks.append(wave_cross)
    names += [f"cross_band_lag_{p}" for p in range(n_pairs)] + [
        f"waveform_log_rms_view{v}" for v in range(n_views)
    ]
    groups["waveform_crossband"] = (start, start + wave_cross.shape[1])

    mark = np.concatenate(blocks, axis=1).astype(np.float32)
    mark[~np.isfinite(mark)] = 0.0

    return EventFeatures(
        t_abs=np.asarray(t_abs, dtype=np.float64),
        count_features=count.astype(np.float32),
        mark_features=mark,
        mark_group_slices=groups,
        count_feature_names=count_names,
        mark_feature_names=names,
        participation=part,
        size=size,
        band_available=avail,
    )


# --------------------------------------------------------------------------- targets


@dataclass
class BlockTargets:
    """Sparse future-block truth for a set of anchors at one horizon."""

    anchor: np.ndarray            # (A,) float64
    horizon_seconds: float
    count: np.ndarray             # (A,) int64        -- events in [anchor, anchor+H)
    has_events: np.ndarray        # (A,) bool
    mark_mean: np.ndarray         # (A, F) float32    -- mean mark over block events
    contact_active: np.ndarray    # (A, C) bool
    contact_rate: np.ndarray      # (A, C) float32    -- fraction of block events
    n_anchors: int

    def as_meta(self) -> dict[str, Any]:
        return {
            "horizon_seconds": float(self.horizon_seconds),
            "n_anchors": int(self.n_anchors),
            "n_anchors_with_events": int(self.has_events.sum()),
            "median_count": float(np.median(self.count)) if self.count.size else float("nan"),
        }


def build_block_targets(
    features: EventFeatures,
    anchors: np.ndarray,
    horizon_seconds: float,
) -> BlockTargets:
    """Aggregate the frozen marks over ``[anchor, anchor + horizon)``.

    Prefix sums, not a loop over anchors: the longest patient has 235k events and
    3k anchors, and the naive form is quadratic.  The half-open convention matters
    -- an event exactly at the anchor belongs to the block being predicted, not to
    the history that predicts it.
    """

    t = features.t_abs
    lo = np.searchsorted(t, anchors, side="left")
    hi = np.searchsorted(t, anchors + float(horizon_seconds), side="left")
    count = (hi - lo).astype(np.int64)
    has = count > 0

    mark_cs = np.concatenate(
        [np.zeros((1, features.mark_features.shape[1]), dtype=np.float64),
         np.cumsum(features.mark_features.astype(np.float64), axis=0)],
        axis=0,
    )
    part_cs = np.concatenate(
        [np.zeros((1, features.participation.shape[1]), dtype=np.float64),
         np.cumsum(features.participation.astype(np.float64), axis=0)],
        axis=0,
    )
    denom = np.maximum(count, 1)[:, None]
    mark_mean = ((mark_cs[hi] - mark_cs[lo]) / denom).astype(np.float32)
    contact_sum = part_cs[hi] - part_cs[lo]
    contact_rate = (contact_sum / denom).astype(np.float32)
    contact_active = contact_sum > 0

    return BlockTargets(
        anchor=np.asarray(anchors, dtype=np.float64),
        horizon_seconds=float(horizon_seconds),
        count=count,
        has_events=has,
        mark_mean=mark_mean,
        contact_active=contact_active,
        contact_rate=contact_rate,
        n_anchors=int(np.asarray(anchors).size),
    )


def train_standardiser(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust location/scale from TRAIN rows only.

    Estimating these on the whole stream leaks the held-out distribution into the
    model's own normalisation constants -- quiet, but a real leak.
    """

    rows = np.asarray(x, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if rows.size == 0:
        return np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32)
    loc = np.median(rows, axis=0)
    scale = np.median(np.abs(rows - loc), axis=0) * 1.4826
    scale = np.where(scale > 1e-6, scale, np.std(rows, axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return loc.astype(np.float32), scale.astype(np.float32)


def write_features_meta(path: Path, features: EventFeatures) -> None:
    payload = {
        "n_events": features.n_events,
        "count_feature_names": features.count_feature_names,
        "mark_feature_names": features.mark_feature_names,
        "mark_group_slices": {k: list(v) for k, v in features.mark_group_slices.items()},
        "band_available": features.band_available.tolist(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
