"""Three non-interchangeable level controls for v0.3.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LevelControl:
    values: np.ndarray
    provenance: dict[str, Any]


def _as_2d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("values must be a finite vector or matrix")
    return x


def fit_train_mean_adapter(values: np.ndarray, train_mask: np.ndarray, *, n_output: int | None = None) -> LevelControl:
    """Static deployable calibration estimated strictly on TRAIN."""

    x = _as_2d(values)
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != (x.shape[0],) or not mask.any():
        raise ValueError("train_mask must select at least one row")
    mean = x[mask].mean(axis=0)
    n = x.shape[0] if n_output is None else int(n_output)
    if n < 0:
        raise ValueError("n_output must be non-negative")
    return LevelControl(
        values=np.repeat(mean[None, :], n, axis=0),
        provenance={
            "name": "train_mean_adapter",
            "causal_at_evaluation": True,
            "fit_partition": "TRAIN_only",
            "uses_future_inputs_within_evaluation_period": False,
            "uses_future_labels": False,
            "role": "deployable_static_patient_calibration",
        },
    )


def rolling_prefix_level(
    observations: np.ndarray,
    *,
    observation_available_at: np.ndarray,
    query_times: np.ndarray,
    observation_segment: np.ndarray,
    query_segment: np.ndarray,
    initial_level: np.ndarray,
    decay_seconds: float | None = None,
) -> LevelControl:
    """Estimate a slow level using only observations already available at query time.

    For a future-block target starting at ``t`` and ending at ``t+h``, pass
    ``observation_available_at=t+h``.  The target then cannot update the level
    until its entire block has elapsed.  State resets at every coverage segment.
    """

    obs = _as_2d(observations)
    available = np.asarray(observation_available_at, dtype=np.float64)
    qtime = np.asarray(query_times, dtype=np.float64)
    oseg = np.asarray(observation_segment, dtype=np.int64)
    qseg = np.asarray(query_segment, dtype=np.int64)
    init = np.asarray(initial_level, dtype=np.float64).reshape(-1)
    if obs.shape[1] != init.size:
        raise ValueError("initial_level width does not match observations")
    if available.shape != (obs.shape[0],) or oseg.shape != (obs.shape[0],):
        raise ValueError("observation metadata shape mismatch")
    if qtime.shape != qseg.shape or qtime.ndim != 1:
        raise ValueError("query metadata shape mismatch")
    if decay_seconds is not None and (not np.isfinite(decay_seconds) or decay_seconds <= 0):
        raise ValueError("decay_seconds must be finite and positive")

    out = np.empty((qtime.size, obs.shape[1]), dtype=np.float64)
    for seg in np.unique(qseg):
        q_idx = np.flatnonzero(qseg == seg)
        o_idx = np.flatnonzero(oseg == seg)
        q_idx = q_idx[np.argsort(qtime[q_idx], kind="stable")]
        o_idx = o_idx[np.argsort(available[o_idx], kind="stable")]
        acc = init.copy()
        weight = 0.0
        last_update = None
        cursor = 0
        for qi in q_idx:
            t = float(qtime[qi])
            while cursor < o_idx.size and available[o_idx[cursor]] <= t:
                oi = int(o_idx[cursor])
                ot = float(available[oi])
                if decay_seconds is not None and last_update is not None:
                    decay = float(np.exp(-(ot - last_update) / decay_seconds))
                    weight *= decay
                acc = (acc * weight + obs[oi]) / (weight + 1.0)
                weight += 1.0
                last_update = ot
                cursor += 1
            out[qi] = acc
    return LevelControl(
        values=out,
        provenance={
            "name": "rolling_prefix_level",
            "causal_at_evaluation": True,
            "update_rule": "observation enters only when observation_available_at <= query_time",
            "resets_at": "coverage_segment",
            "decay_seconds": decay_seconds,
            "uses_future_inputs_within_evaluation_period": False,
            "uses_future_labels": False,
            "role": "causal_slow_level_candidate",
        },
    )


def selection_period_mean_input_oracle(
    input_state: np.ndarray, selection_mask: np.ndarray, *, source_semantics: str
) -> LevelControl:
    """Noncausal *input-side* period oracle; outcome targets are rejected."""

    if source_semantics != "input_state":
        raise ValueError("selection_period_mean accepts input_state only, never target/outcome values")
    state = _as_2d(input_state)
    mask = np.asarray(selection_mask, dtype=bool)
    if mask.shape != (state.shape[0],) or not mask.any():
        raise ValueError("selection_mask must select at least one state row")
    mean = state[mask].mean(axis=0)
    return LevelControl(
        values=np.repeat(mean[None, :], state.shape[0], axis=0),
        provenance={
            "name": "selection_period_mean",
            "causal_at_evaluation": False,
            "source_semantics": "input_state",
            "uses_future_inputs_within_evaluation_period": True,
            "uses_future_labels": False,
            "role": "noncausal_input_oracle_diagnostic_only",
        },
    )
