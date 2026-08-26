"""Full-event causal exposure screen for H3/T2 time scales.

This is deliberately a screen, not the persistent generator model.  It asks
whether a decayed history of cross-fitted IED-load innovations improves the
next exact interval or spatial mark beyond the complete explicit event
history, and compares it with a same-form exposure driven by older events.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import contract
from .bridge import BridgeArrays, _explicit_history, _uniform_take


EXPOSURE_REVISION = "h3_s0_full_event_physical_time_placebo_v3"


@dataclass
class ExposureDataset:
    arrays: BridgeArrays
    metadata: dict


def pre_event_innovation_predictors(history: np.ndarray,
                                    participation: np.ndarray) -> np.ndarray:
    """Covariates available immediately before the current event.

    The event's own load, contact set, rank and group count are intentionally
    absent.  The stored traces include the current event, so its unit jump is
    removed before they are used to estimate expected load.
    """
    n_contacts = participation.shape[1]
    trace30_pre = history[:, 1:2] - 1.0
    trace120_pre = history[:, 2:3] - 1.0
    contact_trace_pre = history[:, -n_contacts:] - participation.astype(np.float32)
    return np.concatenate([
        history[:, 0:1], trace30_pre, trace120_pre,
        history[:, 7:10], contact_trace_pre,
    ], axis=1).astype(np.float32)


def cross_fitted_load_innovation(predictors: np.ndarray, load: np.ndarray,
                                 split: np.ndarray) -> np.ndarray:
    """TRAIN cross-fit and TRAIN-only development prediction."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = np.flatnonzero(split == 0)
    validation = np.flatnonzero(split == 1)
    if len(train) < 20:
        raise ValueError("too few training events for innovation cross-fit")
    expected = np.zeros(len(load), dtype=np.float64)
    folds = [fold for fold in np.array_split(train, min(5, len(train))) if len(fold)]

    def estimator():
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    for fold in folds:
        fit = np.setdiff1d(train, fold, assume_unique=True)
        model = estimator().fit(predictors[fit], load[fit])
        expected[fold] = model.predict(predictors[fold])
    final = estimator().fit(predictors[train], load[train])
    if len(validation):
        expected[validation] = final.predict(predictors[validation])
    innovation = load.astype(np.float64) - expected
    center = float(np.mean(innovation[train]))
    scale = max(float(np.std(innovation[train])), 1e-4)
    return ((innovation - center) / scale).astype(np.float32)


def cross_fitted_participation_innovation(predictors: np.ndarray,
                                          participation: np.ndarray,
                                          split: np.ndarray) -> np.ndarray:
    """Cross-fitted per-contact surprise using pre-event covariates only."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = np.flatnonzero(split == 0)
    validation = np.flatnonzero(split == 1)
    if len(train) < 20:
        raise ValueError("too few training events for participation innovation")
    target = participation.astype(np.float64)
    expected = np.zeros_like(target)
    folds = [fold for fold in np.array_split(train, min(5, len(train))) if len(fold)]

    def estimator():
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    for fold in folds:
        fit = np.setdiff1d(train, fold, assume_unique=True)
        model = estimator().fit(predictors[fit], target[fit])
        expected[fold] = model.predict(predictors[fold])
    final = estimator().fit(predictors[train], target[train])
    if len(validation):
        expected[validation] = final.predict(predictors[validation])
    innovation = target - expected
    center = np.mean(innovation[train], axis=0)
    scale = np.maximum(np.std(innovation[train], axis=0), 0.05)
    return ((innovation - center) / scale).astype(np.float32)


def exposure_pair(times: np.ndarray, innovation: np.ndarray,
                  session: np.ndarray, split: np.ndarray,
                  tau_minutes: float, *, decay_clock: str = "physical_time",
                  event_count_step_minutes: float | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Real exposure and a causal older-event placebo, reset by segment.

    ``event_count`` is a clock-control arm: every transition receives the same
    decay implied by the TRAIN-only median IEI.  Its nominal tau therefore has
    the same typical memory as the physical-time exposure, but ignores actual
    interval variability.
    """
    if decay_clock not in {"physical_time", "event_count"}:
        raise ValueError(f"unsupported exposure decay clock {decay_clock!r}")
    if decay_clock == "event_count" and (
        event_count_step_minutes is None or event_count_step_minutes <= 0
    ):
        raise ValueError("event_count clock requires a positive TRAIN-only step")
    value_shape = innovation.shape[1:]
    real = np.zeros((len(times), *value_shape), dtype=np.float64)
    placebo = np.zeros((len(times), *value_shape), dtype=np.float64)
    shifts: list[dict] = []
    for split_code in (0, 1):
        for session_id in np.unique(session[split == split_code]):
            idx = np.flatnonzero((split == split_code) & (session == session_id))
            if not len(idx):
                continue
            # Placebo is shifted on the same physical clock as tau, not by an
            # event count whose duration changes with IED rate.  Use at least
            # 30 min or 3*tau, capped at one third of the observed segment so
            # short records retain a nonzero comparison interval.
            segment_times = times[idx]
            duration_minutes = max(
                (float(segment_times[-1]) - float(segment_times[0])) / 60.0, 0.0
            )
            requested_delay = max(30.0, 3.0 * float(tau_minutes))
            effective_delay = min(
                requested_delay,
                max(duration_minutes / 3.0, np.finfo(float).eps),
            )
            delayed = np.zeros((len(idx), *value_shape), dtype=np.float64)
            event_shifts = []
            for source_position, event_index in enumerate(idx.tolist()):
                destination = int(np.searchsorted(
                    segment_times,
                    float(times[event_index]) + effective_delay * 60.0,
                    side="left",
                ))
                destination = max(destination, source_position + 1)
                if destination < len(idx):
                    delayed[destination] += innovation[event_index]
                    event_shifts.append(destination - source_position)
            u_real = np.zeros(value_shape, dtype=np.float64)
            u_placebo = np.zeros(value_shape, dtype=np.float64)
            previous = float(times[idx[0]])
            for local, event_index in enumerate(idx.tolist()):
                physical_dt = max((float(times[event_index]) - previous) / 60.0, 0.0)
                dt_minutes = (
                    physical_dt if decay_clock == "physical_time"
                    else 0.0 if local == 0
                    else float(event_count_step_minutes)
                )
                decay = math.exp(-dt_minutes / float(tau_minutes))
                u_real *= decay
                u_placebo *= decay
                u_real += innovation[event_index]
                u_placebo += delayed[local]
                real[event_index] = u_real
                placebo[event_index] = u_placebo
                previous = float(times[event_index])
            shifts.append({
                "split": int(split_code), "session": int(session_id),
                "n_events": int(len(idx)),
                "requested_delay_minutes": float(requested_delay),
                "effective_delay_minutes": float(effective_delay),
                "median_delay_events": (
                    float(np.median(event_shifts)) if event_shifts else None
                ),
                "decay_clock": decay_clock,
                "event_count_step_minutes": (
                    float(event_count_step_minutes)
                    if event_count_step_minutes is not None else None
                ),
            })
    return real.astype(np.float32), placebo.astype(np.float32), shifts


def build_exposure_dataset(subject: str, tau_minutes: float,
                           exposure_kind: str = "load",
                           decay_clock: str = "physical_time",
                           max_train: int = 20000,
                           max_validation: int = 10000) -> ExposureDataset:
    payload = torch.load(contract.COHORT_CACHE, map_location="cpu", weights_only=False)[subject]
    times = payload["event_time"].numpy().astype(np.float64)
    session = payload["session_index"].numpy().astype(np.int64)
    participation = payload["participation"].numpy().astype(bool)
    n_groups = payload["n_groups"].numpy().astype(np.int64)
    marks = payload["marks"].numpy().astype(np.float32)
    load = payload["load"].numpy().astype(np.float32)
    dataset = str(payload["dataset"])
    history = _explicit_history(
        times, session, participation, n_groups, load, marks[:, :, 1], dataset
    )
    bound = contract.load_split(subject)
    split = np.full(len(times), 2, dtype=np.int8)
    split[times < bound.dev_end_epoch] = 1
    split[times < bound.train_end_epoch] = 0
    predictors = pre_event_innovation_predictors(history, participation)
    if exposure_kind == "load":
        innovation = cross_fitted_load_innovation(predictors, load, split)
    elif exposure_kind == "participation":
        innovation = cross_fitted_participation_innovation(
            predictors, participation, split
        )
    else:
        raise ValueError(f"unsupported exposure kind {exposure_kind!r}")
    train_pair = (
        (split[1:] == 0) & (split[:-1] == 0)
        & (session[1:] == session[:-1]) & (np.diff(times) > 0)
    )
    train_iei_minutes = np.diff(times)[train_pair] / 60.0
    if not len(train_iei_minutes):
        raise ValueError(f"{subject}: no TRAIN interval for exposure clock")
    event_count_step_minutes = float(np.median(train_iei_minutes))
    real, placebo, shifts = exposure_pair(
        times, innovation, session, split, tau_minutes,
        decay_clock=decay_clock,
        event_count_step_minutes=event_count_step_minutes,
    )

    pair_ok = (
        (session[1:] == session[:-1])
        & (split[1:] == split[:-1])
        & (np.diff(times) > 0)
        & (split[:-1] < 2)
    )
    train = _uniform_take(np.flatnonzero(pair_ok & (split[:-1] == 0)), max_train)
    valid = _uniform_take(np.flatnonzero(pair_ok & (split[:-1] == 1)), max_validation)
    idx = np.concatenate([train, valid])
    row_split = np.r_[
        np.zeros(len(train), dtype=np.int8), np.ones(len(valid), dtype=np.int8)
    ]
    nxt = idx + 1
    arrays = BridgeArrays(
        subject=subject,
        history=history[idx],
        spectral=real[idx, None] if real.ndim == 1 else real[idx],
        raw=placebo[idx, None] if placebo.ndim == 1 else placebo[idx],
        log_next_iei=np.log(np.maximum(times[nxt] - times[idx], 1e-3)).astype(np.float32),
        participation=participation[nxt].astype(np.float32),
        rank=marks[nxt, :, 1].astype(np.float32),
        stop_fraction=(n_groups[nxt] / participation.shape[1]).astype(np.float32),
        split=row_split,
        current_time=times[idx], next_time=times[nxt],
        current_event_index=idx,
        observation_valid_fraction=np.ones(len(idx), dtype=np.float32),
    )
    arrays.validate()
    train_events = split == 0
    metadata = {
        "contract": contract.REVISION,
        "exposure_revision": EXPOSURE_REVISION,
        "subject": subject,
        "exposure_kind": exposure_kind,
        "tau_minutes": float(tau_minutes),
        "decay_clock": decay_clock,
        "event_count_step_minutes_train_median": event_count_step_minutes,
        "n_train": int(len(train)),
        "n_validation": int(len(valid)),
        "innovation_train_mean_max_abs": float(np.max(np.abs(
            np.mean(innovation[train_events], axis=0)
        ))),
        "innovation_train_std_median": float(np.median(
            np.std(innovation[train_events], axis=0)
        )),
        "real_placebo_train_correlation": float(np.corrcoef(
            real[train_events].reshape(-1), placebo[train_events].reshape(-1)
        )[0, 1]),
        "placebo_semantics": (
            "causal physical-time delayed innovations without circular wrap; "
            "requested max(30 min, 3*tau), segment-capped"
        ),
        "segment_shifts": shifts,
        "sealed_opened": False,
        "claim_boundary": "predictive distributed-exposure screen; not generator causality",
    }
    return ExposureDataset(arrays=arrays, metadata=metadata)
