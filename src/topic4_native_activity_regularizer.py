"""Causal, label-free proxy for unsupported native-sheet recruitment.

The statistic is deliberately narrower than a causal-mechanism claim.  It asks
whether above-background activity in a saved sheet bin can be reached from
activity that was already classified as supported through the frozen delayed
E-to-E graph.  Unsupported activity is not allowed to legitimise itself on the
next frame.  One largest connected seed is exempt after a sufficiently long
global reset, so separate self-limited episodes in one observation window are
not automatically penalised.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label as connected_component_labels


@dataclass(frozen=True)
class UnsupportedWindowResult:
    start_frame: int
    stop_frame: int
    baseline_start_frame: int
    baseline_stop_frame: int
    eligible_excess_mass: float
    unsupported_mass: float
    unsupported_fraction: float
    exempt_seed_mass: float
    n_episode_seeds: int
    low_count_excess_mass: float
    background_mass_per_frame: float
    background_active_bin_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def _largest_patch(active: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Return the mass-largest 8-connected active patch, deterministically."""
    labels, n_labels = connected_component_labels(
        np.asarray(active, bool), structure=np.ones((3, 3), bool),
    )
    if n_labels == 0:
        return np.zeros_like(active, bool)
    totals = np.asarray([
        float(np.sum(mass[labels == index]))
        for index in range(1, n_labels + 1)
    ])
    # np.argmax provides a deterministic lower-label tie break.
    return labels == int(np.argmax(totals) + 1)


def unsupported_activity_for_window(
        activity_counts: np.ndarray, support_by_lag: np.ndarray, *,
        start_frame: int, stop_frame: int,
        baseline_start_frame: int, baseline_stop_frame: int,
        minimum_active_neurons: int = 2,
        minimum_parent_support: float = 0.001,
        reset_frames: int = 18,
        background_quantile: float = 0.95,
        response_tail_frames: int = 13,
        response_tail_fraction: float = 0.5) -> UnsupportedWindowResult:
    """Measure unsupported above-background activity in one observation window.

    ``support_by_lag[lag, target_bin, source_bin]`` must be the delayed frozen
    E-to-E support returned by :func:`binned_ee_delay_support`.  Only activity
    already classified as supported can support later activity.  At the first
    active frame after ``reset_frames`` globally inactive frames, the largest
    connected patch is an exempt seed; simultaneous disconnected patches must
    earn delayed support or contribute to the numerator.

    The ratio denominator contains above-background mass in bins meeting the
    activity threshold.  Lower-count mass and pre-window background are returned
    separately so neither can dilute the score silently.
    """
    counts = np.asarray(activity_counts, float)
    support = np.asarray(support_by_lag, float)
    start, stop = int(start_frame), int(stop_frame)
    base_start, base_stop = int(baseline_start_frame), int(baseline_stop_frame)
    minimum_active_neurons = int(minimum_active_neurons)
    threshold = float(minimum_parent_support)
    reset_frames = int(reset_frames)
    quantile = float(background_quantile)
    response_tail_frames = int(response_tail_frames)
    response_tail_fraction = float(response_tail_fraction)
    if counts.ndim != 3 or np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("activity_counts must be finite nonnegative [time,y,x]")
    n_bins = counts.shape[1] * counts.shape[2]
    if (support.ndim != 3 or support.shape[1:] != (n_bins, n_bins)
            or np.any(support < 0) or not np.all(np.isfinite(support))):
        raise ValueError("support_by_lag does not match the activity grid")
    if not (0 <= base_start < base_stop <= start < stop <= len(counts)):
        raise ValueError("baseline and analysis frame bounds are invalid")
    if (minimum_active_neurons <= 0 or threshold <= 0 or reset_frames <= 0
            or not 0 <= quantile <= 1 or response_tail_frames <= 0
            or not 0 < response_tail_fraction <= 1):
        raise ValueError("regularizer thresholds are invalid")

    baseline = counts[base_start:base_stop]
    background = np.quantile(baseline, quantile, axis=0)
    local_counts = counts[base_start:stop]
    excess = np.maximum(local_counts - background[None], 0.0)
    active = (local_counts >= minimum_active_neurons) & (excess > 0.0)
    flat_excess = excess.reshape(len(local_counts), n_bins)
    flat_active = active.reshape(len(local_counts), n_bins)
    supported_history = np.zeros((len(local_counts), n_bins), float)
    filtered_history = np.zeros_like(supported_history)
    if response_tail_frames == 1:
        tail_weights = np.ones(1, float)
    else:
        tail_weights = np.geomspace(
            1.0, response_tail_fraction, response_tail_frames,
        )

    episode_open = False
    silent_run = reset_frames
    unsupported_mass = 0.0
    eligible_mass = 0.0
    low_count_mass = 0.0
    seed_mass = 0.0
    n_seeds = 0
    for frame in range(base_start, stop):
        local = frame - base_start
        active_bins = flat_active[local]
        if frame >= start:
            eligible_mass += float(np.sum(flat_excess[local, active_bins]))
            low_count_mass += float(np.sum(flat_excess[local, ~active_bins]))
        if not np.any(active_bins):
            silent_run += 1
            if silent_run >= reset_frames:
                episode_open = False
            n_tail = min(response_tail_frames, local + 1)
            indices = local - np.arange(n_tail)
            filtered_history[local] = (
                tail_weights[:n_tail] @ supported_history[indices]
            )
            continue

        arriving = np.zeros(n_bins, float)
        target_bins = np.flatnonzero(active_bins)
        for lag in range(1, min(len(support), local + 1)):
            prior = filtered_history[local - lag]
            source_bins = np.flatnonzero(prior > 0.0)
            if len(source_bins):
                arriving[target_bins] += (
                    support[lag][np.ix_(target_bins, source_bins)]
                    @ prior[source_bins]
                )
        support_fraction = np.clip(arriving / threshold, 0.0, 1.0)
        supported = active_bins & (support_fraction >= 1.0)

        if not episode_open and silent_run >= reset_frames:
            seed = _largest_patch(
                active[local], excess[local],
            ).ravel()
            supported |= seed
            if frame >= start:
                seed_mass += float(np.sum(flat_excess[local, seed]))
            n_seeds += int(frame >= start)
            episode_open = True

        unsupported = active_bins & ~supported
        if frame >= start and np.any(unsupported):
            unsupported_mass += float(np.sum(
                flat_excess[local, unsupported]
                * (1.0 - support_fraction[unsupported])
            ))
        supported_history[local, supported] = flat_excess[local, supported]
        n_tail = min(response_tail_frames, local + 1)
        indices = local - np.arange(n_tail)
        filtered_history[local] = (
            tail_weights[:n_tail] @ supported_history[indices]
        )
        silent_run = 0

    fraction = (
        float(unsupported_mass / eligible_mass) if eligible_mass > 0.0 else 0.0
    )
    return UnsupportedWindowResult(
        start_frame=start, stop_frame=stop,
        baseline_start_frame=base_start, baseline_stop_frame=base_stop,
        eligible_excess_mass=float(eligible_mass),
        unsupported_mass=float(unsupported_mass),
        unsupported_fraction=fraction,
        exempt_seed_mass=float(seed_mass), n_episode_seeds=int(n_seeds),
        low_count_excess_mass=float(low_count_mass),
        background_mass_per_frame=float(np.mean(np.sum(baseline, axis=(1, 2)))),
        background_active_bin_fraction=float(np.mean(
            baseline >= minimum_active_neurons
        )),
    )


def summarize_network_windows(rows: list[UnsupportedWindowResult]) -> dict:
    """Aggregate events within one physical network; networks remain the unit."""
    if not rows:
        return {
            "n_events": 0, "mean_unsupported_fraction": None,
            "median_unsupported_fraction": None,
        }
    values = np.asarray([row.unsupported_fraction for row in rows], float)
    return {
        "n_events": int(len(rows)),
        "mean_unsupported_fraction": float(np.mean(values)),
        "median_unsupported_fraction": float(np.median(values)),
        "q25_unsupported_fraction": float(np.quantile(values, 0.25)),
        "q75_unsupported_fraction": float(np.quantile(values, 0.75)),
        "total_unsupported_mass": float(sum(row.unsupported_mass for row in rows)),
        "total_eligible_excess_mass": float(
            sum(row.eligible_excess_mass for row in rows)
        ),
        "mean_background_mass_per_frame": float(np.mean([
            row.background_mass_per_frame for row in rows
        ])),
        "mean_background_active_bin_fraction": float(np.mean([
            row.background_active_bin_fraction for row in rows
        ])),
    }
