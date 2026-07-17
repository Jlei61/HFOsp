"""Episode-resolved, scaffold-blind spectral-onset diagnostics.

This module deliberately separates three operations:

1. prepare per-band/per-contact energy without consulting onset labels;
2. calibrate level and step gates from other seizures' distal background;
3. detect every sustained broadband episode, then assign one episode to an
   annotated seizure in a separate function.

The contract is provisional until the blinded pilot review is complete.  It
must not be used as a clinical seizure-onset detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpectralOnsetConfig:
    baseline: tuple[float, float] = (-120.0, -90.0)
    smooth_sec: float = 1.0
    change_flank_sec: float = 2.0
    level_quantile: float = 0.95
    step_z_threshold: float = 3.0
    min_bands: int = 3
    low_band_indices: tuple[int, ...] = (0, 1, 2)
    high_band_indices: tuple[int, ...] = (3, 4)
    spatial_fraction: float = 0.25
    min_contacts: int = 2
    min_episode_sec: float = 5.0
    max_gap_sec: float = 1.0
    min_state_occupancy: float = 0.60
    change_search_radius_sec: float = 3.0
    onset_bridge_sec: float = 10.0
    assignment_post_sec: float = 20.0
    n_boot: int = 100
    max_ci_width_sec: float = 3.0
    min_robust_scale: float = 0.05


@dataclass
class PreparedSpectralEvent:
    rel_t: np.ndarray
    smoothed: np.ndarray
    band_trace: np.ndarray
    consensus_trace: np.ndarray
    cell_step: np.ndarray
    cell_post_level: np.ndarray
    consensus_step: np.ndarray
    valid_step: np.ndarray
    baseline_mask: np.ndarray
    dt: float


@dataclass
class SpectralCalibrationSamples:
    level: np.ndarray
    cell_step: np.ndarray
    consensus_step: np.ndarray


@dataclass
class SpectralCalibration:
    level_threshold: np.ndarray
    cell_step_center: np.ndarray
    cell_step_scale: np.ndarray
    consensus_step_center: float
    consensus_step_scale: float
    n_level_samples: int
    n_step_samples: int
    n_source_seizures: int


@dataclass(frozen=True)
class SpectralEpisode:
    start_sec: float
    end_sec: float
    duration_sec: float
    state_occupancy: float
    change_sec: float
    change_step_z: float
    n_level_bands_at_change: int
    n_level_contacts_at_change: int
    n_step_bands: int
    n_step_contacts: int
    low_step_supported: bool
    high_step_supported: bool
    automatic_change_gate: bool
    bootstrap_q05_sec: float
    bootstrap_q95_sec: float
    bootstrap_ci_width_sec: float
    stable_candidate_time: bool
    precise_time: bool


@dataclass
class SpectralDiagnostics:
    rel_t: np.ndarray
    band_trace: np.ndarray
    consensus_trace: np.ndarray
    consensus_step_z: np.ndarray
    n_level_bands: np.ndarray
    n_level_contacts: np.ndarray
    contact_active_band_count: np.ndarray
    state_mask: np.ndarray
    episodes: list[SpectralEpisode]
    min_spatial_contacts: int
    calibration: SpectralCalibration


@dataclass(frozen=True)
class TargetEpisodeAssignment:
    status: str
    target_index: int | None
    anchor_start_sec: float
    anchor_end_sec: float
    n_connected_episodes: int
    n_prior_episodes: int


def _validate_time(rel_t: np.ndarray) -> float:
    t = np.asarray(rel_t, dtype=float)
    if t.ndim != 1 or t.size < 3:
        raise ValueError("rel_t must be a one-dimensional array with at least three samples")
    delta = np.diff(t)
    if not np.all(np.isfinite(delta)) or np.any(delta <= 0.0):
        raise ValueError("rel_t must be finite and strictly increasing")
    dt = float(np.median(delta))
    if not np.allclose(delta, dt, rtol=0.02, atol=max(1e-6, 0.02 * dt)):
        raise ValueError("rel_t must be approximately uniform")
    return dt


def _interval_mean(x: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``x[..., i+left:i+right]`` for every center i."""
    arr = np.asarray(x, dtype=float)
    if right <= left:
        raise ValueError("right must be greater than left")
    n_t = arr.shape[-1]
    center = np.arange(n_t)
    lo = center + int(left)
    hi = center + int(right)
    valid = (lo >= 0) & (hi <= n_t)
    lo_clip = np.clip(lo, 0, n_t)
    hi_clip = np.clip(hi, 0, n_t)
    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0.0)
    csum = np.concatenate([np.zeros((*arr.shape[:-1], 1)), np.cumsum(values, axis=-1)], axis=-1)
    cnum = np.concatenate(
        [np.zeros((*arr.shape[:-1], 1)), np.cumsum(finite.astype(float), axis=-1)], axis=-1
    )
    total = np.take(csum, hi_clip, axis=-1) - np.take(csum, lo_clip, axis=-1)
    count = np.take(cnum, hi_clip, axis=-1) - np.take(cnum, lo_clip, axis=-1)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    mean[..., ~valid] = np.nan
    return mean, valid


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if int(width) <= 1:
        return np.asarray(x, dtype=float).copy()
    arr = np.asarray(x, dtype=float)
    kernel = np.ones(int(width), dtype=float)
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.empty_like(flat)
    for row_idx, row in enumerate(flat):
        finite = np.isfinite(row)
        num = np.convolve(np.where(finite, row, 0.0), kernel, mode="same")
        den = np.convolve(finite.astype(float), kernel, mode="same")
        out[row_idx] = np.divide(
            num, den, out=np.full(row.shape, np.nan), where=den > 0
        )
    return out.reshape(arr.shape)


def prepare_spectral_event(
    z_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
    *,
    config: SpectralOnsetConfig = SpectralOnsetConfig(),
) -> PreparedSpectralEvent:
    """Re-centre, smooth, and compute label-blind level/step diagnostics."""
    z = np.asarray(z_band_contact_time, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if z.ndim != 3:
        raise ValueError("z_band_contact_time must be [band, contact, time]")
    if z.shape[2] != t.size:
        raise ValueError("time dimension must match rel_t")
    if z.shape[0] < config.min_bands:
        raise ValueError("fewer bands than min_bands")
    if z.shape[1] < config.min_contacts:
        raise ValueError("fewer contacts than min_contacts")
    dt = _validate_time(t)
    baseline = (t >= config.baseline[0]) & (t < config.baseline[1])
    if np.sum(baseline) < 3:
        raise ValueError(f"insufficient distal baseline in {config.baseline}")
    center = np.nanmedian(z[:, :, baseline], axis=2, keepdims=True)
    delta = z - center
    smooth_frames = max(1, int(round(config.smooth_sec / dt)))
    smoothed = _moving_average(delta, smooth_frames)
    flank = max(1, int(round(config.change_flank_sec / dt)))
    before, valid_before = _interval_mean(smoothed, -flank, 0)
    after, valid_after = _interval_mean(smoothed, 0, flank)
    cell_step = after - before
    valid_step = valid_before & valid_after
    post_frames = max(1, int(round(config.min_episode_sec / dt)))
    cell_post_level, _ = _interval_mean(smoothed, 0, post_frames)
    band_trace = np.nanquantile(smoothed, 0.75, axis=1)
    consensus = np.nanmedian(band_trace, axis=0)
    consensus_before, _ = _interval_mean(consensus, -flank, 0)
    consensus_after, _ = _interval_mean(consensus, 0, flank)
    consensus_step = consensus_after - consensus_before
    return PreparedSpectralEvent(
        rel_t=t,
        smoothed=smoothed,
        band_trace=band_trace,
        consensus_trace=consensus,
        cell_step=cell_step,
        cell_post_level=cell_post_level,
        consensus_step=consensus_step,
        valid_step=valid_step,
        baseline_mask=baseline,
        dt=dt,
    )


def calibration_samples(event: PreparedSpectralEvent) -> SpectralCalibrationSamples:
    """Extract this event's distal-background samples for LOSO calibration."""
    t = event.rel_t
    baseline_centers = (
        event.baseline_mask
        & event.valid_step
        & (t - event.dt >= t[event.baseline_mask][0])
        & np.isfinite(event.consensus_step)
    )
    # Requiring all flanks inside the baseline avoids transition samples that
    # borrow from the post-baseline search region.
    flank_margin = max(0.0, float(np.nanmax(t[event.baseline_mask]) - np.nanmin(t[event.baseline_mask])))
    del flank_margin  # documented intent; exact containment is checked below
    b0 = float(t[event.baseline_mask][0])
    b1 = float(t[event.baseline_mask][-1] + event.dt)
    half = np.where(event.valid_step)[0]
    if half.size:
        # Infer the symmetric step half-width from the first valid center.
        n_half = int(half[0])
        baseline_centers &= (t >= b0 + n_half * event.dt) & (t <= b1 - n_half * event.dt)
    if not np.any(baseline_centers):
        raise ValueError("no complete step centers inside distal baseline")
    return SpectralCalibrationSamples(
        level=np.asarray(event.smoothed[:, :, event.baseline_mask], dtype=float),
        cell_step=np.asarray(event.cell_step[:, :, baseline_centers], dtype=float),
        consensus_step=np.asarray(event.consensus_step[baseline_centers], dtype=float),
    )


def fit_spectral_calibration(
    samples: Sequence[SpectralCalibrationSamples],
    *,
    config: SpectralOnsetConfig = SpectralOnsetConfig(),
) -> SpectralCalibration:
    """Fit one subject-level calibration from one or more background sets."""
    if not samples:
        raise ValueError("at least one calibration sample is required")
    shape = samples[0].level.shape[:2]
    if any(sample.level.shape[:2] != shape for sample in samples):
        raise ValueError("band/contact shapes differ across calibration seizures")
    level = np.concatenate([sample.level for sample in samples], axis=2)
    step = np.concatenate([sample.cell_step for sample in samples], axis=2)
    consensus = np.concatenate([sample.consensus_step for sample in samples])
    level_threshold = np.nanquantile(level, config.level_quantile, axis=2)
    step_center = np.nanmedian(step, axis=2)
    step_scale = 1.4826 * np.nanmedian(np.abs(step - step_center[:, :, None]), axis=2)
    step_scale = np.maximum(step_scale, config.min_robust_scale)
    consensus_center = float(np.nanmedian(consensus))
    consensus_scale = float(1.4826 * np.nanmedian(np.abs(consensus - consensus_center)))
    consensus_scale = max(consensus_scale, config.min_robust_scale)
    return SpectralCalibration(
        level_threshold=level_threshold,
        cell_step_center=step_center,
        cell_step_scale=step_scale,
        consensus_step_center=consensus_center,
        consensus_step_scale=consensus_scale,
        n_level_samples=int(level.shape[2]),
        n_step_samples=int(step.shape[2]),
        n_source_seizures=len(samples),
    )


def _fill_short_false_runs(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0 or not np.any(out):
        return out
    false_mask = ~out
    padded = np.r_[False, false_mask, False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for start, stop in edges.reshape(-1, 2):
        start -= 1
        stop -= 1
        # Do not extend episodes into leading/trailing background.
        if start > 0 and stop < out.size and stop - start <= max_gap_frames:
            out[start:stop] = True
    return out


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _persistent_episode_mask(
    state_mask: np.ndarray,
    *,
    min_frames: int,
    max_gap_frames: int,
    min_occupancy: float,
) -> np.ndarray:
    """Return episode envelopes seeded by future-window state occupancy.

    A seed is an observed broadband-state frame whose following
    ``min_frames`` window is occupied above ``min_occupancy`` after only
    brief gaps have been filled.  Each qualifying seed contributes that
    full future window to the episode envelope.  This implements the stated
    persistence contract without incorrectly requiring five uninterrupted
    seconds of broadband state.
    """
    raw = np.asarray(state_mask, dtype=bool)
    n_t = raw.size
    width = max(1, int(min_frames))
    if n_t < width or not np.any(raw):
        return np.zeros(n_t, dtype=bool)
    gap_filled = _fill_short_false_runs(raw, max_gap_frames=max_gap_frames)
    occupancy, valid = _interval_mean(gap_filled.astype(float), 0, width)
    seeds = raw & valid & (occupancy >= float(min_occupancy))
    seed_idx = np.flatnonzero(seeds)
    if seed_idx.size == 0:
        return np.zeros(n_t, dtype=bool)
    # Difference-array union of [seed, seed + width) intervals.
    delta = np.zeros(n_t + 1, dtype=np.int32)
    np.add.at(delta, seed_idx, 1)
    np.add.at(delta, np.minimum(seed_idx + width, n_t), -1)
    return np.cumsum(delta[:-1]) > 0


def _bootstrap_change_interval(
    event: PreparedSpectralEvent,
    center_idx: int,
    *,
    config: SpectralOnsetConfig,
    seed: int,
) -> tuple[float, float]:
    if config.n_boot <= 0:
        t = float(event.rel_t[center_idx])
        return t, t
    radius = float(config.change_search_radius_sec)
    use = np.flatnonzero(
        (event.rel_t >= event.rel_t[center_idx] - radius)
        & (event.rel_t <= event.rel_t[center_idx] + radius)
        & event.valid_step
    )
    if use.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n_band, n_contact, _ = event.smoothed.shape
    flank = max(1, int(round(config.change_flank_sec / event.dt)))
    lo = max(0, int(use[0]) - flank)
    hi = min(event.rel_t.size, int(use[-1]) + flank + 1)
    local_use = use - lo
    peaks = np.empty(config.n_boot, dtype=float)
    for boot in range(config.n_boot):
        band_idx = rng.integers(0, n_band, size=n_band)
        contact_idx = rng.integers(0, n_contact, size=n_contact)
        sampled = event.smoothed[band_idx][:, contact_idx, lo:hi]
        band_trace = np.nanquantile(sampled, 0.75, axis=1)
        consensus = np.nanmedian(band_trace, axis=0)
        before, _ = _interval_mean(consensus, -flank, 0)
        after, _ = _interval_mean(consensus, 0, flank)
        step = after - before
        peaks[boot] = float(
            event.rel_t[use[int(np.nanargmax(step[local_use]))]]
        )
    return tuple(float(v) for v in np.nanquantile(peaks, [0.05, 0.95]))


def detect_spectral_episodes(
    event: PreparedSpectralEvent,
    calibration: SpectralCalibration,
    *,
    search: tuple[float, float],
    config: SpectralOnsetConfig = SpectralOnsetConfig(),
    seed: int = 20260714,
) -> SpectralDiagnostics:
    """Detect every sustained broadband episode without reading onset labels."""
    if calibration.level_threshold.shape != event.smoothed.shape[:2]:
        raise ValueError("calibration and event band/contact shapes differ")
    t = event.rel_t
    n_band, n_contact, _ = event.smoothed.shape
    min_contacts = max(config.min_contacts, int(np.ceil(config.spatial_fraction * n_contact)))
    level_active = event.smoothed > calibration.level_threshold[:, :, None]
    band_level_active = np.sum(level_active, axis=1) >= min_contacts
    contact_band_count = np.sum(level_active, axis=0)
    contact_level_active = contact_band_count >= config.min_bands
    n_level_bands = np.sum(band_level_active, axis=0)
    n_level_contacts = np.sum(contact_level_active, axis=0)
    low = np.any(band_level_active[np.asarray(config.low_band_indices, dtype=int)], axis=0)
    high = np.any(band_level_active[np.asarray(config.high_band_indices, dtype=int)], axis=0)
    search_mask = (t >= float(search[0])) & (t <= float(search[1]))
    state_mask = (
        (n_level_bands >= config.min_bands)
        & low
        & high
        & (n_level_contacts >= min_contacts)
        & search_mask
    )
    episode_mask = _persistent_episode_mask(
        state_mask,
        min_frames=max(1, int(round(config.min_episode_sec / event.dt))),
        max_gap_frames=max(0, int(round(config.max_gap_sec / event.dt))),
        min_occupancy=config.min_state_occupancy,
    )
    cell_step_z = (
        event.cell_step - calibration.cell_step_center[:, :, None]
    ) / calibration.cell_step_scale[:, :, None]
    consensus_step_z = (
        event.consensus_step - calibration.consensus_step_center
    ) / calibration.consensus_step_scale
    step_cell_active_all = (
        (cell_step_z >= config.step_z_threshold)
        & (event.cell_post_level > calibration.level_threshold[:, :, None])
    )
    band_step_active_all = np.sum(step_cell_active_all, axis=1) >= min_contacts
    contact_step_active_all = np.sum(step_cell_active_all, axis=0) >= config.min_bands
    n_step_bands_all = np.sum(band_step_active_all, axis=0)
    n_step_contacts_all = np.sum(contact_step_active_all, axis=0)
    low_step_all = np.any(
        band_step_active_all[np.asarray(config.low_band_indices, dtype=int)], axis=0
    )
    high_step_all = np.any(
        band_step_active_all[np.asarray(config.high_band_indices, dtype=int)], axis=0
    )
    automatic_step_mask = (
        (consensus_step_z >= config.step_z_threshold)
        & (n_step_bands_all >= config.min_bands)
        & (n_step_contacts_all >= min_contacts)
        & low_step_all
        & high_step_all
        & event.valid_step
        & search_mask
    )
    episodes: list[SpectralEpisode] = []
    for run_index, (start, stop) in enumerate(_true_runs(episode_mask)):
        duration = float((stop - start) * event.dt)
        occupancy = float(np.mean(state_mask[start:stop]))
        # Duration is guaranteed by the seed envelope; keep the explicit
        # numerical guard for non-standard frame grids and future refactors.
        if duration + 0.5 * event.dt < config.min_episode_sec:
            continue
        radius = config.change_search_radius_sec
        change_use = np.flatnonzero(
            (t >= t[start] - config.onset_bridge_sec)
            & (t <= t[start] + radius)
            & event.valid_step
            & search_mask
            & np.isfinite(consensus_step_z)
        )
        if change_use.size == 0:
            continue
        gated_use = change_use[automatic_step_mask[change_use]]
        if gated_use.size:
            # Consecutive points are one change candidate.  Select the peak of
            # the earliest complete candidate connected to the later state,
            # not a larger step after the state has already stabilized.
            local_runs = _true_runs(automatic_step_mask[change_use])
            first_start, first_stop = local_runs[0]
            first_candidate = change_use[first_start:first_stop]
            change_idx = int(
                first_candidate[int(np.nanargmax(consensus_step_z[first_candidate]))]
            )
        else:
            change_idx = int(change_use[int(np.nanargmax(consensus_step_z[change_use]))])
        n_step_bands = int(n_step_bands_all[change_idx])
        n_step_contacts = int(n_step_contacts_all[change_idx])
        low_step = bool(low_step_all[change_idx])
        high_step = bool(high_step_all[change_idx])
        automatic_change = bool(automatic_step_mask[change_idx])
        # Every sustained connected broadband episode receives a candidate
        # timing interval.  The stricter automatic step gate determines
        # primary eligibility, not whether uncertainty is reported at all.
        q05, q95 = _bootstrap_change_interval(
            event,
            change_idx,
            config=config,
            seed=int(seed + 1009 * run_index),
        )
        width = float(q95 - q05) if np.isfinite(q05) and np.isfinite(q95) else float("nan")
        stable_candidate = bool(np.isfinite(width) and width <= 5.0)
        precise = bool(automatic_change and np.isfinite(width) and width <= config.max_ci_width_sec)
        episodes.append(
            SpectralEpisode(
                start_sec=float(t[start]),
                end_sec=float(t[stop - 1]),
                duration_sec=duration,
                state_occupancy=occupancy,
                change_sec=float(t[change_idx]),
                change_step_z=float(consensus_step_z[change_idx]),
                n_level_bands_at_change=int(n_level_bands[change_idx]),
                n_level_contacts_at_change=int(n_level_contacts[change_idx]),
                n_step_bands=n_step_bands,
                n_step_contacts=n_step_contacts,
                low_step_supported=low_step,
                high_step_supported=high_step,
                automatic_change_gate=automatic_change,
                bootstrap_q05_sec=q05,
                bootstrap_q95_sec=q95,
                bootstrap_ci_width_sec=width,
                stable_candidate_time=stable_candidate,
                precise_time=precise,
            )
        )
    return SpectralDiagnostics(
        rel_t=t,
        band_trace=event.band_trace,
        consensus_trace=event.consensus_trace,
        consensus_step_z=consensus_step_z,
        n_level_bands=n_level_bands,
        n_level_contacts=n_level_contacts,
        contact_active_band_count=contact_band_count,
        state_mask=state_mask,
        episodes=episodes,
        min_spatial_contacts=min_contacts,
        calibration=calibration,
    )


def assign_target_episode(
    episodes: Sequence[SpectralEpisode],
    *,
    eeg_onset_sec: float,
    clinical_onset_sec: float,
    config: SpectralOnsetConfig = SpectralOnsetConfig(),
) -> TargetEpisodeAssignment:
    """Assign a label-blind episode to the target seizure using annotations only here."""
    anchor_start = float(min(eeg_onset_sec, clinical_onset_sec))
    anchor_end = float(max(eeg_onset_sec, clinical_onset_sec) + config.assignment_post_sec)
    connected = [
        i
        for i, episode in enumerate(episodes)
        if episode.end_sec >= anchor_start and episode.start_sec <= anchor_end
    ]
    prior = [i for i, episode in enumerate(episodes) if episode.end_sec < anchor_start]
    if connected:
        target = min(connected, key=lambda i: episodes[i].change_sec)
        status = (
            "confirmed_precise_T"
            if episodes[target].precise_time
            else "broadband_but_imprecise_T"
        )
    elif prior:
        target = None
        status = "separate_prior_episode"
    else:
        target = None
        status = "no_detectable_broadband_transition"
    return TargetEpisodeAssignment(
        status=status,
        target_index=target,
        anchor_start_sec=anchor_start,
        anchor_end_sec=anchor_end,
        n_connected_episodes=len(connected),
        n_prior_episodes=len(prior),
    )


def episode_to_dict(episode: SpectralEpisode | None) -> dict:
    return {} if episode is None else asdict(episode)


def config_to_dict(config: SpectralOnsetConfig) -> dict:
    return asdict(config)
