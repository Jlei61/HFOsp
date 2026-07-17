"""Refine ``T_spectral`` inside an already frozen spectral class.

This module does not classify seizures.  The caller must provide one of the
three committed, frequency-defined ``simple_phenotype`` labels produced by
``plot_topic5_early_spectral_phenotypes.py``.  The label only selects which
band trace is allowed to localize the onset:

* ``broadband_1_150``: at least five of the six 1--150 Hz bands;
* ``gamma_nonbroadband``: the committed 30--80 Hz gamma trace;
* ``low_frequency_only``: at least two of delta/theta/alpha.

No LVFA/HYP morphology label, interictal scaffold, field similarity, outcome,
or subject-level phenotype vote is read here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from src.topic5_energy_timing import (
    detect_sustained_enhancement,
)


BANDS = (
    "delta_HYP_slow",
    "theta_preictal_PAC",
    "alpha_sharp_leq13",
    "beta_LVFA_low",
    "gamma_LVFA",
    "hg_low_ripple",
)

FROZEN_TYPES = (
    "broadband_1_150",
    "gamma_nonbroadband",
    "low_frequency_only",
)

TYPE_BAND_INDICES = {
    "broadband_1_150": (0, 1, 2, 3, 4, 5),
    "gamma_nonbroadband": (4,),
    "low_frequency_only": (0, 1, 2),
}

TYPE_REQUIRED_BANDS = {
    "broadband_1_150": 5,
    "gamma_nonbroadband": 1,
    "low_frequency_only": 2,
}


@dataclass(frozen=True)
class TypeRefinementConfig:
    quiet_pool_sec: tuple[float, float] = (-120.0, -20.0)
    quiet_window_sec: float = 20.0
    quiet_step_sec: float = 5.0
    early_domain_sec: tuple[float, float] = (-15.0, 20.0)
    local_pre_anchor_sec: float = 10.0
    local_post_anchor_sec: float = 3.0
    smooth_sec: float = 2.0
    flank_sec: float = 2.0
    post_sec: float = 5.0
    sustain_sec: float = 2.0
    baseline_quantile: float = 0.99
    spatial_quantile: float = 0.75


@dataclass(frozen=True)
class QuietBaseline:
    start_sec: float
    end_sec: float
    score: float
    n_candidates: int


@dataclass(frozen=True)
class TypeRefinedOnset:
    simple_phenotype: str
    detected: bool
    onset_sec: float
    baseline_start_sec: float
    baseline_end_sec: float
    n_band_post_sustained: int
    n_required_bands: int
    search_start_sec: float
    search_end_sec: float


def _validate_input(
    z_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_band_contact_time, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if z.ndim != 3 or z.shape[0] != len(BANDS):
        raise ValueError("expected six 1-150 Hz bands in [band,contact,time] order")
    if t.ndim != 1 or z.shape[2] != t.size:
        raise ValueError("rel_t must match the time dimension")
    if z.shape[1] < 1:
        raise ValueError("at least one timing contact is required")
    if not np.isfinite(z).all() or not np.isfinite(t).all():
        raise ValueError("type refinement requires finite cache arrays")
    if t.size < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError("rel_t must be strictly increasing")
    return z, t


def smooth_band_contacts(
    z_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
    *,
    smooth_sec: float,
) -> np.ndarray:
    """Smooth finite band/contact traces on their native time grid."""
    z, t = _validate_input(z_band_contact_time, rel_t)
    dt = float(np.median(np.diff(t)))
    width = max(1, int(round(float(smooth_sec) / dt)))
    if width == 1:
        return z.copy()
    return uniform_filter1d(z, size=width, axis=2, mode="nearest")


def select_quiet_baseline(
    smoothed_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
    *,
    config: TypeRefinementConfig = TypeRefinementConfig(),
) -> QuietBaseline:
    """Select the lowest, most stable complete pre-onset baseline window."""
    z, t = _validate_input(smoothed_band_contact_time, rel_t)
    spatial = np.quantile(z, config.spatial_quantile, axis=1)
    starts = np.arange(
        float(config.quiet_pool_sec[0]),
        float(config.quiet_pool_sec[1]) - float(config.quiet_window_sec)
        + 0.5 * float(config.quiet_step_sec),
        float(config.quiet_step_sec),
    )
    candidates: list[tuple[float, float, float]] = []
    for start in starts:
        end = float(start + config.quiet_window_sec)
        use = (t >= start) & (t < end)
        if int(np.sum(use)) < 3:
            continue
        values = spatial[:, use]
        level = np.median(values, axis=1)
        variability = np.median(np.abs(values - level[:, None]), axis=1)
        score = float(
            np.median(level)
            + 0.50 * np.median(variability)
            + 0.25 * np.quantile(values, 0.90)
        )
        candidates.append((score, float(start), end))
    if not candidates:
        raise ValueError("no complete quiet-baseline candidate")
    score, start, end = min(candidates, key=lambda item: (item[0], item[1]))
    return QuietBaseline(start, end, score, len(candidates))


def _refine_from_smoothed(
    smoothed_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
    simple_phenotype: str,
    anchor_sec: float,
    baseline: QuietBaseline,
    *,
    config: TypeRefinementConfig,
) -> TypeRefinedOnset:
    if simple_phenotype not in FROZEN_TYPES:
        raise ValueError(
            f"simple_phenotype must be one of {FROZEN_TYPES}; got {simple_phenotype!r}"
        )
    if not np.isfinite(anchor_sec):
        raise ValueError("a finite frozen-label timing anchor is required")
    z, t = _validate_input(smoothed_band_contact_time, rel_t)
    baseline_mask = (t >= baseline.start_sec) & (t < baseline.end_sec)
    if not np.any(baseline_mask):
        raise ValueError("selected quiet baseline is absent from time grid")
    centered = z - np.median(z[:, :, baseline_mask], axis=2, keepdims=True)
    band_traces = np.quantile(centered, config.spatial_quantile, axis=1)
    indices = TYPE_BAND_INDICES[simple_phenotype]
    required = TYPE_REQUIRED_BANDS[simple_phenotype]
    selected = band_traces[np.asarray(indices, dtype=int)]
    consensus = np.median(selected, axis=0)
    if simple_phenotype == "low_frequency_only":
        rises: list[float] = []
        for trace in selected:
            result = detect_sustained_enhancement(
                trace,
                t,
                baseline=(baseline.start_sec, baseline.end_sec),
                search=config.early_domain_sec,
                baseline_quantile=config.baseline_quantile,
                sustain_sec=config.sustain_sec,
            )
            if result.detected:
                rises.append(float(result.rise_sec))
        rises.sort()
        detected = len(rises) >= required
        onset_sec = float(rises[required - 1]) if detected else float("nan")
        # A rise at the very start of the fixed early domain is left-censored:
        # the energy was already increasing before the inspected window, so the
        # boundary itself is not an onset estimate.
        if detected and onset_sec <= (
            float(config.early_domain_sec[0]) + float(config.flank_sec)
        ):
            detected = False
        return TypeRefinedOnset(
            simple_phenotype=simple_phenotype,
            detected=detected,
            onset_sec=onset_sec,
            baseline_start_sec=float(baseline.start_sec),
            baseline_end_sec=float(baseline.end_sec),
            n_band_post_sustained=int(len(rises)),
            n_required_bands=int(required),
            search_start_sec=float(config.early_domain_sec[0]),
            search_end_sec=float(config.early_domain_sec[1]),
        )
    search_start = max(
        float(config.early_domain_sec[0]),
        float(anchor_sec) - float(config.local_pre_anchor_sec),
    )
    search_end = min(
        float(config.early_domain_sec[1]),
        float(anchor_sec) + float(config.local_post_anchor_sec),
    )
    if search_end <= search_start:
        raise ValueError("frozen-label anchor gives an empty early search window")
    dt = float(np.median(np.diff(t)))
    flank_frames = max(1, int(round(float(config.flank_sec) / dt)))
    centers = np.arange(flank_frames, t.size - flank_frames + 1, dtype=int)
    cumulative = np.concatenate(([0.0], np.cumsum(consensus)))
    before = (
        cumulative[centers] - cumulative[centers - flank_frames]
    ) / flank_frames
    after = (
        cumulative[centers + flank_frames] - cumulative[centers]
    ) / flank_frames
    step = after - before
    center_times = t[centers]
    baseline_step = (
        (center_times - config.flank_sec >= baseline.start_sec)
        & (center_times + config.flank_sec <= baseline.end_sec)
    )
    search_step = (
        (center_times >= search_start)
        & (center_times + config.flank_sec <= search_end)
    )
    if not np.any(baseline_step) or not np.any(search_step):
        raise ValueError("no complete change windows in baseline or anchored search")
    step_threshold = float(
        np.quantile(step[baseline_step], config.baseline_quantile)
    )
    search_indices = np.flatnonzero(search_step)
    local_best = int(search_indices[np.argmax(step[search_indices])])
    transition_sec = float(center_times[local_best])
    transition_detected = bool(
        float(step[local_best]) > step_threshold
        and transition_sec
        > float(search_start + config.flank_sec)
    )
    post_end = min(float(t[-1]), float(transition_sec + config.post_sec))
    if post_end - transition_sec < float(config.sustain_sec):
        consensus_sustained = False
        n_band_sustained = 0
    else:
        consensus_sustained = detect_sustained_enhancement(
            consensus,
            t,
            baseline=(baseline.start_sec, baseline.end_sec),
            search=(transition_sec, post_end),
            baseline_quantile=config.baseline_quantile,
            sustain_sec=config.sustain_sec,
        ).detected
        n_band_sustained = int(
            sum(
                detect_sustained_enhancement(
                    trace,
                    t,
                    baseline=(baseline.start_sec, baseline.end_sec),
                    search=(transition_sec, post_end),
                    baseline_quantile=config.baseline_quantile,
                    sustain_sec=config.sustain_sec,
                ).detected
                for trace in selected
            )
        )
    detected = bool(
        transition_detected
        and consensus_sustained
        and n_band_sustained >= required
    )
    return TypeRefinedOnset(
        simple_phenotype=simple_phenotype,
        detected=detected,
        onset_sec=transition_sec,
        baseline_start_sec=float(baseline.start_sec),
        baseline_end_sec=float(baseline.end_sec),
        n_band_post_sustained=n_band_sustained,
        n_required_bands=int(required),
        search_start_sec=search_start,
        search_end_sec=search_end,
    )


def refine_frozen_type_onset(
    z_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
    simple_phenotype: str,
    anchor_sec: float,
    *,
    config: TypeRefinementConfig = TypeRefinementConfig(),
) -> TypeRefinedOnset:
    """Localize onset for a caller-supplied frozen type without reclassification."""
    smoothed = smooth_band_contacts(
        z_band_contact_time,
        rel_t,
        smooth_sec=config.smooth_sec,
    )
    baseline = select_quiet_baseline(smoothed, rel_t, config=config)
    return _refine_from_smoothed(
        smoothed,
        rel_t,
        simple_phenotype,
        anchor_sec,
        baseline,
        config=config,
    )


def bootstrap_frozen_type_onset(
    z_band_contact_time: np.ndarray,
    rel_t: np.ndarray,
    simple_phenotype: str,
    anchor_sec: float,
    *,
    n_boot: int = 100,
    seed: int = 20260716,
    config: TypeRefinementConfig = TypeRefinementConfig(),
) -> dict[str, float | int]:
    """Contact-bootstrap timing stability for the same frozen type.

    The quiet interval is selected once from the full event.  Resamples test
    whether the same type-specific change remains spatially supported; they do
    not vote on the event's class.
    """
    smoothed = smooth_band_contacts(
        z_band_contact_time,
        rel_t,
        smooth_sec=config.smooth_sec,
    )
    baseline = select_quiet_baseline(smoothed, rel_t, config=config)
    full = _refine_from_smoothed(
        smoothed,
        rel_t,
        simple_phenotype,
        anchor_sec,
        baseline,
        config=config,
    )
    if int(n_boot) <= 0:
        return {
            "n_boot": 0,
            "n_detected": 0,
            "support_fraction": float("nan"),
            "q05_sec": float(full.onset_sec),
            "q95_sec": float(full.onset_sec),
            "width_sec": 0.0,
            "consistency_1s": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    n_contacts = smoothed.shape[1]
    detected: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.integers(0, n_contacts, size=n_contacts)
        result = _refine_from_smoothed(
            smoothed[:, sample],
            rel_t,
            simple_phenotype,
            anchor_sec,
            baseline,
            config=config,
        )
        if result.detected:
            detected.append(float(result.onset_sec))
    if not detected:
        return {
            "n_boot": int(n_boot),
            "n_detected": 0,
            "support_fraction": 0.0,
            "q05_sec": float("nan"),
            "q95_sec": float("nan"),
            "width_sec": float("nan"),
            "consistency_1s": 0.0,
        }
    values = np.asarray(detected, dtype=float)
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return {
        "n_boot": int(n_boot),
        "n_detected": int(values.size),
        "support_fraction": float(values.size / int(n_boot)),
        "q05_sec": float(q05),
        "q95_sec": float(q95),
        "width_sec": float(q95 - q05),
        "consistency_1s": float(np.mean(np.abs(values - full.onset_sec) <= 1.0)),
    }
