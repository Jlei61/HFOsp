"""
Event periodicity analysis for interictal HFO population events.

Quantifies temporal structure at two levels:
  - per-channel: individual electrode IEI and PSD
  - population (group): packed population-event IEI and PSD

Metrics produced:
  - Welch PSD of binary pulse trains
  - specparam (FOOOF) periodic/aperiodic decomposition
  - IEI distribution fitting (MLE power-law + model comparison)
  - Peak significance via ISI-shuffle surrogate and Gamma renewal null
  - Phase 2 follow-up tools: packing sweep, centroid anchor comparison,
    hazard visualization, return map, and propagation stereotypy

Interpretation guardrails:
  - Group-level ~2 Hz PSD peaks should not be read as evidence for an intrinsic
    oscillator unless they survive refractory/null-model checks.
  - The current hazard estimator is a qualitative visualization, not a formal
    survival-analysis estimator.
  - The current return-map correlation is computed on log(IEI[n]), log(IEI[n+1]);
    its Pearson p-value is descriptive only because adjacent pairs are dependent.
  - The centroid-bypass timestamps remain tied to the legacy lagPatRaw->window
    mapping and do not constitute a fully independent reconstruction.
  - The propagation stereotypy metric is a sub-sampled summary statistic and is
    best treated as exploratory until re-checked with multi-seed / mixed-model
    analyses.

Consumes existing legacy assets (*_gpu.npz, *_lagPat.npz, *_packedTimes.npy)
without re-running detection or packing.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import welch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PSDResult:
    freqs: np.ndarray
    power: np.ndarray
    fs_pulse: float
    nperseg: int

@dataclass
class SpecparamResult:
    peaks: np.ndarray          # (n_peaks, 3): center_freq, power, bandwidth
    aperiodic_params: np.ndarray
    r_squared: float
    error: float
    freqs: np.ndarray          # log10 scale from specparam
    power_spectrum: np.ndarray # log10
    fooofed_spectrum: np.ndarray
    ap_fit: np.ndarray

@dataclass
class IEIFitResult:
    alpha: float               # MLE power-law exponent
    xmin: float
    n_tail: int                # samples used in tail fit
    pl_vs_ln_R: float          # log-likelihood ratio: power-law vs lognormal
    pl_vs_ln_p: float
    pl_vs_exp_R: float
    pl_vs_exp_p: float
    iei_mean: float
    iei_median: float
    iei_min: float
    n_total: int

@dataclass
class SurrogateResult:
    real_peak_power: float
    real_peak_freq: float
    null_peak_powers: np.ndarray   # (n_surrogates,)
    p_value: float
    n_surrogates: int
    method: str                    # 'isi_shuffle' or 'gamma_renewal'

@dataclass
class ChannelPeriodicityResult:
    channel: str
    psd: Optional[PSDResult] = None
    specparam: Optional[SpecparamResult] = None
    iei_fit: Optional[IEIFitResult] = None
    surrogate_isi: Optional[SurrogateResult] = None
    surrogate_gamma: Optional[SurrogateResult] = None
    n_events: int = 0
    recording_duration_sec: float = 0.0

@dataclass
class SubjectPeriodicityResult:
    subject: str
    dataset: str
    channels: List[ChannelPeriodicityResult] = field(default_factory=list)
    group: Optional[ChannelPeriodicityResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

def load_yuquan_subject_events(
    subject_dir: Path,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str], List[Tuple[float, float]]]:
    """Load per-channel events and packed times from Yuquan legacy assets.

    Returns
    -------
    per_ch_events : dict[str, ndarray(N,2)]
        Absolute-time event [start, end] per channel, sorted by start.
    packed_times : ndarray(M,2)
        Group population event times (absolute, sorted).
    ch_names : list[str]
        Channel names from lagPat.
    block_ranges : list[tuple(float,float)]
        (start_sec, end_sec) for each recording block, for gap detection.
    """
    subject_dir = Path(subject_dir)

    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        raise FileNotFoundError(f"No *_lagPat.npz in {subject_dir}")

    tmp = np.load(lagpat_files[0], allow_pickle=True)
    pack_chns = list(tmp["chnNames"])

    per_ch_events: Dict[str, list] = {c: [] for c in pack_chns}
    block_ranges: List[Tuple[float, float]] = []

    edf_files = sorted(subject_dir.glob("*.edf"))
    for edf in edf_files:
        gpu_file = edf.with_name(edf.stem + "_gpu.npz")
        if not gpu_file.exists():
            continue
        gpu = np.load(gpu_file, allow_pickle=True)
        dets = gpu["whole_dets"]
        start_t = float(gpu["start_time"])
        chns = list(gpu["chns_names"])

        block_dur = 2 * 3600.0
        block_ranges.append((start_t, start_t + block_dur))

        for chn in pack_chns:
            if chn in chns:
                idx = chns.index(chn)
                ch_dets = np.array(dets[idx]).reshape(-1, 2) + start_t
                per_ch_events[chn].append(ch_dets)

    for chn in pack_chns:
        if per_ch_events[chn]:
            cat = np.concatenate(per_ch_events[chn], axis=0)
            per_ch_events[chn] = cat[cat[:, 0].argsort()]
        else:
            per_ch_events[chn] = np.zeros((0, 2))

    packed_all = []
    for edf in edf_files:
        lagpat_f = edf.with_name(edf.stem + "_lagPat.npz")
        packed_f = edf.with_name(edf.stem + "_packedTimes.npy")
        gpu_f = edf.with_name(edf.stem + "_gpu.npz")
        if not (lagpat_f.exists() and packed_f.exists() and gpu_f.exists()):
            continue

        lp = np.load(lagpat_f, allow_pickle=True)
        lag_raw = lp["lagPatRaw"]
        packed = np.load(packed_f)
        gpu = np.load(gpu_f, allow_pickle=True)
        start_t = float(gpu["start_time"])

        if lag_raw.size == 0 or packed.size == 0:
            continue
        if 0 in lag_raw.shape or 0 in packed.shape:
            continue

        lag_min = np.min(lag_raw, axis=0)
        lag_max = np.max(lag_raw, axis=0)
        mean_win = np.mean(packed[:, 1] - packed[:, 0])

        offsets = np.zeros((len(lag_min), 2))
        offsets[:, 0] = lag_min % mean_win
        offsets[:, 1] = offsets[:, 0] + (lag_max - lag_min)

        calibrated = packed[:, 0:1] + offsets
        packed_abs = calibrated + start_t
        packed_all.append(packed_abs)

    if packed_all:
        packed_times = np.concatenate(packed_all, axis=0)
        packed_times = packed_times[packed_times[:, 0].argsort()]
    else:
        packed_times = np.zeros((0, 2))

    return per_ch_events, packed_times, pack_chns, block_ranges


def _try_load_gpu(path: Path) -> Optional[dict]:
    """Safely load a _gpu.npz file, returning None if corrupt."""
    try:
        import os
        if os.path.getsize(path) < 500:
            return None
        data = np.load(path, allow_pickle=True)
        _ = data["whole_dets"]
        return data
    except Exception:
        return None


def load_epilepsiae_subject_events(
    subject_dir: Path,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str], List[Tuple[float, float]]]:
    """Load events from Epilepsiae legacy assets.

    Epilepsiae _gpu.npz files may be corrupt stubs. Falls back to
    lagPat.npz for start_t and channel names, skipping per-channel
    raw detections when gpu is unavailable.
    """
    subject_dir = Path(subject_dir)

    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        raise FileNotFoundError(f"No *_lagPat.npz in {subject_dir}")

    tmp = np.load(lagpat_files[0], allow_pickle=True)
    pack_chns = list(tmp["chnNames"])

    per_ch_events: Dict[str, list] = {c: [] for c in pack_chns}
    block_ranges: List[Tuple[float, float]] = []

    block_dur_guess = 3600.0
    gpu_available = False

    for lp_file in lagpat_files:
        stem = lp_file.stem.replace("_lagPat", "")
        gpu_file = lp_file.with_name(stem + "_gpu.npz")

        lp = np.load(lp_file, allow_pickle=True)
        start_t = float(lp["start_t"])
        block_ranges.append((start_t, start_t + block_dur_guess))

        gpu = _try_load_gpu(gpu_file) if gpu_file.exists() else None
        if gpu is not None:
            gpu_available = True
            dets = gpu["whole_dets"]
            chns = list(gpu["chns_names"])
            for chn in pack_chns:
                if chn in chns:
                    idx = chns.index(chn)
                    ch_dets = np.array(dets[idx]).reshape(-1, 2) + start_t
                    per_ch_events[chn].append(ch_dets)

    if not gpu_available:
        logger.warning(f"No valid _gpu.npz for {subject_dir.name}; per-channel analysis unavailable")

    for chn in pack_chns:
        if per_ch_events[chn]:
            cat = np.concatenate(per_ch_events[chn], axis=0)
            per_ch_events[chn] = cat[cat[:, 0].argsort()]
        else:
            per_ch_events[chn] = np.zeros((0, 2))

    packed_all = []
    for lp_file in lagpat_files:
        stem = lp_file.stem.replace("_lagPat", "")
        packed_f = lp_file.with_name(stem + "_packedTimes.npy")
        if not packed_f.exists():
            continue

        lp = np.load(lp_file, allow_pickle=True)
        lag_raw = lp["lagPatRaw"]
        start_t = float(lp["start_t"])
        packed = np.load(packed_f)

        if lag_raw.size == 0 or packed.size == 0:
            continue
        if 0 in lag_raw.shape or 0 in packed.shape:
            continue

        lag_min = np.min(lag_raw, axis=0)
        lag_max = np.max(lag_raw, axis=0)
        mean_win = np.mean(packed[:, 1] - packed[:, 0])

        offsets = np.zeros((len(lag_min), 2))
        offsets[:, 0] = lag_min % mean_win
        offsets[:, 1] = offsets[:, 0] + (lag_max - lag_min)

        calibrated = packed[:, 0:1] + offsets
        packed_abs = calibrated + start_t
        packed_all.append(packed_abs)

    if packed_all:
        packed_times = np.concatenate(packed_all, axis=0)
        packed_times = packed_times[packed_times[:, 0].argsort()]
    else:
        packed_times = np.zeros((0, 2))

    return per_ch_events, packed_times, pack_chns, block_ranges


# ---------------------------------------------------------------------------
# Pulse train + PSD
# ---------------------------------------------------------------------------

def build_pulse_train(
    events: np.ndarray,
    fs: float = 100.0,
    mode: str = "rectangle",
    block_ranges: Optional[List[Tuple[float, float]]] = None,
    gap_threshold_sec: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert event times to a binary pulse train.

    Parameters
    ----------
    events : ndarray(N,2) — [start, end] in seconds (absolute time)
    fs : sampling rate for pulse train
    mode : 'rectangle' (legacy) or 'delta' (impulse at start only)
    block_ranges : if provided, detect gaps > gap_threshold_sec
    gap_threshold_sec : gaps larger than this are masked

    Returns
    -------
    pulse : 1-d array
    mask : 1-d bool array (True = valid, False = gap)
    t0 : start time offset
    """
    if len(events) < 2:
        return np.zeros(0), np.zeros(0, dtype=bool), 0.0

    t0 = events[0, 0]
    t_end = events[-1, 1]
    n_samples = int(fs * (t_end - t0)) + 1
    pulse = np.zeros(n_samples, dtype=np.float32)
    mask = np.ones(n_samples, dtype=bool)

    if mode == "rectangle":
        for s, e in events:
            i0 = int((s - t0) * fs)
            i1 = int((e - t0) * fs)
            pulse[i0:i1] = 1.0
    elif mode == "delta":
        for s, _ in events:
            i0 = int((s - t0) * fs)
            if 0 <= i0 < n_samples:
                pulse[i0] = 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if block_ranges:
        sorted_blocks = sorted(block_ranges, key=lambda x: x[0])
        for i in range(1, len(sorted_blocks)):
            prev_end = sorted_blocks[i - 1][1]
            curr_start = sorted_blocks[i][0]
            gap = curr_start - prev_end
            if gap > gap_threshold_sec:
                g_start = int((prev_end - t0) * fs)
                g_end = int((curr_start - t0) * fs)
                g_start = max(0, g_start)
                g_end = min(n_samples, g_end)
                mask[g_start:g_end] = False

    return pulse, mask, t0


def compute_event_psd(
    pulse: np.ndarray,
    mask: np.ndarray,
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
    max_freq: float = 10.0,
    per_block_average: bool = True,
) -> PSDResult:
    """Compute Welch PSD from pulse train, respecting gap mask.

    If per_block_average=True, splits at mask gaps, computes PSD per segment,
    and returns the weighted average.
    """
    nperseg = int(fs * nperseg_sec)

    if per_block_average and not mask.all():
        segments = _split_by_mask(pulse, mask, min_length=nperseg)
        if not segments:
            segments = [pulse]
    else:
        segments = [pulse]

    psd_sum = None
    weight_sum = 0.0
    freqs = None

    for seg in segments:
        if len(seg) < nperseg:
            seg_nperseg = len(seg)
        else:
            seg_nperseg = nperseg
        ff, pp = welch(seg, fs=fs, nperseg=seg_nperseg)
        freq_mask = ff <= max_freq
        ff = ff[freq_mask]
        pp = pp[freq_mask]
        w = len(seg)

        if psd_sum is None:
            psd_sum = pp * w
            freqs = ff
        else:
            if len(ff) == len(freqs):
                psd_sum += pp * w
            else:
                min_len = min(len(ff), len(freqs))
                psd_sum = psd_sum[:min_len] + pp[:min_len] * w
                freqs = freqs[:min_len]
        weight_sum += w

    power = psd_sum / weight_sum if weight_sum > 0 else psd_sum
    return PSDResult(freqs=freqs, power=power, fs_pulse=fs, nperseg=nperseg)


def _split_by_mask(arr: np.ndarray, mask: np.ndarray, min_length: int) -> List[np.ndarray]:
    """Split array at contiguous False regions in mask, keep segments >= min_length."""
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]

    segs = []
    for s, e in zip(starts, ends):
        if e - s >= min_length:
            segs.append(arr[s:e])
    return segs


# ---------------------------------------------------------------------------
# specparam (FOOOF) decomposition
# ---------------------------------------------------------------------------

def fit_psd_periodic(
    psd_result: PSDResult,
    freq_range: Tuple[float, float] = (0.5, 10.0),
    aperiodic_mode: str = "knee",
    max_n_peaks: int = 2,
    min_peak_height: float = 0.1,
    peak_width_limits: Tuple[float, float] = (0.6, 12.0),
) -> Optional[SpecparamResult]:
    """Fit specparam model to event PSD.

    Compatible with both specparam >=2.0 and legacy fooof.
    """
    try:
        from specparam import SpectralModel
        _is_v2 = True
    except ImportError:
        from fooof import FOOOF as SpectralModel
        _is_v2 = False

    freqs = psd_result.freqs
    power = psd_result.power

    if len(freqs) < 5 or np.all(power == 0):
        return None

    model = SpectralModel(
        aperiodic_mode=aperiodic_mode,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_width_limits=peak_width_limits,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(freqs, power, freq_range=list(freq_range))

    if _is_v2:
        peaks_raw = model.get_params("peak")
        peaks = np.atleast_2d(peaks_raw) if peaks_raw is not None and len(peaks_raw) > 0 else np.empty((0, 3))
        ap_params = model.get_params("aperiodic")
        metrics = model.results.metrics.results
        r2 = metrics.get("gof_rsquared", 0.0)
        err = metrics.get("error_mae", 0.0)
        fit_freqs = model.data.freqs
        power_spec = model.data.power_spectrum
        modeled = model.results.model.modeled_spectrum
        ap_fit = model.results.model._ap_fit
    else:
        peaks_raw = model.peak_params_
        peaks = np.atleast_2d(peaks_raw) if peaks_raw is not None else np.empty((0, 3))
        ap_params = model.aperiodic_params_
        r2 = model.r_squared_
        err = model.error_
        fit_freqs = model.freqs
        power_spec = model.power_spectrum
        modeled = model.fooofed_spectrum_
        ap_fit = model._ap_fit

    return SpecparamResult(
        peaks=peaks,
        aperiodic_params=np.asarray(ap_params),
        r_squared=float(r2),
        error=float(err),
        freqs=np.asarray(fit_freqs),
        power_spectrum=np.asarray(power_spec),
        fooofed_spectrum=np.asarray(modeled),
        ap_fit=np.asarray(ap_fit),
    )


# ---------------------------------------------------------------------------
# IEI distribution fitting
# ---------------------------------------------------------------------------

def compute_iei(events: np.ndarray, block_ranges: Optional[List[Tuple[float, float]]] = None,
                gap_threshold_sec: float = 300.0) -> np.ndarray:
    """Compute inter-event intervals, excluding cross-block gaps."""
    if len(events) < 2:
        return np.array([])

    starts = events[:, 0]
    iei = np.diff(starts)

    if block_ranges:
        sorted_blocks = sorted(block_ranges, key=lambda x: x[0])
        gap_starts = []
        gap_ends = []
        for i in range(1, len(sorted_blocks)):
            prev_end = sorted_blocks[i - 1][1]
            curr_start = sorted_blocks[i][0]
            if curr_start - prev_end > gap_threshold_sec:
                gap_starts.append(prev_end)
                gap_ends.append(curr_start)

        if gap_starts:
            mid_times = (starts[:-1] + starts[1:]) / 2.0
            keep = np.ones(len(iei), dtype=bool)
            for gs, ge in zip(gap_starts, gap_ends):
                keep &= ~((mid_times > gs) & (mid_times < ge))
            iei = iei[keep]

    return iei


def fit_iei_distribution(
    iei: np.ndarray,
    xmin_lower: float = 0.5,
) -> Optional[IEIFitResult]:
    """Fit IEI distribution with MLE power-law and compare with alternatives."""
    import powerlaw

    iei_pos = iei[iei > 0]
    if len(iei_pos) < 50:
        return None

    fit = powerlaw.Fit(iei_pos, xmin=max(xmin_lower, np.min(iei_pos)),
                       verbose=False)

    try:
        R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    except Exception:
        R_ln, p_ln = 0.0, 1.0

    try:
        R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    except Exception:
        R_exp, p_exp = 0.0, 1.0

    n_tail = int(np.sum(iei_pos >= fit.power_law.xmin))

    return IEIFitResult(
        alpha=fit.power_law.alpha,
        xmin=fit.power_law.xmin,
        n_tail=n_tail,
        pl_vs_ln_R=R_ln,
        pl_vs_ln_p=p_ln,
        pl_vs_exp_R=R_exp,
        pl_vs_exp_p=p_exp,
        iei_mean=float(np.mean(iei_pos)),
        iei_median=float(np.median(iei_pos)),
        iei_min=float(np.min(iei_pos)),
        n_total=len(iei_pos),
    )


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def _find_primary_peak(sp_result: SpecparamResult,
                       freq_range: Tuple[float, float] = (0.5, 4.0)) -> Tuple[float, float]:
    """Find the strongest periodic peak within freq_range."""
    peaks = sp_result.peaks
    if len(peaks) == 0:
        return 0.0, 0.0

    in_range = (peaks[:, 0] >= freq_range[0]) & (peaks[:, 0] <= freq_range[1])
    if not np.any(in_range):
        return 0.0, 0.0

    candidates = peaks[in_range]
    best_idx = np.argmax(candidates[:, 1])
    residual = sp_result.power_spectrum - sp_result.ap_fit
    freq_idx = np.argmin(np.abs(sp_result.freqs - candidates[best_idx, 0]))
    peak_power = residual[freq_idx]
    return candidates[best_idx, 0], peak_power


def test_peak_significance_isi_shuffle(
    events: np.ndarray,
    real_sp: SpecparamResult,
    n_surrogates: int = 200,
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
    block_ranges: Optional[List[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SurrogateResult]:
    """ISI-shuffle surrogate test: randomize IEI order, recompute PSD peak."""
    if rng is None:
        rng = np.random.default_rng(42)

    real_freq, real_power = _find_primary_peak(real_sp)
    if real_power <= 0:
        return None

    iei = np.diff(events[:, 0])
    if len(iei) < 10:
        return None

    null_powers = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        shuffled_iei = rng.permutation(iei)
        surr_starts = np.cumsum(np.r_[events[0, 0], shuffled_iei])
        mean_dur = np.mean(events[:, 1] - events[:, 0])
        surr_events = np.column_stack([surr_starts, surr_starts + mean_dur])

        pulse, mask, _ = build_pulse_train(surr_events, fs=fs, mode="rectangle",
                                           block_ranges=block_ranges)
        if len(pulse) < int(fs * nperseg_sec):
            continue
        psd = compute_event_psd(pulse, mask, fs=fs, nperseg_sec=nperseg_sec)
        sp = fit_psd_periodic(psd)
        if sp is None:
            continue
        _, surr_power = _find_primary_peak(sp)
        null_powers[i] = surr_power

    p_val = float(np.mean(null_powers >= real_power))
    return SurrogateResult(
        real_peak_power=real_power,
        real_peak_freq=real_freq,
        null_peak_powers=null_powers,
        p_value=p_val,
        n_surrogates=n_surrogates,
        method="isi_shuffle",
    )


def test_peak_significance_gamma_renewal(
    events: np.ndarray,
    real_sp: SpecparamResult,
    n_surrogates: int = 200,
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
    block_ranges: Optional[List[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SurrogateResult]:
    """Gamma renewal process null: match firing rate and refractory period."""
    if rng is None:
        rng = np.random.default_rng(42)

    real_freq, real_power = _find_primary_peak(real_sp)
    if real_power <= 0:
        return None

    iei = np.diff(events[:, 0])
    if len(iei) < 10:
        return None

    mean_iei = np.mean(iei)
    min_iei = np.percentile(iei, 2)
    total_duration = events[-1, 0] - events[0, 0]
    n_events = len(events)
    mean_dur = np.mean(events[:, 1] - events[:, 0])

    shifted = iei - min_iei
    shifted_mean = np.mean(shifted)
    shifted_var = np.var(shifted)

    if shifted_mean <= 0 or shifted_var <= 0:
        shape = 1.0
        scale = mean_iei
    else:
        shape = (shifted_mean ** 2) / shifted_var
        scale = shifted_var / shifted_mean

    null_powers = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        surr_iei = rng.gamma(shape, scale, size=n_events - 1) + min_iei
        surr_starts = np.cumsum(np.r_[events[0, 0], surr_iei])
        surr_events = np.column_stack([surr_starts, surr_starts + mean_dur])

        pulse, mask, _ = build_pulse_train(surr_events, fs=fs, mode="rectangle",
                                           block_ranges=block_ranges)
        if len(pulse) < int(fs * nperseg_sec):
            continue
        psd = compute_event_psd(pulse, mask, fs=fs, nperseg_sec=nperseg_sec)
        sp = fit_psd_periodic(psd)
        if sp is None:
            continue
        _, surr_power = _find_primary_peak(sp)
        null_powers[i] = surr_power

    p_val = float(np.mean(null_powers >= real_power))
    return SurrogateResult(
        real_peak_power=real_power,
        real_peak_freq=real_freq,
        null_peak_powers=null_powers,
        p_value=p_val,
        n_surrogates=n_surrogates,
        method="gamma_renewal",
    )


# ---------------------------------------------------------------------------
# Phase 2 follow-up (PR1): Analytic renewal PSD + SOZ dead-time split
# ---------------------------------------------------------------------------

def compute_renewal_psd_analytic(
    iei: np.ndarray,
    freqs: np.ndarray,
    event_dur: float = 0.0,
) -> np.ndarray:
    """Analytic PSD of a shifted-gamma renewal process.

    Parameters
    ----------
    iei : array of inter-event intervals (seconds)
    freqs : 1-D frequency array (Hz), must not contain 0
    event_dur : mean event duration for sinc^2 rectangle correction; 0 = skip

    Returns
    -------
    S : one-sided PSD prediction on ``freqs``.

    Notes
    -----
    Uses a shifted-gamma characteristic function
    ``phi(w) = exp(i*w*tau_r) * (1 - i*w*theta)^(-k)``, where:
      - ``tau_r`` = 2nd percentile of IEI
      - ``k, theta`` are method-of-moments estimates on ``iei - tau_r``.
    DC is not defined (phi=1 -> 0/0), so callers should pass ``freqs > 0``.
    """
    iei = np.asarray(iei, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    iei = iei[np.isfinite(iei) & (iei > 0)]

    if iei.size < 5:
        return np.zeros_like(freqs, dtype=float)
    if freqs.size == 0:
        return np.array([], dtype=float)
    if np.any(freqs <= 0):
        raise ValueError("freqs must be strictly positive (exclude DC bin)")

    tau_r = float(np.percentile(iei, 2))
    mu = float(np.mean(iei))
    sigma2 = float(np.var(iei))
    shifted_mu = mu - tau_r

    eps = 1e-12
    lam = 1.0 / max(mu, eps)
    if shifted_mu <= eps or sigma2 <= eps:
        k = 1.0
        theta = max(shifted_mu, eps)
    else:
        k = (shifted_mu ** 2) / sigma2
        theta = sigma2 / shifted_mu

    omega = 2.0 * np.pi * freqs
    z = 1.0 - 1j * omega * theta
    phi_gamma = np.exp(-k * np.log(z))
    phi = np.exp(1j * omega * tau_r) * phi_gamma

    denom = 1.0 - phi
    denom[np.abs(denom) < eps] = eps
    s_delta = lam * np.real((1.0 + phi) / denom)
    s_delta = np.clip(s_delta, 0.0, None)

    if event_dur > 0:
        s_delta = s_delta * (np.sinc(freqs * float(event_dur)) ** 2)

    return s_delta


def _load_group_events_with_soz_labels(
    subject_dir: Path,
    dataset: str,
    soz_channels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load group-event absolute times and SOZ participation labels."""
    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    soz_set = set(soz_channels or [])

    if not lagpat_files:
        return {
            "warning": "no_lagpat_files",
            "events": np.zeros((0, 2), dtype=float),
            "is_soz": np.zeros(0, dtype=bool),
            "n_channels_total": 0,
            "n_soz_channels_input": len(soz_set),
            "n_soz_channels_matched": 0,
        }

    def _to_abs_events(lp_file: Path, lp: Any) -> np.ndarray:
        stem = lp_file.stem.replace("_lagPat", "")
        packed_f = lp_file.with_name(stem + "_packedTimes.npy")
        if not packed_f.exists():
            return np.zeros((0, 2), dtype=float)

        packed = np.load(packed_f)
        lag_raw = lp["lagPatRaw"]
        if packed.size == 0 or lag_raw.size == 0 or 0 in packed.shape or 0 in lag_raw.shape:
            return np.zeros((0, 2), dtype=float)

        if dataset == "yuquan":
            gpu_f = lp_file.with_name(stem + "_gpu.npz")
            if not gpu_f.exists():
                return np.zeros((0, 2), dtype=float)
            gpu = _try_load_gpu(gpu_f)
            if gpu is None:
                return np.zeros((0, 2), dtype=float)
            start_t = float(gpu["start_time"])
        else:
            start_t = float(np.ravel(lp["start_t"])[0])

        lag_min = np.min(lag_raw, axis=0)
        lag_max = np.max(lag_raw, axis=0)
        mean_win = float(np.mean(packed[:, 1] - packed[:, 0]))
        if mean_win <= 0:
            return np.zeros((0, 2), dtype=float)

        offsets = np.zeros((len(lag_min), 2))
        offsets[:, 0] = lag_min % mean_win
        offsets[:, 1] = offsets[:, 0] + (lag_max - lag_min)
        return packed[:, 0:1] + offsets + start_t

    all_events = []
    all_is_soz = []
    all_chn_names = []

    for lp_file in lagpat_files:
        lp = np.load(lp_file, allow_pickle=True)
        events_bool = lp["eventsBool"]
        chn_names = [str(x) for x in list(lp["chnNames"])]
        all_chn_names.extend(chn_names)

        packed_abs = _to_abs_events(lp_file, lp)
        if packed_abs.size == 0:
            continue

        n_ev = packed_abs.shape[0]
        n_ev_bool = events_bool.shape[1] if events_bool.ndim == 2 else 0
        if n_ev_bool == 0:
            continue
        if n_ev != n_ev_bool:
            min_ev = min(n_ev, n_ev_bool)
            packed_abs = packed_abs[:min_ev]
            events_bool = events_bool[:, :min_ev]

        soz_mask_ch = np.array([c in soz_set for c in chn_names], dtype=bool)
        if np.any(soz_mask_ch):
            is_soz_ev = np.any(events_bool[soz_mask_ch, :] > 0, axis=0)
        else:
            is_soz_ev = np.zeros(events_bool.shape[1], dtype=bool)

        all_events.append(packed_abs)
        all_is_soz.append(is_soz_ev)

    soz_overlap = len(set(all_chn_names) & soz_set)
    if not all_events:
        out = {
            "warning": "no_group_events",
            "events": np.zeros((0, 2), dtype=float),
            "is_soz": np.zeros(0, dtype=bool),
            "n_channels_total": len(set(all_chn_names)),
            "n_soz_channels_input": len(soz_set),
            "n_soz_channels_matched": soz_overlap,
        }
        if soz_set and soz_overlap == 0:
            out["warning"] = "no_soz_channel_match"
        return out

    events = np.concatenate(all_events, axis=0)
    is_soz = np.concatenate(all_is_soz, axis=0)
    order = np.argsort(events[:, 0])

    out = {
        "events": events[order],
        "is_soz": is_soz[order],
        "n_channels_total": len(set(all_chn_names)),
        "n_soz_channels_input": len(soz_set),
        "n_soz_channels_matched": soz_overlap,
    }
    if soz_set and soz_overlap == 0:
        out["warning"] = "no_soz_channel_match"
    return out


def compute_soz_stratified_deadtime(
    subject_dir: Path,
    dataset: str,
    soz_channels: List[str],
    block_ranges: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """Split group events by SOZ participation and summarize IEI dead-time stats."""
    loaded = _load_group_events_with_soz_labels(subject_dir, dataset, soz_channels)
    events = loaded["events"]
    is_soz = loaded["is_soz"]

    def _stats(ev: np.ndarray) -> Dict[str, Any]:
        if ev.shape[0] < 2:
            return {"n_events": int(ev.shape[0]), "n_iei": 0}
        iei = compute_iei(ev, block_ranges=block_ranges)
        if iei.size == 0:
            return {"n_events": int(ev.shape[0]), "n_iei": 0}
        return {
            "n_events": int(ev.shape[0]),
            "n_iei": int(iei.size),
            "iei_min": float(np.min(iei)),
            "iei_median": float(np.median(iei)),
            "iei_mean": float(np.mean(iei)),
            "iei_p02": float(np.percentile(iei, 2)),
        }

    out: Dict[str, Any] = {
        "n_channels_total": int(loaded["n_channels_total"]),
        "n_soz_channels_input": int(loaded["n_soz_channels_input"]),
        "n_soz_channels_matched": int(loaded["n_soz_channels_matched"]),
        "all": _stats(events),
        "soz": _stats(events[is_soz]),
        "nonsoz": _stats(events[~is_soz]),
    }
    if "warning" in loaded:
        out["warning"] = loaded["warning"]
    return out


# ---------------------------------------------------------------------------
# Subject-level runner
# ---------------------------------------------------------------------------

def run_subject_periodicity(
    subject_dir: Path,
    dataset: str,
    subject_name: str,
    n_surrogates: int = 200,
    fs_pulse: float = 100.0,
    nperseg_sec: float = 500.0,
    run_surrogates: bool = True,
) -> SubjectPeriodicityResult:
    """Run full periodicity analysis for one subject."""
    loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events
    per_ch, packed, ch_names, block_ranges = loader(subject_dir)

    result = SubjectPeriodicityResult(
        subject=subject_name,
        dataset=dataset,
        metadata={"subject_dir": str(subject_dir), "n_channels": len(ch_names)},
    )

    for chn in ch_names:
        events = per_ch.get(chn, np.zeros((0, 2)))
        ch_result = _analyze_event_series(
            events, chn, block_ranges, fs_pulse, nperseg_sec,
            0,  # surrogates only for group level
        )
        result.channels.append(ch_result)

    if len(packed) >= 2:
        grp = _analyze_event_series(
            packed, "group", block_ranges, fs_pulse, nperseg_sec,
            n_surrogates if run_surrogates else 0,
        )
        result.group = grp

    return result


def _analyze_event_series(
    events: np.ndarray,
    name: str,
    block_ranges: List[Tuple[float, float]],
    fs: float,
    nperseg_sec: float,
    n_surrogates: int,
) -> ChannelPeriodicityResult:
    """Analyze a single event series (channel or group)."""
    res = ChannelPeriodicityResult(channel=name, n_events=len(events))

    if len(events) < 10:
        return res

    res.recording_duration_sec = float(events[-1, 1] - events[0, 0])

    pulse, mask, _ = build_pulse_train(events, fs=fs, mode="rectangle",
                                       block_ranges=block_ranges)
    if len(pulse) < int(fs * 10):
        return res

    psd = compute_event_psd(pulse, mask, fs=fs, nperseg_sec=nperseg_sec)
    res.psd = psd

    sp = fit_psd_periodic(psd)
    res.specparam = sp

    iei = compute_iei(events, block_ranges=block_ranges)
    if len(iei) >= 50:
        res.iei_fit = fit_iei_distribution(iei)

    if sp is not None and n_surrogates > 0:
        logger.info(f"  {name}: running {n_surrogates} ISI-shuffle surrogates...")
        res.surrogate_isi = test_peak_significance_isi_shuffle(
            events, sp, n_surrogates=n_surrogates, fs=fs,
            nperseg_sec=nperseg_sec, block_ranges=block_ranges,
        )
        logger.info(f"  {name}: running {n_surrogates} Gamma renewal surrogates...")
        res.surrogate_gamma = test_peak_significance_gamma_renewal(
            events, sp, n_surrogates=n_surrogates, fs=fs,
            nperseg_sec=nperseg_sec, block_ranges=block_ranges,
        )

    return res


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2: PackWinLen sweep (Experiment 1)
# ---------------------------------------------------------------------------

def load_raw_detections_yuquan_per_block(
    subject_dir: Path,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], float]], List[str], List[Tuple[float, float]]]:
    """Load per-channel raw detections from _gpu.npz, one dict per block.

    Returns
    -------
    blocks : list of (per_ch_events_relative, start_t)
        per_ch_events_relative has zero-based times within the 2h block.
    ch_names : list[str]
    block_ranges : list[tuple(float,float)] — absolute epoch ranges.
    """
    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        raise FileNotFoundError(f"No *_lagPat.npz in {subject_dir}")

    pack_chns = list(np.load(lagpat_files[0], allow_pickle=True)["chnNames"])
    blocks: List[Tuple[Dict[str, np.ndarray], float]] = []
    block_ranges: List[Tuple[float, float]] = []

    for edf in sorted(subject_dir.glob("*.edf")):
        gpu_file = edf.with_name(edf.stem + "_gpu.npz")
        if not gpu_file.exists():
            continue
        gpu = _try_load_gpu(gpu_file)
        if gpu is None:
            continue
        dets = gpu["whole_dets"]
        start_t = float(gpu["start_time"])
        chns = list(gpu["chns_names"])
        block_dur = 2 * 3600.0
        block_ranges.append((start_t, start_t + block_dur))

        per_ch: Dict[str, np.ndarray] = {}
        for chn in pack_chns:
            if chn in chns:
                idx = chns.index(chn)
                ch_dets = np.array(dets[idx]).reshape(-1, 2)
                per_ch[chn] = ch_dets[ch_dets[:, 0].argsort()]
            else:
                per_ch[chn] = np.zeros((0, 2))
        blocks.append((per_ch, start_t))

    return blocks, pack_chns, block_ranges


def repack_and_analyze(
    blocks: List[Tuple[Dict[str, np.ndarray], float]],
    window_sec: float,
    block_ranges: List[Tuple[float, float]],
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
) -> Dict[str, Any]:
    """Re-pack raw detections with a given window_sec, then compute PSD + specparam.

    Processes each block separately (relative time), then converts to absolute.
    """
    from src.group_event_analysis import build_windows_from_detections

    all_events = []
    for per_ch_rel, start_t in blocks:
        windows = build_windows_from_detections(per_ch_rel, window_sec=window_sec)
        for w in windows:
            all_events.append([w.start + start_t, w.end + start_t])

    n_events = len(all_events)
    result: Dict[str, Any] = {
        "window_sec": window_sec,
        "n_events": n_events,
        "peak_freq": None,
        "peak_power": None,
        "f_theory": 1.0 / window_sec,
    }

    if n_events < 10:
        return result

    events = np.array(all_events)
    events = events[events[:, 0].argsort()]

    pulse, mask, _ = build_pulse_train(events, fs=fs, mode="rectangle",
                                       block_ranges=block_ranges)
    if len(pulse) < int(fs * 10):
        return result

    psd = compute_event_psd(pulse, mask, fs=fs, nperseg_sec=nperseg_sec)
    sp = fit_psd_periodic(psd)
    if sp is not None:
        freq, power = _find_primary_peak(sp)
        if power > 0:
            result["peak_freq"] = float(freq)
            result["peak_power"] = float(power)

    iei = compute_iei(events, block_ranges=block_ranges)
    if len(iei) > 0:
        result["iei_min"] = float(np.min(iei))
        result["iei_mean"] = float(np.mean(iei))
        result["iei_median"] = float(np.median(iei))

    return result


def run_packing_sweep(
    subject_dir: Path,
    window_values: Optional[Sequence[float]] = None,
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
) -> List[Dict[str, Any]]:
    """Run Experiment 1: sweep window_sec, track f_peak.

    Parameters
    ----------
    window_values : window sizes in seconds; defaults to
        [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    """
    if window_values is None:
        window_values = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    blocks, ch_names, block_ranges = load_raw_detections_yuquan_per_block(subject_dir)
    results = []
    for w in window_values:
        logger.info(f"  PackWinLen sweep: W={w:.3f}s")
        r = repack_and_analyze(blocks, w, block_ranges, fs=fs, nperseg_sec=nperseg_sec)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Phase 2: Centroid bypass (Experiment 2)
# ---------------------------------------------------------------------------

def load_centroid_event_times(
    subject_dir: Path,
    dataset: str,
) -> Dict[str, np.ndarray]:
    """Extract three event time series from lagPatRaw for anchor comparison.

    Returns dict with keys:
      - 'window_start': (N,2) from packedTimes[:, 0]
      - 'mean_centroid': (N,2) using mean centroid across participating channels
      - 'ignition_centroid': (N,2) using earliest channel centroid

    All times are absolute (epoch seconds), but the centroid-based timestamps are
    still reconstructed inside the legacy packed-window frame via ``% mean_win``.
    They are useful for testing sensitivity to within-window anchor choice, not
    for claiming a fully independent absolute-timestamp reconstruction.
    """
    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        raise FileNotFoundError(f"No *_lagPat.npz in {subject_dir}")

    series: Dict[str, list] = {"window_start": [], "mean_centroid": [], "ignition_centroid": []}

    for lp_file in lagpat_files:
        stem = lp_file.stem.replace("_lagPat", "")
        packed_f = lp_file.with_name(stem + "_packedTimes.npy")
        if not packed_f.exists():
            continue

        lp = np.load(lp_file, allow_pickle=True)
        lag_raw = lp["lagPatRaw"]
        events_bool = lp["eventsBool"]

        if dataset == "yuquan":
            gpu_f = lp_file.with_name(stem + "_gpu.npz")
            if not gpu_f.exists():
                continue
            gpu = _try_load_gpu(gpu_f)
            if gpu is None:
                continue
            start_t = float(gpu["start_time"])
        else:
            start_t = float(lp["start_t"])

        packed = np.load(packed_f)
        if lag_raw.size == 0 or packed.size == 0:
            continue
        if 0 in lag_raw.shape or 0 in packed.shape:
            continue

        n_events = packed.shape[0]
        mean_win = np.mean(packed[:, 1] - packed[:, 0])
        event_dur_approx = mean_win * 0.1

        for i in range(n_events):
            mask = events_bool[:, i] > 0
            if not np.any(mask):
                continue

            centroids_i = lag_raw[mask, i]
            c_min = np.min(centroids_i)
            c_mean = np.mean(centroids_i)

            ws = packed[i, 0] + start_t
            ign = packed[i, 0] + (c_min % mean_win) + start_t
            mc = packed[i, 0] + (c_mean % mean_win) + start_t

            series["window_start"].append([ws, ws + event_dur_approx])
            series["ignition_centroid"].append([ign, ign + event_dur_approx])
            series["mean_centroid"].append([mc, mc + event_dur_approx])

    out = {}
    for k, v in series.items():
        if v:
            arr = np.array(v)
            out[k] = arr[arr[:, 0].argsort()]
        else:
            out[k] = np.zeros((0, 2))
    return out


def run_centroid_bypass(
    subject_dir: Path,
    dataset: str,
    block_ranges: List[Tuple[float, float]],
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
    n_surrogates: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Run Experiment 2: compare PSD for alternative within-window anchors.

    This tests whether peak frequency is sensitive to the choice of anchor inside
    the already-packed window. It does not fully remove the legacy packing /
    lagPatRaw dependency chain.
    """
    centroid_series = load_centroid_event_times(subject_dir, dataset)
    results = {}

    for method, events in centroid_series.items():
        r: Dict[str, Any] = {
            "method": method,
            "n_events": len(events),
            "peak_freq": None,
            "peak_power": None,
        }

        if len(events) < 10:
            results[method] = r
            continue

        pulse, mask, _ = build_pulse_train(events, fs=fs, mode="delta",
                                           block_ranges=block_ranges)
        if len(pulse) < int(fs * 10):
            results[method] = r
            continue

        psd = compute_event_psd(pulse, mask, fs=fs, nperseg_sec=nperseg_sec)
        sp = fit_psd_periodic(psd)
        if sp is not None:
            freq, power = _find_primary_peak(sp)
            if power > 0:
                r["peak_freq"] = float(freq)
                r["peak_power"] = float(power)

        iei = compute_iei(events, block_ranges=block_ranges)
        if len(iei) > 0:
            r["iei_min"] = float(np.min(iei))
            r["iei_mean"] = float(np.mean(iei))
            r["iei_median"] = float(np.median(iei))

        if sp is not None and n_surrogates > 0:
            sr = test_peak_significance_isi_shuffle(
                events, sp, n_surrogates=n_surrogates,
                fs=fs, nperseg_sec=nperseg_sec, block_ranges=block_ranges)
            if sr is not None:
                r["isi_shuffle_p"] = sr.p_value

        results[method] = r

    return results


# ---------------------------------------------------------------------------
# Phase 2: Hazard function (Experiment 3)
# ---------------------------------------------------------------------------

def compute_hazard_function(
    iei: np.ndarray,
    n_points: int = 500,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a smoothed hazard-like curve H(t) = f(t) / (1 - F(t)).

    Parameters
    ----------
    iei : 1-d array of inter-event intervals
    n_points : evaluation grid size
    bandwidth : KDE bandwidth in seconds; auto-selected if None

    Returns
    -------
    t : evaluation grid (seconds)
    hazard : H(t)
    pdf : f(t)

    Notes
    -----
    This implementation uses KDE smoothing, numerical CDF integration, and
    clipping for stability. It is suitable for qualitative visualization of
    dead-time / refractory structure, but not for formal survival-analysis
    inference or parameter comparison across groups.
    """
    from scipy.stats import gaussian_kde

    iei = iei[iei > 0]
    if len(iei) < 20:
        return np.array([]), np.array([]), np.array([])

    if bandwidth is None:
        bandwidth = 0.5 * np.std(iei) * len(iei) ** (-1.0 / 5.0)
        bandwidth = max(bandwidth, 0.01)

    kde = gaussian_kde(iei, bw_method=bandwidth / np.std(iei))

    t = np.linspace(0.0, np.percentile(iei, 99), n_points)
    pdf = kde(t)
    cdf = np.cumsum(pdf) * (t[1] - t[0])
    cdf = np.clip(cdf, 0.0, 1.0 - 1e-10)

    hazard = pdf / (1.0 - cdf)
    hazard = np.clip(hazard, 0.0, np.percentile(hazard[np.isfinite(hazard)], 99) * 3)

    return t, hazard, pdf


# ---------------------------------------------------------------------------
# Phase 2: Return map / Poincaré plot (Experiment 4)
# ---------------------------------------------------------------------------

def _log_iei_pairs(
    iei: np.ndarray,
    lag: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return log-IEI pairs separated by ``lag`` intervals."""
    iei = np.asarray(iei, dtype=float)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    iei = iei[np.isfinite(iei) & (iei > 0)]
    if iei.size <= lag:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.log(iei[:-lag])
    y = np.log(iei[lag:])
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Safe Pearson correlation helper."""
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan

    try:
        r, p = pearsonr(x, y)
    except Exception:
        return np.nan, np.nan
    return (float(r) if np.isfinite(r) else np.nan,
            float(p) if np.isfinite(p) else np.nan)


def _serial_corr_decay_from_sequences(
    iei_sequences: Sequence[np.ndarray],
    max_lag: int = 100,
    min_pairs: int = 50,
) -> Dict[str, Any]:
    """Pool lag-k serial correlation across multiple IEI sequences."""
    lags: List[int] = []
    rs: List[float] = []
    n_pairs_list: List[int] = []

    max_lag = max(1, int(max_lag))
    min_pairs = max(2, int(min_pairs))

    for lag in range(1, max_lag + 1):
        xs = []
        ys = []
        n_pairs = 0
        for seq in iei_sequences:
            x, y = _log_iei_pairs(seq, lag=lag)
            if x.size == 0:
                continue
            xs.append(x)
            ys.append(y)
            n_pairs += int(x.size)
        if n_pairs < min_pairs:
            break
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        r, _ = _pearson_corr(x_all, y_all)
        lags.append(lag)
        rs.append(float(r) if np.isfinite(r) else np.nan)
        n_pairs_list.append(n_pairs)

    half_life_lag = np.nan
    valid_rs = np.asarray(rs, dtype=float)
    if valid_rs.size > 0 and np.isfinite(valid_rs[0]) and valid_rs[0] > 0:
        target = 0.5 * valid_rs[0]
        prev_lag = float(lags[0])
        prev_r = float(valid_rs[0])
        if prev_r <= target:
            half_life_lag = prev_lag
        else:
            for lag, r in zip(lags[1:], valid_rs[1:]):
                if not np.isfinite(r):
                    prev_lag = float(lag)
                    continue
                if r <= target:
                    denom = prev_r - r
                    frac = 0.0 if abs(denom) < 1e-12 else (prev_r - target) / denom
                    frac = float(np.clip(frac, 0.0, 1.0))
                    half_life_lag = prev_lag + frac * (float(lag) - prev_lag)
                    break
                prev_lag = float(lag)
                prev_r = float(r)

    return {
        "lags": np.asarray(lags, dtype=int),
        "rs": np.asarray(rs, dtype=float),
        "n_pairs": np.asarray(n_pairs_list, dtype=int),
        "half_life_lag": half_life_lag,
    }


def _rolling_log_iei_residuals(
    event_times: np.ndarray,
    iei: np.ndarray,
    window_sec: float = 600.0,
    min_local_events: int = 5,
) -> Dict[str, Any]:
    """Estimate log-IEI residuals after local-median detrending."""
    event_times = np.asarray(event_times, dtype=float)
    iei = np.asarray(iei, dtype=float)
    iei = iei[np.isfinite(iei) & (iei > 0)]

    if iei.size < 3:
        return {
            "residual": np.array([], dtype=float),
            "valid_interval": np.zeros(0, dtype=bool),
            "valid_pair": np.zeros(0, dtype=bool),
            "local_baseline": np.array([], dtype=float),
            "local_counts": np.array([], dtype=int),
        }

    if event_times.size == iei.size + 1:
        centers = 0.5 * (event_times[:-1] + event_times[1:])
    elif event_times.size == iei.size:
        centers = event_times
    else:
        raise ValueError("event_times must have len(iei) or len(iei)+1")

    half_window = 0.5 * float(window_sec)
    min_local_events = max(3, int(min_local_events))

    baseline = np.full(iei.shape, np.nan, dtype=float)
    counts = np.zeros(iei.shape, dtype=int)

    order = np.argsort(centers)
    sorted_c = centers[order]
    sorted_iei = iei[order]

    for idx in range(len(sorted_c)):
        left = int(np.searchsorted(sorted_c, sorted_c[idx] - half_window, side="left"))
        right = int(np.searchsorted(sorted_c, sorted_c[idx] + half_window, side="right"))
        cnt = right - left
        orig_i = order[idx]
        counts[orig_i] = cnt
        if cnt >= min_local_events:
            baseline[orig_i] = float(np.median(sorted_iei[left:right]))

    valid_interval = np.isfinite(baseline) & (baseline > 0)
    residual = np.full(iei.shape, np.nan, dtype=float)
    residual[valid_interval] = np.log(iei[valid_interval]) - np.log(baseline[valid_interval])
    valid_pair = valid_interval[:-1] & valid_interval[1:]

    return {
        "residual": residual,
        "valid_interval": valid_interval,
        "valid_pair": valid_pair,
        "local_baseline": baseline,
        "local_counts": counts,
    }


def _split_events_by_block(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
) -> List[np.ndarray]:
    """Split an event series into block-contained sub-series."""
    events = np.asarray(events, dtype=float)
    if events.size == 0:
        return []

    starts = events[:, 0]
    blocks = []
    for block_start, block_end in sorted(block_ranges, key=lambda x: x[0]):
        mask = (starts >= block_start) & (starts < block_end)
        if np.any(mask):
            blocks.append(events[mask])
    return blocks


def compute_iei_return_map(
    iei: np.ndarray,
) -> Dict[str, Any]:
    """Compute IEI return-map statistics from consecutive intervals.

    Returns
    -------
    dict with keys:
      - iei_n, iei_n1: arrays for scatter
      - serial_corr: Pearson correlation of log(IEI[n]) and log(IEI[n+1])
      - serial_corr_p: naive Pearson p-value (descriptive only)

    Notes
    -----
    The correlation is computed on log-IEI because the raw IEI distribution is
    strongly skewed. The reported p-value should not be treated as a formal
    subject-level significance test because adjacent IEI pairs are not
    independent.
    """
    iei = np.asarray(iei, dtype=float)
    iei = iei[np.isfinite(iei) & (iei > 0)]
    if len(iei) < 10:
        return {"iei_n": np.array([]), "iei_n1": np.array([]),
                "serial_corr": np.nan, "serial_corr_p": np.nan}

    iei_n = iei[:-1]
    iei_n1 = iei[1:]
    r, p = _pearson_corr(np.log(iei_n), np.log(iei_n1))

    return {
        "iei_n": iei_n,
        "iei_n1": iei_n1,
        "serial_corr": float(r) if np.isfinite(r) else 0.0,
        "serial_corr_p": float(p) if np.isfinite(p) else 1.0,
    }


def compute_serial_correlation_decay(
    iei: np.ndarray,
    max_lag: int = 100,
    min_pairs: int = 50,
) -> Dict[str, Any]:
    """Pearson r(log IEI[n], log IEI[n+k]) for k = 1..max_lag."""
    out = _serial_corr_decay_from_sequences([iei], max_lag=max_lag, min_pairs=min_pairs)
    return {
        "lags": out["lags"],
        "rs": out["rs"],
        "n_pairs": out["n_pairs"],
        "half_life_lag": out["half_life_lag"],
        "lag1_r": float(out["rs"][0]) if len(out["rs"]) else np.nan,
    }


def compute_detrended_serial_correlation(
    event_times: np.ndarray,
    iei: np.ndarray,
    window_sec: float = 600.0,
) -> Dict[str, Any]:
    """Estimate lag-1 serial correlation after local-median detrending."""
    iei = np.asarray(iei, dtype=float)
    iei = iei[np.isfinite(iei) & (iei > 0)]
    raw_x, raw_y = _log_iei_pairs(iei, lag=1)
    raw_r, _ = _pearson_corr(raw_x, raw_y)

    rolled = _rolling_log_iei_residuals(event_times, iei, window_sec=window_sec)
    residual = rolled["residual"]
    valid_pair = rolled["valid_pair"]
    det_x = residual[:-1][valid_pair]
    det_y = residual[1:][valid_pair]
    detrended_r, _ = _pearson_corr(det_x, det_y)

    detrend_fraction = np.nan
    if np.isfinite(raw_r) and abs(raw_r) > 1e-12 and np.isfinite(detrended_r):
        detrend_fraction = 1.0 - (detrended_r / raw_r)

    local_counts = rolled["local_counts"]
    return {
        "raw_r": raw_r,
        "detrended_r": detrended_r,
        "detrend_fraction": detrend_fraction,
        "window_sec": float(window_sec),
        "min_local_events": 5,
        "n_intervals": int(iei.size),
        "n_valid_intervals": int(np.sum(rolled["valid_interval"])),
        "n_valid_pairs": int(det_x.size),
        "median_local_count": float(np.median(local_counts)) if local_counts.size else np.nan,
    }


def compute_within_block_serial_corr(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """Compute lag-1 serial correlation separately inside each valid block."""
    blocks = _split_events_by_block(events, block_ranges)
    per_block = []
    pooled_x = []
    pooled_y = []

    for idx, block_events in enumerate(blocks):
        block_start = float(block_events[0, 0]) if len(block_events) else np.nan
        block_end = float(block_events[-1, 1]) if len(block_events) else np.nan
        iei = compute_iei(block_events)
        x, y = _log_iei_pairs(iei, lag=1)
        r, _ = _pearson_corr(x, y)

        per_block.append({
            "block_idx": idx,
            "start_sec": block_start,
            "end_sec": block_end,
            "duration_sec": float(block_end - block_start) if np.isfinite(block_start) and np.isfinite(block_end) else np.nan,
            "n_events": int(block_events.shape[0]),
            "n_iei": int(iei.size),
            "serial_corr": r,
            "n_pairs": int(x.size),
        })
        if x.size:
            pooled_x.append(x)
            pooled_y.append(y)

    if pooled_x:
        pooled_r, _ = _pearson_corr(np.concatenate(pooled_x), np.concatenate(pooled_y))
        pooled_n_pairs = int(sum(arr.size for arr in pooled_x))
    else:
        pooled_r = np.nan
        pooled_n_pairs = 0

    block_rs = np.asarray(
        [blk["serial_corr"] for blk in per_block if np.isfinite(blk["serial_corr"])],
        dtype=float,
    )
    return {
        "n_blocks_total": len(block_ranges),
        "n_blocks_with_events": len(blocks),
        "n_blocks_used": int(np.sum([blk["n_pairs"] > 0 for blk in per_block])),
        "pooled_r": pooled_r,
        "pooled_n_pairs": pooled_n_pairs,
        "median_block_r": float(np.median(block_rs)) if block_rs.size else np.nan,
        "per_block": per_block,
    }


def compute_serial_corr_soz_stratified(
    subject_dir: Path,
    dataset: str,
    soz_channels: List[str],
    block_ranges: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """Build SOZ-only and nonSOZ-only packed event sequences and compute lag-1 r."""
    loaded = _load_group_events_with_soz_labels(subject_dir, dataset, soz_channels)
    events = loaded["events"]
    is_soz = loaded["is_soz"]

    def _summary(ev: np.ndarray) -> Dict[str, Any]:
        base = {
            "n_events": int(ev.shape[0]),
            "n_iei": int(max(ev.shape[0] - 1, 0)),
        }
        within = compute_within_block_serial_corr(ev, block_ranges)
        base.update(within)
        base["lag1_r"] = within["pooled_r"]
        if ev.shape[0] < 50:
            base["warning"] = "insufficient_events_lt_50"
        return base

    out: Dict[str, Any] = {
        "n_channels_total": int(loaded["n_channels_total"]),
        "n_soz_channels_input": int(loaded["n_soz_channels_input"]),
        "n_soz_channels_matched": int(loaded["n_soz_channels_matched"]),
        "all": _summary(events),
        "soz": _summary(events[is_soz]),
        "nonsoz": _summary(events[~is_soz]),
    }
    if "warning" in loaded:
        out["warning"] = loaded["warning"]
    return out


# ---------------------------------------------------------------------------
# Phase 2: Propagation stereotypy (Experiment 5)
# ---------------------------------------------------------------------------

def compute_propagation_stereotypy(
    subject_dir: Path,
    dataset: str,
    soz_channels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute exploratory channel-order consistency (Kendall tau) across events.

    Parameters
    ----------
    soz_channels : SOZ channel names; if provided, events are split into
        SOZ-participating and non-SOZ groups.

    Returns
    -------
    dict with keys: mean_tau, n_events, n_channels,
        soz_mean_tau, soz_n_events, nonsoz_mean_tau, nonsoz_n_events

    Notes
    -----
    The estimator sub-samples at most 200 events per subject to avoid O(n^2)
    blow-up. The resulting mean tau is therefore a stable summary statistic, not
    an exact full-population value, and should be treated as exploratory until
    rechecked with multi-seed or model-based analyses.
    """
    from scipy.stats import kendalltau

    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))

    all_ranks = []
    all_bools = []
    ch_names = None

    for lp_file in lagpat_files:
        lp = np.load(lp_file, allow_pickle=True)
        rank = lp["lagPatRank"]
        eb = lp["eventsBool"]
        if ch_names is None:
            ch_names = list(lp["chnNames"])
        if rank.size == 0:
            continue
        all_ranks.append(rank)
        all_bools.append(eb)

    if not all_ranks:
        return {"mean_tau": np.nan, "n_events": 0}

    ranks = np.concatenate(all_ranks, axis=1)
    bools = np.concatenate(all_bools, axis=1)

    n_ch, n_ev = ranks.shape
    min_participating = 3

    def _mean_pairwise_tau(event_indices):
        if len(event_indices) < 2:
            return np.nan, len(event_indices)
        taus = []
        pairs = min(len(event_indices), 200)
        rng = np.random.default_rng(42)
        sampled = rng.choice(event_indices, size=min(pairs, len(event_indices)), replace=False)
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                ei, ej = sampled[i], sampled[j]
                mask = (bools[:, ei] > 0) & (bools[:, ej] > 0)
                if np.sum(mask) < min_participating:
                    continue
                tau, _ = kendalltau(ranks[mask, ei], ranks[mask, ej])
                if np.isfinite(tau):
                    taus.append(tau)
        if not taus:
            return np.nan, len(event_indices)
        return float(np.mean(taus)), len(event_indices)

    all_idx = np.arange(n_ev)
    valid = np.sum(bools > 0, axis=0) >= min_participating
    all_valid = all_idx[valid]

    mean_tau, n_total = _mean_pairwise_tau(all_valid)
    result: Dict[str, Any] = {
        "mean_tau": mean_tau,
        "n_events": n_total,
        "n_channels": n_ch,
    }

    if soz_channels and ch_names:
        soz_set = set(soz_channels)
        soz_mask_ch = np.array([c in soz_set for c in ch_names])

        soz_events = []
        nonsoz_events = []
        for i in all_valid:
            participating = bools[:, i] > 0
            if np.any(participating & soz_mask_ch):
                soz_events.append(i)
            else:
                nonsoz_events.append(i)

        soz_tau, soz_n = _mean_pairwise_tau(np.array(soz_events))
        nonsoz_tau, nonsoz_n = _mean_pairwise_tau(np.array(nonsoz_events))

        result["soz_mean_tau"] = soz_tau
        result["soz_n_events"] = soz_n
        result["nonsoz_mean_tau"] = nonsoz_tau
        result["nonsoz_n_events"] = nonsoz_n

    return result


# ---------------------------------------------------------------------------
# PR-2.5: Multi-scale modulation anatomy
# ---------------------------------------------------------------------------

def compute_multiscale_detrend_fraction(
    event_times: np.ndarray,
    iei: np.ndarray,
    windows: Sequence[float] = (60, 180, 600, 1800, 3600, 7200),
) -> Dict[str, Any]:
    """Detrend fraction at multiple window sizes → modulation anatomy.

    Smaller windows track the series more aggressively (removing both slow
    and fast structure), while larger windows only remove slow modulation.
    Use ``delta_frac`` (the difference between adjacent scales) to identify
    which timescale concentrates the most modulation energy.
    """
    iei = np.asarray(iei, dtype=float)
    iei = iei[np.isfinite(iei) & (iei > 0)]
    event_times = np.asarray(event_times, dtype=float)

    raw_x, raw_y = _log_iei_pairs(iei, lag=1)
    raw_r, _ = _pearson_corr(raw_x, raw_y)

    windows = sorted(windows)
    per_window: List[Dict[str, Any]] = []

    for win_sec in windows:
        rolled = _rolling_log_iei_residuals(event_times, iei, window_sec=win_sec)
        resid = rolled["residual"]
        valid_pair = rolled["valid_pair"]

        if np.any(valid_pair):
            det_r, _ = _pearson_corr(resid[:-1][valid_pair], resid[1:][valid_pair])
        else:
            det_r = np.nan

        frac = np.nan
        if np.isfinite(raw_r) and abs(raw_r) > 1e-12 and np.isfinite(det_r):
            frac = 1.0 - (det_r / raw_r)

        per_window.append({
            "window_sec": float(win_sec),
            "detrended_r": float(det_r) if np.isfinite(det_r) else np.nan,
            "detrend_fraction": float(frac) if np.isfinite(frac) else np.nan,
            "n_valid_pairs": int(np.sum(valid_pair)),
        })

    delta_frac: List[Dict[str, Any]] = []
    fracs = [pw["detrend_fraction"] for pw in per_window]
    for i in range(len(windows) - 1):
        f_small = fracs[i]
        f_large = fracs[i + 1]
        midpoint = float(np.sqrt(windows[i] * windows[i + 1]))

        df = np.nan
        if np.isfinite(f_small) and np.isfinite(f_large):
            df = float(f_small - f_large)

        delta_frac.append({
            "midpoint_sec": midpoint,
            "window_lo": float(windows[i]),
            "window_hi": float(windows[i + 1]),
            "delta_frac": df,
        })

    return {
        "raw_r": float(raw_r) if np.isfinite(raw_r) else np.nan,
        "windows": [float(w) for w in windows],
        "per_window": per_window,
        "delta_frac": delta_frac,
    }


def _compute_half_life(lags: Sequence[int], rs: np.ndarray) -> float:
    """Interpolate lag at which r drops to half of r[0]."""
    if rs.size == 0 or not np.isfinite(rs[0]) or rs[0] <= 0:
        return np.nan
    target = 0.5 * rs[0]
    prev_lag, prev_r = float(lags[0]), float(rs[0])
    if prev_r <= target:
        return prev_lag
    for lag, r in zip(list(lags)[1:], rs[1:]):
        if not np.isfinite(r):
            prev_lag = float(lag)
            continue
        if r <= target:
            denom = prev_r - r
            frac = 0.0 if abs(denom) < 1e-12 else (prev_r - target) / denom
            frac = float(np.clip(frac, 0.0, 1.0))
            return prev_lag + frac * (float(lag) - prev_lag)
        prev_lag, prev_r = float(lag), float(r)
    return np.nan


def compute_nparticipating_autocorrelation(
    subject_dir: Path,
    dataset: str,
    block_ranges: List[Tuple[float, float]],
    max_lag: int = 100,
    min_pairs: int = 50,
) -> Dict[str, Any]:
    """Lag-k Spearman autocorrelation of n_participating (integer counts).

    Returns decay curve and cross-correlation with the IEI Pearson decay.
    Spearman is used because n_participating is a small discrete integer.
    """
    from scipy.stats import spearmanr

    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))

    all_n_part: List[np.ndarray] = []
    all_starts: List[np.ndarray] = []

    for lp_file in lagpat_files:
        lp = np.load(lp_file, allow_pickle=True)
        events_bool = lp["eventsBool"]
        stem = lp_file.stem.replace("_lagPat", "")
        packed_f = lp_file.with_name(stem + "_packedTimes.npy")
        if not packed_f.exists():
            continue

        if dataset == "yuquan":
            gpu_f = lp_file.with_name(stem + "_gpu.npz")
            if not gpu_f.exists():
                continue
            gpu = _try_load_gpu(gpu_f)
            if gpu is None:
                continue
            start_t = float(gpu["start_time"])
        else:
            start_t = float(np.ravel(lp["start_t"])[0])

        packed = np.load(packed_f)
        if events_bool.size == 0 or packed.size == 0:
            continue
        n_ev = min(events_bool.shape[1], packed.shape[0])
        n_part = events_bool[:, :n_ev].sum(axis=0).astype(int)
        starts = packed[:n_ev, 0] + start_t

        all_n_part.append(n_part)
        all_starts.append(starts)

    if not all_n_part:
        return {"warning": "no_data", "n_events": 0}

    n_part_all = np.concatenate(all_n_part)
    starts_all = np.concatenate(all_starts)
    order = np.argsort(starts_all)
    n_part_all = n_part_all[order]
    starts_all = starts_all[order]

    block_seqs: List[np.ndarray] = []
    for b_start, b_end in sorted(block_ranges, key=lambda x: x[0]):
        mask = (starts_all >= b_start) & (starts_all < b_end)
        if np.sum(mask) >= 3:
            block_seqs.append(n_part_all[mask])

    lags_out: List[int] = []
    rs_out: List[float] = []
    n_pairs_out: List[int] = []

    for lag in range(1, max_lag + 1):
        xs, ys = [], []
        n_pairs = 0
        for seq in block_seqs:
            if seq.size <= lag:
                continue
            xs.append(seq[:-lag])
            ys.append(seq[lag:])
            n_pairs += seq.size - lag
        if n_pairs < min_pairs:
            break
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        try:
            rho, _ = spearmanr(x_all, y_all)
            rho = float(rho) if np.isfinite(rho) else np.nan
        except Exception:
            rho = np.nan
        lags_out.append(lag)
        rs_out.append(rho)
        n_pairs_out.append(n_pairs)

    rs_arr = np.asarray(rs_out, dtype=float)
    half_life = _compute_half_life(lags_out, rs_arr)

    return {
        "n_events": int(n_part_all.size),
        "n_blocks_used": len(block_seqs),
        "n_part_median": float(np.median(n_part_all)),
        "n_part_mean": float(np.mean(n_part_all)),
        "n_part_min": int(np.min(n_part_all)),
        "n_part_max": int(np.max(n_part_all)),
        "lags": np.asarray(lags_out, dtype=int),
        "rs": rs_arr,
        "n_pairs": np.asarray(n_pairs_out, dtype=int),
        "half_life_lag": half_life,
        "lag1_r": float(rs_out[0]) if rs_out else np.nan,
    }


def merge_contiguous_blocks(
    block_ranges: List[Tuple[float, float]],
    max_gap_sec: float = 5.0,
) -> List[Tuple[float, float]]:
    """Merge adjacent blocks where gap in [0, max_gap_sec]."""
    if not block_ranges:
        return []
    sorted_blocks = sorted(block_ranges, key=lambda x: x[0])
    merged = [list(sorted_blocks[0])]
    for b_start, b_end in sorted_blocks[1:]:
        gap = b_start - merged[-1][1]
        if 0 <= gap <= max_gap_sec:
            merged[-1][1] = max(merged[-1][1], b_end)
        elif gap < 0:
            merged[-1][1] = max(merged[-1][1], b_end)
        else:
            merged.append([b_start, b_end])
    return [(s, e) for s, e in merged]


def _epoch_to_hour(epoch_sec: float, timezone: str) -> float:
    """Convert Unix epoch to fractional hour of local day."""
    import datetime
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone)
    except ImportError:
        import pytz
        tz = pytz.timezone(timezone)
    dt = datetime.datetime.fromtimestamp(epoch_sec, tz=tz)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def compute_daynight_stratified_detrending(
    event_times: np.ndarray,
    iei: np.ndarray,
    dataset: str,
    window_sec: float = 600.0,
    day_hours: Tuple[float, float] = (8.0, 20.0),
) -> Dict[str, Any]:
    """Detrend IEI within day and night segments separately.

    Yuquan uses Asia/Shanghai; Epilepsiae uses Europe/Berlin.
    """
    iei = np.asarray(iei, dtype=float)
    mask_pos = np.isfinite(iei) & (iei > 0)
    iei = iei[mask_pos]
    event_times = np.asarray(event_times, dtype=float)
    if event_times.size == iei.size + 1:
        centers = 0.5 * (event_times[:-1] + event_times[1:])
    elif event_times.size == iei.size:
        centers = event_times.copy()
    else:
        return {"warning": "event_times_iei_length_mismatch"}
    centers = centers[mask_pos] if mask_pos.size == centers.size else centers

    tz = "Asia/Shanghai" if dataset == "yuquan" else "Europe/Berlin"
    hours = np.array([_epoch_to_hour(t, tz) for t in centers])
    is_day = (hours >= day_hours[0]) & (hours < day_hours[1])

    def _detrend_segment(idx: np.ndarray) -> Dict[str, Any]:
        if idx.size < 20:
            return {"n_iei": int(idx.size), "raw_r": np.nan,
                    "detrended_r": np.nan, "detrend_fraction": np.nan}
        seg_iei = iei[idx]
        seg_times = centers[idx]
        raw_x, raw_y = _log_iei_pairs(seg_iei, lag=1)
        raw_r, _ = _pearson_corr(raw_x, raw_y)

        rolled = _rolling_log_iei_residuals(seg_times, seg_iei, window_sec=window_sec)
        resid = rolled["residual"]
        vp = rolled["valid_pair"]
        if np.any(vp):
            det_r, _ = _pearson_corr(resid[:-1][vp], resid[1:][vp])
        else:
            det_r = np.nan

        frac = np.nan
        if np.isfinite(raw_r) and abs(raw_r) > 1e-12 and np.isfinite(det_r):
            frac = 1.0 - (det_r / raw_r)
        return {
            "n_iei": int(idx.size),
            "raw_r": float(raw_r) if np.isfinite(raw_r) else np.nan,
            "detrended_r": float(det_r) if np.isfinite(det_r) else np.nan,
            "detrend_fraction": float(frac) if np.isfinite(frac) else np.nan,
        }

    day_idx = np.where(is_day)[0]
    night_idx = np.where(~is_day)[0]

    return {
        "timezone": tz,
        "day_hours": list(day_hours),
        "n_day": int(day_idx.size),
        "n_night": int(night_idx.size),
        "day": _detrend_segment(day_idx),
        "night": _detrend_segment(night_idx),
        "combined": _detrend_segment(np.arange(iei.size)),
    }


# ---------------------------------------------------------------------------
# PR-2.6: Continuous long-timescale analysis
# ---------------------------------------------------------------------------

def _dataset_timezone(dataset: str) -> str:
    """Canonical timezone used for local-clock summaries."""
    return "Asia/Shanghai" if dataset == "yuquan" else "Europe/Berlin"


def _hour_to_hms(hour_float: float) -> Tuple[int, int, int]:
    """Convert fractional hour to (hour, minute, second)."""
    total_seconds = int(round(float(hour_float) * 3600.0))
    total_seconds %= 24 * 3600
    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    second = total_seconds % 60
    return hour, minute, second


def _contiguous_true_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return contiguous [start, end) spans where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _segmented_moving_average(
    values: np.ndarray,
    valid_mask: np.ndarray,
    window_bins: int,
) -> np.ndarray:
    """Centered moving average applied independently to each valid span."""
    values = np.asarray(values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    window_bins = max(1, int(window_bins))

    out = np.full(values.shape, np.nan, dtype=float)
    kernel = np.ones(window_bins, dtype=float)

    for start, end in _contiguous_true_spans(valid_mask):
        seg = values[start:end]
        if seg.size == 0:
            continue
        if seg.size <= window_bins:
            out[start:end] = float(np.mean(seg))
            continue
        numer = np.convolve(seg, kernel, mode="same")
        denom = np.convolve(np.ones(seg.size, dtype=float), kernel, mode="same")
        out[start:end] = numer / np.maximum(denom, 1.0)
    return out


def _segmented_autocorr(
    values: np.ndarray,
    valid_mask: np.ndarray,
    lag_bins: int,
    min_pairs: int = 10,
) -> Dict[str, Any]:
    """Pearson autocorrelation at a fixed lag across contiguous valid spans."""
    values = np.asarray(values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    lag_bins = max(1, int(lag_bins))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for start, end in _contiguous_true_spans(valid_mask):
        seg = values[start:end]
        if seg.size <= lag_bins:
            continue
        xs.append(seg[:-lag_bins])
        ys.append(seg[lag_bins:])

    if not xs:
        return {"r": np.nan, "n_pairs": 0}

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    if x_all.size < min_pairs:
        return {"r": np.nan, "n_pairs": int(x_all.size)}

    r, _ = _pearson_corr(x_all, y_all)
    return {"r": r, "n_pairs": int(x_all.size)}


def summarize_block_continuity(
    block_ranges: List[Tuple[float, float]],
    merge_gap_sec: float = 5.0,
) -> Dict[str, Any]:
    """Summarize observation continuity after merging tiny inter-block gaps."""
    merged = merge_contiguous_blocks(block_ranges, max_gap_sec=merge_gap_sec)
    if not merged:
        return {
            "n_blocks_original": 0,
            "n_runs_merged": 0,
            "total_observed_hours": 0.0,
            "total_span_hours": 0.0,
            "coverage_fraction": np.nan,
            "longest_run_hours": 0.0,
            "median_run_hours": 0.0,
            "merged_runs": [],
        }

    durations = np.asarray([end - start for start, end in merged], dtype=float)
    total_observed = float(np.sum(durations))
    total_span = float(merged[-1][1] - merged[0][0])
    return {
        "n_blocks_original": int(len(block_ranges)),
        "n_runs_merged": int(len(merged)),
        "total_observed_hours": total_observed / 3600.0,
        "total_span_hours": total_span / 3600.0,
        "coverage_fraction": float(total_observed / total_span) if total_span > 0 else np.nan,
        "longest_run_hours": float(np.max(durations) / 3600.0),
        "median_run_hours": float(np.median(durations) / 3600.0),
        "merged_runs": [[float(s), float(e)] for s, e in merged],
    }


def build_continuous_rate_trace(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    bin_sec: float = 300.0,
    merge_gap_sec: float = 5.0,
    min_coverage_frac: float = 0.95,
) -> Dict[str, Any]:
    """Build a continuous-time rate trace on the real observation axis.

    The trace is binned on an absolute clock. Bins with incomplete observation
    coverage are retained in ``coverage_frac`` but excluded from long-timescale
    summaries via ``valid_mask``.
    """
    events = np.asarray(events, dtype=float)
    if events.size == 0 or not block_ranges:
        return {"warning": "no_events_or_blocks"}

    continuity = summarize_block_continuity(block_ranges, merge_gap_sec=merge_gap_sec)
    merged = [(float(s), float(e)) for s, e in continuity["merged_runs"]]
    if not merged:
        return {"warning": "no_merged_runs"}

    bin_sec = float(bin_sec)
    t0 = merged[0][0]
    t1 = merged[-1][1]
    edges = np.arange(t0, t1 + bin_sec, bin_sec, dtype=float)
    if edges[-1] < t1:
        edges = np.r_[edges, edges[-1] + bin_sec]
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = centers.size

    observed_sec = np.zeros(n_bins, dtype=float)
    for run_start, run_end in merged:
        i0 = max(0, int(np.floor((run_start - t0) / bin_sec)))
        i1 = min(n_bins - 1, int(np.floor((run_end - t0 - 1e-9) / bin_sec)))
        for idx in range(i0, i1 + 1):
            overlap = max(0.0, min(edges[idx + 1], run_end) - max(edges[idx], run_start))
            observed_sec[idx] += overlap

    starts = events[:, 0]
    counts = np.histogram(starts, bins=edges)[0].astype(float)
    rate_per_hour = np.full(n_bins, np.nan, dtype=float)
    valid_obs = observed_sec > 0
    rate_per_hour[valid_obs] = counts[valid_obs] * 3600.0 / observed_sec[valid_obs]
    coverage_frac = observed_sec / bin_sec
    valid_mask = np.isfinite(rate_per_hour) & (coverage_frac >= float(min_coverage_frac))

    return {
        "bin_sec": bin_sec,
        "time_start": float(t0),
        "time_end": float(t1),
        "bin_edges": edges,
        "bin_centers": centers,
        "count": counts,
        "observed_sec": observed_sec,
        "coverage_frac": coverage_frac,
        "rate_per_hour": rate_per_hour,
        "valid_mask": valid_mask,
        "continuity": continuity,
    }


def compute_long_timescale_rate_summary(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    bin_sec: float = 300.0,
    smooth_windows_sec: Sequence[float] = (1800, 3600, 7200, 14400, 28800),
    autocorr_lag_hours: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
    merge_gap_sec: float = 5.0,
    min_coverage_frac: float = 0.95,
) -> Dict[str, Any]:
    """Summarize long-timescale modulation on the real continuous time axis."""
    trace = build_continuous_rate_trace(
        events=events,
        block_ranges=block_ranges,
        bin_sec=bin_sec,
        merge_gap_sec=merge_gap_sec,
        min_coverage_frac=min_coverage_frac,
    )
    if "warning" in trace:
        return trace

    rate = np.asarray(trace["rate_per_hour"], dtype=float)
    valid = np.asarray(trace["valid_mask"], dtype=bool)
    rate_summary: List[Dict[str, Any]] = []
    smooth_traces: Dict[str, List[float]] = {}

    for window_sec in smooth_windows_sec:
        window_bins = max(1, int(round(float(window_sec) / float(bin_sec))))
        smooth = _segmented_moving_average(rate, valid, window_bins)
        smooth_valid = np.isfinite(smooth) & valid

        fluct_strength = np.nan
        if np.sum(smooth_valid) >= 5:
            smooth_vals = smooth[smooth_valid]
            med = float(np.median(smooth_vals))
            if med > 0:
                q75, q25 = np.percentile(smooth_vals, [75, 25])
                fluct_strength = float((q75 - q25) / med)

        rate_summary.append({
            "window_sec": float(window_sec),
            "window_hours": float(window_sec) / 3600.0,
            "window_bins": int(window_bins),
            "n_valid_bins": int(np.sum(smooth_valid)),
            "fluct_strength_iqr_over_median": fluct_strength,
        })
        smooth_traces[f"smooth_{int(round(window_sec))}s"] = smooth.tolist()

    rate_autocorr: List[Dict[str, Any]] = []
    for lag_h in autocorr_lag_hours:
        lag_bins = max(1, int(round(float(lag_h) * 3600.0 / float(bin_sec))))
        ac = _segmented_autocorr(rate, valid, lag_bins)
        rate_autocorr.append({
            "lag_hours": float(lag_h),
            "lag_bins": int(lag_bins),
            "r": ac["r"],
            "n_pairs": int(ac["n_pairs"]),
        })

    trace["continuity"]["is_near_24h_continuous"] = bool(
        trace["continuity"]["longest_run_hours"] >= 22.0
    )

    return {
        "bin_sec": float(bin_sec),
        "continuity": trace["continuity"],
        "trace": {
            "bin_centers": np.asarray(trace["bin_centers"], dtype=float),
            "bin_hours_from_start": (
                (np.asarray(trace["bin_centers"], dtype=float) - float(trace["time_start"])) / 3600.0
            ),
            "rate_per_hour": rate,
            "coverage_frac": np.asarray(trace["coverage_frac"], dtype=float),
            "valid_mask": valid,
            **smooth_traces,
        },
        "rate_summary": rate_summary,
        "rate_autocorr": rate_autocorr,
    }


def _next_local_transition(
    epoch_sec: float,
    timezone: str,
    day_hours: Tuple[float, float],
) -> float:
    """Return next local day/night transition after epoch_sec."""
    import datetime
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone)
    except ImportError:
        import pytz
        tz = pytz.timezone(timezone)

    dt = datetime.datetime.fromtimestamp(epoch_sec, tz=tz)
    h_day_start, m_day_start, s_day_start = _hour_to_hms(day_hours[0])
    h_day_end, m_day_end, s_day_end = _hour_to_hms(day_hours[1])
    day_start = dt.replace(hour=h_day_start, minute=m_day_start,
                           second=s_day_start, microsecond=0)
    day_end = dt.replace(hour=h_day_end, minute=m_day_end,
                         second=s_day_end, microsecond=0)

    if dt < day_start:
        next_dt = day_start
    elif dt < day_end:
        next_dt = day_end
    else:
        next_dt = day_start + datetime.timedelta(days=1)
    return float(next_dt.timestamp())


def split_contiguous_daynight_segments(
    block_ranges: List[Tuple[float, float]],
    dataset: str,
    day_hours: Tuple[float, float] = (8.0, 20.0),
    merge_gap_sec: float = 5.0,
) -> List[Dict[str, Any]]:
    """Split continuous observed runs into continuous day/night segments."""
    timezone = _dataset_timezone(dataset)
    merged = merge_contiguous_blocks(block_ranges, max_gap_sec=merge_gap_sec)
    segments: List[Dict[str, Any]] = []

    for run_idx, (run_start, run_end) in enumerate(merged):
        cursor = float(run_start)
        while cursor < run_end - 1e-9:
            local_hour = _epoch_to_hour(cursor, timezone)
            label = "day" if day_hours[0] <= local_hour < day_hours[1] else "night"
            boundary = _next_local_transition(cursor + 1e-6, timezone, day_hours)
            seg_end = min(float(run_end), boundary)
            segments.append({
                "run_idx": int(run_idx),
                "label": label,
                "start_sec": float(cursor),
                "end_sec": float(seg_end),
                "duration_hours": float((seg_end - cursor) / 3600.0),
            })
            cursor = seg_end

    return segments


def compute_contiguous_daynight_detrending(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    dataset: str,
    window_sec: float = 600.0,
    day_hours: Tuple[float, float] = (8.0, 20.0),
    merge_gap_sec: float = 5.0,
    min_segment_iei: int = 20,
) -> Dict[str, Any]:
    """Compute serial correlation inside continuous day/night segments."""
    events = np.asarray(events, dtype=float)
    segments = split_contiguous_daynight_segments(
        block_ranges=block_ranges,
        dataset=dataset,
        day_hours=day_hours,
        merge_gap_sec=merge_gap_sec,
    )

    if events.size == 0 or not segments:
        return {"warning": "no_events_or_segments", "segments": []}

    starts = events[:, 0]
    segment_records: List[Dict[str, Any]] = []
    pooled: Dict[str, Dict[str, List[np.ndarray]]] = {
        "day": {"raw_x": [], "raw_y": [], "det_x": [], "det_y": []},
        "night": {"raw_x": [], "raw_y": [], "det_x": [], "det_y": []},
    }

    for seg in segments:
        mask = (starts >= seg["start_sec"]) & (starts < seg["end_sec"])
        seg_events = events[mask]
        n_events = int(seg_events.shape[0])
        if n_events < 2:
            segment_records.append({**seg, "n_events": n_events, "n_iei": 0})
            continue

        seg_iei = compute_iei(seg_events)
        n_iei = int(seg_iei.size)
        if n_iei < min_segment_iei:
            segment_records.append({
                **seg,
                "n_events": n_events,
                "n_iei": n_iei,
                "raw_r": np.nan,
                "detrended_r": np.nan,
                "detrend_fraction": np.nan,
            })
            continue

        seg_starts = seg_events[:, 0]
        det = compute_detrended_serial_correlation(
            event_times=seg_starts,
            iei=seg_iei,
            window_sec=window_sec,
        )
        record = {
            **seg,
            "n_events": n_events,
            "n_iei": n_iei,
            "raw_r": det["raw_r"],
            "detrended_r": det["detrended_r"],
            "detrend_fraction": det["detrend_fraction"],
        }
        segment_records.append(record)

        label = seg["label"]
        raw_x, raw_y = _log_iei_pairs(seg_iei, lag=1)
        if raw_x.size:
            pooled[label]["raw_x"].append(raw_x)
            pooled[label]["raw_y"].append(raw_y)

        rolled = _rolling_log_iei_residuals(seg_starts, seg_iei, window_sec=window_sec)
        resid = rolled["residual"]
        valid_pair = rolled["valid_pair"]
        if resid.size and np.any(valid_pair):
            pooled[label]["det_x"].append(resid[:-1][valid_pair])
            pooled[label]["det_y"].append(resid[1:][valid_pair])

    def _label_summary(label: str) -> Dict[str, Any]:
        label_segments = [rec for rec in segment_records if rec["label"] == label]
        used_segments = [rec for rec in label_segments if np.isfinite(rec.get("detrended_r", np.nan))]

        raw_r, _ = _pearson_corr(
            np.concatenate(pooled[label]["raw_x"]) if pooled[label]["raw_x"] else np.array([]),
            np.concatenate(pooled[label]["raw_y"]) if pooled[label]["raw_y"] else np.array([]),
        )
        det_r, _ = _pearson_corr(
            np.concatenate(pooled[label]["det_x"]) if pooled[label]["det_x"] else np.array([]),
            np.concatenate(pooled[label]["det_y"]) if pooled[label]["det_y"] else np.array([]),
        )
        detrend_fraction = np.nan
        if np.isfinite(raw_r) and abs(raw_r) > 1e-12 and np.isfinite(det_r):
            detrend_fraction = 1.0 - (det_r / raw_r)

        detrended_vals = np.asarray(
            [rec["detrended_r"] for rec in used_segments if np.isfinite(rec["detrended_r"])],
            dtype=float,
        )
        return {
            "n_segments_total": int(len(label_segments)),
            "n_segments_used": int(len(used_segments)),
            "total_hours": float(np.sum([rec["duration_hours"] for rec in label_segments])),
            "total_iei": int(np.sum([rec.get("n_iei", 0) for rec in used_segments])),
            "pooled_raw_r": raw_r,
            "pooled_detrended_r": det_r,
            "pooled_detrend_fraction": detrend_fraction,
            "median_segment_detrended_r": (
                float(np.median(detrended_vals)) if detrended_vals.size else np.nan
            ),
        }

    return {
        "timezone": _dataset_timezone(dataset),
        "day_hours": list(day_hours),
        "merge_gap_sec": float(merge_gap_sec),
        "window_sec": float(window_sec),
        "segments": segment_records,
        "day": _label_summary("day"),
        "night": _label_summary("night"),
    }


def compute_detrended_psd_backfill(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    window_sec: float = 600.0,
    fs: float = 100.0,
    nperseg_sec: float = 500.0,
    n_surrogates: int = 200,
) -> Dict[str, Any]:
    """Detrend IEI by removing slow rate modulation, rebuild pulse train, recompute PSD.

    For the PR-1 escape subjects (1084, 1096): if the peak disappears after
    detrending, the peak was caused by slow rate drift rather than an oscillator.
    """
    if len(events) < 10:
        return {"warning": "insufficient_events"}

    iei = compute_iei(events, block_ranges=block_ranges)
    if iei.size < 10:
        return {"warning": "insufficient_iei"}

    starts = events[:, 0]
    if starts.size == iei.size + 1:
        ev_times = starts
    else:
        ev_times = starts[:iei.size]

    rolled = _rolling_log_iei_residuals(ev_times, iei, window_sec=window_sec)
    baseline = rolled["local_baseline"]
    valid = rolled["valid_interval"]

    global_median = float(np.median(iei[iei > 0]))
    detrended_iei = iei.copy()
    ok = valid & np.isfinite(baseline) & (baseline > 0)
    detrended_iei[ok] = iei[ok] / baseline[ok] * global_median

    new_starts = np.empty(len(detrended_iei) + 1)
    new_starts[0] = events[0, 0]
    new_starts[1:] = new_starts[0] + np.cumsum(detrended_iei)
    new_events = np.column_stack([new_starts, new_starts + 0.01])

    pulse_raw, mask_raw, _ = build_pulse_train(
        events, fs=fs, mode="delta", block_ranges=block_ranges)
    psd_raw = compute_event_psd(pulse_raw, mask_raw, fs=fs, nperseg_sec=nperseg_sec)
    sp_raw = fit_psd_periodic(psd_raw)

    pulse_det, mask_det, _ = build_pulse_train(
        new_events, fs=fs, mode="delta", block_ranges=block_ranges)
    psd_det = compute_event_psd(pulse_det, mask_det, fs=fs, nperseg_sec=nperseg_sec)
    sp_det = fit_psd_periodic(psd_det)

    raw_peak_freq, raw_peak_power = (0.0, 0.0)
    if sp_raw is not None:
        raw_peak_freq, raw_peak_power = _find_primary_peak(sp_raw)

    det_peak_freq, det_peak_power = (0.0, 0.0)
    if sp_det is not None:
        det_peak_freq, det_peak_power = _find_primary_peak(sp_det)

    gamma_p_raw = np.nan
    gamma_p_det = np.nan
    if sp_raw is not None and n_surrogates > 0:
        sr = test_peak_significance_gamma_renewal(
            events, sp_raw, n_surrogates=n_surrogates,
            fs=fs, nperseg_sec=nperseg_sec, block_ranges=block_ranges)
        if sr is not None:
            gamma_p_raw = sr.p_value
    if sp_det is not None and n_surrogates > 0 and det_peak_power > 0:
        sr = test_peak_significance_gamma_renewal(
            new_events, sp_det, n_surrogates=n_surrogates,
            fs=fs, nperseg_sec=nperseg_sec, block_ranges=block_ranges)
        if sr is not None:
            gamma_p_det = sr.p_value

    return {
        "n_events": int(events.shape[0]),
        "n_iei": int(iei.size),
        "window_sec": float(window_sec),
        "raw": {
            "peak_freq": float(raw_peak_freq),
            "peak_power": float(raw_peak_power),
            "gamma_p": float(gamma_p_raw),
            "freqs": psd_raw.freqs,
            "power": psd_raw.power,
        },
        "detrended": {
            "peak_freq": float(det_peak_freq),
            "peak_power": float(det_peak_power),
            "gamma_p": float(gamma_p_det),
            "freqs": psd_det.freqs,
            "power": psd_det.power,
        },
        "peak_disappeared": bool(det_peak_power <= 0 and raw_peak_power > 0),
        "peak_insignificant": bool(gamma_p_det >= 0.05 if np.isfinite(gamma_p_det) else False),
    }


# ---------------------------------------------------------------------------
# PR-2.7: Rate-trace spectral characterization + seizure proximity
# ---------------------------------------------------------------------------


def compute_rate_trace_psd(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    bin_sec: float = 300.0,
    merge_gap_sec: float = 5.0,
    min_span_bins: int = 48,
    nperseg_max: int = 64,
    fit_freq_range_mhz: Tuple[float, float] = (0.02, 0.5),
) -> Dict[str, Any]:
    """Welch PSD of the continuous rate trace with OLS 1/f slope fit.

    Only contiguous valid spans (>= min_span_bins) are used.  PSDs from
    multiple spans are averaged weighted by span length.  The 1/f slope β
    is estimated via OLS on log10(PSD) vs log10(f) within fit_freq_range_mhz.
    """
    from scipy.signal import welch as scipy_welch

    trace = build_continuous_rate_trace(
        events=events, block_ranges=block_ranges,
        bin_sec=bin_sec, merge_gap_sec=merge_gap_sec,
    )
    if "warning" in trace:
        return {"warning": trace["warning"]}

    rate = np.asarray(trace["rate_per_hour"], dtype=float)
    valid = np.asarray(trace["valid_mask"], dtype=bool)
    spans = _contiguous_true_spans(valid)

    usable = [(s, e) for s, e in spans if (e - s) >= min_span_bins]
    if not usable:
        return {"warning": "no_usable_spans"}

    span_lengths = [e - s for s, e in usable]
    min_span_len = min(span_lengths)
    common_nperseg = min(int(nperseg_max), max(8, int(min_span_len // 2)))
    if common_nperseg < 8:
        return {"warning": "insufficient_span_length_for_psd"}
    usable = [(s, e) for s, e in usable if (e - s) >= common_nperseg]
    if not usable:
        return {"warning": "no_usable_spans_after_common_nperseg"}

    fs_hz = 1.0 / float(bin_sec)
    all_psd: List[np.ndarray] = []
    all_weights: List[float] = []
    common_freqs: Optional[np.ndarray] = None

    for s, e in usable:
        seg = rate[s:e].copy()
        seg -= np.nanmean(seg)
        freqs, pxx = scipy_welch(
            seg,
            fs=fs_hz,
            nperseg=common_nperseg,
            noverlap=common_nperseg // 2,
            detrend="linear",
        )
        if common_freqs is None:
            common_freqs = freqs
        if len(freqs) == len(common_freqs) and np.allclose(freqs, common_freqs):
            all_psd.append(pxx)
            all_weights.append(float(e - s))

    if not all_psd or common_freqs is None:
        return {"warning": "psd_computation_failed"}

    weights = np.array(all_weights, dtype=float)
    weights /= weights.sum()
    avg_psd = np.zeros_like(all_psd[0])
    for w, p in zip(weights, all_psd):
        avg_psd += w * p

    freqs_mhz = common_freqs * 1000.0

    f_lo, f_hi = fit_freq_range_mhz
    fit_mask = (freqs_mhz >= f_lo) & (freqs_mhz <= f_hi) & (avg_psd > 0) & (common_freqs > 0)

    beta = np.nan
    beta_r2 = np.nan
    n_fit_pts = int(np.sum(fit_mask))
    if n_fit_pts >= 3:
        log_f = np.log10(freqs_mhz[fit_mask])
        log_p = np.log10(avg_psd[fit_mask])
        coeffs = np.polyfit(log_f, log_p, 1)
        beta = -float(coeffs[0])
        predicted = np.polyval(coeffs, log_f)
        ss_res = np.sum((log_p - predicted) ** 2)
        ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
        beta_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    total_hours = sum((e - s) * bin_sec / 3600.0 for s, e in usable)

    return {
        "freqs_mhz": freqs_mhz,
        "psd": avg_psd,
        "beta": beta,
        "beta_r2": beta_r2,
        "n_fit_points": n_fit_pts,
        "n_spans": len(usable),
        "nperseg": int(common_nperseg),
        "total_hours": float(total_hours),
        "bin_sec": float(bin_sec),
    }


def compute_rate_npart_coherence(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    subject_dir: Path,
    dataset: str,
    bin_sec: float = 300.0,
    merge_gap_sec: float = 5.0,
    min_span_bins: int = 48,
    nperseg_max: int = 64,
    coherence_band_mhz: Tuple[float, float] = (0.02, 0.5),
) -> Dict[str, Any]:
    """Cross-spectral coherence between rate trace and n_participating trace.

    n_participating is computed as mean(eventsBool.sum(axis=0)) per bin.
    Only contiguous spans where both rate and npart are valid are used.
    """
    from scipy.signal import coherence as scipy_coherence

    trace = build_continuous_rate_trace(
        events=events, block_ranges=block_ranges,
        bin_sec=bin_sec, merge_gap_sec=merge_gap_sec,
    )
    if "warning" in trace:
        return {"warning": trace["warning"]}

    rate = np.asarray(trace["rate_per_hour"], dtype=float)
    valid_rate = np.asarray(trace["valid_mask"], dtype=bool)
    bin_edges = np.asarray(trace["bin_edges"], dtype=float)
    n_bins = len(rate)

    npart_arr = _build_npart_trace(events, subject_dir, dataset, bin_edges, n_bins)
    if npart_arr is None:
        return {"warning": "npart_trace_failed"}

    valid_npart = np.isfinite(npart_arr)
    joint_valid = valid_rate & valid_npart
    spans = _contiguous_true_spans(joint_valid)
    usable = [(s, e) for s, e in spans if (e - s) >= min_span_bins]

    if not usable:
        return {"warning": "no_usable_joint_spans"}

    span_lengths = [e - s for s, e in usable]
    min_span_len = min(span_lengths)
    common_nperseg = min(int(nperseg_max), max(8, int(min_span_len // 2)))
    if common_nperseg < 8:
        return {"warning": "insufficient_span_length_for_coherence"}
    usable = [(s, e) for s, e in usable if (e - s) >= common_nperseg]
    if not usable:
        return {"warning": "no_usable_joint_spans_after_common_nperseg"}

    fs_hz = 1.0 / float(bin_sec)
    all_coh: List[np.ndarray] = []
    all_weights: List[float] = []
    common_freqs: Optional[np.ndarray] = None

    for s, e in usable:
        r_seg = rate[s:e].copy()
        n_seg = npart_arr[s:e].copy()
        r_seg -= np.nanmean(r_seg)
        n_seg -= np.nanmean(n_seg)
        freqs, cxy = scipy_coherence(
            r_seg,
            n_seg,
            fs=fs_hz,
            nperseg=common_nperseg,
            noverlap=common_nperseg // 2,
        )
        if common_freqs is None:
            common_freqs = freqs
        if len(freqs) == len(common_freqs) and np.allclose(freqs, common_freqs):
            all_coh.append(cxy)
            all_weights.append(float(e - s))

    if not all_coh or common_freqs is None:
        return {"warning": "coherence_computation_failed"}

    weights = np.array(all_weights, dtype=float)
    weights /= weights.sum()
    avg_coh = np.zeros_like(all_coh[0])
    for w, c in zip(weights, all_coh):
        avg_coh += w * c

    freqs_mhz = common_freqs * 1000.0
    f_lo, f_hi = coherence_band_mhz
    band_mask = (freqs_mhz >= f_lo) & (freqs_mhz <= f_hi)
    median_coh = float(np.median(avg_coh[band_mask])) if np.any(band_mask) else np.nan

    return {
        "freqs_mhz": freqs_mhz,
        "coherence": avg_coh,
        "median_coherence": median_coh,
        "n_spans": len(usable),
        "nperseg": int(common_nperseg),
        "bin_sec": float(bin_sec),
    }


def _build_npart_trace(
    events: np.ndarray,
    subject_dir: Path,
    dataset: str,
    bin_edges: np.ndarray,
    n_bins: int,
) -> Optional[np.ndarray]:
    """Build mean(n_participating) trace from lagPat files.

    Returns an array of length n_bins with NaN for bins without events.
    """
    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        return None

    all_npart: List[float] = []
    all_times: List[float] = []

    for lp_file in lagpat_files:
        try:
            lp = np.load(lp_file, allow_pickle=True)
            events_bool = lp["eventsBool"]
            start_t = float(lp["start_t"])
        except Exception:
            continue

        n_ch, n_events = events_bool.shape
        npart_per_event = events_bool.sum(axis=0).astype(float)

        packed_file = lp_file.parent / lp_file.name.replace("_lagPat.npz", "_packedTimes.npy")
        wfc_packed = lp_file.parent / lp_file.name.replace(
            "_lagPat.npz", "_packedTimes_withFreqCent.npy"
        )
        for pf in (wfc_packed, packed_file):
            if pf.exists():
                try:
                    packed_times = np.load(pf, allow_pickle=True)
                    if len(packed_times) >= n_events:
                        for i in range(n_events):
                            ev = packed_times[i]
                            if hasattr(ev, '__len__') and len(ev) >= 2:
                                t = start_t + float(ev[0])
                            else:
                                t = start_t + float(ev)
                            all_times.append(t)
                            all_npart.append(float(npart_per_event[i]))
                        break
                except Exception:
                    continue
        else:
            block_dur = 2 * 3600.0 if dataset == "yuquan" else 3600.0
            spacing = block_dur / max(n_events, 1)
            for i in range(n_events):
                all_times.append(start_t + i * spacing)
                all_npart.append(float(npart_per_event[i]))

    if not all_times:
        return None

    times_arr = np.array(all_times, dtype=float)
    npart_vals = np.array(all_npart, dtype=float)

    result = np.full(n_bins, np.nan, dtype=float)
    bin_idx = np.digitize(times_arr, bin_edges) - 1
    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            result[i] = float(np.mean(npart_vals[mask]))

    return result


def compute_seizure_triggered_rate(
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    seizure_times: List[float],
    bin_sec: float = 300.0,
    window_hours: float = 12.0,
    merge_gap_sec: float = 5.0,
    min_valid_frac: float = 0.5,
) -> Dict[str, Any]:
    """Seizure-triggered average (STA) of the rate trace.

    For each seizure, extracts a ±window_hours window from the continuous
    rate trace, z-scores it, then averages across usable seizures.
    """
    if not seizure_times:
        return {"warning": "no_seizure_times"}

    trace = build_continuous_rate_trace(
        events=events, block_ranges=block_ranges,
        bin_sec=bin_sec, merge_gap_sec=merge_gap_sec,
    )
    if "warning" in trace:
        return {"warning": trace["warning"]}

    rate = np.asarray(trace["rate_per_hour"], dtype=float)
    valid = np.asarray(trace["valid_mask"], dtype=bool)
    centers = np.asarray(trace["bin_centers"], dtype=float)

    global_vals = rate[valid & np.isfinite(rate)]
    if len(global_vals) < 5:
        return {"warning": "insufficient_valid_bins"}
    g_mean = float(np.mean(global_vals))
    g_std = float(np.std(global_vals))
    if g_std < 1e-12:
        return {"warning": "zero_rate_variance"}

    window_sec = window_hours * 3600.0
    n_window_bins = int(round(2 * window_sec / bin_sec))
    half_bins = n_window_bins // 2
    time_axis = (np.arange(-half_bins, half_bins, dtype=float) * float(bin_sec)) / 3600.0

    usable_windows: List[np.ndarray] = []
    for sz_t in seizure_times:
        center_idx = np.argmin(np.abs(centers - sz_t))
        i0 = center_idx - half_bins
        i1 = center_idx + half_bins
        if i0 < 0 or i1 > len(rate):
            continue
        window_rate = rate[i0:i1].copy()
        window_valid = valid[i0:i1].copy()
        valid_frac = float(np.sum(window_valid)) / len(window_valid)
        if valid_frac < min_valid_frac:
            continue
        z_rate = (window_rate - g_mean) / g_std
        z_rate[~window_valid] = np.nan
        usable_windows.append(z_rate)

    n_usable = len(usable_windows)
    if n_usable == 0:
        return {
            "warning": "no_usable_seizure_windows",
            "n_seizures_total": len(seizure_times),
            "n_seizures_usable": 0,
        }
    if n_usable < 2:
        return {
            "warning": "insufficient_usable_seizure_windows",
            "n_seizures_total": len(seizure_times),
            "n_seizures_usable": n_usable,
        }

    stacked = np.array(usable_windows, dtype=float)
    valid_counts = np.sum(np.isfinite(stacked), axis=0)
    sum_vals = np.nansum(stacked, axis=0)
    sta_mean = np.full(stacked.shape[1], np.nan, dtype=float)
    mean_mask = valid_counts > 0
    sta_mean[mean_mask] = sum_vals[mean_mask] / valid_counts[mean_mask]

    if n_usable > 1:
        centered = stacked - sta_mean[None, :]
        centered[~np.isfinite(stacked)] = np.nan
        ss = np.nansum(centered ** 2, axis=0)
        denom = np.maximum(valid_counts - 1, 1)
        std = np.full_like(sta_mean, np.nan)
        std[valid_counts > 1] = np.sqrt(ss[valid_counts > 1] / denom[valid_counts > 1])
        sta_sem = np.full_like(sta_mean, np.nan)
        sta_sem[valid_counts > 1] = std[valid_counts > 1] / np.sqrt(valid_counts[valid_counts > 1])
    else:
        sta_sem = np.zeros_like(sta_mean)

    bins_per_hour = 3600.0 / bin_sec
    def _mean_z_in_range(h_lo: float, h_hi: float) -> float:
        i0 = int(round((h_lo + window_hours) * bins_per_hour))
        i1 = int(round((h_hi + window_hours) * bins_per_hour))
        i0 = max(0, min(i0, len(sta_mean)))
        i1 = max(0, min(i1, len(sta_mean)))
        seg = sta_mean[i0:i1]
        return float(np.nanmean(seg)) if len(seg) > 0 else np.nan

    pre_rate = _mean_z_in_range(-6.0, -1.0)
    baseline_rate = _mean_z_in_range(-12.0, -6.0)
    post_rate = _mean_z_in_range(1.0, 6.0)
    post_baseline = _mean_z_in_range(6.0, 12.0)

    return {
        "time_hours": time_axis,
        "sta_mean": sta_mean,
        "sta_sem": sta_sem,
        "valid_counts_per_bin": valid_counts.astype(int),
        "n_seizures_total": len(seizure_times),
        "n_seizures_usable": n_usable,
        "pre_rate": pre_rate,
        "baseline_rate": baseline_rate,
        "post_rate": post_rate,
        "post_baseline": post_baseline,
    }


def load_seizure_times(subject: str, dataset: str) -> List[float]:
    """Load seizure onset times (Unix epoch seconds) from existing results.

    Yuquan: reads results/seizure_detection/pr1_seizure_<subject>.json
    Epilepsiae: reads results/epilepsiae_seizure_inventory.csv
    """
    import csv

    times: List[float] = []
    if dataset == "yuquan":
        path = Path("results/seizure_detection") / f"pr1_seizure_{subject}.json"
        if not path.exists():
            return times
        with open(path) as f:
            data = json.load(f)
        for file_rec in data.get("files", []):
            for si in file_rec.get("seizure_intervals", []):
                onset = si.get("onset_epoch")
                if onset is not None:
                    times.append(float(onset))
    elif dataset == "epilepsiae":
        path = Path("results/epilepsiae_seizure_inventory.csv")
        if not path.exists():
            return times
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("subject") == subject:
                    onset = row.get("eeg_onset_epoch", "")
                    if onset:
                        try:
                            times.append(float(onset))
                        except ValueError:
                            pass
    times.sort()
    return times


# ---------------------------------------------------------------------------
# Per-channel spatial modulation analysis
# ---------------------------------------------------------------------------


def _normalize_channel_name(name: str) -> str:
    """Strip whitespace, uppercase, remove known prefixes."""
    s = name.strip().upper()
    for prefix in ("EEG ", "EEG_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def match_bipolar_soz(ch_name: str, soz_set: set) -> str:
    """Match bipolar channel X-Y against SOZ set; any contact in SOZ -> 'soz'."""
    normalized = _normalize_channel_name(ch_name)
    parts = normalized.split("-")
    for p in parts:
        p = p.strip()
        if p in soz_set:
            return "soz"
    return "non_soz"


def match_bipolar_focus_rel(
    ch_name: str, focus_rel: Dict[str, list]
) -> str:
    """Match bipolar channel to Epilepsiae i/l/e; priority i > l > e."""
    normalized = _normalize_channel_name(ch_name)
    parts = [p.strip() for p in normalized.split("-")]
    for label in ("i", "l", "e"):
        label_set = {_normalize_channel_name(c) for c in focus_rel.get(label, [])}
        for p in parts:
            if p in label_set:
                return label
    return "unknown"


def load_perchannel_events_relaxed(
    subject_dir: Path,
    dataset: str,
    refine_k: float = 0.0,
    min_count: int = 100,
    min_rate: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Load per-channel events from *_gpu.npz with relaxed refine threshold.

    Unlike load_yuquan_subject_events, the channel set comes from gpu.npz
    full channel names + relaxed refine, not from lagPat chnNames.

    Returns None if no valid gpu.npz found.
    """
    from src.group_event_analysis import select_core_channels_by_event_count

    subject_dir = Path(subject_dir)

    all_ch_names: List[str] = []
    ch_name_set: set = set()
    sum_events_count: Dict[str, int] = {}
    per_ch_events_raw: Dict[str, list] = {}
    block_ranges: List[Tuple[float, float]] = []
    total_hours = 0.0

    if dataset == "yuquan":
        block_dur = 2 * 3600.0
        edf_files = sorted(subject_dir.glob("*.edf"))
        for edf in edf_files:
            gpu_file = edf.with_name(edf.stem + "_gpu.npz")
            gpu = _try_load_gpu(gpu_file) if gpu_file.exists() else None
            if gpu is None:
                continue

            start_t = float(gpu["start_time"])
            block_ranges.append((start_t, start_t + block_dur))
            total_hours += block_dur / 3600.0

            chns = list(gpu["chns_names"])
            dets = gpu["whole_dets"]
            try:
                ecnt = np.array(gpu["events_count"], dtype=int).ravel()
            except Exception:
                ecnt = np.zeros(len(chns), dtype=int)

            for i, ch in enumerate(chns):
                if ch not in ch_name_set:
                    ch_name_set.add(ch)
                    all_ch_names.append(ch)
                    per_ch_events_raw[ch] = []
                cnt = int(ecnt[i]) if i < len(ecnt) else 0
                sum_events_count[ch] = sum_events_count.get(ch, 0) + cnt

                if cnt > 0:
                    try:
                        ch_dets = np.array(dets[i]).reshape(-1, 2)
                        per_ch_events_raw[ch].append(ch_dets + start_t)
                    except Exception:
                        pass

    elif dataset == "epilepsiae":
        block_dur = 3600.0
        lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))

        start_t_map = {}
        for lp_file in lagpat_files:
            stem = lp_file.stem.replace("_lagPat", "")
            try:
                lp = np.load(lp_file, allow_pickle=True)
                start_t_map[stem] = float(lp["start_t"])
            except Exception:
                pass

        gpu_files = sorted(subject_dir.glob("*_gpu.npz"))
        for gpu_file in gpu_files:
            gpu = _try_load_gpu(gpu_file)
            if gpu is None:
                continue

            stem = gpu_file.stem.replace("_gpu", "")
            if "start_time" in gpu:
                start_t = float(gpu["start_time"])
            elif stem in start_t_map:
                start_t = start_t_map[stem]
            else:
                continue

            block_ranges.append((start_t, start_t + block_dur))
            total_hours += block_dur / 3600.0

            chns = list(gpu["chns_names"])
            dets = gpu["whole_dets"]
            try:
                ecnt = np.array(gpu["events_count"], dtype=int).ravel()
            except Exception:
                ecnt = np.zeros(len(chns), dtype=int)

            for i, ch in enumerate(chns):
                if ch not in ch_name_set:
                    ch_name_set.add(ch)
                    all_ch_names.append(ch)
                    per_ch_events_raw[ch] = []
                cnt = int(ecnt[i]) if i < len(ecnt) else 0
                sum_events_count[ch] = sum_events_count.get(ch, 0) + cnt

                if cnt > 0:
                    try:
                        ch_dets = np.array(dets[i]).reshape(-1, 2)
                        per_ch_events_raw[ch].append(ch_dets + start_t)
                    except Exception:
                        pass
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not all_ch_names or total_hours == 0:
        return None

    counts_arr = np.array(
        [sum_events_count.get(c, 0) for c in all_ch_names], dtype=float,
    )
    selected = select_core_channels_by_event_count(
        events_count=counts_arr,
        ch_names=all_ch_names,
        method="mean_std",
        k=refine_k,
        min_count=1,
    )
    selected = [
        ch for ch in selected
        if sum_events_count.get(ch, 0) >= min_count
        and (sum_events_count.get(ch, 0) / total_hours) >= min_rate
    ]

    if not selected:
        return None

    per_ch_events: Dict[str, np.ndarray] = {}
    for ch in selected:
        if per_ch_events_raw.get(ch):
            cat = np.concatenate(per_ch_events_raw[ch], axis=0)
            per_ch_events[ch] = cat[cat[:, 0].argsort()]
        else:
            per_ch_events[ch] = np.zeros((0, 2))

    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    lagpat_channels: set = set()
    if lagpat_files:
        try:
            tmp = np.load(lagpat_files[0], allow_pickle=True)
            lagpat_channels = set(tmp["chnNames"])
        except Exception:
            pass

    return {
        "per_ch_events": per_ch_events,
        "ch_names": selected,
        "lagpat_channels": lagpat_channels,
        "block_ranges": block_ranges,
        "total_hours": total_hours,
        "events_count_all": sum_events_count,
    }


def compute_perchannel_metrics(
    ch_name: str,
    events: np.ndarray,
    block_ranges: List[Tuple[float, float]],
    total_hours: float,
    iei_min_threshold: float = 0.01,
    cv_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Compute per-channel temporal metrics with quality control.

    Reuses compute_serial_correlation_decay and
    compute_detrended_serial_correlation on single-channel event sequences.

    Quality control:
      1. IEI < iei_min_threshold (10ms) dropped (detector artifact)
      2. Channels with CV > cv_threshold flagged artifact_suspect
      3. Caller enforces min_count + min_rate double threshold
    """
    n_events = len(events)
    if n_events < 2:
        return _empty_channel_metrics(ch_name, n_events, total_hours)

    iei = compute_iei(events, block_ranges=block_ranges)
    if len(iei) == 0:
        return _empty_channel_metrics(ch_name, n_events, total_hours)

    iei_clean = iei[iei >= iei_min_threshold]
    n_iei_dropped = len(iei) - len(iei_clean)
    iei = iei_clean

    if len(iei) < 10:
        return _empty_channel_metrics(ch_name, n_events, total_hours)

    event_rate = n_events / total_hours if total_hours > 0 else 0.0
    iei_mean = float(np.mean(iei))
    iei_std = float(np.std(iei))
    iei_cv = iei_std / iei_mean if iei_mean > 0 else 0.0
    artifact_suspect = iei_cv > cv_threshold

    iei_median = float(np.median(iei))
    iei_p02 = float(np.percentile(iei, 2))

    decay = compute_serial_correlation_decay(iei, max_lag=50, min_pairs=20)
    lag1_r = decay["lag1_r"]
    half_life_lag = decay["half_life_lag"]

    event_starts = events[:, 0]
    detrended = compute_detrended_serial_correlation(event_starts, iei, window_sec=600.0)
    detrended_r = detrended["detrended_r"]
    detrend_fraction = detrended["detrend_fraction"]

    return {
        "ch_name": ch_name,
        "n_events": n_events,
        "event_rate": round(event_rate, 2),
        "n_iei": len(iei),
        "n_iei_dropped": n_iei_dropped,
        "iei_mean": round(iei_mean, 6),
        "iei_median": round(iei_median, 6),
        "iei_p02": round(iei_p02, 6),
        "iei_cv": round(iei_cv, 3),
        "artifact_suspect": artifact_suspect,
        "iei_lag1_r": round(lag1_r, 4) if np.isfinite(lag1_r) else None,
        "iei_half_life": half_life_lag,
        "iei_detrended_r": round(detrended_r, 4) if np.isfinite(detrended_r) else None,
        "detrend_fraction": round(detrend_fraction, 3) if np.isfinite(detrend_fraction) else None,
    }


def _empty_channel_metrics(ch_name: str, n_events: int, total_hours: float) -> Dict[str, Any]:
    return {
        "ch_name": ch_name,
        "n_events": n_events,
        "event_rate": round(n_events / total_hours, 2) if total_hours > 0 else 0.0,
        "n_iei": 0,
        "n_iei_dropped": 0,
        "iei_mean": None,
        "iei_median": None,
        "iei_p02": None,
        "iei_cv": None,
        "artifact_suspect": True,
        "iei_lag1_r": None,
        "iei_half_life": None,
        "iei_detrended_r": None,
        "detrend_fraction": None,
    }


def annotate_channels_soz(
    ch_names: List[str],
    soz_channels: List[str],
    focus_rel: Optional[Dict[str, list]] = None,
) -> Dict[str, str]:
    """Annotate each channel with SOZ label using bipolar-any matching.

    For Yuquan: binary soz/non_soz.
    For Epilepsiae with focus_rel: three-tier i/l/e.
    Channels not found in any SOZ/focus_rel set are 'unknown'.
    """
    soz_set = {_normalize_channel_name(c) for c in soz_channels}

    labels = {}
    for ch in ch_names:
        if focus_rel is not None:
            label = match_bipolar_focus_rel(ch, focus_rel)
            if label != "unknown":
                labels[ch] = label
                continue

        label = match_bipolar_soz(ch, soz_set)
        labels[ch] = label

    return labels


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_subject_result(result: SubjectPeriodicityResult, out_path: Path) -> None:
    """Save result as JSON (ndarrays converted to lists)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (ChannelPeriodicityResult, SubjectPeriodicityResult,
                            PSDResult, SpecparamResult, IEIFitResult, SurrogateResult)):
            return asdict(obj)
        return obj

    data = asdict(result)

    with open(out_path, "w") as f:
        json.dump(data, f, default=_convert, indent=2)

    logger.info(f"Saved {out_path}")
