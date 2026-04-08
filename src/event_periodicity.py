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
            start_t = float(lp["start_t"])

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

    for i, center in enumerate(centers):
        in_window = np.abs(centers - center) <= half_window
        counts[i] = int(np.sum(in_window))
        if counts[i] >= min_local_events:
            baseline[i] = float(np.median(iei[in_window]))

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
