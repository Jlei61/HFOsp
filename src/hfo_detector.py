"""
HFO Detector (Hilbert-envelope + hysteresis thresholds)

Design goals:
- Explicit, boring, testable API
- No "guessing" about referencing or channel lists
- Stream-friendly: support chunked detection to avoid huge memory spikes

This module detects HFO-like events in a band (ripple / fast-ripple) using:
1) Bandpass filter
2) Hilbert envelope (analytic amplitude)
3) Envelope smoothing (moving average)
4) Hysteresis thresholding (high threshold starts events; low threshold ends events)
5) Merge-close + duration filtering

Author: HFOsp Team
Date: 2026-01-14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


def cupy_hilbert(x_gpu):
    """
    Compute analytic signal on GPU via FFT (Hilbert transform).

    Parameters
    ----------
    x_gpu : cupy.ndarray
        Shape (..., n_samples). Real-valued signal.

    Returns
    -------
    analytic : cupy.ndarray (complex)
        Analytic signal of x_gpu along last axis.
    """
    if not _HAS_CUPY:
        raise RuntimeError("cupy_hilbert requires CuPy installed.")
    n = x_gpu.shape[-1]
    Xf = cp.fft.fft(x_gpu, axis=-1)
    h = cp.zeros((n,), dtype=Xf.dtype)

    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    return cp.fft.ifft(Xf * h, axis=-1)


@dataclass(frozen=True)
class HFODetectionConfig:
    """Configuration for HFO detection."""

    # Detection algorithm:
    # - 'bqk': match src/utils/bqk_utils.py (dual-threshold on Hilbert envelope + merge + min_last)
    # - 'mad_hysteresis': this module's MAD + hysteresis detector (kept for comparison)
    algorithm: str = "bqk"

    band: str = "ripple"  # 'ripple' or 'fast_ripple'
    bandpass: Optional[Tuple[float, float]] = None  # overrides default band

    # Envelope smoothing (moving average) in milliseconds
    smooth_ms: float = 5.0

    # Robust baseline on envelope: median + k * MAD
    # High threshold triggers event; low threshold terminates it (hysteresis)
    high_k: float = 6.0
    low_k: float = 3.0

    # Event post-processing
    min_duration_ms: float = 6.0
    max_duration_ms: float = 200.0
    min_gap_ms: float = 10.0

    # bqk-style thresholds (see src/utils/bqk_utils.py: find_high_enveTimes)
    rel_thresh: float = 3.0
    abs_thresh: float = 3.0
    min_last_ms: float = 50.0

    # Chunking (seconds). If None, process full signal at once.
    chunk_sec: Optional[float] = 30.0
    chunk_overlap_sec: float = 1.0

    # Experimental: ictal exclusion for baseline
    exclude_ictal: bool = False
    ictal_min_duration_sec: float = 3.0
    ictal_k: float = 8.0

    # If True and CuPy is available, use GPU FFT Hilbert for envelope (big speedup on long signals).
    # Note: bandpass filtering is still CPU in this detector (to avoid subtle numeric/API differences).
    use_gpu: bool = False


@dataclass
class HFODetectionResult:
    """Detection output."""

    sfreq: float
    ch_names: List[str]
    band: str
    config: HFODetectionConfig

    # Each element: array shape (n_events, 2) in seconds [start, end]
    events_by_channel: List[np.ndarray]

    # Convenience stats
    events_count: np.ndarray  # shape (n_channels,)
    baseline_median: np.ndarray  # shape (n_channels,)
    baseline_mad: np.ndarray  # shape (n_channels,)
    used_gpu: bool = False

    meta: Dict = field(default_factory=dict)


def _mad(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Median absolute deviation (MAD), unscaled."""
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average along last axis."""
    if win <= 1:
        return x
    # Use cumulative sum for speed; preserve dtype as float64 to reduce numerical junk.
    x = x.astype(np.float64, copy=False)
    c = np.cumsum(x, axis=-1)
    c[..., win:] = c[..., win:] - c[..., :-win]
    out = c[..., win - 1 :] / float(win)
    # Pad to original length (left pad with first value to keep alignment simple)
    pad = np.repeat(out[..., :1], win - 1, axis=-1)
    return np.concatenate([pad, out], axis=-1)


def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive runs [(start, end), ...] where mask is True."""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    dm = np.diff(m, prepend=0, append=0)
    starts = np.where(dm == 1)[0]
    ends = np.where(dm == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _detect_events_hysteresis_1d(
    env: np.ndarray,
    sfreq: float,
    thr_low: float,
    thr_high: float,
    min_dur_s: float,
    max_dur_s: float,
    min_gap_s: float,
) -> np.ndarray:
    """
    Detect events with hysteresis thresholds on a 1D envelope.

    Event starts when env crosses above thr_high.
    Event extends until env falls below thr_low.
    """
    if env.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    high = env >= thr_high
    if not np.any(high):
        return np.zeros((0, 2), dtype=np.float64)

    low = env >= thr_low
    high_runs = _find_runs(high)
    events: List[Tuple[int, int]] = []

    for hs, he in high_runs:
        # extend left to last low=False
        s = hs
        while s > 0 and low[s - 1]:
            s -= 1
        # extend right while low=True
        e = he
        n = env.shape[0]
        while e < n and low[e]:
            e += 1
        events.append((s, e))

    # Sort and merge-close
    events.sort()
    merged: List[Tuple[int, int]] = []
    gap = int(round(min_gap_s * sfreq))
    for s, e in events:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    # Duration filter
    min_len = int(round(min_dur_s * sfreq))
    max_len = int(round(max_dur_s * sfreq))
    kept = [(s, e) for s, e in merged if (e - s) >= min_len and (e - s) <= max_len]

    if not kept:
        return np.zeros((0, 2), dtype=np.float64)

    out = np.array(kept, dtype=np.float64) / float(sfreq)
    return out


def _default_bandpass_for(band: str) -> Tuple[float, float]:
    band = band.lower().strip()
    if band in ("ripple", "ripples"):
        return (80.0, 250.0)
    if band in ("fast_ripple", "fast-ripple", "fr", "fast"):
        return (250.0, 500.0)
    raise ValueError(f"Unknown band='{band}'. Use 'ripple' or 'fast_ripple'.")


def _bandpass_sos(low: float, high: float, sfreq: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    if high >= nyq:
        # Clip rather than crash; still warn via metadata upstream.
        high = max(nyq - 1.0, low + 1.0)
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _detect_ictal_mask(
    data: np.ndarray, sfreq: float, min_duration_sec: float, k: float
) -> np.ndarray:
    """
    Very simple ictal detector: channel-averaged absolute amplitude, smoothed,
    thresholded by median + k*MAD.
    """
    # channel-avg absolute amplitude
    x = np.mean(np.abs(data), axis=0)
    win = max(1, int(round(0.5 * sfreq)))  # 500ms smoothing
    xs = _moving_average(x, win)
    med = np.median(xs)
    mad = _mad(xs)
    thr = med + k * mad
    mask = xs >= thr

    # Enforce min duration
    runs = _find_runs(mask)
    keep = np.zeros_like(mask, dtype=bool)
    min_len = int(round(min_duration_sec * sfreq))
    for s, e in runs:
        if e - s >= min_len:
            keep[s:e] = True
    return keep


class HFODetector:
    """
    HFO detector for a preprocessed signal (notch/resample done upstream).

    Typical usage:
        pre = SEEGPreprocessor(reference='bipolar', crop_seconds=120).run(edf)
        det = HFODetector(HFODetectionConfig(band='ripple')).detect(pre)
    """

    def __init__(self, config: Optional[HFODetectionConfig] = None):
        self.config = config or HFODetectionConfig()

        self._used_gpu = bool(self.config.use_gpu and _HAS_CUPY)

    def detect(self, x: Union["PreprocessingResult", np.ndarray], sfreq: Optional[float] = None,
               ch_names: Optional[List[str]] = None) -> HFODetectionResult:
        """
        Detect events from a PreprocessingResult or raw ndarray.

        If x is ndarray, sfreq and ch_names must be provided.
        """
        # Avoid importing preprocessing at module import time.
        if hasattr(x, "data") and hasattr(x, "sfreq") and hasattr(x, "ch_names"):
            data = getattr(x, "data")
            sfreq_ = float(getattr(x, "sfreq"))
            ch_names_ = list(getattr(x, "ch_names"))
        else:
            if sfreq is None or ch_names is None:
                raise ValueError("When passing ndarray, sfreq and ch_names are required.")
            data = x  # type: ignore[assignment]
            sfreq_ = float(sfreq)
            ch_names_ = list(ch_names)

        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected data shape (n_channels, n_samples), got {data.shape}")
        if data.shape[0] != len(ch_names_):
            raise ValueError("len(ch_names) must match data.shape[0]")

        cfg = self.config
        low, high = cfg.bandpass if cfg.bandpass is not None else _default_bandpass_for(cfg.band)

        meta: Dict = {
            "bandpass": (low, high),
            "note": "Hilbert uses GPU FFT when use_gpu=True and CuPy is available; bandpass remains CPU.",
        }
        if cfg.use_gpu and not _HAS_CUPY:
            meta["gpu_warning"] = "use_gpu=True requested but CuPy not installed; running CPU."

        # Optional ictal exclusion mask for baseline
        ictal_mask = None
        if cfg.exclude_ictal:
            ictal_mask = _detect_ictal_mask(
                data=data, sfreq=sfreq_, min_duration_sec=cfg.ictal_min_duration_sec, k=cfg.ictal_k
            )
            meta["ictal_excluded_fraction"] = float(np.mean(ictal_mask))

        # Detection
        algo = (cfg.algorithm or "bqk").lower().strip()
        if algo == "bqk":
            events_by_channel, med, mad = self._detect_bqk_chunked(
                data=data, sfreq=sfreq_, freqband=(low, high)
            )
            meta["algorithm"] = "bqk_utils"
        elif algo in ("mad_hysteresis", "hysteresis", "mad"):
            events_by_channel, med, mad = self._detect_chunked(
                data=data, sfreq=sfreq_, bandpass=(low, high), ictal_mask=ictal_mask
            )
            meta["algorithm"] = "mad_hysteresis"
        else:
            raise ValueError(f"Unknown algorithm='{cfg.algorithm}'. Use 'bqk' or 'mad_hysteresis'.")
        events_count = np.array([ev.shape[0] for ev in events_by_channel], dtype=np.int64)

        return HFODetectionResult(
            sfreq=sfreq_,
            ch_names=ch_names_,
            band=cfg.band,
            config=cfg,
            events_by_channel=events_by_channel,
            events_count=events_count,
            baseline_median=med,
            baseline_mad=mad,
            used_gpu=self._used_gpu,
            meta=meta,
        )

    def _detect_bqk_chunked(
        self,
        data: np.ndarray,
        sfreq: float,
        freqband: Tuple[float, float],
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Detection that matches src/utils/bqk_utils.py:
        - envelope = return_hil_enve_norm(data, fs, freqband)
        - high = (env > rel_thresh*ch_median) & (env > abs_thresh*global_median)
        - ranges = return_timeRanges + merge_timeRanges
        - keep events longer than min_last_ms

        We do it chunked to bound memory; then merge across chunks.
        """
        try:
            from src.utils import bqk_utils as bqk
        except Exception:
            from .utils import bqk_utils as bqk  # type: ignore

        cfg = self.config
        n_ch, n_samp = data.shape

        # chunking
        if cfg.chunk_sec is None:
            chunk_len = n_samp
        else:
            chunk_len = int(round(cfg.chunk_sec * sfreq))
        chunk_len = max(1, min(chunk_len, n_samp))
        overlap = int(round(cfg.chunk_overlap_sec * sfreq))
        overlap = max(0, min(overlap, chunk_len // 2))
        step = max(1, chunk_len - overlap)

        all_events: List[List[List[float]]] = [[] for _ in range(n_ch)]

        for base_start in range(0, n_samp, step):
            base_end = min(n_samp, base_start + chunk_len)
            pad_start = max(0, base_start - overlap)
            pad_end = min(n_samp, base_end + overlap)

            x = data[:, pad_start:pad_end]
            if x.shape[1] < int(0.2 * sfreq):
                continue

            # bqk envelope (includes bandpass+hilbert, plus filterbank sum)
            env = bqk.return_hil_enve_norm(x, sfreq, freqband)
            seg_events = bqk.find_high_enveTimes(
                env,
                chns_nums=env.shape[0],
                fs=sfreq,
                rel_thresh=cfg.rel_thresh,
                abs_thresh=cfg.abs_thresh,
                min_gap=cfg.min_gap_ms,
                min_last=cfg.min_last_ms,
                start_time=pad_start / sfreq,
            )

            # Keep only events in [base_start, base_end)
            t0 = base_start / sfreq
            t1 = base_end / sfreq
            for ci in range(n_ch):
                ch_ev = seg_events[ci]
                if not ch_ev:
                    continue
                for s, e in ch_ev:
                    if e < t0 or s > t1:
                        continue
                    s2 = max(s, t0)
                    e2 = min(e, t1)
                    all_events[ci].append([float(s2), float(e2)])

        # Merge per-channel (bqk merge_timeRanges uses ms min_gap)
        merged_out: List[np.ndarray] = []
        for ci in range(n_ch):
            if not all_events[ci]:
                merged_out.append(np.zeros((0, 2), dtype=np.float64))
                continue
            # sort
            ev = sorted(all_events[ci], key=lambda x: x[0])
            ev = bqk.merge_timeRanges(np.array(ev), min_gap=cfg.min_gap_ms)
            merged_out.append(np.array(ev, dtype=np.float64) if len(ev) else np.zeros((0, 2), dtype=np.float64))

        # baseline placeholders (not used by bqk algorithm)
        env0 = bqk.return_hil_enve_norm(data[:, : min(n_samp, int(2 * sfreq))], sfreq, freqband)
        baseline_med = np.median(env0, axis=-1)
        baseline_mad = np.zeros_like(baseline_med)
        return merged_out, baseline_med, baseline_mad

    def _detect_chunked(
        self,
        data: np.ndarray,
        sfreq: float,
        bandpass: Tuple[float, float],
        ictal_mask: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        cfg = self.config
        n_ch, n_samp = data.shape

        smooth_win = max(1, int(round(cfg.smooth_ms * 1e-3 * sfreq)))
        min_dur_s = cfg.min_duration_ms * 1e-3
        max_dur_s = cfg.max_duration_ms * 1e-3
        min_gap_s = cfg.min_gap_ms * 1e-3

        # Bandpass filter coefficients
        sos = _bandpass_sos(bandpass[0], bandpass[1], sfreq=sfreq, order=4)

        # Baseline is estimated from *envelope*; but envelope requires filtering.
        # For practicality, compute per-channel baseline from a subsampled set of chunks.
        # (Still deterministic, not "guessy".)
        baseline_med = np.zeros((n_ch,), dtype=np.float64)
        baseline_mad = np.zeros((n_ch,), dtype=np.float64)

        # Chunking setup
        if cfg.chunk_sec is None:
            chunk_len = n_samp
        else:
            chunk_len = int(round(cfg.chunk_sec * sfreq))
        chunk_len = max(1, min(chunk_len, n_samp))

        overlap = int(round(cfg.chunk_overlap_sec * sfreq))
        overlap = max(0, min(overlap, chunk_len // 2))

        # First pass: baseline per channel (median/MAD on smoothed envelope).
        # We sample up to ~10 chunks evenly across the signal.
        if n_samp <= chunk_len:
            sample_starts = [0]
        else:
            n_samples = min(10, int(np.ceil(n_samp / chunk_len)))
            sample_starts = np.linspace(0, n_samp - chunk_len, n_samples).astype(int).tolist()

        env_samples: List[np.ndarray] = []
        for s0 in sample_starts:
            s1 = s0 + chunk_len
            chunk = data[:, s0:s1]
            if ictal_mask is not None:
                # Keep non-ictal portion only for baseline
                m = ~ictal_mask[s0:s1]
                if np.any(m):
                    chunk = chunk[:, m]
                else:
                    continue
            if chunk.shape[1] < int(0.5 * sfreq):
                continue
            filt = sosfiltfilt(sos, chunk, axis=-1)
            if cfg.use_gpu and _HAS_CUPY:
                filt_gpu = cp.asarray(filt)
                env = cp.abs(cupy_hilbert(filt_gpu)).astype(cp.float64, copy=False)
                env = cp.asnumpy(env)
            else:
                env = np.abs(hilbert(filt, axis=-1))
            env = _moving_average(env, smooth_win)
            env_samples.append(env)

        if not env_samples:
            # Fall back: compute on first chunk (even if ictal mask excluded it all).
            chunk = data[:, :chunk_len]
            filt = sosfiltfilt(sos, chunk, axis=-1)
            if cfg.use_gpu and _HAS_CUPY:
                filt_gpu = cp.asarray(filt)
                env = cp.abs(cupy_hilbert(filt_gpu)).astype(cp.float64, copy=False)
                env = cp.asnumpy(env)
            else:
                env = np.abs(hilbert(filt, axis=-1))
            env = _moving_average(env, smooth_win)
            env_samples = [env]

        env_cat = np.concatenate(env_samples, axis=-1)
        baseline_med = np.median(env_cat, axis=-1)
        baseline_mad = _mad(env_cat, axis=-1)
        baseline_mad = np.maximum(baseline_mad, 1e-12)  # avoid zero MAD

        thr_high = baseline_med + cfg.high_k * baseline_mad
        thr_low = baseline_med + cfg.low_k * baseline_mad

        # Second pass: event detection per channel, chunked with overlap.
        events_by_channel: List[List[np.ndarray]] = [[] for _ in range(n_ch)]

        step = chunk_len - overlap
        if step <= 0:
            step = chunk_len

        for base_start in range(0, n_samp, step):
            base_end = min(n_samp, base_start + chunk_len)

            # Expand for overlap padding to reduce edge artifacts.
            pad_start = max(0, base_start - overlap)
            pad_end = min(n_samp, base_end + overlap)

            chunk = data[:, pad_start:pad_end]
            if chunk.shape[1] < 8:  # too short
                continue

            filt = sosfiltfilt(sos, chunk, axis=-1)
            if cfg.use_gpu and _HAS_CUPY:
                filt_gpu = cp.asarray(filt)
                env = cp.abs(cupy_hilbert(filt_gpu)).astype(cp.float64, copy=False)
                env = cp.asnumpy(env)
            else:
                env = np.abs(hilbert(filt, axis=-1))
            env = _moving_average(env, smooth_win)

            # Only keep detections in the central (non-padded) region.
            keep_s = base_start - pad_start
            keep_e = keep_s + (base_end - base_start)

            for ci in range(n_ch):
                ev = _detect_events_hysteresis_1d(
                    env=env[ci, :],
                    sfreq=sfreq,
                    thr_low=float(thr_low[ci]),
                    thr_high=float(thr_high[ci]),
                    min_dur_s=min_dur_s,
                    max_dur_s=max_dur_s,
                    min_gap_s=min_gap_s,
                )
                if ev.shape[0] == 0:
                    continue

                # Convert to samples to filter by keep region, then back to seconds.
                ev_samp = np.round(ev * sfreq).astype(int)
                # keep events that overlap central region
                mask = (ev_samp[:, 1] > keep_s) & (ev_samp[:, 0] < keep_e)
                ev_samp = ev_samp[mask]
                if ev_samp.shape[0] == 0:
                    continue
                # Clip to central region
                ev_samp[:, 0] = np.maximum(ev_samp[:, 0], keep_s)
                ev_samp[:, 1] = np.minimum(ev_samp[:, 1], keep_e)
                ev_sec = (ev_samp.astype(np.float64) / sfreq) + (pad_start / sfreq)
                events_by_channel[ci].append(ev_sec)

        # Merge per-channel across chunks
        merged_out: List[np.ndarray] = []
        min_gap_samp = int(round(min_gap_s * sfreq))
        for ci in range(n_ch):
            if not events_by_channel[ci]:
                merged_out.append(np.zeros((0, 2), dtype=np.float64))
                continue
            ev = np.concatenate(events_by_channel[ci], axis=0)
            ev = ev[np.argsort(ev[:, 0])]

            # Merge close again (across chunk boundaries)
            merged: List[Tuple[int, int]] = []
            ev_s = np.round(ev * sfreq).astype(int)
            for s, e in ev_s:
                if not merged:
                    merged.append((s, e))
                    continue
                ps, pe = merged[-1]
                if s - pe <= min_gap_samp:
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))

            merged_ev = (np.array(merged, dtype=np.float64) / sfreq) if merged else np.zeros((0, 2), dtype=np.float64)
            merged_out.append(merged_ev)

        return merged_out, baseline_med, baseline_mad

