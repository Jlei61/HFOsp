"""
HFO Detector (BQK algorithm)

Design goals:
- Explicit, boring, testable API
- No "guessing" about referencing or channel lists
- Stream-friendly: support chunked detection to avoid huge memory spikes

This module detects HFO-like events using the BQK algorithm:
1) Multi-band envelope construction (subdivide into 20Hz sub-bands)
2) Bandpass filter + Hilbert transform per sub-band
3) Sum envelopes across sub-bands
4) Dual-threshold detection (rel_thresh × local_median AND abs_thresh × global_median)
5) Merge-close + duration filtering

Author: HFOsp Team
Date: 2026-01-31
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from .utils.logging_utils import get_run_logger, log_section


@dataclass(frozen=True)
class HFODetectionConfig:
    """Configuration for BQK HFO detection."""

    band: str = "ripple"  # 'ripple' or 'fast_ripple'
    bandpass: Optional[Tuple[float, float]] = None  # overrides default band

    # BQK sub-band parameters
    subband_width: float = 20.0  # Width of each sub-band in Hz (BQK standard)

    # BQK dual-threshold parameters
    rel_thresh: float = 3.0  # Relative threshold (× channel median)
    abs_thresh: float = 3.0  # Absolute threshold (× global median)
    
    # Event post-processing
    min_gap_ms: float = 20.0     # Minimum gap between events (ms)
    min_last_ms: float = 50.0    # Minimum event duration (ms)

    # Chunking (seconds). If None, process full signal at once.
    chunk_sec: Optional[float] = 30.0
    chunk_overlap_sec: float = 1.0

    # Parallelization: number of jobs for multi-band envelope computation
    # -1 = use all CPUs, 1 = serial (recommended for chunked processing)
    # NOTE: Parallel overhead (joblib process creation ~500ms) exceeds compute time
    #       for typical chunk sizes (30s). Use n_jobs=1 unless chunk_sec=None.
    n_jobs: int = 1
    verbose: int = 0  # joblib verbosity level


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

    meta: Dict = field(default_factory=dict)


def _default_bandpass_for(band: str) -> Tuple[float, float]:
    """Return default frequency band for ripple or fast-ripple."""
    band = band.lower().strip()
    if band in ("ripple", "ripples"):
        return (80.0, 250.0)
    if band in ("fast_ripple", "fast-ripple", "fr", "fast"):
        return (250.0, 500.0)
    raise ValueError(f"Unknown band='{band}'. Use 'ripple' or 'fast_ripple'.")


class HFODetector:
    """
    BQK HFO detector for a preprocessed signal (notch/resample done upstream).

    Uses the BQK algorithm with optimized multi-band envelope computation:
    - Multi-band subdivision (default: 20Hz sub-bands)
    - Parallel processing via joblib (if n_jobs != 1)
    - Dual-threshold detection (relative + absolute)

    Typical usage:
        pre = SEEGPreprocessor(reference='bipolar').run(edf)
        det = HFODetector(HFODetectionConfig(band='ripple', n_jobs=-1))
        result = det.detect(pre)
    """

    def __init__(self, config: Optional[HFODetectionConfig] = None):
        self.config = config or HFODetectionConfig()

    def detect(self, x: Union["PreprocessingResult", np.ndarray], sfreq: Optional[float] = None,
               ch_names: Optional[List[str]] = None) -> HFODetectionResult:
        """
        Detect HFO events using BQK algorithm.

        Parameters
        ----------
        x : PreprocessingResult or np.ndarray
            Preprocessed signal. If ndarray, must provide sfreq and ch_names.
        sfreq : float, optional
            Sampling frequency (required if x is ndarray).
        ch_names : List[str], optional
            Channel names (required if x is ndarray).

        Returns
        -------
        HFODetectionResult
            Detection results with events per channel.
        """
        t_start = time.time()
        logger = get_run_logger("hfo_detect")
        log_section(logger, "HFO DETECT START (BQK)")
        logger.info("band=%s", str(self.config.band))
        logger.info("n_jobs=%d", int(self.config.n_jobs))

        # Extract data, sfreq, ch_names
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
            "algorithm": "bqk_parallel",
            "n_jobs": cfg.n_jobs,
        }

        # BQK detection (chunked + parallel)
        events_by_channel, med = self._detect_bqk_chunked(
            data=data, sfreq=sfreq_, freqband=(low, high)
        )
        events_count = np.array([ev.shape[0] for ev in events_by_channel], dtype=np.int64)

        res = HFODetectionResult(
            sfreq=sfreq_,
            ch_names=ch_names_,
            band=cfg.band,
            config=cfg,
            events_by_channel=events_by_channel,
            events_count=events_count,
            baseline_median=med,
            meta=meta,
        )
        
        log_section(logger, "HFO DETECT SUMMARY")
        logger.info("channels=%d", int(len(ch_names_)))
        logger.info("total_events=%d", int(np.sum(events_count)))
        logger.info("mean_events_per_channel=%.3f", float(np.mean(events_count)))
        logger.info("elapsed_sec=%.3f", float(time.time() - t_start))
        log_section(logger, "HFO DETECT END")
        return res

    def _detect_bqk_chunked(
        self,
        data: np.ndarray,
        sfreq: float,
        freqband: Tuple[float, float],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        BQK detection with chunked processing and parallel envelope computation.

        Uses BQKDetector class for optimized multi-band envelope:
        - Pre-computed filter coefficients (no redundant butter() calls)
        - Parallel sub-band processing via joblib (if n_jobs != 1)
        - Chunked processing to bound memory usage
        - Proper overlap handling to avoid edge artifacts

        Returns
        -------
        events_by_channel : List[np.ndarray]
            Events per channel, each shape (n_events, 2) [start, end] in seconds.
        baseline_median : np.ndarray
            Baseline envelope median per channel, shape (n_channels,).
        """
        try:
            from src.utils.bqk_utils import BQKDetector, merge_timeRanges
        except ImportError:
            from .utils.bqk_utils import BQKDetector, merge_timeRanges  # type: ignore

        cfg = self.config
        n_ch, n_samp = data.shape

        # Create BQK detector (pre-computes filter coefficients)
        detector = BQKDetector(
            sfreq=sfreq,
            freqband=freqband,
            subband_width=cfg.subband_width,
            rel_thresh=cfg.rel_thresh,
            abs_thresh=cfg.abs_thresh,
            min_gap=cfg.min_gap_ms,
            min_last=cfg.min_last_ms,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )

        # Chunking parameters
        if cfg.chunk_sec is None:
            chunk_len = n_samp
        else:
            chunk_len = int(round(cfg.chunk_sec * sfreq))
        chunk_len = max(1, min(chunk_len, n_samp))
        overlap = int(round(cfg.chunk_overlap_sec * sfreq))
        overlap = max(0, min(overlap, chunk_len // 2))
        step = max(1, chunk_len - overlap)

        all_events: List[List[List[float]]] = [[] for _ in range(n_ch)]

        # Process chunks
        for base_start in range(0, n_samp, step):
            base_end = min(n_samp, base_start + chunk_len)
            pad_start = max(0, base_start - overlap)
            pad_end = min(n_samp, base_end + overlap)

            x = data[:, pad_start:pad_end]
            if x.shape[1] < int(0.2 * sfreq):  # Skip tiny chunks
                continue

            # Detect events in chunk (parallel envelope computation inside)
            seg_events = detector.detect_events(x, start_time=pad_start / sfreq)

            # Keep only events in core region [base_start, base_end)
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

        # Merge events across chunks per channel
        merged_out: List[np.ndarray] = []
        for ci in range(n_ch):
            if not all_events[ci]:
                merged_out.append(np.zeros((0, 2), dtype=np.float64))
                continue
            # Sort and merge
            ev = sorted(all_events[ci], key=lambda x: x[0])
            ev = merge_timeRanges(np.array(ev), min_gap=cfg.min_gap_ms)
            merged_out.append(
                np.array(ev, dtype=np.float64) if len(ev) else np.zeros((0, 2), dtype=np.float64)
            )

        # Estimate baseline median from first 2 seconds
        env0 = detector.compute_envelope(data[:, : min(n_samp, int(2 * sfreq))])
        baseline_med = np.median(env0, axis=-1)

        return merged_out, baseline_med


