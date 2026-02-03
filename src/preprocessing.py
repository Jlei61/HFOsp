"""
SEEG Data Preprocessing Pipeline for HFO Analysis

This module handles:
- EDF file loading with electrode name parsing
- Re-referencing strategies:
  - Bipolar (adjacent contacts on same shaft)
  - Common Average Reference (CAR) per shaft
  - Detection of pre-bipolar data (A1-A2 labeled as A1)
- Resampling with Nyquist-aware warnings
- Notch and bandpass filtering
- Channel quality assessment
- GPU acceleration support (optional, via CuPy)

Author: HFOsp Team
Date: 2026-01-14
"""

import json
import time
import numpy as np
import mne
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy import signal
from .utils.logging_utils import get_run_logger, log_section

# GPU support (optional)
try:
    import cupy as cp
    from cupyx.scipy.signal import sosfilt as cupy_sosfilt
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


@dataclass
class PreprocessingResult:
    """Container for preprocessing output"""
    data: np.ndarray              # (n_channels, n_samples)
    sfreq: float                  # Sampling frequency
    ch_names: List[str]           # Channel names (monopolar or bipolar)
    original_ch_names: List[str]  # Original EDF channel names
    bipolar_pairs: Optional[List[Tuple[str, str]]] = None  # (ch1, ch2) pairs if bipolar
    shaft_mapping: Dict[str, List[str]] = field(default_factory=dict)
    bad_channels: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    reference_type: str = 'unknown'  # 'monopolar', 'bipolar', 'car'
    excluded_channels: List[str] = field(default_factory=list)
    used_gpu: bool = False        # Whether GPU was used for processing


def _moving_sum_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Fast moving sum for 1D arrays."""
    if win <= 1:
        return x.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    c = np.cumsum(x)
    out = c[win - 1 :] - np.concatenate(([0.0], c[:-win]))
    return out


def _robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score using median and MAD."""
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + eps
    return (x - med) / mad


def detect_seizure_onsets_from_data(
    data: np.ndarray,
    sfreq: float,
    *,
    ll_win_sec: float = 1.0,
    ll_step_sec: float = 0.2,
    ll_k: float = 6.0,
    rms_win_sec: float = 1.0,
    rms_step_sec: float = 0.2,
    rms_k: float = 6.0,
    min_duration_sec: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Detect ictal segments using channel-averaged line length + RMS.
    Returns onset/offset times and sample-level ictal mask.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")

    n_samples = data.shape[1]
    x_mean = np.mean(data, axis=0)

    def _windowed_values(x: np.ndarray, win_sec: float, step_sec: float, func) -> Tuple[np.ndarray, np.ndarray]:
        win = max(1, int(round(win_sec * sfreq)))
        step = max(1, int(round(step_sec * sfreq)))
        if n_samples < win:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        starts = np.arange(0, n_samples - win + 1, step, dtype=np.int64)
        vals = np.zeros((starts.shape[0],), dtype=np.float64)
        for i, s in enumerate(starts):
            seg = x[s : s + win]
            vals[i] = func(seg)
        times = starts.astype(np.float64) / sfreq
        return times, vals

    ll_t, ll_vals = _windowed_values(
        x_mean, ll_win_sec, ll_step_sec, lambda seg: float(np.sum(np.abs(np.diff(seg))))
    )
    rms_t, rms_vals = _windowed_values(
        x_mean, rms_win_sec, rms_step_sec, lambda seg: float(np.sqrt(np.mean(seg**2))))

    if ll_vals.size == 0 or rms_vals.size == 0:
        return {
            "onsets_sec": np.zeros((0,), dtype=np.float64),
            "offsets_sec": np.zeros((0,), dtype=np.float64),
            "ictal_mask": np.zeros((n_samples,), dtype=bool),
        }

    ll_z = _robust_z(ll_vals)
    rms_z = _robust_z(rms_vals)

    ll_flag = ll_z >= float(ll_k)
    rms_flag = rms_z >= float(rms_k)

    # Align to a common timeline by mapping onto the denser grid (use ll_t).
    rms_interp = np.interp(ll_t, rms_t, rms_flag.astype(np.float64), left=0.0, right=0.0) >= 0.5
    ictal_flag = ll_flag | rms_interp

    # Enforce min duration on ll timeline
    runs = []
    cur = None
    for i, val in enumerate(ictal_flag):
        if val and cur is None:
            cur = i
        elif not val and cur is not None:
            runs.append((cur, i))
            cur = None
    if cur is not None:
        runs.append((cur, len(ictal_flag)))

    onsets = []
    offsets = []
    ictal_mask = np.zeros((n_samples,), dtype=bool)
    for s_idx, e_idx in runs:
        t0 = ll_t[s_idx]
        t1 = ll_t[e_idx - 1] + float(ll_step_sec) + float(ll_win_sec)
        if (t1 - t0) < float(min_duration_sec):
            continue
        onsets.append(float(t0))
        offsets.append(float(t1))
        i0 = max(0, int(round(t0 * sfreq)))
        i1 = min(n_samples, int(round(t1 * sfreq)))
        if i1 > i0:
            ictal_mask[i0:i1] = True

    return {
        "onsets_sec": np.asarray(onsets, dtype=np.float64),
        "offsets_sec": np.asarray(offsets, dtype=np.float64),
        "ictal_mask": ictal_mask,
    }


def save_seizure_onsets_json(
    *,
    subject_id: str,
    onsets_sec: np.ndarray,
    offsets_sec: np.ndarray,
    sfreq: float,
    output_dir: Union[str, Path] = "results/seizure_onset",
    params: Optional[Dict] = None,
    method: str = "line_length_rms_channel_avg",
) -> str:
    """
    Save subject-level seizure onset JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}.json"
    payload = {
        "subject_id": str(subject_id),
        "sfreq": float(sfreq),
        "method": str(method),
        "params": params or {},
        "onsets_sec": [float(x) for x in np.asarray(onsets_sec).ravel().tolist()],
        "offsets_sec": [float(x) for x in np.asarray(offsets_sec).ravel().tolist()],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return str(out_path)

def get_array_module(use_gpu: bool = False):
    """Get numpy or cupy based on availability and preference."""
    if use_gpu and HAS_GPU:
        return cp
    return np


def to_numpy(arr):
    """Convert cupy array to numpy if needed."""
    if HAS_GPU and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def to_gpu(arr):
    """Convert numpy array to cupy if GPU available."""
    if HAS_GPU:
        return cp.asarray(arr)
    return arr
    
    
class ElectrodeParser:
    """
    Parses SEEG electrode names into (prefix, number) tuples.
    
    Handles formats:
    - Standard: A1, B10, K3
    - With prime: A'1, B'10
    - With prefixes: POL A1, EEG A1-Ref
    """
    
    # Common prefixes to strip
    STRIP_PREFIXES = ['POL ', 'EEG ', 'SEEG ']
    STRIP_SUFFIXES = ['-Ref', '-REF', '-ref']
    
    # Channels to exclude (case-insensitive patterns)
    EXCLUDE_PATTERNS = ['ECG', 'EKG', 'EMG', 'EOG', 'MK', 'DC', 'Annotations', 
                        'Status', 'Trigger', 'Event', 'STI', 'RESP', 'SpO2']
    
    # Valid SEEG pattern: letters + optional prime + numbers
    SEEG_PATTERN = re.compile(r"^([a-zA-Z]+[']?)([0-9]+)$")
    
    @classmethod
    def clean_name(cls, raw_name: str) -> str:
        """Remove common prefixes and suffixes from channel name."""
        name = raw_name
        for prefix in cls.STRIP_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix in cls.STRIP_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip()
    
    @classmethod
    def is_excluded(cls, name: str) -> bool:
        """Check if channel should be excluded."""
        name_lower = name.lower()
        for pattern in cls.EXCLUDE_PATTERNS:
            if pattern.lower() in name_lower:
                return True
        return False
    
    @classmethod
    def parse(cls, ch_name: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse electrode name into (prefix, number).
        
        Examples:
            'A1' -> ('A', 1)
            'K10' -> ('K', 10)
            "A'5" -> ("A'", 5)
            
        Returns:
            (prefix, number) or (None, None) if invalid
        """
        clean = cls.clean_name(ch_name)
        match = cls.SEEG_PATTERN.match(clean)
        if match:
            return match.group(1).upper(), int(match.group(2))
        return None, None
    
    @classmethod
    def is_valid_seeg(cls, ch_name: str) -> bool:
        """Check if channel name is valid SEEG format."""
        if cls.is_excluded(ch_name):
            return False
        prefix, num = cls.parse(ch_name)
        return prefix is not None


class CommonAverageReferencer:
    """
    Applies Common Average Reference (CAR) per electrode shaft.
    
    For each shaft (e.g., all A contacts), subtracts the mean of all
    contacts on that shaft from each contact.
    """
    
    def compute_car(self, data: np.ndarray, ch_names: List[str]
                    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply CAR per shaft.
        
        Args:
            data: (n_channels, n_samples)
            ch_names: Channel names
            
        Returns:
            car_data: (n_channels, n_samples) CAR referenced data
            ch_names: Same channel names (unchanged)
        """
        car_data = data.copy()
        ch_indices = {name: i for i, name in enumerate(ch_names)}
        
        # Group by shaft
        shaft_groups = {}
        for name in ch_names:
            prefix, num = ElectrodeParser.parse(name)
            if prefix:
                if prefix not in shaft_groups:
                    shaft_groups[prefix] = []
                shaft_groups[prefix].append(name)
        
        # Apply CAR per shaft
        for prefix, channels in shaft_groups.items():
            indices = [ch_indices[ch] for ch in channels]
            shaft_mean = np.mean(data[indices], axis=0, keepdims=True)
            car_data[indices] -= shaft_mean
        
        return car_data, ch_names


class FilterBackend:
    """
    Abstract interface for filtering operations.
    
    Eliminates runtime if/else branching between CPU and GPU implementations.
    """
    
    def apply_notch(self, data: np.ndarray, sfreq: float, freqs: List[float]) -> np.ndarray:
        """Apply notch filter at specified frequencies."""
        raise NotImplementedError
    
    def apply_bandpass(self, data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter."""
        raise NotImplementedError


class CpuFilterBackend(FilterBackend):
    """CPU-based filtering using scipy."""
    
    def apply_notch(self, data: np.ndarray, sfreq: float, freqs: List[float]) -> np.ndarray:
        """Apply notch filter for power line harmonics."""
        for freq in freqs:
            if freq < sfreq / 2:  # Must be below Nyquist
                Q = 30  # Quality factor
                b, a = iirnotch(freq, Q, sfreq)
                data = filtfilt(b, a, data, axis=-1)
        return data
    
    def apply_bandpass(self, data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter."""
        nyq = sfreq / 2
        
        if high >= nyq:
            warnings.warn(f"Bandpass high ({high}Hz) >= Nyquist ({nyq}Hz), clipping to {nyq-1}Hz")
            high = nyq - 1
            
        sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
        data = sosfiltfilt(sos, data, axis=-1)
        return data


class GpuFilterBackend(FilterBackend):
    """GPU-based filtering using CuPy."""
    
    def __init__(self, chunk_sec: float = 20.0):
        """
        Args:
            chunk_sec: Chunk size in seconds for processing (avoids OOM on long recordings)
        """
        if not HAS_GPU:
            raise RuntimeError("GPU requested but CuPy not available")
        self.chunk_sec = float(chunk_sec)
    
    def _sosfiltfilt_reflect_gpu(self, x_gpu, sos_gpu, pad_len: int):
        """
        Approximate filtfilt on GPU via reflect padding + forward/backward sosfilt.
        Avoids worst edge transients from naive forward->flip->forward.
        """
        if pad_len <= 0:
            y = cupy_sosfilt(sos_gpu, x_gpu, axis=-1)
            y = cp.flip(y, axis=-1)
            y = cupy_sosfilt(sos_gpu, y, axis=-1)
            return cp.flip(y, axis=-1)

        x_pad = cp.pad(x_gpu, ((0, 0), (pad_len, pad_len)), mode="reflect")
        y = cupy_sosfilt(sos_gpu, x_pad, axis=-1)
        y = cp.flip(y, axis=-1)
        y = cupy_sosfilt(sos_gpu, y, axis=-1)
        y = cp.flip(y, axis=-1)
        return y[:, pad_len:-pad_len]
    
    def _filter_in_chunks(self, x_gpu, sos_gpu, pad_len: int, sfreq: float):
        """Filter data in chunks to avoid GPU OOM."""
        n_samples = int(x_gpu.shape[1])
        chunk_len = int(max(1, round(self.chunk_sec * sfreq)))
        
        if n_samples <= chunk_len:
            return self._sosfiltfilt_reflect_gpu(x_gpu, sos_gpu, pad_len)
        
        out = cp.zeros_like(x_gpu, dtype=x_gpu.dtype)
        step = max(1, chunk_len)
        
        for start in range(0, n_samples, step):
            end = min(n_samples, start + step)
            ext_start = max(0, start - pad_len)
            ext_end = min(n_samples, end + pad_len)
            chunk = x_gpu[:, ext_start:ext_end]
            filt = self._sosfiltfilt_reflect_gpu(chunk, sos_gpu, pad_len)
            s0 = start - ext_start
            s1 = s0 + (end - start)
            out[:, start:end] = filt[:, s0:s1]
        
        return out
    
    def apply_notch(self, data: np.ndarray, sfreq: float, freqs: List[float]) -> np.ndarray:
        """Apply notch filter on GPU."""
        # Transfer to GPU (float32 to reduce memory)
        data_gpu = cp.asarray(data, dtype=cp.float32)
        
        for freq in freqs:
            if freq < sfreq / 2:
                Q = 30
                b, a = iirnotch(freq, Q, sfreq)
                from scipy.signal import tf2sos
                sos = tf2sos(b, a)
                sos_gpu = cp.asarray(sos)
                pad_len = int(min(data_gpu.shape[1] // 4, max(1, round(0.5 * sfreq))))
                data_gpu = self._filter_in_chunks(data_gpu, sos_gpu, pad_len, sfreq)
        
        return cp.asnumpy(data_gpu)
    
    def apply_bandpass(self, data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter on GPU."""
        nyq = sfreq / 2
        
        if high >= nyq:
            high = nyq - 1
            
        sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
        
        # Transfer to GPU
        data_gpu = cp.asarray(data, dtype=cp.float32)
        sos_gpu = cp.asarray(sos)
        pad_len = int(min(data_gpu.shape[1] // 4, max(1, round(0.5 * sfreq))))
        data_gpu = self._filter_in_chunks(data_gpu, sos_gpu, pad_len, sfreq)
        
        return cp.asnumpy(data_gpu)




class BipolarReferencer:
    """
    Applies bipolar re-referencing to SEEG data.
    
    Critical rules:
    1. Only subtract adjacent contacts on the SAME shaft (same prefix)
    2. Handle non-consecutive numbering (e.g., K3, K5, K6 -> K3-K5, K5-K6)
    3. Naming convention: explicit pair name (A1-A2, A2-A3, ...)
    """
    
    def __init__(self, allow_gap: int = 2):
        """
        Args:
            allow_gap: Maximum allowed gap in contact numbers.
                       1 = only consecutive (A1-A2)
                       2 = allow one missing (A1-A3 if A2 doesn't exist)
        """
        self.allow_gap = allow_gap
        
    def group_by_shaft(self, ch_names: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """
        Group channels by electrode shaft prefix.
        
        Args:
            ch_names: List of channel names
        
        Returns:
            Dict mapping prefix -> [(number, name), ...] sorted by number
        """
        shaft_groups = {}
        
        for name in ch_names:
            prefix, num = ElectrodeParser.parse(name)
            if prefix is not None:
                if prefix not in shaft_groups:
                    shaft_groups[prefix] = []
                shaft_groups[prefix].append((num, name))
        
        # Sort each group by contact number
        for prefix in shaft_groups:
            shaft_groups[prefix].sort(key=lambda x: x[0])
            
        return shaft_groups
    
    def compute_bipolar(self, data: np.ndarray, ch_names: List[str]
                        ) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
        """
        Compute bipolar referenced data.
        
        Args:
            data: (n_channels, n_samples) array
            ch_names: List of channel names
            
        Returns:
            bipolar_data: (n_bipolar, n_samples) array
            bipolar_names: List of bipolar channel names
            bipolar_pairs: List of (ch1, ch2) pairs that were subtracted
        """
        ch_indices = {name: i for i, name in enumerate(ch_names)}
        shaft_groups = self.group_by_shaft(ch_names)
        
        bipolar_data = []
        bipolar_names = []
        bipolar_pairs = []
        
        for prefix, contacts in shaft_groups.items():
            # contacts is [(num, name), ...] sorted by number
            for i in range(len(contacts) - 1):
                curr_num, curr_name = contacts[i]
                next_num, next_name = contacts[i + 1]
                
                # Check gap between contact numbers
                gap = next_num - curr_num
                
                if gap <= self.allow_gap:
                    idx_curr = ch_indices[curr_name]
                    idx_next = ch_indices[next_name]
                    
                    # Bipolar: proximal - distal (lower - higher number)
                    diff_signal = data[idx_curr] - data[idx_next]
                    
                    bipolar_data.append(diff_signal)
                    curr_clean = ElectrodeParser.clean_name(curr_name)
                    next_clean = ElectrodeParser.clean_name(next_name)
                    bipolar_names.append(f"{curr_clean}-{next_clean}")
                    bipolar_pairs.append((curr_name, next_name))
                else:
                    warnings.warn(
                        f"Skipping pair {curr_name}-{next_name}: gap={gap} > allow_gap={self.allow_gap}"
                    )
        
        if not bipolar_data:
            raise RuntimeError("Bipolar referencing resulted in 0 channels. Check channel naming.")
            
        return np.array(bipolar_data), bipolar_names, bipolar_pairs


class ChannelQualityChecker:
    """
    Assesses channel quality for SEEG data.
    
    Detects:
    - High amplitude outliers (artifacts)
    - Low variance channels (disconnected/dead)
    - High-frequency noise
    """
    
    def __init__(self, 
                 zscore_threshold: float = 5.0,
                 variance_percentile_low: float = 1.0,
                 variance_percentile_high: float = 99.0):
        """
        Args:
            zscore_threshold: Channels with |z-score| > threshold are flagged
            variance_percentile_low: Below this percentile = low variance (dead)
            variance_percentile_high: Above this percentile = high variance (noisy)
        """
        self.zscore_threshold = zscore_threshold
        self.var_pctl_low = variance_percentile_low
        self.var_pctl_high = variance_percentile_high
        
    def check_quality(self, data: np.ndarray, ch_names: List[str]
                      ) -> Tuple[List[str], Dict[str, float]]:
        """
        Check channel quality.
        
        Args:
            data: (n_channels, n_samples)
            ch_names: Channel names
            
        Returns:
            bad_channels: List of channels flagged as bad
            quality_scores: Dict of channel -> score (higher = better, 0-1)
        """
        n_channels = data.shape[0]
        bad_channels = []
        quality_scores = {}
        
        # Compute per-channel statistics
        channel_vars = np.var(data, axis=1)
        channel_means = np.mean(np.abs(data), axis=1)
        
        # Global statistics for z-scoring
        global_var = np.median(channel_vars)
        global_std = np.std(channel_vars)
        
        # Percentile thresholds
        var_low = np.percentile(channel_vars, self.var_pctl_low)
        var_high = np.percentile(channel_vars, self.var_pctl_high)
        
        for i, name in enumerate(ch_names):
            score = 1.0  # Start with perfect score
            reasons = []
            
            # Check 1: Variance z-score
            if global_std > 0:
                z = (channel_vars[i] - global_var) / global_std
                if np.abs(z) > self.zscore_threshold:
                    score *= 0.5
                    reasons.append(f"z-score={z:.1f}")
            
            # Check 2: Low variance (dead channel)
            if channel_vars[i] < var_low:
                score *= 0.3
                reasons.append("low_variance")
            
            # Check 3: High variance (noisy/artifact)
            if channel_vars[i] > var_high:
                score *= 0.5
                reasons.append("high_variance")
            
            # Check 4: Flat line detection (very low variance)
            if channel_vars[i] < 1e-12:
                score = 0.0
                reasons.append("flat_line")
            
            quality_scores[name] = score
            
            if score < 0.5:
                bad_channels.append(name)
                
        return bad_channels, quality_scores


class SEEGPreprocessor:
    """
    Complete preprocessing pipeline for SEEG HFO analysis.
    
    Pipeline:
    1. Load EDF and filter to valid SEEG channels
    2. Apply re-referencing strategy (bipolar, CAR, or none)
    3. Optionally apply explicit channel include/exclude lists
    5. Resample (with Nyquist warnings)
    6. Apply notch and bandpass filters (GPU accelerated if available)
    7. Check channel quality
    
    Re-referencing strategies:
    - 'none': Keep as-is (for pre-bipolar data or monopolar analysis)
    - 'bipolar': Apply bipolar referencing (A1-A2, A2-A3, ...)
    - 'car': Common Average Reference per shaft
    - 'auto': Alias for 'bipolar' (kept for backwards compatibility; no guessing)
    
    Example:
        >>> # Explicitly bipolar-reference (recommended for SEEG analysis)
        >>> preprocessor = SEEGPreprocessor(reference='bipolar')
        >>> result = preprocessor.run('/path/to/file.edf')
        
        >>> # With GPU acceleration
        >>> preprocessor = SEEGPreprocessor(use_gpu=True)
        >>> result = preprocessor.run('/path/to/file.edf')
    """
    
    def __init__(self,
                 target_sfreq: Optional[Union[float, str]] = None,
                 target_band: str = 'ripple',
                 reference: str = 'auto',
                 bipolar_gap: int = 2,
                 include_channels: Optional[List[str]] = None,
                 exclude_channels: Optional[List[str]] = None,
                 notch_freqs: Optional[List[float]] = None,
                 bandpass: Optional[Tuple[float, float]] = None,
                 check_quality: bool = True,
                 crop_seconds: Optional[float] = None,
                 use_gpu: bool = False,
                 gpu_chunk_sec: float = 20.0,
                 use_fif_cache: bool = True,
                 fif_cache_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            target_sfreq: Target sampling rate (Hz), or 'auto'/'none'.
                          If None or 'auto', auto-select based on target_band:
                          - 'ripple': 1000 Hz
                          - 'fast_ripple': 2000 Hz (or keep original if >= 2000)
            target_band: 'ripple' (80-250Hz) or 'fast_ripple' (250-500Hz)
            reference: Re-referencing strategy:
                       - 'none': Keep original (EDF raw; do not guess)
                       - 'bipolar': Apply bipolar referencing
                       - 'car': Common Average Reference per shaft
                       - 'auto': Alias for 'bipolar' (no guessing)
            bipolar_gap: Max gap in contact numbers for bipolar (default=2)
            include_channels: Explicit channel whitelist (after cleaning names). If set,
                              keeps only these channels (intersection, order preserved).
            exclude_channels: Explicit channel blacklist (after cleaning names).
            notch_freqs: Notch filter frequencies (default: 50Hz harmonics)
            bandpass: (low, high) bandpass filter (default: None, applied in detector)
            check_quality: Whether to run quality checks
            crop_seconds: If set, only load first N seconds (for testing/memory)
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.target_band = target_band
        self.reference = reference
        self.bipolar_gap = bipolar_gap
        self.include_channels = include_channels
        self.exclude_channels = exclude_channels
        self.check_quality = check_quality
        self.crop_seconds = crop_seconds
        self.use_gpu = use_gpu and HAS_GPU
        self.gpu_chunk_sec = float(gpu_chunk_sec)
        self.use_fif_cache = bool(use_fif_cache)
        self.fif_cache_dir = Path(fif_cache_dir) if fif_cache_dir is not None else None
        
        if use_gpu and not HAS_GPU:
            warnings.warn("GPU requested but CuPy not available. Using CPU.")
        
        # Initialize filter backend (决定 CPU/GPU 一次,消除运行时判断)
        if self.use_gpu:
            self.filter_backend: FilterBackend = GpuFilterBackend(chunk_sec=self.gpu_chunk_sec)
        else:
            self.filter_backend: FilterBackend = CpuFilterBackend()
        
        self._skip_resample = False
        if isinstance(target_sfreq, str):
            ts = target_sfreq.strip().lower()
            if ts == "none":
                self._skip_resample = True
                self.target_sfreq = None
            elif ts == "auto":
                target_sfreq = None
            else:
                target_sfreq = float(target_sfreq)

        # Auto-select sampling rate based on target band
        if target_sfreq is None and not self._skip_resample:
            if target_band == 'fast_ripple':
                self.target_sfreq = 2000.0  # Keep high for FR
            else:
                self.target_sfreq = 1000.0  # Safe for Ripple
        elif not self._skip_resample:
            self.target_sfreq = float(target_sfreq)

        # Validate Nyquist for target band
        if not self._skip_resample:
            self._validate_nyquist()
        
        # Default notch frequencies (50Hz power line and harmonics)
        if notch_freqs is None:
            self.notch_freqs = [50.0, 100.0, 150.0, 200.0]
        else:
            self.notch_freqs = notch_freqs
            
        self.bandpass = bandpass
        
        # Components
        self.bipolar_ref = BipolarReferencer(allow_gap=bipolar_gap)
        self.car_ref = CommonAverageReferencer()
        self.quality_checker = ChannelQualityChecker()
        
        # State
        self._raw = None
        self._original_sfreq = None
        self._original_ch_names = []
        self._excluded_channels = []
        self._used_gpu = False
        
    def _validate_nyquist(self):
        """Warn if sampling rate is too low for target band."""
        if self.target_band == 'fast_ripple':
            # FR needs up to 500Hz, Nyquist = 1000Hz minimum
            # But for clean signal, we want 2x margin
            if self.target_sfreq < 1200:
                warnings.warn(
                    f"Target sfreq {self.target_sfreq}Hz is marginal for Fast Ripple (250-500Hz). "
                    f"Recommend >= 1500Hz, ideally 2000Hz."
                )
        elif self.target_band == 'ripple':
            # Ripple needs up to 250Hz, Nyquist = 500Hz minimum
            if self.target_sfreq < 600:
                warnings.warn(
                    f"Target sfreq {self.target_sfreq}Hz is too low for Ripple (80-250Hz). "
                    f"Nyquist limit violated!"
                )
    
    def _get_fif_cache_path(self, edf_path: Path) -> Path:
        base = edf_path.stem
        if self.crop_seconds is not None:
            base = f"{base}_crop{int(self.crop_seconds)}s"
        cache_dir = self.fif_cache_dir if self.fif_cache_dir is not None else edf_path.parent
        return Path(cache_dir) / f"{base}.fif"

    def load_edf(self, file_path: Union[str, Path]) -> 'SEEGPreprocessor':
        """
        Load EDF file and filter to valid SEEG channels.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            self (for chaining)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {file_path}")
            
        print(f"Loading: {file_path.name}")

        # Optional: use cached FIF to speed repeated reads
        if self.use_fif_cache:
            fif_path = self._get_fif_cache_path(file_path)
            if fif_path.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._raw = mne.io.read_raw_fif(
                        str(fif_path),
                        preload=True,
                        verbose=False,
                    )
                self._original_sfreq = self._raw.info['sfreq']
                self._original_ch_names = list(self._raw.ch_names)
                print(f"  Loaded cached FIF: {fif_path.name}")
                return self
        
        # First pass: read header to get channel list
        # Use latin1 encoding for Chinese hospital EDF files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False, encoding='latin1')
        
        all_chs = temp_raw.ch_names
        self._original_ch_names = all_chs
        
        # Filter to valid SEEG channels
        keep_chs = [ch for ch in all_chs if ElectrodeParser.is_valid_seeg(ch)]
        
        if not keep_chs:
            raise ValueError(
                f"No valid SEEG channels found. "
                f"Sample channels: {all_chs[:5]}"
            )
        
        # Load only valid channels (initially without preload for cropping)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._raw = mne.io.read_raw_edf(
                file_path, 
                include=keep_chs, 
                preload=False, 
                verbose=False,
                encoding='latin1'
            )
        
        self._original_sfreq = self._raw.info['sfreq']
        total_duration = self._raw.times[-1]
        
        # Crop if requested (for memory management)
        if self.crop_seconds is not None and self.crop_seconds < total_duration:
            self._raw.crop(tmax=self.crop_seconds)
            print(f"  Cropped to first {self.crop_seconds} seconds")
        
        # Now load data into memory
        self._raw.load_data()
        
        # Clean channel names
        rename_map = {ch: ElectrodeParser.clean_name(ch) for ch in self._raw.ch_names}
        self._raw.rename_channels(rename_map)
        
        print(f"  Loaded {len(self._raw.ch_names)} SEEG channels")
        print(f"  Original sfreq: {self._original_sfreq} Hz")
        print(f"  Duration: {self._raw.times[-1]:.1f} s ({self._raw.times[-1]/3600:.2f} h)")

        if self.use_fif_cache:
            try:
                fif_path = self._get_fif_cache_path(file_path)
                fif_path.parent.mkdir(parents=True, exist_ok=True)
                self._raw.save(str(fif_path), overwrite=True, verbose=False)
                print(f"  Saved FIF cache: {fif_path.name}")
            except Exception:
                pass
        
        return self
    
    def run(self, file_path: Union[str, Path]) -> PreprocessingResult:
        """
        Run complete preprocessing pipeline.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            PreprocessingResult containing processed data and metadata
        """
        t_start = time.time()
        logger = get_run_logger(f"preprocess_{Path(file_path).stem}")
        log_section(logger, "PREPROCESS START")
        logger.info("file_path=%s", str(file_path))
        logger.info("target_band=%s", str(self.target_band))
        logger.info("reference=%s", str(self.reference))
        logger.info("target_sfreq=%s", str(self.target_sfreq))
        logger.info("skip_resample=%s", str(self._skip_resample))
        logger.info("crop_seconds=%s", str(self.crop_seconds))
        logger.info("use_gpu=%s", str(self.use_gpu))

        # Step 1: Load EDF
        self.load_edf(file_path)
        
        # Get raw data
        data = self._raw.get_data()  # (n_channels, n_samples)
        ch_names = list(self._raw.ch_names)
        sfreq = self._original_sfreq
        original_ch_count = len(ch_names)
        
        # Step 2: Apply re-referencing (explicit; no guessing)
        reference_type = 'monopolar'
        effective_reference = self.reference
        if effective_reference == 'auto':
            effective_reference = 'bipolar'

        # Step 3: Apply re-referencing
        bipolar_pairs = None
        
        if effective_reference == 'bipolar':
            print("Applying bipolar re-referencing...")
            pre_count = len(ch_names)
            data, ch_names, bipolar_pairs = self.bipolar_ref.compute_bipolar(data, ch_names)
            reference_type = 'bipolar'
            print(f"  {pre_count} -> {len(ch_names)} bipolar channels")
            
        elif effective_reference == 'car':
            print("Applying Common Average Reference per shaft...")
            data, ch_names = self.car_ref.compute_car(data, ch_names)
            reference_type = 'car'
            print(f"  Applied CAR to {len(ch_names)} channels")
            
        elif effective_reference == 'none':
            reference_type = 'monopolar'
            print("Keeping original reference (monopolar)")
        else:
            raise ValueError(f"Unknown reference='{self.reference}'. Use 'none'/'bipolar'/'car'.")

        # Step 4: Apply explicit channel include/exclude lists (post-reference)
        excluded_channels: List[str] = []
        if self.include_channels is not None:
            include_set = set(self.include_channels)
            keep_indices = [i for i, nm in enumerate(ch_names) if nm in include_set]
            excluded_channels.extend([nm for nm in ch_names if nm not in include_set])
            data = data[keep_indices]
            ch_names = [ch_names[i] for i in keep_indices]

        if self.exclude_channels is not None:
            exclude_set = set(self.exclude_channels)
            keep_indices = [i for i, nm in enumerate(ch_names) if nm not in exclude_set]
            excluded_channels.extend([nm for nm in ch_names if nm in exclude_set])
            data = data[keep_indices]
            ch_names = [ch_names[i] for i in keep_indices]

        self._excluded_channels = sorted(set(excluded_channels))
        
        # Step 5: Resample if needed
        if self._skip_resample:
            print(f"Skipping resample (target_sfreq=none); keeping {sfreq} Hz")
        elif sfreq != self.target_sfreq:
            print(f"Resampling: {sfreq} -> {self.target_sfreq} Hz...")
            data = self._resample(data, sfreq, self.target_sfreq)
            sfreq = self.target_sfreq
        else:
            print(f"Sampling rate matches target ({sfreq} Hz), skipping resample")
        
        # Step 6: Filtering (with optional GPU acceleration)
        gpu_status = "(GPU)" if self.use_gpu else "(CPU)"
        print(f"Filtering {gpu_status}: notch={self.notch_freqs}, bandpass={self.bandpass}")
        data = self._apply_filters(data, sfreq)
        self._used_gpu = self.use_gpu
        
        # Step 7: Quality check
        bad_channels = []
        quality_scores = {}
        if self.check_quality:
            print("Checking channel quality...")
            bad_channels, quality_scores = self.quality_checker.check_quality(data, ch_names)
            if bad_channels:
                print(f"  Flagged {len(bad_channels)} potentially bad channels: {bad_channels[:5]}...")
        
        # Build shaft mapping
        shaft_mapping = {}
        for name in ch_names:
            prefix, _ = ElectrodeParser.parse(name)
            if prefix:
                if prefix not in shaft_mapping:
                    shaft_mapping[prefix] = []
                shaft_mapping[prefix].append(name)
        
        print(f"Preprocessing complete: {data.shape[0]} channels, {data.shape[1]/sfreq:.1f}s at {sfreq}Hz")
        log_section(logger, "PREPROCESS SUMMARY")
        logger.info("input_channels=%d", int(original_ch_count))
        logger.info("output_channels=%d", int(data.shape[0]))
        logger.info("reference_type=%s", str(reference_type))
        logger.info("sfreq=%.3f", float(sfreq))
        logger.info("duration_sec=%.3f", float(data.shape[1] / sfreq))
        logger.info("excluded_channels=%d", int(len(set(excluded_channels))))
        logger.info("bad_channels=%d", int(len(bad_channels)))
        logger.info("elapsed_sec=%.3f", float(time.time() - t_start))
        log_section(logger, "PREPROCESS END")
        
        return PreprocessingResult(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            original_ch_names=list(self._raw.ch_names),
            bipolar_pairs=bipolar_pairs,
            shaft_mapping=shaft_mapping,
            bad_channels=bad_channels,
            quality_scores=quality_scores,
            reference_type=reference_type,
            excluded_channels=excluded_channels,
            used_gpu=self._used_gpu
        )
    
    def _resample(self, data: np.ndarray, orig_sfreq: float, target_sfreq: float) -> np.ndarray:
        """Resample data using polyphase filtering."""
        from fractions import Fraction
        
        # Find up/down factors
        frac = Fraction(int(target_sfreq), int(orig_sfreq)).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        
        # Use scipy's resample_poly (handles anti-aliasing)
        from scipy.signal import resample_poly
        resampled = resample_poly(data, up, down, axis=-1)
        
        return resampled
    
    def _apply_filters(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Apply notch and bandpass filters using configured backend."""
        # Notch filter for power line harmonics
        if self.notch_freqs:
            data = self.filter_backend.apply_notch(data, sfreq, self.notch_freqs)
        
        # Bandpass filter (optional - usually applied in detector)
        if self.bandpass is not None:
            low, high = self.bandpass
            data = self.filter_backend.apply_bandpass(data, sfreq, low, high)
        
        return data
    
    def get_shaft_info(self) -> Dict[str, int]:
        """Get number of contacts per electrode shaft."""
        if self._raw is None:
            raise RuntimeError("No data loaded")
        shaft_groups = self.bipolar_ref.group_by_shaft(list(self._raw.ch_names))
        return {k: len(v) for k, v in shaft_groups.items()}


# =============================================================================
# Convenience Functions
# =============================================================================

def save_raw_cache(
    edf_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_prefix: str,
    *,
    reference: str = "none",
    include_channels: Optional[List[str]] = None,
    exclude_channels: Optional[List[str]] = None,
    crop_seconds: Optional[float] = None,
) -> str:
    """
    Save raw waveform cache (no resample, no filtering) for visualization.

    The cache is generated using preprocessing utilities (clean names + reference),
    but skips resampling and filtering to preserve raw waveform.
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    output_dir = Path(output_dir)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{output_prefix}_rawCache_{reference}.npz"

    # Read EDF header for channel list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding="latin1")
    all_chs = temp_raw.ch_names

    # Keep valid SEEG channels
    keep_chs = [ch for ch in all_chs if ElectrodeParser.is_valid_seeg(ch)]
    if not keep_chs:
        raise ValueError("No valid SEEG channels found in EDF.")

    # Load data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(
            edf_path,
            include=keep_chs,
            preload=False,
            verbose=False,
            encoding="latin1",
        )

    if crop_seconds is not None and crop_seconds < raw.times[-1]:
        raw.crop(tmax=crop_seconds)
    raw.load_data()

    # Clean channel names
    rename_map = {ch: ElectrodeParser.clean_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    data = raw.get_data()
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    # Apply explicit channel include/exclude after cleaning
    excluded_channels: List[str] = []
    if include_channels is not None:
        include_set = set(include_channels)
        keep_indices = [i for i, nm in enumerate(ch_names) if nm in include_set]
        excluded_channels.extend([nm for nm in ch_names if nm not in include_set])
        data = data[keep_indices]
        ch_names = [ch_names[i] for i in keep_indices]
    if exclude_channels is not None:
        exclude_set = set(exclude_channels)
        keep_indices = [i for i, nm in enumerate(ch_names) if nm not in exclude_set]
        excluded_channels.extend([nm for nm in ch_names if nm in exclude_set])
        data = data[keep_indices]
        ch_names = [ch_names[i] for i in keep_indices]

    # Apply reference (no filtering/resampling)
    reference_type = "monopolar"
    if reference == "bipolar":
        ref = BipolarReferencer(allow_gap=2)
        data, ch_names, _ = ref.compute_bipolar(data, ch_names)
        reference_type = "bipolar"
    elif reference == "car":
        ref = CommonAverageReferencer()
        data, ch_names = ref.compute_car(data, ch_names)
        reference_type = "car"
    elif reference == "none":
        reference_type = "monopolar"
    else:
        raise ValueError("reference must be 'none'|'bipolar'|'car'")

    np.savez_compressed(
        out_path,
        data=data.astype(np.float32, copy=False),
        sfreq=np.array([sfreq], dtype=np.float64),
        ch_names=np.array(ch_names),
        reference_type=np.array([reference_type]),
        original_ch_names=np.array(all_chs),
        excluded_channels=np.array(sorted(set(excluded_channels))),
    )
    return str(out_path)


def load_raw_cache(cache_npz_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load raw cache saved by save_raw_cache()."""
    meta = np.load(str(cache_npz_path), allow_pickle=True)
    return {k: meta[k] for k in meta.files}


def preprocess_edf(file_path: Union[str, Path],
                   target_band: str = 'ripple',
                   notch_freqs: Optional[List[float]] = None
                   ) -> Tuple[np.ndarray, float, List[str]]:
    """
    Convenience function for quick preprocessing.
    
    Args:
        file_path: Path to EDF file
        target_band: 'ripple' or 'fast_ripple'
        notch_freqs: Notch filter frequencies
        
    Returns:
        data: (n_channels, n_samples) preprocessed data
        sfreq: Sampling frequency
        ch_names: Channel names
    """
    preprocessor = SEEGPreprocessor(
        target_band=target_band,
        notch_freqs=notch_freqs
    )
    result = preprocessor.run(file_path)
    return result.data, result.sfreq, result.ch_names




# =============================================================================
# Main / Test
# =============================================================================

if __name__ == '__main__':
    # Minimal smoke test (local)
    edf_path = '/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf'
    print("=" * 70)
    print("Smoke test: SEEGPreprocessor (bipolar)")
    print("=" * 70)
    pre = SEEGPreprocessor(reference='bipolar', crop_seconds=10)
    out = pre.run(edf_path)
    print(f"OK: {out.data.shape} @ {out.sfreq}Hz, ref={out.reference_type}")
