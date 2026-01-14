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

import numpy as np
import mne
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy import signal

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


class PreBipolarDetector:
    """
    Deprecated.
    
    We do NOT guess whether EDF data is already bipolar referenced.
    The Yuquan *_gpu.npz channel list differs from EDF mainly because GPU-side
    processing selected a subset of contacts (e.g., often dropping distal contacts),
    which is a channel-selection decision, not proof of bipolar referencing.
    
    Keep this class only for backwards compatibility; it always returns False.
    """
    
    @staticmethod
    def detect(ch_names: List[str], expected_contacts_per_shaft: Optional[Dict[str, int]] = None
               ) -> Tuple[bool, str]:
        return False, "PreBipolarDetector is deprecated; use explicit reference='bipolar'/'car'/'none'."


class BipolarReferencer:
    """
    Applies bipolar re-referencing to SEEG data.
    
    Critical rules:
    1. Only subtract adjacent contacts on the SAME shaft (same prefix)
    2. Handle non-consecutive numbering (e.g., K3, K5, K6 -> K3-K5, K5-K6)
    3. Naming convention: explicit pair name (A1-A2, A2-A3, ...)
    """
    
    def __init__(self, allow_gap: int = 2, exclude_last_n: int = 0):
        """
        Args:
            allow_gap: Maximum allowed gap in contact numbers.
                       1 = only consecutive (A1-A2)
                       2 = allow one missing (A1-A3 if A2 doesn't exist)
            exclude_last_n: Deprecated; do not bake dataset-specific contact dropping
                            into referencing. Prefer explicit include_channels/exclude_channels
                            in SEEGPreprocessor if you need to match an external channel list.
        """
        self.allow_gap = allow_gap
        self.exclude_last_n = exclude_last_n
        
    def group_by_shaft(self, ch_names: List[str], 
                       exclude_last_n: Optional[int] = None) -> Dict[str, List[Tuple[int, str]]]:
        """
        Group channels by electrode shaft prefix.
        
        Args:
            ch_names: List of channel names
            exclude_last_n: Override instance setting for excluding last N contacts
        
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
        
        # Exclude last N contacts if requested
        n_exclude = exclude_last_n if exclude_last_n is not None else self.exclude_last_n
        if n_exclude > 0:
            for prefix in shaft_groups:
                if len(shaft_groups[prefix]) > n_exclude:
                    shaft_groups[prefix] = shaft_groups[prefix][:-n_exclude]
            
        return shaft_groups
    
    def compute_bipolar(self, data: np.ndarray, ch_names: List[str],
                        exclude_last_n: Optional[int] = None
                        ) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
        """
        Compute bipolar referenced data.
        
        Args:
            data: (n_channels, n_samples) array
            ch_names: List of channel names
            exclude_last_n: Override instance setting for excluding last N contacts
            
        Returns:
            bipolar_data: (n_bipolar, n_samples) array
            bipolar_names: List of bipolar channel names
            bipolar_pairs: List of (ch1, ch2) pairs that were subtracted
        """
        ch_indices = {name: i for i, name in enumerate(ch_names)}
        shaft_groups = self.group_by_shaft(ch_names, exclude_last_n)
        
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
                 target_sfreq: Optional[float] = None,
                 target_band: str = 'ripple',
                 reference: str = 'auto',
                 bipolar_gap: int = 2,
                 include_channels: Optional[List[str]] = None,
                 exclude_channels: Optional[List[str]] = None,
                 exclude_last_n: int = 0,
                 notch_freqs: Optional[List[float]] = None,
                 bandpass: Optional[Tuple[float, float]] = None,
                 check_quality: bool = True,
                 crop_seconds: Optional[float] = None,
                 use_gpu: bool = False):
        """
        Args:
            target_sfreq: Target sampling rate. If None, auto-select based on target_band:
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
            exclude_last_n: Deprecated. Avoid dataset-specific "drop last N contacts" rules.
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
        self.exclude_last_n = exclude_last_n
        self.check_quality = check_quality
        self.crop_seconds = crop_seconds
        self.use_gpu = use_gpu and HAS_GPU
        
        if use_gpu and not HAS_GPU:
            warnings.warn("GPU requested but CuPy not available. Using CPU.")
        
        # Auto-select sampling rate based on target band
        if target_sfreq is None:
            if target_band == 'fast_ripple':
                self.target_sfreq = 2000.0  # Keep high for FR
            else:
                self.target_sfreq = 1000.0  # Safe for Ripple
        else:
            self.target_sfreq = float(target_sfreq)
            
        # Validate Nyquist for target band
        self._validate_nyquist()
        
        # Default notch frequencies (50Hz power line and harmonics)
        if notch_freqs is None:
            self.notch_freqs = [50.0, 100.0, 150.0, 200.0]
        else:
            self.notch_freqs = notch_freqs
            
        self.bandpass = bandpass
        
        # Components
        self.bipolar_ref = BipolarReferencer(allow_gap=bipolar_gap, exclude_last_n=0)
        self.car_ref = CommonAverageReferencer()
        # Deprecated detector kept for backwards compatibility only; we don't use it.
        self.prebipolar_detector = PreBipolarDetector()
        self.quality_checker = ChannelQualityChecker()
        
        # State
        self._raw = None
        self._original_sfreq = None
        self._original_ch_names = []
        self._excluded_channels = []
        self._is_pre_bipolar = False  # deprecated field; kept for compatibility
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
        
        return self
    
    def run(self, file_path: Union[str, Path]) -> PreprocessingResult:
        """
        Run complete preprocessing pipeline.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            PreprocessingResult containing processed data and metadata
        """
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

        # Step 4b: Deprecated distal-contact dropping (kept only for backwards compatibility)
        if self.exclude_last_n > 0:
            print(f"[DEPRECATED] Excluding last {self.exclude_last_n} contacts per shaft...")
            shaft_groups = self.bipolar_ref.group_by_shaft(ch_names, exclude_last_n=0)
            keep_names = set()
            for _, contacts in shaft_groups.items():
                keep_contacts = contacts[:-self.exclude_last_n] if len(contacts) > self.exclude_last_n else contacts
                for _, name in keep_contacts:
                    keep_names.add(name)
            keep_indices = [i for i, name in enumerate(ch_names) if name in keep_names]
            excluded_channels.extend([name for name in ch_names if name not in keep_names])
            data = data[keep_indices]
            ch_names = [ch_names[i] for i in keep_indices]
            print(f"  {original_ch_count} -> {len(ch_names)} channels (excluded {len(set(excluded_channels))})")

        self._excluded_channels = sorted(set(excluded_channels))
        
        # Step 5: Resample if needed
        if sfreq != self.target_sfreq:
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
        """Apply notch and bandpass filters with optional GPU acceleration."""
        
        if self.use_gpu and HAS_GPU:
            return self._apply_filters_gpu(data, sfreq)
        else:
            return self._apply_filters_cpu(data, sfreq)
    
    def _apply_filters_cpu(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Apply filters on CPU."""
        # Notch filter for power line harmonics
        if self.notch_freqs:
            for freq in self.notch_freqs:
                if freq < sfreq / 2:  # Must be below Nyquist
                    Q = 30  # Quality factor
                    b, a = iirnotch(freq, Q, sfreq)
                    data = filtfilt(b, a, data, axis=-1)
        
        # Bandpass filter (optional - usually applied in detector)
        if self.bandpass is not None:
            low, high = self.bandpass
            nyq = sfreq / 2
            
            if high >= nyq:
                warnings.warn(f"Bandpass high ({high}Hz) >= Nyquist ({nyq}Hz), clipping to {nyq-1}Hz")
                high = nyq - 1
                
            sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
            data = sosfiltfilt(sos, data, axis=-1)
        
        return data
    
    def _apply_filters_gpu(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Apply filters on GPU using CuPy."""
        if not HAS_GPU:
            return self._apply_filters_cpu(data, sfreq)
        
        # Transfer to GPU
        data_gpu = cp.asarray(data)

        def _sosfiltfilt_reflect_gpu(x_gpu, sos_gpu, pad_len: int):
            """
            Approximate filtfilt on GPU via reflect padding + forward/backward sosfilt.
            This avoids the worst edge transients from naive forward->flip->forward.
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
        
        # Notch filter - use forward-only filter on GPU (no filtfilt in cupy)
        # We use reflect padding + forward/backward to suppress edge transients.
        if self.notch_freqs:
            for freq in self.notch_freqs:
                if freq < sfreq / 2:
                    Q = 30
                    b, a = iirnotch(freq, Q, sfreq)
                    # Convert to SOS for stability
                    from scipy.signal import tf2sos
                    sos = tf2sos(b, a)
                    sos_gpu = cp.asarray(sos)
                    # Heuristic pad length: similar to scipy filtfilt default (3*(ntaps-1)),
                    # but we don't have direct ntaps for SOS; be conservative in seconds.
                    pad_len = int(min(data_gpu.shape[1] // 4, max(1, round(0.5 * sfreq))))
                    data_gpu = _sosfiltfilt_reflect_gpu(data_gpu, sos_gpu, pad_len)
        
        # Bandpass filter
        if self.bandpass is not None:
            low, high = self.bandpass
            nyq = sfreq / 2
            
            if high >= nyq:
                high = nyq - 1
                
            sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
            sos_gpu = cp.asarray(sos)
            pad_len = int(min(data_gpu.shape[1] // 4, max(1, round(0.5 * sfreq))))
            data_gpu = _sosfiltfilt_reflect_gpu(data_gpu, sos_gpu, pad_len)
        
        # Transfer back to CPU
        return cp.asnumpy(data_gpu)
    
    def get_shaft_info(self) -> Dict[str, int]:
        """Get number of contacts per electrode shaft."""
        if self._raw is None:
            raise RuntimeError("No data loaded")
        shaft_groups = self.bipolar_ref.group_by_shaft(list(self._raw.ch_names))
        return {k: len(v) for k, v in shaft_groups.items()}


# =============================================================================
# Convenience Functions
# =============================================================================

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


def validate_against_gpu_results(result: PreprocessingResult, 
                                  gpu_ch_names: np.ndarray) -> Dict:
    """
    Deprecated.
    
    Do not infer referencing/channel-dropping rules from *_gpu.npz.
    If you need to match GPU channel selection, pass include_channels explicitly:
      include_channels=[str(n) for n in gpu['chns_names']]
    """
    raise RuntimeError(
        "validate_against_gpu_results() is deprecated. "
        "Use include_channels/exclude_channels for explicit channel selection."
    )


# =============================================================================
# Main / Test
# =============================================================================

if __name__ == '__main__':
    # Minimal smoke test (local)
    edf_path = '/Volumes/Elements/yuquan_24h_edf/chengshuai/FC10477Q.edf'
    print("=" * 70)
    print("Smoke test: SEEGPreprocessor (bipolar)")
    print("=" * 70)
    pre = SEEGPreprocessor(reference='bipolar', crop_seconds=10)
    out = pre.run(edf_path)
    print(f"OK: {out.data.shape} @ {out.sfreq}Hz, ref={out.reference_type}")
