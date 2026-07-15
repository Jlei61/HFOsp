"""Shared spectrogram kernel for Figure 1 Panels a1 and a2."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram


ALGORITHM_ID = "fig1_gaussian_smoothed_magnitude_v1"


def compute_smoothed_magnitude_spectrogram(
    signal: np.ndarray,
    fs: float,
    *,
    window: str = "hann",
    window_sec: float = 0.18,
    overlap_sec: float = 0.16,
    freq_range_hz: tuple[float, float] = (0.0, 240.0),
    gaussian_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the common Figure 1 magnitude/Gaussian spectrogram quantity."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"signal must be 1D, got {values.shape}")
    nperseg = int(round(float(window_sec) * float(fs)))
    noverlap = int(round(float(overlap_sec) * float(fs)))
    if nperseg < 2 or noverlap < 0 or noverlap >= nperseg:
        raise ValueError(
            f"invalid spectrogram window/overlap: nperseg={nperseg}, noverlap={noverlap}"
        )
    freqs, times, magnitude = spectrogram(
        values,
        fs=float(fs),
        window=str(window),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        mode="magnitude",
    )
    low, high = map(float, freq_range_hz)
    fmask = (freqs >= low) & (freqs <= high)
    if not np.any(fmask):
        raise ValueError(f"no spectrogram bins in requested range {freq_range_hz}")
    smoothed = gaussian_filter(magnitude[fmask], sigma=float(gaussian_sigma))
    return freqs[fmask], times, np.maximum(smoothed, 0.0)
