"""What the sustained high-activity state actually is.

The project's 2026-08-08 finding was that this engine's sustained regime is a
burst train re-igniting from population silence, not a continuous carrier. That
was measured at a different work point, so it is context, not a result of this
round; these functions recompute it on THIS round's trajectories. Reporting the
transition without reporting the morphology of the state entered invites the
figure to be read as "a seizure was reproduced".
"""
from __future__ import annotations

import numpy as np


def _runs(mask):
    """Start/stop index pairs of each True run."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.empty((0, 2), int)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2)


def _slice(values, dt_ms, window_ms):
    lo = int(round(window_ms[0] / dt_ms))
    hi = int(round(window_ms[1] / dt_ms))
    return np.asarray(values, float)[lo:hi]


def characterize_state(rate_E_hz, *, dt_ms, window_ms, silence_threshold_hz,
                       zero_window_ms=20.0):
    rate = _slice(rate_E_hz, dt_ms, window_ms)
    active = rate > float(silence_threshold_hz)
    active_runs = _runs(active)
    silent_runs = _runs(~active)
    onsets = active_runs[:, 0] * dt_ms if len(active_runs) else np.empty(0)
    intervals = np.diff(onsets) if len(onsets) > 1 else np.empty(0)
    window_steps = max(1, int(round(zero_window_ms / dt_ms)))
    n_windows = len(rate) // window_steps
    trimmed = rate[:n_windows * window_steps].reshape(n_windows, window_steps)
    return {
        "active_durations_ms": (active_runs[:, 1] - active_runs[:, 0]) * dt_ms,
        "silent_durations_ms": (silent_runs[:, 1] - silent_runs[:, 0]) * dt_ms,
        "n_bursts": int(len(active_runs)),
        "burst_interval_ms": float(np.median(intervals)) if len(intervals) else float("nan"),
        "reignition_rate_hz": (1000.0 / float(np.median(intervals))
                               if len(intervals) else 0.0),
        "zero_spike_window_fraction": float(np.mean(trimmed.max(axis=1) <= 0.0))
                                      if n_windows else float("nan"),
        "peak_rate_hz": float(rate.max()) if len(rate) else float("nan"),
        "median_rate_hz": float(np.median(rate)) if len(rate) else float("nan"),
        "mean_rate_hz": float(rate.mean()) if len(rate) else float("nan"),
        "window_ms": tuple(float(v) for v in window_ms),
    }


def interictal_reference(rate_E_hz, *, dt_ms, window_ms):
    """Length-matched interictal comparison window.

    Every state statistic is compared against a window of the SAME length, so a
    difference cannot be an artefact of estimating one number over 500 ms and
    the other over 20 s.
    """
    rate = _slice(rate_E_hz, dt_ms, window_ms)
    return {
        "window_ms": tuple(float(v) for v in window_ms),
        "n_steps": int(len(rate)),
        "median_rate_hz": float(np.median(rate)) if len(rate) else float("nan"),
        "percentile_95_hz": float(np.percentile(rate, 95)) if len(rate) else float("nan"),
        "percentile_99_hz": float(np.percentile(rate, 99)) if len(rate) else float("nan"),
        "max_rate_hz": float(rate.max()) if len(rate) else float("nan"),
    }


def band_proxy(rate_E_hz, *, dt_ms, band_hz=(30.0, 80.0)):
    """In-band power proxy, with its own resolution limit reported.

    500 ms gives ~15 cycles and ~2 Hz resolution at 30 Hz. Those numbers travel
    with the estimate so nobody reads a band-power difference as a spectral
    finding it cannot support.
    """
    rate = np.asarray(rate_E_hz, float)
    n = len(rate)
    if n < 8:
        raise ValueError("band proxy needs at least 8 samples")
    window_ms = n * dt_ms
    windowed = (rate - rate.mean()) * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt_ms * 1e-3)
    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    band_power = float(spectrum[in_band].sum())
    peak = float(freqs[in_band][int(np.argmax(spectrum[in_band]))]) if in_band.any() else float("nan")
    return {
        "band_hz": tuple(float(v) for v in band_hz),
        "band_power": band_power,
        "total_power": float(spectrum.sum()),
        "band_power_fraction": band_power / float(spectrum.sum()) if spectrum.sum() else float("nan"),
        "peak_frequency_hz": peak,
        "window_ms": float(window_ms),
        "frequency_resolution_hz": 1000.0 / window_ms,
        "n_cycles_at_band_low": band_hz[0] * window_ms / 1000.0,
    }
