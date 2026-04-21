"""PR-6-A Step 2 TDD tests for ictal onset extraction primitives.

Step 2 scope: ER (gamma + broad band-pair configurations) and baseline
z-score normalization. Page-Hinkley CUSUM, tie/unreached flagging,
C_common re-rank, and dual-cohort gating are added in subsequent steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ictal_onset_extraction import (
    GAMMA_ER_BANDS,
    BROAD_ER_BANDS,
    baseline_zscore_er,
    compute_er,
)


def _synth_signal_with_burst(
    fs: float,
    duration_sec: float,
    n_channels: int,
    burst_start_sec: list[float],
    burst_freq_hz: float = 80.0,
    burst_dur_sec: float = 1.5,
    burst_amplitude: float = 5.0,
    background_gamma_amp: list[float] | None = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """Build a multi-channel signal with optional per-channel burst injection.

    Each channel is white-noise baseline. ``background_gamma_amp[ch]``
    injects a *persistent* low-amplitude oscillation at ``burst_freq_hz``
    across the entire trace; this is the only knob that produces a real
    per-channel identity bias in ER (energy ratio is scale-invariant, so
    multiplicative noise gain alone cannot do it). A transient burst at
    ``burst_freq_hz`` is added at ``burst_start_sec[ch]``; if the start
    time is NaN/None the channel keeps the (background + noise) trace
    only.
    """

    rng = np.random.default_rng(rng_seed)
    n_samples = int(round(fs * duration_sec))
    t = np.arange(n_samples) / fs
    if background_gamma_amp is None:
        background_gamma_amp = [0.0] * n_channels
    sig = np.empty((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        noise = rng.standard_normal(n_samples)
        bg = background_gamma_amp[ch] * np.sin(2.0 * np.pi * burst_freq_hz * t)
        sig[ch] = noise + bg
        bs = burst_start_sec[ch]
        if bs is None or (isinstance(bs, float) and np.isnan(bs)):
            continue
        i0 = int(round(bs * fs))
        i1 = i0 + int(round(burst_dur_sec * fs))
        i1 = min(i1, n_samples)
        if i0 >= n_samples:
            continue
        burst = burst_amplitude * np.sin(2.0 * np.pi * burst_freq_hz * t[i0:i1])
        sig[ch, i0:i1] += burst
    return sig


# ---------------------------------------------------------------------------
# T1 — ER channel-independent ranks earlier-burst before later-burst.
# Required for both gamma_ER (60-100 / 4-20) and broad_ER (12-127 / 4-20).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band_key,bands", [("gamma_ER", GAMMA_ER_BANDS), ("broad_ER", BROAD_ER_BANDS)])
def test_er_channel_independence(band_key: str, bands: dict) -> None:
    fs = 1000.0
    duration = 8.0
    win_sec = 1.0
    hop_sec = 0.1
    burst_starts = [2.0, 5.0]
    sig = _synth_signal_with_burst(
        fs=fs,
        duration_sec=duration,
        n_channels=2,
        burst_start_sec=burst_starts,
        burst_freq_hz=80.0,
        burst_dur_sec=1.5,
        burst_amplitude=8.0,
    )
    er = compute_er(
        sig, fs=fs, fast_band=bands["fast"], slow_band=bands["slow"],
        win_sec=win_sec, hop_sec=hop_sec,
    )
    assert er.shape[0] == 2
    expected_frames = (sig.shape[1] - int(round(win_sec * fs))) // int(round(hop_sec * fs)) + 1
    assert er.shape[1] == expected_frames

    nd_ch0 = int(np.argmax(er[0]))
    nd_ch1 = int(np.argmax(er[1]))
    assert nd_ch0 < nd_ch1, (
        f"{band_key}: expected ch0 ER peak before ch1 (ch0_idx={nd_ch0}, ch1_idx={nd_ch1})"
    )

    def _frame_to_time(i: int) -> float:
        return i * hop_sec + win_sec / 2.0

    nd_ch0_t = _frame_to_time(nd_ch0)
    nd_ch1_t = _frame_to_time(nd_ch1)
    assert abs(nd_ch0_t - burst_starts[0]) < 1.5, (
        f"{band_key}: ch0 peak time {nd_ch0_t:.2f}s far from {burst_starts[0]}s"
    )
    assert abs(nd_ch1_t - burst_starts[1]) < 1.5, (
        f"{band_key}: ch1 peak time {nd_ch1_t:.2f}s far from {burst_starts[1]}s"
    )


# ---------------------------------------------------------------------------
# T2 — baseline z-score removes channel-level identity bias (constant ER offset).
# ---------------------------------------------------------------------------


def test_baseline_zscore_removes_identity_bias() -> None:
    fs = 1000.0
    duration = 80.0
    win_sec = 1.0
    hop_sec = 0.1
    bursts_at = [70.0, 70.0]
    sig = _synth_signal_with_burst(
        fs=fs,
        duration_sec=duration,
        n_channels=2,
        burst_start_sec=bursts_at,
        burst_freq_hz=80.0,
        burst_dur_sec=1.5,
        burst_amplitude=6.0,
        background_gamma_amp=[1.0, 0.0],
    )
    er = compute_er(
        sig, fs=fs, fast_band=GAMMA_ER_BANDS["fast"], slow_band=GAMMA_ER_BANDS["slow"],
        win_sec=win_sec, hop_sec=hop_sec,
    )

    burst_first_idx = int(round((bursts_at[0] - win_sec / 2.0) / hop_sec))
    baseline_window = (0, max(1, burst_first_idx - 50))

    raw_baseline_mean = er[:, baseline_window[0] : baseline_window[1]].mean(axis=1)
    assert raw_baseline_mean[0] != pytest.approx(raw_baseline_mean[1], abs=0.05), (
        "Synthetic baseline offsets must produce different raw ER baselines; got "
        f"{raw_baseline_mean}"
    )

    z = baseline_zscore_er(er, baseline_idx_window=baseline_window, hop_sec=hop_sec)
    assert z.shape == er.shape
    assert not np.isnan(z).any(), "z-ER must not be all-NaN when baseline >= 60s"

    z_baseline_mean = z[:, baseline_window[0] : baseline_window[1]].mean(axis=1)
    assert np.all(np.abs(z_baseline_mean) < 1e-6), (
        f"z-score baseline mean must be ~0, got {z_baseline_mean}"
    )

    nd_z_ch0 = int(np.argmax(z[0]))
    nd_z_ch1 = int(np.argmax(z[1]))
    delta_z = abs(nd_z_ch0 - nd_z_ch1)
    assert delta_z <= 5, (
        f"After z-score, both channels must peak near the same hop "
        f"(nd_ch0={nd_z_ch0}, nd_ch1={nd_z_ch1})"
    )

    def _frame_to_time(i: int) -> float:
        return i * hop_sec + win_sec / 2.0

    peak_t_ch0 = _frame_to_time(nd_z_ch0)
    assert abs(peak_t_ch0 - bursts_at[0]) < 1.5


def test_baseline_zscore_marks_short_baseline_as_nan() -> None:
    fs = 1000.0
    duration = 4.0
    win_sec = 1.0
    hop_sec = 0.1
    sig = _synth_signal_with_burst(
        fs=fs, duration_sec=duration, n_channels=2,
        burst_start_sec=[2.0, 2.5], burst_amplitude=5.0,
    )
    er = compute_er(
        sig, fs=fs, fast_band=GAMMA_ER_BANDS["fast"], slow_band=GAMMA_ER_BANDS["slow"],
        win_sec=win_sec, hop_sec=hop_sec,
    )
    z = baseline_zscore_er(er, baseline_idx_window=(0, 5), hop_sec=hop_sec)
    assert np.isnan(z).all(), "Below-min-baseline window must produce all-NaN z-ER"


def test_compute_er_rejects_bands_above_nyquist() -> None:
    fs = 200.0
    sig = np.random.default_rng(0).standard_normal((2, int(fs * 4)))
    with pytest.raises(ValueError):
        compute_er(sig, fs=fs, fast_band=(60.0, 150.0), slow_band=(4.0, 20.0))
