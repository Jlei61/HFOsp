"""PR-6-A Step 2 TDD tests for ictal onset extraction primitives.

Step 2 scope: ER (gamma + broad band-pair configurations) and baseline
z-score normalization. Page-Hinkley CUSUM, tie/unreached flagging,
C_common re-rank, and dual-cohort gating are added in subsequent steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ictal_onset_extraction import (
    BROAD_ER_BANDS,
    GAMMA_ER_BANDS,
    MIN_BASELINE_VALID_SEC,
    baseline_zscore_er,
    compute_er,
    detect_er_onset_preview,
    preview_threshold_from_baseline,
    resolve_detection_window,
    resolve_baseline_window,
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


# ---------------------------------------------------------------------------
# T_baseline_clip — EEG-onset-aware baseline window resolver
# ---------------------------------------------------------------------------


def test_baseline_window_uses_clinical_only_when_no_eeg_onset() -> None:
    """When eeg_onset is missing, baseline ends at clin_onset - buffer (legacy contract)."""

    n_t = int(round((300 + 30) / 0.1))
    win = resolve_baseline_window(
        n_t,
        hop_sec=0.1,
        pre_sec=300.0,
        buffer_sec=60.0,
        eeg_onset_rel_sec=None,
    )
    assert win.valid is True
    assert win.clipped_by_eeg_onset is False
    assert win.end_sec == pytest.approx(-60.0, abs=0.05)
    assert win.start_sec == pytest.approx(-300.0, abs=0.05)
    assert win.valid_sec == pytest.approx(240.0, abs=0.1)


def test_baseline_window_no_clip_when_eeg_at_or_after_clin() -> None:
    """eeg_onset later than (or equal to) clin_onset must not pull baseline back."""

    n_t = int(round((300 + 30) / 0.1))
    win = resolve_baseline_window(
        n_t,
        hop_sec=0.1,
        pre_sec=300.0,
        buffer_sec=60.0,
        eeg_onset_rel_sec=5.0,
    )
    assert win.valid is True
    assert win.clipped_by_eeg_onset is False
    assert win.end_sec == pytest.approx(-60.0, abs=0.05)


def test_baseline_window_clips_to_eeg_onset_when_earlier() -> None:
    """eeg_onset earlier than clin_onset must move baseline end back by buffer_sec."""

    n_t = int(round((300 + 30) / 0.1))
    win = resolve_baseline_window(
        n_t,
        hop_sec=0.1,
        pre_sec=300.0,
        buffer_sec=60.0,
        eeg_onset_rel_sec=-100.0,
    )
    assert win.valid is True
    assert win.clipped_by_eeg_onset is True
    assert win.end_sec == pytest.approx(-160.0, abs=0.1)
    assert win.start_sec == pytest.approx(-300.0, abs=0.1)
    assert win.valid_sec == pytest.approx(140.0, abs=0.2)


def test_baseline_window_marks_invalid_when_clipped_below_min_and_does_not_fall_back() -> None:
    """If EEG-aware clip leaves <60s of baseline, return invalid; do NOT silently
    fall back to the legacy clinical window (that would re-introduce the
    pre-ictal contamination this fix exists to remove)."""

    n_t = int(round((300 + 30) / 0.1))
    win = resolve_baseline_window(
        n_t,
        hop_sec=0.1,
        pre_sec=300.0,
        buffer_sec=60.0,
        eeg_onset_rel_sec=-220.0,
        min_baseline_valid_sec=MIN_BASELINE_VALID_SEC,
    )
    assert win.valid is False
    assert win.clipped_by_eeg_onset is True
    assert win.valid_sec < MIN_BASELINE_VALID_SEC
    assert win.end_sec == pytest.approx(-280.0, abs=0.2)
    assert win.end_sec < -60.0 - 1e-6, "Must not fall back to legacy clinical baseline edge"


def test_baseline_zscore_uses_clipped_window_and_drops_pre_ictal_offset() -> None:
    """End-to-end: a per-channel pre-ictal ER bump that lives only in [-200, -60]
    must be excluded from the baseline z-score statistics once eeg_onset_rel_sec
    is supplied. Channel 1 has the bump; channel 0 is clean. Without clipping,
    channel 1's z-ER baseline mean would drift positive; with EEG-aware
    clipping it must stay near 0."""

    fs = 1000.0
    duration = 360.0
    win_sec = 1.0
    hop_sec = 0.1
    pre_sec = 300.0
    buffer_sec = 60.0
    eeg_onset_rel_sec = -150.0

    rng = np.random.default_rng(0)
    n_samples = int(round(fs * duration))
    t = np.arange(n_samples) / fs

    sig = rng.standard_normal((2, n_samples))
    pre_ictal_mask = (t >= 100.0) & (t < 240.0)
    bump = 2.0 * np.sin(2.0 * np.pi * 80.0 * t)
    sig[1, pre_ictal_mask] += bump[pre_ictal_mask]

    er = compute_er(
        sig, fs=fs,
        fast_band=GAMMA_ER_BANDS["fast"], slow_band=GAMMA_ER_BANDS["slow"],
        win_sec=win_sec, hop_sec=hop_sec,
    )

    clipped = resolve_baseline_window(
        er.shape[1], hop_sec=hop_sec, pre_sec=pre_sec, buffer_sec=buffer_sec,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
    )
    legacy = resolve_baseline_window(
        er.shape[1], hop_sec=hop_sec, pre_sec=pre_sec, buffer_sec=buffer_sec,
        eeg_onset_rel_sec=None,
    )
    assert clipped.valid and legacy.valid
    assert clipped.end_sec < legacy.end_sec - 1.0

    z_clip = baseline_zscore_er(
        er, baseline_idx_window=(clipped.start_idx, clipped.end_idx),
        hop_sec=hop_sec,
    )
    z_legacy = baseline_zscore_er(
        er, baseline_idx_window=(legacy.start_idx, legacy.end_idx),
        hop_sec=hop_sec,
    )

    bl_clip = z_clip[1, clipped.start_idx:clipped.end_idx]
    bl_legacy = z_legacy[1, legacy.start_idx:legacy.end_idx]
    assert abs(np.nanmean(bl_clip)) < 0.05, (
        f"clipped baseline mean must be ~0 for biased channel, got {np.nanmean(bl_clip)}"
    )
    assert np.nanstd(bl_legacy) < np.nanstd(z_clip[1]), (
        "legacy baseline std underestimates true ER variability when pre-ictal bump leaks in"
    )


# ---------------------------------------------------------------------------
# T_detection_preview — pre-clinical ER onset preview before formal Step 3.
# ---------------------------------------------------------------------------


def test_detection_window_starts_at_later_of_baseline_end_and_start_floor() -> None:
    n_t = int(round((300.0 + 30.0) / 0.1))
    win = resolve_detection_window(
        n_t,
        hop_sec=0.1,
        pre_sec=300.0,
        baseline_end_sec=-80.0,
        start_floor_sec=-120.0,
        end_sec=30.0,
    )
    assert win.start_sec == pytest.approx(-80.0, abs=0.05)
    assert win.end_sec == pytest.approx(30.0, abs=0.05)
    assert win.valid is True

    win2 = resolve_detection_window(
        n_t,
        hop_sec=0.1,
        pre_sec=300.0,
        baseline_end_sec=-160.0,
        start_floor_sec=-120.0,
        end_sec=30.0,
    )
    assert win2.start_sec == pytest.approx(-120.0, abs=0.05)
    assert win2.end_sec == pytest.approx(30.0, abs=0.05)
    assert win2.valid is True


def test_detect_er_onset_preview_returns_first_preclinical_crossing_and_none_when_unreached() -> None:
    hop_sec = 0.1
    pre_sec = 300.0
    n_t = int(round((pre_sec + 30.0) / hop_sec))
    t_axis = (np.arange(n_t) * hop_sec) - pre_sec

    det_win = resolve_detection_window(
        n_t,
        hop_sec=hop_sec,
        pre_sec=pre_sec,
        baseline_end_sec=-120.0,
        start_floor_sec=-120.0,
        end_sec=30.0,
    )

    z = np.zeros(n_t, dtype=float)
    rise_mask = t_axis >= -40.0
    z[rise_mask] = 2.0
    hit = detect_er_onset_preview(
        z,
        t_axis,
        detection_idx_window=(det_win.start_idx, det_win.end_idx),
        bias=0.5,
        threshold=4.0,
    )
    assert hit.detected is True
    assert hit.onset_sec is not None
    assert -40.1 <= hit.onset_sec <= -39.5

    miss = detect_er_onset_preview(
        np.zeros(n_t, dtype=float),
        t_axis,
        detection_idx_window=(det_win.start_idx, det_win.end_idx),
        bias=0.5,
        threshold=4.0,
    )
    assert miss.detected is False
    assert miss.onset_idx is None
    assert miss.onset_sec is None


def test_preview_threshold_from_baseline_filters_small_drift_but_keeps_real_rise() -> None:
    hop_sec = 0.1
    pre_sec = 300.0
    n_t = int(round((pre_sec + 30.0) / hop_sec))
    t_axis = (np.arange(n_t) * hop_sec) - pre_sec

    baseline_mask = (t_axis >= -270.0) & (t_axis < -120.0)
    detection_mask = (t_axis >= -120.0) & (t_axis <= 30.0)

    z = np.zeros(n_t, dtype=float)
    z[baseline_mask] = 0.8
    z[detection_mask] = 0.8
    z[t_axis >= -20.0] = 5.0

    threshold = preview_threshold_from_baseline(
        z,
        baseline_idx_window=(int(np.where(baseline_mask)[0][0]), int(np.where(baseline_mask)[0][-1]) + 1),
        bias=0.5,
        threshold_margin=1.0,
    )
    assert threshold > 5.0

    early_only = detect_er_onset_preview(
        np.where(t_axis < -20.0, z, 0.0),
        t_axis,
        detection_idx_window=(int(np.where(detection_mask)[0][0]), int(np.where(detection_mask)[0][-1]) + 1),
        bias=0.5,
        threshold=threshold,
    )
    assert early_only.detected is False

    full = detect_er_onset_preview(
        z,
        t_axis,
        detection_idx_window=(int(np.where(detection_mask)[0][0]), int(np.where(detection_mask)[0][-1]) + 1),
        bias=0.5,
        threshold=threshold,
    )
    assert full.detected is True
    assert full.onset_sec is not None
    assert -20.1 <= full.onset_sec <= -15.0
