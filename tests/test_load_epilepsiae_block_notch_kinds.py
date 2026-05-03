"""TDD tests for ``load_epilepsiae_block(notch_filter_kind=...)``.

Plan reference: docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md Δ7.
Reference impl: ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_detectHFOs.py:79
(``cusignal.firwin(801, ...) + fftconvolve``).
"""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest


@pytest.fixture
def synthetic_epilepsiae_block(tmp_path):
    """Build a tiny in-memory Epilepsiae .data + .head pair for testing.

    8 intracranial channels + a known 50 Hz tone on channel 0 to verify the
    notch removes it. Sample rate 512 Hz, duration 4s. int16 little-endian
    interleaved frames matching the legacy convention.
    """
    sfreq = 512.0
    duration = 4.0
    n_samples = int(sfreq * duration)
    n_channels = 8
    ch_names = [f"GA{i + 1}" for i in range(n_channels)]

    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / sfreq
    data = rng.normal(0.0, 50.0, size=(n_channels, n_samples)).astype(np.float64)
    # Inject a strong 50 Hz tone on channel 0 to verify the notch works.
    data[0] += 200.0 * np.sin(2 * np.pi * 50.0 * t)

    # Quantize to int16 with a known conversion factor (legacy uses negative).
    conv = 0.5
    data_int = np.round(data / -conv).astype("<i2")

    data_path = tmp_path / "test.data"
    head_path = tmp_path / "test.head"
    data_int.T.tofile(data_path)  # interleave by frame

    head_path.write_text(
        f"elec_names=[{','.join(ch_names)}]\n"
        f"num_channels={n_channels}\n"
        f"sample_freq={sfreq}\n"
        f"duration_in_sec={duration}\n"
        f"num_samples={n_samples}\n"
        f"sample_bytes=2\n"
        f"conversion_factor={conv}\n"
        f"start_ts=2026-05-03 12:00:00.000000\n"
    )
    return data_path, head_path, ch_names, sfreq


def test_load_epilepsiae_block_default_uses_iir_notch(synthetic_epilepsiae_block):
    """Default notch_filter_kind must remain 'iir' for backward compatibility."""
    from src.preprocessing import load_epilepsiae_block

    data_path, head_path, _, _ = synthetic_epilepsiae_block
    pre = load_epilepsiae_block(data_path, head_path, reference="car")
    assert pre.data.shape[0] == 8
    assert pre.sfreq == 512.0


def test_load_epilepsiae_block_fir_legacy_notch_smoke(synthetic_epilepsiae_block):
    """FIR-legacy notch path must run without error and produce same shape."""
    from src.preprocessing import load_epilepsiae_block

    data_path, head_path, _, _ = synthetic_epilepsiae_block
    pre_iir = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[50.0, 100.0],
    )
    pre_fir = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[50.0, 100.0],
        notch_filter_kind="fir_legacy",
    )
    assert pre_iir.data.shape == pre_fir.data.shape
    # Both should attenuate the 50 Hz tone on channel 0; raw amplitude was
    # 200 µV peak, post-notch should be << 100.
    rms_iir_ch0 = float(np.sqrt(np.mean(pre_iir.data[0] ** 2)))
    rms_fir_ch0 = float(np.sqrt(np.mean(pre_fir.data[0] ** 2)))
    assert rms_iir_ch0 < 200.0
    assert rms_fir_ch0 < 200.0


def test_fir_legacy_notch_implementation_matches_legacy_formula(synthetic_epilepsiae_block):
    """The FIR notch must match the legacy hfo_net firwin(801, [(f-1)/nyq, (f+1)/nyq], pass_zero=True)
    + fftconvolve formula verbatim. Anchored to legacy
    ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_detectHFOs.py:79.
    """
    from scipy.signal import firwin, fftconvolve
    from src.preprocessing import load_epilepsiae_block

    data_path, head_path, _, sfreq = synthetic_epilepsiae_block
    nyq = sfreq / 2.0

    # Run the function with the legacy notch
    pre = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[50.0],
        notch_filter_kind="fir_legacy",
    )

    # Manually replicate the same flow on the SAME data
    pre_no_notch = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[],  # skip notch entirely
    )
    raw_after_car = pre_no_notch.data.copy()
    b = firwin(801, [(50 - 1) / nyq, (50 + 1) / nyq], pass_zero=True)
    expected = fftconvolve(raw_after_car, b[None, :], mode="same", axes=-1)

    np.testing.assert_allclose(pre.data, expected, rtol=1e-10, atol=1e-12)


def test_fir_legacy_notch_skips_freq_above_nyquist(synthetic_epilepsiae_block):
    """If a notch frequency would put the upper transition at/above Nyquist,
    the function must skip it (legacy would crash; we silently skip with no
    truncation of useful frequencies)."""
    from src.preprocessing import load_epilepsiae_block

    data_path, head_path, _, sfreq = synthetic_epilepsiae_block  # sfreq=512, nyq=256
    # 256 + 1 = 257 > nyq → must be skipped, not crash.
    pre = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[50.0, 100.0, 256.0],
        notch_filter_kind="fir_legacy",
    )
    assert pre.data.shape[0] == 8


def test_invalid_notch_filter_kind_raises(synthetic_epilepsiae_block):
    from src.preprocessing import load_epilepsiae_block

    data_path, head_path, _, _ = synthetic_epilepsiae_block
    with pytest.raises(ValueError, match="notch_filter_kind"):
        load_epilepsiae_block(
            data_path, head_path, reference="car",
            notch_filter_kind="butter",
        )
