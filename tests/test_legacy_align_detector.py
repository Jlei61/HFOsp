"""Unit tests for legacy_align flag plumbing in the HFO detector + preprocessor.

Covers Phase D fixes from `docs/archive/yuquan_lagpat/yuquan_detector_drift_root_cause.plan.md`:
- R02: refine uses global pickChn_thresh=1.0; pack uses per-subject pick_k
- D03: legacy resample factors `up=2, down=round(2*fs/800)`
- D04: legacy chunk_sec = 200 s
- D13: notch_freqs includes 250 Hz under legacy_align
- D14: notch impl is FIR firwin(801, ±2 Hz) under legacy_align (not iirnotch)
- D18: chunk-edge events with empty side windows are REJECTED under legacy_align
- D21: chunk_overlap_sec forced to 0 under legacy_align
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing import SEEGPreprocessor
from src.hfo_detector import HFODetectionConfig, HFODetector
from src.utils.bqk_utils import BQKDetector, find_high_enveTimes


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config / contract tests
# ---------------------------------------------------------------------------

def test_yuquan_defaults_have_legacy_flags():
    """yuquan _defaults must enable legacy_align + 250Hz notch + chunk_sec=200."""
    p = _PROJECT_ROOT / "config" / "subject_params.json"
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    yq = cfg["yuquan"]["_defaults"]
    assert yq["legacy_align"] is True
    assert yq["refine_pick_k"] == 1.0
    assert 250 in yq["notch_freqs"]
    assert yq["notch_freqs"] == [50, 100, 150, 200, 250]
    assert yq["chunk_sec"] == 200.0


def test_per_subject_pick_k_unchanged_for_pack_stage():
    """`pick_k` per subject in JSON is now consumed only by the pack stage
    (run_pipeline.py). Detection (run_hfo_detection.py) reads `refine_pick_k`."""
    p = _PROJECT_ROOT / "config" / "subject_params.json"
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    yq = cfg["yuquan"]
    # Sanity: per-subject pack pick_k still recorded so pack stage can read it
    assert yq["gaolan"]["pick_k"] == 1.9
    assert yq["dongyiming"]["pick_k"] == 0.5
    # And the new refine default (1.0) is in _defaults
    assert yq["_defaults"]["refine_pick_k"] == 1.0


# ---------------------------------------------------------------------------
# HFODetectionConfig + BQKDetector flag plumbing
# ---------------------------------------------------------------------------

def test_hfo_config_has_legacy_align_field():
    cfg = HFODetectionConfig(legacy_align=True)
    assert cfg.legacy_align is True
    assert HFODetectionConfig().legacy_align is False


def test_bqk_detector_legacy_align_uses_fir_bank():
    """legacy_align => CPU bank is FIR firwin(201), not Butter SOS (D15)."""
    det_legacy = BQKDetector(
        sfreq=800.0, freqband=(80, 250), use_gpu=False, legacy_align=True,
    )
    assert det_legacy.legacy_align is True
    assert len(det_legacy.filter_bank_fir_cpu) > 0
    assert all(b.shape[0] == 201 for b in det_legacy.filter_bank_fir_cpu)
    # SOS bank not built in legacy mode
    assert det_legacy.filter_bank_sos == []

    det_default = BQKDetector(
        sfreq=800.0, freqband=(80, 250), use_gpu=False, legacy_align=False,
    )
    assert det_default.legacy_align is False
    assert len(det_default.filter_bank_sos) > 0
    assert det_default.filter_bank_fir_cpu == []


# ---------------------------------------------------------------------------
# D18: chunk-edge events with empty side windows
# ---------------------------------------------------------------------------

def test_d18_find_high_enveTimes_accepts_legacy_align_param():
    """find_high_enveTimes accepts the new legacy_align kwarg without
    breaking the existing API. Default (False) preserves prior behaviour;
    True enables the empty-side-window rejection branch (only triggers
    inside chunked detector when an event crosses a chunk boundary —
    not directly reachable from this unit test, see D18 in plan).
    """
    fs = 800.0
    n = int(0.5 * fs)
    rng = np.random.default_rng(0)
    enve = (np.abs(rng.normal(0, 1, n)) + 1.0)[None, :].astype(np.float64)
    # Spike a short transient
    enve[0, 200:220] *= 8.0

    common = dict(
        chns_nums=1, fs=fs,
        rel_thresh=2.0, abs_thresh=2.0,
        min_gap=20, min_last=10, max_last=200,
        side_thresh=1.5, start_time=0.0,
    )

    out_default = find_high_enveTimes(enve, **common, legacy_align=False)
    out_legacy = find_high_enveTimes(enve, **common, legacy_align=True)

    # Both calls succeed and return a list with 1 channel entry.
    assert isinstance(out_default, list) and len(out_default) == 1
    assert isinstance(out_legacy, list) and len(out_legacy) == 1
    # Legacy is a subset of default (it can only reject more, never accept more)
    assert len(out_legacy[0]) <= len(out_default[0])


def test_d18_gpu_path_threads_legacy_align_into_init():
    """BQKDetector.legacy_align is read by both CPU find_high_enveTimes and
    GPU _find_high_enveTimes_gpu. Smoke check the constructor wires it
    through as a public attribute."""
    det = BQKDetector(
        sfreq=800.0, freqband=(80, 250), use_gpu=False,
        side_thresh=1.5, legacy_align=True,
    )
    assert det.legacy_align is True
    assert det.side_thresh == 1.5


# ---------------------------------------------------------------------------
# D21: chunk_overlap_sec forced to 0 under legacy_align
# ---------------------------------------------------------------------------

def test_d21_legacy_align_zero_chunk_overlap():
    """In _detect_bqk_chunked, when cfg.legacy_align=True the effective overlap
    must be 0 even if chunk_overlap_sec is non-zero in the config."""
    fs = 800.0
    n_ch = 2
    duration_sec = 60.0  # >> chunk_sec so chunking actually happens
    n_samp = int(duration_sec * fs)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (n_ch, n_samp)).astype(np.float64)

    class _StubPre:
        def __init__(self, d, f, ch):
            self.data = d
            self.sfreq = f
            self.ch_names = ch

    pre = _StubPre(data, fs, [f"CH{i}" for i in range(n_ch)])

    cfg = HFODetectionConfig(
        bandpass=(80, 250),
        chunk_sec=20.0, chunk_overlap_sec=2.0,
        n_jobs=1, use_gpu=False,
        legacy_align=True,
    )
    det = HFODetector(cfg)
    res = det.detect(pre)

    # Sanity: detect ran without crashing under legacy_align
    assert res.events_count.shape == (n_ch,)
    # No assertion on event counts (signal is white noise) — this test is
    # primarily a smoke + plumbing check that the legacy code path executes.


# ---------------------------------------------------------------------------
# D03: legacy resample factors
# ---------------------------------------------------------------------------

def test_d03_legacy_resample_factors_at_2048hz():
    """fs=2048 under legacy_resample => up=2, down=5 => effective ~819.2 Hz."""
    pre = SEEGPreprocessor(
        target_sfreq=800.0,
        check_quality=False,
        use_fif_cache=False,
        legacy_align=True,
    )
    fs_in = 2048.0
    n = int(2.0 * fs_in)
    # Single-channel sine for shape check
    data = np.sin(2 * np.pi * 100.0 * np.arange(n) / fs_in)[None, :]

    out = pre._resample(data, fs_in, 800.0)

    # Legacy: up=2, down=round(2*2048/800)=round(5.12)=5 => out_len = n * 2 / 5
    expected_len = int(n * 2 / 5)
    # resample_poly may add 0-1 sample of edge — accept ±1
    assert abs(out.shape[-1] - expected_len) <= 1


def test_d03_default_resample_uses_fraction_at_2048hz():
    """Default (legacy_align=False, legacy_resample=False) uses Fraction =>
    up=25, down=64 at 2048 Hz, exact 800 Hz."""
    pre = SEEGPreprocessor(
        target_sfreq=800.0,
        check_quality=False,
        use_fif_cache=False,
        legacy_align=False,
    )
    fs_in = 2048.0
    n = int(2.0 * fs_in)
    data = np.sin(2 * np.pi * 100.0 * np.arange(n) / fs_in)[None, :]

    out = pre._resample(data, fs_in, 800.0)
    # Fraction(800, 2048) = 25/64 => out_len = n * 25 / 64 = exact 800 Hz @ 2 s = 1600 samples
    assert abs(out.shape[-1] - 1600) <= 1


# ---------------------------------------------------------------------------
# D13 + D14: notch_freqs includes 250Hz; FIR notch removes 250 Hz line noise
# ---------------------------------------------------------------------------

def test_d13_d14_legacy_notch_removes_250hz_preserves_other_bands():
    """Under legacy_align with 250Hz in notch_freqs:
    - 250Hz line is suppressed ≥ 20 dB
    - 120Hz (NOT in notch list) is preserved within 3 dB

    The second assertion is critical: it catches the failure mode where the
    notch kernel is misapplied as a bandpass (which would kill 120Hz too).
    """
    fs = 800.0
    duration = 4.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    bg = 0.1 * rng.normal(0, 1, n)
    line250 = 1.0 * np.sin(2 * np.pi * 250.0 * t)
    sig120 = 1.0 * np.sin(2 * np.pi * 120.0 * t)
    data = (bg + line250 + sig120)[None, :]

    pre_legacy = SEEGPreprocessor(
        target_sfreq=800.0,
        check_quality=False,
        use_fif_cache=False,
        legacy_align=True,
    )
    out_legacy = pre_legacy._apply_notch_legacy_fir(data, fs, [50, 100, 150, 200, 250])

    pre_default = SEEGPreprocessor(
        target_sfreq=800.0,
        check_quality=False,
        use_fif_cache=False,
        legacy_align=False,
        notch_freqs=[50, 100, 150, 200],  # missing 250
    )
    out_default = pre_default.filter_backend.apply_notch(
        data, fs, pre_default.notch_freqs,
    )

    def _power_at(x, freq):
        X = np.fft.rfft(x[0])
        freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
        idx = int(np.argmin(np.abs(freqs - freq)))
        return float(np.sum(np.abs(X[max(0, idx - 1):idx + 2]) ** 2))

    p_in_250 = _power_at(data, 250.0)
    p_in_120 = _power_at(data, 120.0)
    p_legacy_250 = _power_at(out_legacy, 250.0)
    p_legacy_120 = _power_at(out_legacy, 120.0)
    p_default_250 = _power_at(out_default, 250.0)

    # Legacy must suppress 250 Hz by ≥ 20 dB
    assert p_legacy_250 < p_in_250 / 100.0, (
        f"Legacy FIR notch at 250 Hz failed: in={p_in_250:.3e}, "
        f"out={p_legacy_250:.3e}"
    )
    # Legacy must PRESERVE 120 Hz (within 3 dB / factor 2). This is the
    # critical sanity check that catches the bandpass-instead-of-bandstop
    # failure mode.
    assert p_legacy_120 > p_in_120 / 2.0, (
        f"Legacy FIR notch at 250 Hz incorrectly killed 120 Hz too: "
        f"in={p_in_120:.3e}, out={p_legacy_120:.3e}. "
        f"Likely the notch kernel is being inverted into a bandpass."
    )
    # Default (no 250 Hz notch) must leave the line essentially intact
    assert p_default_250 > p_in_250 / 2.0, (
        f"Default backend should NOT suppress 250 Hz: in={p_in_250:.3e}, "
        f"out={p_default_250:.3e}"
    )
