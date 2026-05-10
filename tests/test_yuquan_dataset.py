"""Tests for src/yuquan_dataset.py.

Synthetic-EDF only — does NOT touch real /mnt/yuquan_data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.yuquan_dataset import (
    classify_yuquan_channel,
    normalize_yuquan_channel_name,
    load_yuquan_record,
)


@pytest.mark.parametrize("raw_name,expected", [
    # scalp ref
    ("EEG A1-Ref", "scalp_ref"),
    ("EEG A2-Ref", "scalp_ref"),
    # 10-20 scalp (extended)
    ("EEG Fp1-Ref", "scalp"),
    ("EEG C3-Ref", "scalp"),
    ("EEG F10-Ref", "scalp"),
    ("EEG C5-Ref", "scalp"),
    # intracranial (single letter + optional apostrophe + 1-3 digits)
    ("POL A3", "intracranial"),
    ("POL A'1", "intracranial"),
    ("POL B'16", "intracranial"),
    ("POL K11", "intracranial"),
    ("POL E10", "intracranial"),
    ("POL F16", "intracranial"),
    # aux: DC channels (false-positive of naive regex — must be aux)
    ("POL DC01", "aux"),
    ("POL DC10", "aux"),
    # aux: physio
    ("POL ECG", "aux"),
    ("POL EMG1", "aux"),
    ("POL EMGLR", "aux"),
    ("POL OSAT", "aux"),
    ("POL PULSE", "aux"),
    # aux: bare letter, no digits
    ("POL E", "aux"),
    # SEEG-named channel under EEG prefix (real subjects sometimes use this)
    ("EEG K11-Ref", "intracranial"),
])
def test_classify_yuquan_channel(raw_name, expected):
    assert classify_yuquan_channel(raw_name) == expected


@pytest.mark.parametrize("raw_name,expected", [
    ("EEG A1-Ref", "A1"),
    ("POL K11", "K11"),
    ("POL E10", "E10"),
    ("POL A'1", "A'1"),
    ("POL B'16", "B'16"),
    ("EEG K11-Ref", "K11"),
])
def test_normalize_yuquan_channel_name(raw_name, expected):
    assert normalize_yuquan_channel_name(raw_name) == expected


def _make_synthetic_yuquan_edf(tmp_path: Path) -> Path:
    """Use mne.io.RawArray + export_raw to write a small EDF with mixed channel
    types so we can test classify + load on a real file format.

    Skips the test if the optional ``edfio`` writer dependency is missing
    (mne.export.export_raw needs it; we MUST NOT pip-install on the user's behalf).
    """
    pytest.importorskip("edfio")
    import mne
    sfreq = 200.0
    duration = 60.0
    n_samples = int(sfreq * duration)
    # 4 intracranial + 1 scalp + 1 aux
    ch_names = ["POL K3", "POL K4", "POL K5", "POL K6", "EEG Fp1-Ref", "POL DC01"]
    data = np.random.RandomState(0).randn(len(ch_names), n_samples).astype(np.float64) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    edf_path = tmp_path / "FAKE001.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)
    return edf_path


def test_load_yuquan_record_intracranial_filter(tmp_path: Path):
    pytest.importorskip("mne")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)

    pre = load_yuquan_record(edf_path, reference="car", intracranial_only=True)

    # Only the 4 POL K* channels should remain
    assert pre.data.shape[0] == 4
    assert all(name in {"K3", "K4", "K5", "K6"} for name in pre.ch_names)
    assert abs(pre.sfreq - 200.0) < 1e-3


def test_load_yuquan_record_car_zero_mean(tmp_path: Path):
    pytest.importorskip("mne")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)

    pre = load_yuquan_record(edf_path, reference="car", intracranial_only=True)

    # CAR must zero-mean across channels at every sample
    sample_means = pre.data.mean(axis=0)
    assert np.allclose(sample_means, 0.0, atol=1e-12)


def test_load_yuquan_record_bipolar_reduces_count(tmp_path: Path):
    pytest.importorskip("mne")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)

    pre = load_yuquan_record(edf_path, reference="bipolar", intracranial_only=True)

    # 4 contiguous K3..K6 → 3 bipolar pairs K3-K4, K4-K5, K5-K6
    assert pre.data.shape[0] == 3
    assert pre.ch_names == ["K3-K4", "K4-K5", "K5-K6"]


def test_load_yuquan_record_raises_on_missing_file(tmp_path: Path):
    """File-not-found surfaces as FileNotFoundError with the path."""
    missing = tmp_path / "nonexistent.edf"
    with pytest.raises(FileNotFoundError):
        load_yuquan_record(missing)


def test_load_yuquan_record_raises_on_unknown_reference(tmp_path: Path):
    """Unknown reference must raise ValueError listing accepted values."""
    pytest.importorskip("edfio")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)
    with pytest.raises(ValueError, match="reference must be"):
        load_yuquan_record(edf_path, reference="laplacian")


def test_load_yuquan_record_raises_when_no_intracranial_channels(tmp_path: Path):
    """If filtering removes all channels, raise with diagnostic info."""
    pytest.importorskip("edfio")
    import mne
    sfreq = 200.0
    n_samples = int(sfreq * 60.0)
    # All-aux EDF: scalp + DC + EMG, zero intracranial
    ch_names = ["EEG Fp1-Ref", "EEG C3-Ref", "POL DC01", "POL EMG1"]
    data = np.random.RandomState(7).randn(len(ch_names), n_samples) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    edf_path = tmp_path / "ALLAUX.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)

    with pytest.raises(ValueError, match="No intracranial channels"):
        load_yuquan_record(edf_path, intracranial_only=True)


def test_load_yuquan_record_bipolar_keeps_prime_shafts_separate(tmp_path: Path):
    """A and A' are different SEEG shafts; bipolar must NOT pair across them."""
    pytest.importorskip("edfio")
    import mne
    sfreq = 200.0
    n_samples = int(sfreq * 60.0)
    # 2 contacts on shaft A + 2 on shaft A' (prime/contralateral)
    ch_names = ["POL A1", "POL A2", "POL A'1", "POL A'2"]
    data = np.random.RandomState(11).randn(len(ch_names), n_samples) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    edf_path = tmp_path / "PRIMED.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)

    pre = load_yuquan_record(edf_path, reference="bipolar", intracranial_only=True)

    # Expect exactly 2 pairs: A1-A2 and A'1-A'2. No cross-shaft pair.
    assert pre.data.shape[0] == 2
    expected_pairs = {"A1-A2", "A'1-A'2"}
    assert set(pre.ch_names) == expected_pairs, (
        f"got {pre.ch_names}; cross-prime pair would corrupt this set"
    )
