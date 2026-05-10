"""Integration tests for yuquan branch of extract_seizure_window.

Uses synthetic inventories + a small EDF written to tmp_path.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.ictal_onset_extraction import extract_seizure_window


def _write_inventory_pair(tmp_root: Path, edf_path: Path, fs: float, duration: float):
    inv_dir = tmp_root / "dataset_inventory"
    inv_dir.mkdir(parents=True)

    block_csv = inv_dir / "yuquan_block_inventory.csv"
    seizure_csv = inv_dir / "yuquan_seizure_inventory.csv"

    block_start = 1700000000.0
    block_end = block_start + duration

    with open(block_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject", "recording_id", "block_id", "block_stem",
            "block_start_epoch", "block_end_epoch", "duration_sec",
            "sample_rate", "n_channels_total", "head_path", "data_path", "edf_path",
        ])
        w.writerow([
            "fakesid", edf_path.stem, edf_path.stem, edf_path.stem,
            f"{block_start}", f"{block_end}", f"{duration}",
            f"{fs}", "6", "", str(edf_path), str(edf_path),
        ])

    onset_epoch = block_start + duration / 2.0
    with open(seizure_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject", "patient_code", "recording_id", "record", "seizure_id",
            "eeg_onset_epoch", "eeg_offset_epoch", "eeg_duration_sec",
            "has_complete_eeg_interval", "timezone_name",
            "eeg_onset_local_hour", "eeg_onset_day_night",
            "record_start_epoch", "record_end_epoch",
        ])
        w.writerow([
            "fakesid", "fakesid", edf_path.stem, edf_path.stem, "fakesid_sz_001",
            f"{onset_epoch}", f"{onset_epoch + 60}", "60.0",
            "True", "Asia/Shanghai", "", "",
            f"{block_start}", f"{block_end}",
        ])


def _make_yuquan_synthetic_edf(tmp_path: Path) -> tuple[Path, float, float]:
    pytest.importorskip("mne")
    pytest.importorskip("edfio")
    import mne
    sfreq = 500.0
    duration = 800.0
    n_samples = int(sfreq * duration)
    ch_names = ["POL K3", "POL K4", "POL K5", "POL K6", "EEG Fp1-Ref", "POL DC01"]
    data = np.random.RandomState(1).randn(len(ch_names), n_samples) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    edf_path = tmp_path / "FAKEY1.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)
    return edf_path, sfreq, duration


def test_extract_seizure_window_yuquan_branch(tmp_path: Path):
    edf_path, sfreq, duration = _make_yuquan_synthetic_edf(tmp_path)
    _write_inventory_pair(tmp_path, edf_path, sfreq, duration)

    sw = extract_seizure_window(
        "yuquan/fakesid",
        seizure_idx=0,
        pre_sec=300.0,
        post_sec=30.0,
        results_root=tmp_path,
        reference="car",
    )

    assert sw.subject == "yuquan/fakesid"
    assert sw.seizure_id == "fakesid_sz_001"
    assert sw.fs == sfreq
    assert sw.signal.shape[0] == 4  # 4 intracranial channels (K3..K6)
    assert sw.signal.shape[1] == int((300.0 + 30.0) * sfreq)
    assert sw.t_axis[0] == pytest.approx(-300.0)
    assert sw.t_axis[-1] == pytest.approx(30.0 - 1.0 / sfreq, abs=1e-3)


def test_extract_seizure_window_yuquan_window_overruns_block(tmp_path: Path):
    edf_path, sfreq, duration = _make_yuquan_synthetic_edf(tmp_path)
    _write_inventory_pair(tmp_path, edf_path, sfreq, duration)

    # Request pre_sec longer than the seizure offset from block start
    # (onset = block_start + duration/2 = block_start + 400; pre_sec=500 → before block_start)
    with pytest.raises(ValueError, match="before block_start"):
        extract_seizure_window(
            "yuquan/fakesid",
            seizure_idx=0,
            pre_sec=500.0,
            post_sec=30.0,
            results_root=tmp_path,
        )


def test_cohort_selector_includes_yuquan_audit_eligible():
    from scripts.run_ictal_er_rank import _cohort_subject_list
    included, excluded = _cohort_subject_list()
    yuquan_in = [s for s in included if s.startswith("yuquan/")]
    epi_in = [s for s in included if s.startswith("epilepsiae/")]
    assert len(yuquan_in) == 9, f"expected 9 yuquan, got {yuquan_in}"
    # 15 audit_eligible + sentinel-only epilepsiae/916 = 16; locks cohort=25 milestone
    assert len(epi_in) == 16, f"expected 16 epilepsiae, got {epi_in}"
    assert "yuquan/gaolan" in yuquan_in
    assert "yuquan/zhangjinhan" in yuquan_in
    # No yuquan should be in `excluded` after this PR
    assert not any(s.startswith("yuquan/") for s in excluded), excluded
