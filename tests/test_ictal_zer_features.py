"""TDD tests for src.ictal_zer_features — z-ER binned tensor feature.

Topic5 PR-1 Step 2: upgrade clustering feature from (channel,) t_onset
vector to (channel × time-bin) z-ER mean tensor. Schroeder/Panagiotopoulou
style — captures both pre-onset rank AND post-onset extent/intensity.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# bin_zer_to_features


def test_bin_zer_assigns_correct_time_bins():
    from src.ictal_zer_features import bin_zer_to_features
    # 5 channels, 100 frames covering t = -200..+200 s in 4-s steps
    n_ch, n_t = 5, 100
    t_axis = np.linspace(-200.0, 200.0, n_t, endpoint=False)
    # z-ER: each channel has constant value = (channel_idx + 1) for all bins
    z = np.tile((np.arange(n_ch, dtype=float) + 1)[:, None], (1, n_t))
    bins = [(-200, -50), (-50, 0), (0, 50), (50, 150), (150, 200)]
    feature, bin_idx_used = bin_zer_to_features(z, t_axis, bins)
    # Shape (n_ch * n_bins,) = (25,)
    assert feature.shape == (5 * 5,)
    # Channel 0 has value 1 → all 5 bins for channel 0 should be 1.0
    # Layout: feature[ch * n_bins + bin] (channel-major)
    for ch in range(5):
        for b in range(5):
            assert feature[ch * 5 + b] == pytest.approx(float(ch + 1))


def test_bin_zer_handles_partial_coverage():
    """If t_axis doesn't cover all bins, missing bins should be NaN."""
    from src.ictal_zer_features import bin_zer_to_features
    # t_axis only covers [-100, +100], so bins [-200,-50] and [150,200] partial/missing
    n_ch = 3
    n_t = 50
    t_axis = np.linspace(-100.0, 100.0, n_t, endpoint=False)
    z = np.ones((n_ch, n_t))
    bins = [(-200, -50), (-50, 0), (0, 50), (50, 150), (150, 200)]
    feature, _ = bin_zer_to_features(z, t_axis, bins)
    # Bin 0 [-200,-50]: t_axis covers part of [-100, -50] → has data
    # Bin 4 [150, 200]: t_axis stops at 100 → all NaN
    for ch in range(n_ch):
        assert np.isnan(feature[ch * 5 + 4])  # bin 4 entirely uncovered


def test_bin_zer_nan_input_propagates_to_bin():
    from src.ictal_zer_features import bin_zer_to_features
    n_ch, n_t = 3, 100
    t_axis = np.linspace(-200.0, 200.0, n_t, endpoint=False)
    z = np.zeros((n_ch, n_t))
    z[1, :] = np.nan  # channel 1 entirely NaN
    bins = [(-200, -50), (-50, 0), (0, 50), (50, 150), (150, 200)]
    feature, _ = bin_zer_to_features(z, t_axis, bins)
    # Channel 1 → all 5 bins NaN
    for b in range(5):
        assert np.isnan(feature[1 * 5 + b])
    # Channels 0, 2 → all bins 0
    for ch in (0, 2):
        for b in range(5):
            assert feature[ch * 5 + b] == pytest.approx(0.0)


# ===========================================================================
# stack_features_to_matrix


def test_stack_features_handles_jagged_skipped_seizures():
    """If some seizures fail z-ER extraction, returned matrix omits them."""
    from src.ictal_zer_features import stack_features_to_matrix
    feats = [
        np.array([1.0, 2.0, 3.0]),
        None,                       # extraction failed
        np.array([10.0, 20.0, 30.0]),
        np.array([np.nan, 5.0, 6.0]),
    ]
    seizure_ids = ["sz0", "sz1", "sz2", "sz3"]
    feature_matrix, kept_ids = stack_features_to_matrix(feats, seizure_ids)
    # sz1 dropped
    assert feature_matrix.shape == (3, 3)
    assert kept_ids == ["sz0", "sz2", "sz3"]
    np.testing.assert_array_equal(feature_matrix[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(feature_matrix[:, 1], [10.0, 20.0, 30.0])
    assert np.isnan(feature_matrix[0, 2])


# ===========================================================================
# Channel-order consistency check (audit 2026-05-10)


def test_extract_zer_drops_seizure_on_channel_order_mismatch(monkeypatch):
    """If a later seizure's ch_names disagrees with the first successful
    extraction's, the seizure must be dropped (drop_reason populated) —
    not silently stacked into the feature matrix.
    """
    import src.ictal_zer_features as mod

    class FakeWindow:
        def __init__(self, seizure_id, ch_names, n_t=200, fs=512.0):
            self.seizure_id = seizure_id
            self.ch_names = ch_names
            self.fs = fs
            self.pre_sec = 100.0
            self.signal = np.zeros((len(ch_names), n_t))
            self.eeg_onset_epoch = None
            self.clin_onset_epoch = 0.0

    def fake_extract(subject, sz_idx, **kw):
        # sz 0, 1: standard 3-channel order; sz 2: REORDERED → mismatch
        chs = ["A", "B", "C"] if sz_idx != 2 else ["B", "A", "C"]
        return FakeWindow(seizure_id=f"id_{sz_idx}", ch_names=chs)

    def fake_compute_er(signal, **kw):
        return np.zeros((signal.shape[0], 50))

    class _BW:
        valid = True
        start_idx = 0
        end_idx = 10

    def fake_resolve(*a, **k):
        return _BW()

    def fake_zscore(er, *a, **k):
        return np.zeros_like(er)

    # Patch the imports inside the function (lazy-imported)
    monkeypatch.setattr(
        "src.ictal_onset_extraction.extract_seizure_window", fake_extract,
        raising=False,
    )
    monkeypatch.setattr(
        "src.ictal_onset_extraction.compute_er", fake_compute_er,
        raising=False,
    )
    monkeypatch.setattr(
        "src.ictal_onset_extraction.resolve_baseline_window", fake_resolve,
        raising=False,
    )
    monkeypatch.setattr(
        "src.ictal_onset_extraction.baseline_zscore_er", fake_zscore,
        raising=False,
    )
    feats, sids, ch_names, drops = mod.extract_zer_binned_for_subject(
        "fake/0", "gamma_ER", [0, 1, 2],
    )
    assert ch_names == ["A", "B", "C"]
    # sz 0 + 1: extracted; sz 2: dropped with channel_order_mismatch
    assert feats[0] is not None and feats[1] is not None
    assert feats[2] is None
    assert "channel_order_mismatch" in drops[2]
