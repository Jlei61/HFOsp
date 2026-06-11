"""Synthetic TDD for the Topic 5 Stage-2 recruitment instrument (Phase 1, pure math)."""
import numpy as np
import pytest

from src.topic5_ictal_recruitment import (
    line_length_trace, band_power_trace, spectral_edge_trace, baseline_robust_z,
    detect_contact_onset, calibrate_feature_lambda, resolve_global_onset,
    fuse_recruitment_rank, feature_agreement, bipolar_alias_label, assert_channel_identity,
)


# --- Task 1: line_length_trace -------------------------------------------------
def test_line_length_trace_shape_and_hop():
    fs = 200.0
    sig = np.zeros((3, int(round(10 * fs))))
    tr, t = line_length_trace(sig, fs, win_sec=1.0, hop_sec=0.1)
    assert tr.shape[0] == 3
    assert tr.shape[1] == t.shape[0]
    assert tr.shape[1] == pytest.approx(91, abs=1)
    assert np.allclose(t[:3], [0.0, 0.1, 0.2])


def test_line_length_trace_rises_with_fast_activity():
    fs = 200.0
    n = int(round(10 * fs))
    rng = np.random.default_rng(0)
    sig = 0.01 * rng.standard_normal((1, n))
    tt = np.arange(n) / fs
    sig[0, n // 2:] += np.sin(2 * np.pi * 40 * tt[n // 2:])
    tr, t = line_length_trace(sig, fs, win_sec=1.0, hop_sec=0.1)
    early = tr[0, t < 4.0].mean()
    late = tr[0, t > 6.0].mean()
    assert late > 5 * early


# --- Task 2: band_power_trace + spectral_edge_trace ----------------------------
def test_band_power_trace_hfa_requires_nyquist():
    fs = 200.0
    sig = np.random.default_rng(0).standard_normal((2, int(round(5 * fs))))
    with pytest.raises(ValueError):
        band_power_trace(sig, fs, band=(80.0, 150.0), win_sec=0.5, hop_sec=0.1)


def test_band_power_trace_tracks_injected_band():
    fs = 500.0
    n = int(round(8 * fs))
    tt = np.arange(n) / fs
    sig = np.zeros((1, n))
    sig[0, n // 2:] += np.sin(2 * np.pi * 100 * tt[n // 2:])
    tr, t = band_power_trace(sig, fs, band=(80.0, 150.0), win_sec=0.5, hop_sec=0.1)
    assert tr[0, t > 5].mean() > tr[0, t < 3].mean() + 2.0


def test_spectral_edge_trace_rises_to_fast():
    fs = 500.0
    n = int(round(8 * fs))
    tt = np.arange(n) / fs
    rng = np.random.default_rng(1)
    sig = np.zeros((1, n))
    sig[0, :n // 2] = np.sin(2 * np.pi * 5 * tt[:n // 2])
    sig[0, n // 2:] = np.sin(2 * np.pi * 90 * tt[n // 2:])
    sig += 0.01 * rng.standard_normal((1, n))
    sef, t = spectral_edge_trace(sig, fs, edge=0.9, win_sec=1.0, hop_sec=0.1)
    assert sef[0, t > 6].mean() > sef[0, t < 2].mean() + 30.0


# --- Task 3: baseline_robust_z -------------------------------------------------
def test_baseline_robust_z_centers_and_scales():
    rng = np.random.default_rng(2)
    tr = rng.standard_normal((1, 200))
    tr[0, 150] += 50.0
    z = baseline_robust_z(tr, (0, 50), hop_sec=0.1, min_baseline_valid_sec=2.0)
    assert abs(np.median(z[0, :50])) < 0.5
    assert z[0, 150] > 10.0


def test_baseline_robust_z_zero_mad_returns_nan():
    tr = np.ones((1, 100))
    z = baseline_robust_z(tr, (0, 50), hop_sec=0.1, min_baseline_valid_sec=2.0)
    assert np.all(np.isnan(z[0]))


def test_baseline_robust_z_insufficient_baseline_returns_nan():
    tr = np.random.default_rng(3).standard_normal((1, 100))
    z = baseline_robust_z(tr, (0, 5), hop_sec=0.1, min_baseline_valid_sec=2.0)
    assert np.all(np.isnan(z[0]))


# --- Task 4: detect_contact_onset ----------------------------------------------
def test_detect_contact_onset_sustained_step_fires():
    z = np.zeros(200)
    z[80:] = 6.0
    res = detect_contact_onset(z, lam=5.0, detection_idx_window=(0, 200),
                               hop_sec=0.1, win_sec=1.0, pre_sec=0.0)
    assert res["detected"] is True
    assert 78 <= res["onset_frame"] <= 90
    assert res["onset_sec"] == pytest.approx(res["onset_frame"] * 0.1 + 0.5, abs=0.2)


def test_detect_contact_onset_transient_is_ambiguous():
    z = np.zeros(200)
    z[80:83] = 6.0
    res = detect_contact_onset(z, lam=5.0, detection_idx_window=(0, 200),
                               hop_sec=0.1, win_sec=1.0, pre_sec=0.0)
    assert res["detected"] is False
    assert res["reason"] == "ambiguous"


def test_detect_contact_onset_never_crosses():
    z = 0.5 * np.ones(200)
    res = detect_contact_onset(z, lam=5.0, detection_idx_window=(0, 200),
                               hop_sec=0.1, win_sec=1.0, pre_sec=0.0)
    assert res["detected"] is False
    assert res["reason"] == "unreached"
    assert np.isnan(res["onset_frame"])


# --- Task 5: calibrate_feature_lambda ------------------------------------------
def test_calibrate_feature_lambda_pooled_ok():
    rng = np.random.default_rng(4)
    pooled = rng.standard_normal((10, 7000))
    out = calibrate_feature_lambda(pooled, fpr_target_per_hour=1.0, hop_sec=0.1,
                                   min_pooled_baseline_sec=600.0)
    assert out["calibration_unstable"] is False
    assert out["pooled_baseline_sec"] == pytest.approx(700.0)
    assert 1.0 <= out["lambda"] <= 100.0


def test_calibrate_feature_lambda_nan_frames_do_not_shrink_duration():
    rng = np.random.default_rng(9)
    pooled = rng.standard_normal((10, 7000))
    pooled[3, :] = np.nan
    out = calibrate_feature_lambda(pooled, fpr_target_per_hour=1.0, hop_sec=0.1,
                                   min_pooled_baseline_sec=600.0)
    assert out["pooled_baseline_sec"] == pytest.approx(700.0)
    assert out["calibration_unstable"] is False


def test_calibrate_feature_lambda_too_short_is_unstable():
    rng = np.random.default_rng(5)
    pooled = rng.standard_normal((10, 100))
    out = calibrate_feature_lambda(pooled, fpr_target_per_hour=1.0, hop_sec=0.1,
                                   min_pooled_baseline_sec=600.0)
    assert out["calibration_unstable"] is True
    assert np.isnan(out["lambda"])


# --- Task 6: resolve_global_onset ----------------------------------------------
def test_resolve_global_onset_reaches_fraction():
    onsets = np.full(20, 100.0)
    onsets[0] = 10.0
    res = resolve_global_onset(onsets, n_valid=20, frac=0.15)
    assert res["global_onset_resolved"] is True
    assert res["t_global"] == pytest.approx(100, abs=1)


def test_resolve_global_onset_single_transient_not_global():
    onsets = np.full(20, np.nan)
    onsets[0] = 10.0
    res = resolve_global_onset(onsets, n_valid=20, frac=0.15)
    assert res["global_onset_resolved"] is False
    assert np.isnan(res["t_global"])


def test_resolve_global_onset_unresolved_when_too_few():
    onsets = np.full(20, np.nan)
    onsets[:2] = [10.0, 11.0]
    res = resolve_global_onset(onsets, n_valid=20, frac=0.15)
    assert res["global_onset_resolved"] is False


# --- Task 7: fuse_recruitment_rank + feature_agreement -------------------------
def test_fuse_recruitment_rank_median_ignores_nan_and_excludes_er():
    per_feat = {
        "line_length": np.array([10.0, 20.0, 30.0, 40.0]),
        "broadband":   np.array([11.0, 19.0, np.nan, 41.0]),
        "hfa":         np.array([9.0, 21.0, 31.0, 39.0]),
        "spectral_edge": np.array([10.0, 20.0, 30.0, 40.0]),
    }
    fused_rank, fused_onset = fuse_recruitment_rank(per_feat)
    assert np.argsort(fused_rank).tolist() == [0, 1, 2, 3]
    assert np.isfinite(fused_onset[2])


def test_feature_agreement_flag_amplitude_only():
    n = 10
    base = np.arange(n, dtype=float)
    per_rank = {
        "line_length": base.copy(),
        "broadband": base.copy(),
        "hfa": base.copy(),
        "spectral_edge": base[::-1].copy(),
    }
    ag = feature_agreement(per_rank, amplitude=("line_length", "broadband", "hfa"),
                           spectral="spectral_edge", early_k=3)
    assert ag["feature_agreement_flag"] is True
    assert ag["spectral_support"] < 0
    assert ag["spectral_conflict_flag"] is True


def test_feature_agreement_flag_false_when_amplitude_disagrees():
    n = 10
    rng = np.random.default_rng(6)
    per_rank = {
        "line_length": np.arange(n, dtype=float),
        "broadband": rng.permutation(n).astype(float),
        "hfa": rng.permutation(n).astype(float),
        "spectral_edge": np.arange(n, dtype=float),
    }
    ag = feature_agreement(per_rank, amplitude=("line_length", "broadband", "hfa"),
                           spectral="spectral_edge", early_k=3)
    assert ag["feature_agreement_flag"] is False


# --- Task 8: montage helpers ---------------------------------------------------
def test_bipolar_alias_label_left_contact():
    assert bipolar_alias_label("HRA1-HRA2") == "HRA1"
    assert bipolar_alias_label("BFRA5-BFRA6") == "BFRA5"


def test_assert_channel_identity_ok_when_montage_matches():
    assert_channel_identity(template_montage="bipolar_aliased_left",
                            ictal_montage="bipolar_aliased_left")


def test_assert_channel_identity_hard_fail_on_montage_mismatch():
    with pytest.raises(ValueError, match="montage"):
        assert_channel_identity(template_montage="bipolar_aliased_left",
                                ictal_montage="car_monopolar")


# --- Task 9: echo-core reuse smoke ---------------------------------------------
def test_recruitment_rank_feeds_stage1_echo_core():
    from src.topic5_echo_gate import compute_echo_strength
    template = np.arange(12, dtype=float)
    recruitment_rank = np.arange(12, dtype=float)
    res = compute_echo_strength(recruitment_rank, [template], B=1000,
                                rng=np.random.default_rng(7), min_ch=8)
    assert res["r_obs"] == pytest.approx(1.0)
    assert res["e_k"] > 3.0
