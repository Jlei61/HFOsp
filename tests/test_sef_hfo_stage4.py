"""TDD for the pure Stage 4 helpers (src/sef_hfo_stage4.py).
Spec: docs/superpowers/specs/2026-06-15-sef-hfo-snn-stage4-extended-patch-stochastic-readout-design.md
Plan: docs/superpowers/plans/2026-06-15-sef-hfo-snn-stage4-extended-patch.md
"""
import numpy as np
from src.sef_hfo_stage4 import (nucleation_centroid, readout_direction_distribution,
                                first_contact_entropy, correspondence_two_stage, _auc,
                                nucleation_dispersion, hotspot_degeneracy)


def test_isolated_early_spike_does_not_pollute_centroid():
    # 6-cell cluster at x~2 fires steps 10-11; ONE stray at x=18 fires step 2 (way early).
    # The onset is anchored on the k_min-th earliest spike (in the cluster), so the stray
    # is outside the [onset-tau_nuc, onset+tau_nuc] window and cannot drag the centroid.
    posE = np.array([[2, 5], [2.1, 5], [1.9, 5.1], [2, 4.9], [2.05, 5.0], [1.95, 5.0], [18, 5]], float)
    spk = np.zeros((40, 7), bool)
    spk[10, 0] = spk[10, 1] = spk[11, 2] = spk[10, 3] = spk[11, 4] = spk[10, 5] = True  # cluster
    spk[2, 6] = True                                                                    # stray
    out = nucleation_centroid(spk, np.arange(7), posE, t_on_idx=0, tau_nuc_steps=4,
                              axis_unit=np.array([1.0, 0.0]), patch_center=np.array([2.0, 5.0]),
                              k_min=5)
    assert out is not None
    assert abs(out["centroid_xy"][0] - 2.0) < 0.5      # stray (x=18) excluded
    assert out["n_early_cells"] == 6                    # cluster only
    assert abs(out["s_nuc"]) < 0.5


def test_too_few_early_returns_none():
    posE = np.array([[2, 5], [2, 5], [2, 5]], float)
    spk = np.zeros((20, 3), bool); spk[10, 0] = spk[11, 1] = True   # only 2 fire, k_min=5
    out = nucleation_centroid(spk, np.arange(3), posE, t_on_idx=0, tau_nuc_steps=4,
                              axis_unit=np.array([1.0, 0.0]), patch_center=np.array([2.0, 5.0]),
                              k_min=5)
    assert out is None


def test_sign_entropy_captures_bidirectionality_axis_concentration_separate():
    # 6 forward, 4 reverse, 2 unreadable; all readable angles hug the 0/180 axis line.
    signs = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, None, None]
    angles = [2, -3, 1, 0, 4, -1, 178, 182, 179, 176, np.nan, np.nan]
    out = readout_direction_distribution(signs, angles, axis_angle_deg=0.0)
    assert out["n_readable"] == 10 and out["n_unreadable"] == 2
    assert abs(out["forward_frac"] - 0.6) < 1e-9
    assert out["sign_entropy"] > 0.9          # bidirectional mix present (H(0.6) ~ 0.971)
    assert out["axis_concentration"] > 0.9    # all readable hug the 0/180 axis line
    assert out["near_axis_frac"] > 0.9


def test_sign_entropy_zero_when_unidirectional():
    out = readout_direction_distribution([1, 1, 1, 1], [1, 2, -1, 0], axis_angle_deg=0.0)
    assert out["sign_entropy"] == 0.0         # one sign -> NOT bidirectional
    assert out["axis_concentration"] > 0.9    # but still tightly on-axis


def test_first_contact_entropy_uniform_vs_degenerate():
    assert first_contact_entropy(["c2", "c2", "c2", "c2"], n_contacts=4) == 0.0
    assert abs(first_contact_entropy(["c0", "c1", "c2", "c3"], n_contacts=4) - 1.0) < 1e-9


def test_auc_values_ties_and_oneclass():
    assert _auc([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]) == 1.0    # perfectly separable
    assert _auc([1, 2, 3, 4], [0, 1, 1, 0]) == 0.5                 # genuine chance
    assert _auc([1, 2, 3, 4], [0, 1, 0, 1]) == 0.75               # the corrected value
    assert _auc([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5                # all ties -> 0.5
    assert np.isnan(_auc([1, 2, 3], [1, 1, 1]))                   # one class -> nan


def test_two_stage_none_safe_and_beats_shuffle():
    rng = np.random.default_rng(0); n = 80
    s_nuc = rng.uniform(-1, 1, n); r_off = rng.uniform(-0.3, 0.3, n)
    readable = (np.abs(s_nuc) > 0.3) & (np.abs(r_off) < 0.2)
    sign = [(1 if s_nuc[i] > 0 else -1) if readable[i] else None for i in range(n)]  # None unreadable
    out = correspondence_two_stage(s_nuc, r_off, readable, sign,
                                   rng=np.random.default_rng(1), n_shuffle=200)
    assert out["stage1_auc_s_nuc"] > 0.7       # end-like position predicts readability
    assert out["stage2_auc_sign"] > 0.9        # s_nuc predicts sign
    assert out["stage2_p_shuffle"] < 0.05      # beats the within-event shuffle null


def test_correspondence_raises_on_readable_none_sign():
    # a readable event whose sign is None must raise (not be silently treated as reverse)
    import pytest
    s_nuc = np.array([0.5, -0.5, 0.6, -0.6, 0.7]); r_off = np.zeros(5)
    readable = np.array([True, True, True, True, True])
    sign = [1, -1, None, -1, 1]                       # index 2 readable but sign None
    with pytest.raises(ValueError):
        correspondence_two_stage(s_nuc, r_off, readable, sign,
                                 rng=np.random.default_rng(0), n_shuffle=10)


# --- T0 continuous-patch gate (anti-two-hotspot), Phase 2 plan 2026-06-17 ---

def test_hotspot_low_n_indeterminate():
    # < n_min nucleation-valid events -> cannot judge healthy vs degenerate (NOT a pass)
    s = np.array([-3.0, -2.9, 3.0, 2.8]); r = np.zeros(4)
    out = hotspot_degeneracy(s, r, patch_r=8.0, n_min=6)
    assert out["verdict"] == "indeterminate_low_n"


def test_hotspot_two_tight_clusters_degenerate():
    # two SPATIALLY TIGHT blobs at s=-4 / s=+4 (each RMS radius ~0.4 mm << 1.6 mm) = covert two-focus
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(-4, 0.3, 9), rng.normal(0, 0.3, 9)])
    b = np.column_stack([rng.normal(+4, 0.3, 9), rng.normal(0, 0.3, 9)])
    pts = np.vstack([a, b])
    out = hotspot_degeneracy(pts[:, 0], pts[:, 1], patch_r=8.0)
    assert out["verdict"] == "two_hotspot_degenerate"
    assert max(out["cluster_radii"]) < out["tight_thresh"]


def test_hotspot_continuous_along_axis_healthy():
    s = np.linspace(-7, 7, 20); r = np.zeros(20)
    out = hotspot_degeneracy(s, r, patch_r=8.0)
    assert out["verdict"] == "healthy"


def test_hotspot_end_favouring_but_continuous_not_killed():
    # CRITICAL: denser at the two ends BUT each end-mode is SPREAD (std ~2.5) -> continuous, not
    # tight -> must stay healthy. spec §3.1 expects end-favouring; only fixed TIGHT points degenerate.
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.normal(-4.5, 2.5, 14), rng.normal(4.5, 2.5, 14)])
    r = rng.normal(0, 1.0, 28)
    out = hotspot_degeneracy(s, r, patch_r=8.0)
    assert out["verdict"] == "healthy"          # spread end-modes -> radius > tight_thresh


def test_nucleation_dispersion_continuous_vs_tight_and_low_n():
    cont = nucleation_dispersion(np.linspace(-6, 6, 20), np.zeros(20), patch_r=8.0)
    rng = np.random.default_rng(2)
    tight_s = np.concatenate([rng.normal(-4, 0.2, 10), rng.normal(4, 0.2, 10)])
    tight = nucleation_dispersion(tight_s, np.zeros(20), patch_r=8.0)
    assert cont["spatial_entropy"] > tight["spatial_entropy"]   # continuous spreads over more cells
    assert tight["top2_occupancy"] > cont["top2_occupancy"]     # tight blobs concentrate (WARNING)
    assert tight["top2_occupancy"] > 0.8
    d1 = nucleation_dispersion(np.array([0.0]), np.array([0.0]), patch_r=8.0)
    assert np.isnan(d1["spatial_entropy"])                      # n<2 safe, no crash


def test_compute_t0_gate_packages_and_filters_invalid():
    from src.sef_hfo_stage4 import compute_t0_gate
    # 12 valid centroids (continuous along axis) + 3 invalid (nan) that must be excluded
    s = list(np.linspace(-6, 6, 12)) + [np.nan, np.nan, np.nan]
    r = list(np.zeros(12)) + [np.nan, 0.0, np.nan]
    g = compute_t0_gate(np.array(s), np.array(r), patch_r=8.0)
    assert g["n_valid_nucleation"] == 12                       # 3 nan-rows dropped
    assert g["hotspot_degeneracy"]["verdict"] == "healthy"
    assert "spatial_entropy" in g["nucleation_dispersion"]
    # too few valid -> indeterminate (never a pass)
    g2 = compute_t0_gate(np.array([-3.0, 3.0, np.nan]), np.array([0.0, 0.0, np.nan]), patch_r=8.0)
    assert g2["n_valid_nucleation"] == 2
    assert g2["hotspot_degeneracy"]["verdict"] == "indeterminate_low_n"


def test_hotspot_elongation_aware(  ):
    # P1-3 (reviewer 2026-06-17): T0 gate must normalize to the (possibly elliptical) patch. An
    # elongated patch (elongation=1.5, patch_r=8 -> semi-major 12) with a continuous spread along the
    # LONG axis is HEALTHY; under isotropic patch_r the far-along-axis points would look artificially
    # spread. Two TIGHT blobs stay degenerate even when elongation-aware.
    s_cont = np.linspace(-11, 11, 18)
    el = hotspot_degeneracy(s_cont, np.zeros(18), patch_r=8.0, elongation=1.5)
    assert el["verdict"] == "healthy"
    rng = np.random.default_rng(0)
    s_blob = np.concatenate([rng.normal(-9, 0.3, 9), rng.normal(9, 0.3, 9)])
    td = hotspot_degeneracy(s_blob, np.zeros(18), patch_r=8.0, elongation=1.5)
    assert td["verdict"] == "two_hotspot_degenerate"
