import numpy as np
from src.topic5_v2_band_scan import (
    load_phase1_config, line_noise_bin_mask, band_bin_selection,
    masked_band_power_trace, robust_z_with_flags, channel_artifact_flags,
    contact_alignment, spatial_constrained_permute, common_field_residual,
    aperiodic_corrected_excess_power,
)
def test_config_rev2_keys():
    c = load_phase1_config()
    assert c["power"]["spectrogram_win_sec"] == 1.0
    assert c["epoch"]["field_window_sec"] == 10.0
    assert c["bands"]["primary_interval"] == "half_open"
    assert c["repro_bands"]["hfa"] == "legacy_hfa_60_100"
    assert c["nulls"]["n_perm_smoke"] == 20 and c["common_field"]["leave_one_band_out"] is True


def test_config_band_edges_and_null_params_locked():
    """Foundation lock: a silent edit to any load-bearing band edge / null param
    (e.g. ripple_high 150->105) would corrupt every downstream gate but pass the
    minimal smoke test above. Pin the exact values here."""
    c = load_phase1_config()
    assert c["bands"]["primary"] == [
        ["delta_HYP_slow", 1, 4], ["theta_preictal_PAC", 4, 8], ["alpha_sharp_leq13", 8, 13],
        ["beta_LVFA_low", 13, 30], ["gamma_LVFA", 30, 80], ["hg_low_ripple", 80, 150],
        ["ripple_high", 150, 250],
    ]
    assert c["bands"]["composites"] == [
        ["low_HYP_1_13", 1, 13], ["LVFA_13_80", 13, 80],
        ["ripple_full_80_250", 80, 250], ["ripple_safe_80_220", 80, 220],
    ]
    assert c["bands"]["composite_interval"] == "closed"
    assert c["repro_bands"]["bb"] == "legacy_bb_1_45"
    assert c["line_noise"]["harmonics_hz"] == [50, 100, 150, 200, 250]
    assert c["line_noise"]["halfwidth_hz"] == 2.0
    assert c["line_noise"]["min_effective_bandwidth_frac"] == 0.5
    assert c["edge"]["fs512_hi_safe_hz"] == 220.0
    n = c["nulls"]
    assert (n["seed"], n["alpha"], n["n_perm_final"]) == (20260701, 0.05, 1000)
    assert n["spatial"] == "within_shaft" and n["min_group_for_shaft"] == 4
    assert n["order_null_min_corr_to_geo"] == 0.90
    assert c["tolerances"]["legacy_subject_median_abs"] == 0.02
    assert c["cohorts"]["axis_sets"] == ["broad", "narrow"] and c["cohorts"]["never_pool_axis_sets"] is True


def test_half_open_bands_do_not_share_boundary_bin():
    f = np.arange(0, 251, 1.0); lm = line_noise_bin_mask(f, [50,100,150,200,250], 2.0)
    # delta [1,4) and theta [4,8): 4 Hz belongs to theta only
    dmask,_,_ = band_bin_selection(f, 1, 4, lm, half_open=True)
    tmask,_,_ = band_bin_selection(f, 4, 8, lm, half_open=True)
    assert not dmask[f==4].any() and tmask[f==4].all()
    # composite closed keeps hi
    cmask,_,_ = band_bin_selection(f, 80, 250, lm, half_open=False)
    assert not cmask[f==250].any()             # 250 is line harmonic -> masked anyway
    _, eff_bb, _ = band_bin_selection(f, 1, 45, lm, half_open=False)
    assert eff_bb == 1.0


def test_band_power_flags_and_edge():
    rng=np.random.default_rng(0); fs=1024.0; sig=rng.standard_normal((4,int(fs*40)))
    out=masked_band_power_trace(sig, fs, 80, 250, 1.0, 0.1, [50,100,150,200,250], 2.0, 220.0)
    assert out["fs_edge_flag"] is False and 0<out["eff_frac"]<1
    out512=masked_band_power_trace(sig[:,:int(512*40)], 512.0, 80, 250, 1.0, 0.1, [50,100,150,200,250], 2.0, 220.0)
    assert out512["fs_edge_flag"] is True
    n_t=out["logpower"].shape[1]; z,low=robust_z_with_flags(out["logpower"], (0,n_t//3), 0.1, 1.0)
    zz=z.copy(); zz[1,:]=50.0                                        # saturate ch1
    fl=channel_artifact_flags(out["logpower"], zz, 12.0, 0.02, 1e-9)
    assert fl["saturation"][1] and fl["bad_channel"][1]


def test_signed_orientation_is_fixed_not_posthoc():
    names=[f"c{i}" for i in range(8)]
    ra={n:float(i) for i,n in enumerate(names)}; rb={n:float(7-i) for i,n in enumerate(names)}
    vals={n:float(7-i) for i,n in enumerate(names)}     # tracks B (anti-A)
    out=contact_alignment(vals, ra, rb, oriented_template="a")
    assert out["signed_spearman_a"] < -0.9              # anti-correlated with A
    assert out["align_signed_oriented"] == out["signed_spearman_a"]   # fixed to A regardless
    assert out["align_signed_posthoc_max"] > 0.9        # posthoc would pick B (positive)


def test_spatial_fallback_reports_strength():
    names=["A1","A2","A3","A4","B1"]                      # B has 1 -> singleton
    vals={n:float(i) for i,n in enumerate(names)}
    shaft={"A1":"A","A2":"A","A3":"A","A4":"A","B1":"B"}
    coord={n:(float(i),0.0) for i,n in enumerate(names)}
    perm,st=spatial_constrained_permute(names,vals,shaft,coord,np.random.default_rng(0),"within_shaft",4)
    assert sorted(perm[n] for n in ["A1","A2","A3","A4"])==[0.0,1.0,2.0,3.0]
    assert st["n_singleton_groups"]>=1 and "spatial_null_strength" in st


def test_common_field_residual_collinear_is_zero():
    names = [f"c{i}" for i in range(8)]
    cf = {n: v for n, v in zip(names, [1.0, 2.0, 3.5, 4.0, 6.0, 7.5, 9.0, 10.0])}
    band = {n: 2.0 * cf[n] + 1.0 for n in names}          # exact line -> nothing left over
    resid = common_field_residual(band, cf)
    assert set(resid) == set(names)
    assert all(abs(resid[n]) < 1e-9 for n in names)


def test_common_field_residual_band_specific_bump_survives():
    names = [f"c{i}" for i in range(30)]                  # enough contacts that one outlier
    cf = {n: float(i) for i, n in enumerate(names)}        # doesn't dominate the OLS fit
    band = dict(cf)                                        # band tracks common field 1:1...
    bump, delta = names[15], 5.0
    band[bump] = cf[bump] + delta                          # ...except one spatially-specific bump
    resid = common_field_residual(band, cf)
    assert abs(resid[bump] - delta) < 0.3                  # bump survives residualization (Gate B)
    assert all(abs(resid[n]) < 0.3 for n in names if n != bump)


def test_common_field_residual_below_three_shared_points_is_empty():
    cf = {"c0": 1.0, "c1": 2.0}                            # only 2 shared finite points
    band = {"c0": 2.0, "c1": 4.0}
    assert common_field_residual(band, cf) == {}


def test_common_field_residual_excludes_nonshared_and_nonfinite():
    names = [f"c{i}" for i in range(6)]
    cf = {n: v for n, v in zip(names, [1.0, 2.0, 3.5, 4.0, 6.0, 7.5])}
    band = {n: 2.0 * cf[n] + 1.0 for n in names}           # exact line -> clean residual ~0
    band["nan_in_band"], cf["nan_in_band"] = np.nan, 100.0  # band value NaN, cf finite
    band["nan_in_cf"], cf["nan_in_cf"] = 100.0, np.nan      # band finite, cf value NaN
    band["band_only"] = 999.0                               # key present in band dict only
    cf["cf_only"] = 999.0                                   # key present in cf dict only
    resid = common_field_residual(band, cf)
    assert set(resid) == set(names)                         # all 4 injected contacts excluded
    assert all(abs(resid[n]) < 1e-9 for n in names)          # exclusion happened before the OLS
                                                              # fit -> clean contacts undisturbed


def test_aperiodic_corrected_excess_power_recovers_pure_1f():
    freqs = np.arange(1, 201, 1.0)
    slope_true, offset_true = -2.0, 2.0
    psd = 10 ** offset_true * freqs ** slope_true          # exact power law, no bump anywhere
    line_mask = np.zeros_like(freqs, dtype=bool)
    out = aperiodic_corrected_excess_power(freqs, psd, 90, 110, line_mask)
    assert out["ok"] is True
    assert abs(out["slope"] - slope_true) < 1e-6
    assert out["fit_r2"] > 1 - 1e-6
    assert abs(out["excess_power"]) < 1e-6                 # no bump -> ~0 (floating-point noise only)


def test_aperiodic_corrected_excess_power_detects_band_localized_bump():
    freqs = np.arange(1, 201, 1.0)
    psd_pure = 10 ** 2.0 * freqs ** -2.0
    line_mask = np.zeros_like(freqs, dtype=bool)
    bump = 0.05 * np.exp(-0.5 * ((freqs - 100.0) / 5.0) ** 2)   # localized Gaussian bump @100Hz
    psd_bump = psd_pure + bump
    out_bump = aperiodic_corrected_excess_power(freqs, psd_bump, 90, 110, line_mask)
    out_ctrl = aperiodic_corrected_excess_power(freqs, psd_bump, 30, 50, line_mask)  # no bump here
    assert out_bump["ok"] is True and out_ctrl["ok"] is True
    assert 0.5 <= out_bump["fit_r2"] < 1.0                 # a real (non-iterative) fit is nudged
                                                            # down by the bump, yet still clears min_r2
    assert out_bump["excess_power"] > 0.3                  # clear detection (true bump mass ~0.6)
    assert out_ctrl["excess_power"] < 0.05                 # no bump present -> near zero
    assert out_bump["excess_power"] > out_ctrl["excess_power"] + 0.2   # clearly larger, not tautological
