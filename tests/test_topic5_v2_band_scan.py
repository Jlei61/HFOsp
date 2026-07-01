import numpy as np
import pytest
from src.topic5_v2_band_scan import (
    load_phase1_config, line_noise_bin_mask, band_bin_selection,
    masked_band_power_trace, robust_z_with_flags, channel_artifact_flags,
    contact_alignment, spatial_constrained_permute, common_field_residual,
    aperiodic_corrected_excess_power, confound_residual_rank,
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


def test_channel_artifact_flags_excludes_nan_frames_from_saturation_fraction():
    # A channel with SOME NaN frames (no baseline coverage in that stretch) plus a
    # saturating fraction among the FINITE frames only: NaN frames must be excluded
    # from the denominator of the saturation fraction, not counted as "not saturated"
    # (that would dilute a genuinely-saturated channel below sat_frac and hide it).
    logpower = np.zeros((2, 10))
    z = np.zeros((2, 10))
    z[1, :8] = np.nan             # 8/10 frames NaN for channel 1
    z[1, 8:] = 50.0                # both remaining finite frames saturate (|z|>12)
    fl = channel_artifact_flags(logpower, z, sat_abs_z=12.0, sat_frac=0.5, flatline_mad_eps=1e-9)
    assert fl["saturation"][1], "finite-only fraction = 2/2 = 1.0 > 0.5 must flag saturated"
    assert fl["bad_channel"][1]
    assert not fl["flatline"][1]          # 2 finite frames -> not all-nonfinite
    assert not fl["saturation"][0]        # untouched all-zero channel stays clean


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


def test_spatial_permute_within_shaft_no_cross_shaft_leakage_when_both_qualify():
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]        # both shafts >= min_group=4
    vals = {"A1": 0.0, "A2": 1.0, "A3": 2.0, "A4": 3.0,
            "B1": 10.0, "B2": 11.0, "B3": 12.0, "B4": 13.0}
    shaft = {n: n[0] for n in names}
    coord = {n: (float(i), 0.0) for i, n in enumerate(names)}
    perm, st = spatial_constrained_permute(names, vals, shaft, coord,
                                            np.random.default_rng(1), "within_shaft", 4)
    assert sorted(perm[n] for n in ("A1", "A2", "A3", "A4")) == [0.0, 1.0, 2.0, 3.0]   # A's own
    assert sorted(perm[n] for n in ("B1", "B2", "B3", "B4")) == [10.0, 11.0, 12.0, 13.0]  # B's own
    assert st["spatial_null_strength"] == "within_shaft_strong"
    assert st["n_effectively_permutable"] == 8


def test_spatial_permute_tier2_distance_bin_pools_two_small_shafts():
    names = ["C1", "C2", "D1", "D2"]                    # each shaft has 2 < min_group=4
    vals = {"C1": 1.0, "C2": 2.0, "D1": 3.0, "D2": 4.0}
    shaft = {"C1": "C", "C2": "C", "D1": "D", "D2": "D"}
    coord = {"C1": (0.0, 0.0), "C2": (1.0, 0.0), "D1": (2.0, 0.0), "D2": (3.0, 0.0)}
    perm, st = spatial_constrained_permute(names, vals, shaft, coord,
                                            np.random.default_rng(2), "within_shaft", 4)
    assert st["spatial_null_strength"] == "distance_bin_fallback"
    assert st["n_effectively_permutable"] == 4
    assert sorted(perm[n] for n in names) == [1.0, 2.0, 3.0, 4.0]   # pooled bin permutes together


def test_spatial_permute_tier3_subject_wide_real_shuffle_when_leftover_ge_2():
    names = ["E1", "E2", "F1"]                          # 3 leftover, below min_group=4,
    vals = {"E1": 5.0, "E2": 6.0, "F1": 7.0}            # too few even for one distance bin
    shaft = {"E1": "E", "E2": "E", "F1": "F"}
    coord = {"E1": (0.0, 0.0), "E2": (1.0, 0.0), "F1": (2.0, 0.0)}
    perm, st = spatial_constrained_permute(names, vals, shaft, coord,
                                            np.random.default_rng(3), "within_shaft", 4)
    assert st["spatial_null_strength"] == "subject_wide_weak"
    assert st["n_effectively_permutable"] == 3          # real shuffle applied (size>=2)
    assert sorted(perm[n] for n in names) == [5.0, 6.0, 7.0]


def test_spatial_permute_unsupported_mode_raises():
    names = ["A1", "A2"]
    vals = {"A1": 0.0, "A2": 1.0}
    shaft = {"A1": "A", "A2": "A"}
    coord = {"A1": (0.0, 0.0), "A2": (1.0, 0.0)}
    with pytest.raises(ValueError):
        spatial_constrained_permute(names, vals, shaft, coord,
                                     np.random.default_rng(0), "unconstrained", 4)


def test_spatial_permute_min_group_below_2_raises():
    names = ["A1", "A2"]
    vals = {"A1": 0.0, "A2": 1.0}
    shaft = {"A1": "A", "A2": "A"}
    coord = {"A1": (0.0, 0.0), "A2": (1.0, 0.0)}
    with pytest.raises(AssertionError):
        spatial_constrained_permute(names, vals, shaft, coord,
                                     np.random.default_rng(0), "within_shaft", 1)


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


def test_aperiodic_corrected_excess_power_zero_valid_band_bins_is_nan_not_zero():
    freqs = np.arange(1, 201, 1.0)
    slope_true, offset_true = -2.0, 2.0
    psd = 10 ** offset_true * freqs ** slope_true          # exact power law everywhere...
    band_idx = (freqs >= 90) & (freqs <= 110)
    psd = psd.copy()
    psd[band_idx] = np.nan                                  # ...except ZERO valid bins in [90,110]
    line_mask = np.zeros_like(freqs, dtype=bool)
    out = aperiodic_corrected_excess_power(freqs, psd, 90, 110, line_mask)  # fit range [1,200] intact
    assert out["ok"] is True                                # fit still succeeds off-band (179 valid pts)
    assert np.isnan(out["excess_power"])                    # zero real band bins != flat spectrum -> nan


def test_confound_residual_rank_single_always_computed_and_correct():
    names = [f"c{i}" for i in range(8)]
    rank = {n: 2.0 * i + 1.0 for i, n in enumerate(names)}          # G_HFO rank stand-in, linear in index
    collinear = {n: float(i) for i, n in enumerate(names)}          # rank = 2*collinear + 1 exactly
    unrelated = {n: abs(i - 3.5) for i, n in enumerate(names)}      # symmetric V-shape: exactly zero
                                                                     # correlation with the linear rank above
    out = confound_residual_rank(rank, {"collinear": collinear, "unrelated": unrelated})
    assert set(out["single"]) == {"collinear", "unrelated"}         # single always has every covariate
    assert all(abs(v) < 1e-9 for v in out["single"]["collinear"].values())      # exact line -> ~0
    mean_rank = float(np.mean(list(rank.values())))
    for n in names:                                                 # zero-corr cov -> slope 0, resid = rank - mean
        assert abs(out["single"]["unrelated"][n] - (rank[n] - mean_rank)) < 1e-9


def test_confound_residual_rank_combined_guarded_by_overfit_ratio():
    # below threshold: 2 covariates -> need >= 3*2+3=9 contacts; only 8 given -> combined None
    names8 = [f"c{i}" for i in range(8)]
    cov8 = {"cov1": {n: float(i) for i, n in enumerate(names8)},
            "cov2": {n: float(i % 3) for i, n in enumerate(names8)}}
    rank8 = {n: float(i) for i, n in enumerate(names8)}
    out_below = confound_residual_rank(rank8, cov8)
    assert out_below["single"] and out_below["combined"] is None    # single unaffected by the guard

    # at threshold: 9 contacts meets 3*2+3=9 exactly (>= is inclusive); rank is an EXACT
    # linear combo of the two covariates -> combined residuals ~0
    names9 = [f"c{i}" for i in range(9)]
    cov1 = {n: float(i) for i, n in enumerate(names9)}
    cov2 = {n: float(i % 3) for i, n in enumerate(names9)}
    rank9 = {n: 3.0 * cov1[n] - 2.0 * cov2[n] + 5.0 for n in names9}
    out_at = confound_residual_rank(rank9, {"cov1": cov1, "cov2": cov2})
    assert isinstance(out_at["combined"], dict)
    assert all(abs(v) < 1e-6 for v in out_at["combined"].values())


def test_confound_residual_rank_combined_uses_jointly_aligned_n_not_len_rank():
    # covariate map covers FEWER names than rank (3 entirely absent) -- the overfit guard
    # and the combined dict's count must both use the jointly-aligned n_contacts, not len(rank).
    names_rank = [f"c{i}" for i in range(10)]
    rank = {n: float(i) for i, n in enumerate(names_rank)}
    partial_cov = {f"c{i}": float(i) for i in range(7)}     # c7,c8,c9 missing entirely
    out = confound_residual_rank(rank, {"partial_cov": partial_cov})
    assert isinstance(out["combined"], dict)                 # 7 contacts >= 3*1+3=6 -> guard passes
    assert len(out["combined"]) == 7                         # jointly-aligned count...
    assert len(out["combined"]) < len(rank)                  # ...strictly fewer than len(rank)=10
    assert set(out["combined"]) == set(partial_cov)


def test_order_null_pair_preserves_counts_both_templates():
    from src.topic5_v2_band_scan import rebuild_typical_rank, order_null_rank_pair
    eb=np.array([[1,1,1,0],[1,1,1,1],[1,0,1,1]],bool); lag=np.array([[0,1,2,np.nan],[0,1,2,3],[0,np.nan,1,2]],float)
    r=rebuild_typical_rank(eb,lag); assert np.nanargmin(r)==0 and np.nanargmax(r)==3
    ra,rb=order_null_rank_pair(eb,lag,eb,lag,np.random.default_rng(0))
    assert (eb.sum(0)>0).tolist()==np.isfinite(ra).tolist()==np.isfinite(rb).tolist()


def _gate_kwargs(**overrides):
    # "everything passes" baseline (Gate A+B+C all True on a ripple band); each test
    # below flips exactly the variable(s) it is probing and leaves the rest passing.
    base = dict(spatial_p=0.01, spatial_delta=1.0, spatial_strength="within_shaft_strong",
                order_p=0.01, order_delta=1.0, order_strength="strong",
                common_resid_p=0.01, common_resid_delta=1.0,
                aperiodic_p=0.01, aperiodic_delta=1.0,
                band_max_over_bands_p=0.01, band="ripple_safe_80_220",
                fs_subset=True, alpha=0.05)
    base.update(overrides)
    return base


def test_gate_a_spatial_needs_within_shaft_strong_not_weaker_fallback_tiers():
    from src.topic5_v2_band_scan import gate_pass_flags
    assert gate_pass_flags(**_gate_kwargs())["gate_A_spatial"] is True
    for weak in ("subject_wide_weak", "distance_bin_fallback"):     # P1-c: neither weaker
        flags = gate_pass_flags(**_gate_kwargs(spatial_strength=weak))  # spatial-null tier
        assert flags["gate_A_spatial"] is False                     # can pass formal Gate A


def test_gate_a_order_rejects_weak_downgrade_strength():
    from src.topic5_v2_band_scan import gate_pass_flags
    assert gate_pass_flags(**_gate_kwargs())["gate_A_order"] is True
    flags = gate_pass_flags(**_gate_kwargs(order_strength="weak_downgrade"))
    assert flags["gate_A_order"] is False


def test_gate_a_requires_both_spatial_and_order_p1b():
    from src.topic5_v2_band_scan import gate_pass_flags
    assert gate_pass_flags(**_gate_kwargs())["gate_A"] is True                 # both pass
    spatial_only = gate_pass_flags(**_gate_kwargs(order_strength="weak_downgrade"))
    assert spatial_only["gate_A_spatial"] is True and spatial_only["gate_A"] is False
    order_only = gate_pass_flags(**_gate_kwargs(spatial_strength="subject_wide_weak"))
    assert order_only["gate_A_order"] is True and order_only["gate_A"] is False
    neither = gate_pass_flags(**_gate_kwargs(spatial_strength="subject_wide_weak",
                                              order_strength="weak_downgrade"))
    assert neither["gate_A"] is False


def test_gate_flags_reject_each_pvalue_and_delta_at_alpha_boundary():
    from src.topic5_v2_band_scan import gate_pass_flags
    # p==alpha (not <alpha) and delta==0 (not >0) must fail the SAME variable's own
    # sub-gate, not just ride along with some other passing variable in the rule.
    assert gate_pass_flags(**_gate_kwargs(spatial_p=0.05))["gate_A_spatial"] is False
    assert gate_pass_flags(**_gate_kwargs(spatial_delta=0.0))["gate_A_spatial"] is False
    assert gate_pass_flags(**_gate_kwargs(order_p=0.05))["gate_A_order"] is False
    assert gate_pass_flags(**_gate_kwargs(order_delta=0.0))["gate_A_order"] is False
    assert gate_pass_flags(**_gate_kwargs(common_resid_p=0.05))["gate_B_freq_specific"] is False
    assert gate_pass_flags(**_gate_kwargs(aperiodic_p=0.05))["gate_C_hfo_specific"] is False


def test_gate_flags_additional_alpha_boundary_and_order_strength_missing():
    from src.topic5_v2_band_scan import gate_pass_flags
    # band_max_over_bands_p==alpha (not <alpha): the observed band delta must beat the
    # max-over-bands null strictly, not tie it.
    assert gate_pass_flags(**_gate_kwargs(band_max_over_bands_p=0.05))["gate_B_freq_specific"] is False
    # common_resid_delta==0.0 (not >0): same boundary discipline as the other delta checks.
    assert gate_pass_flags(**_gate_kwargs(common_resid_delta=0.0))["gate_B_freq_specific"] is False
    # aperiodic_delta==0.0 (not >0): fails gate_C on its own.
    assert gate_pass_flags(**_gate_kwargs(aperiodic_delta=0.0))["gate_C_hfo_specific"] is False
    # order_strength=="missing" (anything other than the literal "weak_downgrade") must still
    # pass gate_A_order -- proves the check is a != negative-match against one banned value,
    # not a =="strong" positive allowlist.
    assert gate_pass_flags(**_gate_kwargs(order_strength="missing"))["gate_A_order"] is True


def test_gate_b_needs_positive_common_resid_delta_and_band_beats_max_over_bands_null():
    from src.topic5_v2_band_scan import gate_pass_flags
    assert gate_pass_flags(**_gate_kwargs())["gate_B_freq_specific"] is True
    neg_delta = gate_pass_flags(**_gate_kwargs(common_resid_delta=-1.0))
    assert neg_delta["gate_B_freq_specific"] is False
    not_band_specific = gate_pass_flags(**_gate_kwargs(band_max_over_bands_p=0.5))  # this band
    assert not_band_specific["gate_B_freq_specific"] is False        # is not beating the null
    gate_a_fails_first = gate_pass_flags(**_gate_kwargs(order_strength="weak_downgrade"))
    assert gate_a_fails_first["gate_B_freq_specific"] is False       # gate_B needs gate_A first


def test_gate_c_needs_ripple_band_membership_and_aperiodic_pass():
    from src.topic5_v2_band_scan import gate_pass_flags
    full_cohort = gate_pass_flags(**_gate_kwargs(band="ripple_safe_80_220"))
    assert full_cohort["gate_C_hfo_specific"] is True
    fs1024_subset = gate_pass_flags(**_gate_kwargs(band="ripple_full_80_250", fs_subset=False))
    assert fs1024_subset["gate_C_hfo_specific"] is True     # fs_subset itself doesn't gate --
                                                              # both ripple bands are eligible
    non_ripple = gate_pass_flags(**_gate_kwargs(band="gamma_LVFA"))
    assert non_ripple["gate_B_freq_specific"] is True and non_ripple["gate_C_hfo_specific"] is False
    bad_aperiodic = gate_pass_flags(**_gate_kwargs(aperiodic_delta=-1.0))
    assert bad_aperiodic["gate_C_hfo_specific"] is False


def test_gate_tier_maps_flags_to_interpretation():
    from src.topic5_v2_band_scan import gate_pass_flags, gate_tier
    strongest_flags = gate_pass_flags(**_gate_kwargs(band="ripple_safe_80_220"))
    assert gate_tier(strongest_flags, "ripple_safe_80_220") == "strongest"

    freq_specific_flags = gate_pass_flags(**_gate_kwargs(band="gamma_LVFA"))
    assert gate_tier(freq_specific_flags, "gamma_LVFA") == "frequency_specific"

    broadband_flags = gate_pass_flags(**_gate_kwargs(band_max_over_bands_p=0.5))
    assert gate_tier(broadband_flags, "ripple_safe_80_220") == "broadband_recruitment"

    weak_flags = gate_pass_flags(**_gate_kwargs(spatial_strength="subject_wide_weak"))
    assert gate_tier(weak_flags, "ripple_safe_80_220") == "weak_negative"
