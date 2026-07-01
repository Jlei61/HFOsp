from src.topic5_v2_band_scan import load_phase1_config
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
