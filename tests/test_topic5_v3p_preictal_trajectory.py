from src.topic5_v3p_preictal_trajectory import load_v3p_config

def test_v3p_config_keys():
    c = load_v3p_config()
    assert c["preictal"]["phases"] == ["P0", "P1", "P2", "P3"]
    assert c["preictal"]["span_full_rel"] == [-120.0, -10.0]
    assert c["preictal"]["span_guard_rel"] == [-120.0, -20.0]      # rev1 onset guard
    assert c["trend"]["estimator"] == "theil_sen"
    assert c["gates"]["h3b_require"] == ["p_label", "p_rate", "lag1_specific_positive"]
    assert c["gates"]["h3c_require"] == ["p_label", "p_phase", "p_block"]
    assert c["gates"]["require_both_spans"] is True
    assert c["nulls_v3p"]["rate_null_per_window"] is True and c["nulls_v3p"]["time_order_null"] is True
    assert c["residualization"]["primary_adjudicator"] == "label_null_slope"
    assert c["co_primary"]["correction"] == "holm"
    assert c["co_primary"]["endpoints"] == ["net_offaxis_flux_surplus_slope", "mode_shift_density_surplus_slope"]
    assert c["inherit_v3_config"] == "topic5_v3" and c["cohorts"]["primary"] == "narrow"
