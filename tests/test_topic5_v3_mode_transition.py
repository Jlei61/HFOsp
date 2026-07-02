from src.topic5_v3_mode_transition import load_v3_config


def test_v3_config_keys():
    c = load_v3_config()
    assert c["phases"]["I1_rel"] == [10.0, 30.0] and c["dynamics"]["finite_horizon_k"] == 3
    assert c["dynamics"]["demean_within_window"] is True and c["dynamics"]["mode_shift_normalization"] == "density"
    assert c["avalanche"]["flux_normalization"] == "source_mean"
    assert c["statistics"]["co_primary_correction"] == "holm" and c["cohorts"]["primary"] == "narrow"
