from src.topic5_v2_band_scan import load_phase1_config
def test_config_rev2_keys():
    c = load_phase1_config()
    assert c["power"]["spectrogram_win_sec"] == 1.0
    assert c["epoch"]["field_window_sec"] == 10.0
    assert c["bands"]["primary_interval"] == "half_open"
    assert c["repro_bands"]["hfa"] == "legacy_hfa_60_100"
    assert c["nulls"]["n_perm_smoke"] == 20 and c["common_field"]["leave_one_band_out"] is True
