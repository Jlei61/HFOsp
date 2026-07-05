from src.topic5_v3_mode_transition import load_v3_config


def test_v3c_config_keys():
    v = load_v3_config()["v3c"]
    assert v["z_cross"] == 2.0 and v["window_sec"] == 30.0 and v["hop_sec"] == 0.1
    assert v["assay_qc"]["t0_frac_max"] == 0.50 and v["assay_qc"]["finite_frac_min"] == 0.40
    assert v["interpretation"]["auc_hb_min"] == 0.60 and v["interpretation"]["auc_ha_band"] == [0.45, 0.55]
    assert v["latency"]["min_surplus"] == 3 and v["latency"]["min_covered_soz"] == 3
    assert v["spatial"]["min_subjects_for_primary"] == 3
    assert v["cohorts"]["primary"] == "broad" and v["nulls"]["n_perm"] == 1000
