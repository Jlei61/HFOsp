from scripts.run_topic4_zm_target_stage1b_controller import _candidates


def test_stage1b_grid_preserves_frozen_adaptation_mass_formula():
    config = {
        "fit": {
            "reference_i_th": 100.0,
            "reference_eta_m": 0.01,
            "stage1b_s_i": 0.8,
            "stage1b_tau_z_ms": 2500.0,
            "stage1b_tau_m_ms": [62.5, 250.0, 500.0, 1000.0],
            "stage1b_g_m_ratio": [0.5, 1.0, 1.5],
            "stage1b_eta_tau_reference": 5.0,
        }
    }
    candidates = _candidates(config)
    assert len(candidates) == 12
    for row in candidates:
        p = row["parameters"]
        ratio = p["eta_m"] * p["tau_adp"] / 5.0
        assert ratio in {0.5, 1.0, 1.5}
        assert p["I_th_EI"] == 80.0
        assert p["tau_z"] == 2500.0
        assert p["E_to_E_dose"] == p["E_to_I_dose"] == 1.0
