from scripts.analyze_topic4_frozen_q_atlas import _classify_static_mode


def _inputs():
    rate = {
        "median_rate_hz": 180.0,
        "minimum_subwindow_median_rate_hz": 150.0,
    }
    recruitment = {"joint_global_recruitment_duty": 0.9}
    rhythm = {
        "contact_fraction_consistently_rhythmic": 0.9,
        "median_contact_peak_hz": 44.0,
        "contact_peak_mad_hz": 4.0,
        "median_peak_power_fraction": 0.4,
        "median_band_power_ratio_over_q1_reference": 3.0,
    }
    return rate, recruitment, rhythm


def test_static_atlas_gate_accepts_stable_global_narrowband_mode():
    got = _classify_static_mode(0.7, *_inputs())
    assert got["all_checks_pass"] is True


def test_static_atlas_gate_rejects_tonic_contact_signal():
    rate, recruitment, rhythm = _inputs()
    rhythm["median_peak_power_fraction"] = 0.01
    got = _classify_static_mode(0.7, rate, recruitment, rhythm)
    assert got["all_checks_pass"] is False
    assert got["checks"]["rhythm_is_narrowband"] is False


def test_q1_reference_does_not_need_to_exceed_itself():
    rate, recruitment, rhythm = _inputs()
    rhythm["median_band_power_ratio_over_q1_reference"] = 1.0
    got = _classify_static_mode(1.0, rate, recruitment, rhythm)
    assert got["checks"]["band_power_exceeds_q1_reference"] is True
