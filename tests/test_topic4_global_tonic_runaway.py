from src.topic4_global_recruited_oscillation import (
    classify_global_tonic_runaway,
)


def _rates():
    return {
        "median_pre_hz": 57.45,
        "q95_pre_hz": 102.0,
        "median_post_hz": 394.3,
        "q05_post_hz": 346.0,
        "median_ratio_post_over_pre": 6.86,
    }


def _recruitment():
    return {
        "median_active_neuron_fraction_20ms": 1.0,
        "median_recruited_spatial_fraction_1mm": 1.0,
        "joint_global_recruitment_duty": 1.0,
    }


def test_b0_like_tonic_plateau_is_positive_without_rhythm_fields():
    result = classify_global_tonic_runaway(
        onset_ms=540.0,
        observed_post_transition_ms=1711.6,
        rates=_rates(),
        recruitment=_recruitment(),
    )
    assert result["status"] == "TONIC_GLOBAL_RUNAWAY"
    assert result["all_checks_pass"]
    assert "30-80 Hz contact peak" in result["explicitly_not_required"]


def test_400ms_low_state_is_readable_without_reusing_rhythm_pre_window():
    result = classify_global_tonic_runaway(
        onset_ms=400.0,
        observed_post_transition_ms=1731.0,
        rates=_rates(),
        recruitment=_recruitment(),
    )
    assert result["all_checks_pass"]


def test_immediate_runaway_without_300ms_low_state_is_rejected():
    result = classify_global_tonic_runaway(
        onset_ms=250.0,
        observed_post_transition_ms=1731.0,
        rates=_rates(),
        recruitment=_recruitment(),
    )
    assert not result["all_checks_pass"]
    assert not result["checks"]["readable_low_state_dwell"]


def test_deep_high_state_troughs_do_not_reintroduce_a_modulation_gate():
    rates = _rates()
    rates["q05_post_hz"] = 224.5
    result = classify_global_tonic_runaway(
        onset_ms=620.0,
        observed_post_transition_ms=1729.3,
        rates=rates,
        recruitment=_recruitment(),
    )
    assert result["all_checks_pass"]
    assert result["observed"]["post_q05_rate_hz"] == 224.5


def test_deeply_modulated_but_global_high_state_is_in_scope():
    recruitment = _recruitment()
    recruitment.update({
        "median_active_neuron_fraction_20ms": 0.899,
        "median_recruited_spatial_fraction_1mm": 0.896,
        "joint_global_recruitment_duty": 0.83,
    })
    rates = _rates()
    rates.update({"median_post_hz": 349.2, "q05_post_hz": 126.5})
    result = classify_global_tonic_runaway(
        onset_ms=400.0,
        observed_post_transition_ms=1731.0,
        rates=rates,
        recruitment=recruitment,
    )
    assert result["all_checks_pass"]


def test_high_rate_without_global_recruitment_is_not_tonic_global_runaway():
    recruitment = _recruitment()
    recruitment["joint_global_recruitment_duty"] = 0.5
    result = classify_global_tonic_runaway(
        onset_ms=540.0,
        observed_post_transition_ms=1711.6,
        rates=_rates(),
        recruitment=recruitment,
    )
    assert not result["all_checks_pass"]
    assert not result["checks"]["global_plateau_is_sustained"]


def test_short_post_record_fails_persistence_even_if_near_saturated():
    result = classify_global_tonic_runaway(
        onset_ms=540.0,
        observed_post_transition_ms=500.0,
        rates=_rates(),
        recruitment=_recruitment(),
    )
    assert not result["all_checks_pass"]
    assert not result["checks"]["global_plateau_is_sustained"]
