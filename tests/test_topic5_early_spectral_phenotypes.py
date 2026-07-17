from scripts.plot_topic5_early_spectral_phenotypes import (
    BANDS,
    classify_band_hits,
    classify_overlap_state,
    classify_simple_state,
    is_early_anchor,
)


def _hits(*active: str) -> dict[str, bool]:
    return {band: band in set(active) for band in BANDS}


def test_broadband_requires_low_and_fast_support():
    assert (
        classify_band_hits(_hits(*BANDS))
        == "broadband_1_150"
    )
    assert (
        classify_band_hits(_hits(*BANDS[:2], *BANDS[3:]))
        == "broadband_1_150"
    )


def test_gamma_plus_fast_without_low_is_fast_frequency_dominant():
    assert (
        classify_band_hits(_hits("beta_LVFA_low", "gamma_LVFA"))
        == "fast_frequency_dominant_13_150"
    )
    assert (
        classify_band_hits(
            _hits("alpha_sharp_leq13", "gamma_LVFA", "hg_low_ripple")
        )
        == "fast_frequency_dominant_13_150"
    )


def test_low_frequency_support_without_fast_is_low_frequency_dominant():
    assert (
        classify_band_hits(_hits("delta_HYP_slow", "theta_preictal_PAC"))
        == "low_frequency_dominant_le13"
    )


def test_mixed_or_isolated_patterns_remain_other():
    assert (
        classify_band_hits(_hits("delta_HYP_slow", "gamma_LVFA")) == "other"
    )
    assert classify_band_hits(_hits("gamma_LVFA")) == "other"


def test_early_anchor_domain_is_inclusive_and_rejects_late_recruitment():
    assert is_early_anchor(-15.0)
    assert is_early_anchor(20.0)
    assert not is_early_anchor(-15.01)
    assert not is_early_anchor(20.01)


def test_overlap_states_partition_broadband_gamma_and_low_support():
    assert (
        classify_overlap_state(_hits(*BANDS))
        == "broadband_gamma_low_overlap"
    )
    assert (
        classify_overlap_state(
            _hits(
                "delta_HYP_slow",
                "theta_preictal_PAC",
                "alpha_sharp_leq13",
                "beta_LVFA_low",
                "hg_low_ripple",
            )
        )
        == "broadband_low_no_gamma"
    )
    assert (
        classify_overlap_state(
            _hits("delta_HYP_slow", "theta_preictal_PAC", "gamma_LVFA")
        )
        == "gamma_low_nonbroadband"
    )
    assert classify_overlap_state(_hits("gamma_LVFA")) == "gamma_only"
    assert (
        classify_overlap_state(_hits("delta_HYP_slow", "theta_preictal_PAC"))
        == "low_frequency_only"
    )
    assert classify_overlap_state(_hits("beta_LVFA_low")) == "neither_defined_support"


def test_simple_state_uses_broadband_then_gamma_then_low_priority():
    assert classify_simple_state(_hits(*BANDS)) == "broadband_1_150"
    assert (
        classify_simple_state(
            _hits("delta_HYP_slow", "theta_preictal_PAC", "gamma_LVFA")
        )
        == "gamma_nonbroadband"
    )
    assert classify_simple_state(_hits("gamma_LVFA")) == "gamma_nonbroadband"
    assert (
        classify_simple_state(_hits("delta_HYP_slow", "theta_preictal_PAC"))
        == "low_frequency_only"
    )
    assert classify_simple_state(_hits("beta_LVFA_low")) == "other"
