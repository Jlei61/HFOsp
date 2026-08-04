import numpy as np
import pytest

from scripts.analyze_topic4_zm_subtractive_pool_carrier import (
    adjudicate, arm_key, modulation_band, spectral_peak,
)


def _summary(beta, g, *, seed=1, state="bounded_late__peak", T_ms=2500.0,
             rho=0.0, som_tau_d=60.0, e_exc=60.0):
    mech = {
        "pv_som_inhibitory_subtypes": {
            "som_source_fraction_realized": .25,
            "som_slow_integrated_budget_fraction": .35,
            "som_recruit_delay_scale": 3., "tau_d_som_ms": som_tau_d,
            "seed": seed,
        },
        "state_selective_mode_H": {
            "rho_mode_H": rho, "tau_mode_H": 250., "tau_mode_H_down": 250.,
            "mode_H_common_subtraction": 0., "mode_H_persistent_g_max": g,
            "mode_H_persistent_e_exc": e_exc, "m_mode_half": 30.,
        },
    }
    if beta > 0.0:
        mech["subtractive_pool"] = {"beta_SG": beta, "alpha_G": 16.0,
                                    "tau_S_ms": 120.0, "S_max": 1.0}
    return {"state": state, "T_ms": T_ms, "mechanism": mech}


def test_arm_key_carries_strength_substrate_and_wiring():
    assert arm_key(_summary(1.9585, 0.32)) == (1.9585, 0.32, 1)
    assert arm_key(_summary(0.0, 0.0)) == (0.0, 0.0, 1)
    assert arm_key(_summary(1.9585, 0.32, seed=3)) == (1.9585, 0.32, 3)


def test_arm_key_refuses_anything_outside_the_locked_comparison():
    assert arm_key(_summary(1.9585, 0.32, som_tau_d=30.)) is None
    assert arm_key(_summary(1.9585, 0.32, e_exc=40.)) is None
    assert arm_key(_summary(1.9585, 0.32, rho=0.5)) is None
    assert arm_key(_summary(1.9585, 0.32, state="bounded_mid__peak")) is None
    # A substrate level outside the locked factorial is not a third row.
    assert arm_key(_summary(1.9585, 0.16)) is None


def test_modulation_band_is_the_three_bands_the_spec_locked():
    assert modulation_band(0.048) == "tonic"
    assert modulation_band(0.099) == "tonic"
    assert modulation_band(0.10) == "ambiguous"
    assert modulation_band(0.25) == "ambiguous"
    assert modulation_band(0.2501) == "clean"
    assert modulation_band(0.532) == "clean"
    assert modulation_band(None) is None


def test_spectral_peak_finds_a_rhythm_and_reports_its_prominence():
    fs = 500.0                                  # 2 ms bins
    t = np.arange(0, 4.0, 1.0 / fs)
    rhythmic = 200.0 + 40.0 * np.sin(2 * np.pi * 12.0 * t)
    peak_hz, prominence = spectral_peak(rhythmic, fs=fs)
    assert peak_hz == pytest.approx(12.0, abs=1.0)
    assert prominence > 10.0
    # A constant carries no rhythm, so the prominence must not manufacture one.
    flat_hz, flat_prom = spectral_peak(np.full(t.size, 200.0), fs=fs)
    assert flat_prom < 10.0


def _row(*, beta, g, gate7, cv, seed=1):
    return {"beta_SG": beta, "persistent_g": g, "som_seed": seed,
            "credible_carrier": gate7, "sustained_core_cv": cv,
            "modulation_band": modulation_band(cv),
            "post_onset_deep_gap_fraction": 0.05 if gate7 else 0.7}


def test_a_clean_candidate_needs_all_seven_gates_and_a_clean_band():
    verdict = adjudicate([
        _row(beta=0.0, g=0.32, gate7=True, cv=0.048),
        _row(beta=0.98, g=0.32, gate7=True, cv=0.40),
    ])
    assert verdict["verdict"] == "SUBTRACTIVE_POOL_MODULATED_CANDIDATE"
    assert verdict["candidate_arms"] == [{"beta_SG": 0.98, "som_seed": 1}]


def test_a_clean_band_that_fails_a_gate_is_not_a_candidate():
    """The arm that misses energy occupancy is informative, not a candidate."""
    verdict = adjudicate([
        _row(beta=0.0, g=0.32, gate7=True, cv=0.048),
        _row(beta=1.9585, g=0.32, gate7=False, cv=0.532),
    ])
    assert verdict["verdict"] != "SUBTRACTIVE_POOL_MODULATED_CANDIDATE"
    assert verdict["candidate_arms"] == []
    assert verdict["broke_the_fixed_point"] is True


def test_breaking_the_fixed_point_is_reported_separately_from_passing():
    """Two different findings; collapsing them would overstate the result."""
    none_broken = adjudicate([
        _row(beta=0.0, g=0.32, gate7=True, cv=0.048),
        _row(beta=0.39, g=0.32, gate7=True, cv=0.06),
    ])
    assert none_broken["broke_the_fixed_point"] is False
    assert none_broken["verdict"] == "SUBTRACTIVE_POOL_LEAVES_THE_FIXED_POINT_INTACT"


def test_the_bare_substrate_row_decides_whether_the_two_are_complementary():
    """If the subtractive term alone already carries, the excitation is surplus."""
    verdict = adjudicate([
        _row(beta=0.0, g=0.32, gate7=True, cv=0.048),
        _row(beta=0.98, g=0.32, gate7=True, cv=0.40),
        _row(beta=0.98, g=0.0, gate7=True, cv=0.40),
    ])
    assert verdict["verdict"] == "SUBTRACTIVE_POOL_CARRIES_WITHOUT_THE_EXCITATION"
    verdict_needs_both = adjudicate([
        _row(beta=0.0, g=0.32, gate7=True, cv=0.048),
        _row(beta=0.98, g=0.32, gate7=True, cv=0.40),
        _row(beta=0.98, g=0.0, gate7=False, cv=1.8),
    ])
    assert verdict_needs_both["verdict"] == "SUBTRACTIVE_POOL_MODULATED_CANDIDATE"


def test_replicate_wirings_are_reported_but_never_pooled_with_wiring_one():
    verdict = adjudicate([
        _row(beta=0.0, g=0.32, gate7=True, cv=0.048),
        _row(beta=1.9585, g=0.32, gate7=True, cv=0.53),
        _row(beta=1.9585, g=0.32, gate7=False, cv=0.51, seed=2),
    ])
    assert verdict["candidate_arms"] == [{"beta_SG": 1.9585, "som_seed": 1}]
    assert verdict["wiring_replication"]["1.9585"] == {
        "seeds_tested": [1, 2], "seeds_passing": [1],
    }
