import numpy as np
import pytest

from scripts.analyze_topic4_zm_subtractive_pool_carrier import (
    adjudicate, arm_key, cv_block_profile, long_arm_key, long_run_class,
    modulation_amplitude_hz, modulation_band, spectral_peak,
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


def test_spectral_peak_does_not_read_a_ramp_as_a_rhythm():
    """A monotone drift has no period; reporting one would invent the result."""
    fs = 500.0
    t = np.arange(0, 4.0, 1.0 / fs)
    ramp = 150.0 + 60.0 * t / t[-1]
    peak_hz, prominence = spectral_peak(ramp, fs=fs)
    assert prominence < 10.0
    # A rhythm riding on the same drift must still be found, at its own rate.
    riding = ramp + 30.0 * np.sin(2 * np.pi * 18.0 * t)
    peak_hz, prominence = spectral_peak(riding, fs=fs)
    assert peak_hz == pytest.approx(18.0, abs=1.0)
    assert prominence > 10.0


def test_spectral_band_excludes_frequencies_the_window_cannot_resolve():
    """Fewer than a few cycles in the window is not a measured frequency."""
    fs = 500.0
    one_second = np.zeros(int(fs))
    # A 1 s window cannot support a 2 Hz claim under a five-cycle rule.
    assert spectral_peak(one_second, fs=fs, min_cycles=5.0)[0] is None or (
        spectral_peak(one_second, fs=fs, min_cycles=5.0)[0] >= 5.0
    )
    t = np.arange(0, 8.0, 1.0 / fs)
    slow = 200.0 + 20.0 * np.sin(2 * np.pi * 3.0 * t)
    # Eight seconds does support 3 Hz, so the same rule must admit it.
    assert spectral_peak(slow, fs=fs, min_cycles=5.0)[0] == pytest.approx(3.0, abs=0.5)


def test_cv_block_profile_shows_whether_modulation_decays():
    fs = 500.0
    t = np.arange(0, 6.0, 1.0 / fs)
    sustained = 200.0 + 60.0 * np.sin(2 * np.pi * 10.0 * t)
    profile = cv_block_profile(sustained, fs=fs, block_ms=2000.0)
    assert len(profile) == 3
    assert max(profile) - min(profile) < 0.05      # holds across the run
    decaying = 200.0 + 60.0 * np.exp(-t) * np.sin(2 * np.pi * 10.0 * t)
    decayed = cv_block_profile(decaying, fs=fs, block_ms=2000.0)
    assert decayed[0] > 5.0 * decayed[-1]          # the rhythm dies out


def _long(*, profile, gap, gate7=True):
    return {"cv_block_profile": profile, "post_onset_deep_gap_fraction": gap,
            "credible_carrier": gate7}


def test_a_sustained_burst_train_is_not_called_a_decay():
    """High strength holds its modulation; calling that a decay is simply wrong."""
    assert long_run_class(_long(
        profile=[0.979, 0.916, 0.883, 0.900, 0.912, 0.873], gap=0.348, gate7=False,
    )) == "persistent_deep_gap_burst_train"


def test_a_transient_that_settles_flat_is_a_decay():
    assert long_run_class(_long(
        profile=[0.849, 0.173, 0.041, 0.043, 0.039, 0.038], gap=0.005,
    )) == "decays_to_tonic_fixed_point"
    # A longer transient is still a decay, not a different outcome.
    assert long_run_class(_long(
        profile=[0.908, 0.754, 0.078, 0.040, 0.036, 0.037], gap=0.057,
    )) == "decays_to_tonic_fixed_point"


def test_a_run_that_never_modulated_is_reported_as_such():
    assert long_run_class(_long(
        profile=[0.06, 0.04, 0.04, 0.04, 0.04, 0.04], gap=0.0,
    )) == "tonic_throughout"


def test_the_target_outcome_needs_modulation_continuity_and_every_gate():
    assert long_run_class(_long(
        profile=[0.5, 0.48, 0.51, 0.49, 0.50, 0.52], gap=0.05,
    )) == "continuous_modulated_carrier"
    # Continuous and modulated but failing a gate is not the target outcome.
    assert long_run_class(_long(
        profile=[0.5, 0.48, 0.51, 0.49, 0.50, 0.52], gap=0.05, gate7=False,
    )) == "persistent_modulated_below_gate"


def test_long_arm_key_admits_a_run_with_no_short_counterpart():
    """A 12 s arm run on its own is evidence; dropping it would hide half a ladder."""
    assert long_arm_key(_summary(5.2228, 0.32, T_ms=12000.0)) == (5.2228, 0.32, 1)
    # The short-window key still refuses it, so the two panels stay separate.
    assert arm_key(_summary(5.2228, 0.32, T_ms=12000.0)) is None
    # Everything else about the locked comparison still has to match.
    assert long_arm_key(_summary(5.2228, 0.32, T_ms=12000.0, som_tau_d=30.)) is None
    assert long_arm_key(_summary(5.2228, 0.32, T_ms=2500.0)) is None


def test_modulation_amplitude_is_absolute_not_only_relative():
    """A large relative peak on a nearly flat trace is not a large rhythm."""
    fs = 500.0
    t = np.arange(0, 4.0, 1.0 / fs)
    flat_ish = 200.0 + 0.5 * np.sin(2 * np.pi * 47.0 * t)
    bursty = 100.0 + 90.0 * np.sin(2 * np.pi * 4.0 * t)
    assert modulation_amplitude_hz(flat_ish) < 1.0
    assert modulation_amplitude_hz(bursty) > 50.0


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
