import numpy as np

from src.topic4_zm_phase_diagram import (
    adjudicate_seed_family,
    classify_paired_initial_states,
    classify_stationary_branch,
    stationary_metrics,
)


def _metrics(rate, active, sheet, duty, q95=None, longest_high_ms=0.0):
    return {
        "median_rate_hz": rate,
        "q95_rate_hz": rate if q95 is None else q95,
        "median_active_E_fraction_20ms": active,
        "median_recruited_sheet_fraction_1mm": sheet,
        "joint_global_recruitment_duty": duty,
        "longest_rate_ge_120_hz_ms": longest_high_ms,
    }


def test_branch_classifier_preserves_low_intermediate_and_tonic_high():
    low = classify_stationary_branch(_metrics(55, 0.2, 0.2, 0.1, q95=95))
    middle = classify_stationary_branch(_metrics(140, 0.6, 0.6, 0.5))
    high = classify_stationary_branch(_metrics(390, 1.0, 1.0, 1.0))
    assert low["label"] == "LOW"
    assert middle["label"] == "INTERMEDIATE"
    assert high["label"] == "TONIC_HIGH"


def test_low_state_allows_isolated_events_but_not_sustained_high_rate():
    eventful_low = classify_stationary_branch(
        _metrics(55, 0.2, 0.2, 0.05, q95=150, longest_high_ms=40))
    persistent = classify_stationary_branch(
        _metrics(55, 0.2, 0.2, 0.05, q95=90, longest_high_ms=100))
    assert eventful_low["label"] == "LOW"
    assert persistent["label"] == "INTERMEDIATE"


def test_paired_initial_state_labels_are_fail_closed():
    assert classify_paired_initial_states("LOW", "TONIC_HIGH") == (
        "BISTABLE_CANDIDATE")
    assert classify_paired_initial_states("LOW", "LOW") == (
        "LOW_MONOSTABLE_CANDIDATE")
    assert classify_paired_initial_states("TONIC_HIGH", "TONIC_HIGH") == (
        "HIGH_MONOSTABLE_CANDIDATE")
    assert classify_paired_initial_states("INTERMEDIATE", "TONIC_HIGH") == (
        "MIXED_OR_UNRESOLVED")


def test_family_requires_three_of_three_for_robust_bistability():
    robust = adjudicate_seed_family(["BISTABLE_CANDIDATE"] * 3)
    partial = adjudicate_seed_family(
        ["BISTABLE_CANDIDATE", "BISTABLE_CANDIDATE",
         "LOW_MONOSTABLE_CANDIDATE"])
    incomplete = adjudicate_seed_family(["BISTABLE_CANDIDATE"] * 2)
    assert robust["verdict"] == "ROBUST_SNN_BISTABILITY_CANDIDATE"
    assert partial["verdict"] == (
        "STOCHASTIC_OR_METASTABLE_BISTABILITY_CANDIDATE")
    assert incomplete["verdict"] == "INCOMPLETE_SEED_DENOMINATOR"


def test_stationary_metrics_uses_only_post_burn_in_segment():
    dt_ms = 1.0
    n_steps, n_e = 200, 20
    spikes = np.zeros((n_steps, n_e), bool)
    spikes[:100, :] = True
    spikes[100:, :2] = True
    rate = np.r_[np.full(100, 500.0), np.full(100, 100.0)]
    x = np.linspace(0.1, 1.9, n_e)
    positions = np.c_[x, np.full(n_e, 1.0)]
    got = stationary_metrics(
        rate, spikes, positions, dt_ms=dt_ms, sheet_l_mm=2.0,
        burn_in_ms=100.0,
    )
    assert got["scoring_duration_ms"] == 100.0
    assert got["median_rate_hz"] == 100.0
    assert got["median_active_E_fraction_20ms"] == 0.1
    assert got["first_half_median_rate_hz"] == 100.0
    assert got["longest_rate_ge_120_hz_ms"] == 0.0
