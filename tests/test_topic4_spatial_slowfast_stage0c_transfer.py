from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.sef_hfo_lif import TAU_ME, TREF_E, lif_rate
from src.topic4_spatial_slowfast_stage0b import ForkClassifierThresholds
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state
from src.topic4_spatial_slowfast_stage0c import moments_from_state as primary_moments_from_state
from src.topic4_spatial_slowfast_stage0c_transfer import (
    ExtendedSiegertTransfer,
    TransferResolution,
    TransferSupport,
    classify_extended_batch,
    direct_exact_error_audit,
    moments_from_prepared,
    prepare_pool_parameters,
    resolution_pair_status,
    simulate_extended_forks,
    stable_siegert_log_rate,
    stable_siegert_rate,
    transfer_axes,
    temporal_refinement_status,
)


def test_stable_siegert_matches_canonical_on_regular_overlap() -> None:
    for mu in (-30.0, -10.0, 0.0, 10.0, 18.0, 40.0, 100.0):
        for sigma in (2.0, 5.0, 10.0, 20.0, 30.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = lif_rate(mu, sigma, TAU_ME, TREF_E)
            observed = stable_siegert_rate(mu, sigma, TAU_ME, TREF_E)
            assert observed == pytest.approx(expected, rel=2e-10, abs=1e-13)


def test_extreme_low_mu_has_finite_log_rate_and_is_monotone() -> None:
    mus = np.asarray([-2500.0, -1800.0, -1000.0, -500.0, -250.0, -100.0, -40.0])
    log_rates = np.asarray([stable_siegert_log_rate(mu, 3.0, TAU_ME, TREF_E) for mu in mus])
    assert np.all(np.isfinite(log_rates))
    assert np.all(np.diff(log_rates) > 0.0)
    assert stable_siegert_rate(-2500.0, 3.0, TAU_ME, TREF_E) == 0.0


def test_endpoint_branch_is_continuous() -> None:
    sigma = 3.0
    branch_mu = 18.0 - 6.0 * sigma
    left = stable_siegert_log_rate(branch_mu - 1e-7, sigma, TAU_ME, TREF_E)
    right = stable_siegert_log_rate(branch_mu + 1e-7, sigma, TAU_ME, TREF_E)
    assert abs(left - right) < 2e-5


def _tiny_transfer() -> ExtendedSiegertTransfer:
    support = TransferSupport(-80.0, -20.0, 40.0, 2.0, 10.0)
    resolution = TransferResolution("test", 2.0, 1.0, 8)
    return ExtendedSiegertTransfer.build(support, resolution)


def test_extended_transfer_never_clips_or_extrapolates() -> None:
    transfer = _tiny_transfer()
    inside = transfer.rate(np.asarray([-20.0, 10.0]), np.asarray([3.0, 5.0]), "E")
    outside = transfer.rate(np.asarray([-81.0, 10.0]), np.asarray([3.0, 11.0]), "E")
    assert np.all(np.isfinite(inside))
    assert np.all(np.isnan(outside))
    assert transfer.support_mask(np.asarray([-20.0]), np.asarray([3.0])).item()
    assert not transfer.support_mask(np.asarray([-81.0]), np.asarray([3.0])).item()


def test_transfer_axes_cover_exact_locked_endpoints() -> None:
    support = TransferSupport()
    mu, sigma = transfer_axes(support, TransferResolution("test", 5.0, 2.5, 8))
    assert mu[0] == pytest.approx(-2500.0)
    assert mu[-1] == pytest.approx(120.0)
    assert sigma[0] == pytest.approx(0.5)
    assert sigma[-1] == pytest.approx(50.0)
    assert np.all(np.diff(mu) > 0.0)


def test_extra_fine_resolution_is_explicitly_supported() -> None:
    resolution = TransferResolution("extra_fine", 0.125, 0.0625, 256).validate()
    assert resolution.name == "extra_fine"
    assert resolution.mu_core_step_mv < 0.25


def test_simulation_audits_every_euler_state() -> None:
    transfer = _tiny_transfer()
    # This equilibrium state remains inside the tiny support for the short smoke run.
    state = equilibrium_state((0.001, 0.003))[None, :]
    params = [PoolParameters(0.9, 2.0)]
    simulation = simulate_extended_forks(
        state,
        params,
        transfer,
        dt_ms=0.25,
        duration_ms=20.0,
        save_stride=4,
        audit_tail_fraction=0.4,
    )
    assert int(simulation["audit_n_euler_states"]) == 81
    assert simulation["support_violation_step_count"].shape == (1,)
    assert simulation["rE_khz"].shape[0] == 21


def test_prepared_moments_match_primary_stage0c_algebra() -> None:
    params = [PoolParameters(0.8, 12.0), PoolParameters(0.84, 24.0)]
    states = np.vstack([equilibrium_state((0.02, 0.04)), equilibrium_state((0.08, 0.15))])
    states[1, 8] = 0.37
    expected = primary_moments_from_state(states, params)
    observed = moments_from_prepared(states, prepare_pool_parameters(params))
    for left, right in zip(expected, observed):
        assert np.allclose(left, right, rtol=2e-15, atol=2e-15)


def test_nonfinite_divisor_fails_closed_without_clipping_or_batch_crash() -> None:
    params = [PoolParameters(0.8, 12.0)]
    state = equilibrium_state((0.02, 0.04))[None, :]
    state[0, 8] = -1.0
    moments = moments_from_prepared(state, prepare_pool_parameters(params))
    assert np.isnan(moments[0][0])
    simulation = simulate_extended_forks(
        state,
        params,
        _tiny_transfer(),
        dt_ms=0.25,
        duration_ms=20.0,
        save_stride=4,
        audit_tail_fraction=0.4,
    )
    assert not bool(simulation["finite"][0])
    assert int(simulation["support_violation_step_count"][0]) > 0


def test_direct_exact_audit_is_fail_closed_per_fork() -> None:
    transfer = _tiny_transfer()
    state = equilibrium_state((0.001, 0.003))[None, :]
    simulation = simulate_extended_forks(
        state,
        [PoolParameters(0.9, 2.0)],
        transfer,
        dt_ms=0.25,
        duration_ms=20.0,
        save_stride=4,
        audit_tail_fraction=0.4,
    )
    audit = direct_exact_error_audit(simulation, transfer, max_points_per_fork=4)
    assert len(audit["per_fork"]) == 1
    assert audit["per_fork"][0]["fork_index"] == 0
    assert audit["per_fork"][0]["n_samples"] > 2 * 4
    assert audit["all_forks_pass"] == audit["per_fork"][0]["pass"]


def test_resolution_status_has_four_locked_outcomes() -> None:
    base = {
        "finite": True,
        "classification": "bounded_oscillatory_candidate",
        "tail_mean_hz": 9.0,
        "dominant_frequency_hz": 2.0,
        "support_violation_step_count": 0,
        "pool_bound_step_count": 0,
        "rate_bound_step_count": 0,
        "synapse_bound_step_count": 0,
        "over_100hz_tail_step_count": 0,
    }
    assert resolution_pair_status(base, base, exact_error_pass=True) == "candidate_survives"
    low = {**base, "classification": "low_fixed_point"}
    assert resolution_pair_status(low, low, exact_error_pass=True) == "collapses_low"
    high = {**base, "over_100hz_tail_step_count": 1}
    assert resolution_pair_status(high, high, exact_error_pass=True) == "becomes_over_100"
    bad = {**base, "support_violation_step_count": 1}
    assert resolution_pair_status(bad, bad, exact_error_pass=True) == "numerical_unresolved"
    negative = {**base, "negative_rate_step_count": 1}
    assert resolution_pair_status(negative, negative, exact_error_pass=True) == "numerical_unresolved"
    i_ceiling = {**base, "i_refractory_tail_occupancy_stepwise": 0.051}
    assert resolution_pair_status(i_ceiling, i_ceiling, exact_error_pass=True) == "numerical_unresolved"


def test_dt_half_must_match_confirmed_rate_and_frequency() -> None:
    base = {
        "finite": True,
        "support_violation_step_count": 0,
        "pool_bound_step_count": 0,
        "rate_bound_step_count": 0,
        "synapse_bound_step_count": 0,
        "negative_rate_step_count": 0,
        "over_100hz_tail_step_count": 0,
        "e_refractory_tail_occupancy_stepwise": 0.0,
        "i_refractory_tail_occupancy_stepwise": 0.0,
    }
    confirm = {
        **base,
        "classification": "bounded_oscillatory_candidate",
        "tail_mean_hz": 8.0,
        "dominant_frequency_hz": 2.0,
    }
    matched = {**confirm, "tail_mean_hz": 8.3, "dominant_frequency_hz": 2.1}
    assert temporal_refinement_status(confirm, matched, exact_error_pass=True) == "candidate_survives"
    wrong_rate = {**matched, "tail_mean_hz": 12.0}
    assert temporal_refinement_status(confirm, wrong_rate, exact_error_pass=True) == "numerical_unresolved"
    wrong_frequency = {**matched, "dominant_frequency_hz": 3.0}
    assert temporal_refinement_status(confirm, wrong_frequency, exact_error_pass=True) == "numerical_unresolved"
