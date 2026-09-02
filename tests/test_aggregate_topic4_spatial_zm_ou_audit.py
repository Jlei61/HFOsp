import pytest

from scripts.aggregate_topic4_spatial_zm_ou_audit import (
    runtime_certified,
    select_working_point,
)


def _record(sigma=0.10, ell=0.38, tau=20.0, seed=1801, eligible=True, **updates):
    row = {
        "seed": seed,
        "sigma_rate_per_ms": sigma,
        "tau_ms": tau,
        "ell_mm": ell,
        "called_every_membrane_step": True,
        "measured_sd_rate_per_ms": 0.92 * sigma,
        "measured_mean_rate_per_ms": 0.0,
        "measured_tau_ms": tau,
        "measured_correlation_length_mm": 2.0 * ell,
        "sd_ratio_after_over_before": 1.0,
        "all_clauses_pass": eligible,
    }
    row.update(updates)
    return row


def test_a_run_whose_drive_was_never_stepped_is_not_certified():
    got = runtime_certified(_record(called_every_membrane_step=False))
    assert got["all_pass"] is False
    assert got["checks"]["stepped_every_membrane_step"] is False


def test_measured_length_is_checked_against_twice_the_declared_kernel_width():
    """A Gaussian kernel of width ell gives a 1/e crossing near 2*ell."""
    assert runtime_certified(_record())["all_pass"] is True
    wrong = _record(measured_correlation_length_mm=0.38)
    assert runtime_certified(wrong)["checks"][
        "correlation_length_matches_declared"] is False


def test_amplitude_change_across_the_split_is_not_certified():
    got = runtime_certified(_record(sd_ratio_after_over_before=1.4))
    assert got["checks"]["stationary_across_split"] is False
    assert got["all_pass"] is False


def test_declared_baseline_is_taken_whenever_it_qualifies():
    rows = []
    for seed in (1801, 1802, 1803):
        rows.append(_record(sigma=0.10, ell=0.38, seed=seed))
        rows.append(_record(sigma=0.05, ell=0.38, seed=seed))
    decision = select_working_point(rows)
    assert decision["selected"]["is_declared_baseline"] is True
    assert decision["selection_reason"] == "declared baseline qualifies"


def test_lowest_qualifying_amplitude_is_taken_when_the_baseline_fails():
    rows = []
    for seed in (1801, 1802, 1803):
        rows.append(_record(sigma=0.10, ell=0.38, seed=seed, eligible=False))
        rows.append(_record(sigma=0.20, ell=0.38, seed=seed))
        rows.append(_record(sigma=0.35, ell=0.38, seed=seed))
    decision = select_working_point(rows)
    assert decision["selected"]["sigma_rate_per_ms"] == pytest.approx(0.20)
    assert "baseline failed" in decision["selection_reason"]


def test_a_rung_with_an_uncertified_seed_cannot_be_selected():
    """Low-state eligibility is not enough; the drive must be proven to run."""
    rows = []
    for seed in (1801, 1802, 1803):
        rows.append(_record(sigma=0.10, ell=0.38, seed=seed,
                            called_every_membrane_step=(seed != 1803)))
        rows.append(_record(sigma=0.20, ell=0.38, seed=seed))
    decision = select_working_point(rows)
    baseline = next(rung for rung in decision["rungs"]
                    if rung["is_declared_baseline"])
    assert baseline["n_seeds_low_state_eligible"] == 3
    assert baseline["qualifies"] is False
    assert decision["selected"]["sigma_rate_per_ms"] == pytest.approx(0.20)


def test_no_qualifying_rung_reports_no_selection_rather_than_a_default():
    rows = [_record(seed=seed, eligible=False) for seed in (1801, 1802, 1803)]
    decision = select_working_point(rows)
    assert decision["selected"] is None
    assert decision["selection_reason"] == "no rung qualified"
