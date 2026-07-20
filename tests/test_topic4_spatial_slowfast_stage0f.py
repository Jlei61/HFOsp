"""Regression tests for the locked Stage-0F derivative certificate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.run_topic4_spatial_slowfast_stage0f as runner
import src.topic4_spatial_slowfast_stage0f as stage0f


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "config/topic4_spatial_slowfast_stage0f.yaml").read_text(encoding="utf-8")
    )


def test_locked_config_validates_and_rejects_parameter_search() -> None:
    cfg = _config()
    runner._validate_config(cfg)
    drifted = deepcopy(cfg)
    drifted["points"].append({"z": 0.85, "alpha_G": 17.0})
    with pytest.raises(ValueError, match="fixed points"):
        runner._validate_config(drifted)


@pytest.mark.parametrize(
    "mu,sigma,pop",
    [(-120.0, 4.0, "E"), (-30.0, 8.0, "I"), (10.0, 12.0, "E"), (55.0, 16.0, "I")],
)
def test_exact_siegert_moving_boundary_derivatives_match_finite_difference(
    mu: float, sigma: float, pop: str
) -> None:
    rate, d_mu, d_sigma = stage0f.exact_siegert_rate_derivatives_scalar(mu, sigma, pop)
    h = 1e-4
    plus_mu = stage0f.exact_siegert_rate_derivatives_scalar(mu + h, sigma, pop)[0]
    minus_mu = stage0f.exact_siegert_rate_derivatives_scalar(mu - h, sigma, pop)[0]
    plus_sigma = stage0f.exact_siegert_rate_derivatives_scalar(mu, sigma + h, pop)[0]
    minus_sigma = stage0f.exact_siegert_rate_derivatives_scalar(mu, sigma - h, pop)[0]
    assert np.isfinite(rate)
    assert d_mu == pytest.approx((plus_mu - minus_mu) / (2.0 * h), rel=3e-5, abs=2e-10)
    assert d_sigma == pytest.approx((plus_sigma - minus_sigma) / (2.0 * h), rel=3e-5, abs=2e-10)


def test_smooth_transfer_interpolates_table_and_refuses_extrapolation() -> None:
    mu = np.linspace(-2.0, 2.0, 9)
    sigma = np.linspace(1.0, 3.0, 9)
    table = mu[:, None] ** 2 + 0.5 * sigma[None, :] ** 2 + mu[:, None] * sigma[None, :]
    domain = stage0f.SmoothDomain(-2.0, 2.0, 1.0, 3.0)
    transfer = stage0f.SmoothSiegertTransfer(mu, sigma, table, domain=domain)
    observed, d_mu, d_sigma = transfer.log_integral_and_derivatives(
        np.asarray([-1.5, 0.25, 1.5]), np.asarray([1.25, 2.0, 2.75])
    )
    expected = np.asarray([-1.5, 0.25, 1.5]) ** 2 + 0.5 * np.asarray([1.25, 2.0, 2.75]) ** 2 + np.asarray([-1.5, 0.25, 1.5]) * np.asarray([1.25, 2.0, 2.75])
    np.testing.assert_allclose(observed, expected, atol=1e-10)
    np.testing.assert_allclose(d_mu, 2.0 * np.asarray([-1.5, 0.25, 1.5]) + np.asarray([1.25, 2.0, 2.75]), atol=1e-9)
    np.testing.assert_allclose(d_sigma, np.asarray([1.25, 2.0, 2.75]) + np.asarray([-1.5, 0.25, 1.5]), atol=1e-9)
    assert not transfer.support_mask(np.asarray([2.1]), np.asarray([2.0]))[0]
    assert np.isnan(transfer.log_integral_and_derivatives(np.asarray([2.1]), np.asarray([2.0]))[0][0])


def test_recruitment_sensor_derivative_has_correct_threshold_and_slope() -> None:
    assert stage0f.recruitment_sensor_derivative(stage0f.E0_KHZ) == 0.0
    value = 0.02
    h = 1e-7
    from src.topic4_spatial_slowfast_stage0c import recruitment_sensor

    expected = float((recruitment_sensor(value + h) - recruitment_sensor(value - h)) / (2.0 * h))
    assert stage0f.recruitment_sensor_derivative(value) == pytest.approx(expected, rel=1e-7)


def test_event_located_tangent_includes_crossing_time_sensitivity() -> None:
    section = stage0f.SectionDefinition()
    before = np.linspace(0.02, 0.10, 9)
    after = before + np.linspace(0.01, 0.09, 9)
    before[8] = 0.14
    after[8] = 0.17
    rng = np.random.default_rng(4)
    a_before = rng.normal(scale=0.02, size=(9, 8))
    a_after = rng.normal(scale=0.02, size=(9, 8))
    _, crossing, tangent = stage0f._event_located_tangent(
        before, after, a_before, a_after, section=section
    )
    assert crossing[8] == pytest.approx(0.15)
    assert np.max(np.abs(tangent[8])) < 1e-14
    epsilon = 1e-6
    for column in range(8):
        plus_before = before + epsilon * a_before[:, column]
        plus_after = after + epsilon * a_after[:, column]
        minus_before = before - epsilon * a_before[:, column]
        minus_after = after - epsilon * a_after[:, column]
        plus_fraction = (section.level - plus_before[8]) / (plus_after[8] - plus_before[8])
        minus_fraction = (section.level - minus_before[8]) / (minus_after[8] - minus_before[8])
        plus_crossing = plus_before + plus_fraction * (plus_after - plus_before)
        minus_crossing = minus_before + minus_fraction * (minus_after - minus_before)
        observed = (plus_crossing - minus_crossing) / (2.0 * epsilon)
        np.testing.assert_allclose(observed, tangent[:, column], atol=2e-9, rtol=2e-7)


def test_normalized_frobenius_difference_uses_explicit_near_zero_floor() -> None:
    zero = np.zeros((8, 8))
    small = np.eye(8) * 1e-10
    observed = stage0f.normalized_frobenius_difference(zero, small, norm_floor=1e-8)
    assert observed == pytest.approx(np.linalg.norm(small, ord="fro") / 1e-8)


def _variational_result(scale: float = 1.0) -> dict:
    matrices = {
        "chain_rule": np.diag(np.linspace(0.01, 0.08, 8)) * scale,
        "centered_1e-5": np.diag(np.linspace(0.01001, 0.08001, 8)) * scale,
        "centered_3e-6": np.diag(np.linspace(0.010005, 0.080005, 8)) * scale,
    }
    return {
        "valid": True,
        "poincare_matrices": matrices,
        "multipliers": {name: np.linalg.eigvals(matrix) for name, matrix in matrices.items()},
        "spectral_radii": {name: float(max(abs(np.linalg.eigvals(matrix)))) for name, matrix in matrices.items()},
        "section_row_max_abs": {name: 0.0 for name in matrices},
        "transversality_per_ms": 0.005,
    }


def test_variational_gate_requires_matrix_not_only_radius_agreement() -> None:
    cfg = _config()["variational"]
    good = _variational_result()
    assert stage0f.variational_consistency_summary(good, cfg)["pass"]
    broken = deepcopy(good)
    broken["poincare_matrices"]["centered_3e-6"] = good["poincare_matrices"]["centered_3e-6"].copy()
    broken["poincare_matrices"]["centered_3e-6"][0, 1] = 0.2
    broken["spectral_radii"]["centered_3e-6"] = good["spectral_radii"]["centered_3e-6"]
    assert not stage0f.variational_consistency_summary(broken, cfg)["pass"]


def test_stability_certificate_is_fail_closed_near_unit_circle() -> None:
    cfg = _config()["stability"]
    base = _variational_result()
    half = _variational_result(0.99)
    assert stage0f.stability_certificate_summary(base, half, cfg)["pass"]
    near = _variational_result(12.4)
    summary = stage0f.stability_certificate_summary(near, near, cfg)
    assert not summary["pass"]
    assert summary["all_nontrivial_multipliers_inside_unit_circle"]
    assert not summary["unit_circle_margin"] >= summary["required_margin"]


def test_overall_verdict_never_opens_stage1() -> None:
    rows = [
        {"alpha_G": 15.0, "outcome": "stable_periodic_orbit_derivative_certified"},
        {"alpha_G": 16.0, "outcome": "stable_periodic_orbit_derivative_certified"},
    ]
    assert runner._overall_verdict(rows, True) == "STAGE0F_DERIVATIVE_CERTIFIED_ALPHA15_AND_ALPHA16"
    assert runner._overall_verdict(rows, False) == "STAGE0F_ENGINEERING_OR_PROVENANCE_FAIL"
