from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.topic4_rev14_field_projection import project_fourier_to_frozen_node
from src.topic4_rev14_fourier_field import (
    evaluate,
    mode_inventory,
    uniform_quadrature,
    weighted_centered_surface_rms,
)


MODES = mode_inventory(2)


def _reference(n_e: int = 96, n_i: int = 24) -> dict:
    rng = np.random.default_rng(1701)
    return {
        "positions": rng.uniform(0.0, 20.0, size=(n_e, 2)),
        "depth": rng.normal(0.6, 0.2, size=n_e),
        "n_total": n_e + n_i,
        "target": 31.0,
    }


def _project(coefficients: np.ndarray, reference: dict | None = None, **kwargs) -> dict:
    frozen = _reference() if reference is None else reference
    contract = {
        "frozen_signed_depth": frozen["depth"],
        "n_total": frozen["n_total"],
        "target_count": frozen["target"],
        "v_base": 18.0,
        **kwargs,
    }
    return project_fourier_to_frozen_node(
        coefficients,
        frozen["positions"],
        MODES,
        **contract,
    )


def test_zero_coefficients_produce_uniform_field_not_exact_off_geometry():
    reference = _reference()
    result = _project(np.zeros((len(MODES), 2)), reference)
    expected = np.full(len(reference["positions"]), reference["target"] / len(reference["positions"]))
    np.testing.assert_allclose(result["h"], expected, rtol=0.0, atol=1e-14)
    assert result["audit"]["zero_coefficients"] is True
    assert result["audit"]["zero_uniform_max_abs_error"] < 1e-14
    assert result["audit"]["historical_field_inherited"] is False


def test_nonzero_projection_preserves_mass_and_frozen_signed_depth():
    reference = _reference()
    coefficients = np.zeros((len(MODES), 2))
    coefficients[0] = (0.55, -0.35)
    coefficients[-1] = (0.10, 0.20)
    result = _project(coefficients, reference)
    assert np.isclose(result["h"].sum(), reference["target"], atol=1e-8)
    assert np.all((result["h"] > 0.0) & (result["h"] < 1.0))
    np.testing.assert_array_equal(result["signed_depth"], reference["depth"])
    np.testing.assert_allclose(
        result["vtheta"][:len(reference["positions"])],
        18.0 - result["h"] * reference["depth"], rtol=0.0, atol=1e-14,
    )
    np.testing.assert_array_equal(
        result["vtheta"][len(reference["positions"]):],
        np.full(reference["n_total"] - len(reference["positions"]), 18.0),
    )
    assert result["audit"]["h_delta_from_uniform_rms"] > 0.0
    assert result["audit"]["threshold_modulation_rms_mV"] > 0.0


def test_centered_physical_sheet_rms_is_reported_exactly():
    coefficients = np.zeros((len(MODES), 2))
    coefficients[1] = (0.4, 0.7)
    result = _project(coefficients)
    grid, weights = uniform_quadrature(128, L=20.0)
    expected = weighted_centered_surface_rms(
        evaluate(coefficients, grid, MODES, L=20.0), weights,
    )
    assert result["audit"]["centered_latent_surface_rms"] == pytest.approx(expected, abs=1e-14)


def test_hashes_are_deterministic_and_include_physical_outputs():
    reference = _reference()
    coefficients = np.arange(len(MODES) * 2, dtype=np.float64).reshape(len(MODES), 2) / 40.0
    first = _project(coefficients, reference)
    second = _project(np.asfortranarray(coefficients), reference)
    assert first["hashes"] == second["hashes"]
    assert np.array_equal(first["h"], second["h"])
    assert len(first["hashes"]["projection_sha256"]) == 64


def test_one_shot_mode_iterable_is_materialized_once():
    reference = _reference()
    result = project_fourier_to_frozen_node(
        np.zeros((len(MODES), 2)), reference["positions"], (mode for mode in MODES),
        frozen_signed_depth=reference["depth"], n_total=reference["n_total"],
        target_count=reference["target"], v_base=18.0,
    )
    assert result["audit"]["zero_coefficients"] is True


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_closed(bad):
    coefficients = np.zeros((len(MODES), 2))
    coefficients[0, 0] = bad
    with pytest.raises(ValueError):
        _project(coefficients)


def test_bounds_mass_and_index_contract_fail_closed():
    reference = _reference()
    bad_positions = reference["positions"].copy()
    bad_positions[0, 0] = 20.1
    with pytest.raises(ValueError, match="physical sheet"):
        project_fourier_to_frozen_node(
            np.zeros((len(MODES), 2)), bad_positions, MODES,
            frozen_signed_depth=reference["depth"], n_total=reference["n_total"],
            target_count=reference["target"], v_base=18.0,
        )
    with pytest.raises(ValueError, match="target_count"):
        _project(np.zeros((len(MODES), 2)), reference, target_count=len(reference["positions"]))
    with pytest.raises(ValueError, match="n_total"):
        _project(np.zeros((len(MODES), 2)), reference, n_total=10)


def test_api_cannot_receive_contact_or_historical_field_basis():
    parameters = set(inspect.signature(project_fourier_to_frozen_node).parameters)
    forbidden = {
        "contact_names", "contact_positions", "shaft_ids", "gaussian_centers",
        "spline_coefficients", "exact_off_h", "exact_off_vtheta",
    }
    assert parameters.isdisjoint(forbidden)
    source = inspect.getsource(project_fourier_to_frozen_node).lower()
    for token in ("contact", "shaft", "gaussian"):
        assert token not in source
