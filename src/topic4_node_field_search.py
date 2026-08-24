"""Observation-invariant whole-sheet residuals for rev12-ND Node refitting."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from src.topic4_continuous_field import continuous_surface, tensor_basis
from src.topic4_observation_invariant_spline import array_sha256, spline_roughness


def uniform_sheet_grid(points_per_axis: int, *, sheet_mm: float = 20.0) -> np.ndarray:
    axis = np.linspace(0.0, float(sheet_mm), int(points_per_axis))
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def coarse_residual_to_coefficients(control: np.ndarray, *,
                                    target_n_basis: int = 18,
                                    degree: int = 3,
                                    sheet_mm: float = 20.0,
                                    projection_grid_per_axis: int = 31) -> dict:
    """Project a sheet-wide coarse spline onto the stored 18x18 basis."""
    control = np.asarray(control, float)
    if control.ndim != 2 or control.shape[0] != control.shape[1]:
        raise ValueError("coarse control surface must be square")
    if control.shape[0] < int(degree) + 1:
        raise ValueError("coarse control surface has too few basis functions")
    grid = uniform_sheet_grid(
        int(projection_grid_per_axis), sheet_mm=sheet_mm,
    )
    target = continuous_surface(
        control, grid, n_basis=control.shape[0], degree=degree, L=sheet_mm,
    )
    basis = tensor_basis(
        grid, int(target_n_basis), degree=degree, L=sheet_mm,
    )
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    reconstructed = basis @ coefficients
    rms = float(np.sqrt(np.mean(reconstructed ** 2)))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise ValueError("coarse residual has zero sheet-wide RMS")
    coefficients = coefficients / rms
    reconstructed = reconstructed / rms
    target = target / rms
    return {
        "coefficients": coefficients.reshape(int(target_n_basis), int(target_n_basis)),
        "projection_rmse": float(np.sqrt(np.mean((reconstructed - target) ** 2))),
        "surface_mean": float(np.mean(reconstructed)),
        "surface_rms": float(np.sqrt(np.mean(reconstructed ** 2))),
        "coarse_n_basis": int(control.shape[0]),
        "target_n_basis": int(target_n_basis),
        "degree": int(degree),
    }


def sobol_coarse_residuals(*, n_residuals: int, n_basis: int = 4,
                           seed: int = 20260821) -> list[np.ndarray]:
    """Generate deterministic sheet-wide directions without contact coordinates."""
    n_residuals = int(n_residuals)
    if n_residuals <= 0:
        raise ValueError("n_residuals must be positive")
    engine = qmc.Sobol(d=int(n_basis) ** 2, scramble=True, seed=int(seed))
    sample_count = 1 << int(np.ceil(np.log2(max(2, n_residuals))))
    samples = 2.0 * engine.random_base2(int(np.log2(sample_count))) - 1.0
    output = []
    for sample in samples:
        values = sample.reshape(int(n_basis), int(n_basis))
        values = values - np.mean(values)
        norm = float(np.linalg.norm(values))
        if norm > 1e-12:
            output.append(values / norm)
        if len(output) == n_residuals:
            break
    if len(output) != n_residuals:
        raise RuntimeError("Sobol generator did not produce enough residuals")
    return output


def residual_candidate(anchor: dict, residual: np.ndarray, *, amplitude: float,
                       candidate_id: str, residual_index: int,
                       coarse_n_basis: int = 4) -> dict:
    """Add one normalized whole-sheet residual to a frozen spline field."""
    coefficients = np.asarray(anchor["coefficients"], float)
    residual = np.asarray(residual, float)
    if residual.shape != coefficients.shape:
        raise ValueError("anchor and residual coefficient tensors do not align")
    values = coefficients + float(amplitude) * residual
    return {
        "candidate_id": str(candidate_id),
        "field_type": "spline_continuous",
        "n_basis": int(anchor["n_basis"]),
        "degree": int(anchor["degree"]),
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        "role": "rev12_whole_sheet_coarse_residual",
        "source_field_sha256": anchor["field_sha256"],
        "residual_coordinates": {
            "coarse_n_basis": int(coarse_n_basis),
            "residual_index": int(residual_index),
            "signed_log_surface_rms": float(amplitude),
            "observation_coordinates_used": False,
        },
    }


def interpolate_spline_candidates(left: dict, right: dict, *, weight: float,
                                  candidate_id: str) -> dict:
    """Interpolate two whole-sheet spline fields without observation coordinates."""
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("interpolation weight must lie in [0, 1]")
    for key in ("field_type", "n_basis", "degree"):
        if left[key] != right[key]:
            raise ValueError(f"spline candidates differ in {key}")
    if left["field_type"] != "spline_continuous":
        raise ValueError("only spline_continuous fields can be interpolated")
    left_values = np.asarray(left["coefficients"], float)
    right_values = np.asarray(right["coefficients"], float)
    if left_values.shape != right_values.shape:
        raise ValueError("spline coefficient tensors do not align")
    values = (1.0 - weight) * left_values + weight * right_values
    return {
        "candidate_id": str(candidate_id),
        "field_type": "spline_continuous",
        "n_basis": int(left["n_basis"]),
        "degree": int(left["degree"]),
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        "role": "rev12_whole_sheet_candidate_interpolation",
        "source_field_sha256": [left["field_sha256"], right["field_sha256"]],
        "residual_coordinates": {
            "interpolation_weight_toward_right": weight,
            "observation_coordinates_used": False,
        },
    }


def normalize_surface_residual(residuals: list[np.ndarray], weights: np.ndarray,
                               *, n_basis: int = 18, degree: int = 3,
                               sheet_mm: float = 20.0,
                               grid_per_axis: int = 31) -> np.ndarray:
    """Combine coefficient directions and normalize their sheet-surface RMS."""
    values = [np.asarray(residual, float) for residual in residuals]
    weights = np.asarray(weights, float)
    expected = (int(n_basis), int(n_basis))
    if not values or weights.shape != (len(values),):
        raise ValueError("residual directions and weights do not align")
    if any(value.shape != expected for value in values) \
            or not np.all(np.isfinite(weights)):
        raise ValueError("combined residual has invalid shape or weights")
    combined = np.sum([
        weight * value for weight, value in zip(weights, values)
    ], axis=0)
    grid = uniform_sheet_grid(int(grid_per_axis), sheet_mm=sheet_mm)
    surface = continuous_surface(
        combined, grid, n_basis=int(n_basis), degree=int(degree), L=sheet_mm,
    )
    rms = float(np.sqrt(np.mean(surface ** 2)))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise ValueError("combined residual has zero sheet-surface RMS")
    return combined / rms
