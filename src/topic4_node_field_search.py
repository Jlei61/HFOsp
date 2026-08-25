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


def cosine_sheet_residuals(*, maximum_frequency: int = 3,
                           target_n_basis: int = 18, degree: int = 3,
                           sheet_mm: float = 20.0,
                           projection_grid_per_axis: int = 61) -> dict:
    """Build an ordered orthonormal low-frequency basis over the whole sheet.

    The construction uses only the uniform sheet domain.  It does not receive
    contact, shaft, patient-source or manual-core coordinates.  Constant mode
    (0, 0) is excluded because the downstream field budget removes a global
    offset.
    """
    maximum_frequency = int(maximum_frequency)
    target_n_basis = int(target_n_basis)
    degree = int(degree)
    if maximum_frequency <= 0 or target_n_basis < degree + 1:
        raise ValueError("cosine basis dimensions are invalid")
    grid = uniform_sheet_grid(
        int(projection_grid_per_axis), sheet_mm=float(sheet_mm),
    )
    x = grid[:, 0]
    y = grid[:, 1]
    spline = tensor_basis(
        grid, target_n_basis, degree=degree, L=float(sheet_mm),
    )
    coefficient_count = target_n_basis ** 2
    centering = np.eye(coefficient_count) - np.full(
        (coefficient_count, coefficient_count), 1.0 / coefficient_count,
    )
    centered_spline = spline @ centering
    effective_spline = centered_spline - np.mean(centered_spline, axis=0, keepdims=True)
    mode_indices = sorted(
        [
            (kx, ky)
            for kx in range(maximum_frequency + 1)
            for ky in range(maximum_frequency + 1)
            if (kx, ky) != (0, 0)
        ],
        key=lambda pair: (
            pair[0] ** 2 + pair[1] ** 2,
            max(pair), pair[0], pair[1],
        ),
    )
    target_vectors: list[np.ndarray] = []
    coefficient_vectors: list[np.ndarray] = []
    fitted_vectors: list[np.ndarray] = []
    rows = []
    for mode_index, (kx, ky) in enumerate(mode_indices):
        target = (
            np.cos(np.pi * float(kx) * x / float(sheet_mm))
            * np.cos(np.pi * float(ky) * y / float(sheet_mm))
        )
        target = target - float(np.mean(target))
        for previous in target_vectors:
            target = target - float(np.mean(target * previous)) * previous
        target_rms = float(np.sqrt(np.mean(target ** 2)))
        if target_rms <= 1e-12:
            raise RuntimeError("cosine target basis became degenerate")
        target = target / target_rms
        unconstrained, *_ = np.linalg.lstsq(effective_spline, target, rcond=None)
        # Solve directly in the coefficient-mean-zero subspace used by
        # continuous_surface rather than fitting and centering afterwards.
        coefficients = centering @ unconstrained
        fitted = spline @ coefficients
        for previous_coefficients, previous_surface in zip(
                coefficient_vectors, fitted_vectors):
            effective = fitted - float(np.mean(fitted))
            projection = float(np.mean(effective * previous_surface))
            coefficients = coefficients - projection * previous_coefficients
            fitted = fitted - projection * previous_surface
        effective = fitted - float(np.mean(fitted))
        fitted_rms = float(np.sqrt(np.mean(effective ** 2)))
        if fitted_rms <= 1e-12:
            raise RuntimeError("projected cosine basis became degenerate")
        coefficients = coefficients / fitted_rms
        fitted = fitted / fitted_rms
        effective = fitted - float(np.mean(fitted))
        target_vectors.append(target)
        coefficient_vectors.append(coefficients)
        fitted_vectors.append(effective)
        rows.append({
            "mode_index": int(mode_index),
            "kx": int(kx), "ky": int(ky),
            "spatial_frequency_norm": float(np.hypot(kx, ky)),
            "coefficients": coefficients.reshape(
                target_n_basis, target_n_basis,
            ),
            "projection_rmse": float(np.sqrt(np.mean((effective - target) ** 2))),
            "surface_mean": float(np.mean(fitted)),
            "effective_surface_rms": float(np.sqrt(np.mean(effective ** 2))),
        })
    surfaces = np.column_stack(fitted_vectors)
    gram = surfaces.T @ surfaces / len(surfaces)
    off_diagonal = gram - np.eye(len(rows))
    return {
        "rows": rows,
        "maximum_frequency": maximum_frequency,
        "n_modes": int(len(rows)),
        "target_n_basis": target_n_basis,
        "degree": degree,
        "sheet_mm": float(sheet_mm),
        "projection_grid_per_axis": int(projection_grid_per_axis),
        "maximum_absolute_gram_error": float(np.max(np.abs(off_diagonal))),
        "observation_coordinates_used": False,
    }


def sobol_cosine_combinations(*, n_pairs: int, n_modes: int,
                              radii: tuple[float, ...],
                              seed: int = 20260825) -> list[dict]:
    """Generate deterministic antithetic directions in a free cosine span."""
    n_pairs, n_modes = int(n_pairs), int(n_modes)
    radius_values = tuple(float(value) for value in radii)
    if n_pairs <= 0 or n_modes <= 0 or not radius_values:
        raise ValueError("global combination dimensions must be positive")
    if any(not np.isfinite(value) or value <= 0.0 for value in radius_values):
        raise ValueError("global combination radii must be finite and positive")
    engine = qmc.Sobol(d=n_modes, scramble=True, seed=int(seed))
    sample_count = 1 << int(np.ceil(np.log2(max(2, n_pairs))))
    samples = 2.0 * engine.random_base2(int(np.log2(sample_count))) - 1.0
    rows = []
    for pair_index, sample in enumerate(samples[:n_pairs]):
        norm = float(np.linalg.norm(sample))
        if norm <= 1e-12:
            raise RuntimeError("Sobol global direction became degenerate")
        unit = sample / norm
        radius = radius_values[pair_index % len(radius_values)]
        for sign in (-1.0, 1.0):
            coefficients = sign * radius * unit
            rows.append({
                "pair_index": int(pair_index),
                "sign": int(sign),
                "radius": radius,
                "coefficients": coefficients,
                "coefficient_l2": float(np.linalg.norm(coefficients)),
            })
    return rows


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
