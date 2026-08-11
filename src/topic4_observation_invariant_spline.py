"""Stable whole-sheet random fields on a uniform continuous spline basis."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter

from src.topic4_continuous_field import (
    continuous_surface,
    curvature_energy,
    tensor_basis,
)


def array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def uniform_allocation_centers(n_per_axis, *, margin_mm, L=20.0):
    """Uniform probe locations independent of electrode geometry."""
    axis = np.linspace(float(margin_mm), float(L) - float(margin_mm), int(n_per_axis))
    return np.asarray([(x, y) for y in axis for x in axis], dtype=float)


def fit_uniform_surface(surface, positions, *, n_basis, degree=3, L=20.0):
    """Project sheet samples onto the stable uniform spline coordinates."""
    values = np.asarray(surface, dtype=float)
    xy = np.asarray(positions, dtype=float)
    if values.shape != (len(xy),) or not np.isfinite(values).all():
        raise ValueError("surface must provide one finite sample per position")
    basis = tensor_basis(xy, n_basis, degree=degree, L=L)
    fitted, *_ = np.linalg.lstsq(basis, values - values.mean(), rcond=None)
    fitted -= fitted.mean()
    return fitted.reshape(int(n_basis), int(n_basis))


def allocation_direction(center, positions, *, n_basis, width_mm,
                         log_amplitude, degree=3, L=20.0):
    """Project one of a uniformly frozen set of identical smooth probes."""
    center = np.asarray(center, dtype=float)
    xy = np.asarray(positions, dtype=float)
    surface = float(log_amplitude) * np.exp(
        -0.5 * np.sum((xy - center[None, :]) ** 2, axis=1)
        / float(width_mm) ** 2
    )
    return fit_uniform_surface(
        surface, xy, n_basis=n_basis, degree=degree, L=L,
    )


def sample_smooth_residual_pairs(
    *, n_pairs, n_basis, seed, rms_amplitudes, positions,
    smoothing_controls=1.25, degree=3, L=20.0,
):
    """Sample antithetic residuals from an observation-free smooth prior."""
    amplitudes = np.asarray(rms_amplitudes, dtype=float)
    if amplitudes.shape != (int(n_pairs),) or np.any(amplitudes <= 0.0):
        raise ValueError("one positive RMS amplitude is required per pair")
    rng = np.random.default_rng(int(seed))
    output = []
    for index, amplitude in enumerate(amplitudes):
        coefficients = gaussian_filter(
            rng.standard_normal((int(n_basis), int(n_basis))),
            sigma=float(smoothing_controls), mode="reflect",
        )
        coefficients -= coefficients.mean()
        sampled = continuous_surface(
            coefficients, positions, n_basis=n_basis, degree=degree, L=L,
        )
        rms = float(np.sqrt(np.mean((sampled - sampled.mean()) ** 2)))
        if rms <= 1e-12:
            raise RuntimeError("smooth random residual has zero RMS")
        coefficients *= float(amplitude) / rms
        output.append({
            "pair_index": int(index),
            "rms_amplitude": float(amplitude),
            "positive": coefficients.copy(),
            "negative": -coefficients.copy(),
        })
    return output


def spline_roughness(coefficients):
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("spline coefficients must form a square lattice")
    return curvature_energy(values, values.shape[0])
