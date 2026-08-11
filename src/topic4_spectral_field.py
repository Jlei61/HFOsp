"""Observation-invariant continuous fields for Topic 4 rev10-SA.

The field uses real Fourier features over the whole model sheet.  Every spatial
frequency has both cosine and sine phases, so the stationary prior does not
prefer a sheet location.  No electrode, shaft, source, or patient coordinate
enters this module; those quantities belong to the post-simulation observation
operator and objective.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src.topic4_core_field import project_to_budget


def fourier_wavevectors(max_harmonic, *, L=20.0):
    """One vector from every +/- pair inside an isotropic frequency disk."""
    maximum = int(max_harmonic)
    if maximum < 1 or not np.isfinite(L) or float(L) <= 0.0:
        raise ValueError("max_harmonic and sheet length must be positive")
    integers = []
    for kx in range(-maximum, maximum + 1):
        for ky in range(-maximum, maximum + 1):
            if kx * kx + ky * ky > maximum * maximum:
                continue
            if kx > 0 or (kx == 0 and ky > 0):
                integers.append((kx, ky))
    # pi/L gives a 2L fundamental period. Restricting this stationary field to
    # [0,L]^2 avoids forcing opposite boundaries to carry identical values.
    return np.asarray(integers, dtype=float) * (np.pi / float(L))


def fourier_basis_2d(positions, max_harmonic, *, L=20.0):
    """Real cos/sin basis with free phase at every spatial frequency."""
    xy = np.asarray(positions, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("positions must be finite")
    wavevectors = fourier_wavevectors(max_harmonic, L=L)
    phase = xy @ wavevectors.T
    return np.stack([np.cos(phase), np.sin(phase)], axis=2).reshape(len(xy), -1)


def spectral_surface(coefficients, positions, *, max_harmonic, L=20.0):
    """Evaluate real Fourier coefficients with no constant spatial mode."""
    expected = (len(fourier_wavevectors(max_harmonic, L=L)), 2)
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.shape != expected:
        raise ValueError(f"spectral coefficients must have shape {expected}")
    if not np.isfinite(coeff).all():
        raise ValueError("spectral coefficients must be finite")
    return fourier_basis_2d(positions, max_harmonic, L=L) @ coeff.ravel()


def spectral_field_h(coefficients, positions, *, max_harmonic, target_count,
                     L=20.0):
    """Map a free spectral surface to a bounded field with fixed total mass."""
    surface = spectral_surface(
        coefficients, positions, max_harmonic=max_harmonic, L=L,
    )
    centered = surface - float(np.mean(surface))
    q = np.exp(np.clip(centered, -30.0, 30.0))
    h, level = project_to_budget(q, float(target_count))
    return h, {"surface": surface, "q": q, "level": float(level)}


def uniform_sheet_grid(n_grid=128, *, L=20.0):
    """Uniform numerical grid used for projection and stationary priors."""
    n = int(n_grid)
    if n < 4:
        raise ValueError("n_grid must be at least 4")
    axis = (np.arange(n, dtype=float) + 0.5) * float(L) / n
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def project_surface_to_spectral(surface, positions, *, max_harmonic, L=20.0):
    """Least-squares projection sampled uniformly over the whole sheet."""
    values = np.asarray(surface, dtype=float)
    xy = np.asarray(positions, dtype=float)
    if values.shape != (len(xy),) or not np.isfinite(values).all():
        raise ValueError("surface must be one finite value per position")
    basis = fourier_basis_2d(xy, max_harmonic, L=L)
    fitted, *_ = np.linalg.lstsq(basis, values - values.mean(), rcond=None)
    return fitted.reshape(len(fourier_wavevectors(max_harmonic, L=L)), 2)


def stationary_frequency_scale(max_harmonic, *, correlation_harmonics=2.5,
                               smoothness=2.0, active_max_harmonic=None,
                               L=20.0):
    """Equal phase variance with scale determined only by frequency norm."""
    wavevectors = fourier_wavevectors(max_harmonic, L=L)
    harmonic_radius = np.linalg.norm(wavevectors, axis=1) * float(L) / np.pi
    scale = (1.0 + (harmonic_radius / float(correlation_harmonics)) ** 2) ** (
        -0.5 * float(smoothness)
    )
    if active_max_harmonic is not None:
        scale[harmonic_radius > float(active_max_harmonic) + 1e-12] = 0.0
    return np.repeat(scale[:, None], 2, axis=1)


def sample_stationary_residual_pairs(
    *, n_pairs, max_harmonic, seed, rms_amplitudes, n_grid=96, L=20.0,
    correlation_harmonics=2.5, smoothness=2.0, active_max_harmonic=None,
):
    """Draw antithetic stationary residuals normalized by whole-sheet RMS."""
    count = int(n_pairs)
    amplitudes = np.asarray(rms_amplitudes, dtype=float)
    if amplitudes.shape != (count,) or np.any(amplitudes <= 0.0):
        raise ValueError("rms_amplitudes must contain one positive value per pair")
    rng = np.random.default_rng(int(seed))
    scale = stationary_frequency_scale(
        max_harmonic, correlation_harmonics=correlation_harmonics,
        smoothness=smoothness, active_max_harmonic=active_max_harmonic, L=L,
    )
    grid = uniform_sheet_grid(int(n_grid), L=L)
    pairs = []
    for index, amplitude in enumerate(amplitudes):
        residual = rng.standard_normal(scale.shape) * scale
        sampled = spectral_surface(
            residual, grid, max_harmonic=max_harmonic, L=L,
        )
        rms = float(np.sqrt(np.mean((sampled - sampled.mean()) ** 2)))
        if rms <= 1e-12:
            raise RuntimeError("stationary residual has zero whole-sheet RMS")
        residual *= float(amplitude) / rms
        pairs.append({
            "pair_index": int(index), "rms_amplitude": float(amplitude),
            "positive": residual.copy(), "negative": -residual.copy(),
        })
    return pairs


def spectral_roughness(coefficients, *, max_harmonic, L=20.0):
    """Frequency-weighted roughness diagnostic, identical across phase."""
    coeff = np.asarray(coefficients, dtype=float)
    wavevectors = fourier_wavevectors(max_harmonic, L=L)
    if coeff.shape != (len(wavevectors), 2):
        raise ValueError("coefficients and max_harmonic disagree")
    weight = np.linalg.norm(wavevectors, axis=1) ** 4
    return float(np.sum(weight[:, None] * coeff ** 2) / coeff.size)


def array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()
