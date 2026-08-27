"""Observation-free Fourier coordinates for the rev14 static Node field.

This module defines only the latent continuous-sheet mathematics.  It does not
project the surface to neurons, impose a field budget, or run the SNN.
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence

import numpy as np


DEFAULT_SHEET_LENGTH_MM = 20.0


def _positive_float(value: float, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def canonical_mode(nx: int, ny: int) -> tuple[int, int]:
    """Return the unique half-plane representative of a nonzero +/- pair."""
    if isinstance(nx, (bool, np.bool_)) or isinstance(ny, (bool, np.bool_)):
        raise TypeError("mode coordinates must be integers")
    if not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
        raise TypeError("mode coordinates must be integers")
    original_x = int(nx)
    original_y = int(ny)
    if original_x == 0 and original_y == 0:
        raise ValueError("the zero spatial mode is excluded")
    if original_x > 0 or (original_x == 0 and original_y > 0):
        return original_x, original_y
    return -original_x, -original_y


def mode_inventory(max_order: int) -> tuple[tuple[int, int], ...]:
    """Deterministically enumerate one mode per +/- pair in a circular disk."""
    order = _nonnegative_integer(max_order, name="max_order")
    if order < 1:
        raise ValueError("max_order must be at least one")
    modes = [
        (nx, ny)
        for nx in range(0, order + 1)
        for ny in range(-order, order + 1)
        if (nx > 0 or (nx == 0 and ny > 0))
        and nx * nx + ny * ny <= order * order
    ]
    return tuple(sorted(modes, key=lambda mode: (
        mode[0] * mode[0] + mode[1] * mode[1], mode[0], mode[1],
    )))


def mode_shell(inner_order: int, outer_order: int) -> tuple[tuple[int, int], ...]:
    """Return the circular outer shell K_outer minus K_inner."""
    inner = _nonnegative_integer(inner_order, name="inner_order")
    outer = _nonnegative_integer(outer_order, name="outer_order")
    if outer <= inner:
        raise ValueError("outer_order must exceed inner_order")
    inner_squared = inner * inner
    return tuple(
        mode for mode in mode_inventory(outer)
        if mode[0] * mode[0] + mode[1] * mode[1] > inner_squared
    )


def _validated_modes(modes: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    try:
        materialized = tuple(tuple(mode) for mode in modes)
    except TypeError as exc:
        raise TypeError("modes must be an iterable of integer pairs") from exc
    if not materialized:
        raise ValueError("at least one spatial mode is required")
    validated: list[tuple[int, int]] = []
    for mode in materialized:
        if len(mode) != 2:
            raise ValueError("every mode must contain exactly two coordinates")
        nx, ny = mode
        canonical = canonical_mode(nx, ny)
        if canonical != (int(nx), int(ny)):
            raise ValueError("modes must use the frozen half-plane convention")
        validated.append(canonical)
    if len(set(validated)) != len(validated):
        raise ValueError("modes must be unique")
    return tuple(validated)


def wavevectors(
    modes: Iterable[Sequence[int]], *, L: float = DEFAULT_SHEET_LENGTH_MM,
) -> np.ndarray:
    """Convert integer modes to k = pi/L * (nx, ny)."""
    validated = _validated_modes(modes)
    length = _positive_float(L, name="L")
    return np.asarray(validated, dtype=np.float64) * (np.pi / length)


def fourier_basis(
    positions: np.ndarray,
    modes: Iterable[Sequence[int]],
    *,
    L: float = DEFAULT_SHEET_LENGTH_MM,
) -> np.ndarray:
    """Evaluate interleaved cosine/sine columns on arbitrary sheet positions."""
    xy = np.asarray(positions, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (n_positions, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("positions must be finite")
    phase = xy @ wavevectors(modes, L=L).T
    return np.stack((np.cos(phase), np.sin(phase)), axis=2).reshape(len(xy), -1)


def evaluate(
    coefficients: np.ndarray,
    positions: np.ndarray,
    modes: Iterable[Sequence[int]],
    *,
    L: float = DEFAULT_SHEET_LENGTH_MM,
) -> np.ndarray:
    """Evaluate sum_n a_n cos(k_n.x) + b_n sin(k_n.x)."""
    validated = _validated_modes(modes)
    values = np.asarray(coefficients, dtype=np.float64)
    expected = (len(validated), 2)
    if values.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    return fourier_basis(positions, validated, L=L) @ values.ravel()


def uniform_quadrature(
    n_per_axis: int = 128, *, L: float = DEFAULT_SHEET_LENGTH_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return whole-sheet midpoint nodes and equal area weights."""
    count = _nonnegative_integer(n_per_axis, name="n_per_axis")
    if count < 2:
        raise ValueError("n_per_axis must be at least two")
    length = _positive_float(L, name="L")
    axis = (np.arange(count, dtype=np.float64) + 0.5) * length / count
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    weights = np.full(count * count, (length / count) ** 2, dtype=np.float64)
    return positions, weights


def weighted_surface_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute sqrt(integral s^2 / integral 1) from positive weights."""
    surface = np.asarray(values, dtype=np.float64)
    area = np.asarray(weights, dtype=np.float64)
    if surface.ndim != 1 or area.shape != surface.shape or surface.size == 0:
        raise ValueError("values and weights must be nonempty aligned vectors")
    if not np.isfinite(surface).all() or not np.isfinite(area).all():
        raise ValueError("values and weights must be finite")
    if np.any(area <= 0.0):
        raise ValueError("quadrature weights must be positive")
    return float(np.sqrt(np.dot(area, surface * surface) / np.sum(area)))


def weighted_centered_surface_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute RMS after removing the physical-sheet weighted mean.

    The downstream mass projection solves an additive level ``lambda``. Spatial
    constants are therefore a gauge direction and must not consume the frozen
    Fourier dose.
    """
    surface = np.asarray(values, dtype=np.float64)
    area = np.asarray(weights, dtype=np.float64)
    if surface.ndim != 1 or area.shape != surface.shape or surface.size == 0:
        raise ValueError("values and weights must be nonempty aligned vectors")
    if not np.isfinite(surface).all() or not np.isfinite(area).all():
        raise ValueError("values and weights must be finite")
    if np.any(area <= 0.0):
        raise ValueError("quadrature weights must be positive")
    centered = surface - float(np.dot(area, surface) / np.sum(area))
    return weighted_surface_rms(centered, area)


def normalize_shell_rms(
    coefficients: np.ndarray,
    modes: Iterable[Sequence[int]],
    *,
    target_rms: float = 1.0,
    n_per_axis: int = 128,
    L: float = DEFAULT_SHEET_LENGTH_MM,
) -> np.ndarray:
    """Scale one shell to a target centered RMS on the physical sheet."""
    validated = _validated_modes(modes)
    values = np.asarray(coefficients, dtype=np.float64)
    expected = (len(validated), 2)
    if values.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    target = _positive_float(target_rms, name="target_rms")
    positions, weights = uniform_quadrature(n_per_axis, L=L)
    current = weighted_centered_surface_rms(
        evaluate(values, positions, validated, L=L), weights,
    )
    if current <= np.finfo(np.float64).eps:
        raise ValueError("a zero shell cannot be RMS-normalized")
    return values * (target / current)


def spectral_roughness_surrogate(
    coefficients: np.ndarray,
    modes: Iterable[Sequence[int]],
    *,
    L: float = DEFAULT_SHEET_LENGTH_MM,
) -> float:
    """Return the frozen coefficient-space roughness tie-break.

    The 2L-periodic basis is observed only on ``[0, L]^2``. This value is not
    asserted to equal the physical-sheet integral of the squared Laplacian.
    """
    validated = _validated_modes(modes)
    values = np.asarray(coefficients, dtype=np.float64)
    expected = (len(validated), 2)
    if values.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    k_squared = np.sum(wavevectors(validated, L=L) ** 2, axis=1)
    return float(np.sum((k_squared * k_squared) * np.sum(values * values, axis=1)))


def analytic_roughness(
    coefficients: np.ndarray,
    modes: Iterable[Sequence[int]],
    *,
    L: float = DEFAULT_SHEET_LENGTH_MM,
) -> float:
    """Backward-compatible name for :func:`spectral_roughness_surrogate`."""
    return spectral_roughness_surrogate(coefficients, modes, L=L)


def array_sha256(values: np.ndarray) -> str:
    """Hash an array using a canonical little-endian numeric representation."""
    array = np.asarray(values)
    if array.dtype.kind in "iu":
        canonical = np.ascontiguousarray(array, dtype="<i8")
    elif array.dtype.kind == "f":
        canonical = np.ascontiguousarray(array, dtype="<f8")
    else:
        raise TypeError("only numeric arrays can be hashed")
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", canonical.ndim))
    digest.update(struct.pack(f"<{canonical.ndim}Q", *canonical.shape))
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()
