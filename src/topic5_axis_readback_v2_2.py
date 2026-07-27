"""Post-hoc axis read-back helpers for Topic-5 v2.2.

These helpers never fit the recurrent model and never inspect ictal targets.
They compare a frozen, seed-consensus RNN line axis with an independently
defined A/B propagation axis after all formal Claim-2 scores are frozen.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import spearmanr


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("axis must be one finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0:
        raise ValueError("axis must have non-zero norm")
    return value / norm


def line_axis_consensus(axes: np.ndarray) -> np.ndarray:
    """Return a sign-invariant consensus line for several unit axes.

    The dominant eigenvector of ``sum(u u.T)`` is invariant to independent
    sign flips and therefore does not use the external A/B axis to orient the
    RNN result.
    """
    values = np.asarray(axes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
        raise ValueError("axes must have shape [seed, 3]")
    values = np.stack([_unit(row) for row in values], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(values.T @ values)
    consensus = eigenvectors[:, int(np.argmax(eigenvalues))]
    # Canonicalize only for reproducible serialization; downstream scores are
    # sign invariant.
    anchor = int(np.argmax(np.abs(consensus)))
    if consensus[anchor] < 0:
        consensus = -consensus
    return _unit(consensus)


def sign_invariant_cosine(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    return float(abs(np.dot(_unit(axis_a), _unit(axis_b))))


def sign_invariant_projection_spearman(
    coords: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
) -> float:
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        raise ValueError("coords must have shape [contact, 3] with n >= 3")
    if not np.all(np.isfinite(values)):
        raise ValueError("coords must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    projection_a = centered @ _unit(axis_a)
    projection_b = centered @ _unit(axis_b)
    statistic = float(spearmanr(projection_a, projection_b).statistic)
    if not np.isfinite(statistic):
        raise ValueError("projection Spearman is not estimable")
    return abs(statistic)


def frozen_random_axes_by_subject(
    subjects: Iterable[str],
    *,
    seed: int = 20260726,
    n_directions: int = 256,
) -> dict[str, np.ndarray]:
    """Reproduce the Claim-3 subject-ordered random physical axes."""
    if n_directions < 1:
        raise ValueError("n_directions must be positive")
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    for subject in map(str, subjects):
        axes = rng.normal(size=(n_directions, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        anchor = np.argmax(np.abs(axes), axis=1)
        sign = np.sign(axes[np.arange(n_directions), anchor])
        axes *= np.where(sign < 0, -1.0, 1.0)[:, None]
        output[subject] = axes
    return output


def empirical_upper_percentile(observed: float, null: np.ndarray) -> float:
    """Finite-sample percentile for a larger-is-more-aligned statistic."""
    values = np.asarray(null, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        raise ValueError("observed and null values must be finite")
    return float((1 + np.sum(values <= observed)) / (values.size + 1))
