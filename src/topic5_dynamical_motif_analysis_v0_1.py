"""Scoring helpers for the Topic 5.2 dynamical motif run.

Reference-event selection, summary standardisation, mode scoring and the
patient-level paired statistics.  Nothing here re-fits a model.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from scipy.special import softmax
from scipy.stats import wilcoxon

from src.lagpat_rank_audit import build_masked_kmeans_features

PREFIX_RANKS = 3
SUMMARY_KEYS = ("r_last_x", "r_last_y", "r_late_x", "r_late_y",
                "l_axis", "l_orth", "n_rank", "n_contact")


def hash_order(label: str, values: Sequence[int]) -> np.ndarray:
    """Deterministic, result-blind ordering of event indices."""
    keys = [hashlib.sha256(f"{label}|{int(v)}".encode()).hexdigest() for v in values]
    return np.asarray(values, dtype=int)[np.argsort(keys)]


def select_reference_events(
    label: str,
    indices: np.ndarray,
    lengths: np.ndarray,
    mode_label: np.ndarray,
    target: int = 24,
) -> np.ndarray:
    """Hash-stratified reference events, chosen before any result is seen.

    Strata are length tercile x train-only mode, so a patient's reference set
    keeps both short and long events and both templates.
    """
    indices = np.asarray(indices, dtype=int)
    if indices.size <= target:
        return np.sort(indices)
    event_lengths = np.asarray(lengths)[indices]
    edges = np.quantile(event_lengths, [1 / 3, 2 / 3])
    tercile = np.digitize(event_lengths, edges)
    modes = np.asarray(mode_label)[indices]
    strata = sorted({(int(a), int(b)) for a, b in zip(tercile, modes)})
    per_stratum = int(np.ceil(target / max(1, len(strata))))
    chosen: list[int] = []
    for stratum in strata:
        members = indices[(tercile == stratum[0]) & (modes == stratum[1])]
        if members.size == 0:
            continue
        chosen.extend(hash_order(label, members)[:per_stratum].tolist())
    return np.sort(np.asarray(sorted(set(chosen))[:target], dtype=int))


def sequences_to_ranks(sequence: np.ndarray, n_emitted: np.ndarray) -> np.ndarray:
    """Dense rank matrix (``-1`` = absent) from generated rank-set indicators."""
    sequence = np.asarray(sequence)
    batch, steps, n_contacts = sequence.shape
    ranks = np.full((batch, n_contacts), -1, dtype=np.int16)
    within = np.arange(steps)[None, :] <= np.asarray(n_emitted)[:, None]
    for t in range(steps - 1, -1, -1):
        active = within[:, t][:, None] & (sequence[:, t] > 0)
        ranks = np.where(active, np.int16(t), ranks)
    return ranks


def mode_posterior(ranks: np.ndarray, centers: np.ndarray, temperature: float) -> np.ndarray:
    """Train-only template posterior from the first three rank sets."""
    values = np.asarray(ranks, dtype=np.int16)
    present = (values >= 0) & (values < PREFIX_RANKS)
    features = build_masked_kmeans_features(
        values.T.astype(float), present.T, impute="event_median")
    squared = np.mean((features[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2)
    return softmax(-squared / max(float(temperature), 1e-9), axis=1)


def standardise(summary: dict[str, np.ndarray], scale: dict[str, float]) -> np.ndarray:
    """Stack the summary vector S in a fixed key order, scaled to comparable units."""
    columns = [
        summary["r_last"][:, 0], summary["r_last"][:, 1],
        summary["r_late"][:, 0], summary["r_late"][:, 1],
        summary["l_axis"], summary["l_orth"],
        summary["n_rank"].astype(float), summary["n_contact"].astype(float),
    ]
    stacked = np.stack(columns, axis=1)
    divisor = np.asarray([max(scale[key], 1e-6) for key in SUMMARY_KEYS])
    return stacked / divisor[None, :]


def observed_scale(summary: dict[str, np.ndarray]) -> dict[str, float]:
    columns = {
        "r_last_x": summary["r_last"][:, 0], "r_last_y": summary["r_last"][:, 1],
        "r_late_x": summary["r_late"][:, 0], "r_late_y": summary["r_late"][:, 1],
        "l_axis": summary["l_axis"], "l_orth": summary["l_orth"],
        "n_rank": summary["n_rank"].astype(float), "n_contact": summary["n_contact"].astype(float),
    }
    return {key: float(np.std(value)) if np.std(value) > 0 else 1.0
            for key, value in columns.items()}


def energy_score_batch(samples: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Energy score per event for ``(n_events, n_draws, dim)`` samples."""
    samples = np.asarray(samples, dtype=float)
    observations = np.asarray(observations, dtype=float)
    first = np.linalg.norm(samples - observations[:, None, :], axis=-1).mean(axis=1)
    difference = samples[:, :, None, :] - samples[:, None, :, :]
    second = np.linalg.norm(difference, axis=-1).mean(axis=(1, 2))
    return first - 0.5 * second


def covariance_alignment(generated: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Eigen-structure agreement between generated and held-out summary spreads."""
    generated = np.asarray(generated, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if generated.shape[0] < 5 or observed.shape[0] < 5:
        return {"estimable": False}
    cov_generated = np.cov(generated, rowvar=False)
    cov_observed = np.cov(observed, rowvar=False)
    values_g, vectors_g = np.linalg.eigh(cov_generated)
    values_o, vectors_o = np.linalg.eigh(cov_observed)
    order_g = np.argsort(values_g)[::-1]
    order_o = np.argsort(values_o)[::-1]
    principal = float(abs(vectors_g[:, order_g[0]] @ vectors_o[:, order_o[0]]))
    overlap = vectors_g[:, order_g[:3]].T @ vectors_o[:, order_o[:3]]
    return {
        "estimable": True,
        "leading_eigenvector_alignment": principal,
        "subspace3_alignment": float(np.linalg.svd(overlap, compute_uv=False).mean()),
        "leading_eigenvalue_ratio": float(values_g[order_g[0]] / max(values_o[order_o[0]], 1e-12)),
        "total_variance_ratio": float(np.trace(cov_generated) / max(np.trace(cov_observed), 1e-12)),
        "log_total_variance_ratio": float(
            np.log(max(np.trace(cov_generated), 1e-12) / max(np.trace(cov_observed), 1e-12))),
    }


def coverage(samples: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Per-event fraction of draws no further from the sample mean than truth."""
    samples = np.asarray(samples, dtype=float)
    observations = np.asarray(observations, dtype=float)
    centre = samples.mean(axis=1, keepdims=True)
    sample_distance = np.linalg.norm(samples - centre, axis=-1)
    truth_distance = np.linalg.norm(observations - centre[:, 0, :], axis=-1)
    return (sample_distance <= truth_distance[:, None]).mean(axis=1)


def paired_patient_effect(values: np.ndarray, alternative: str = "two-sided") -> dict:
    """Patient-level paired summary with a bootstrap interval on the median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": None, "p_value": None}
    nonzero = values[np.abs(values) > 1e-12]
    p_value = (float(wilcoxon(nonzero, alternative=alternative).pvalue)
               if nonzero.size >= 3 else None)
    rng = np.random.default_rng(20260816)
    boot = np.median(rng.choice(values, size=(4000, values.size), replace=True), axis=1)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "n_positive": int((values > 0).sum()),
        "n_negative": int((values < 0).sum()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value": p_value,
        "alternative": alternative,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni over one pre-registered family."""
    usable = {k: v for k, v in p_values.items() if v is not None and np.isfinite(v)}
    order = sorted(usable, key=lambda k: usable[k])
    n = len(order)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        value = min(1.0, (n - rank) * usable[key])
        running = max(running, value)
        adjusted[key] = running
    for key in p_values:
        adjusted.setdefault(key, None)
    return adjusted
