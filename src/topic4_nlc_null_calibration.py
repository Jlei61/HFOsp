"""Null calibration for the rev11-NLC frozen-substrate acceptance statements.

The rev11-NLC confirmation adjudicates ``DIRECTIONAL_REPERTOIRE`` against a
fixed 0.5 threshold and ``PATIENT_GEOMETRY`` against a fixed 0.0 threshold.
Neither threshold is the null expectation of its statistic:

* ``direction_balanced_alignment`` takes the better of the two cluster-to-mode
  matchings, so its null expectation is strictly above 0.5 for finite samples;
* ``signed_margin`` is produced by assigning events to the patient prototypes
  and then evaluating those same prototypes on the held-out contact fold, and
  the two patient prototypes are strongly anti-correlated, so a model that only
  propagates back and forth along the implanted axis already earns a positive
  margin.

This module builds the two matched nulls that the earlier D5.2 audit of the
same line used (seed-stratified label permutation, patient-matched benchmark)
and adds a contact-correspondence null for the cross-fit margin. The fast
cross-fit path is checked against ``crossfit_patient_readout`` on the observed
data of every arm-network pair before any null draw is trusted.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from src.topic4_d6_natural_kmeans import (
    best_binary_alignment,
    normalize_event_ranks,
    patient_profiles,
    signed_matrix_margin,
)

MINIMUM_SHARED_CONTACTS = 3


def _spearman(left, right):
    """Spearman rho with the tie handling ``scipy.stats.spearmanr`` uses."""
    if len(left) < MINIMUM_SHARED_CONTACTS:
        return np.nan
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    denominator = np.sqrt(np.sum(left_rank ** 2) * np.sum(right_rank ** 2))
    if denominator <= 0.0:
        return np.nan
    return float(np.dot(left_rank, right_rank) / denominator)


def _assign_to_profiles(ranks, profiles, contacts):
    """Vectorised equivalent of ``topic4_d6_natural_kmeans.assign_to_profiles``."""
    selected = ranks[:, contacts]
    distances = np.full((len(ranks), 2), np.nan)
    for mode in (0, 1):
        profile = profiles[mode, contacts]
        finite = np.isfinite(selected) & np.isfinite(profile)[None, :]
        count = finite.sum(axis=1)
        difference = np.where(finite, selected - profile[None, :], 0.0)
        usable = count >= MINIMUM_SHARED_CONTACTS
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.sqrt(np.sum(difference ** 2, axis=1) / np.maximum(count, 1))
        distances[usable, mode] = value[usable]
    labels = np.full(len(ranks), -1, int)
    both = np.all(np.isfinite(distances), axis=1)
    labels[both] = np.argmin(distances[both], axis=1)
    return labels


def _mode_profiles(ranks, labels):
    profiles = np.full((2, ranks.shape[1]), np.nan)
    for mode in (0, 1):
        selected = ranks[labels == mode]
        if not len(selected):
            continue
        finite = np.isfinite(selected)
        count = finite.sum(axis=0)
        total = np.where(finite, selected, 0.0).sum(axis=0)
        profiles[mode] = np.divide(
            total, count, out=np.full(ranks.shape[1], np.nan), where=count > 0,
        )
    return profiles


def _matrix(model, patient, contacts):
    matrix = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            finite = (
                np.isfinite(model[row, contacts])
                & np.isfinite(patient[column, contacts])
            )
            if np.sum(finite) >= MINIMUM_SHARED_CONTACTS:
                matrix[row, column] = _spearman(
                    model[row, contacts][finite], patient[column, contacts][finite],
                )
    return matrix


def crossfit_margin(normalized_ranks, profiles, folds):
    """Cross-fit signed margin for already normalized model event ranks.

    Mirrors ``crossfit_patient_readout``: assign on one within-shaft alternating
    contact fold, evaluate on the disjoint fold, swap, then take the signed
    margin of the fold-averaged matrix.
    """
    stack = []
    for assignment_contacts, evaluation_contacts in (
            (folds[0], folds[1]), (folds[1], folds[0])):
        labels = _assign_to_profiles(normalized_ranks, profiles, assignment_contacts)
        model = _mode_profiles(normalized_ranks, labels)
        stack.append(_matrix(model, profiles, evaluation_contacts))
    stack = np.asarray(stack, float)
    count = np.sum(np.isfinite(stack), axis=0)
    matrix = np.divide(
        np.nansum(stack, axis=0), count,
        out=np.full(stack.shape[1:], np.nan), where=count > 0,
    )
    return signed_matrix_margin(matrix)


def contact_permutation_draws(
        ranks, patient_ranks, patient_labels, folds, *, draws, seed,
        shaft_ids=None):
    """Null draws of the cross-fit margin under permuted contact identity.

    One permutation is drawn per network and applied to every event of that
    network, so each event keeps its own rank profile and the events keep their
    joint structure; only the correspondence between model contact identity and
    patient contact identity is destroyed. ``shaft_ids`` restricts the
    permutation to within-shaft exchanges, which additionally preserves the
    shaft-level ordering that the implantation geometry already fixes.
    """
    normalized = normalize_event_ranks(ranks)
    profiles = patient_profiles(patient_ranks, patient_labels)
    rng = np.random.default_rng(int(seed))
    n_contacts = normalized.shape[1]
    if shaft_ids is None:
        blocks = [np.arange(n_contacts)]
    else:
        shaft_ids = np.asarray(shaft_ids)
        blocks = [
            np.flatnonzero(shaft_ids == shaft) for shaft in np.unique(shaft_ids)
        ]
    values = []
    for _ in range(int(draws)):
        order = np.arange(n_contacts)
        for block in blocks:
            order[block] = rng.permutation(block)
        margin = crossfit_margin(normalized[:, order], profiles, folds)
        if margin is not None and np.isfinite(margin):
            values.append(float(margin))
    return np.asarray(values, float)


def direction_label_permutation_draws(
        cluster_labels, direction_labels, *, draws, seed):
    """Null draws of balanced alignment under permuted direction labels.

    Permuting only the labelled events preserves which events carry a direction
    label and the two mode counts, so the draw distribution contains exactly the
    best-of-two-matchings bias that the 0.5 threshold ignores.
    """
    cluster_labels = np.asarray(cluster_labels, int)
    direction_labels = np.asarray(direction_labels, int)
    labelled = np.flatnonzero(direction_labels >= 0)
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(draws)):
        shuffled = direction_labels.copy()
        shuffled[labelled] = rng.permutation(direction_labels[labelled])
        alignment = best_binary_alignment(cluster_labels, shuffled)
        if alignment["balanced_alignment"] is not None:
            values.append(float(alignment["balanced_alignment"]))
    return np.asarray(values, float)


def equal_network_null(observed_by_network, draws_by_network):
    """Aggregate per-network null draws into an equal-network null distribution.

    Every network contributes one draw per index, so the aggregate keeps the
    equal-network weighting the acceptance contract uses.
    """
    seeds = sorted(draws_by_network, key=str)
    if not seeds:
        return None
    width = min(len(draws_by_network[seed]) for seed in seeds)
    if width <= 0:
        return None
    stack = np.asarray([draws_by_network[seed][:width] for seed in seeds], float)
    means = stack.mean(axis=0)
    observed = float(np.mean([observed_by_network[seed] for seed in seeds]))
    exceed = int(np.sum(means >= observed - 1e-12))
    return {
        "n_networks": len(seeds),
        "n_draws": int(width),
        "observed_equal_network_mean": observed,
        "null_q05": float(np.quantile(means, 0.05)),
        "null_median": float(np.quantile(means, 0.50)),
        "null_q95": float(np.quantile(means, 0.95)),
        "observed_minus_null_median": observed - float(np.quantile(means, 0.50)),
        "one_sided_p": float((exceed + 1) / (width + 1)),
        "observed_above_null_q95": bool(observed > float(np.quantile(means, 0.95))),
    }
