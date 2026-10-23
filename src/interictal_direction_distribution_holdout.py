"""Training-only template densities evaluated on unseen recording blocks.

The primary loss is the negative log density of the *marginal* 3-D direction
distribution. Test directions never select a component. The component axes
remain the existing rank-template gradients; only their spread and prevalence
are estimated from training events. This is a representation test, not a
forecast of independently measured directions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from scripts.run_timing_plus_space_rank_readback import _fit_missing_view_labels
from src.interictal_propagation import assign_events_to_templates, build_cluster_templates
from src.lagpat_rank_audit import build_masked_kmeans_features
from src.topic5_interictal_direction_rose import fit_event_directions_3d

METHODS = ("rank_sequence", "rank_trajectory")
LOG_SPHERE_AREA = float(np.log(4 * np.pi))
KAPPA_CAP = 1000.0


def fit_template_model(ranks, bools, directions, coords, *, method, seed=0):
    """Receive training arrays only; reproduce the current all-event fit."""
    features = build_masked_kmeans_features(ranks, bools, impute="event_median")
    if method == METHODS[0]:
        labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(features)
        scale = None
    elif method == METHODS[1]:
        fit = _fit_missing_view_labels(features, directions, random_state=seed)
        labels, scale = fit["labels"], fit["spatial_scale"]
    else:
        raise ValueError(method)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 20:
        raise ValueError(f"insufficient training cluster support: {counts}")
    templates = build_cluster_templates(ranks, bools, labels, n_clusters=2)
    axes = fit_event_directions_3d(templates.T, coords, min_contacts=3)["directions"]
    if not np.isfinite(axes).all():
        raise ValueError("undefined training template axis")
    return dict(labels=labels, templates=templates, axes=axes,
                train_counts=counts, spatial_scale=scale)


def fit_concentration(mean_cosine):
    """Constrained vMF MLE with fixed axis: coth(kappa)-1/kappa = mean cosine."""
    value = float(mean_cosine)
    if not np.isfinite(value) or value > 1 + 1e-10 or value < -1 - 1e-10:
        raise ValueError("invalid mean cosine")
    if value <= 0:
        return 0.0

    def resultant(k):
        if k < 1e-4:
            return k / 3 - k**3 / 45
        return 1 / np.tanh(k) - 1 / k

    if value >= resultant(KAPPA_CAP):
        return KAPPA_CAP
    return float(brentq(lambda k: resultant(k) - value, 0, KAPPA_CAP))


def component_logpdf(directions, axes, kappas):
    """vMF log densities per steradian on S^2, including the uniform limit."""
    directions, axes, kappas = map(np.asarray, (directions, axes, kappas))
    if directions.ndim != 2 or directions.shape[1] != 3 or axes.shape != (len(kappas), 3):
        raise ValueError("direction/axis dimensions do not align")
    if not np.isfinite(directions).all() or not np.isfinite(axes).all():
        raise ValueError("undefined direction passed to density")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-7):
        raise ValueError("directions must be unit vectors")
    if not np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-7):
        raise ValueError("axes must be unit vectors")
    if np.any(kappas < 0) or not np.isfinite(kappas).all():
        raise ValueError("invalid concentration")
    cosines = np.clip(directions @ axes.T, -1, 1)
    result = np.empty_like(cosines)
    for k, concentration in enumerate(kappas):
        if concentration < 1e-5:
            result[:, k] = (-LOG_SPHERE_AREA - concentration**2 / 6
                            + concentration * cosines[:, k])
        else:
            # Combining k*cos(theta)-k avoids cancellation and overflow.
            result[:, k] = (np.log(concentration) - np.log(2 * np.pi)
                            - np.log(-np.expm1(-2 * concentration))
                            + concentration * (cosines[:, k] - 1))
    return result


def fit_template_density(model, training_directions):
    """No free rotation: density centers are the frozen rank-template axes."""
    directions = np.asarray(training_directions)
    valid = np.isfinite(directions).all(axis=1)
    labels = np.asarray(model["labels"])
    counts, kappas, cosines = [], [], []
    for k, axis in enumerate(model["axes"]):
        subset = directions[valid & (labels == k)]
        if len(subset) < 3:
            raise ValueError("insufficient training directions in a component")
        cosine = float(np.mean(subset @ axis))
        counts.append(len(subset))
        cosines.append(cosine)
        kappas.append(fit_concentration(cosine))
    weights = np.asarray(counts, float) / sum(counts)
    return dict(axes=np.asarray(model["axes"]), kappas=np.asarray(kappas),
                weights=weights, spatial_counts=np.asarray(counts),
                train_mean_cosines=np.asarray(cosines))


def mixture_nll(density, directions):
    logs = component_logpdf(directions, density["axes"], density["kappas"])
    return -logsumexp(logs + np.log(density["weights"])[None, :], axis=1)


def single_direction_density(directions):
    """Patient-specific one-component training baseline, without templates."""
    finite = np.asarray(directions)[np.isfinite(directions).all(axis=1)]
    mean = finite.mean(axis=0)
    resultant = float(np.linalg.norm(mean))
    axis = mean / resultant if resultant > 1e-12 else np.array([1., 0., 0.])
    return dict(axes=axis[None, :], kappas=np.array([fit_concentration(resultant)]),
                weights=np.array([1.]))


def evaluate_models(models, densities, ranks, bools, directions):
    """Primary loss uses every estimable direction; conditional metrics share a mask."""
    finite = np.isfinite(directions).all(axis=1)
    assignments = {m: assign_events_to_templates(ranks, bools, models[m]["templates"],
                                                min_shared_channels=3) for m in METHODS}
    common = finite & (assignments[METHODS[0]] >= 0) & (assignments[METHODS[1]] >= 0)
    if finite.sum() == 0 or common.sum() == 0:
        raise ValueError("no evaluable held-out directions")
    metrics, event_losses = {}, {}
    for m in METHODS:
        density = densities[m]
        marginal = mixture_nll(density, directions[finite])
        labels = assignments[m][common]
        cosines = np.clip(np.sum(directions[common] * models[m]["axes"][labels], axis=1), -1, 1)
        conditional_logs = component_logpdf(directions[common], density["axes"], density["kappas"])
        conditional = -conditional_logs[np.arange(len(labels)), labels]
        counts = np.bincount(labels, minlength=2)
        if counts.min() < 3:
            raise ValueError("insufficient conditional test cluster support")
        old_score = np.mean([cosines[labels == k].mean() for k in (0, 1)])
        metrics[m] = dict(nll=float(marginal.mean()), conditional_nll=float(conditional.mean()),
                          angular_mae_deg=float(np.rad2deg(np.arccos(cosines)).mean()),
                          legacy_direction_score=float(old_score),
                          n_test_assigned=int((assignments[m] >= 0).sum()))
        event_losses[m] = dict(marginal_nll=marginal, conditional_nll=conditional)
    return dict(metrics=metrics, event_losses=event_losses, assignments=assignments,
                finite_mask=finite, common_mask=common)


def shuffle_training_directions(directions, blocks, rng):
    """Break rank/direction pairing while preserving block and missingness."""
    shuffled = np.array(directions, copy=True)
    valid = np.isfinite(directions).all(axis=1)
    for block in np.unique(blocks):
        indices = np.flatnonzero((blocks == block) & valid)
        shuffled[indices] = directions[rng.permutation(indices)]
    return shuffled
