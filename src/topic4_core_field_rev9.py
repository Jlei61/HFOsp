"""Frozen-field reconstruction and exploratory rev9 readout helpers."""
from __future__ import annotations

import hashlib

import numpy as np

from src.topic4_core_field import (build_vth, core_thresholds,
                                   sample_core_quantiles, signed_depth)
from src.topic4_core_field_profile import transform_rank_curves
from src.topic4_core_field_stage3 import params_to_h, unpack


def array_sha256(values):
    arr = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def reconstruct_frozen_node(theta, pos_e, *, n_total, target_count,
                            quantile_seed, core_mean, core_std, v_base,
                            K=3, L=20.0):
    """Rebuild rev8's independent ``h``, signed ``d``, and threshold vector.

    ``d`` is regenerated from the original quantile seed. It is never inferred
    by dividing an already modulated threshold vector by ``h``.
    """
    pos_e = np.asarray(pos_e, float)
    n_e = len(pos_e)
    h = params_to_h(np.asarray(theta, float), pos_e, int(K), float(L),
                    float(target_count))
    return reconstruct_node_from_h(
        h, n_total=n_total, quantile_seed=quantile_seed,
        core_mean=core_mean, core_std=core_std, v_base=v_base,
    )


def reconstruct_node_from_h(h, *, n_total, quantile_seed, core_mean,
                            core_std, v_base):
    """Apply the frozen signed threshold depths to any valid continuous field."""
    h = np.asarray(h, float)
    if h.ndim != 1 or not len(h):
        raise ValueError("h must be a non-empty one-dimensional E-neuron field")
    if not np.isfinite(h).all() or np.any((h < 0.0) | (h > 1.0)):
        raise ValueError("h must be finite and lie in [0, 1]")
    n_e = len(h)
    if int(n_total) < n_e:
        raise ValueError("n_total cannot be smaller than the E-neuron field")
    quantiles = sample_core_quantiles(n_e, int(quantile_seed))
    d = signed_depth(core_thresholds(
        quantiles, float(core_mean), float(core_std)), float(v_base))
    vtheta = build_vth(h, d, n_total=int(n_total), n_E=n_e,
                       v_base=float(v_base))
    return dict(
        h=h, d=d, vtheta=vtheta, delta_vtheta=-h * d,
        hashes=dict(
            h_vector_sha256=array_sha256(h),
            d_vector_sha256=array_sha256(d),
            vtheta_reconstructed_sha256=array_sha256(vtheta),
        ),
    )


def node_reconstruction_error(reconstructed_vtheta, frozen_vtheta):
    left = np.asarray(reconstructed_vtheta)
    right = np.asarray(frozen_vtheta)
    if left.shape != right.shape:
        raise ValueError("reconstructed and frozen threshold vectors differ in shape")
    delta = np.asarray(left, float) - np.asarray(right, float)
    return dict(
        exact=bool(np.array_equal(left, right)),
        max_abs_error=float(np.max(np.abs(delta), initial=0.0)),
        reconstructed_sha256=array_sha256(left),
        frozen_sha256=array_sha256(right),
    )


def component_contributions(theta, positions, *, K=3, L=20.0):
    """Return each Gaussian's raw contribution without the field EPS floor."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    columns = []
    for component in unpack(np.asarray(theta, float), int(K), float(L)):
        cosine, sine = np.cos(component["phi"]), np.sin(component["phi"])
        delta = positions - component["center"]
        along = delta[:, 0] * cosine + delta[:, 1] * sine
        across = -delta[:, 0] * sine + delta[:, 1] * cosine
        columns.append(component["weight"] * np.exp(-0.5 * (
            (along / component["sigma_par"]) ** 2
            + (across / component["sigma_perp"]) ** 2)))
    return np.column_stack(columns) if columns else np.empty((len(positions), 0))


def component_responsibilities(theta, positions, *, K=3, L=20.0):
    """Soft component assignment at locations, based on raw Gaussian mass."""
    contribution = component_contributions(theta, positions, K=K, L=L)
    total = contribution.sum(axis=1, keepdims=True)
    responsibility = np.divide(
        contribution, total, out=np.zeros_like(contribution), where=total > 0.0)
    return dict(
        contributions=contribution,
        responsibilities=responsibility,
        assignments=np.argmax(responsibility, axis=1).astype(int),
        maximum_responsibility=np.max(responsibility, axis=1),
    )


def fit_frozen_mode_classifier(curves, labels, reference, *, ood_quantile=0.99):
    """Freeze nearest-centroid mode assignment in the existing PCA space."""
    curves = np.asarray(curves, float)
    labels = np.asarray(labels, int)
    if curves.ndim != 2 or labels.shape != (len(curves),):
        raise ValueError("curves and labels do not align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("the frozen classifier requires exactly labels 0 and 1")
    quantile = float(ood_quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("ood_quantile must lie in (0, 1)")
    embedded = transform_rank_curves(curves, reference)
    centroids = np.asarray([
        embedded[labels == mode].mean(axis=0) for mode in (0, 1)
    ])
    distance_matrix = np.linalg.norm(
        embedded[:, None, :] - centroids[None, :, :], axis=2)
    assigned = np.argmin(distance_matrix, axis=1)
    if not np.array_equal(assigned, labels):
        mismatches = int(np.count_nonzero(assigned != labels))
        raise RuntimeError(
            f"frozen labels are not nearest-centroid separable ({mismatches} mismatches)")
    assigned_distance = distance_matrix[np.arange(len(labels)), labels]
    thresholds = np.asarray([
        np.quantile(assigned_distance[labels == mode], quantile)
        for mode in (0, 1)
    ])
    return dict(
        embedding_centroids=centroids,
        ood_distance_thresholds=thresholds,
        ood_quantile=quantile,
        baseline_embedded=embedded,
        baseline_assigned_distance=assigned_distance,
        baseline_counts=np.bincount(labels, minlength=2),
    )


def assign_frozen_modes(curves, classifier, reference):
    """Assign events to frozen modes and flag distances beyond baseline p99."""
    embedded = transform_rank_curves(np.asarray(curves, float), reference)
    centroids = np.asarray(classifier["embedding_centroids"], float)
    thresholds = np.asarray(classifier["ood_distance_thresholds"], float)
    if centroids.ndim != 2 or centroids.shape[0] != 2:
        raise ValueError("classifier must contain two embedding centroids")
    if thresholds.shape != (2,):
        raise ValueError("classifier must contain two OOD thresholds")
    distances = np.linalg.norm(
        embedded[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1).astype(int)
    assigned_distance = distances[np.arange(len(labels)), labels]
    return dict(
        labels=labels,
        distance_matrix=distances,
        assigned_distance=assigned_distance,
        ood=assigned_distance > thresholds[labels],
    )
