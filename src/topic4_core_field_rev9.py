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
                            core_std, v_base, depth_shrinkage=1.0,
                            node_gain=1.0):
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
    rho = float(depth_shrinkage)
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("depth_shrinkage must lie in [0, 1]")
    h_mass = float(np.sum(h))
    if h_mass <= 0.0 and (rho != 1.0 or float(node_gain) != 1.0):
        raise ValueError("non-default node modulation requires positive field mass")
    # The main-line default historically permits an all-background field.
    weighted_mean = float(np.sum(h * d) / h_mass) if h_mass > 0.0 else 0.0
    # Preserve the historical floating-point path exactly at rho=1.
    d_shrunk = d if rho == 1.0 else weighted_mean + rho * (d - weighted_mean)
    gain = float(node_gain)
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("node_gain must be finite and nonnegative")
    # Preserve the historical floating-point path exactly at gain=1.
    d_effective = d_shrunk if gain == 1.0 else gain * d_shrunk
    original_budget = float(np.sum(h * d))
    shrinkage_budget = float(np.sum(h * d_shrunk))
    effective_budget = float(np.sum(h * d_effective))
    weights = h / h_mass if h_mass > 0.0 else np.zeros_like(h)
    weighted_sd_original = float(np.sqrt(np.sum(weights * (d - weighted_mean) ** 2)))
    shrinkage_mean = float(np.sum(weights * d_shrunk))
    weighted_sd_shrunk = float(np.sqrt(
        np.sum(weights * (d_shrunk - shrinkage_mean) ** 2)
    ))
    effective_mean = float(np.sum(weights * d_effective))
    weighted_sd_effective = float(np.sqrt(
        np.sum(weights * (d_effective - effective_mean) ** 2)
    ))
    vtheta = build_vth(h, d_effective, n_total=int(n_total), n_E=n_e,
                       v_base=float(v_base))
    return dict(
        h=h, d=d, d_shrunk=d_shrunk, d_effective=d_effective, vtheta=vtheta,
        delta_vtheta=-h * d_effective,
        mapping_audit=dict(
            signed_depth_shrinkage=rho,
            node_gain=gain,
            h_weighted_mean_depth=weighted_mean,
            h_weighted_modulation_original=original_budget,
            h_weighted_modulation_after_shrinkage=shrinkage_budget,
            h_weighted_modulation_effective=effective_budget,
            budget_error=shrinkage_budget - original_budget,
            gain_application_error=(
                effective_budget - gain * shrinkage_budget
            ),
            h_weighted_depth_sd_original=weighted_sd_original,
            h_weighted_depth_sd_after_shrinkage=weighted_sd_shrunk,
            h_weighted_depth_sd_effective=weighted_sd_effective,
            latent_negative_fraction=float(np.mean(d < 0.0)),
            shrunk_negative_fraction=float(np.mean(d_shrunk < 0.0)),
            effective_negative_fraction=float(np.mean(d_effective < 0.0)),
        ),
        hashes=dict(
            h_vector_sha256=array_sha256(h),
            d_vector_sha256=array_sha256(d),
            d_shrunk_vector_sha256=array_sha256(d_shrunk),
            d_effective_vector_sha256=array_sha256(d_effective),
            vtheta_reconstructed_sha256=array_sha256(vtheta),
        ),
    )


def reconstruct_node_from_dual_fields(h_mean, h_dispersion, *, n_total,
                                      quantile_seed, core_mean, core_std,
                                      v_base):
    """Separate the smooth mean-excitability and signed-dispersion envelopes.

    The mapping is exactly the historical ``-h*d`` mapping when the two fields
    are identical.  For distinct fields, the dispersion residual is centered
    under its own field so it cannot silently change the total threshold
    budget carried by the mean field.
    """
    h_mean = np.asarray(h_mean, float)
    h_dispersion = np.asarray(h_dispersion, float)
    if h_mean.ndim != 1 or not len(h_mean) or h_mean.shape != h_dispersion.shape:
        raise ValueError("dual Node fields must be aligned non-empty vectors")
    for name, values in (("h_mean", h_mean), ("h_dispersion", h_dispersion)):
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if float(np.sum(values)) <= 0.0:
            raise ValueError(f"{name} must have positive mass")
    n_e = len(h_mean)
    if int(n_total) < n_e:
        raise ValueError("n_total cannot be smaller than the E-neuron field")
    quantiles = sample_core_quantiles(n_e, int(quantile_seed))
    d = signed_depth(core_thresholds(
        quantiles, float(core_mean), float(core_std)), float(v_base))
    mean_depth = float(np.sum(h_mean * d) / np.sum(h_mean))
    dispersion_center = float(
        np.sum(h_dispersion * d) / np.sum(h_dispersion)
    )
    dispersion_residual = d - dispersion_center
    mean_component = -h_mean * mean_depth
    dispersion_component = -h_dispersion * dispersion_residual
    if not np.isclose(float(np.sum(dispersion_component)), 0.0, atol=1e-9):
        raise RuntimeError("dual Node dispersion channel changed total threshold budget")
    same_field = bool(np.array_equal(h_mean, h_dispersion))
    parity_error = None
    if same_field:
        historical = reconstruct_node_from_h(
            h_mean, n_total=n_total, quantile_seed=quantile_seed,
            core_mean=core_mean, core_std=core_std, v_base=v_base,
        )
        delta_vtheta = historical["delta_vtheta"]
        vtheta = historical["vtheta"]
        parity_error = 0.0
    else:
        delta_vtheta = mean_component + dispersion_component
        vtheta = np.full(int(n_total), float(v_base), dtype=float)
        vtheta[:n_e] += delta_vtheta
    return dict(
        h=h_mean,
        h_mean=h_mean,
        h_dispersion=h_dispersion,
        d=d,
        d_shrunk=d,
        d_effective=d,
        vtheta=vtheta,
        delta_vtheta=delta_vtheta,
        mean_component=mean_component,
        dispersion_component=dispersion_component,
        mapping_audit=dict(
            mapping_type="dual_continuous_mean_dispersion",
            mean_depth=mean_depth,
            dispersion_center=dispersion_center,
            mean_component_sum=float(np.sum(mean_component)),
            dispersion_component_sum=float(np.sum(dispersion_component)),
            decomposition_max_abs_error=float(np.max(np.abs(
                delta_vtheta - (mean_component + dispersion_component)
            ), initial=0.0)),
            total_modulation=float(np.sum(delta_vtheta)),
            identical_fields=same_field,
            historical_parity_max_abs_error=parity_error,
        ),
        hashes=dict(
            h_vector_sha256=array_sha256(h_mean),
            h_mean_vector_sha256=array_sha256(h_mean),
            h_dispersion_vector_sha256=array_sha256(h_dispersion),
            d_vector_sha256=array_sha256(d),
            d_shrunk_vector_sha256=array_sha256(d),
            d_effective_vector_sha256=array_sha256(d),
            mean_component_sha256=array_sha256(mean_component),
            dispersion_component_sha256=array_sha256(dispersion_component),
            delta_vtheta_sha256=array_sha256(delta_vtheta),
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
