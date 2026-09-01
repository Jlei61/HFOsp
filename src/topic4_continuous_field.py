"""Continuous non-component pathology fields for Topic 4 rev10-SA.

The control surface is a tensor-product cubic B-spline.  Its coefficients are
numerical degrees of freedom, not putative cores: the fitted surface may have
zero, one, or several extrema.  The existing level-set projection remains the
only field-mass constraint.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit, logit

from src.topic4_core_field import EPS, TAU_H, project_to_budget


def open_uniform_knots(n_basis, degree=3, L=20.0):
    """Return a clamped open-uniform knot vector on ``[0, L]``."""
    n_basis = int(n_basis)
    degree = int(degree)
    if n_basis < degree + 1:
        raise ValueError("n_basis must be at least degree + 1")
    n_internal = n_basis - degree - 1
    internal = np.linspace(0.0, float(L), n_internal + 2)[1:-1]
    return np.concatenate([
        np.zeros(degree + 1), internal, np.full(degree + 1, float(L)),
    ])


def spline_basis_1d(values, n_basis, degree=3, L=20.0):
    """Evaluate every clamped B-spline basis function at ``values``."""
    values = np.asarray(values, float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if np.any((values < 0.0) | (values > float(L))):
        raise ValueError("spline coordinates must lie inside the sheet")
    knots = open_uniform_knots(n_basis, degree=degree, L=L)
    coefficients = np.eye(int(n_basis))
    basis = BSpline(knots, coefficients, int(degree), extrapolate=False)(values)
    if not np.isfinite(basis).all():
        raise RuntimeError("B-spline basis produced non-finite values")
    return np.asarray(basis, float)


def tensor_basis(positions, n_basis, degree=3, L=20.0):
    """Return the continuous tensor-product basis at 2-D sheet positions."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    bx = spline_basis_1d(positions[:, 0], n_basis, degree=degree, L=L)
    by = spline_basis_1d(positions[:, 1], n_basis, degree=degree, L=L)
    return np.einsum("ni,nj->nij", bx, by).reshape(len(positions), -1)


def continuous_surface(coefficients, positions, *, n_basis, degree=3, L=20.0):
    """Evaluate the unconstrained latent surface ``s(x, y)``."""
    coefficients = np.asarray(coefficients, float)
    expected = int(n_basis) ** 2
    if coefficients.size != expected:
        raise ValueError(f"expected {expected} coefficients, got {coefficients.size}")
    if not np.isfinite(coefficients).all():
        raise ValueError("continuous-field coefficients must be finite")
    # A constant coefficient shift is unidentifiable after mass projection.
    centered = coefficients.reshape(-1) - float(np.mean(coefficients))
    return tensor_basis(positions, n_basis, degree=degree, L=L) @ centered


def continuous_field_h(coefficients, positions, *, n_basis, target_count,
                       degree=3, L=20.0):
    """Map a continuous latent surface to ``h in (0, 1)`` at exact mass."""
    surface = continuous_surface(
        coefficients, positions, n_basis=n_basis, degree=degree, L=L,
    )
    q = np.exp(np.clip(surface, -30.0, 30.0))
    h, threshold = project_to_budget(q, float(target_count))
    return h, {
        "surface_min": float(np.min(surface)),
        "surface_max": float(np.max(surface)),
        "projection_threshold": float(threshold),
    }


def continuous_field_h_with_queries(
        coefficients, reference_positions, query_positions, *, n_basis,
        target_count, degree=3, L=20.0):
    """Evaluate one budget-projected field at reference and query positions.

    The level-set threshold is fitted only on ``reference_positions``. Query
    values use that same threshold, so a second neuron population does not get
    a separately normalized field.
    """
    reference_surface = continuous_surface(
        coefficients, reference_positions, n_basis=n_basis, degree=degree, L=L,
    )
    reference_q = np.exp(np.clip(reference_surface, -30.0, 30.0))
    reference_h, threshold = project_to_budget(reference_q, float(target_count))
    query_surface = continuous_surface(
        coefficients, query_positions, n_basis=n_basis, degree=degree, L=L,
    )
    query_q = np.exp(np.clip(query_surface, -30.0, 30.0))
    query_h = expit((np.log(query_q + EPS) - threshold) / TAU_H)
    return reference_h, query_h, {
        "reference_surface_min": float(np.min(reference_surface)),
        "reference_surface_max": float(np.max(reference_surface)),
        "query_surface_min": float(np.min(query_surface)),
        "query_surface_max": float(np.max(query_surface)),
        "projection_threshold": float(threshold),
    }


def distance_to_segments(positions, segments):
    """Euclidean distance to a union of line segments."""
    positions = np.asarray(positions, float)
    segments = np.asarray(segments, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or not len(segments):
        raise ValueError("segments must have shape (n_segments, 2, 2)")
    start = segments[:, 0]
    delta = segments[:, 1] - start
    denominator = np.sum(delta ** 2, axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("support segments must have positive length")
    offset = positions[:, None, :] - start[None, :, :]
    fraction = np.clip(
        np.sum(offset * delta[None, :, :], axis=2) / denominator[None, :],
        0.0, 1.0,
    )
    projection = start[None, :, :] + fraction[:, :, None] * delta[None, :, :]
    return np.min(
        np.linalg.norm(positions[:, None, :] - projection, axis=2), axis=1,
    )


def continuous_corridor_field_h(segments, positions, *, width_mm, target_count):
    """Project a smooth distance-to-path field to an exact mass budget.

    The path is a capacity positive control, not a collection of components.
    Neither its segment count nor its extrema are interpreted as biological
    cores.
    """
    width_mm = float(width_mm)
    if not np.isfinite(width_mm) or width_mm <= 0.0:
        raise ValueError("width_mm must be positive")
    distance = distance_to_segments(positions, segments)
    q = np.exp(-0.5 * (distance / width_mm) ** 2)
    h, threshold = project_to_budget(q, float(target_count))
    return h, {
        "distance_min_mm": float(np.min(distance)),
        "distance_median_mm": float(np.median(distance)),
        "projection_threshold": float(threshold),
    }


def shaft_polyline_segments(contacts):
    """Build the two observed shaft polylines from the frozen contact contract."""
    segments = []
    for shaft in ("ICL", "SCL"):
        rows = sorted(
            (row for row in contacts if row["shaft_id"] == shaft),
            key=lambda row: int(row["within_shaft_order_by_shared_axis"]),
        )
        points = np.asarray([row["sheet_xy_mm"] for row in rows], float)
        if len(points) < 2:
            raise ValueError(f"shaft {shaft} needs at least two contacts")
        segments.extend(np.stack([points[:-1], points[1:]], axis=1))
    return np.asarray(segments, float)


def closest_cross_shaft_segment(contacts):
    """Return the shortest observed ICL-to-SCL contact bridge."""
    icl = np.asarray([
        row["sheet_xy_mm"] for row in contacts if row["shaft_id"] == "ICL"
    ], float)
    scl = np.asarray([
        row["sheet_xy_mm"] for row in contacts if row["shaft_id"] == "SCL"
    ], float)
    if not len(icl) or not len(scl):
        raise ValueError("both shafts are required")
    distance = np.linalg.norm(icl[:, None, :] - scl[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    return np.asarray([icl[i], scl[j]], float)


def build_continuous_support_candidates(contacts, *, widths_mm):
    """Construct no-K shaft-support controls for the field-capacity assay."""
    shaft_segments = shaft_polyline_segments(contacts)
    bridge = closest_cross_shaft_segment(contacts)
    candidates = []
    for support_id, segments in (
        ("dual_shaft_disconnected", shaft_segments),
        ("dual_shaft_connected", np.concatenate([
            shaft_segments, bridge[None, :, :],
        ], axis=0)),
    ):
        for width_mm in widths_mm:
            metadata = {
                "candidate_id": (
                    f"corridor_{support_id}_w{str(width_mm).replace('.', 'p')}"
                ),
                "field_type": "continuous_corridor",
                "support_id": support_id,
                "width_mm": float(width_mm),
                "segments": np.asarray(segments, float).tolist(),
                "bridge_segment": (
                    bridge.tolist() if support_id == "dual_shaft_connected" else None
                ),
                "component_count": None,
                "peak_count_constraint": None,
                "no_component_or_peak_assignment": True,
            }
            metadata["field_sha256"] = continuous_candidate_hash({
                key: metadata[key] for key in (
                    "field_type", "support_id", "width_mm", "segments",
                )
            })
            candidates.append(metadata)
    if len({row["field_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("continuous support candidates are not unique")
    return {
        "shaft_segments": shaft_segments,
        "bridge_segment": bridge,
        "candidates": candidates,
    }


def second_difference_operator(n_basis):
    """Second differences along both control-surface axes."""
    n_basis = int(n_basis)
    rows = []
    for axis in (0, 1):
        for fixed in range(n_basis):
            for index in range(1, n_basis - 1):
                row = np.zeros((n_basis, n_basis), float)
                if axis == 0:
                    row[index - 1:index + 2, fixed] = [1.0, -2.0, 1.0]
                else:
                    row[fixed, index - 1:index + 2] = [1.0, -2.0, 1.0]
                rows.append(row.ravel())
    return np.asarray(rows, float)


def curvature_energy(coefficients, n_basis):
    coefficients = np.asarray(coefficients, float).reshape(-1)
    operator = second_difference_operator(n_basis)
    return float(np.mean((operator @ coefficients) ** 2))


def shaft_balanced_contact_weights(shaft_ids):
    """Give ICL and SCL equal total weight without assigning field components."""
    shaft_ids = np.asarray(shaft_ids).astype(str)
    unique = sorted(set(shaft_ids.tolist()))
    if unique != ["ICL", "SCL"]:
        raise ValueError("shaft_ids must contain exactly ICL and SCL")
    weights = np.zeros(len(shaft_ids), float)
    for shaft in unique:
        selected = shaft_ids == shaft
        weights[selected] = 0.5 / int(np.sum(selected))
    return weights


def background_anchors(contact_xy, *, L=20.0, spacing_mm=2.5,
                       exclusion_radius_mm=2.0):
    """Sheet-wide baseline anchors selected without shaft labels."""
    contact_xy = np.asarray(contact_xy, float)
    axis = np.arange(0.0, float(L) + 0.5 * float(spacing_mm), float(spacing_mm))
    xx, yy = np.meshgrid(axis, axis)
    anchors = np.column_stack([xx.ravel(), yy.ravel()])
    distance = np.min(
        np.linalg.norm(anchors[:, None, :] - contact_xy[None, :, :], axis=2),
        axis=1,
    )
    selected = anchors[distance >= float(exclusion_radius_mm)]
    if not len(selected):
        raise RuntimeError("background-anchor rule selected no sheet locations")
    return selected


def fit_contact_target(contact_xy, target, shaft_ids, *, n_basis,
                       roughness=0.3, ridge=1e-3, degree=3, L=20.0,
                       background_weight=0.25, background_probability=0.05,
                       background_spacing_mm=2.5,
                       background_exclusion_radius_mm=2.0):
    """Fit a smooth surface to contact values with equal total shaft weight."""
    target = np.asarray(target, float)
    if target.shape != (len(contact_xy),) or not np.isfinite(target).all():
        raise ValueError("target must provide one finite value per contact")
    clipped = np.clip(target, 0.02, 0.98)
    contact_latent = logit(clipped)
    background = background_anchors(
        contact_xy, L=L, spacing_mm=background_spacing_mm,
        exclusion_radius_mm=background_exclusion_radius_mm,
    )
    background_weight = float(background_weight)
    if not 0.0 < background_weight < 1.0:
        raise ValueError("background_weight must lie in (0, 1)")
    contact_weights = ((1.0 - background_weight)
                       * shaft_balanced_contact_weights(shaft_ids))
    background_weights = np.full(
        len(background), background_weight / len(background), float,
    )
    weights = np.concatenate([contact_weights, background_weights])
    latent_target = np.concatenate([
        contact_latent,
        np.full(len(background), logit(float(background_probability))),
    ])
    latent_target -= np.sum(weights * latent_target)
    fit_xy = np.vstack([contact_xy, background])
    basis = tensor_basis(fit_xy, n_basis, degree=degree, L=L)
    root_weight = np.sqrt(weights)
    curvature = second_difference_operator(n_basis)
    identity = np.eye(int(n_basis) ** 2)
    design = np.vstack([
        root_weight[:, None] * basis,
        np.sqrt(float(roughness) / max(1, len(curvature))) * curvature,
        np.sqrt(float(ridge) / len(identity)) * identity,
    ])
    response = np.concatenate([
        root_weight * latent_target,
        np.zeros(len(curvature) + len(identity)),
    ])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    coefficients -= float(np.mean(coefficients))
    fitted = basis @ coefficients
    contact_fit = fitted[:len(contact_xy)]
    background_fit = fitted[len(contact_xy):]
    balanced_contact = shaft_balanced_contact_weights(shaft_ids)
    return coefficients, {
        "weighted_contact_rmse": float(np.sqrt(np.sum(
            balanced_contact * (contact_fit - latent_target[:len(contact_xy)]) ** 2
        ))),
        "background_rmse": float(np.sqrt(np.mean(
            (background_fit - latent_target[len(contact_xy):]) ** 2
        ))),
        "curvature_energy": curvature_energy(coefficients, n_basis),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "n_background_anchors": int(len(background)),
        "background_weight": background_weight,
        "background_probability": float(background_probability),
    }


def patient_contact_targets(onsets, labels):
    """Construct direction-preserving contact targets from patient training data."""
    onsets = np.asarray(onsets, float)
    labels = np.asarray(labels, int)
    if onsets.ndim != 2 or labels.shape != (len(onsets),):
        raise ValueError("onsets and labels do not align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("patient labels must contain modes 0 and 1")
    recruited = np.isfinite(onsets)
    recruitment = np.asarray([
        recruited[labels == mode].mean(axis=0) for mode in (0, 1)
    ])
    earliest = []
    for mode in (0, 1):
        selected = onsets[labels == mode]
        valid = np.isfinite(selected)
        normalized = np.full(selected.shape, np.nan)
        for index, row in enumerate(selected):
            mask = valid[index]
            if not np.any(mask):
                continue
            values = row[mask]
            span = float(np.max(values) - np.min(values))
            normalized[index, mask] = (
                (values - np.min(values)) / (span + 1e-12)
            )
        earliest.append(np.mean(valid & (normalized <= 0.25), axis=0))
    earliest = np.asarray(earliest)
    return {
        "uniform_contact_support": np.full(onsets.shape[1], 0.75),
        "mode_A_recruitment": recruitment[0],
        "mode_B_recruitment": recruitment[1],
        "weakest_mode_recruitment": np.minimum(recruitment[0], recruitment[1]),
        "either_mode_early_support": np.maximum(earliest[0], earliest[1]),
    }


def continuous_candidate_hash(candidate):
    payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_continuous_field_candidates(contact_xy, shaft_ids, onsets, labels, *,
                                      designs, degree=3, L=20.0,
                                      fit_options=None):
    """Build deterministic continuous fields without component or peak labels."""
    targets = patient_contact_targets(onsets, labels)
    fit_options = dict(fit_options or {})
    primary_basis = int(designs[0]["n_basis"])
    uniform = {
        "candidate_id": f"bs{primary_basis}_uniform_sheet",
        "field_type": "continuous_bspline",
        "target_id": "uniform_sheet",
        "n_basis": primary_basis,
        "degree": int(degree),
        "roughness": None,
        "contrast": 0.0,
        "coefficients": np.zeros(primary_basis ** 2).tolist(),
        "fit_diagnostics": {
            "weighted_contact_rmse": 0.0,
            "curvature_energy": 0.0,
            "coefficient_l2": 0.0,
        },
        "no_component_or_peak_assignment": True,
    }
    uniform["field_sha256"] = continuous_candidate_hash({
        key: uniform[key] for key in (
            "field_type", "n_basis", "degree", "coefficients",
        )
    })
    uniform["target_aliases"] = [uniform["target_id"]]
    candidates = [uniform]
    for design in designs:
        n_basis = int(design["n_basis"])
        for target_id, target in targets.items():
            for roughness in design["roughness"]:
                base, fit = fit_contact_target(
                    contact_xy, target, shaft_ids, n_basis=n_basis,
                    roughness=float(roughness), degree=degree, L=L,
                    **fit_options,
                )
                for contrast in design["contrast"]:
                    coefficients = float(contrast) * base
                    metadata = {
                        "field_type": "continuous_bspline",
                        "target_id": target_id,
                        "n_basis": n_basis,
                        "degree": int(degree),
                        "roughness": float(roughness),
                        "contrast": float(contrast),
                        "coefficients": coefficients.tolist(),
                        "fit_diagnostics": fit,
                        "no_component_or_peak_assignment": True,
                    }
                    slug = str(roughness).replace(".", "p")
                    cslug = str(contrast).replace(".", "p")
                    metadata["candidate_id"] = (
                        f"bs{n_basis}_{target_id}_r{slug}_c{cslug}"
                    )
                    metadata["field_sha256"] = continuous_candidate_hash({
                        key: metadata[key] for key in (
                            "field_type", "n_basis", "degree", "coefficients",
                        )
                    })
                    duplicate = next((
                        row for row in candidates
                        if row["field_sha256"] == metadata["field_sha256"]
                    ), None)
                    if duplicate is None:
                        metadata["target_aliases"] = [target_id]
                        candidates.append(metadata)
                    elif target_id not in duplicate["target_aliases"]:
                        duplicate["target_aliases"].append(target_id)
    identifiers = [row["candidate_id"] for row in candidates]
    hashes = [row["field_sha256"] for row in candidates]
    if len(set(identifiers)) != len(identifiers) or len(set(hashes)) != len(hashes):
        raise RuntimeError("continuous-field candidates are not unique")
    return {"targets": targets, "candidates": candidates}
