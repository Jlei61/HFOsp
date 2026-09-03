"""Fast exact-weight audit for the final rev22 ellipse-plus-learned mapper design."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.topic4_local_connectivity import local_pair_features


PATHWAYS = ("E_to_E", "E_to_I")


def _effective_source_count(rows: np.ndarray, data: np.ndarray, n_targets: int) -> np.ndarray:
    first = np.bincount(rows, weights=data, minlength=n_targets)
    second = np.bincount(rows, weights=data * data, minlength=n_targets)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(second > 0.0, first * first / np.where(second > 0.0, second, 1.0), np.nan)


def _elliptical_radius(dx: np.ndarray, dy: np.ndarray, *, length_scale: float,
                       angle_deg: float, aspect_ratio: float) -> np.ndarray:
    if (not np.isfinite([length_scale, angle_deg, aspect_ratio]).all()
            or float(length_scale) <= 0.0 or float(aspect_ratio) < 1.0):
        raise ValueError("ellipse parameters must be finite, with positive scale and aspect >= 1")
    angle = np.deg2rad(float(angle_deg))
    parallel = float(length_scale) * np.sqrt(float(aspect_ratio))
    perpendicular = float(length_scale) / np.sqrt(float(aspect_ratio))
    u = np.cos(angle) * dx + np.sin(angle) * dy
    v = -np.sin(angle) * dx + np.cos(angle) * dy
    return np.sqrt((u / parallel) ** 2 + (v / perpendicular) ** 2)


def ellipse_reweight(graph: Mapping, *, length_scale: float, angle_deg: float,
                     aspect_ratio: float) -> np.ndarray:
    rows, data = np.asarray(graph["row"], int), np.asarray(graph["data"], float)
    log_ratio = (
        -_elliptical_radius(graph["dx"], graph["dy"], length_scale=length_scale,
                            angle_deg=angle_deg, aspect_ratio=aspect_ratio)
        + _elliptical_radius(graph["dx"], graph["dy"], length_scale=length_scale,
                             angle_deg=45.0, aspect_ratio=2.0)
    )
    raw_ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
    incoming = np.bincount(rows, weights=data, minlength=int(graph["n_e"]))
    denominator = np.bincount(rows, weights=data * raw_ratio, minlength=int(graph["n_e"]))
    if np.any((incoming > 0.0) & (denominator <= 0.0)):
        raise RuntimeError("ellipse mapper produced a zero denominator")
    return data * raw_ratio * incoming[rows] / denominator[rows]


def _weighted_connection_geometry(dx: np.ndarray, dy: np.ndarray, data: np.ndarray) -> dict:
    weight = np.asarray(data, float)
    total = float(np.sum(weight))
    moment = np.asarray([
        [np.sum(weight * dx * dx), np.sum(weight * dx * dy)],
        [np.sum(weight * dx * dy), np.sum(weight * dy * dy)],
    ], float) / total
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    if eigenvalues[0] <= 0.0 or not np.isfinite(eigenvalues).all():
        raise RuntimeError("weighted edge covariance is singular or non-finite")
    axis = eigenvectors[:, -1]
    return {
        "achieved_angle_deg": float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0),
        "achieved_aspect_ratio": float(np.sqrt(eigenvalues[-1] / eigenvalues[0])),
        "weighted_rms_distance_mm": float(np.sqrt(np.trace(moment))),
        "axial_anisotropy": float((eigenvalues[-1] - eigenvalues[0]) / np.sum(eigenvalues)),
    }


def flatten_pathway(net: Mapping, pathway: str) -> dict:
    """Flatten one AMPA pathway in the producer's bin-major ordering."""
    if pathway not in PATHWAYS:
        raise ValueError(f"unknown pathway: {pathway}")
    n_e, n_i = int(net["NE"]), int(net["NI"])
    bins, rows, cols, data = [], [], [], []
    for bin_index, matrix in enumerate(net["ampa_by_delay"]):
        coo = matrix.tocoo(copy=False)
        all_rows = np.asarray(coo.row, np.int64)
        keep = all_rows < n_e if pathway == "E_to_E" else all_rows >= n_e
        selected_rows = all_rows[keep]
        local_rows = selected_rows if pathway == "E_to_E" else selected_rows - n_e
        bins.append(np.full(int(np.sum(keep)), bin_index, np.int64))
        rows.append(local_rows)
        cols.append(np.asarray(coo.col[keep], np.int64))
        data.append(np.asarray(coo.data[keep], float))
    n_targets = n_e if pathway == "E_to_E" else n_i
    return {
        "pathway": pathway,
        "n_e": n_e,
        "n_targets": n_targets,
        "bin": np.concatenate(bins) if bins else np.zeros(0, np.int64),
        "row": np.concatenate(rows) if rows else np.zeros(0, np.int64),
        "col": np.concatenate(cols) if cols else np.zeros(0, np.int64),
        "data": np.concatenate(data) if data else np.zeros(0, float),
    }


def pathway_logits(flat: Mapping, positions: np.ndarray, h_all: np.ndarray,
                   coefficient: np.ndarray, *, length_scale: float,
                   raw_logit_clip: float | None, chunk_size: int = 1_000_000) -> np.ndarray:
    """Evaluate the accepted learned local feature rule without allocating all features."""
    coefficient = np.asarray(coefficient, float)
    if coefficient.shape != (6,):
        raise ValueError("one pathway coefficient row must contain six values")
    if not np.isfinite(coefficient).all() or not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("coefficient and length scale must be finite, with positive scale")
    rows, cols = np.asarray(flat["row"], int), np.asarray(flat["col"], int)
    n_e = int(flat["n_e"])
    target_rows = rows if flat["pathway"] == "E_to_E" else rows + n_e
    output = np.empty(len(rows), float)
    for start in range(0, len(rows), int(chunk_size)):
        stop = min(len(rows), start + int(chunk_size))
        feature = local_pair_features(
            positions[target_rows[start:stop]], positions[cols[start:stop]],
            h_all[target_rows[start:stop]], h_all[cols[start:stop]],
            length_scale=float(length_scale),
        )
        output[start:stop] = feature @ coefficient
    if raw_logit_clip is not None:
        output = np.clip(output, -float(raw_logit_clip), float(raw_logit_clip))
    return output


def target_normalized_reweight(rows: np.ndarray, data: np.ndarray, logits: np.ndarray,
                               n_targets: int) -> np.ndarray:
    """Apply the mapper's stable per-target softmax while preserving incoming totals."""
    rows, data, logits = np.asarray(rows, int), np.asarray(data, float), np.asarray(logits, float)
    if rows.shape != data.shape or rows.shape != logits.shape:
        raise ValueError("rows, data and logits must align")
    incoming = np.bincount(rows, weights=data, minlength=int(n_targets))
    target_max = np.full(int(n_targets), -np.inf, float)
    np.maximum.at(target_max, rows, np.log(data) + logits)
    shifted = np.exp(np.log(data) + logits - target_max[rows])
    denominator = np.bincount(rows, weights=shifted, minlength=int(n_targets))
    if np.any((incoming > 0.0) & (denominator <= 0.0)):
        raise RuntimeError("learned mapper produced a zero denominator")
    return incoming[rows] * shifted / denominator[rows]


def candidate_pathway_weights(
    ee: Mapping,
    etoi: Mapping,
    *,
    positions: np.ndarray,
    h_all: np.ndarray,
    coefficients: np.ndarray,
    g_ee: float,
    g_etoi: float,
    length_scale_ee: float,
    length_scale_etoi: float,
    angle_deg: float,
    aspect_ratio: float,
    raw_logit_clip: float | None,
) -> dict:
    """Compute final pathway weights in exactly the producer's transform order."""
    positions, h_all = np.asarray(positions, float), np.asarray(h_all, float)
    coefficients = np.asarray(coefficients, float)
    if coefficients.shape != (2, 6):
        raise ValueError("learned coefficient matrix must have shape (2, 6)")
    ee_graph = {
        "n_e": int(ee["n_targets"]), "row": np.asarray(ee["row"], int),
        "data": np.asarray(ee["data"], float),
        "dx": positions[np.asarray(ee["col"], int), 0]
              - positions[np.asarray(ee["row"], int), 0],
        "dy": positions[np.asarray(ee["col"], int), 1]
              - positions[np.asarray(ee["row"], int), 1],
    }
    base_logits = {
        "E_to_E": pathway_logits(
            dict(ee, n_e=int(ee["n_targets"])), positions, h_all, coefficients[0],
            length_scale=length_scale_ee, raw_logit_clip=None,
        ),
        "E_to_I": pathway_logits(
            dict(etoi, n_e=int(ee["n_targets"])), positions, h_all, coefficients[1],
            length_scale=length_scale_etoi, raw_logit_clip=None,
        ),
    }
    return candidate_pathway_weights_from_logits(
        ee,
        etoi,
        positions=positions,
        base_logits=base_logits,
        g_ee=g_ee,
        g_etoi=g_etoi,
        length_scale_ee=length_scale_ee,
        angle_deg=angle_deg,
        aspect_ratio=aspect_ratio,
        raw_logit_clip=raw_logit_clip,
    )


def candidate_pathway_weights_from_logits(
    ee: Mapping,
    etoi: Mapping,
    *,
    positions: np.ndarray,
    base_logits: Mapping[str, np.ndarray],
    g_ee: float,
    g_etoi: float,
    length_scale_ee: float,
    angle_deg: float,
    aspect_ratio: float,
    raw_logit_clip: float | None,
) -> dict:
    """Fast candidate weights after topology-specific unscaled logits are cached."""
    positions = np.asarray(positions, float)
    if (not np.isfinite([g_ee, g_etoi, length_scale_ee, angle_deg, aspect_ratio]).all()
            or g_ee < 0.0 or g_etoi < 0.0):
        raise ValueError("candidate doses and geometry must be finite; doses must be non-negative")
    ee_graph = {
        "n_e": int(ee["n_targets"]), "row": np.asarray(ee["row"], int),
        "data": np.asarray(ee["data"], float),
        "dx": positions[np.asarray(ee["col"], int), 0]
              - positions[np.asarray(ee["row"], int), 0],
        "dy": positions[np.asarray(ee["col"], int), 1]
              - positions[np.asarray(ee["row"], int), 1],
    }
    ellipse = ellipse_reweight(
        ee_graph, length_scale=float(length_scale_ee), angle_deg=float(angle_deg),
        aspect_ratio=float(aspect_ratio),
    )
    ee_logits = np.asarray(base_logits["E_to_E"], float) * float(g_ee)
    etoi_logits = np.asarray(base_logits["E_to_I"], float) * float(g_etoi)
    if raw_logit_clip is not None:
        ee_logits = np.clip(ee_logits, -float(raw_logit_clip), float(raw_logit_clip))
        etoi_logits = np.clip(etoi_logits, -float(raw_logit_clip), float(raw_logit_clip))
    return {
        "E_to_E": target_normalized_reweight(
            ee["row"], ellipse, ee_logits, int(ee["n_targets"]),
        ),
        "E_to_I": target_normalized_reweight(
            etoi["row"], etoi["data"], etoi_logits, int(etoi["n_targets"]),
        ),
    }


def pathway_structure(flat: Mapping, final_data: np.ndarray) -> dict:
    """Final-versus-unmodified graph ratios and effective-source support."""
    original, final = np.asarray(flat["data"], float), np.asarray(final_data, float)
    rows, n_targets = np.asarray(flat["row"], int), int(flat["n_targets"])
    ratio = final / original
    incoming_original = np.bincount(rows, weights=original, minlength=n_targets)
    incoming_final = np.bincount(rows, weights=final, minlength=n_targets)
    effective_original = _effective_source_count(rows, original, n_targets)
    effective_final = _effective_source_count(rows, final, n_targets)
    valid = np.isfinite(effective_original) & (effective_original > 0.0)
    effective_ratio = effective_final[valid] / effective_original[valid]
    return {
        "maximum_abs_incoming_error": float(np.max(np.abs(incoming_final - incoming_original), initial=0.0)),
        "edge_ratio_p01": float(np.quantile(ratio, 0.01)),
        "edge_ratio_p50": float(np.quantile(ratio, 0.50)),
        "edge_ratio_p99": float(np.quantile(ratio, 0.99)),
        "effective_source_median_ratio": float(np.median(effective_ratio)),
        "effective_source_p05_ratio": float(np.quantile(effective_ratio, 0.05)),
        "n_edges": int(len(final)),
    }


def candidate_structure(ee: Mapping, etoi: Mapping, final: Mapping,
                        positions: np.ndarray) -> dict:
    output = {
        pathway: pathway_structure(flat, final[pathway])
        for pathway, flat in (("E_to_E", ee), ("E_to_I", etoi))
    }
    rows, cols = np.asarray(ee["row"], int), np.asarray(ee["col"], int)
    output["E_to_E"]["achieved_geometry"] = _weighted_connection_geometry(
        positions[cols, 0] - positions[rows, 0],
        positions[cols, 1] - positions[rows, 1],
        final["E_to_E"],
    )
    return output


def structure_passes(structure: Mapping, thresholds: Mapping) -> bool:
    for pathway in PATHWAYS:
        row = structure[pathway]
        if not (
            row["maximum_abs_incoming_error"] <= thresholds["budget_error_max"]
            and row["edge_ratio_p01"] >= thresholds["edge_ratio_p01_min"]
            and row["edge_ratio_p99"] <= thresholds["edge_ratio_p99_max"]
            and row["effective_source_median_ratio"] >= thresholds["effective_source_median_ratio_min"]
            and row["effective_source_p05_ratio"] >= thresholds["effective_source_p05_ratio_min"]
        ):
            return False
    return True


__all__ = [
    "PATHWAYS", "candidate_pathway_weights", "candidate_pathway_weights_from_logits",
    "candidate_structure", "ellipse_reweight", "flatten_pathway", "pathway_logits",
    "pathway_structure", "structure_passes", "target_normalized_reweight",
]
