"""Continuous field-coupled local AMPA redistribution for Topic 4 rev11-NLC."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from src.topic4_core_connectivity import _hash_sparse_bins, _invalidate_ampa_caches


FEATURE_NAMES = (
    "source_field_contrast",
    "source_target_field_interaction",
    "source_field_locality",
    "source_target_field_locality",
    "source_target_flow_x",
    "source_target_flow_y",
)
PATHWAYS = ("E_to_E", "E_to_I")


def _array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def local_pair_features(target_xy, source_xy, target_h, source_h, *, length_scale):
    """Return observation-invariant continuous features for existing E edges."""
    target_xy = np.asarray(target_xy, float)
    source_xy = np.asarray(source_xy, float)
    target_h = np.asarray(target_h, float)
    source_h = np.asarray(source_h, float)
    if target_xy.shape != source_xy.shape or target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("target/source coordinates must align as (edges, 2)")
    if target_h.shape != (len(target_xy),) or source_h.shape != (len(target_xy),):
        raise ValueError("target/source field values must align with edges")
    length_scale = float(length_scale)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    if (not np.isfinite(target_h).all() or not np.isfinite(source_h).all()
            or np.any((target_h < 0.0) | (target_h > 1.0))
            or np.any((source_h < 0.0) | (source_h > 1.0))):
        raise ValueError("field values must be finite and lie in [0, 1]")

    displacement = (source_xy - target_xy) / length_scale
    radius = np.linalg.norm(displacement, axis=1)
    q_target = 2.0 * target_h - 1.0
    q_source = 2.0 * source_h - 1.0
    interaction = q_target * q_source
    return np.column_stack((
        q_source,
        interaction,
        -q_source * radius,
        -interaction * radius,
        interaction * displacement[:, 0],
        interaction * displacement[:, 1],
    ))


def _coefficient_array(coefficients):
    if coefficients is None:
        return np.zeros((2, len(FEATURE_NAMES)), float)
    if isinstance(coefficients, dict):
        unknown = set(coefficients) - set(PATHWAYS)
        if unknown:
            raise ValueError(f"unknown pathways: {sorted(unknown)}")
        values = np.vstack([
            np.asarray(coefficients.get(pathway, np.zeros(len(FEATURE_NAMES))), float)
            for pathway in PATHWAYS
        ])
    else:
        values = np.asarray(coefficients, float)
    if values.shape != (2, len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise ValueError(f"coefficients must have shape (2, {len(FEATURE_NAMES)})")
    return values


def _pathway_mask(rows, n_e, pathway):
    if pathway == "E_to_E":
        return rows < n_e
    if pathway == "E_to_I":
        return rows >= n_e
    raise ValueError(f"unsupported pathway {pathway}")


def _summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"min": None, "p01": None, "median": None, "p99": None, "max": None}
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _incoming_by_pathway(matrices, n_e, n_i, pathway):
    size = n_e if pathway == "E_to_E" else n_i
    total = np.zeros(size, float)
    for matrix in matrices:
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        mask = _pathway_mask(rows, n_e, pathway)
        local_rows = rows[mask] if pathway == "E_to_E" else rows[mask] - n_e
        total += np.bincount(
            local_rows, weights=np.asarray(coo.data[mask], float), minlength=size,
        )
    return total


def continuous_local_e_source_flow(
        net, positions, h_all, coefficients, *, l_ee, l_e_to_i,
        raw_logit_clip=1.5, ratio_sample_limit=500_000):
    """Redistribute existing E->E and E->I weights under separate target budgets.

    AMPA rows are targets and columns are E sources. Every pathway is normalized
    over all delay bins for each postsynaptic target. Topology, delay assignment,
    all GABA matrices and both pathway-specific incoming totals are preserved.
    """
    n_e, n_i = int(net["NE"]), int(net["NI"])
    n_total = n_e + n_i
    positions = np.asarray(positions, float)
    h_all = np.asarray(h_all, float)
    values = _coefficient_array(coefficients)
    if positions.shape != (n_total, 2):
        raise ValueError(f"positions must have shape ({n_total}, 2)")
    if h_all.shape != (n_total,) or not np.isfinite(h_all).all() or np.any((h_all < 0) | (h_all > 1)):
        raise ValueError(f"h_all must be finite in [0, 1] with shape ({n_total},)")
    scales = {"E_to_E": float(l_ee), "E_to_I": float(l_e_to_i)}
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError("pathway length scales must be finite and positive")
    if raw_logit_clip is not None:
        raw_logit_clip = float(raw_logit_clip)
        if not np.isfinite(raw_logit_clip) or raw_logit_clip <= 0.0:
            raise ValueError("raw_logit_clip must be finite and positive")

    old_bins = net["ampa_by_delay"]
    old_topology = _hash_sparse_bins(old_bins, include_data=False)
    old_gaba = _hash_sparse_bins(net["gaba_by_delay"])
    old_data = _hash_sparse_bins(old_bins)
    old_incoming = {
        pathway: _incoming_by_pathway(old_bins, n_e, n_i, pathway)
        for pathway in PATHWAYS
    }
    audit = {
        "mechanism": "continuous_field_coupled_local_E_source_flow_v1",
        "feature_names": list(FEATURE_NAMES),
        "pathways": list(PATHWAYS),
        "coefficients": {pathway: values[index].tolist()
                         for index, pathway in enumerate(PATHWAYS)},
        "coefficients_sha256": _array_sha256(values),
        "length_scales_mm": scales,
        "raw_logit_clip_abs": raw_logit_clip,
    }

    if np.all(values == 0.0):
        new_net = copy.copy(net)
        new_net["ampa_by_delay"] = [matrix.copy() for matrix in old_bins]
        removed = _invalidate_ampa_caches(new_net)
        return new_net, {
            **audit,
            "exact_noop": True,
            "edge_ratio": _summary(np.ones(1)),
            "topology_unchanged": True,
            "delay_assignment_unchanged": True,
            "gaba_unchanged": True,
            "ampa_data_unchanged": True,
            "invalidated_ampa_cache_keys": removed,
            "pathway_audit": {
                pathway: {
                    "n_edges": int(sum(matrix[:n_e, :].nnz for matrix in old_bins))
                    if pathway == "E_to_E" else
                    int(sum(matrix[n_e:, :].nnz for matrix in old_bins)),
                    "max_abs_incoming_error": 0.0,
                    "edge_ratio": _summary(np.ones(1)),
                }
                for pathway in PATHWAYS
            },
        }

    def logits_for(rows, columns, pathway, coefficient):
        feature = local_pair_features(
            positions[rows], positions[columns], h_all[rows], h_all[columns],
            length_scale=scales[pathway],
        )
        raw = feature @ coefficient
        return raw if raw_logit_clip is None else np.clip(raw, -raw_logit_clip, raw_logit_clip)

    target_max = {
        "E_to_E": np.full(n_e, -np.inf, float),
        "E_to_I": np.full(n_i, -np.inf, float),
    }
    edge_counts = {pathway: 0 for pathway in PATHWAYS}
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        columns = np.asarray(coo.col, np.int64)
        data = np.asarray(coo.data, float)
        if np.any(~np.isfinite(data)) or np.any(data <= 0.0):
            raise ValueError("stored AMPA weights must be finite and positive")
        for index, pathway in enumerate(PATHWAYS):
            mask = _pathway_mask(rows, n_e, pathway)
            if not np.any(mask):
                continue
            selected_rows, selected_columns = rows[mask], columns[mask]
            local_rows = selected_rows if pathway == "E_to_E" else selected_rows - n_e
            logits = np.log(data[mask]) + logits_for(
                selected_rows, selected_columns, pathway, values[index],
            )
            np.maximum.at(target_max[pathway], local_rows, logits)
            edge_counts[pathway] += int(np.sum(mask))

    target_sum = {
        "E_to_E": np.zeros(n_e, float),
        "E_to_I": np.zeros(n_i, float),
    }
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        columns = np.asarray(coo.col, np.int64)
        data = np.asarray(coo.data, float)
        for index, pathway in enumerate(PATHWAYS):
            mask = _pathway_mask(rows, n_e, pathway)
            if not np.any(mask):
                continue
            selected_rows, selected_columns = rows[mask], columns[mask]
            local_rows = selected_rows if pathway == "E_to_E" else selected_rows - n_e
            logits = np.log(data[mask]) + logits_for(
                selected_rows, selected_columns, pathway, values[index],
            )
            target_sum[pathway] += np.bincount(
                local_rows,
                weights=np.exp(logits - target_max[pathway][local_rows]),
                minlength=len(target_sum[pathway]),
            )

    sample_strides = {
        pathway: max(1, int(np.ceil(edge_counts[pathway] / max(1, int(ratio_sample_limit)))))
        for pathway in PATHWAYS
    }
    offsets = {pathway: 0 for pathway in PATHWAYS}
    ratio_samples = {pathway: [] for pathway in PATHWAYS}
    new_bins = []
    for matrix in old_bins:
        coo = matrix.tocoo(copy=True)
        rows = np.asarray(coo.row, np.int64)
        columns = np.asarray(coo.col, np.int64)
        original = np.asarray(coo.data, float).copy()
        for index, pathway in enumerate(PATHWAYS):
            mask = _pathway_mask(rows, n_e, pathway)
            if not np.any(mask):
                continue
            selected_rows, selected_columns = rows[mask], columns[mask]
            local_rows = selected_rows if pathway == "E_to_E" else selected_rows - n_e
            logits = np.log(original[mask]) + logits_for(
                selected_rows, selected_columns, pathway, values[index],
            )
            transformed = old_incoming[pathway][local_rows] * np.exp(
                logits - target_max[pathway][local_rows],
            ) / target_sum[pathway][local_rows]
            if np.any(~np.isfinite(transformed)) or np.any(transformed <= 0.0):
                raise RuntimeError(f"{pathway} transform produced invalid weights")
            ratio = transformed / original[mask]
            local = np.flatnonzero(
                (offsets[pathway] + np.arange(len(ratio))) % sample_strides[pathway] == 0,
            )
            if len(local):
                ratio_samples[pathway].append(ratio[local])
            offsets[pathway] += len(ratio)
            coo.data[mask] = transformed
        new_bins.append(coo.tocsc())

    new_net = copy.copy(net)
    new_net["ampa_by_delay"] = new_bins
    removed = _invalidate_ampa_caches(new_net)
    topology_unchanged = _hash_sparse_bins(new_bins, include_data=False) == old_topology
    gaba_unchanged = _hash_sparse_bins(new_net["gaba_by_delay"]) == old_gaba
    pathway_audit = {}
    for pathway in PATHWAYS:
        current = _incoming_by_pathway(new_bins, n_e, n_i, pathway)
        error = np.abs(current - old_incoming[pathway])
        samples = np.concatenate(ratio_samples[pathway]) if ratio_samples[pathway] else np.ones(1)
        pathway_audit[pathway] = {
            "n_edges": edge_counts[pathway],
            "max_abs_incoming_error": float(np.max(error, initial=0.0)),
            "mean_abs_incoming_error": float(np.mean(error)),
            "edge_ratio": _summary(samples),
            "ratio_quantile_sample_size": int(len(samples)),
            "ratio_sample_stride": sample_strides[pathway],
        }
    if (not topology_unchanged or not gaba_unchanged
            or any(item["max_abs_incoming_error"] > 1e-9 for item in pathway_audit.values())):
        raise RuntimeError("local E-source transform violated the structural contract")
    combined_ratio_samples = np.concatenate([
        np.concatenate(ratio_samples[pathway])
        if ratio_samples[pathway] else np.ones(1)
        for pathway in PATHWAYS
    ])

    return new_net, {
        **audit,
        "exact_noop": False,
        "edge_ratio": _summary(combined_ratio_samples),
        "topology_unchanged": topology_unchanged,
        "delay_assignment_unchanged": topology_unchanged,
        "gaba_unchanged": gaba_unchanged,
        "ampa_data_unchanged": _hash_sparse_bins(new_bins) == old_data,
        "invalidated_ampa_cache_keys": removed,
        "pathway_audit": pathway_audit,
    }


__all__ = [
    "FEATURE_NAMES",
    "PATHWAYS",
    "continuous_local_e_source_flow",
    "local_pair_features",
]
