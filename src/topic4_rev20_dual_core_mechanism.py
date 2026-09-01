"""Pure contracts and fixed-topology geometry mapper for rev20-DC."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from src.topic4_core_connectivity import _hash_sparse_bins, _invalidate_ampa_caches


REFERENCE_ANGLE_DEG = 45.0
REFERENCE_ASPECT_RATIO = 2.0


def canonical_sha256(value: Mapping) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def dual_core_field_sha256(centers_mm, target_count: int) -> str:
    centers = np.asarray(centers_mm, float)
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        raise ValueError("dual-core centers must be finite with shape (2, 2)")
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return canonical_sha256({
        "field_type": "manual_dual_core_budget_matched",
        "centers_mm": centers[order].tolist(),
        "target_count": int(target_count),
    })


def build_one_factor_candidates(config: Mapping) -> list[dict]:
    """Expand the frozen atlas while emitting the reference exactly once."""
    reference = dict(config["reference"])
    anchor = dict(config["dual_core_anchor"])
    centers = anchor["centers_mm"]
    target_count = int(anchor["target_count"])

    def row(candidate_id: str, family: str, level, *, budget=target_count,
            node_gain=reference["node_gain"],
            depth=reference["signed_depth_shrinkage"],
            g_ee=reference["g_EE"], g_etoi=reference["g_EtoI"],
            angle=reference["ellipse_angle_deg"],
            aspect=reference["ellipse_aspect_ratio"], is_reference=False):
        field = {
            "field_type": "manual_dual_core_budget_matched",
            "centers_mm": centers,
            "target_count": int(budget),
        }
        field["field_sha256"] = dual_core_field_sha256(
            centers, int(budget),
        )
        return {
            "candidate_id": candidate_id,
            "family": family,
            "level": level,
            "is_reference": bool(is_reference),
            "selection_eligible": True,
            "node_field": field,
            "node_mapping": {
                "node_gain": float(node_gain),
                "signed_depth_shrinkage": float(depth),
            },
            "mechanisms": {
                "g_EE": float(g_ee),
                "g_EtoI": float(g_etoi),
                "ellipse_angle_deg": float(angle),
                "ellipse_aspect_ratio": float(aspect),
                "Z_M": "off",
            },
        }

    rows = [row(
        "dualcore_s39_reference", "reference", "reference", is_reference=True,
    )]
    families = config["parameter_families"]
    specifications = (
        ("node_gain", reference["node_gain"], lambda value: {
            "node_gain": value,
        }),
        ("core_budget_scale", 1.0, lambda value: {
            "budget": int(round(target_count * float(value))),
        }),
        ("signed_depth_shrinkage", reference["signed_depth_shrinkage"],
         lambda value: {"depth": value}),
        ("g_EE", reference["g_EE"], lambda value: {"g_ee": value}),
        ("g_EtoI", reference["g_EtoI"], lambda value: {"g_etoi": value}),
        ("both_scale", 1.0, lambda value: {
            "g_ee": float(reference["g_EE"]) * float(value),
            "g_etoi": float(reference["g_EtoI"]) * float(value),
        }),
        ("ellipse_angle_deg", reference["ellipse_angle_deg"],
         lambda value: {"angle": value}),
        ("ellipse_aspect_ratio", reference["ellipse_aspect_ratio"],
         lambda value: {"aspect": value}),
    )
    for family, reference_level, changes in specifications:
        for level in families[family]:
            if np.isclose(float(level), float(reference_level)):
                continue
            token = str(level).replace("-", "m").replace(".", "p")
            rows.append(row(
                f"dc_{family}_{token}", family, level, **changes(level),
            ))
    ids = [item["candidate_id"] for item in rows]
    if len(rows) != 31 or len(ids) != len(set(ids)):
        raise RuntimeError("rev20-DC atlas must contain 31 unique candidates")
    assert_one_factor_candidates(rows, reference)
    return rows


def assert_one_factor_candidates(candidates: list[Mapping], reference: Mapping) -> None:
    """Reject candidates that silently alter more than their named coordinate."""
    expected = {
        "node_gain": float(reference["node_gain"]),
        "core_budget_scale": 1.0,
        "signed_depth_shrinkage": float(reference["signed_depth_shrinkage"]),
        "g_EE": float(reference["g_EE"]),
        "g_EtoI": float(reference["g_EtoI"]),
        "ellipse_angle_deg": float(reference["ellipse_angle_deg"]),
        "ellipse_aspect_ratio": float(reference["ellipse_aspect_ratio"]),
    }
    reference_row = [row for row in candidates if row["is_reference"]]
    if len(reference_row) != 1:
        raise RuntimeError("atlas must contain exactly one reference")
    anchor_count = int(reference_row[0]["node_field"]["target_count"])
    for item in candidates:
        family = str(item["family"])
        if family == "reference":
            continue
        observed = {
            "node_gain": float(item["node_mapping"]["node_gain"]),
            "core_budget_scale": (
                int(item["node_field"]["target_count"]) / anchor_count
            ),
            "signed_depth_shrinkage": float(
                item["node_mapping"]["signed_depth_shrinkage"]
            ),
            "g_EE": float(item["mechanisms"]["g_EE"]),
            "g_EtoI": float(item["mechanisms"]["g_EtoI"]),
            "ellipse_angle_deg": float(item["mechanisms"]["ellipse_angle_deg"]),
            "ellipse_aspect_ratio": float(
                item["mechanisms"]["ellipse_aspect_ratio"]
            ),
        }
        allowed = {family}
        if family == "both_scale":
            allowed = {"g_EE", "g_EtoI"}
        changed = {
            key for key, value in observed.items()
            if not np.isclose(value, expected[key])
        }
        required = allowed if family == "both_scale" else {family}
        if changed - allowed or not required.issubset(changed):
            raise RuntimeError(
                f"candidate {item['candidate_id']} is not one-factor: {changed}"
            )


def _elliptical_radius(displacement, *, length_scale, angle_deg, aspect_ratio):
    values = np.asarray(displacement, float)
    scale = float(length_scale)
    aspect = float(aspect_ratio)
    angle = np.deg2rad(float(angle_deg))
    if scale <= 0.0 or aspect < 1.0 or not np.isfinite([scale, aspect, angle]).all():
        raise ValueError("ellipse parameters are invalid")
    parallel = scale * np.sqrt(aspect)
    perpendicular = scale / np.sqrt(aspect)
    c, s = np.cos(angle), np.sin(angle)
    u = c * values[:, 0] + s * values[:, 1]
    v = -s * values[:, 0] + c * values[:, 1]
    return np.sqrt((u / parallel) ** 2 + (v / perpendicular) ** 2)


def fixed_topology_ee_ellipse_redistribution(
    net,
    positions,
    *,
    length_scale,
    angle_deg=REFERENCE_ANGLE_DEG,
    aspect_ratio=REFERENCE_ASPECT_RATIO,
    reference_angle_deg=REFERENCE_ANGLE_DEG,
    reference_aspect_ratio=REFERENCE_ASPECT_RATIO,
):
    """Reweight existing E-to-E edges and conserve each target's total input."""
    n_e = int(net["NE"])
    positions = np.asarray(positions, float)
    if positions.shape[0] < n_e or positions.shape[1:] != (2,):
        raise ValueError("positions must cover all neurons with two coordinates")
    old_bins = net["ampa_by_delay"]
    old_topology = _hash_sparse_bins(old_bins, include_data=False)
    old_data = _hash_sparse_bins(old_bins)
    old_gaba = _hash_sparse_bins(net["gaba_by_delay"])
    is_reference = (
        float(angle_deg) == float(reference_angle_deg)
        and float(aspect_ratio) == float(reference_aspect_ratio)
    )
    if is_reference:
        return net, {
            "mechanism": "fixed_topology_EE_ellipse_redistribution_v1",
            "exact_noop": True,
            "angle_deg": float(angle_deg),
            "aspect_ratio": float(aspect_ratio),
            "reference_angle_deg": float(reference_angle_deg),
            "reference_aspect_ratio": float(reference_aspect_ratio),
            "topology_unchanged": True,
            "delay_assignment_unchanged": True,
            "gaba_unchanged": True,
            "ampa_data_unchanged": True,
            "maximum_abs_incoming_EE_error": 0.0,
            "edge_ratio": {"min": 1.0, "median": 1.0, "max": 1.0},
        }

    incoming = np.zeros(n_e, float)
    denominator = np.zeros(n_e, float)
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        mask = np.asarray(coo.row) < n_e
        rows = np.asarray(coo.row[mask], np.int64)
        cols = np.asarray(coo.col[mask], np.int64)
        data = np.asarray(coo.data[mask], float)
        displacement = positions[cols] - positions[rows]
        log_ratio = (
            -_elliptical_radius(
                displacement, length_scale=length_scale,
                angle_deg=angle_deg, aspect_ratio=aspect_ratio,
            )
            + _elliptical_radius(
                displacement, length_scale=length_scale,
                angle_deg=reference_angle_deg,
                aspect_ratio=reference_aspect_ratio,
            )
        )
        ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
        incoming += np.bincount(rows, weights=data, minlength=n_e)
        denominator += np.bincount(rows, weights=data * ratio, minlength=n_e)
    if np.any((incoming > 0.0) & (denominator <= 0.0)):
        raise RuntimeError("ellipse redistribution has a zero denominator")

    new_bins, ratio_samples = [], []
    for matrix in old_bins:
        coo = matrix.tocoo(copy=True)
        mask = np.asarray(coo.row) < n_e
        rows = np.asarray(coo.row[mask], np.int64)
        cols = np.asarray(coo.col[mask], np.int64)
        original = np.asarray(coo.data[mask], float).copy()
        displacement = positions[cols] - positions[rows]
        log_ratio = (
            -_elliptical_radius(
                displacement, length_scale=length_scale,
                angle_deg=angle_deg, aspect_ratio=aspect_ratio,
            )
            + _elliptical_radius(
                displacement, length_scale=length_scale,
                angle_deg=reference_angle_deg,
                aspect_ratio=reference_aspect_ratio,
            )
        )
        raw_ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
        transformed = original * raw_ratio * incoming[rows] / denominator[rows]
        coo.data[mask] = transformed
        if len(transformed):
            stride = max(1, len(transformed) // 100_000)
            ratio_samples.append(transformed[::stride] / original[::stride])
        new_bins.append(coo.tocsc())

    current = np.zeros(n_e, float)
    for matrix in new_bins:
        coo = matrix.tocoo(copy=False)
        mask = np.asarray(coo.row) < n_e
        current += np.bincount(
            np.asarray(coo.row[mask], np.int64),
            weights=np.asarray(coo.data[mask], float), minlength=n_e,
        )
    error = np.abs(current - incoming)
    new_net = copy.copy(net)
    new_net["ampa_by_delay"] = new_bins
    removed = _invalidate_ampa_caches(new_net)
    samples = np.concatenate(ratio_samples) if ratio_samples else np.ones(1)
    audit = {
        "mechanism": "fixed_topology_EE_ellipse_redistribution_v1",
        "exact_noop": False,
        "angle_deg": float(angle_deg),
        "aspect_ratio": float(aspect_ratio),
        "reference_angle_deg": float(reference_angle_deg),
        "reference_aspect_ratio": float(reference_aspect_ratio),
        "topology_unchanged": (
            _hash_sparse_bins(new_bins, include_data=False) == old_topology
        ),
        "delay_assignment_unchanged": (
            _hash_sparse_bins(new_bins, include_data=False) == old_topology
        ),
        "gaba_unchanged": _hash_sparse_bins(new_net["gaba_by_delay"]) == old_gaba,
        "ampa_data_unchanged": _hash_sparse_bins(new_bins) == old_data,
        "maximum_abs_incoming_EE_error": float(np.max(error, initial=0.0)),
        "mean_abs_incoming_EE_error": float(np.mean(error)),
        "edge_ratio": {
            "min": float(np.min(samples)),
            "median": float(np.median(samples)),
            "max": float(np.max(samples)),
        },
        "invalidated_ampa_cache_keys": removed,
    }
    if (not audit["topology_unchanged"] or not audit["gaba_unchanged"]
            or audit["maximum_abs_incoming_EE_error"] > 1e-9):
        raise RuntimeError("ellipse redistribution violated the structural contract")
    return new_net, audit
