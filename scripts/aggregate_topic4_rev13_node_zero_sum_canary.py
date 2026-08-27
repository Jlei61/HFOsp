#!/usr/bin/env python3
"""Aggregate rev13 Node-recovery canaries without opening patient targets.

The formal K=2 endpoint is one-dimensional signed causal displacement along
the frozen substrate axis, evaluated by contiguous-time held-out density.
Whole-sheet onset-map KMeans is retained only as a topology/figure diagnostic.
Contact readouts, patient labels, prototypes and classifiers are forbidden.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_node_dualmode import causal_root_displacement  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
EXPECTED_MANIFEST_STATUS = "REV13_NODE_ZERO_SUM_RECOVERY_CANARY_FROZEN"
EXPECTED_WORKER_STATUS = "REV13_NODE_ZERO_SUM_WORKER_COMPLETE"
FORBIDDEN_INPUT_TERMS = (
    "contact", "patient", "prototype", "classifier", "heldout",
)
ALLOWED_PATIENT_AUDIT_KEYS = frozenset({
    "patient_runtime_inputs_used",
    "patient_label_or_prototype_input_loaded",
    "patient_labels_or_prototypes_loaded",
})
MODEL_INTERNAL_ARRAY_ALLOWLIST = frozenset({
    "source_onset_maps_ms",
    "source_onset_evaluable",
    "event_returned",
    "event_t_on_ms",
    "event_t_off_ms",
    "event_trigger_t_on_ms",
    "event_fragment_count",
    "event_directed_root_id",
    "event_root_count",
    "positions_E",
    "delta_vtheta",
    "source_bin_mm",
})
WORKER_PAYLOAD_ALLOWLIST = frozenset({
    "status",
    "candidate_id",
    "seed",
    "mechanism_freeze",
    "node_accessibility",
    "arrays",
    "event_unit",
    "simulation",
})
MINORITY_FRACTION = 0.20
MIN_DIRECTION_CONSISTENCY = 0.70
MIN_TEMPORAL_SIGN_BLOCKS = 2
N_TEMPORAL_BLOCKS = 3
MIN_FORMAL_EVENTS = 24
PREFERRED_FORMAL_EVENTS = 30
MIN_EVENTS_PER_DIRECTION = 6
MINIMUM_YIELD_RATIO = 0.50
MAXIMUM_COMPOUND_DELTA = 0.15
WIDESPREAD_FRACTION = 0.50
ZERO_SUM_ATOL_MV = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _assert_patient_free_manifest(manifest: Mapping[str, Any]) -> None:
    for name, record in manifest.get("inputs", {}).items():
        path = record.get("path", "") if isinstance(record, Mapping) else ""
        lowered = f"{name} {path}".lower()
        if any(term in lowered for term in FORBIDDEN_INPUT_TERMS):
            raise RuntimeError(f"rev13 aggregator forbids patient input {name!r}")
    for candidate in manifest.get("candidates", []):
        for key in candidate:
            if any(term in str(key).lower() for term in FORBIDDEN_INPUT_TERMS):
                raise RuntimeError("rev13 candidate embeds a patient-side field")


def _project_worker_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only model-internal worker fields required by this aggregator."""
    projected = {
        key: payload[key] for key in WORKER_PAYLOAD_ALLOWLIST if key in payload
    }
    node = projected.get("node_accessibility")
    if isinstance(node, Mapping):
        projected["node_accessibility"] = {
            key: node[key] for key in (
                "mode", "diagnostics", "patient_runtime_inputs_used",
            ) if key in node
        }
    mechanism = projected.get("mechanism_freeze")
    if isinstance(mechanism, Mapping):
        projected["mechanism_freeze"] = {
            key: mechanism[key] for key in ("EE", "E_to_I", "Z_M")
            if key in mechanism
        }
    arrays = projected.get("arrays")
    if isinstance(arrays, Mapping):
        projected["arrays"] = {
            key: arrays[key] for key in ("path", "sha256") if key in arrays
        }
    return projected


def _load_model_internal_arrays(arrays_path: Path) -> dict[str, np.ndarray]:
    """Read an explicit allowlist; contact and patient arrays are never indexed."""
    with np.load(arrays_path, allow_pickle=False) as loaded:
        return {
            name: loaded[name]
            for name in MODEL_INTERNAL_ARRAY_ALLOWLIST
            if name in loaded.files
        }


def _event_map_features(onset_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = np.asarray(onset_maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError("onset maps must have shape (event, y, x)")
    valid = np.sum(np.isfinite(maps), axis=(1, 2)) >= 4
    selected = maps[valid]
    if not len(selected):
        return np.empty((0, 2 * maps.shape[1] * maps.shape[2])), valid
    rows = []
    for onset in selected:
        finite = np.isfinite(onset)
        normalized = np.zeros(onset.shape, dtype=np.float64)
        values = onset[finite]
        span = float(np.ptp(values))
        if span > 0.0:
            normalized[finite] = (values - float(np.min(values))) / span
        rows.append(np.concatenate([finite.ravel(), normalized.ravel()]))
    return np.asarray(rows, dtype=np.float64), valid


def _equal_time_folds(event_t_on_ms: np.ndarray, *,
                      n_folds: int = N_TEMPORAL_BLOCKS) -> list[np.ndarray]:
    """Split chronological events by equal-duration, not equal-count, blocks."""
    times = np.asarray(event_t_on_ms, dtype=np.float64)
    if times.ndim != 1 or len(times) < n_folds or not np.all(np.isfinite(times)):
        raise ValueError("equal-time folds require finite one-dimensional times")
    if np.any(np.diff(times) < 0.0) or float(times[-1]) <= float(times[0]):
        raise ValueError("equal-time folds require increasing temporal support")
    edges = np.linspace(float(times[0]), float(times[-1]), int(n_folds) + 1)
    block_id = np.searchsorted(edges[1:-1], times, side="right")
    return [
        np.flatnonzero(block_id == block).astype(np.int64)
        for block in range(int(n_folds))
    ]


def _fit_directional_gmm(values: np.ndarray, *, n_components: int,
                         seed: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=int(n_components),
        covariance_type="full",
        reg_covar=1e-4,
        n_init=20,
        random_state=int(seed),
    ).fit(np.asarray(values, dtype=np.float64).reshape(-1, 1))


def directional_k2_formal(displacements: np.ndarray,
                          event_t_on_ms: np.ndarray, *, seed: int
                          ) -> dict[str, Any]:
    """Evaluate signed displacement K=2 without shuffled time folds.

    The input must already be chronological and event-count matched to every
    paired arm.  Each equal-duration temporal third is held out once; the full-series K=2
    labels are used only for the frozen direction/occupancy diagnostics.
    """
    values = np.asarray(displacements, dtype=np.float64)
    times = np.asarray(event_t_on_ms, dtype=np.float64)
    if values.ndim != 1 or times.shape != values.shape:
        raise ValueError("signed displacement and event times must align")
    finite = np.isfinite(values) & np.isfinite(times)
    values = values[finite]
    times = times[finite]
    order = np.argsort(times, kind="stable")
    values = values[order]
    times = times[order]
    direction_counts = {
        "negative": int(np.sum(values < 0.0)),
        "positive": int(np.sum(values > 0.0)),
    }
    evaluability_reasons = []
    if len(values) < MIN_FORMAL_EVENTS:
        evaluability_reasons.append("fewer_than_24_isolated_families")
    if min(direction_counts.values()) < MIN_EVENTS_PER_DIRECTION:
        evaluability_reasons.append("fewer_than_6_families_in_one_direction")
    try:
        folds = _equal_time_folds(times)
    except ValueError:
        folds = []
        evaluability_reasons.append("equal_time_blocks_not_constructible")
    if folds and any(len(block) == 0 for block in folds):
        evaluability_reasons.append("one_or_more_equal_time_blocks_empty")
    if evaluability_reasons:
        return {
            "status": "NOT_EVALUABLE",
            "n_events": int(len(values)),
            "events_per_direction": direction_counts,
            "preferred_event_count_reached": bool(
                len(values) >= PREFERRED_FORMAL_EVENTS
            ),
            "evaluability_reasons": evaluability_reasons,
        }

    heldout_deltas = []
    for fold_index, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(values)), test, assume_unique=True)
        fold_scores = []
        for n_components in (1, 2):
            model = _fit_directional_gmm(
                values[train], n_components=n_components,
                seed=int(seed) + 101 * fold_index + n_components,
            )
            fold_scores.append(float(model.score(values[test, None])))
        heldout_deltas.append(fold_scores[1] - fold_scores[0])

    full_model = _fit_directional_gmm(values, n_components=2, seed=int(seed))
    labels = full_model.predict(values[:, None]).astype(np.int64)
    labels, medians = canonicalize_cluster_labels(labels, values)
    counts = np.bincount(labels, minlength=2)
    consistency = []
    for cluster, median in enumerate(medians):
        selected = values[labels == cluster]
        direction = -1.0 if median < 0.0 else 1.0
        consistency.append(float(np.mean(direction * selected > 0.0)))

    sign_support = {"negative": 0, "positive": 0}
    block_counts = []
    for block in folds:
        block_values = values[block]
        negative = int(np.sum(block_values < 0.0))
        positive = int(np.sum(block_values > 0.0))
        sign_support["negative"] += int(negative > 0)
        sign_support["positive"] += int(positive > 0)
        block_counts.append({"negative": negative, "positive": positive})

    return {
        "status": "OK",
        "n_events": int(len(values)),
        "events_per_direction": direction_counts,
        "preferred_event_count_reached": bool(
            len(values) >= PREFERRED_FORMAL_EVENTS
        ),
        "evaluability_reasons": [],
        "temporal_block_definition": "three_equal_duration_blocks",
        "temporal_block_time_edges_ms": [
            float(times[0]),
            *[
                float(times[0] + (times[-1] - times[0]) * block / N_TEMPORAL_BLOCKS)
                for block in range(1, N_TEMPORAL_BLOCKS)
            ],
            float(times[-1]),
        ],
        "heldout_k2_minus_k1_loglik_per_event": float(np.mean(heldout_deltas)),
        "heldout_fold_deltas": heldout_deltas,
        "cluster_labels": labels,
        "cluster_counts": counts.tolist(),
        "minority_fraction": float(np.min(counts) / np.sum(counts)),
        "cluster_median_axis_displacement_mm": medians,
        "same_network_opposite_directions": bool(medians[0] < 0.0 < medians[1]),
        "within_cluster_direction_consistency": consistency,
        "minimum_within_cluster_direction_consistency": float(min(consistency)),
        "temporal_block_sign_counts": block_counts,
        "temporal_blocks_with_each_sign": sign_support,
        "both_signs_in_at_least_two_of_three_blocks": bool(
            sign_support["negative"] >= MIN_TEMPORAL_SIGN_BLOCKS
            and sign_support["positive"] >= MIN_TEMPORAL_SIGN_BLOCKS
        ),
    }


def substrate_pca_axis(positions_e: np.ndarray,
                       static_modulation: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_e, dtype=np.float64)
    modulation = np.asarray(static_modulation, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 \
            or modulation.shape != (len(positions),):
        raise ValueError("substrate PCA inputs do not align")
    weights = np.abs(modulation)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(weights)):
        raise ValueError("substrate PCA inputs must be finite")
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones(len(positions), dtype=np.float64)
    weights = weights / float(np.sum(weights))
    center = np.sum(positions * weights[:, None], axis=0)
    centered = positions - center
    covariance = (centered * weights[:, None]).T @ centered
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    first_nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if not len(first_nonzero):
        raise RuntimeError("substrate PCA axis is degenerate")
    if axis[first_nonzero[0]] < 0.0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def event_axis_displacements(onset_maps: np.ndarray, *, axis_unit: np.ndarray,
                             bin_mm: float) -> np.ndarray:
    axis = np.asarray(axis_unit, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    output = []
    for onset in np.asarray(onset_maps, dtype=np.float64):
        record = causal_root_displacement(onset, bin_mm=float(bin_mm))
        if not record["evaluable"]:
            output.append(np.nan)
            continue
        output.append(float(np.dot(record["displacement_xy_mm"], axis)))
    return np.asarray(output, dtype=np.float64)


def overlap_connected_episode_audit(
        event_t_on_ms: np.ndarray, event_t_off_ms: np.ndarray,
        displacements: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify every family belonging to a temporally overlapping episode.

    A strict interval overlap joins two families.  Connectivity is transitive:
    if A overlaps B and B overlaps C, all three are excluded from the formal
    K=2 endpoint even when A and C do not directly overlap.
    """
    t_on = np.asarray(event_t_on_ms, dtype=np.float64)
    t_off = np.asarray(event_t_off_ms, dtype=np.float64)
    displacement = np.asarray(displacements, dtype=np.float64)
    if t_on.ndim != 1 or t_off.shape != t_on.shape \
            or displacement.shape != t_on.shape:
        raise ValueError("overlap audit arrays must align")
    if not np.all(np.isfinite(t_on)) or not np.all(np.isfinite(t_off)) \
            or np.any(t_off < t_on):
        raise ValueError("overlap audit requires finite ordered intervals")
    if np.any(np.diff(t_on) < 0.0):
        raise ValueError("overlap audit requires chronological families")

    pairs = []
    pair_direction_counts = {"same": 0, "opposite": 0, "zero_involved": 0}
    for left in range(len(t_on)):
        right = left + 1
        while right < len(t_on) and t_on[right] < t_off[left]:
            left_sign = int(np.sign(displacement[left]))
            right_sign = int(np.sign(displacement[right]))
            relation = (
                "zero_involved" if left_sign == 0 or right_sign == 0
                else "same" if left_sign == right_sign else "opposite"
            )
            pair_direction_counts[relation] += 1
            pairs.append({
                "family_indices": [int(left), int(right)],
                "directions": [left_sign, right_sign],
                "direction_relation": relation,
            })
            right += 1

    components: list[list[int]] = []
    if len(t_on):
        current = [0]
        current_right = float(t_off[0])
        for index in range(1, len(t_on)):
            if float(t_on[index]) < current_right:
                current.append(index)
                current_right = max(current_right, float(t_off[index]))
            else:
                components.append(current)
                current = [index]
                current_right = float(t_off[index])
        components.append(current)

    overlap_components = []
    excluded = np.zeros(len(t_on), dtype=bool)
    for component in components:
        if len(component) < 2:
            continue
        excluded[component] = True
        signs = np.sign(displacement[component]).astype(np.int64)
        overlap_components.append({
            "family_indices": [int(value) for value in component],
            "n_families": int(len(component)),
            "negative": int(np.sum(signs < 0)),
            "positive": int(np.sum(signs > 0)),
            "zero": int(np.sum(signs == 0)),
            "contains_opposite_directions": bool(
                np.any(signs < 0) and np.any(signs > 0)
            ),
        })

    return ~excluded, {
        "formal_action": "EXCLUDE_ALL_MEMBERS_OF_OVERLAP_CONNECTED_EPISODES",
        "strict_overlap_definition": "later_t_on_ms < earlier_t_off_ms",
        "n_input_families": int(len(t_on)),
        "n_overlap_pairs": int(len(pairs)),
        "n_overlap_components": int(len(overlap_components)),
        "n_excluded_families": int(np.sum(excluded)),
        "n_isolated_families": int(np.sum(~excluded)),
        "overlap_pair_direction_counts": pair_direction_counts,
        "overlap_pairs": pairs,
        "overlap_components": overlap_components,
    }


def canonicalize_cluster_labels(labels: np.ndarray,
                                displacements: np.ndarray) -> tuple[np.ndarray, list[float]]:
    labels = np.asarray(labels, dtype=np.int64)
    displacement = np.asarray(displacements, dtype=np.float64)
    if labels.shape != displacement.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("two cluster labels and displacements are required")
    medians = [
        float(np.nanmedian(displacement[labels == cluster]))
        for cluster in (0, 1)
    ]
    if not np.all(np.isfinite(medians)):
        raise RuntimeError("both clusters require evaluable causal displacement")
    if medians[0] > medians[1]:
        return 1 - labels, [medians[1], medians[0]]
    return labels.copy(), medians


def whole_sheet_onset_map_kmeans_diagnostic(
        onset_maps: np.ndarray, *, axis_unit: np.ndarray,
        bin_mm: float, seed: int) -> dict[str, Any]:
    features, valid = _event_map_features(onset_maps)
    if len(features) < 8:
        return {
            "status": "INSUFFICIENT_CAUSAL_FAMILIES",
            "n_events": int(len(features)),
        }
    label_sets = [
        KMeans(n_clusters=2, n_init=50, random_state=int(seed) + offset)
        .fit_predict(features)
        for offset in range(8)
    ]
    displacement = event_axis_displacements(
        np.asarray(onset_maps)[valid], axis_unit=axis_unit, bin_mm=bin_mm,
    )
    labels, medians = canonicalize_cluster_labels(label_sets[0], displacement)
    counts = np.bincount(labels, minlength=2)
    stability = [
        adjusted_mutual_info_score(label_sets[0], other)
        for other in label_sets[1:]
    ]
    return {
        "status": "OK",
        "n_events": int(len(labels)),
        "valid_event_mask": valid,
        "cluster_labels": labels,
        "cluster_counts": counts.tolist(),
        "minority_fraction": float(np.min(counts) / np.sum(counts)),
        "cluster_median_axis_displacement_mm": medians,
        "same_network_opposite_directions": bool(medians[0] < 0.0 < medians[1]),
        "kmeans_seed_ami_median": float(np.median(stability)),
        "silhouette": float(silhouette_score(features, labels)),
        "formal_acceptance_role": "TOPOLOGY_VISUALIZATION_DIAGNOSTIC_ONLY",
    }


def _run_lengths(labels: np.ndarray) -> list[int]:
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels):
        return []
    boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
    return np.diff(np.r_[0, boundaries, len(labels)]).astype(int).tolist()


def occupancy_matched_transition_null(labels: np.ndarray, *, draws: int = 4096,
                                      seed: int = 20260827) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 4 or set(np.unique(labels)) != {0, 1}:
        return {"status": "NOT_EVALUABLE", "anti_persistent": None}
    observed = float(np.mean(labels[1:] == labels[:-1]))
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        shuffled = rng.permutation(labels)
        null[index] = np.mean(shuffled[1:] == shuffled[:-1])
    q05 = float(np.quantile(null, 0.05))
    transitions = np.zeros((2, 2), dtype=np.int64)
    np.add.at(transitions, (labels[:-1], labels[1:]), 1)
    return {
        "status": "OK",
        "observed_same_mode_probability": observed,
        "occupancy_matched_null_q05": q05,
        "occupancy_matched_null_median": float(np.median(null)),
        "lower_tail_p": float((1 + np.sum(null <= observed)) / (len(null) + 1)),
        "anti_persistent": bool(observed < q05),
        "transition_counts": transitions.tolist(),
        "run_lengths": _run_lengths(labels),
        "draws": int(draws),
    }


def fragmentation_audit(labels: np.ndarray, root_ids: np.ndarray,
                        fragment_counts: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    root_ids = np.asarray(root_ids, dtype=np.int64)
    fragment_counts = np.asarray(fragment_counts, dtype=np.int64)
    if labels.shape != root_ids.shape or labels.shape != fragment_counts.shape:
        raise ValueError("fragmentation arrays do not align")
    repeated_cross_cluster = []
    for root_id in np.unique(root_ids[root_ids >= 0]):
        selected = root_ids == root_id
        if np.sum(selected) > 1 and len(np.unique(labels[selected])) > 1:
            repeated_cross_cluster.append(int(root_id))
    return {
        "formal_acceptance_role": "DIAGNOSTIC_ONLY",
        "n_families": int(len(labels)),
        "n_unique_directed_roots": int(len(np.unique(root_ids))),
        "mean_detector_fragments_per_family": (
            float(np.mean(fragment_counts)) if len(fragment_counts) else 0.0
        ),
        "cross_cluster_duplicated_root_ids": repeated_cross_cluster,
        "repeated_root_crosses_k2_clusters": bool(repeated_cross_cluster),
    }


def controller_diagnostic(payload: Mapping[str, Any]) -> dict[str, Any]:
    node = payload.get("node_accessibility", {})
    mode = str(node.get("mode", "exact_off"))
    diagnostics = node.get("diagnostics", {}) or {}
    zero_sum = diagnostics.get("maximum_zero_sum_error_mV")
    saturation = diagnostics.get("maximum_support_weighted_saturation_fraction")
    sign_flip = diagnostics.get(
        "maximum_support_weighted_static_modulation_sign_flip_fraction"
    )
    requires_zero_sum = mode in {"zero_sum", "spatial_shift"}
    zero_sum_ok = (
        not requires_zero_sum
        or (zero_sum is not None and float(zero_sum) <= ZERO_SUM_ATOL_MV)
    )
    return {
        "mode": mode,
        "zero_sum_required": requires_zero_sum,
        "zero_sum_ok": bool(zero_sum_ok),
        "maximum_zero_sum_error_mV": zero_sum,
        "maximum_support_weighted_saturation_fraction": saturation,
        "maximum_support_weighted_static_modulation_sign_flip_fraction": sign_flip,
        "widespread_saturation": bool(
            saturation is not None and float(saturation) >= WIDESPREAD_FRACTION
        ),
        "widespread_static_sign_flip": bool(
            sign_flip is not None and float(sign_flip) >= WIDESPREAD_FRACTION
        ),
    }


def _load_worker(json_path: Path, *, expected_candidate_ids: set[str]) -> dict[str, Any]:
    raw_payload = json.loads(json_path.read_text())
    payload = _project_worker_payload(raw_payload)
    if payload.get("status") != EXPECTED_WORKER_STATUS:
        raise RuntimeError(f"incomplete rev13 worker: {json_path}")
    candidate_id = str(payload.get("candidate_id"))
    if candidate_id not in expected_candidate_ids:
        raise RuntimeError("worker candidate is absent from frozen manifest")
    mechanism = payload.get("mechanism_freeze", {})
    if mechanism.get("EE") != "off" or mechanism.get("E_to_I") != "off" \
            or mechanism.get("Z_M") != "off":
        raise RuntimeError("rev13 model-internal aggregate received an open pathway")
    node = payload.get("node_accessibility", {})
    if node.get("patient_runtime_inputs_used") is not False:
        raise RuntimeError("worker did not prove patient-free execution")
    arrays_path = Path(payload["arrays"]["path"])
    if not arrays_path.is_absolute():
        arrays_path = ARTIFACT_ROOT / arrays_path
    if _sha256(arrays_path) != payload["arrays"]["sha256"]:
        raise RuntimeError("worker NPZ hash changed")
    arrays = _load_model_internal_arrays(arrays_path)
    required = set(MODEL_INTERNAL_ARRAY_ALLOWLIST)
    missing = required.difference(arrays)
    if missing:
        raise RuntimeError("worker lacks model-internal arrays: " + ", ".join(sorted(missing)))
    return {"payload": payload, "arrays": arrays, "json_path": json_path}


def per_run_record(worker: Mapping[str, Any], *, seed_offset: int = 0) -> dict[str, Any]:
    payload, arrays = worker["payload"], worker["arrays"]
    seed = int(payload["seed"])
    order = np.argsort(np.asarray(arrays["event_t_on_ms"], dtype=np.float64))
    usable = (
        np.asarray(arrays["event_returned"], dtype=bool)
        & np.asarray(arrays["source_onset_evaluable"], dtype=bool)
    )
    selected = order[usable[order]]
    maps = np.asarray(arrays["source_onset_maps_ms"], dtype=np.float64)[selected]
    t_on = np.asarray(arrays["event_t_on_ms"], dtype=np.float64)[selected]
    t_off = np.asarray(arrays["event_t_off_ms"], dtype=np.float64)[selected]
    trigger_t_on = np.asarray(
        arrays["event_trigger_t_on_ms"], dtype=np.float64,
    )[selected]
    axis = substrate_pca_axis(arrays["positions_E"], arrays["delta_vtheta"])
    bin_mm = float(np.asarray(arrays["source_bin_mm"]).item())
    displacements = event_axis_displacements(
        maps, axis_unit=axis, bin_mm=bin_mm,
    )
    directional_valid = (
        np.isfinite(displacements)
        & np.isfinite(t_on)
        & np.isfinite(t_off)
        & np.isfinite(trigger_t_on)
        & (t_off >= t_on)
    )
    maps = maps[directional_valid]
    displacements = displacements[directional_valid]
    t_on = t_on[directional_valid]
    t_off = t_off[directional_valid]
    trigger_t_on = trigger_t_on[directional_valid]
    root_ids = np.asarray(arrays["event_directed_root_id"])[selected][directional_valid]
    root_counts = np.asarray(arrays["event_root_count"])[selected][directional_valid]
    fragment_counts = np.asarray(arrays["event_fragment_count"])[selected][directional_valid]
    isolated, overlap = overlap_connected_episode_audit(
        t_on, t_off, displacements,
    )
    isolated_maps = maps[isolated]
    isolated_displacements = displacements[isolated]
    isolated_t_on = t_on[isolated]
    topology_diagnostic = whole_sheet_onset_map_kmeans_diagnostic(
        isolated_maps,
        axis_unit=axis,
        bin_mm=bin_mm,
        seed=20260827 + seed + int(seed_offset),
    )
    event_unit = payload.get("event_unit", {})
    compound = event_unit.get("compound_detector_fragment_fraction")
    runaway_time = payload.get("simulation", {}).get("runaway_early_stop_ms")
    controller = controller_diagnostic(payload)
    compact_topology = {
        key: value for key, value in topology_diagnostic.items()
        if key not in {"valid_event_mask", "cluster_labels"}
    }
    return {
        "candidate_id": str(payload["candidate_id"]),
        "seed": seed,
        "worker_json": str(worker["json_path"]),
        "worker_json_sha256": _sha256(worker["json_path"]),
        "worker_npz": str(payload["arrays"]["path"]),
        "worker_npz_sha256": str(payload["arrays"]["sha256"]),
        "n_returned_evaluable_causal_families": int(len(displacements)),
        "n_isolated_returned_evaluable_causal_families": int(
            len(isolated_displacements)
        ),
        "preferred_formal_event_count_reached": bool(
            len(isolated_displacements) >= PREFERRED_FORMAL_EVENTS
        ),
        "raw_detector_fragment_count": int(event_unit.get("raw_detector_fragment_count", 0)),
        "compound_detector_fragment_fraction": (
            None if compound is None else float(compound)
        ),
        "multi_root_event_fraction": event_unit.get("multi_root_event_fraction"),
        "runaway": bool(runaway_time is not None),
        "runaway_early_stop_ms": runaway_time,
        "substrate_pca_axis_xy": axis.tolist(),
        "whole_sheet_onset_map_kmeans_diagnostic": compact_topology,
        "overlap_connected_episode_audit": overlap,
        "causal_family_identity_diagnostic": {
            "formal_acceptance_role": "DIAGNOSTIC_ONLY",
            "all_root_counts_equal_one": bool(
                len(root_counts) and np.all(root_counts == 1)
            ),
            "median_family_onset_lead_vs_detector_trigger_ms": (
                float(np.median(trigger_t_on - t_on)) if len(t_on) else None
            ),
        },
        "_directional_displacements_mm": isolated_displacements,
        "_event_t_on_ms": isolated_t_on,
        "_directed_root_ids": root_ids[isolated],
        "_fragment_counts": fragment_counts[isolated],
        "controller": controller,
        "patient_labels_or_prototypes_loaded": False,
    }


def _finite_or(value: Any, default: float = -np.inf) -> float:
    return default if value is None or not np.isfinite(value) else float(value)


def _coefficient_suffix(candidate_id: str) -> str | None:
    marker = str(candidate_id).rfind("_c")
    return None if marker < 0 else str(candidate_id)[marker:]


def _matched_event_indices(n_available: int, n_matched: int) -> np.ndarray:
    if n_matched < 1 or n_available < n_matched:
        raise ValueError("invalid event-count match")
    if n_available == n_matched:
        return np.arange(n_available, dtype=np.int64)
    return np.rint(np.linspace(0, n_available - 1, n_matched)).astype(np.int64)


def _matched_directional_record(record: Mapping[str, Any], *, n_events: int,
                                seed: int) -> tuple[dict[str, Any], np.ndarray]:
    displacement = np.asarray(record.get("_directional_displacements_mm", []),
                              dtype=np.float64)
    event_t_on = np.asarray(record.get("_event_t_on_ms", []), dtype=np.float64)
    if event_t_on.shape != displacement.shape:
        raise ValueError("directional displacement and event times do not align")
    indices = _matched_event_indices(len(displacement), n_events)
    formal = directional_k2_formal(
        displacement[indices], event_t_on[indices], seed=int(seed),
    )
    return formal, indices


def paired_record(
        active: Mapping[str, Any], off: Mapping[str, Any], *,
        matched_controls: Iterable[Mapping[str, Any]] = (),
        expected_matched_control_ids: Iterable[str] = ()) -> dict[str, Any]:
    if int(active["seed"]) != int(off["seed"]):
        raise ValueError("paired runs must share a network seed")
    controls = list(matched_controls)
    if any(int(row["seed"]) != int(active["seed"]) for row in controls):
        raise ValueError("matched controls must share the active network seed")
    expected_ids = sorted(str(value) for value in expected_matched_control_ids)
    control_by_id = {str(row["candidate_id"]): row for row in controls}
    available_ids = sorted(control_by_id)
    controls_complete = bool(expected_ids) and available_ids == expected_ids

    matched_records = [active, off] + [control_by_id[key] for key in expected_ids
                                      if key in control_by_id]
    available_counts = [
        len(np.asarray(row.get("_directional_displacements_mm", [])))
        for row in matched_records
    ]
    n_matched = min(available_counts) if available_counts else 0
    formal_seed = 20260827 + int(active["seed"])
    formal_records: dict[str, dict[str, Any]] = {}
    matched_indices: dict[str, np.ndarray] = {}
    if n_matched >= 1:
        for row in matched_records:
            candidate_id = str(row["candidate_id"])
            formal, indices = _matched_directional_record(
                row, n_events=n_matched, seed=formal_seed,
            )
            formal_records[candidate_id] = formal
            matched_indices[candidate_id] = indices

    active_formal = formal_records.get(str(active["candidate_id"]), {})
    off_formal = formal_records.get(str(off["candidate_id"]), {})
    active_delta_value = active_formal.get("heldout_k2_minus_k1_loglik_per_event")
    off_delta_value = off_formal.get("heldout_k2_minus_k1_loglik_per_event")
    density_evaluable = bool(
        active_formal.get("status") == "OK" and off_formal.get("status") == "OK"
        and active_delta_value is not None and off_delta_value is not None
        and np.isfinite(active_delta_value) and np.isfinite(off_delta_value)
    )
    active_delta = _finite_or(active_delta_value)
    off_delta = _finite_or(off_delta_value)
    active_yield = int(active.get(
        "n_isolated_returned_evaluable_causal_families",
        len(np.asarray(active.get("_directional_displacements_mm", []))),
    ))
    off_yield = int(off.get(
        "n_isolated_returned_evaluable_causal_families",
        len(np.asarray(off.get("_directional_displacements_mm", []))),
    ))
    yield_ratio = active_yield / max(1, off_yield)
    compound_delta = (
        _finite_or(active.get("compound_detector_fragment_fraction"), 0.0)
        - _finite_or(off.get("compound_detector_fragment_fraction"), 0.0)
    )
    control_deltas = {
        candidate_id: formal_records.get(candidate_id, {}).get(
            "heldout_k2_minus_k1_loglik_per_event"
        )
        for candidate_id in expected_ids
    }
    controls_evaluable = bool(
        controls_complete and expected_ids and all(
            formal_records.get(candidate_id, {}).get("status") == "OK"
            and value is not None and np.isfinite(value)
            for candidate_id, value in control_deltas.items()
        )
    )

    transition = {"status": "NOT_EVALUABLE", "anti_persistent": None}
    fragmentation = {
        "formal_acceptance_role": "DIAGNOSTIC_ONLY",
        "repeated_root_crosses_k2_clusters": None,
        "cross_cluster_duplicated_root_ids": [],
    }
    if active_formal.get("status") == "OK":
        labels = np.asarray(active_formal["cluster_labels"], dtype=np.int64)
        transition = occupancy_matched_transition_null(
            labels, seed=formal_seed,
        )
        active_indices = matched_indices[str(active["candidate_id"])]
        fragmentation = fragmentation_audit(
            labels,
            np.asarray(active.get("_directed_root_ids", []))[active_indices],
            np.asarray(active.get("_fragment_counts", []))[active_indices],
        )

    reasons = []
    checks = {
        "heldout_k2_minus_k1_positive": bool(
            density_evaluable and active_delta > 0.0
        ),
        "directional_k2_above_paired_off": bool(
            density_evaluable and active_delta > off_delta
        ),
        "matched_controls_available": bool(controls_complete),
        "directional_k2_above_same_coefficient_controls": bool(
            density_evaluable and controls_evaluable
            and all(active_delta > float(value) for value in control_deltas.values())
        ),
        "same_network_opposite_directions": bool(
            active_formal.get("same_network_opposite_directions", False)
        ),
        "within_cluster_direction_consistency_at_least_70_percent": bool(
            _finite_or(
                active_formal.get("minimum_within_cluster_direction_consistency"),
                0.0,
            ) >= MIN_DIRECTION_CONSISTENCY
        ),
        "minority_at_least_20_percent": bool(
            _finite_or(active_formal.get("minority_fraction"), 0.0)
            >= MINORITY_FRACTION
        ),
        "both_signs_in_at_least_two_of_three_temporal_blocks": bool(
            active_formal.get(
                "both_signs_in_at_least_two_of_three_blocks", False,
            )
        ),
        "yield_at_least_half_paired_off": bool(yield_ratio >= MINIMUM_YIELD_RATIO),
        "compound_delta_within_limit": bool(compound_delta <= MAXIMUM_COMPOUND_DELTA),
        "not_runaway": not bool(active["runaway"]),
        "not_forced_alternation": not bool(
            transition.get("anti_persistent", True)
        ),
        "controller_zero_sum_valid": bool(active["controller"].get("zero_sum_ok", False)),
        "not_widespread_saturation": not bool(
            active["controller"].get("widespread_saturation", True)
        ),
        "not_widespread_static_sign_flip": not bool(
            active["controller"].get("widespread_static_sign_flip", True)
        ),
    }
    matched_control_status = (
        "MATCHED_CONTROLS_COMPLETE"
        if controls_complete else "MATCHED_CONTROL_EXTENSION_REQUIRED"
    )
    compact_active_formal = {
        key: value for key, value in active_formal.items()
        if key != "cluster_labels"
    }
    compact_control_formal = {
        candidate_id: {
            key: value for key, value in formal_records.get(candidate_id, {}).items()
            if key != "cluster_labels"
        }
        for candidate_id in expected_ids if candidate_id in formal_records
    }
    base_formal_evaluable = bool(
        active_formal.get("status") == "OK" and off_formal.get("status") == "OK"
    )
    comparison_evaluable = bool(
        base_formal_evaluable and controls_complete and controls_evaluable
    )
    model_internal_pass = bool(comparison_evaluable and all(checks.values()))
    if comparison_evaluable:
        reasons = [name for name, passed in checks.items() if not passed]
    not_evaluable_reasons = {}
    for candidate_id, record in formal_records.items():
        if record.get("status") != "OK":
            not_evaluable_reasons[candidate_id] = record.get(
                "evaluability_reasons", [record.get("status")]
            )
    if not base_formal_evaluable or (controls_complete and not controls_evaluable):
        status = "NOT_EVALUABLE"
    elif not controls_complete:
        status = matched_control_status
    elif model_internal_pass:
        status = "MODEL_INTERNAL_NETWORK_PASS"
    else:
        status = "MODEL_INTERNAL_NETWORK_FAIL"
    return {
        "candidate_id": str(active["candidate_id"]),
        "seed": int(active["seed"]),
        "status": status,
        "paired_off_candidate_id": str(off["candidate_id"]),
        "matched_event_count_per_arm": int(n_matched),
        "matched_control_candidate_ids": available_ids,
        "expected_matched_control_candidate_ids": expected_ids,
        "matched_control_status": matched_control_status,
        "formal_comparison_evaluable": comparison_evaluable,
        "not_evaluable_reasons": not_evaluable_reasons,
        "directional_k2_formal": compact_active_formal,
        "matched_control_directional_k2_formal": compact_control_formal,
        "directional_k2_delta_vs_off": (
            float(active_delta - off_delta) if density_evaluable else None
        ),
        "returned_event_yield_ratio_vs_off": float(yield_ratio),
        "compound_fraction_delta_vs_off": float(compound_delta),
        "directional_k2_margins_vs_matched_controls": {
            candidate_id: (
                None if value is None or not np.isfinite(value)
                else float(active_delta - float(value))
            )
            for candidate_id, value in control_deltas.items()
        },
        "transition": transition,
        "fragmentation": fragmentation,
        "checks": checks,
        "model_internal_network_pass": model_internal_pass,
        "failure_reasons": reasons,
    }


def build_paired_rows(
        per_run: Iterable[Mapping[str, Any]], *,
        expected_candidate_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    rows = list(per_run)
    by_key = {(str(row["candidate_id"]), int(row["seed"])): row for row in rows}
    all_candidate_ids = set(
        str(value) for value in (
            expected_candidate_ids
            if expected_candidate_ids is not None
            else {row["candidate_id"] for row in rows}
        )
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    output = []
    for seed in seeds:
        off = by_key.get(("exact_off", seed))
        if off is None:
            continue
        for candidate_id in ("zero_sum_c010", "zero_sum_c020", "zero_sum_c040"):
            active = by_key.get((candidate_id, seed))
            if active is not None:
                suffix = _coefficient_suffix(candidate_id)
                expected_controls = sorted(
                    value for value in all_candidate_ids
                    if value != candidate_id
                    and not value.startswith("zero_sum_")
                    and value != "exact_off"
                    and _coefficient_suffix(value) == suffix
                )
                matched_controls = [
                    by_key[(value, seed)] for value in expected_controls
                    if (value, seed) in by_key
                ]
                output.append(paired_record(
                    active, off,
                    matched_controls=matched_controls,
                    expected_matched_control_ids=expected_controls,
                ))
    return output


def model_internal_decision(per_run: list[dict[str, Any]],
                            paired: list[dict[str, Any]],
                            manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_seeds = sorted(
        int(seed) for seed in (
            manifest["search"]["canary_network_seeds"]
            + manifest["search"]["fit_network_seeds"]
        )
    )
    arm_rows = []
    for candidate_id in ("zero_sum_c010", "zero_sum_c020", "zero_sum_c040"):
        selected = [row for row in paired if row["candidate_id"] == candidate_id]
        pass_count = int(np.sum([
            row["model_internal_network_pass"] for row in selected
        ]))
        arm_rows.append({
            "candidate_id": candidate_id,
            "n_networks": int(len(selected)),
            "network_passes": pass_count,
            "status": (
                "MATCHED_CONTROL_EXTENSION_REQUIRED"
                if selected and all(
                    row["matched_control_status"]
                    == "MATCHED_CONTROL_EXTENSION_REQUIRED" for row in selected
                ) else "NOT_EVALUABLE"
                if any(row["status"] == "NOT_EVALUABLE" for row in selected)
                else "FORMALLY_EVALUABLE"
            ),
            "canary_admissible": bool(selected and selected[0]["model_internal_network_pass"]),
            "capacity_pass_2_of_3": bool(len(selected) >= 3 and pass_count >= 2),
            "not_evaluable_seeds": [
                int(row["seed"]) for row in selected
                if row["status"] == "NOT_EVALUABLE"
            ],
            "scientific_failed_seeds": [
                int(row["seed"]) for row in selected
                if row["status"] == "MODEL_INTERNAL_NETWORK_FAIL"
            ],
        })
    completed = sorted({int(row["seed"]) for row in per_run})
    candidate_ids = [
        row["candidate_id"] for row in arm_rows if row["canary_admissible"]
    ]
    capacity_ids = [
        row["candidate_id"] for row in arm_rows if row["capacity_pass_2_of_3"]
    ]
    all_arms = {row["candidate_id"] for row in manifest["candidates"]}
    complete_seed_arms = {
        seed: sorted(row["candidate_id"] for row in per_run if int(row["seed"]) == seed)
        for seed in completed
    }
    canary_complete = all(
        set(complete_seed_arms.get(seed, [])) == all_arms
        for seed in manifest["search"]["canary_network_seeds"]
    )
    replication_complete = all(
        set(complete_seed_arms.get(seed, [])) == all_arms for seed in expected_seeds
    )
    formally_not_evaluable = any(
        row["status"] == "NOT_EVALUABLE" for row in paired
        if row["candidate_id"] == "zero_sum_c020"
    )
    if replication_complete:
        status = (
            "REV13_ZERO_SUM_NODE_CAPACITY_OBSERVED"
            if capacity_ids else "ZERO_SUM_NODE_CAPACITY_NOT_OBSERVED"
        )
        if formally_not_evaluable and not capacity_ids:
            status = "REV13_DIRECTIONAL_K2_NOT_EVALUABLE"
    elif canary_complete:
        status = (
            "REV13_ZERO_SUM_NODE_CANARY_CANDIDATE_FOUND"
            if candidate_ids else "ZERO_SUM_NODE_CAPACITY_NOT_OBSERVED_IN_CANARY"
        )
        if formally_not_evaluable and not candidate_ids:
            status = "REV13_DIRECTIONAL_K2_NOT_EVALUABLE"
    else:
        status = "REV13_MODEL_INTERNAL_AGGREGATE_INCOMPLETE"
    return {
        "schema_id": "topic4_rev13_node_zero_sum_model_internal_decision_v2",
        "status": status,
        "canary_complete": bool(canary_complete),
        "replication_complete": bool(replication_complete),
        "completed_network_seeds": completed,
        "expected_network_seeds": expected_seeds,
        "canary_candidate_ids": candidate_ids,
        "capacity_candidate_ids": capacity_ids,
        "arms": arm_rows,
        "rules": {
            "minimum_minority_fraction": MINORITY_FRACTION,
            "minimum_returned_yield_ratio_vs_off": MINIMUM_YIELD_RATIO,
            "maximum_compound_fraction_delta_vs_off": MAXIMUM_COMPOUND_DELTA,
            "minimum_capacity_networks": 2,
            "required_capacity_networks_total": 3,
            "same_network_opposite_directions_required": True,
            "formal_endpoint": "signed_causal_displacement_1d",
            "heldout_scheme": "three_equal_duration_time_folds",
            "overlap_policy": (
                "exclude_all_members_of_overlap_connected_episodes_before_matching"
            ),
            "minimum_isolated_families": MIN_FORMAL_EVENTS,
            "preferred_isolated_families": PREFERRED_FORMAL_EVENTS,
            "minimum_families_per_direction": MIN_EVENTS_PER_DIRECTION,
            "event_counts_matched_across_active_off_and_controls": True,
            "minimum_within_cluster_direction_consistency": (
                MIN_DIRECTION_CONSISTENCY
            ),
            "minimum_temporal_blocks_per_sign": MIN_TEMPORAL_SIGN_BLOCKS,
            "temporal_blocks_total": N_TEMPORAL_BLOCKS,
            "forced_alternation_rejected": True,
            "repeated_root_fragmentation_role": "DIAGNOSTIC_ONLY",
            "same_coefficient_matched_controls_must_be_outperformed": True,
            "onset_map_kmeans_role": "TOPOLOGY_VISUALIZATION_DIAGNOSTIC_ONLY",
        },
        "patient_labels_or_prototypes_loaded": False,
        "claim_boundary": (
            "Model-internal causal-family capacity only. Patient labels, "
            "prototypes and contact classifiers were not loaded."
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {}
            for field in fields:
                value: Any = row
                for key in field.split("."):
                    value = value.get(key) if isinstance(value, Mapping) else None
                flat[field] = value
            writer.writerow(flat)


def write_outputs(output_root: Path, per_run: list[dict[str, Any]],
                  paired: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, str]:
    aggregate = output_root / "aggregate"
    analysis = output_root / "analysis"
    paths = {
        "per_run_json": aggregate / "model_internal_per_run.json",
        "per_run_csv": aggregate / "model_internal_per_run.csv",
        "paired_json": aggregate / "model_internal_paired.json",
        "paired_csv": aggregate / "model_internal_paired.csv",
        "decision_json": analysis / "model_internal_decision.json",
        "decision_csv": analysis / "model_internal_decision.csv",
    }
    public_per_run = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in per_run
    ]
    _write_json(paths["per_run_json"], {"rows": public_per_run})
    _write_json(paths["paired_json"], {"rows": paired})
    _write_json(paths["decision_json"], decision)
    _write_csv(paths["per_run_csv"], public_per_run, [
        "candidate_id", "seed", "n_returned_evaluable_causal_families",
        "n_isolated_returned_evaluable_causal_families",
        "preferred_formal_event_count_reached",
        "overlap_connected_episode_audit.n_overlap_pairs",
        "overlap_connected_episode_audit.n_overlap_components",
        "overlap_connected_episode_audit.n_excluded_families",
        "whole_sheet_onset_map_kmeans_diagnostic.minority_fraction",
        "whole_sheet_onset_map_kmeans_diagnostic.same_network_opposite_directions",
        "whole_sheet_onset_map_kmeans_diagnostic.kmeans_seed_ami_median",
        "compound_detector_fragment_fraction", "runaway",
        "controller.maximum_zero_sum_error_mV",
        "controller.maximum_support_weighted_saturation_fraction",
        "controller.maximum_support_weighted_static_modulation_sign_flip_fraction",
    ])
    _write_csv(paths["paired_csv"], paired, [
        "candidate_id", "seed", "status", "matched_event_count_per_arm",
        "matched_control_status",
        "directional_k2_formal.heldout_k2_minus_k1_loglik_per_event",
        "directional_k2_delta_vs_off",
        "returned_event_yield_ratio_vs_off", "compound_fraction_delta_vs_off",
        "transition.anti_persistent",
        "fragmentation.repeated_root_crosses_k2_clusters",
        "formal_comparison_evaluable",
        "model_internal_network_pass",
    ])
    _write_csv(paths["decision_csv"], decision.get("arms", []), [
        "candidate_id", "n_networks", "network_passes",
        "canary_admissible", "capacity_pass_2_of_3",
    ])
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_rev13_node_zero_sum_recovery.json",
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    artifact_root = args.artifact_root.resolve()
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise RuntimeError("rev13 candidate manifest is not frozen")
    if manifest.get("config_sha256") != _sha256(args.config.resolve()):
        raise RuntimeError("rev13 candidate manifest is stale")
    _assert_patient_free_manifest(manifest)
    candidate_ids = {str(row["candidate_id"]) for row in manifest["candidates"]}
    output_root = artifact_root / config["output_root"]
    worker_paths = sorted((output_root / "workers").glob("*.json"))
    workers = [
        _load_worker(path, expected_candidate_ids=candidate_ids)
        for path in worker_paths
    ]
    per_run = [per_run_record(worker) for worker in workers]
    paired = build_paired_rows(per_run, expected_candidate_ids=candidate_ids)
    decision = model_internal_decision(per_run, paired, manifest)
    paths = write_outputs(output_root, per_run, paired, decision)
    print(json.dumps({
        "status": decision["status"],
        "n_runs": len(per_run),
        "n_pairs": len(paired),
        "outputs": paths,
    }, indent=2))


if __name__ == "__main__":
    main()
