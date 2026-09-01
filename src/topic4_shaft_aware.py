"""Shaft-aware event representation and distances for Topic 4 rev10-SA.

The module deliberately has no SNN dependency.  It keeps contact identity and
missing recruitment explicit so an absent shaft cannot disappear from a loss
through finite-value filtering.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score


SHAFT_ORDER = ("ICL", "SCL")
PAIR_CLASS_ORDER = ("ICL-ICL", "SCL-SCL", "ICL-SCL")
PAIR_STATE_ORDER = ("i_before_j", "j_before_i", "tie", "not_jointly_recruited")


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contact_parts(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)([0-9]+)", str(name))
    if match is None:
        raise ValueError(f"contact name does not encode shaft and number: {name!r}")
    shaft, number = match.group(1).upper(), int(match.group(2))
    if shaft not in SHAFT_ORDER:
        raise ValueError(f"unsupported shaft {shaft!r} in contact {name!r}")
    return shaft, number


def build_contact_contract(
    contact_names: Sequence[str],
    sheet_xy_mm: np.ndarray,
    shared_axis_coordinate_mm: Sequence[float],
    readout_parameters: Mapping,
) -> dict:
    """Build the canonical 15-contact geometry and unordered-pair contract."""
    names = [str(name) for name in contact_names]
    xy = np.asarray(sheet_xy_mm, dtype=float)
    axial = np.asarray(shared_axis_coordinate_mm, dtype=float)
    if len(names) != 15 or len(set(names)) != 15:
        raise ValueError("rev10-SA requires exactly 15 unique contacts")
    if xy.shape != (15, 2) or axial.shape != (15,):
        raise ValueError("contact geometry must be (15,2) with 15 axial coordinates")
    if not np.isfinite(xy).all() or not np.isfinite(axial).all():
        raise ValueError("contact geometry must be finite")

    parsed = [_contact_parts(name) for name in names]
    by_shaft = {
        shaft: sorted(
            (index for index, (value, _) in enumerate(parsed) if value == shaft),
            key=lambda index: (axial[index], names[index]),
        )
        for shaft in SHAFT_ORDER
    }
    expected = {"ICL": 11, "SCL": 4}
    observed = {shaft: len(indices) for shaft, indices in by_shaft.items()}
    if observed != expected:
        raise ValueError(f"unexpected shaft counts: {observed}")
    within_order = {
        index: order
        for shaft in SHAFT_ORDER
        for order, index in enumerate(by_shaft[shaft])
    }

    contacts = []
    for index, (name, (shaft, number)) in enumerate(zip(names, parsed)):
        contacts.append({
            "contact_name": name,
            "contact_index": index,
            "shaft_id": shaft,
            "contact_number": number,
            "within_shaft_order_by_shared_axis": within_order[index],
            "sheet_xy_mm": [float(value) for value in xy[index]],
            "shared_axis_coordinate_mm": float(axial[index]),
        })

    pairs = []
    pair_counts = {name: 0 for name in PAIR_CLASS_ORDER}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            shafts = {parsed[left][0], parsed[right][0]}
            if len(shafts) == 1:
                shaft = parsed[left][0]
                pair_class = f"{shaft}-{shaft}"
            else:
                pair_class = "ICL-SCL"
            pair_counts[pair_class] += 1
            pairs.append({
                "i": left,
                "j": right,
                "contact_i": names[left],
                "contact_j": names[right],
                "pair_class": pair_class,
            })
    if pair_counts != {"ICL-ICL": 55, "SCL-SCL": 6, "ICL-SCL": 44}:
        raise RuntimeError(f"pair contract is inconsistent: {pair_counts}")

    shaft_assignment = [{
        "contact_name": row["contact_name"],
        "contact_index": row["contact_index"],
        "shaft_id": row["shaft_id"],
        "contact_number": row["contact_number"],
        "within_shaft_order_by_shared_axis": row["within_shaft_order_by_shared_axis"],
    } for row in contacts]
    geometry = [{
        "contact_name": row["contact_name"],
        "contact_index": row["contact_index"],
        "sheet_xy_mm": row["sheet_xy_mm"],
        "shared_axis_coordinate_mm": row["shared_axis_coordinate_mm"],
    } for row in contacts]
    pair_identity = [{
        "i": row["i"], "j": row["j"], "pair_class": row["pair_class"],
    } for row in pairs]
    readout = json.loads(json.dumps(dict(readout_parameters), allow_nan=False))
    return {
        "schema": "topic4_rev10_sa_contact_contract_v1",
        "contacts": contacts,
        "shaft_counts": observed,
        "pairs": pairs,
        "pair_counts": pair_counts,
        "pair_state_order": list(PAIR_STATE_ORDER),
        "readout_parameters": readout,
        "hashes": {
            "contact_geometry_sha256": _canonical_sha256(geometry),
            "shaft_assignment_sha256": _canonical_sha256(shaft_assignment),
            "pair_class_sha256": _canonical_sha256(pair_identity),
            "readout_parameters_sha256": _canonical_sha256(readout),
        },
    }


def contract_groups(contract: Mapping) -> dict[str, np.ndarray]:
    groups = {}
    for shaft in SHAFT_ORDER:
        groups[shaft] = np.asarray([
            row["contact_index"] for row in contract["contacts"]
            if row["shaft_id"] == shaft
        ], dtype=int)
    return groups


def contract_pairs(contract: Mapping) -> dict[str, np.ndarray]:
    output = {}
    for pair_class in PAIR_CLASS_ORDER:
        output[pair_class] = np.asarray([
            (row["i"], row["j"]) for row in contract["pairs"]
            if row["pair_class"] == pair_class
        ], dtype=int).reshape((-1, 2))
    return output


def normalized_event_onsets(onsets: np.ndarray) -> np.ndarray:
    """Normalize finite contact onsets independently within every event."""
    values = np.asarray(onsets, dtype=float)
    if values.ndim != 2:
        raise ValueError("onsets must be events x contacts")
    output = np.full(values.shape, np.nan, dtype=float)
    for event_index, row in enumerate(values):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        event = row[finite]
        span = float(np.ptp(event))
        output[event_index, finite] = (
            np.zeros(len(event), dtype=float)
            if span <= 1e-12 else (event - float(event.min())) / span
        )
    return output


def build_event_features(
    onsets: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> dict:
    """Return the fixed-identity rev10-SA feature and named components."""
    values = np.asarray(onsets, dtype=float)
    if values.ndim != 2:
        raise ValueError("onsets must be events x contacts")
    mask = np.isfinite(values)
    normalized = normalized_event_onsets(values)
    recruited_onset = np.where(mask, normalized, 0.0)
    fractions = np.column_stack([
        mask[:, np.asarray(groups[shaft], dtype=int)].mean(axis=1)
        for shaft in SHAFT_ORDER
    ])
    first = {}
    present = {}
    for shaft in SHAFT_ORDER:
        indices = np.asarray(groups[shaft], dtype=int)
        shaft_values = normalized[:, indices]
        present[shaft] = np.isfinite(shaft_values).any(axis=1)
        first[shaft] = np.min(
            np.where(np.isfinite(shaft_values), shaft_values, np.inf), axis=1,
        )
    delta_valid = present["ICL"] & present["SCL"]
    delta_first = np.zeros(len(values), dtype=float)
    delta_first[delta_valid] = (
        first["SCL"][delta_valid] - first["ICL"][delta_valid]
    )
    features = np.column_stack([
        mask.astype(float), recruited_onset, fractions,
        delta_first, delta_valid.astype(float),
    ])
    return {
        "features": features,
        "mask": mask,
        "normalized_onsets": normalized,
        "recruited_onset": recruited_onset,
        "shaft_fraction": fractions,
        "delta_first": delta_first,
        "delta_valid": delta_valid,
    }


def pair_state_distributions(
    onsets: np.ndarray,
    pairs_by_class: Mapping[str, np.ndarray],
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Four-state probabilities for every unordered contact pair."""
    values = np.asarray(onsets, dtype=float)
    if values.ndim != 2:
        raise ValueError("onsets must be events x contacts")
    output = {}
    for pair_class in PAIR_CLASS_ORDER:
        pairs = np.asarray(pairs_by_class[pair_class], dtype=int).reshape((-1, 2))
        probabilities = np.zeros((len(pairs), len(PAIR_STATE_ORDER)), dtype=float)
        for pair_index, (left, right) in enumerate(pairs):
            left_values, right_values = values[:, left], values[:, right]
            joint = np.isfinite(left_values) & np.isfinite(right_values)
            difference = left_values - right_values
            probabilities[pair_index, 0] = np.mean(
                joint & (difference < -float(tie_tolerance))
            )
            probabilities[pair_index, 1] = np.mean(
                joint & (difference > float(tie_tolerance))
            )
            probabilities[pair_index, 2] = np.mean(
                joint & (np.abs(difference) <= float(tie_tolerance))
            )
            probabilities[pair_index, 3] = np.mean(~joint)
        output[pair_class] = probabilities
    return output


def describe_events(
    onsets: np.ndarray,
    groups: Mapping[str, np.ndarray],
    pairs_by_class: Mapping[str, np.ndarray],
    *,
    tie_tolerance: float = 1e-12,
) -> dict:
    values = np.asarray(onsets, dtype=float)
    features = build_event_features(values, groups)
    recruitment = {
        shaft: features["mask"][:, np.asarray(groups[shaft], dtype=int)].mean(axis=0)
        for shaft in SHAFT_ORDER
    }
    profile = {
        shaft: features["recruited_onset"][:, np.asarray(groups[shaft], dtype=int)].mean(axis=0)
        for shaft in SHAFT_ORDER
    }
    valid = features["delta_valid"]
    profile["cross"] = np.asarray([
        float(np.mean(valid)),
        float(np.mean(features["delta_first"][valid])) if valid.any() else 0.0,
    ])
    return {
        "n_events": int(len(values)),
        "recruitment": recruitment,
        "precedence": pair_state_distributions(
            values, pairs_by_class, tie_tolerance=tie_tolerance,
        ),
        "profile": profile,
        "features": features["features"],
        "multishaft_fraction": float(np.mean(valid)) if len(valid) else float("nan"),
    }


def _js_divergence_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("probability tables must have matching 2-D shapes")
    midpoint = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        left_term = np.where(left > 0.0, left * np.log2(left / midpoint), 0.0)
        right_term = np.where(right > 0.0, right * np.log2(right / midpoint), 0.0)
    return 0.5 * (left_term.sum(axis=1) + right_term.sum(axis=1))


def descriptor_distances(candidate: Mapping, target: Mapping) -> dict:
    """Raw component distances before patient-floor calibration."""
    recruitment = {
        shaft: float(np.mean(np.abs(
            np.asarray(candidate["recruitment"][shaft], dtype=float)
            - np.asarray(target["recruitment"][shaft], dtype=float)
        ))) for shaft in SHAFT_ORDER
    }
    precedence = {
        pair_class: float(np.mean(_js_divergence_rows(
            candidate["precedence"][pair_class], target["precedence"][pair_class],
        ))) for pair_class in PAIR_CLASS_ORDER
    }
    profile = {
        shaft: float(np.mean(np.abs(
            np.asarray(candidate["profile"][shaft], dtype=float)
            - np.asarray(target["profile"][shaft], dtype=float)
        ))) for shaft in SHAFT_ORDER
    }
    profile["cross"] = float(np.mean(np.abs(
        np.asarray(candidate["profile"]["cross"], dtype=float)
        - np.asarray(target["profile"]["cross"], dtype=float)
    )))
    return {"recruitment": recruitment, "precedence": precedence, "profile": profile}


def fit_patient_embedding(
    features: np.ndarray,
    *,
    variance_fraction: float = 0.95,
    max_components: int = 12,
    reference_n: int = 4096,
    n_directions: int = 64,
    seed: int = 20260811,
) -> dict:
    """Fit and serialize a patient-only standardized PCA/SW contract."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("embedding features must be a finite 2-D patient table")
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (values - center) / scale
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    variance = singular ** 2
    cumulative = np.cumsum(variance) / variance.sum()
    n_components = int(np.searchsorted(cumulative, float(variance_fraction)) + 1)
    n_components = max(1, min(n_components, int(max_components), right.shape[0]))
    components = right[:n_components]
    scores = standardized @ components.T

    rng = np.random.default_rng(int(seed))
    take = min(len(scores), int(reference_n))
    reference_indices = np.sort(rng.choice(len(scores), size=take, replace=False))
    directions = rng.normal(size=(int(n_directions), n_components))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return {
        "center": center,
        "scale": scale,
        "components": components,
        "explained_variance_fraction": variance[:n_components] / variance.sum(),
        "reference_indices": reference_indices,
        "reference_z": scores[reference_indices],
        "directions": directions,
        "seed": int(seed),
        "n_train": int(len(values)),
        "n_components": n_components,
    }


def transform_patient_embedding(features: np.ndarray, embedding: Mapping) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    center = np.asarray(embedding["center"], dtype=float)
    scale = np.asarray(embedding["scale"], dtype=float)
    components = np.asarray(embedding["components"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(center):
        raise ValueError("features do not match the frozen shaft-aware embedding")
    return ((values - center) / scale) @ components.T


def sliced_event_cloud_distance(
    features: np.ndarray,
    embedding: Mapping,
    *,
    reference_z: np.ndarray | None = None,
) -> float:
    z = transform_patient_embedding(features, embedding)
    reference = np.asarray(
        embedding["reference_z"] if reference_z is None else reference_z,
        dtype=float,
    )
    directions = np.asarray(embedding["directions"], dtype=float)
    if len(z) < 2:
        return float("nan")
    return float(np.mean([
        wasserstein_distance(z @ direction, reference @ direction)
        for direction in directions
    ]))


def consensus_kmeans(
    z: np.ndarray,
    *,
    n_clusters: int,
    seeds: Sequence[int],
    n_init: int = 20,
) -> dict:
    """Choose the KMeans run with highest mean AMI to all other runs."""
    values = np.asarray(z, dtype=float)
    seeds = [int(seed) for seed in seeds]
    if values.ndim != 2 or len(values) < int(n_clusters) or len(seeds) < 2:
        raise ValueError("consensus KMeans needs a finite table and at least two seeds")
    labels = np.asarray([
        KMeans(
            n_clusters=int(n_clusters), n_init=int(n_init), random_state=seed,
        ).fit_predict(values)
        for seed in seeds
    ], dtype=int)
    similarity = np.eye(len(seeds), dtype=float)
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            value = adjusted_mutual_info_score(labels[left], labels[right])
            similarity[left, right] = similarity[right, left] = value
    mean_similarity = (similarity.sum(axis=1) - 1.0) / (len(seeds) - 1)
    selected = int(np.argmax(mean_similarity))
    return {
        "labels": labels[selected],
        "selected_seed": seeds[selected],
        "selected_run_index": selected,
        "mean_pairwise_ami": float(np.mean(similarity[np.triu_indices(len(seeds), 1)])),
        "minimum_pairwise_ami": float(np.min(similarity[np.triu_indices(len(seeds), 1)])),
        "selected_mean_ami": float(mean_similarity[selected]),
        "cluster_counts": np.bincount(labels[selected], minlength=int(n_clusters)),
        "n_init_per_seed": int(n_init),
        "similarity_matrix": similarity,
    }


def align_cluster_labels(labels: np.ndarray, reference_labels: np.ndarray) -> dict:
    """Hungarian-align arbitrary cluster ids to frozen reference ids."""
    candidate = np.asarray(labels, dtype=int)
    reference = np.asarray(reference_labels, dtype=int)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("candidate and reference labels must be matching vectors")
    n_candidate = int(candidate.max()) + 1
    n_reference = int(reference.max()) + 1
    contingency = np.zeros((n_candidate, n_reference), dtype=int)
    for left, right in zip(candidate, reference):
        contingency[left, right] += 1
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {int(row): int(column) for row, column in zip(rows, columns)}
    aligned = np.asarray([mapping.get(int(value), -1) for value in candidate], dtype=int)
    return {
        "labels": aligned,
        "mapping": mapping,
        "ami": float(adjusted_mutual_info_score(reference, candidate)),
        "accuracy": float(np.mean(aligned == reference)),
        "contingency": contingency,
    }


def centered_smooth_max(values: Sequence[float], tau: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("smooth maximum needs a non-empty finite vector")
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    maximum = float(values.max())
    return float(maximum + tau * np.log(np.mean(np.exp((values - maximum) / tau))))


def floor_excess(value: float, floor_median: float, floor_q95: float) -> float:
    denominator = float(floor_q95) - float(floor_median)
    if denominator <= 1e-12:
        return 0.0 if float(value) <= float(floor_q95) else float("inf")
    return max(0.0, (float(value) - float(floor_median)) / denominator)
