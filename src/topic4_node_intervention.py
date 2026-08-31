"""Model-internal target selection for rev12-ND same-checkpoint intervention."""
from __future__ import annotations

import numpy as np

from src.topic4_node_dualmode import (
    event_features,
    normalize_event_ranks,
    source_topology_features,
)


def early_support_probability(onset_maps: np.ndarray,
                              fraction: float = 0.10) -> np.ndarray:
    """Equal-event probability that a bin belongs to an event's earliest support."""
    maps = np.asarray(onset_maps, float)
    if maps.ndim != 3 or not len(maps):
        raise ValueError("onset maps must have shape (event, y, x) and be non-empty")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    support = np.zeros_like(maps, dtype=float)
    evaluable = np.zeros(len(maps), bool)
    for index, onset in enumerate(maps):
        finite = np.isfinite(onset)
        if not np.any(finite):
            continue
        count = int(np.sum(finite))
        early_count = min(count, max(1, int(np.ceil(float(fraction) * count))))
        threshold = np.partition(onset[finite], early_count - 1)[early_count - 1]
        support[index, finite & (onset <= threshold)] = 1.0
        evaluable[index] = True
    if not np.any(evaluable):
        raise ValueError("onset maps contain no evaluable event")
    return np.mean(support[evaluable], axis=0)


def network_balanced_early_support(onset_maps_by_network: list[np.ndarray],
                                   labels_by_network: list[np.ndarray],
                                   mode: int, fraction: float = 0.10) -> np.ndarray:
    """Average within-network early probabilities so event-rich seeds do not dominate."""
    if len(onset_maps_by_network) != len(labels_by_network):
        raise ValueError("network maps and labels do not align")
    probabilities = []
    for maps, labels in zip(onset_maps_by_network, labels_by_network):
        maps = np.asarray(maps, float)
        labels = np.asarray(labels, int)
        if maps.ndim != 3 or labels.shape != (len(maps),):
            raise ValueError("one network map bundle does not align")
        selected = maps[labels == int(mode)]
        if len(selected):
            probabilities.append(early_support_probability(selected, fraction=fraction))
    if not probabilities:
        raise ValueError(f"mode {mode} has no evaluable network")
    return np.mean(probabilities, axis=0)


def select_representative_seed(network_scores: list[dict],
                               source_counts: dict[int, int]) -> int:
    """Select a fully dual-mode seed nearest the candidate's median objective."""
    eligible = [
        row for row in network_scores
        if np.all(np.asarray(row.get("mode_counts", []), int) > 0)
        and int(source_counts.get(int(row["seed"]), 0)) >= 2
        and np.isfinite(float(row["objective"]))
    ]
    if not eligible:
        raise ValueError("no dual-mode source-evaluable confirmation seed")
    objectives = np.asarray([float(row["objective"]) for row in eligible])
    events = np.asarray([int(row["n_events"]) for row in eligible])
    objective_median = float(np.median(objectives))
    event_median = float(np.median(events))
    ranked = sorted(
        eligible,
        key=lambda row: (
            abs(float(row["objective"]) - objective_median),
            abs(int(row["n_events"]) - event_median),
            int(row["seed"]),
        ),
    )
    return int(ranked[0]["seed"])


def grid_covariates(positions_e: np.ndarray, h_e: np.ndarray,
                    baseline_spikes: np.ndarray, *, dt_ms: float,
                    sheet_mm: float, bin_mm: float) -> dict:
    """Bin h, local E density and baseline E rate on the source-topology grid."""
    positions = np.asarray(positions_e, float)
    h = np.asarray(h_e, float)
    spikes = np.asarray(baseline_spikes, bool)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or h.shape != (len(positions),)
            or spikes.ndim != 2 or spikes.shape[1] != len(positions)):
        raise ValueError("positions, h and baseline spikes do not align")
    size = int(round(float(sheet_mm) / float(bin_mm)))
    if size <= 0 or not np.isclose(size * float(bin_mm), float(sheet_mm)):
        raise ValueError("bin size must tile the sheet")
    xy = np.floor(positions / float(bin_mm)).astype(int)
    xy = np.clip(xy, 0, size - 1)
    flat = xy[:, 1] * size + xy[:, 0]
    density = np.bincount(flat, minlength=size * size).astype(float)
    h_sum = np.bincount(flat, weights=h, minlength=size * size)
    duration_seconds = len(spikes) * float(dt_ms) * 1e-3
    if duration_seconds <= 0.0:
        raise ValueError("baseline window must be non-empty")
    spike_sum = np.bincount(
        flat, weights=np.sum(spikes, axis=0), minlength=size * size,
    )
    h_mean = np.divide(h_sum, density, out=np.full_like(h_sum, np.nan),
                       where=density > 0)
    rate = np.divide(
        spike_sum, density * duration_seconds,
        out=np.full_like(spike_sum, np.nan), where=density > 0,
    )
    return {
        "h_mean": h_mean.reshape(size, size),
        "e_density": density.reshape(size, size),
        "baseline_rate_hz": rate.reshape(size, size),
    }


def intervention_footprint_covariates(
        positions_e: np.ndarray, h_e: np.ndarray,
        baseline_spikes: np.ndarray, *, dt_ms: float,
        sheet_mm: float, bin_mm: float, target_radius_mm: float,
        additional_node_covariates: dict[str, np.ndarray] | None = None) -> dict:
    """Measure matching covariates over the actual circular pulse footprint."""
    positions = np.asarray(positions_e, float)
    h = np.asarray(h_e, float)
    spikes = np.asarray(baseline_spikes, bool)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or h.shape != (len(positions),)
            or spikes.ndim != 2 or spikes.shape[1] != len(positions)):
        raise ValueError("positions, h and baseline spikes do not align")
    size = int(round(float(sheet_mm) / float(bin_mm)))
    if size <= 0 or not np.isclose(size * float(bin_mm), float(sheet_mm)):
        raise ValueError("bin size must tile the sheet")
    if float(target_radius_mm) <= 0.0:
        raise ValueError("target radius must be positive")
    duration_seconds = len(spikes) * float(dt_ms) * 1e-3
    if duration_seconds <= 0.0:
        raise ValueError("baseline window must be non-empty")

    centers = _grid_centers((size, size), bin_mm)
    neuron_rates = np.sum(spikes, axis=0) / duration_seconds
    additional = {
        str(key): np.asarray(value, float)
        for key, value in (additional_node_covariates or {}).items()
    }
    if any(value.shape != (len(positions),) for value in additional.values()):
        raise ValueError("additional node covariates do not align to E neurons")
    counts = np.zeros(len(centers), float)
    h_mean = np.full(len(centers), np.nan)
    rate = np.full(len(centers), np.nan)
    additional_means = {
        key: np.full(len(centers), np.nan) for key in additional
    }
    for index, center in enumerate(centers):
        selected = np.linalg.norm(positions - center[None, :], axis=1) <= float(
            target_radius_mm
        )
        counts[index] = int(np.sum(selected))
        if np.any(selected):
            h_mean[index] = float(np.mean(h[selected]))
            rate[index] = float(np.mean(neuron_rates[selected]))
            for key, value in additional.items():
                additional_means[key][index] = float(np.mean(value[selected]))
    return {
        "h_mean": h_mean.reshape(size, size),
        # Kept under the established key so the matcher remains reusable. Here
        # this is the exact number of E neurons affected by the circular pulse.
        "e_density": counts.reshape(size, size),
        "baseline_rate_hz": rate.reshape(size, size),
        "covariate_footprint": "circular_intervention_target",
        "target_radius_mm": float(target_radius_mm),
        **{
            key: value.reshape(size, size)
            for key, value in additional_means.items()
        },
    }


def _grid_centers(shape: tuple[int, int], bin_mm: float) -> np.ndarray:
    y, x = np.indices(shape)
    return np.column_stack([
        (x.ravel() + 0.5) * float(bin_mm),
        (y.ravel() + 0.5) * float(bin_mm),
    ])


def select_hotspot_triplet(
        early_probability: np.ndarray, covariates: dict, *,
        bin_mm: float, minimum_separation_mm: float = 3.0,
        off_template_quantile: float = 0.25,
        competing_probability: np.ndarray | None = None,
        require_positive_contrast: bool = False,
        maximum_standardized_l1: float | None = None,
        maximum_standardized_component: float | None = None,
        covariate_keys: tuple[str, ...] | list[str] | None = None) -> dict:
    """Select dominant, separated secondary and covariate-matched control bins."""
    probability = np.asarray(early_probability, float)
    if probability.ndim != 2 or not np.all(np.isfinite(probability)):
        raise ValueError("early probability must be a finite 2D map")
    competing = None
    if competing_probability is not None:
        competing = np.asarray(competing_probability, float)
        if competing.shape != probability.shape or not np.all(np.isfinite(competing)):
            raise ValueError("competing probability must align and be finite")
    required = tuple(covariate_keys or (
        "h_mean", "e_density", "baseline_rate_hz",
    ))
    if "e_density" not in required or len(set(required)) != len(required):
        raise ValueError("matching covariates must uniquely include e_density")
    arrays = {key: np.asarray(covariates[key], float) for key in required}
    if any(value.shape != probability.shape for value in arrays.values()):
        raise ValueError("covariate maps must align to the early-probability map")
    valid = arrays["e_density"] > 0
    valid &= np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    if np.sum(valid) < 3:
        raise ValueError("fewer than three populated bins are available")
    centers = _grid_centers(probability.shape, bin_mm)
    p = probability.ravel()
    other = np.zeros_like(p) if competing is None else competing.ravel()
    selection_score = p if competing is None else p - other
    valid_flat = valid.ravel()
    dominant_pool = valid_flat & (p > 0.0)
    if require_positive_contrast:
        dominant_pool &= selection_score > 0.0
    if not np.any(dominant_pool):
        raise ValueError("no supported mode-discriminative hotspot is available")
    dominant_candidates = np.flatnonzero(dominant_pool)
    dominant = int(dominant_candidates[np.argmax(selection_score[dominant_candidates])])
    distance_to_dominant = np.linalg.norm(centers - centers[dominant], axis=1)
    secondary_pool = (
        valid_flat
        & (distance_to_dominant >= minimum_separation_mm)
        & (p > 0.0)
    )
    if not np.any(secondary_pool):
        raise ValueError("no supported spatially separated secondary hotspot is available")
    secondary_candidates = np.flatnonzero(secondary_pool)
    secondary = int(secondary_candidates[np.argmax(p[secondary_candidates])])

    distance_to_secondary = np.linalg.norm(centers - centers[secondary], axis=1)
    union_probability = np.maximum(p, other)
    threshold = float(np.quantile(
        union_probability[valid_flat], float(off_template_quantile),
    ))
    control_pool = (
        valid_flat
        & (distance_to_dominant >= minimum_separation_mm)
        & (distance_to_secondary >= minimum_separation_mm)
        & (union_probability <= threshold)
    )
    if not np.any(control_pool):
        raise ValueError("no separated off-template control bin is available")
    covariate_matrix = np.column_stack([
        arrays[key].ravel() for key in required
    ])
    scale = np.nanpercentile(covariate_matrix[valid_flat], 75, axis=0) - np.nanpercentile(
        covariate_matrix[valid_flat], 25, axis=0,
    )
    scale = np.where(scale > 1e-12, scale, 1.0)
    control_candidates = np.flatnonzero(control_pool)
    mismatch = np.sum(
        np.abs(covariate_matrix[control_candidates] - covariate_matrix[dominant])
        / scale,
        axis=1,
    )
    # Lower template probability is the deterministic secondary key.
    order = np.lexsort((control_candidates, p[control_candidates], mismatch))
    control = int(control_candidates[order[0]])
    standardized = np.abs(
        covariate_matrix[control] - covariate_matrix[dominant]
    ) / scale
    standardized_l1 = float(np.sum(standardized))
    standardized_max = float(np.max(standardized))
    match_acceptable = bool(
        (maximum_standardized_l1 is None
         or standardized_l1 <= float(maximum_standardized_l1))
        and (maximum_standardized_component is None
             or standardized_max <= float(maximum_standardized_component))
    )

    def record(index: int) -> dict:
        row, column = np.unravel_index(index, probability.shape)
        record = {
            "flat_index": index,
            "row": int(row),
            "column": int(column),
            "xy_mm": centers[index].tolist(),
            "early_probability": float(p[index]),
            "competing_mode_early_probability": float(other[index]),
            "mode_probability_contrast": float(p[index] - other[index]),
        }
        record.update({
            key: float(covariate_matrix[index, position])
            for position, key in enumerate(required)
        })
        return record

    return {
        "dominant": record(dominant),
        "secondary": record(secondary),
        "matched_off_template": record(control),
        "minimum_separation_mm": float(minimum_separation_mm),
        "off_template_quantile": float(off_template_quantile),
        "match_quality": {
            "covariates": list(required),
            "standardized_absolute_difference": standardized.tolist(),
            "standardized_l1": standardized_l1,
            "maximum_standardized_component": standardized_max,
            "maximum_allowed_standardized_l1": maximum_standardized_l1,
            "maximum_allowed_standardized_component": (
                maximum_standardized_component
            ),
            "acceptable": match_acceptable,
            "control_pool_size": int(len(control_candidates)),
            "scale": scale.tolist(),
        },
    }


def representative_event_index(onset_maps: np.ndarray, ranks: np.ndarray,
                               labels: np.ndarray, mode: int) -> int:
    """Choose a joint source-topology/contact-rank medoid without visual selection."""
    maps = np.asarray(onset_maps, float)
    ranks = np.asarray(ranks, float)
    labels = np.asarray(labels, int)
    if maps.ndim != 3 or ranks.ndim != 2 or labels.shape != (len(maps),):
        raise ValueError("event maps, ranks and labels do not align")
    selected = np.flatnonzero(labels == int(mode))
    if not len(selected):
        raise ValueError(f"mode {mode} has no evaluable event")
    source = source_topology_features(maps[selected])
    contact = event_features(normalize_event_ranks(ranks[selected]))
    source /= np.sqrt(max(1, source.shape[1]))
    contact /= np.sqrt(max(1, contact.shape[1]))
    joint = np.concatenate([source, contact], axis=1)
    centroid = np.mean(joint, axis=0)
    distance = np.linalg.norm(joint - centroid, axis=1)
    return int(selected[int(np.argmin(distance))])


def native_mode_outcome(
    onset: np.ndarray, *, patient_mode: int, ood: bool, native_mode: int,
    groups: dict[str, np.ndarray],
) -> dict:
    """Classify whether a branch retained its original supported patient mode."""
    onset = np.asarray(onset, float)
    if onset.ndim != 1 or any(shaft not in groups for shaft in ("ICL", "SCL")):
        raise ValueError("onset and shaft groups do not define a contact event")
    shaft_indices = {
        shaft: np.asarray(groups[shaft], dtype=int) for shaft in ("ICL", "SCL")
    }
    if any(
        indices.ndim != 1 or not len(indices)
        or np.any(indices < 0) or np.any(indices >= len(onset))
        for indices in shaft_indices.values()
    ):
        raise ValueError("shaft groups contain invalid contact indices")
    dual_shaft = all(
        np.isfinite(onset[shaft_indices[shaft]]).any()
        for shaft in ("ICL", "SCL")
    )
    formal_clean = bool(dual_shaft and not bool(ood))
    retained = bool(formal_clean and int(patient_mode) == int(native_mode))
    if not dual_shaft:
        outcome = "single_shaft_or_contact_unreadable"
    elif bool(ood):
        outcome = "classifier_ood"
    elif not retained:
        outcome = "patient_mode_switch"
    else:
        outcome = "native_patient_mode_retained"
    return {
        "dual_shaft": bool(dual_shaft),
        "formal_clean": formal_clean,
        "native_mode_retained": retained,
        "event_outcome": outcome,
    }


def ordered_suppression_effect(sham: dict, branch: dict) -> tuple[int, float]:
    """Prioritize loss of the native supported mode, then its onset delay."""
    if not bool(sham.get("event_occurred")):
        raise ValueError("same-checkpoint sham must reproduce the native event")
    if sham.get("native_mode_retained") is not True:
        raise ValueError("same-checkpoint sham must retain the native patient mode")
    if "native_mode_retained" not in branch:
        raise ValueError("intervention branch lacks native-mode retention status")
    if not bool(branch.get("native_mode_retained")):
        return 1, 0.0
    sham_latency = float(sham["latency_from_checkpoint_ms"])
    branch_latency = float(branch["latency_from_checkpoint_ms"])
    if not np.isfinite(sham_latency) or not np.isfinite(branch_latency):
        raise ValueError("surviving intervention events require finite latencies")
    return 0, max(0.0, branch_latency - sham_latency)


def crossed_hotspot_selectivity(
    records_by_network: list[dict], *, required_networks: int = 2,
) -> dict:
    """Audit whether a mode hotspot has a crossed, spatially specific effect.

    Each network record contains native modes ``0`` and ``1``. Each native-mode
    branch dictionary must contain ``sham``, ``mode0_hotspot``,
    ``mode0_matched_off_template``, ``mode1_hotspot`` and
    ``mode1_matched_off_template``.
    """
    if int(required_networks) < 1:
        raise ValueError("required selective networks must be positive")
    if len(records_by_network) < int(required_networks):
        raise ValueError("fewer networks than the selective-effect requirement")
    results = {}
    for hotspot_mode in (0, 1):
        per_network = []
        for network in records_by_network:
            native = network["native_modes"]
            own = native[str(hotspot_mode)]
            opposite = native[str(1 - hotspot_mode)]
            own_effect = ordered_suppression_effect(
                own["sham"], own[f"mode{hotspot_mode}_hotspot"],
            )
            opposite_effect = ordered_suppression_effect(
                opposite["sham"], opposite[f"mode{hotspot_mode}_hotspot"],
            )
            matched_effect = ordered_suppression_effect(
                own["sham"],
                own[f"mode{hotspot_mode}_matched_off_template"],
            )
            matched_control_acceptable = bool(
                own[f"mode{hotspot_mode}_matched_off_template"].get(
                    "control_match_acceptable", True,
                )
            )
            cross_mode_hotspots_distinct = bool(
                own.get("cross_mode_hotspots_distinct", True)
            )
            selective = bool(
                own_effect > opposite_effect and own_effect > matched_effect
                and matched_control_acceptable and cross_mode_hotspots_distinct
            )
            per_network.append({
                "network_seed": int(network["network_seed"]),
                "own_mode_effect_event_abolished_then_delay": list(own_effect),
                "opposite_mode_effect_event_abolished_then_delay": list(opposite_effect),
                "matched_control_effect_event_abolished_then_delay": list(matched_effect),
                "matched_control_acceptable": matched_control_acceptable,
                "cross_mode_hotspots_distinct": cross_mode_hotspots_distinct,
                "selective": selective,
            })
        count = int(sum(row["selective"] for row in per_network))
        results[str(hotspot_mode)] = {
            "selective_network_count": count,
            "required_network_count": int(required_networks),
            "pass": bool(count >= int(required_networks)),
            "per_network": per_network,
        }
    passed = [int(mode) for mode, row in results.items() if row["pass"]]
    return {
        "node_freeze_permitted": bool(passed),
        "selective_hotspot_modes": passed,
        "modes": results,
        "primary_ordered_effect": (
            "native supported-mode loss first, then nonnegative onset delay"
        ),
        "continuous_endpoints_are_explanatory_not_additional_gates": True,
    }
