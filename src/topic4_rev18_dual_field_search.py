"""Robust multi-coordinate search helpers for the rev18 dual Node field."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.topic4_node_field_search import sobol_cosine_combinations


DIMENSION = 30


def global_blueprints(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build deterministic whole-sheet directions without observation geometry."""
    pairs = int(design["n_antithetic_pairs"])
    radii = tuple(float(value) for value in design["radii"])
    rows = sobol_cosine_combinations(
        n_pairs=pairs, n_modes=DIMENSION, radii=radii,
        seed=int(design["sobol_seed"]),
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        radius = float(row["radius"])
        coefficients = np.asarray(row["coefficients"], dtype=float)
        direction = coefficients / radius
        sign = "m" if int(row["sign"]) < 0 else "p"
        output.append({
            "candidate_id": (
                f"rev18_s{int(row['pair_index']):02d}_{sign}_"
                f"r{int(round(100 * radius)):02d}"
            ),
            "family": "sobol_multicoordinate_dual_field",
            "nomination_strategy": "frozen_uniform_sheet_sobol",
            "radius": radius,
            "direction": direction.tolist(),
            "pair_index": int(row["pair_index"]),
            "sign": int(row["sign"]),
            "observation_coordinates_used": False,
        })

    for record in design.get("sentinel_directions", []):
        direction = np.asarray(record["direction"], dtype=float)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (DIMENSION,) or not np.isfinite(direction).all() \
                or norm <= 1e-12:
            raise ValueError("rev18 sentinel direction is invalid")
        direction = direction / norm
        output.append({
            "candidate_id": str(record["candidate_id"]),
            "family": "rev17_generalization_sentinel",
            "nomination_strategy": "frozen_prior_round_sentinel",
            "radius": float(record["radius"]),
            "direction": direction.tolist(),
            "pair_index": None,
            "sign": None,
            "observation_coordinates_used": False,
        })
    identifiers = [row["candidate_id"] for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("rev18 candidate identifiers are duplicated")
    if any(
        not np.isclose(np.linalg.norm(row["direction"]), 1.0, atol=1e-10)
        for row in output
    ):
        raise RuntimeError("rev18 candidate direction lost unit norm")
    return output


def _positive_ratio(value: float, reference: float) -> float:
    value, reference = float(value), float(reference)
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("rev18 normalized endpoint is not positive and finite")
    return value / reference


def network_loss(
    row: Mapping[str, Any], anchor: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, float]:
    """Compute one equal-network contribution to the frozen robust objective."""
    target_support = float(contract["target_effective_support_per_mode"])
    target_cluster = float(contract["target_natural_cluster_count"])
    target_alignment = float(contract["target_natural_balanced_alignment"])
    b_ratio = float(contract["B_protection_ratio"])
    weights = contract["weights"]
    natural = row.get("natural_kmeans") or {}
    if natural.get("status") == "OK":
        alignment = natural.get("direction_balanced_alignment")
        alignment = 0.0 if alignment is None else float(alignment)
        counts = np.asarray(natural.get("cluster_counts", []), dtype=float)
        minimum_cluster = float(np.min(counts)) if counts.size == 2 else 0.0
        kmeans_not_evaluable = 0.0
    else:
        alignment, minimum_cluster = 0.0, 0.0
        kmeans_not_evaluable = 1.0
    components = {
        "J14_ratio": _positive_ratio(row["j14"], anchor["j14"]),
        "A_ratio": _positive_ratio(row["mode_0_mean"], anchor["mode_0_mean"]),
        "B_excess": max(
            0.0,
            _positive_ratio(row["mode_1_mean"], anchor["mode_1_mean"])
            / b_ratio - 1.0,
        ),
        "A_support_deficit": max(
            0.0, 1.0 - float(row["mode_0_effective_events"]) / target_support,
        ),
        "B_support_deficit": max(
            0.0, 1.0 - float(row["mode_1_effective_events"]) / target_support,
        ),
        "KMeans_alignment_deficit": max(
            0.0, 1.0 - alignment / target_alignment,
        ),
        "KMeans_cluster_deficit": max(
            0.0, 1.0 - minimum_cluster / target_cluster,
        ),
        "KMeans_not_evaluable": kmeans_not_evaluable,
        "OOD_fraction": float(row["ood_count"]) / max(
            1.0, float(row["n_contact_primary"]),
        ),
    }
    total = float(sum(float(weights[name]) * value for name, value in components.items()))
    return {**components, "loss": total}


def evaluate_candidate(
    candidate_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate equal-network losses with frozen dispersion and worst terms."""
    candidates = {int(row["seed"]): row for row in candidate_rows}
    anchors = {int(row["seed"]): row for row in anchor_rows}
    if candidates.keys() != anchors.keys() or not candidates:
        raise ValueError("rev18 candidate and anchor networks do not align")
    valid = all(row.get("run_status") == "VALID" for row in candidates.values())
    if not valid:
        return {
            "valid_all_networks": False,
            "robust_loss": float(contract["invalid_candidate_loss"]),
            "per_network": [],
        }
    per_network = []
    for seed in sorted(candidates):
        components = network_loss(candidates[seed], anchors[seed], contract)
        per_network.append({"seed": seed, **components})
    values = np.asarray([row["loss"] for row in per_network], dtype=float)
    robust = (
        float(np.mean(values))
        + float(contract["network_sd_weight"]) * float(np.std(values))
        + float(contract["worst_network_weight"]) * float(np.max(values))
    )
    return {
        "valid_all_networks": True,
        "robust_loss": robust,
        "mean_network_loss": float(np.mean(values)),
        "network_loss_sd": float(np.std(values)),
        "worst_network_loss": float(np.max(values)),
        "minimum_A_support": float(min(
            row["mode_0_effective_events"] for row in candidates.values()
        )),
        "minimum_B_support": float(min(
            row["mode_1_effective_events"] for row in candidates.values()
        )),
        "minimum_natural_alignment": float(min(
            (row.get("natural_kmeans") or {}).get(
                "direction_balanced_alignment", 0.0,
            ) or 0.0
            for row in candidates.values()
        )),
        "per_network": per_network,
    }


def nominate(
    evaluations: Sequence[Mapping[str, Any]], *, maximum_candidates: int,
) -> list[dict[str, Any]]:
    """Rank the complete screen by its predeclared scalar; no endpoint gates."""
    eligible = [row for row in evaluations if row["valid_all_networks"]]
    eligible.sort(key=lambda row: (
        float(row["robust_loss"]), float(row["worst_network_loss"]),
        str(row["candidate_id"]),
    ))
    return [dict(row) for row in eligible[:int(maximum_candidates)]]
