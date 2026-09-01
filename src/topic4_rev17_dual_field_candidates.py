"""Nominate and reconstruct bounded rev17 dual-field selection candidates."""
from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from src.topic4_node_field_search import cosine_sheet_residuals, residual_candidate
from src.topic4_rev17_dual_field_residual import mapping_sha256


AGGREGATE_STATUS = "REV17_DUAL_FIELD_RESIDUAL_ATLAS_AGGREGATE_COMPLETE"
CHANNELS = ("mean", "dispersion")


def _validate_nomination_boundary(aggregate: Mapping[str, Any]) -> None:
    if aggregate.get("status") != AGGREGATE_STATUS:
        raise RuntimeError("rev17 response aggregate is incomplete")
    if not aggregate.get("inventory", {}).get("complete_cartesian_product"):
        raise RuntimeError("rev17 response atlas Cartesian product is incomplete")
    boundaries = aggregate.get("boundaries", {})
    if (
        any(boundaries.get(key) is not False for key in (
            "natural_kmeans_used", "patient_heldout_used", "ictal_data_used",
            "figure_used",
        ))
        or boundaries.get("EE_EtoI_ZM") != "off"
    ):
        raise RuntimeError("rev17 candidate nomination crossed a forbidden boundary")


def _discrete_nomination_blueprint(
    aggregate: Mapping[str, Any], contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Nominate measured coordinates when the event response is not linear."""
    rules = contract.get("discrete_response_fallback") or {}
    if rules.get("enabled") is not True:
        raise RuntimeError("rev17 atlas has no locally linear direction and no fallback")
    tensor = aggregate.get("response_tensor") or {}
    seeds = [int(value) for value in tensor.get("network_seeds", [])]
    if len(seeds) != 3:
        raise RuntimeError("rev17 discrete nomination requires three fit networks")
    amplitude = float(tensor.get("amplitude", 0.0))
    if amplitude <= 0.0:
        raise RuntimeError("rev17 discrete nomination amplitude is invalid")
    rows = list(aggregate.get("scored_runs") or [])
    lookup = {
        (str(row["candidate_id"]), int(row["seed"])): row for row in rows
    }
    if len(lookup) != len(rows):
        raise RuntimeError("rev17 scored-run identity is duplicated")
    anchors = [lookup.get(("exact_dual_anchor", seed)) for seed in seeds]
    if any(row is None for row in anchors):
        raise RuntimeError("rev17 discrete nomination lacks paired anchors")
    candidate_ids = sorted({
        str(row["candidate_id"]) for row in rows
        if row.get("candidate_id") != "exact_dual_anchor"
    })
    b_ratio = float(rules["B_protection_ratio"])
    support_minimum = float(rules["minimum_effective_support_per_mode"])
    maximum_candidates = int(rules["maximum_candidates"])
    proposals = []
    for source_id in candidate_ids:
        candidate_rows = [lookup.get((source_id, seed)) for seed in seeds]
        if any(row is None for row in candidate_rows):
            raise RuntimeError(f"rev17 discrete candidate is incomplete: {source_id}")
        coordinate_identity = {
            (str(row["channel"]), int(row["mode_index"]), int(row["orientation"]))
            for row in candidate_rows
        }
        if len(coordinate_identity) != 1:
            raise RuntimeError("rev17 discrete coordinate changed across networks")
        channel, mode_index, orientation = next(iter(coordinate_identity))
        if (
            channel not in CHANNELS or not 0 <= mode_index < 15
            or orientation not in (-1, 1)
        ):
            raise RuntimeError("rev17 discrete coordinate identity is invalid")
        direction = np.zeros(30, dtype=float)
        direction[CHANNELS.index(channel) * 15 + mode_index] = float(orientation)
        changes = {
            "J14": np.asarray([
                float(row["j14"]) - float(anchor["j14"])
                for row, anchor in zip(candidate_rows, anchors)
            ]),
            "A": np.asarray([
                float(row["mode_0_mean"]) - float(anchor["mode_0_mean"])
                for row, anchor in zip(candidate_rows, anchors)
            ]),
            "B": np.asarray([
                float(row["mode_1_mean"]) - float(anchor["mode_1_mean"])
                for row, anchor in zip(candidate_rows, anchors)
            ]),
        }
        support_a = np.asarray([
            float(row["mode_0_effective_events"]) for row in candidate_rows
        ])
        support_b = np.asarray([
            float(row["mode_1_effective_events"]) for row in candidate_rows
        ])
        gates = {
            "valid_all_networks": all(
                row.get("run_status") == "VALID" for row in candidate_rows
            ),
            "J14_improves_all_networks": bool(np.all(changes["J14"] < 0.0)),
            "A_improves_all_networks": bool(np.all(changes["A"] < 0.0)),
            "B_within_ratio_all_networks": bool(np.all([
                float(row["mode_1_mean"])
                <= b_ratio * float(anchor["mode_1_mean"]) + 1e-12
                for row, anchor in zip(candidate_rows, anchors)
            ])),
            "A_support_at_least_minimum_all_networks": bool(np.all(
                support_a >= support_minimum
            )),
            "B_support_at_least_minimum_all_networks": bool(np.all(
                support_b >= support_minimum
            )),
        }
        proposals.append({
            "candidate_id": f"direct_{source_id}",
            "source_candidate_id": source_id,
            "family": "discrete_single_coordinate",
            "nomination_strategy": "measured_antithetic_endpoint",
            "radius": amplitude,
            "direction": direction.tolist(),
            "fit_observed_changes": {
                name: value.tolist() for name, value in changes.items()
            },
            "fit_observed_support": {
                "A": support_a.tolist(), "B": support_b.tolist(),
            },
            "fit_observed_gates": gates,
            "predicted_advancement_eligible": bool(all(gates.values())),
            "worst_delta_J14": float(np.max(changes["J14"])),
            "worst_delta_A": float(np.max(changes["A"])),
            "mean_delta_J14": float(np.mean(changes["J14"])),
            "mean_delta_A": float(np.mean(changes["A"])),
            "minimum_support": float(min(np.min(support_a), np.min(support_b))),
        })
    eligible = [row for row in proposals if row["predicted_advancement_eligible"]]
    eligible.sort(key=lambda row: (
        row["worst_delta_J14"], row["worst_delta_A"],
        row["mean_delta_J14"], row["mean_delta_A"], row["candidate_id"],
    ))
    nominated = eligible[:maximum_candidates]
    return nominated, {
        "nomination_strategy": "measured_antithetic_endpoint",
        "fit_network_seeds": seeds,
        "source_amplitude": amplitude,
        "minimum_effective_support_per_mode": support_minimum,
        "B_protection_ratio": b_ratio,
        "maximum_candidates": maximum_candidates,
        "eligible_before_cap": len(eligible),
        "nominated_candidate_ids": [row["candidate_id"] for row in nominated],
        "nominated_candidate_count": len(nominated),
        "all_proposals": proposals,
        "natural_kmeans_used": False,
        "patient_heldout_used": False,
        "EE_EtoI_ZM": "off",
    }


def _unit_direction(record: Mapping[str, Any], dimension: int) -> np.ndarray | None:
    raw = record.get("direction")
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float)
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise RuntimeError("rev17 robust direction is malformed")
    if not np.isclose(np.linalg.norm(values), 1.0, rtol=0.0, atol=1e-7):
        raise RuntimeError("rev17 robust direction is not unit norm")
    return values


def nomination_blueprint(
    aggregate: Mapping[str, Any], contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the predeclared predicted advancement rule to local directions."""
    _validate_nomination_boundary(aggregate)
    tensor = aggregate.get("response_tensor") or {}
    gradients = {
        name: np.asarray(values, dtype=float)
        for name, values in (tensor.get("gradients") or {}).items()
    }
    required = ("A", "B", "J14", "support_A", "support_B")
    if set(gradients) != set(required):
        raise RuntimeError("rev17 response gradient inventory changed")
    if {value.shape for value in gradients.values()} != {(3, 30)}:
        raise RuntimeError("rev17 response gradients are not 3 by 30")
    anchor_rows = sorted(
        [
            row for row in aggregate.get("scored_runs", [])
            if row.get("candidate_id") == "exact_dual_anchor"
        ],
        key=lambda row: int(row["seed"]),
    )
    if [int(row["seed"]) for row in anchor_rows] != list(tensor["network_seeds"]):
        raise RuntimeError("rev17 anchor rows do not match response networks")
    anchor = {
        "A": np.asarray([row["mode_0_mean"] for row in anchor_rows], dtype=float),
        "B": np.asarray([row["mode_1_mean"] for row in anchor_rows], dtype=float),
        "J14": np.asarray([row["j14"] for row in anchor_rows], dtype=float),
        "support_A": np.asarray(
            [row["mode_0_effective_events"] for row in anchor_rows], dtype=float,
        ),
        "support_B": np.asarray(
            [row["mode_1_effective_events"] for row in anchor_rows], dtype=float,
        ),
    }
    rules = contract["direction_construction"]
    radii = [float(value) for value in rules["candidate_radii"]]
    if not radii or any(value <= 0.0 for value in radii):
        raise RuntimeError("rev17 candidate radii are invalid")
    families = list(rules["families"])
    b_ratio = float(rules["predicted_B_protection_ratio"])
    support_minimum = float(rules["predicted_minimum_support_per_mode"])
    directions = aggregate.get("robust_directions") or {}
    if directions.get("analysis_status") == "NO_LOCALLY_LINEAR_COORDINATE":
        return _discrete_nomination_blueprint(aggregate, contract)
    nominated, audit = [], []
    for family in families:
        record = directions.get(family) or {}
        direction = _unit_direction(record, 30)
        if direction is None:
            audit.append({"family": family, "status": "DIRECTION_INFEASIBLE"})
            continue
        changes = {name: gradients[name] @ direction for name in required}
        for radius in radii:
            predicted = {name: anchor[name] + radius * changes[name] for name in required}
            gates = {
                "J14_improves_all_networks": bool(np.all(radius * changes["J14"] < 0.0)),
                "A_improves_all_networks": bool(np.all(radius * changes["A"] < 0.0)),
                "B_within_ratio_all_networks": bool(np.all(
                    predicted["B"] <= b_ratio * anchor["B"] + 1e-12
                )),
                "A_support_at_least_minimum_all_networks": bool(np.all(
                    predicted["support_A"] >= support_minimum
                )),
                "B_support_at_least_minimum_all_networks": bool(np.all(
                    predicted["support_B"] >= support_minimum
                )),
            }
            eligible = bool(all(gates.values()))
            candidate_id = f"dual_{family}_r{int(round(100 * radius)):02d}"
            proposal = {
                "candidate_id": candidate_id,
                "family": family,
                "radius": radius,
                "direction": direction.tolist(),
                "predicted_changes": {
                    name: (radius * changes[name]).tolist() for name in required
                },
                "predicted_absolute": {
                    name: predicted[name].tolist() for name in required
                },
                "predicted_gates": gates,
                "predicted_advancement_eligible": eligible,
            }
            audit.append(proposal)
            if eligible:
                nominated.append(proposal)
    if len({row["candidate_id"] for row in nominated}) != len(nominated):
        raise RuntimeError("rev17 nominated candidate identifiers are duplicated")
    return nominated, {
        "fit_network_seeds": list(tensor["network_seeds"]),
        "families": families,
        "candidate_radii": radii,
        "nominated_candidate_ids": [row["candidate_id"] for row in nominated],
        "nominated_candidate_count": len(nominated),
        "all_proposals": audit,
        "natural_kmeans_used": False,
        "patient_heldout_used": False,
        "EE_EtoI_ZM": "off",
    }


def build_candidates(
    atlas_manifest: Mapping[str, Any], blueprints: list[Mapping[str, Any]], *,
    maximum_frequency: int, target_n_basis: int, degree: int,
    sheet_mm: float, projection_grid_per_axis: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map nominated 30-D directions back to the accepted two spline fields."""
    anchors = [
        row for row in atlas_manifest.get("candidates", [])
        if row.get("candidate_id") == "exact_dual_anchor"
    ]
    if len(anchors) != 1:
        raise RuntimeError("rev17 atlas lacks one exact dual anchor")
    anchor = copy.deepcopy(anchors[0])
    anchor["role"] = "exact_dual_anchor_fresh_network_reference"
    anchor["selection_eligible"] = False
    basis = cosine_sheet_residuals(
        maximum_frequency=int(maximum_frequency),
        target_n_basis=int(target_n_basis), degree=int(degree),
        sheet_mm=float(sheet_mm),
        projection_grid_per_axis=int(projection_grid_per_axis),
    )
    modes = [np.asarray(row["coefficients"], dtype=float) for row in basis["rows"]]
    if len(modes) != 15:
        raise RuntimeError("rev17 selection basis no longer has 15 spatial modes")
    candidates = [anchor]
    audits = []
    atlas_candidates = {
        str(row["candidate_id"]): row
        for row in atlas_manifest.get("candidates", [])
    }
    for blueprint in blueprints:
        direction = np.asarray(blueprint["direction"], dtype=float)
        radius = float(blueprint["radius"])
        if direction.shape != (30,):
            raise RuntimeError("rev17 selection direction has wrong dimension")
        source_id = blueprint.get("source_candidate_id")
        if source_id is not None:
            source = atlas_candidates.get(str(source_id))
            if source is None:
                raise RuntimeError("rev17 discrete source candidate is absent")
            candidate = copy.deepcopy(source)
            candidate.update({
                "candidate_id": blueprint["candidate_id"],
                "role": (
                    "dual_continuous_node_measured_endpoint_selection_candidate"
                ),
                "selection_eligible": True,
                "residual_coordinates": {
                    **copy.deepcopy(source.get("residual_coordinates") or {}),
                    "family": blueprint["family"],
                    "nomination_strategy": "measured_antithetic_endpoint",
                    "source_candidate_id": source_id,
                    "joint_dual_field_radius": radius,
                    "direction": direction.tolist(),
                    "mean_direction_l2": float(np.linalg.norm(direction[:15])),
                    "dispersion_direction_l2": float(np.linalg.norm(direction[15:])),
                    "observation_coordinates_used": False,
                    "zero_is_exact_dual_anchor": True,
                },
                "predicted_response": copy.deepcopy(dict(blueprint)),
                "pathways": copy.deepcopy(anchor["pathways"]),
            })
            candidates.append(candidate)
            audits.append({
                "candidate_id": candidate["candidate_id"],
                "mapping_sha256": candidate["node_mapping"]["mapping_sha256"],
                "mean_field_sha256": candidate["node_field"]["field_sha256"],
                "dispersion_field_sha256": candidate[
                    "node_dispersion_field"
                ]["field_sha256"],
                "joint_dual_field_radius": radius,
                "mean_direction_l2": float(np.linalg.norm(direction[:15])),
                "dispersion_direction_l2": float(np.linalg.norm(direction[15:])),
                "source_candidate_id": source_id,
                "reproduces_source_candidate": True,
            })
            continue
        fields = {}
        channel_norms = {}
        for channel_index, (channel, source_key) in enumerate((
            ("mean", "node_field"), ("dispersion", "node_dispersion_field"),
        )):
            weights = direction[channel_index * 15:(channel_index + 1) * 15]
            channel_norm = float(np.linalg.norm(weights))
            channel_norms[channel] = channel_norm
            if channel_norm <= 1e-12:
                fields[source_key] = copy.deepcopy(anchor[source_key])
                continue
            residual = sum(weight * mode for weight, mode in zip(weights, modes))
            field = residual_candidate(
                anchor[source_key], residual, amplitude=radius,
                candidate_id=f"{blueprint['candidate_id']}_{channel}",
                residual_index=-1, coarse_n_basis=int(maximum_frequency) + 1,
            )
            field["role"] = f"rev17_joint_{channel}_continuous_residual"
            field["residual_coordinates"].update({
                "basis_family": "uniform_sheet_cosine",
                "channel": channel,
                "joint_direction": True,
                "direction_weights": weights.tolist(),
                "channel_direction_l2": channel_norm,
                "channel_sheet_rms": radius * channel_norm,
                "joint_dual_field_radius": radius,
                "observation_coordinates_used": False,
            })
            fields[source_key] = field
        mapping_hash = mapping_sha256(
            fields["node_field"]["field_sha256"],
            fields["node_dispersion_field"]["field_sha256"],
        )
        candidate = {
            "candidate_id": blueprint["candidate_id"],
            "role": "dual_continuous_node_response_nominated_selection_candidate",
            "selection_eligible": True,
            "source_candidate_ids": copy.deepcopy(anchor.get("source_candidate_ids", {})),
            **fields,
            "node_mapping": {
                "mapping_type": "dual_continuous_mean_dispersion",
                "signed_depth_shrinkage": 1.0,
                "node_gain": 1.0,
                "mapping_sha256": mapping_hash,
            },
            "residual_coordinates": {
                "family": blueprint["family"],
                "nomination_strategy": blueprint.get(
                    "nomination_strategy", "locally_linear_direction"
                ),
                "source_candidate_id": blueprint.get("source_candidate_id"),
                "joint_dual_field_radius": radius,
                "direction": direction.tolist(),
                "mean_direction_l2": channel_norms["mean"],
                "dispersion_direction_l2": channel_norms["dispersion"],
                "observation_coordinates_used": False,
                "zero_is_exact_dual_anchor": True,
            },
            "predicted_response": copy.deepcopy(dict(blueprint)),
            "pathways": copy.deepcopy(anchor["pathways"]),
        }
        candidates.append(candidate)
        audits.append({
            "candidate_id": candidate["candidate_id"],
            "mapping_sha256": mapping_hash,
            "mean_field_sha256": fields["node_field"]["field_sha256"],
            "dispersion_field_sha256": fields["node_dispersion_field"]["field_sha256"],
            "joint_dual_field_radius": radius,
            "mean_direction_l2": channel_norms["mean"],
            "dispersion_direction_l2": channel_norms["dispersion"],
            "source_candidate_id": None,
            "reproduces_source_candidate": None,
        })
    mapping_hashes = [row["node_mapping"]["mapping_sha256"] for row in candidates]
    if len(mapping_hashes) != len(set(mapping_hashes)):
        raise RuntimeError("rev17 selection candidates contain duplicate mappings")
    return candidates, {
        "candidate_count_including_anchor": len(candidates),
        "selection_candidate_count": len(candidates) - 1,
        "candidate_audit": audits,
        "maximum_absolute_basis_gram_error": basis["maximum_absolute_gram_error"],
    }
