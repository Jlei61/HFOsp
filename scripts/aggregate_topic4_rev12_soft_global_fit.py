#!/usr/bin/env python3
"""Aggregate the fit-only global continuous-field screen with soft mode scores."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_topic4_rev12_cascade_fit import (  # noqa: E402
    load_event_sensitivity_workers,
    patient_direction_contract,
)
from scripts.audit_topic4_rev12_soft_mode_target import (  # noqa: E402
    _atomic_json,
    _candidate_summary,
    _classifier_contract,
    _frozen_patient_reference,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
    _resolve,
    _sha256,
    _soft_score,
    _source_bundle,
    _worker_diagnostics,
)
from scripts.rescore_topic4_rev12_node_historical import DEFAULT_ARTIFACT_ROOT  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
    soft_topology_network_reproducibility,
)


def _metric(row: dict, axis: str) -> float:
    name, direction = axis.split(":")
    value = float(row[name])
    return value if direction == "min" else -value


def _selection_evaluable(row: dict, axes: list[str]) -> bool:
    if not row.get("fit_valid", False):
        return False
    if not row.get("candidate", {}).get("selection_eligible", True):
        return False
    return bool(np.all(np.isfinite([
        _metric(row, axis) for axis in axes
    ])))


def _pareto_ids(rows: list[dict], axes: list[str]) -> list[str]:
    valid = [row for row in rows if _selection_evaluable(row, axes)]
    selected = []
    for row in valid:
        values = np.asarray([_metric(row, axis) for axis in axes], float)
        dominated = False
        for other in valid:
            if other is row:
                continue
            comparison = np.asarray([_metric(other, axis) for axis in axes], float)
            if np.all(comparison <= values) and np.any(comparison < values):
                dominated = True
                break
        if not dominated:
            selected.append(row["candidate_id"])
    return selected


def _radius(row: dict) -> float | None:
    coordinates = row["candidate"]["node_field"].get("residual_coordinates", {})
    value = coordinates.get("radius")
    return None if value is None else float(value)


def _nominate(rows: list[dict], selection: dict) -> dict:
    axes = list(selection["axes"])
    valid = [row for row in rows if _selection_evaluable(row, axes)]
    if not valid:
        return {"status": "NO_VALID_GLOBAL_FIELD", "candidate_ids": []}
    pareto = set(_pareto_ids(valid, axes))
    metric_matrix = np.asarray([
        [_metric(row, axis) for axis in axes] for row in valid
    ], float)
    low = np.min(metric_matrix, axis=0)
    span = np.ptp(metric_matrix, axis=0)
    span[span <= 1e-12] = 1.0
    for row, values in zip(valid, metric_matrix):
        row["pareto_normalized_regret"] = float(np.mean((values - low) / span))
        row["pareto_member"] = row["candidate_id"] in pareto
    best_soft = min(valid, key=lambda row: row["mean_soft_objective"])["candidate_id"]
    best_direction = max(
        valid, key=lambda row: row["mean_soft_causal_direction"],
    )["candidate_id"]
    maximum = int(selection["maximum_nominees"])
    maximum_per_radius = int(selection.get("maximum_per_radius", maximum))
    nominated = []
    radius_counts: dict[str, int] = {}

    def add(candidate_id: str) -> None:
        if candidate_id in nominated or len(nominated) >= maximum:
            return
        row = next(item for item in valid if item["candidate_id"] == candidate_id)
        radius = _radius(row)
        key = "sentinel" if radius is None else f"{radius:.6f}"
        if radius is not None and radius_counts.get(key, 0) >= maximum_per_radius:
            return
        nominated.append(candidate_id)
        radius_counts[key] = radius_counts.get(key, 0) + 1

    add(best_soft)
    add(best_direction)
    ranked = sorted(valid, key=lambda row: (
        not row["pareto_member"], row["pareto_normalized_regret"],
        row["candidate_id"],
    ))
    for row in ranked:
        add(row["candidate_id"])
    return {
        "status": "GLOBAL_FIT_NOMINEES_FROZEN_FROM_PREDECLARED_PARETO_RULE",
        "candidate_ids": nominated,
        "best_soft_objective": best_soft,
        "best_causal_direction": best_direction,
        "pareto_candidate_ids": sorted(pareto),
        "axes": axes,
        "maximum_nominees": maximum,
        "maximum_per_radius": maximum_per_radius,
        "patient_heldout_used": False,
        "natural_kmeans_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--seed-pool", choices=("fit",), default="fit")
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"global aggregate input changed: {record['path']}")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("global screen manifest is stale")

    cohort = json.loads(_resolve(
        artifact_root, config["inputs"]["cohort_config"]["path"],
    ).read_text())
    classifier_config = json.loads(_resolve(
        artifact_root, config["inputs"]["classifier_config"]["path"],
    ).read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    objective = config["soft_objective"]
    patient_reference = _frozen_patient_reference(
        patient, per_mode=int(objective["patient_reference_events_per_mode"]),
        seed=int(objective["projection_seed"]),
    )
    patient_reference["contact_names"] = patient["contact_names"]
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]),
        n_directions=int(objective["projection_count"]),
        seed=int(objective["projection_seed"]),
    )
    calibration = calibrate_component_scales(
        patient["train_ranks"], patient["train_labels"], patient["train_blocks"],
        patient["contact_names"], projections,
        sample_size=int(objective["calibration_sample_size"]),
        draws=int(objective["calibration_draws"]),
        seed=int(objective["calibration_seed"]),
    )
    seeds = [int(value) for value in config["search"]["fit_network_seeds"]]
    output_root = artifact_root / config["output_root"]
    first = output_root / "workers" / (
        f"{manifest['candidates'][0]['candidate_id']}_seed_{seeds[0]}.npz"
    )
    with np.load(first, allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        xy = np.asarray(loaded["contact_xy_mm"], float)
    order = [int(np.flatnonzero(names == name)[0]) for name in patient["contact_names"]]
    direction = patient_direction_contract(
        patient["train_ranks"], patient["train_labels"], xy[order],
    )

    rows = []
    for candidate in manifest["candidates"]:
        per_network, topology_maps, topology_probabilities = [], [], []
        invalid_reasons = []
        for seed in seeds:
            npz_path = output_root / "workers" / (
                f"{candidate['candidate_id']}_seed_{seed}.npz"
            )
            json_path = npz_path.with_suffix(".json")
            if not npz_path.exists() or not json_path.exists():
                invalid_reasons.append(f"missing_seed_{seed}")
                continue
            payload = json.loads(json_path.read_text())
            if payload["simulation"].get("runaway_early_stop_ms") is not None:
                invalid_reasons.append(f"runaway_seed_{seed}")
            worker = _load_network_worker(
                npz_path, patient["contact_names"], classifier,
                semantics["raw_to_patient"],
            )
            if not len(worker["ranks"]):
                invalid_reasons.append(f"zero_returned_seed_{seed}")
                continue
            maps, probabilities = _source_bundle(npz_path, worker)
            topology_maps.append(maps)
            topology_probabilities.append(probabilities)
            diagnostic = _worker_diagnostics(
                worker, npz_path, patient_reference, projections,
                calibration, objective, direction,
                seed=int(objective["projection_seed"]) + seed,
            )
            sensitivity = load_event_sensitivity_workers(
                npz_path, patient["contact_names"], classifier,
                semantics["raw_to_patient"],
            )
            sensitivity_scores = []
            for variant in sensitivity:
                current = variant["worker"]
                if len(current["ranks"]):
                    sensitivity_scores.append(_soft_score(
                        current["ranks"], current["probability_B"],
                        patient_reference, projections, calibration, objective,
                    )["objective"])
            diagnostic["event_sensitivity"] = {
                "n_evaluable_variants": len(sensitivity_scores),
                "objective_mean": (
                    float(np.mean(sensitivity_scores)) if sensitivity_scores else None
                ),
                "objective_sd": (
                    float(np.std(sensitivity_scores)) if sensitivity_scores else None
                ),
                "objective_max": (
                    float(np.max(sensitivity_scores)) if sensitivity_scores else None
                ),
                "selection_role": "diagnostic_only_primary_event_unit_ranks_screen",
            }
            diagnostic["ood_fraction"] = (
                float(np.mean(worker["ood"])) if len(worker["ood"]) else 1.0
            )
            diagnostic["compound_fraction"] = float(
                payload["event_unit"]["compound_detector_fragment_fraction"]
            )
            per_network.append(diagnostic)
        row = _candidate_summary(candidate, per_network) if per_network else {
            "candidate_id": candidate["candidate_id"],
            "role": candidate.get("role"),
            "field_sha256": candidate["node_field"]["field_sha256"],
            "n_networks": 0,
        }
        row["candidate"] = candidate
        row["fit_valid"] = not invalid_reasons and len(per_network) == len(seeds)
        row["invalid_reasons"] = invalid_reasons
        if per_network:
            topology = soft_topology_network_reproducibility(
                topology_maps, topology_probabilities,
            )
            row["soft_source_topology"] = topology
            row["soft_topology_across_network"] = topology[
                "mean_across_network_template_cosine"
            ]
            row["soft_topology_mode_separation"] = topology[
                "equal_network_between_mode_distance"
            ]
            sensitivity_sds = [
                item["event_sensitivity"]["objective_sd"] for item in per_network
                if item["event_sensitivity"]["objective_sd"] is not None
            ]
            row["mean_event_sensitivity_objective_sd"] = (
                float(np.mean(sensitivity_sds)) if sensitivity_sds else None
            )
            row["mean_ood_fraction"] = float(np.mean([
                item["ood_fraction"] for item in per_network
            ]))
            row["mean_compound_fraction"] = float(np.mean([
                item["compound_fraction"] for item in per_network
            ]))
        else:
            for key in (
                "mean_soft_objective", "mean_soft_causal_direction",
                "mean_soft_causal_monotonicity", "soft_topology_across_network",
                "soft_topology_mode_separation",
            ):
                row[key] = float("nan")
        row["selection_evaluable"] = _selection_evaluable(
            row, list(config["pareto_selection"]["axes"]),
        )
        rows.append(row)
    decision = _nominate(rows, config["pareto_selection"])
    rows.sort(key=lambda row: (
        not row["fit_valid"], row.get("pareto_normalized_regret", float("inf")),
        row["candidate_id"],
    ))
    payload = {
        "schema_id": "topic4_rev12_nd_global_soft_field_fit_aggregate_v1",
        "status": (
            "REV12ND_GLOBAL_SOFT_FIELD_FIT_COMPLETE"
            if decision["candidate_ids"] else "REV12ND_GLOBAL_SOFT_FIELD_NO_NOMINEE"
        ),
        "scientific_role": config["scientific_role"],
        "requested_seeds": seeds,
        "rows": rows,
        "nomination_decision": decision,
        "selection_contract": {
            "soft_objective": objective,
            "pareto": config["pareto_selection"],
            "patient_data": "training only",
            "natural_kmeans": "diagnostic only and absent from nomination code",
            "patient_heldout": "not scored",
            "event_sensitivity": "diagnostic only; primary causal-family unit ranks fit",
        },
        "inputs": {
            "config": str(config_path), "config_sha256": _sha256(config_path),
            "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    aggregate = output_root / "aggregate"
    _atomic_json(aggregate / "fit_soft_global_summary.json", payload)
    _atomic_json(aggregate / "fit_nomination_decision.json", decision)
    aggregate.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id", "role", "fit_valid", "selection_evaluable", "n_networks",
        "mean_soft_objective", "mean_soft_mode_0", "mean_soft_mode_1",
        "mean_soft_causal_direction", "mean_soft_causal_monotonicity",
        "soft_topology_across_network", "soft_topology_mode_separation",
        "mean_ambiguity", "mean_contrast_alignment", "mean_ood_fraction",
        "mean_compound_fraction", "mean_event_sensitivity_objective_sd",
        "pareto_member", "pareto_normalized_regret",
    ]
    with (aggregate / "fit_soft_global_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(rows),
        "nominees": decision["candidate_ids"], "output": str(aggregate),
    }, indent=2))


if __name__ == "__main__":
    main()
