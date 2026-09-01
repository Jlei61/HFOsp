#!/usr/bin/env python3
"""Build the final Node and frozen-pathway analysis from completed phases."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic4_rev10_sa_shaft_aware_target import (  # noqa: E402
    _calibrated_mode_score,
    _distance_row,
    _flatten_distances,
)
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    natural_kmeans,
    normalize_event_ranks,
)
from src.topic4_dual_core_ood import load_embedding  # noqa: E402
from src.topic4_nlc_pathway_mechanism import (  # noqa: E402
    bootstrap_mean,
    factorial_bootstrap,
    paired_bootstrap,
)
from src.topic4_shaft_aware import (  # noqa: E402
    centered_smooth_max,
    contract_groups,
    contract_pairs,
)


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"
ARM_ORDER = (
    "frozen_dualcore_node",
    "frozen_dualcore_ee",
    "frozen_dualcore_etoi",
    "frozen_dualcore_both",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _patient_target(loaded, mode: int) -> dict[str, Any]:
    return {
        "descriptor": {
            "recruitment": {
                shaft: np.asarray(loaded[f"mode_{mode}_recruitment_{shaft.lower()}"], float)
                for shaft in ("ICL", "SCL")
            },
            "precedence": {
                key: np.asarray(loaded[f"mode_{mode}_precedence_{key.lower().replace('-', '_')}"], float)
                for key in ("ICL-ICL", "SCL-SCL", "ICL-SCL")
            },
            "profile": {
                "ICL": np.asarray(loaded[f"mode_{mode}_profile_icl"], float),
                "SCL": np.asarray(loaded[f"mode_{mode}_profile_scl"], float),
                "cross": np.asarray(loaded[f"mode_{mode}_profile_cross"], float),
            },
        },
        "reference_z": np.asarray(loaded[f"mode_{mode}_reference_z"], float),
    }


def _patient_bundle(target_path: Path) -> dict[str, Any]:
    with np.load(target_path, allow_pickle=False) as loaded:
        labels = np.asarray(loaded["patient_train_old_labels"], int)
        ranks = np.asarray(loaded["patient_train_ranks"], float)
        normalized = normalize_event_ranks(ranks)
        return {
            "contact_names": np.asarray(loaded["contact_names"]).astype(str),
            "profiles": np.asarray([
                np.nanmean(normalized[labels == mode], axis=0)
                for mode in (0, 1)
            ]),
            "mode_counts": np.bincount(labels, minlength=2),
            "mode_fraction": np.bincount(labels, minlength=2) / len(labels),
            "targets": {mode: _patient_target(loaded, mode) for mode in (0, 1)},
        }


def _profile_matrix(ranks: np.ndarray, labels: np.ndarray, patient: np.ndarray) -> np.ndarray:
    normalized = normalize_event_ranks(ranks)
    model = np.asarray([
        np.nanmean(normalized[labels == mode], axis=0)
        if np.any(labels == mode) else np.full(ranks.shape[1], np.nan)
        for mode in (0, 1)
    ])
    output = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            finite = np.isfinite(model[row]) & np.isfinite(patient[column])
            if np.sum(finite) >= 3:
                output[row, column] = float(spearmanr(
                    model[row, finite], patient[column, finite],
                ).statistic)
    return output


def _fixed_indices(indices: np.ndarray, count: int) -> np.ndarray | None:
    indices = np.asarray(indices, int)
    if len(indices) < count:
        return None
    local = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return indices[local]


def score_network_secondary(
    npz_path: Path, aggregate_row: dict[str, Any], *, patient: dict[str, Any],
    groups: dict[str, np.ndarray], pairs: dict[str, np.ndarray], embedding: dict[str, Any],
    floors: dict[str, Any], sa_config: dict[str, Any], fixed_count: int = 10,
) -> dict[str, Any]:
    events = aggregate_row["events"]
    labels = np.asarray([row["mode"] for row in events], int)
    support = np.asarray([row["in_support"] for row in events], bool)
    with np.load(npz_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["onsets"], float)
        ranks = np.asarray(loaded["ranks"], float)
    joint_shaft = (
        np.isfinite(onsets[:, groups["ICL"]]).any(axis=1)
        & np.isfinite(onsets[:, groups["SCL"]]).any(axis=1)
    )
    natural = natural_kmeans(
        ranks[support & joint_shaft], labels[support & joint_shaft],
        random_state=int(aggregate_row["seed"]),
    )
    modes, mode_scores = {}, []
    for mode in (0, 1):
        selected = _fixed_indices(np.flatnonzero(support & (labels == mode)), fixed_count)
        if selected is None:
            modes[str(mode)] = {
                "status": "INSUFFICIENT_FOR_SECONDARY_FIXED_COUNT",
                "n_available": int(np.sum(support & (labels == mode))),
                "n_required": int(fixed_count),
            }
            continue
        raw = _flatten_distances(_distance_row(
            onsets[selected], patient["targets"][mode], groups, pairs, embedding,
        ))
        objective = _calibrated_mode_score(
            raw, floors[str(fixed_count)][str(mode)], sa_config,
        )
        modes[str(mode)] = {
            "status": "OK", "n_available": int(np.sum(support & (labels == mode))),
            "n_scored": int(fixed_count), "selected_indices": selected,
            "raw": raw, "objective": objective,
        }
        mode_scores.append(objective)
    profile_matrix = _profile_matrix(
        ranks[support], labels[support], patient["profiles"],
    )
    counts = np.bincount(labels[support], minlength=2)
    output = {
        "natural_kmeans": {
            key: value for key, value in natural.items()
            if key not in {"valid_event_mask", "cluster_labels"}
        },
        "patient_profile_spearman_matrix": profile_matrix,
        "mode_counts_in_support": counts,
        "mode_fraction_in_support": counts / max(1, counts.sum()),
        "modes": modes,
        "secondary_distribution_status": (
            "OK" if len(mode_scores) == 2 else "INSUFFICIENT_MODE_SUPPORT"
        ),
    }
    if len(mode_scores) == 2:
        scalar = {
            key: float(np.mean([row[key] for row in mode_scores]))
            for key in ("recruitment", "precedence", "profile", "event_cloud")
        }
        output.update({
            **scalar,
            "weakest_mode_error": centered_smooth_max(
                [row["mode_score"] for row in mode_scores], 0.25,
            ),
        })
    return output


def _arm_rows(
    phase_aggregate: dict[str, Any], *, patient: dict[str, Any],
    groups: dict[str, np.ndarray], pairs: dict[str, np.ndarray], embedding: dict[str, Any],
    floors: dict[str, Any], sa_config: dict[str, Any], fixed_count: int,
) -> dict[str, dict[str, Any]]:
    output = {}
    for candidate in phase_aggregate["ranking"]:
        seed_rows = {}
        for row in candidate["per_network"]:
            secondary = score_network_secondary(
                ROOT / row["worker_npz"], row, patient=patient, groups=groups,
                pairs=pairs, embedding=embedding, floors=floors,
                sa_config=sa_config, fixed_count=fixed_count,
            )
            seed_rows[str(row["seed"])] = {**row, **secondary}
        output[candidate["candidate_id"]] = {
            "summary": candidate, "per_network": seed_rows,
        }
    return output


def _metric_vector(arm: dict[str, Any], seeds: list[int], metric: str) -> np.ndarray:
    return np.asarray([
        arm["per_network"][str(seed)].get(metric, np.nan) for seed in seeds
    ], float)


def _pathway_inference(
    arms: dict[str, dict[str, Any]], seeds: list[int], *, draws: int, seed: int,
) -> dict[str, Any]:
    metrics = {
        "ood_all_returned": "lower",
        "weakest_mode_error": "lower",
        "natural_kmeans_alignment": "higher",
        "mode_1_fraction": "descriptive",
        "returned_events": "higher",
    }
    values: dict[str, dict[str, np.ndarray]] = {metric: {} for metric in metrics}
    for arm_id in ARM_ORDER:
        arm = arms[arm_id]
        values["ood_all_returned"][arm_id] = _metric_vector(
            arm, seeds, "ood_all_returned",
        )
        values["weakest_mode_error"][arm_id] = _metric_vector(
            arm, seeds, "weakest_mode_error",
        )
        values["natural_kmeans_alignment"][arm_id] = np.asarray([
            arm["per_network"][str(network_seed)]["natural_kmeans"].get(
                "direction_balanced_alignment", np.nan,
            ) for network_seed in seeds
        ], float)
        values["mode_1_fraction"][arm_id] = np.asarray([
            arm["per_network"][str(network_seed)]["mode_fraction_in_support"][1]
            for network_seed in seeds
        ], float)
        values["returned_events"][arm_id] = _metric_vector(
            arm, seeds, "n_returned",
        )
    output = {"metrics": {}, "arm_order": list(ARM_ORDER)}
    node = ARM_ORDER[0]
    for metric_index, (metric, direction) in enumerate(metrics.items()):
        metric_output = {
            "direction": direction,
            "by_arm": {
                arm_id: bootstrap_mean(
                    values[metric][arm_id], draws=draws,
                    seed=seed + 100 * metric_index + arm_index,
                ) for arm_index, arm_id in enumerate(ARM_ORDER)
            },
            "paired_arm_minus_node": {
                arm_id: paired_bootstrap(
                    values[metric][arm_id], values[metric][node], draws=draws,
                    seed=seed + 1000 + 100 * metric_index + arm_index,
                ) for arm_index, arm_id in enumerate(ARM_ORDER[1:], start=1)
            },
            "factorial_interaction": factorial_bootstrap(
                values[metric][ARM_ORDER[0]], values[metric][ARM_ORDER[1]],
                values[metric][ARM_ORDER[2]], values[metric][ARM_ORDER[3]],
                draws=draws, seed=seed + 2000 + metric_index,
            ),
        }
        output["metrics"][metric] = metric_output
    return output


def analyze(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    confirmation_path = output_root / "confirmation/aggregate.json"
    pathway_path = output_root / "pathway/aggregate.json"
    confirmation = json.loads(confirmation_path.read_text())
    pathway = json.loads(pathway_path.read_text())
    if any(payload.get("status") != "DUAL_CORE_OOD_PHASE_COMPLETE"
           for payload in (confirmation, pathway)):
        raise RuntimeError("confirmation and pathway phases must both be complete")
    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text())
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    patient = _patient_bundle(target_path)
    embedding = load_embedding(str(target_path))
    floor_path = ROOT / config["inputs"]["shaft_aware_floors"]["path"]
    floors_payload = json.loads(floor_path.read_text())
    floors = floors_payload["full_timing_floors"]
    sa_config = json.loads((
        ROOT / config["inputs"]["shaft_aware_scoring_config"]["path"]
    ).read_text())
    fixed_count = 10
    confirmation_arms = _arm_rows(
        confirmation, patient=patient, groups=groups, pairs=pairs,
        embedding=embedding, floors=floors, sa_config=sa_config,
        fixed_count=fixed_count,
    )
    pathway_arms = _arm_rows(
        pathway, patient=patient, groups=groups, pairs=pairs,
        embedding=embedding, floors=floors, sa_config=sa_config,
        fixed_count=fixed_count,
    )
    seeds = sorted({
        int(seed) for arm in pathway_arms.values()
        for seed in arm["per_network"]
    })
    inference = _pathway_inference(
        pathway_arms, seeds,
        draws=int(config["search"]["paired_network_bootstrap"]["draws"]),
        seed=int(config["search"]["paired_network_bootstrap"]["seed"]),
    )
    node_id = confirmation["ranking"][0]["candidate_id"]
    node = confirmation_arms[node_id]
    node_rows = list(node["per_network"].values())
    output = {
        "status": "DUAL_CORE_OOD_NODE_AND_PATHWAYS_ANALYZED",
        "primary_metric": "OOD_all_returned; unreadable returned events count as OOD",
        "natural_kmeans_population": (
            "returned, readable, cross-shaft events inside frozen patient support only"
        ),
        "node_candidate_id": node_id,
        "node_field": node["summary"]["node_field"],
        "node_confirmation": {
            "n_networks": len(node_rows),
            "networks_with_both_modes": int(sum(
                row["both_modes_in_support"] for row in node_rows
            )),
            "ood_all_returned": bootstrap_mean(
                [row["ood_all_returned"] for row in node_rows], draws=4096,
                seed=20260830,
            ),
            "returned_events": bootstrap_mean(
                [row["n_returned"] for row in node_rows], draws=4096,
                seed=20260831,
            ),
            "natural_kmeans_alignment": bootstrap_mean([
                row["natural_kmeans"].get("direction_balanced_alignment", np.nan)
                for row in node_rows
            ], draws=4096, seed=20260832),
            "weakest_mode_error": bootstrap_mean([
                row.get("weakest_mode_error", np.nan) for row in node_rows
            ], draws=4096, seed=20260833),
            "per_network": node["per_network"],
        },
        "patient_training_mode_fraction": patient["mode_fraction"],
        "secondary_fixed_count_per_mode": fixed_count,
        "secondary_fixed_count_is_not_an_eligibility_gate": True,
        "pathway_arms": pathway_arms,
        "pathway_inference": inference,
        "inputs": {
            "confirmation_aggregate": str(confirmation_path.relative_to(ROOT)),
            "pathway_aggregate": str(pathway_path.relative_to(ROOT)),
            "patient_target": str(target_path.relative_to(ROOT)),
            "patient_floors": str(floor_path.relative_to(ROOT)),
        },
        "claim_boundary": config["claim_boundary"],
    }
    _atomic_json(output_root / "final_analysis.json", output)
    csv_path = output_root / "pathway_per_network.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "seed", "ood_all_returned", "n_returned",
            "mode_0_in_support", "mode_1_in_support",
            "natural_kmeans_alignment", "weakest_mode_error",
        ])
        writer.writeheader()
        for arm_id, arm in pathway_arms.items():
            for network_seed, row in arm["per_network"].items():
                writer.writerow({
                    "candidate_id": arm_id, "seed": network_seed,
                    "ood_all_returned": row["ood_all_returned"],
                    "n_returned": row["n_returned"],
                    "mode_0_in_support": row["mode_counts_in_support"][0],
                    "mode_1_in_support": row["mode_counts_in_support"][1],
                    "natural_kmeans_alignment": row["natural_kmeans"].get(
                        "direction_balanced_alignment"
                    ),
                    "weakest_mode_error": row.get("weakest_mode_error"),
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output = analyze(args.config)
    print(json.dumps({
        "status": output["status"],
        "node_candidate_id": output["node_candidate_id"],
        "node_ood": output["node_confirmation"]["ood_all_returned"],
    }, indent=2))


if __name__ == "__main__":
    main()
