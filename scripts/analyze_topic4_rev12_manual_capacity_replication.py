#!/usr/bin/env python3
"""Close out the non-selectable Stage-AD Node directional-capacity control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from scripts.analyze_topic4_rev12_global_soft_field_expansion import (  # noqa: E402
    bootstrap_mean_interval,
    paired_utilities,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _rows(summary: dict) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in summary["rows"]}


def capacity_audit(capacity: dict, comparisons: dict[str, dict], *,
                   description: dict, draws: int, confidence: float,
                   seed: int) -> dict:
    selection_eligible = capacity.get(
        "selection_eligible",
        capacity.get("candidate", {}).get("selection_eligible", True),
    )
    if selection_eligible:
        raise RuntimeError("manual capacity control became selection eligible")
    workers = sorted(capacity["per_network"], key=lambda row: int(row["seed"]))
    minimum_events = float(description["minimum_effective_events_per_soft_mode"])
    direction_threshold = float(description["direction_sign_threshold"])
    monotonicity_threshold = float(description["monotonicity_sign_threshold"])
    per_network = []
    for row in workers:
        modes = row["soft_objective"]["modes"]
        km = row.get("natural_kmeans_final_validation_diagnostic", {})
        record = {
            "seed": int(row["seed"]),
            "n_events": int(row["n_events"]),
            "soft_objective": float(row["soft_objective"]["objective"]),
            "soft_mode_0": float(modes["0"]["mean"]),
            "soft_mode_1": float(modes["1"]["mean"]),
            "mode_0_effective_events": float(modes["0"]["effective_events"]),
            "mode_1_effective_events": float(modes["1"]["effective_events"]),
            "mode_0_soft_occupancy": float(modes["0"]["soft_occupancy"]),
            "mode_1_soft_occupancy": float(modes["1"]["soft_occupancy"]),
            "causal_direction": float(row["soft_causal_direction"]["score"]),
            "causal_monotonicity": float(
                row["soft_causal_monotonicity"]["score"]
            ),
            "ood_fraction": float(row["ood_fraction"]),
            "natural_kmeans_status": km.get("status"),
            "natural_kmeans_cluster_counts": km.get("cluster_counts"),
            "natural_kmeans_direction_balanced_alignment": km.get(
                "direction_balanced_alignment"
            ),
            "natural_kmeans_seed_ami_median": km.get("kmeans_seed_ami_median"),
        }
        record["both_soft_modes_supported"] = bool(
            record["mode_0_effective_events"] >= minimum_events
            and record["mode_1_effective_events"] >= minimum_events
        )
        record["direction_positive"] = bool(
            record["causal_direction"] > direction_threshold
        )
        record["monotonicity_positive"] = bool(
            record["causal_monotonicity"] > monotonicity_threshold
        )
        per_network.append(record)

    n_networks = len(per_network)
    counts = {
        "both_soft_modes_supported": sum(
            row["both_soft_modes_supported"] for row in per_network
        ),
        "direction_positive": sum(row["direction_positive"] for row in per_network),
        "monotonicity_positive": sum(
            row["monotonicity_positive"] for row in per_network
        ),
        "natural_kmeans_k2_evaluable": sum(
            row["natural_kmeans_status"] == "OK"
            and isinstance(row["natural_kmeans_cluster_counts"], list)
            and len(row["natural_kmeans_cluster_counts"]) == 2
            and min(row["natural_kmeans_cluster_counts"]) > 0
            for row in per_network
        ),
        "natural_kmeans_direction_alignment_above_chance": sum(
            row["natural_kmeans_direction_balanced_alignment"] is not None
            and row["natural_kmeans_direction_balanced_alignment"] > 0.5
            for row in per_network
        ),
    }
    full_support = bool(
        n_networks > 0
        and counts["both_soft_modes_supported"] == n_networks
        and counts["direction_positive"] == n_networks
        and counts["monotonicity_positive"] == n_networks
    )
    majority_support = bool(
        n_networks > 0
        and all(counts[key] >= int(np.ceil(2 * n_networks / 3)) for key in (
            "both_soft_modes_supported", "direction_positive",
            "monotonicity_positive",
        ))
    )
    status = (
        "NODE_ONLY_DIRECTIONAL_CAPACITY_POSITIVE_PATIENT_JOINT_RECOVERY_UNRESOLVED"
        if full_support else
        "NODE_ONLY_DIRECTIONAL_CAPACITY_PARTIAL_PATIENT_JOINT_RECOVERY_UNRESOLVED"
        if majority_support else "NODE_ONLY_DIRECTIONAL_CAPACITY_UNRESOLVED"
    )

    paired = {}
    for comparison_index, (candidate_id, candidate) in enumerate(comparisons.items()):
        endpoints = {
            name: bootstrap_mean_interval(
                values, draws=draws, confidence=confidence,
                seed=seed + 1000 * comparison_index + endpoint_index,
            )
            for endpoint_index, (name, values) in enumerate(
                paired_utilities(candidate, capacity).items()
            )
        }
        paired[candidate_id] = {
            "orientation": (
                "positive utility means the manual capacity control is better"
            ),
            "endpoints": endpoints,
            "aggregate_topology_delta": {
                "across_network": float(
                    capacity["soft_topology_across_network"]
                    - candidate["soft_topology_across_network"]
                ),
                "mode_separation": float(
                    capacity["soft_topology_mode_separation"]
                    - candidate["soft_topology_mode_separation"]
                ),
            },
        }
    return {
        "status": status,
        "n_networks": n_networks,
        "capacity_support_counts": counts,
        "full_network_sign_support": full_support,
        "two_thirds_network_sign_support": majority_support,
        "minimums": {
            "mode_0_effective_events": min(
                row["mode_0_effective_events"] for row in per_network
            ),
            "mode_1_effective_events": min(
                row["mode_1_effective_events"] for row in per_network
            ),
            "causal_direction": min(row["causal_direction"] for row in per_network),
            "causal_monotonicity": min(
                row["causal_monotonicity"] for row in per_network
            ),
        },
        "means": {
            "soft_objective": float(capacity["mean_soft_objective"]),
            "soft_mode_0": float(capacity["mean_soft_mode_0"]),
            "soft_mode_1": float(capacity["mean_soft_mode_1"]),
            "causal_direction": float(capacity["mean_soft_causal_direction"]),
            "causal_monotonicity": float(capacity["mean_soft_causal_monotonicity"]),
            "ood_fraction": float(capacity["mean_ood_fraction"]),
            "topology_across_network": float(
                capacity["soft_topology_across_network"]
            ),
            "topology_mode_separation": float(
                capacity["soft_topology_mode_separation"]
            ),
        },
        "paired_comparisons": paired,
        "per_network": per_network,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    artifact_root = args.artifact_root.resolve()
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Stage-AD audit input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if loaded["stage_ad_complete"].get("status") != (
            "REV12ND_MANUAL_CAPACITY_REPLICATION_COMPLETE"):
        raise RuntimeError("Stage-AD did not complete")
    if loaded["stage_ac_paired_audit"].get("status") != (
            "LOCAL_INTERPOLATION_DIRECTION_ONLY_NO_BALANCED_NODE_FIELD"):
        raise RuntimeError("Stage-AC no longer justifies the capacity audit")
    stage_ad = _rows(loaded["stage_ad_summary"])
    stage_aa = _rows(loaded["stage_aa_summary"])
    candidate_id = config["capacity_candidate_id"]
    if set(stage_ad) != {candidate_id}:
        raise RuntimeError("Stage-AD must contain exactly the capacity control")
    comparison_ids = list(config["comparison_candidate_ids"])
    if any(candidate_id not in stage_aa for candidate_id in comparison_ids):
        raise RuntimeError("Stage-AA comparison field is absent")
    bootstrap = config["bootstrap"]
    result = capacity_audit(
        stage_ad[candidate_id],
        {candidate_id: stage_aa[candidate_id] for candidate_id in comparison_ids},
        description=config["capacity_description"],
        draws=int(bootstrap["draws"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
    )
    payload = {
        "schema_id": "topic4_rev12_nd_manual_capacity_paired_audit_v1",
        **result,
        "capacity_description": config["capacity_description"],
        "bootstrap": bootstrap,
        "selection_eligible": False,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
        "inputs": inputs,
        "claim_boundary": config["claim_boundary"],
    }
    output_root = artifact_root / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "paired_capacity_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    with (output_root / "paired_capacity_per_network.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_network"][0]))
        writer.writeheader()
        writer.writerows(result["per_network"])
    print(json.dumps({
        "status": result["status"],
        "support_counts": result["capacity_support_counts"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
