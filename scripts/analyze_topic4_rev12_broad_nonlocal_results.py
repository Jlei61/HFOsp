#!/usr/bin/env python3
"""Paired network-level audit of the Stage-AG broad Node-field screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from scripts.analyze_topic4_rev12_orthogonal_response_calibration import (
    ENDPOINTS,
    _utility,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def paired_candidate_audit(
    summary: dict,
    *,
    anchor_candidate_id: str,
    primary_endpoints: list[str],
    diagnostic_endpoints: list[str],
    bootstrap_draws: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict:
    endpoints = primary_endpoints + diagnostic_endpoints
    if len(set(endpoints)) != len(endpoints) or any(key not in ENDPOINTS for key in endpoints):
        raise ValueError("unknown or repeated endpoint")
    rows = {row["candidate_id"]: row for row in summary["rows"]}
    if anchor_candidate_id not in rows:
        raise ValueError("anchor candidate missing")
    per_network = {
        candidate_id: {int(record["seed"]): record for record in row["per_network"]}
        for candidate_id, row in rows.items()
    }
    seeds = sorted(per_network[anchor_candidate_id])
    if any(sorted(records) != seeds for records in per_network.values()):
        raise ValueError("candidate network sets are not paired")
    if bootstrap_draws < 1 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("invalid bootstrap contract")
    rng = np.random.default_rng(bootstrap_seed)
    draw_indices = rng.integers(0, len(seeds), size=(bootstrap_draws, len(seeds)))
    alpha = (1.0 - bootstrap_confidence) / 2.0
    anchor = per_network[anchor_candidate_id]
    candidates = []
    for candidate_id in sorted(rows):
        if candidate_id == anchor_candidate_id:
            continue
        endpoint_rows = {}
        utility_matrix = []
        for endpoint in endpoints:
            values = np.asarray([
                _utility(per_network[candidate_id][seed], endpoint)
                - _utility(anchor[seed], endpoint)
                for seed in seeds
            ], dtype=float)
            draws = np.mean(values[draw_indices], axis=1)
            endpoint_rows[endpoint] = {
                "mean_utility": float(np.mean(values)),
                "median_utility": float(np.median(values)),
                "bootstrap_ci": [
                    float(np.quantile(draws, alpha)),
                    float(np.quantile(draws, 1.0 - alpha)),
                ],
                "positive_networks": int(np.sum(values > 0.0)),
                "zero_networks": int(np.sum(values == 0.0)),
                "utilities_by_seed": {
                    str(seed): float(value) for seed, value in zip(seeds, values)
                },
            }
            utility_matrix.append(values)
        matrix = np.asarray(utility_matrix).T
        primary_indices = [endpoints.index(endpoint) for endpoint in primary_endpoints]
        diagnostic_indices = [endpoints.index(endpoint) for endpoint in diagnostic_endpoints]
        primary_means_positive = all(
            endpoint_rows[endpoint]["mean_utility"] > 0.0
            for endpoint in primary_endpoints
        )
        candidate_result = {
            "candidate_id": candidate_id,
            "role": rows[candidate_id].get("role"),
            "n_networks": len(seeds),
            "endpoints": endpoint_rows,
            "all_primary_mean_utilities_positive": primary_means_positive,
            "all_primary_ci_lower_bounds_positive": all(
                endpoint_rows[endpoint]["bootstrap_ci"][0] > 0.0
                for endpoint in primary_endpoints
            ),
            "all_seven_mean_utilities_positive": bool(
                primary_means_positive and all(
                    endpoint_rows[endpoint]["mean_utility"] > 0.0
                    for endpoint in diagnostic_endpoints
                )
            ),
            "joint_primary_positive_networks": int(np.sum(
                np.all(matrix[:, primary_indices] > 0.0, axis=1)
            )),
            "joint_diagnostic_positive_networks": int(np.sum(
                np.all(matrix[:, diagnostic_indices] > 0.0, axis=1)
            )),
            "joint_all_endpoint_positive_networks": int(np.sum(
                np.all(matrix > 0.0, axis=1)
            )),
        }
        candidates.append(candidate_result)
    balanced = [
        row["candidate_id"] for row in candidates
        if row["all_primary_mean_utilities_positive"]
    ]
    best_fit = max(
        candidates,
        key=lambda row: row["endpoints"]["soft_objective"]["mean_utility"],
    )
    best_mode_1_direction = max(
        candidates,
        key=lambda row: row["endpoints"]["mode_1_direction"]["mean_utility"],
    )
    status = (
        "BROAD_FIELD_BALANCED_FIT_DIRECTION_CANDIDATE_FOUND"
        if balanced else
        "BROAD_FIELD_SEARCH_PATIENT_DIRECTION_TRADEOFF_PERSISTS_NO_BALANCED_NODE_CANDIDATE"
    )
    return {
        "status": status,
        "anchor_candidate_id": anchor_candidate_id,
        "network_seeds": seeds,
        "n_candidates_compared": len(candidates),
        "primary_endpoints": primary_endpoints,
        "diagnostic_endpoints": diagnostic_endpoints,
        "balanced_candidate_ids": balanced,
        "best_soft_objective_candidate": best_fit["candidate_id"],
        "best_mode_1_direction_candidate": best_mode_1_direction["candidate_id"],
        "candidates": candidates,
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
            raise RuntimeError(f"Stage-AG audit input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if loaded["stage_ag_complete"].get("status") != (
        "REV12ND_BROAD_NONLOCAL_FIELD_SCREEN_COMPLETE"
    ):
        raise RuntimeError("Stage-AG did not complete")
    manifest_ids = {row["candidate_id"] for row in loaded["candidate_manifest"]["candidates"]}
    summary_ids = {row["candidate_id"] for row in loaded["fit_summary"]["rows"]}
    if manifest_ids != summary_ids:
        raise RuntimeError("Stage-AG manifest and aggregate differ")
    result = paired_candidate_audit(
        loaded["fit_summary"],
        anchor_candidate_id=config["anchor_candidate_id"],
        primary_endpoints=list(config["primary_endpoints"]),
        diagnostic_endpoints=list(config["diagnostic_endpoints"]),
        bootstrap_draws=int(config["bootstrap_draws"]),
        bootstrap_confidence=float(config["bootstrap_confidence"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    payload = {
        "schema_id": config["schema_id"],
        **result,
        "bootstrap_contract": {
            "draws": int(config["bootstrap_draws"]),
            "confidence": float(config["bootstrap_confidence"]),
            "seed": int(config["bootstrap_seed"]),
            "independent_unit": "network_seed",
        },
        "patient_heldout_used": bool(config["patient_heldout_used"]),
        "natural_kmeans_used_for_selection": bool(
            config["natural_kmeans_used_for_selection"]
        ),
        "selection_opened": bool(config["selection_opened"]),
        "inputs": inputs,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_root"] / "broad_nonlocal_paired_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "balanced_candidate_ids": payload["balanced_candidate_ids"],
        "best_soft_objective_candidate": payload["best_soft_objective_candidate"],
        "best_mode_1_direction_candidate": payload["best_mode_1_direction_candidate"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
