#!/usr/bin/env python3
"""Compare Stage-AH mappings with their rho=1 fields and the global anchor."""
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

from scripts.analyze_topic4_rev12_orthogonal_response_calibration import (  # noqa: E402
    ENDPOINTS,
    _utility,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _paired_endpoints(candidate: dict, reference: dict, endpoints: list[str],
                      *, indices: np.ndarray, confidence: float) -> dict:
    candidate_by_seed = {int(row["seed"]): row for row in candidate["per_network"]}
    reference_by_seed = {int(row["seed"]): row for row in reference["per_network"]}
    seeds = sorted(reference_by_seed)
    if sorted(candidate_by_seed) != seeds:
        raise RuntimeError("mapping and reference network sets differ")
    alpha = (1.0 - confidence) / 2.0
    output = {}
    for endpoint in endpoints:
        values = np.asarray([
            _utility(candidate_by_seed[seed], endpoint)
            - _utility(reference_by_seed[seed], endpoint)
            for seed in seeds
        ], float)
        draws = np.mean(values[indices], axis=1)
        output[endpoint] = {
            "mean_utility": float(np.mean(values)),
            "bootstrap_ci": [
                float(np.quantile(draws, alpha)),
                float(np.quantile(draws, 1.0 - alpha)),
            ],
            "positive_networks": int(np.sum(values > 0.0)),
            "utilities_by_seed": {
                str(seed): float(value) for seed, value in zip(seeds, values)
            },
        }
    return output


def mapping_audit(stage_ah_summary: dict, stage_ag_summary: dict,
                  manifest: dict, *, primary_endpoints: list[str],
                  diagnostic_endpoints: list[str],
                  minimum_positive_networks: int, bootstrap_draws: int,
                  bootstrap_confidence: float, bootstrap_seed: int) -> dict:
    endpoints = primary_endpoints + diagnostic_endpoints
    if any(endpoint not in ENDPOINTS for endpoint in endpoints):
        raise ValueError("unknown mapping endpoint")
    ah_rows = {row["candidate_id"]: row for row in stage_ah_summary["rows"]}
    ag_rows = {row["candidate_id"]: row for row in stage_ag_summary["rows"]}
    meta = {row["candidate_id"]: row for row in manifest["candidates"]}
    if set(ah_rows) != set(meta):
        raise RuntimeError("Stage-AH manifest and summary differ")
    global_anchor = ag_rows["stage_ag_anchor"]
    seeds = sorted(int(row["seed"]) for row in global_anchor["per_network"])
    if not 0 < minimum_positive_networks <= len(seeds):
        raise ValueError("invalid network-majority contract")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seeds), size=(bootstrap_draws, len(seeds)))
    candidates = []
    for candidate_id in sorted(ah_rows):
        source_id = meta[candidate_id]["source_candidate_ids"][0]
        same_field = ag_rows[source_id]
        relative = _paired_endpoints(
            ah_rows[candidate_id], same_field, endpoints,
            indices=indices, confidence=bootstrap_confidence,
        )
        absolute = _paired_endpoints(
            ah_rows[candidate_id], global_anchor, endpoints,
            indices=indices, confidence=bootstrap_confidence,
        )
        balanced_absolute = bool(all(
            absolute[endpoint]["mean_utility"] > 0.0
            and absolute[endpoint]["positive_networks"] >= minimum_positive_networks
            for endpoint in primary_endpoints
        ))
        candidates.append({
            "candidate_id": candidate_id,
            "source_candidate_id": source_id,
            "field_sha256": meta[candidate_id]["node_field"]["field_sha256"],
            "signed_depth_shrinkage": float(
                meta[candidate_id]["node_mapping"]["signed_depth_shrinkage"]
            ),
            "mapping_sha256": meta[candidate_id]["node_mapping"]["mapping_sha256"],
            "relative_to_same_field_rho1": relative,
            "relative_to_stage_ag_anchor_rho1": absolute,
            "balanced_absolute_primary_improvement": balanced_absolute,
        })
    balanced = [
        row["candidate_id"] for row in candidates
        if row["balanced_absolute_primary_improvement"]
    ]
    return {
        "status": (
            "SIGNED_DEPTH_SHRINKAGE_BALANCED_NODE_MAPPING_CANDIDATE_FOUND"
            if balanced else
            "SIGNED_DEPTH_SHRINKAGE_DOES_NOT_RESOLVE_TWO_MODE_TRADEOFF"
        ),
        "network_seeds": seeds,
        "primary_endpoints": primary_endpoints,
        "diagnostic_endpoints": diagnostic_endpoints,
        "minimum_positive_networks_per_primary_endpoint": minimum_positive_networks,
        "balanced_candidate_ids": balanced,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    static = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"mapping analysis input changed: {record['path']}")
        static[name] = json.loads(path.read_text())
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("Stage-AH manifest is stale")
    ah_summary_path = artifact_root / config["output_root"] / "aggregate" / "fit_soft_global_summary.json"
    ah_summary = json.loads(ah_summary_path.read_text())
    analysis = config["analysis"]
    result = mapping_audit(
        ah_summary, static["stage_ag_fit_summary"], manifest,
        primary_endpoints=list(analysis["primary_endpoints"]),
        diagnostic_endpoints=list(analysis["diagnostic_endpoints"]),
        minimum_positive_networks=int(
            analysis["minimum_positive_networks_per_primary_endpoint"]
        ),
        bootstrap_draws=int(analysis["bootstrap_draws"]),
        bootstrap_confidence=float(analysis["bootstrap_confidence"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]),
    )
    payload = {
        "schema_id": "topic4_rev12_nd_signed_depth_mapping_result_v1",
        **result,
        "bootstrap_contract": {
            "draws": int(analysis["bootstrap_draws"]),
            "confidence": float(analysis["bootstrap_confidence"]),
            "seed": int(analysis["bootstrap_seed"]),
            "independent_unit": "network_seed",
        },
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
        "selection_opened": False,
        "inputs": {
            "stage_ah_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "stage_ah_fit_summary": {"path": str(ah_summary_path), "sha256": _sha256(ah_summary_path)},
            "stage_ag_fit_summary": config["inputs"]["stage_ag_fit_summary"],
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_root"] / "analysis" / "signed_depth_mapping_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "balanced_candidate_ids": payload["balanced_candidate_ids"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
