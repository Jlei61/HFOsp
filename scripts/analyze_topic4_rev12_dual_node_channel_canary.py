#!/usr/bin/env python3
"""Pair Stage-AK dual Node channels on fresh networks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic4_rev12_orthogonal_response_calibration import ENDPOINTS
from scripts.analyze_topic4_rev12_signed_depth_mapping_audit import _paired_endpoints


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def dual_channel_audit(summary: dict, manifest: dict, *, anchor_id: str,
                       primary_endpoints: list[str],
                       diagnostic_endpoints: list[str],
                       minimum_positive_networks: int, bootstrap_draws: int,
                       bootstrap_confidence: float,
                       bootstrap_seed: int) -> dict:
    endpoints = primary_endpoints + diagnostic_endpoints
    if any(endpoint not in ENDPOINTS for endpoint in endpoints):
        raise ValueError("unknown dual-channel endpoint")
    rows = {row["candidate_id"]: row for row in summary["rows"]}
    meta = {row["candidate_id"]: row for row in manifest["candidates"]}
    if set(rows) != set(meta) or anchor_id not in rows:
        raise RuntimeError("dual-channel manifest, summary or anchor changed")
    anchor = rows[anchor_id]
    seeds = sorted(int(row["seed"]) for row in anchor["per_network"])
    if not 0 < minimum_positive_networks <= len(seeds):
        raise ValueError("invalid network-majority contract")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seeds), size=(bootstrap_draws, len(seeds)))

    coupled_by_source = {}
    for candidate_id, record in meta.items():
        source = record["source_candidate_ids"]
        if source["mean"] == source["dispersion"]:
            coupled_by_source[source["mean"]] = candidate_id
    candidates = []
    for candidate_id in sorted(rows):
        record = meta[candidate_id]
        source = record["source_candidate_ids"]
        coupled = source["mean"] == source["dispersion"]
        relative_anchor = _paired_endpoints(
            rows[candidate_id], anchor, endpoints,
            indices=indices, confidence=bootstrap_confidence,
        )
        relative_mean_control = _paired_endpoints(
            rows[candidate_id], rows[coupled_by_source[source["mean"]]], endpoints,
            indices=indices, confidence=bootstrap_confidence,
        )
        relative_dispersion_control = _paired_endpoints(
            rows[candidate_id], rows[coupled_by_source[source["dispersion"]]], endpoints,
            indices=indices, confidence=bootstrap_confidence,
        )
        balanced = bool(not coupled and all(
            relative_anchor[endpoint]["mean_utility"] > 0.0
            and relative_anchor[endpoint]["positive_networks"] >= minimum_positive_networks
            for endpoint in primary_endpoints
        ))
        candidates.append({
            "candidate_id": candidate_id,
            "mean_source_candidate_id": source["mean"],
            "dispersion_source_candidate_id": source["dispersion"],
            "coupled_control": coupled,
            "mean_events": float(rows[candidate_id]["mean_events"]),
            "fit_valid": bool(rows[candidate_id]["fit_valid"]),
            "invalid_reasons": list(rows[candidate_id]["invalid_reasons"]),
            "relative_to_anchor_coupled": relative_anchor,
            "relative_to_mean_coupled_control": relative_mean_control,
            "relative_to_dispersion_coupled_control": relative_dispersion_control,
            "balanced_absolute_primary_improvement": balanced,
        })
    balanced_ids = [
        row["candidate_id"] for row in candidates
        if row["balanced_absolute_primary_improvement"]
    ]
    return {
        "status": (
            "DUAL_CONTINUOUS_NODE_CHANNEL_BALANCED_CANDIDATE_FOUND"
            if balanced_ids else
            "DUAL_CONTINUOUS_NODE_CHANNEL_DOES_NOT_RESOLVE_TWO_MODE_TRADEOFF"
        ),
        "anchor_candidate_id": anchor_id,
        "network_seeds": seeds,
        "primary_endpoints": primary_endpoints,
        "diagnostic_endpoints": diagnostic_endpoints,
        "minimum_positive_networks_per_primary_endpoint": minimum_positive_networks,
        "balanced_candidate_ids": balanced_ids,
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
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("Stage-AK manifest is stale")
    summary_path = artifact_root / config["output_root"] / "aggregate" / "fit_soft_global_summary.json"
    summary = json.loads(summary_path.read_text())
    analysis = config["analysis"]
    anchor_id = next(
        row["candidate_id"] for row in manifest["candidates"]
        if row["source_candidate_ids"] == {
            "mean": config["dual_node_channel"]["anchor_candidate_id"],
            "dispersion": config["dual_node_channel"]["anchor_candidate_id"],
        }
    )
    result = dual_channel_audit(
        summary, manifest, anchor_id=anchor_id,
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
        "schema_id": "topic4_rev12_nd_dual_node_channel_result_v1",
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
            "stage_ak_manifest": {
                "path": str(manifest_path), "sha256": _sha256(manifest_path),
            },
            "stage_ak_fit_summary": {
                "path": str(summary_path), "sha256": _sha256(summary_path),
            },
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_root"] / "analysis" / "dual_node_channel_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "balanced_candidate_ids": payload["balanced_candidate_ids"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
