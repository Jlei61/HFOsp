#!/usr/bin/env python3
"""Paired fresh-network audit of the omitted Stage-Z Pareto field."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_topic4_rev12_global_soft_field_expansion import (
    bootstrap_mean_interval,
    paired_utilities,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def paired_recovery_audit(anchor: dict, candidate: dict, *, draws: int,
                          confidence: float, seed: int) -> dict:
    utilities = paired_utilities(anchor, candidate)
    endpoints = {
        name: bootstrap_mean_interval(
            values, draws=draws, confidence=confidence,
            seed=seed + endpoint_index,
        )
        for endpoint_index, (name, values) in enumerate(utilities.items())
    }
    required = ("soft_objective", "mode_0", "mode_1", "causal_direction")
    balanced_mean = bool(all(endpoints[name]["mean"] > 0.0 for name in required))
    stable_threshold = int(np.ceil(2 * endpoints["mode_0"]["n_networks"] / 3))
    both_modes_stable = bool(
        endpoints["mode_0"]["positive_networks"] >= stable_threshold
        and endpoints["mode_1"]["positive_networks"] >= stable_threshold
    )
    mode_1_only = bool(
        endpoints["mode_1"]["positive_networks"] >= stable_threshold
        and endpoints["mode_0"]["positive_networks"] < stable_threshold
    )
    topology = {
        "across_network_delta": float(
            candidate["soft_topology_across_network"]
            - anchor["soft_topology_across_network"]
        ),
        "mode_separation_delta": float(
            candidate["soft_topology_mode_separation"]
            - anchor["soft_topology_mode_separation"]
        ),
    }
    status = (
        "OMITTED_PARETO_BALANCED_AND_MODE_STABLE" if balanced_mean and both_modes_stable
        else "OMITTED_PARETO_MODE1_ONLY_NO_BALANCED_STABILITY" if mode_1_only
        else "OMITTED_PARETO_NO_BALANCED_STABILITY"
    )
    return {
        "status": status,
        "endpoints": endpoints,
        "balanced_mean": balanced_mean,
        "both_modes_stable": both_modes_stable,
        "mode_1_only_network_tendency": mode_1_only,
        "stable_network_threshold": stable_threshold,
        "aggregate_topology": topology,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"recovery audit input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if loaded["stage_ab_complete"].get("status") != (
            "REV12ND_OMITTED_PARETO_RECOVERY_COMPLETE"):
        raise RuntimeError("Stage-AB did not complete")
    aa_rows = {row["candidate_id"]: row for row in loaded["stage_aa_summary"]["rows"]}
    ab_rows = {row["candidate_id"]: row for row in loaded["stage_ab_summary"]["rows"]}
    anchor_id = config["anchor_candidate_id"]
    candidate_id = config["recovery_candidate_id"]
    if anchor_id not in aa_rows or candidate_id not in ab_rows:
        raise RuntimeError("paired Stage-AB fields are absent")
    bootstrap = config["bootstrap"]
    result = paired_recovery_audit(
        aa_rows[anchor_id], ab_rows[candidate_id],
        draws=int(bootstrap["draws"]), confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
    )
    payload = {
        "schema_id": "topic4_rev12_nd_omitted_pareto_recovery_paired_audit_v1",
        **result,
        "anchor_candidate_id": anchor_id,
        "recovery_candidate_id": candidate_id,
        "bootstrap": bootstrap,
        "inputs": inputs,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_root"] / "paired_recovery_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "balanced_mean": result["balanced_mean"],
        "both_modes_stable": result["both_modes_stable"], "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
