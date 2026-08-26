#!/usr/bin/env python3
"""Audit whether an apparent mean scalar-gain crossing is network robust."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _utilities(record: dict, endpoint: str) -> dict[int, float]:
    values = record[endpoint]["utilities_by_seed"]
    return {int(seed): float(value) for seed, value in values.items()}


def crossing_audit(left: dict, right: dict, *, left_gain: float,
                   right_gain: float, endpoints: list[str], grid_points: int,
                   minimum_positive_networks: int) -> dict:
    if not 0 <= left_gain < right_gain:
        raise ValueError("gain interval must be increasing and nonnegative")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    left_by_endpoint = {name: _utilities(left, name) for name in endpoints}
    right_by_endpoint = {name: _utilities(right, name) for name in endpoints}
    seeds = sorted(left_by_endpoint[endpoints[0]])
    if not 0 < minimum_positive_networks <= len(seeds):
        raise ValueError("invalid network-majority contract")
    for name in endpoints:
        if sorted(left_by_endpoint[name]) != seeds:
            raise RuntimeError(f"left seed mismatch for {name}")
        if sorted(right_by_endpoint[name]) != seeds:
            raise RuntimeError(f"right seed mismatch for {name}")

    rows = []
    for gain in np.linspace(left_gain, right_gain, grid_points):
        fraction = float((gain - left_gain) / (right_gain - left_gain))
        values = {}
        positive_sets = []
        for name in endpoints:
            per_seed = {
                seed: ((1.0 - fraction) * left_by_endpoint[name][seed]
                       + fraction * right_by_endpoint[name][seed])
                for seed in seeds
            }
            positive = [seed for seed, value in per_seed.items() if value > 0.0]
            positive_sets.append(set(positive))
            values[name] = {
                "mean_utility": float(np.mean(list(per_seed.values()))),
                "positive_networks": len(positive),
            }
        joint = sorted(set.intersection(*positive_sets))
        mean_positive = all(values[name]["mean_utility"] > 0.0 for name in endpoints)
        endpoint_majority = all(
            values[name]["positive_networks"] >= minimum_positive_networks
            for name in endpoints
        )
        rows.append({
            "gain": float(gain),
            "endpoints": values,
            "all_endpoint_means_positive": mean_positive,
            "all_endpoints_have_network_majority": endpoint_majority,
            "joint_positive_networks": joint,
            "formal_corridor_point": bool(mean_positive and endpoint_majority),
        })

    mean_crossing = [row for row in rows if row["all_endpoint_means_positive"]]
    robust = [row for row in rows if row["formal_corridor_point"]]
    maximum_joint = max(len(row["joint_positive_networks"]) for row in rows)
    representative = max(
        rows,
        key=lambda row: (
            min(row["endpoints"][name]["mean_utility"] for name in endpoints),
            len(row["joint_positive_networks"]),
        ),
    )
    return {
        "status": (
            "NETWORK_ROBUST_SCALAR_GAIN_CORRIDOR_FOUND"
            if robust else
            "MEAN_GAIN_CROSSING_WITHOUT_NETWORK_ROBUST_CORRIDOR"
        ),
        "network_seeds": seeds,
        "endpoints": endpoints,
        "minimum_positive_networks_per_endpoint": minimum_positive_networks,
        "left_gain": left_gain,
        "right_gain": right_gain,
        "grid_points": grid_points,
        "mean_positive_gain_interval": (
            [mean_crossing[0]["gain"], mean_crossing[-1]["gain"]]
            if mean_crossing else None
        ),
        "formal_corridor_gain_interval": (
            [robust[0]["gain"], robust[-1]["gain"]] if robust else None
        ),
        "maximum_joint_positive_networks": maximum_joint,
        "best_minimum_mean_point": representative,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    loaded = {}
    resolved = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"crossing-audit input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        resolved[name] = {"path": str(path), "sha256": _sha256(path)}

    design = config["interpolation"]
    left = next(
        row["relative_to_stage_ag_anchor_gain1"]
        for row in loaded["stage_aj_result"]["candidates"]
        if row["candidate_id"] == design["left_candidate_id"]
    )
    right = next(
        row["endpoints"]
        for row in loaded["stage_ag_paired_audit"]["candidates"]
        if row["candidate_id"] == design["right_candidate_id"]
    )
    result = crossing_audit(
        left, right,
        left_gain=float(design["left_gain"]),
        right_gain=float(design["right_gain"]),
        endpoints=list(design["primary_endpoints"]),
        grid_points=int(design["grid_points"]),
        minimum_positive_networks=int(
            design["minimum_positive_networks_per_endpoint"]
        ),
    )
    payload = {
        "schema_id": "topic4_rev12_nd_scalar_gain_crossing_audit_v1",
        **result,
        "linear_interpolation_only": True,
        "patient_heldout_used": False,
        "inputs": resolved,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "mean_positive_gain_interval": payload["mean_positive_gain_interval"],
        "formal_corridor_gain_interval": payload["formal_corridor_gain_interval"],
        "maximum_joint_positive_networks": payload["maximum_joint_positive_networks"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
