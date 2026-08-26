#!/usr/bin/env python3
"""Test Stage-AF's common direction on held-out fit-network slopes."""
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
    solve_common_direction,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def network_gradients(manifest: dict, summary: dict) -> tuple[list[int], dict[str, np.ndarray]]:
    rows = {row["candidate_id"]: row for row in summary["rows"]}
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if rows.keys() != candidates.keys():
        raise RuntimeError("Stage-AF manifest and summary differ")
    modes = {}
    radius = None
    for candidate_id, candidate in candidates.items():
        residual = candidate["node_field"]["residual_coordinates"]
        modes.setdefault(int(residual["mode_index"]), {})[
            int(residual["sign"])
        ] = candidate_id
        observed_radius = float(residual["radius"])
        if radius is None:
            radius = observed_radius
        elif not np.isclose(radius, observed_radius):
            raise RuntimeError("Stage-AF pair radii differ")
    if any(set(pair) != {-1, 1} for pair in modes.values()):
        raise RuntimeError("Stage-AF pair is incomplete")
    worker_maps = {
        candidate_id: {int(row["seed"]): row for row in candidate["per_network"]}
        for candidate_id, candidate in rows.items()
    }
    seeds = sorted(next(iter(worker_maps.values())))
    if any(sorted(records) != seeds for records in worker_maps.values()):
        raise RuntimeError("Stage-AF network sets differ")
    output = {endpoint: [] for endpoint in ENDPOINTS}
    for seed in seeds:
        for endpoint in ENDPOINTS:
            output[endpoint].append([
                (
                    _utility(worker_maps[modes[mode][1]][seed], endpoint)
                    - _utility(worker_maps[modes[mode][-1]][seed], endpoint)
                ) / (2.0 * radius)
                for mode in sorted(modes)
            ])
    return seeds, {key: np.asarray(value, float) for key, value in output.items()}


def crossvalidated_common_direction(
        gradients_by_network: dict[str, np.ndarray], seeds: list[int],
        endpoints: list[str], *, minimum_positive_networks_per_endpoint: int,
        minimum_joint_positive_networks: int,
        minimum_median_heldout_margin: float) -> dict:
    if any(len(values) != len(seeds) for values in gradients_by_network.values()):
        raise RuntimeError("network gradient count changed")
    heldout = []
    for index, seed in enumerate(seeds):
        train = {
            endpoint: np.mean(np.delete(gradients_by_network[endpoint], index, axis=0), axis=0)
            for endpoint in endpoints
        }
        solved = solve_common_direction(train, endpoints)
        if not solved["success"]:
            heldout.append({"seed": int(seed), "success": False, "reason": solved["reason"]})
            continue
        direction = np.asarray(solved["direction"], float)
        raw, normalized = {}, {}
        for endpoint in endpoints:
            value = float(np.dot(gradients_by_network[endpoint][index], direction))
            scale = float(np.linalg.norm(train[endpoint]))
            raw[endpoint] = value
            normalized[endpoint] = value / scale if scale > 1e-12 else float("nan")
        finite = all(np.isfinite(value) for value in normalized.values())
        heldout.append({
            "seed": int(seed),
            "success": bool(finite),
            "train_common_margin": float(solved["common_normalized_margin"]),
            "heldout_raw_utilities_per_unit_rms": raw,
            "heldout_normalized_utilities": normalized,
            "heldout_joint_margin": float(min(normalized.values())) if finite else None,
            "all_endpoints_positive": bool(finite and all(value > 0.0 for value in raw.values())),
            "direction": solved["direction"],
        })
    successful = [row for row in heldout if row["success"]]
    counts = {
        endpoint: int(sum(
            row["heldout_raw_utilities_per_unit_rms"][endpoint] > 0.0
            for row in successful
        ))
        for endpoint in endpoints
    }
    joint_positive = int(sum(row["all_endpoints_positive"] for row in successful))
    margins = [row["heldout_joint_margin"] for row in successful]
    median_margin = float(np.median(margins)) if margins else float("nan")
    supported = bool(
        len(successful) == len(seeds)
        and all(value >= minimum_positive_networks_per_endpoint for value in counts.values())
        and joint_positive >= minimum_joint_positive_networks
        and median_margin > minimum_median_heldout_margin
    )
    return {
        "status": (
            "CROSSVALIDATED_COMMON_DIRECTION_SUPPORTED" if supported else
            "CROSSVALIDATED_COMMON_DIRECTION_NOT_SUPPORTED"
        ),
        "supported": supported,
        "positive_networks_per_endpoint": counts,
        "joint_positive_networks": joint_positive,
        "median_heldout_joint_margin": median_margin,
        "minimum_heldout_joint_margin": float(min(margins)) if margins else None,
        "heldout_networks": heldout,
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
            raise RuntimeError(f"crossvalidation input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if loaded["stage_af_complete"].get("status") != (
            "REV12ND_ORTHOGONAL_RESPONSE_CALIBRATION_COMPLETE"):
        raise RuntimeError("Stage-AF did not complete")
    if loaded["stage_af_audit_v1"].get("status") != (
            "ORTHOGONAL_RESPONSE_COMMON_DIRECTION_IDENTIFIED"):
        raise RuntimeError("Stage-AF v1 did not propose a common direction")
    seeds, gradients = network_gradients(
        loaded["stage_af_manifest"], loaded["stage_af_summary"],
    )
    result = crossvalidated_common_direction(
        gradients, seeds, list(config["required_joint_endpoints"]),
        minimum_positive_networks_per_endpoint=int(
            config["minimum_positive_networks_per_endpoint"]
        ),
        minimum_joint_positive_networks=int(config["minimum_joint_positive_networks"]),
        minimum_median_heldout_margin=float(config["minimum_median_heldout_margin"]),
    )
    payload = {
        "schema_id": "topic4_rev12_nd_orthogonal_response_crossvalidation_v1",
        **result,
        "contract": {
            key: config[key] for key in (
                "required_joint_endpoints",
                "minimum_positive_networks_per_endpoint",
                "minimum_joint_positive_networks",
                "minimum_median_heldout_margin",
            )
        },
        "patient_heldout_used": False,
        "manual_field_used": False,
        "selection_eligible": False,
        "inputs": inputs,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_root"] / "orthogonal_response_crossvalidation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "positive_networks": result["positive_networks_per_endpoint"],
        "joint_positive_networks": result["joint_positive_networks"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
