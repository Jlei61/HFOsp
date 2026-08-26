#!/usr/bin/env python3
"""Estimate orthogonal Node-field response gradients from Stage-AF pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


ENDPOINTS = {
    "soft_objective": ("soft_objective", "objective", "lower"),
    "mode_0": ("soft_objective", "modes", "0", "mean", "lower"),
    "mode_1": ("soft_objective", "modes", "1", "mean", "lower"),
    "mode_0_direction": (
        "soft_causal_direction", "modes", "0", "alignment_score", "higher",
    ),
    "mode_1_direction": (
        "soft_causal_direction", "modes", "1", "alignment_score", "higher",
    ),
    "mode_0_monotonicity": (
        "soft_causal_monotonicity", "modes", "0", "alignment_score", "higher",
    ),
    "mode_1_monotonicity": (
        "soft_causal_monotonicity", "modes", "1", "alignment_score", "higher",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(record: dict, path: tuple[str, ...]) -> float:
    value = record
    for key in path:
        value = value[key]
    return float(value)


def _utility(record: dict, endpoint: str) -> float:
    contract = ENDPOINTS[endpoint]
    value = _get(record, tuple(contract[:-1]))
    return -value if contract[-1] == "lower" else value


def solve_common_direction(gradients: dict[str, np.ndarray],
                           endpoints: list[str]) -> dict:
    normalized = []
    for endpoint in endpoints:
        gradient = np.asarray(gradients[endpoint], float)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= 1e-12:
            return {"success": False, "reason": f"zero gradient: {endpoint}"}
        normalized.append(gradient / norm)
    matrix = np.asarray(normalized)
    start_direction = np.sum(matrix, axis=0)
    norm = float(np.linalg.norm(start_direction))
    if norm <= 1e-12:
        start_direction = matrix[0]
    else:
        start_direction /= norm
    start_margin = float(np.min(matrix @ start_direction))
    start = np.r_[start_direction, start_margin]
    constraints = [
        {"type": "ineq", "fun": lambda x: 1.0 - float(np.dot(x[:-1], x[:-1]))}
    ]
    for row in matrix:
        constraints.append({
            "type": "ineq",
            "fun": lambda x, row=row: float(np.dot(row, x[:-1]) - x[-1]),
        })
    result = minimize(
        lambda x: -float(x[-1]), start, method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        return {"success": False, "reason": str(result.message)}
    direction = np.asarray(result.x[:-1], float)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-12:
        direction /= norm
    predicted = {
        endpoint: float(np.dot(np.asarray(gradients[endpoint]), direction))
        for endpoint in endpoints
    }
    return {
        "success": True,
        "common_normalized_margin": float(np.min(matrix @ direction)),
        "direction": direction.tolist(),
        "predicted_raw_utilities_per_unit_rms": predicted,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def response_audit(manifest: dict, summary: dict, *, analysis: dict) -> dict:
    candidate_rows = {row["candidate_id"]: row for row in summary["rows"]}
    candidate_meta = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_rows.keys() != candidate_meta.keys():
        raise RuntimeError("Stage-AF manifest and aggregate differ")
    modes = {}
    for candidate_id, row in candidate_meta.items():
        residual = row["node_field"]["residual_coordinates"]
        mode_index = int(residual["mode_index"])
        sign = int(residual["sign"])
        modes.setdefault(mode_index, {})[sign] = candidate_id
    expected_modes = list(range(len(modes)))
    if sorted(modes) != expected_modes or any(set(pair) != {-1, 1} for pair in modes.values()):
        raise RuntimeError("Stage-AF orthogonal pairs are incomplete")
    seeds = sorted(int(row["seed"]) for row in next(iter(candidate_rows.values()))["per_network"])
    worker_maps = {
        candidate_id: {int(row["seed"]): row for row in candidate["per_network"]}
        for candidate_id, candidate in candidate_rows.items()
    }
    if any(sorted(records) != seeds for records in worker_maps.values()):
        raise RuntimeError("Stage-AF network pairing changed")
    radius = float(next(iter(candidate_meta.values()))["node_field"][
        "residual_coordinates"
    ]["radius"])
    gradients_by_network = {endpoint: [] for endpoint in ENDPOINTS}
    for seed in seeds:
        for endpoint in ENDPOINTS:
            slopes = []
            for mode_index in expected_modes:
                plus = worker_maps[modes[mode_index][1]][seed]
                minus = worker_maps[modes[mode_index][-1]][seed]
                slopes.append(
                    (_utility(plus, endpoint) - _utility(minus, endpoint))
                    / (2.0 * radius)
                )
            gradients_by_network[endpoint].append(slopes)
    gradients_by_network = {
        key: np.asarray(value, float) for key, value in gradients_by_network.items()
    }
    draws = int(analysis["bootstrap_draws"])
    confidence = float(analysis["bootstrap_confidence"])
    rng = np.random.default_rng(int(analysis["bootstrap_seed"]))
    indices = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    alpha = (1.0 - confidence) / 2.0
    endpoint_audit, mean_gradients = {}, {}
    for endpoint, values in gradients_by_network.items():
        mean = np.mean(values, axis=0)
        mean_gradients[endpoint] = mean
        sampled = np.mean(values[indices], axis=1)
        leave_one = [
            np.mean(np.delete(values, omitted, axis=0), axis=0)
            for omitted in range(len(seeds))
        ]
        endpoint_audit[endpoint] = {
            "mean_gradient": mean.tolist(),
            "coefficient_ci_low": np.quantile(sampled, alpha, axis=0).tolist(),
            "coefficient_ci_high": np.quantile(sampled, 1.0 - alpha, axis=0).tolist(),
            "network_gradient_pairwise_cosine_median": float(np.median([
                _cosine(values[left], values[right])
                for left in range(len(seeds)) for right in range(left + 1, len(seeds))
            ])),
            "leave_network_out_cosines_to_full": [
                _cosine(row, mean) for row in leave_one
            ],
        }
    required = list(analysis["required_joint_endpoints"])
    common = solve_common_direction(mean_gradients, required)
    bootstrap_margins, bootstrap_directions = [], []
    for sample in indices:
        gradients = {
            endpoint: np.mean(values[sample], axis=0)
            for endpoint, values in gradients_by_network.items()
        }
        solved = solve_common_direction(gradients, required)
        if solved["success"]:
            bootstrap_margins.append(solved["common_normalized_margin"])
            bootstrap_directions.append(np.asarray(solved["direction"], float))
    if not common["success"] or len(bootstrap_margins) < 0.95 * draws:
        status = "ORTHOGONAL_RESPONSE_COMMON_DIRECTION_NOT_IDENTIFIED"
        common["bootstrap_successes"] = len(bootstrap_margins)
    else:
        margins = np.asarray(bootstrap_margins)
        direction = np.asarray(common["direction"])
        leave_network_directions = []
        for omitted in range(len(seeds)):
            gradients = {
                endpoint: np.mean(np.delete(values, omitted, axis=0), axis=0)
                for endpoint, values in gradients_by_network.items()
            }
            solved = solve_common_direction(gradients, required)
            if solved["success"]:
                leave_network_directions.append(np.asarray(solved["direction"]))
        leave_cosines = [_cosine(row, direction) for row in leave_network_directions]
        common.update({
            "bootstrap_successes": len(bootstrap_margins),
            "bootstrap_margin_ci": [
                float(np.quantile(margins, alpha)),
                float(np.quantile(margins, 1.0 - alpha)),
            ],
            "bootstrap_direction_cosine_to_full_median": float(np.median([
                _cosine(row, direction) for row in bootstrap_directions
            ])),
            "leave_network_out_direction_cosines": leave_cosines,
        })
        identified = bool(
            common["bootstrap_margin_ci"][0]
            > float(analysis["minimum_bootstrap_common_margin"])
            and min(leave_cosines)
            >= float(analysis["minimum_leave_network_out_direction_cosine"])
        )
        status = (
            "ORTHOGONAL_RESPONSE_COMMON_DIRECTION_IDENTIFIED" if identified else
            "ORTHOGONAL_RESPONSE_COMMON_DIRECTION_NOT_IDENTIFIED"
        )
    return {
        "status": status,
        "n_networks": len(seeds),
        "n_basis_modes": len(expected_modes),
        "radius": radius,
        "endpoint_gradients": endpoint_audit,
        "common_direction": common,
        "analysis_contract": analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    artifact_root = args.artifact_root.resolve()
    output_root = artifact_root / config["output_root"]
    manifest_path = artifact_root / config["candidate_manifest"]
    summary_path = output_root / "aggregate" / "fit_soft_global_summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != "REV12ND_ORTHOGONAL_RESPONSE_CALIBRATION_FROZEN":
        raise RuntimeError("Stage-AF manifest is not frozen")
    result = response_audit(
        manifest, summary, analysis=config["response_analysis"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_orthogonal_response_audit_v1",
        **result,
        "selection_eligible": False,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
        "manual_field_used": False,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        },
        "claim_boundary": config["claim_boundary"],
    }
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    output = analysis_root / "orthogonal_response_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
