#!/usr/bin/env python3
"""Paired-network audit of the Stage-AA global Node-field expansion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


ENDPOINTS = {
    "soft_objective": ("soft_objective", "objective", "lower"),
    "mode_0": ("soft_objective", "modes", "0", "mean", "lower"),
    "mode_1": ("soft_objective", "modes", "1", "mean", "lower"),
    "causal_direction": ("soft_causal_direction", "score", "higher"),
    "causal_monotonicity": ("soft_causal_monotonicity", "score", "higher"),
    "ood_fraction": ("ood_fraction", "lower"),
    "compound_fraction": ("compound_fraction", "lower"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _get(record: dict, path: tuple[str, ...]) -> float:
    value = record
    for key in path:
        value = value[key]
    return float(value)


def paired_utilities(anchor: dict, candidate: dict) -> dict[str, np.ndarray]:
    anchor_rows = {int(row["seed"]): row for row in anchor["per_network"]}
    candidate_rows = {int(row["seed"]): row for row in candidate["per_network"]}
    if anchor_rows.keys() != candidate_rows.keys():
        raise RuntimeError("anchor and candidate networks do not align")
    output = {}
    for name, contract in ENDPOINTS.items():
        direction = contract[-1]
        path = tuple(contract[:-1])
        values = []
        for seed in sorted(anchor_rows):
            reference = _get(anchor_rows[seed], path)
            observed = _get(candidate_rows[seed], path)
            values.append(
                reference - observed if direction == "lower"
                else observed - reference
            )
        output[name] = np.asarray(values, float)
    return output


def bootstrap_mean_interval(values: np.ndarray, *, draws: int,
                            confidence: float, seed: int) -> dict:
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires finite one-dimensional values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(draws), len(values)))
    means = np.mean(values[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "positive_networks": int(np.sum(values > 0.0)),
        "zero_networks": int(np.sum(values == 0.0)),
        "n_networks": int(len(values)),
        "values": values.tolist(),
    }


def audit_summary(summary: dict, *, anchor_id: str, draws: int,
                  confidence: float, seed: int) -> dict:
    rows = {row["candidate_id"]: row for row in summary["rows"]}
    if anchor_id not in rows:
        raise RuntimeError("paired anchor is absent")
    if not rows[anchor_id].get("fit_valid", False):
        raise RuntimeError("paired anchor is invalid")
    output = {}
    for index, candidate_id in enumerate(sorted(rows)):
        if candidate_id == anchor_id:
            continue
        row = rows[candidate_id]
        if not row.get("fit_valid", False):
            raise RuntimeError(f"invalid expansion candidate: {candidate_id}")
        utilities = paired_utilities(rows[anchor_id], row)
        endpoints = {
            name: bootstrap_mean_interval(
                values, draws=draws, confidence=confidence,
                seed=seed + 1009 * index + endpoint_index,
            )
            for endpoint_index, (name, values) in enumerate(utilities.items())
        }
        means = [endpoints[key]["mean"] for key in (
            "soft_objective", "mode_0", "mode_1", "causal_direction",
        )]
        output[candidate_id] = {
            "endpoints": endpoints,
            "all_four_mean_utilities_positive": bool(all(value > 0 for value in means)),
            "both_mode_means_positive": bool(
                endpoints["mode_0"]["mean"] > 0
                and endpoints["mode_1"]["mean"] > 0
            ),
            "both_modes_at_least_two_thirds_networks": bool(
                endpoints["mode_0"]["positive_networks"]
                >= np.ceil(2 * endpoints["mode_0"]["n_networks"] / 3)
                and endpoints["mode_1"]["positive_networks"]
                >= np.ceil(2 * endpoints["mode_1"]["n_networks"] / 3)
            ),
            "aggregate_topology": {
                "across_network_delta": float(
                    row["soft_topology_across_network"]
                    - rows[anchor_id]["soft_topology_across_network"]
                ),
                "mode_separation_delta": float(
                    row["soft_topology_mode_separation"]
                    - rows[anchor_id]["soft_topology_mode_separation"]
                ),
            },
        }
    return output


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
            raise RuntimeError(f"paired audit input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    bootstrap = config["bootstrap"]
    fresh = audit_summary(
        loaded["stage_aa_summary"], anchor_id=config["anchor_candidate_id"],
        draws=int(bootstrap["draws"]), confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
    )
    earlier = audit_summary(
        loaded["stage_z_summary"], anchor_id=config["anchor_candidate_id"],
        draws=int(bootstrap["draws"]), confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]) + 100000,
    )
    common = sorted(set(fresh) & set(earlier))
    combined = {}
    for candidate_id in common:
        combined[candidate_id] = {"endpoints": {}}
        for endpoint in ENDPOINTS:
            values = np.asarray(
                earlier[candidate_id]["endpoints"][endpoint]["values"]
                + fresh[candidate_id]["endpoints"][endpoint]["values"], float,
            )
            combined[candidate_id]["endpoints"][endpoint] = bootstrap_mean_interval(
                values, draws=int(bootstrap["draws"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]) + 200000 + 997 * common.index(candidate_id),
            )
    balanced = [
        candidate_id for candidate_id, row in fresh.items()
        if row["all_four_mean_utilities_positive"]
    ]
    stable = [
        candidate_id for candidate_id in balanced
        if fresh[candidate_id]["both_modes_at_least_two_thirds_networks"]
    ]
    status = (
        "FRESH_FIT_BALANCED_AND_MODE_STABLE_CANDIDATE_FOUND" if stable
        else "FRESH_FIT_BALANCED_MEAN_CANDIDATE_MODE_STABILITY_UNRESOLVED"
        if balanced else "FRESH_FIT_DIRECTION_PATIENT_FIT_TRADEOFF_UNRESOLVED"
    )
    payload = {
        "schema_id": "topic4_rev12_nd_global_soft_field_expansion_paired_audit_v1",
        "status": status,
        "fresh_nine_networks_primary": fresh,
        "earlier_three_networks_sensitivity": earlier,
        "combined_twelve_networks_sensitivity": combined,
        "balanced_mean_candidate_ids": balanced,
        "balanced_and_mode_stable_candidate_ids": stable,
        "automatic_stage_aa_nomination": loaded["stage_aa_nomination"],
        "bootstrap": bootstrap,
        "inputs": inputs,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
        "claim_boundary": config["claim_boundary"],
    }
    output_root = artifact_root / config["output_root"]
    _atomic_json(output_root / "paired_expansion_audit.json", payload)
    fields = [
        "candidate_id", "endpoint", "mean", "ci_low", "ci_high",
        "positive_networks", "n_networks",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "paired_expansion_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate_id, row in fresh.items():
            for endpoint, record in row["endpoints"].items():
                writer.writerow({
                    "candidate_id": candidate_id, "endpoint": endpoint,
                    **{key: record[key] for key in fields[2:]},
                })
    print(json.dumps({
        "status": status, "balanced": balanced, "stable": stable,
        "output": str(output_root / "paired_expansion_audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
