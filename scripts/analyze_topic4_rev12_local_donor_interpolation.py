#!/usr/bin/env python3
"""Paired candidate and donor-dose audit for the Stage-AC response surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from scripts.analyze_topic4_rev12_global_soft_field_expansion import (  # noqa: E402
    ENDPOINTS,
    _get,
    bootstrap_mean_interval,
    paired_utilities,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _row_map(summary: dict) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in summary["rows"]}


def _coordinates(manifest: dict) -> dict[str, tuple[float, float]]:
    output = {}
    for row in manifest["candidates"]:
        record = row["node_field"]["residual_coordinates"]
        output[row["candidate_id"]] = (
            float(record["mode_1_dose"]), float(record["direction_dose"]),
        )
    return output


def _endpoint_value(worker: dict, endpoint: str) -> float:
    contract = ENDPOINTS[endpoint]
    return _get(worker, tuple(contract[:-1]))


def _oriented_delta(reference: float, observed: float, endpoint: str) -> float:
    return reference - observed if ENDPOINTS[endpoint][-1] == "lower" else observed - reference


def candidate_audit(reference: dict, candidate: dict, *, draws: int,
                    confidence: float, seed: int, advancement: dict) -> dict:
    endpoints = {
        name: bootstrap_mean_interval(
            values, draws=draws, confidence=confidence,
            seed=seed + index,
        )
        for index, (name, values) in enumerate(
            paired_utilities(reference, candidate).items()
        )
    }
    required = list(advancement["required_positive_mean_endpoints"])
    mean_balance = bool(all(endpoints[name]["mean"] > 0.0 for name in required))
    minimum = int(advancement["minimum_positive_networks_per_mode"])
    mode_stability = bool(
        endpoints["mode_0"]["positive_networks"] >= minimum
        and endpoints["mode_1"]["positive_networks"] >= minimum
    )
    direction_protected = bool(
        endpoints["causal_direction"]["mean"] >= 0.0
        if advancement["direction_must_not_be_negative"] else True
    )
    return {
        "endpoints": endpoints,
        "all_required_means_positive": mean_balance,
        "both_modes_network_stable": mode_stability,
        "direction_protected": direction_protected,
        "advances": bool(mean_balance and mode_stability and direction_protected),
    }


def dose_main_effects(rows: dict[str, dict], coordinates: dict[str, tuple[float, float]],
                      origin: dict, *, draws: int, confidence: float,
                      seed: int) -> dict:
    cells = {(0.0, 0.0): origin}
    cells.update({coordinates[candidate_id]: row for candidate_id, row in rows.items()})
    mode_levels = sorted({coordinate[0] for coordinate in cells})
    direction_levels = sorted({coordinate[1] for coordinate in cells})
    if len(cells) != len(mode_levels) * len(direction_levels):
        raise RuntimeError("Stage-AC response surface is incomplete")
    seeds = sorted(int(row["seed"]) for row in origin["per_network"])
    workers = {
        coordinate: {int(row["seed"]): row for row in candidate["per_network"]}
        for coordinate, candidate in cells.items()
    }
    if any(sorted(records) != seeds for records in workers.values()):
        raise RuntimeError("Stage-AC response surface networks do not align")

    output = {"mode_1_dose": {}, "direction_dose": {}}
    for factor, levels, other_levels, factor_index in (
        ("mode_1_dose", mode_levels, direction_levels, 0),
        ("direction_dose", direction_levels, mode_levels, 1),
    ):
        for level in levels[1:]:
            endpoint_records = {}
            for endpoint in ENDPOINTS:
                values = []
                for network_seed in seeds:
                    deltas = []
                    for other in other_levels:
                        treated_coordinate = (
                            (level, other) if factor_index == 0 else (other, level)
                        )
                        reference_coordinate = (
                            (0.0, other) if factor_index == 0 else (other, 0.0)
                        )
                        observed = _endpoint_value(
                            workers[treated_coordinate][network_seed], endpoint,
                        )
                        reference = _endpoint_value(
                            workers[reference_coordinate][network_seed], endpoint,
                        )
                        deltas.append(_oriented_delta(reference, observed, endpoint))
                    values.append(float(np.mean(deltas)))
                endpoint_records[endpoint] = bootstrap_mean_interval(
                    np.asarray(values), draws=draws, confidence=confidence,
                    seed=seed + 10000 * factor_index + int(round(level * 1000))
                    + list(ENDPOINTS).index(endpoint),
                )
            output[factor][str(level)] = endpoint_records
    return output


def response_surface_audit(stage_aa: dict, manifest: dict, stage_ac: dict, *,
                           anchor_id: str, center_id: str, draws: int,
                           confidence: float, seed: int,
                           advancement: dict) -> dict:
    references = _row_map(stage_aa)
    candidates = _row_map(stage_ac)
    if anchor_id not in references or center_id not in references:
        raise RuntimeError("Stage-AC references are absent")
    coordinates = _coordinates(manifest)
    if candidates.keys() != coordinates.keys():
        raise RuntimeError("Stage-AC manifest and aggregate differ")
    versus_anchor, versus_center = {}, {}
    for index, candidate_id in enumerate(sorted(candidates)):
        versus_anchor[candidate_id] = candidate_audit(
            references[anchor_id], candidates[candidate_id], draws=draws,
            confidence=confidence, seed=seed + 1000 * index,
            advancement=advancement,
        )
        versus_center[candidate_id] = candidate_audit(
            references[center_id], candidates[candidate_id], draws=draws,
            confidence=confidence, seed=seed + 100000 + 1000 * index,
            advancement=advancement,
        )
        versus_anchor[candidate_id]["coordinates"] = {
            "mode_1_dose": coordinates[candidate_id][0],
            "direction_dose": coordinates[candidate_id][1],
        }
    advancing = [
        candidate_id for candidate_id, record in versus_anchor.items()
        if record["advances"]
    ]
    effects = dose_main_effects(
        candidates, coordinates, references[center_id], draws=draws,
        confidence=confidence, seed=seed + 500000,
    )
    direction_only = any(
        record["causal_direction"]["ci_low"] > 0.0
        and not (
            record["mode_0"]["ci_low"] > 0.0
            and record["mode_1"]["ci_low"] > 0.0
        )
        for record in effects["direction_dose"].values()
    )
    status = (
        "LOCAL_INTERPOLATION_BALANCED_FIT_CANDIDATE_FOUND" if advancing
        else "LOCAL_INTERPOLATION_DIRECTION_ONLY_NO_BALANCED_NODE_FIELD"
        if direction_only else "LOCAL_INTERPOLATION_NO_BALANCED_NODE_FIELD"
    )
    return {
        "status": status,
        "candidate_vs_anchor": versus_anchor,
        "candidate_vs_balanced_center": versus_center,
        "advancing_candidate_ids": advancing,
        "dose_main_effects": effects,
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
            raise RuntimeError(f"Stage-AC analysis input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if loaded["stage_ac_complete"].get("status") != (
            "REV12ND_LOCAL_DONOR_INTERPOLATION_COMPLETE"):
        raise RuntimeError("Stage-AC did not complete")
    bootstrap = config["bootstrap"]
    result = response_surface_audit(
        loaded["stage_aa_summary"], loaded["stage_ac_manifest"],
        loaded["stage_ac_summary"], anchor_id=config["anchor_candidate_id"],
        center_id=config["balanced_center_id"], draws=int(bootstrap["draws"]),
        confidence=float(bootstrap["confidence"]), seed=int(bootstrap["seed"]),
        advancement=config["advancement"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_local_donor_interpolation_paired_audit_v1",
        **result,
        "bootstrap": bootstrap,
        "advancement": config["advancement"],
        "inputs": inputs,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output_root"] / "paired_response_surface_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "advancing": result["advancing_candidate_ids"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
