#!/usr/bin/env python3
"""Validate and rank rev17 dual-field candidates on fresh networks."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev17_dual_field_selection.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as flat  # noqa: E402
from scripts import aggregate_topic4_rev17_dual_field_residual_atlas as atlas  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402


STATUS = "REV17_DUAL_FIELD_FRESH_SELECTION_AGGREGATE_COMPLETE"
MANIFEST_STATUS = "REV17_DUAL_FIELD_SELECTION_CANDIDATES_FROZEN"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(canary._jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_contract(config_path: Path, root: Path) -> tuple[dict, dict, Path]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_dual_field_selection_v1":
        raise RuntimeError("rev17 selection schema changed")
    if config.get("scientific_role") != (
        "development_only_dual_continuous_node_residual_selection"
    ):
        raise RuntimeError("rev17 selection scientific role changed")
    if config.get("pathways") != {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off", "Z_M": "off",
    }:
        raise RuntimeError("rev17 selection opened a forbidden pathway")
    for name, record in config["inputs"].items():
        path = atlas._resolve(root, record["path"])
        if not path.is_file() or atlas._sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev17 selection input changed: {name}")
    manifest_path = root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != MANIFEST_STATUS:
        raise RuntimeError("rev17 selection manifest is not formally frozen")
    if manifest.get("config_sha256") != atlas._sha256(config_path):
        raise RuntimeError("rev17 selection manifest/config hash mismatch")
    if not manifest.get("provenance", {}).get("formal_ready"):
        raise RuntimeError("rev17 selection manifest provenance is not formal")
    expected = int(config["dual_field_selection"]["candidate_count_including_anchor"])
    if len(manifest.get("candidates", [])) != expected:
        raise RuntimeError("rev17 selection candidate count changed")
    return config, manifest, manifest_path


def evaluate_candidates(
    rows: list[dict[str, Any]], candidates: Mapping[str, Mapping[str, Any]],
    seeds: list[int], selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply all fresh-network gates with the network as the independent unit."""
    exact = {
        int(row["seed"]): row for row in rows
        if row["candidate_id"] == "exact_dual_anchor"
    }
    if sorted(exact) != sorted(seeds):
        raise RuntimeError("rev17 fresh selection lacks exact anchors")
    b_ratio = float(selection["B_protection_ratio"])
    support_minimum = float(selection["minimum_effective_support_per_mode_per_network"])
    evaluations = []
    for candidate_id, candidate in candidates.items():
        if not candidate["selection_eligible"]:
            continue
        candidate_rows = sorted(
            [row for row in rows if row["candidate_id"] == candidate_id],
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in candidate_rows] != sorted(seeds):
            raise RuntimeError(f"rev17 fresh selection is incomplete: {candidate_id}")
        paired = []
        for row in candidate_rows:
            seed = int(row["seed"])
            reference = exact[seed]
            paired.append({
                "seed": seed,
                "delta_J14": float(row["j14"]) - float(reference["j14"]),
                "delta_A": float(row["mode_0_mean"]) - float(reference["mode_0_mean"]),
                "B_ratio": float(row["mode_1_mean"]) / max(
                    float(reference["mode_1_mean"]), 1e-12
                ),
                "A_support": float(row["mode_0_effective_events"]),
                "B_support": float(row["mode_1_effective_events"]),
                "valid": row["run_status"] == "VALID",
            })
        gates = {
            "valid_all_networks": all(row["valid"] for row in paired),
            "J14_improves_all_networks": all(row["delta_J14"] < 0.0 for row in paired),
            "A_improves_all_networks": all(row["delta_A"] < 0.0 for row in paired),
            "B_protected_all_networks": all(row["B_ratio"] <= b_ratio for row in paired),
            "A_support_all_networks": all(row["A_support"] >= support_minimum for row in paired),
            "B_support_all_networks": all(row["B_support"] >= support_minimum for row in paired),
        }
        record = {
            "candidate_id": candidate_id,
            "family": (candidate.get("residual_coordinates") or {}).get("family"),
            "radius": (candidate.get("residual_coordinates") or {}).get(
                "joint_dual_field_radius"
            ),
            "passes_all_fresh_selection_gates": bool(all(gates.values())),
            **gates,
            "worst_delta_J14": float(max(row["delta_J14"] for row in paired)),
            "mean_delta_J14": float(np.mean([row["delta_J14"] for row in paired])),
            "worst_delta_A": float(max(row["delta_A"] for row in paired)),
            "mean_delta_A": float(np.mean([row["delta_A"] for row in paired])),
            "worst_B_ratio": float(max(row["B_ratio"] for row in paired)),
            "minimum_A_support": float(min(row["A_support"] for row in paired)),
            "minimum_B_support": float(min(row["B_support"] for row in paired)),
            "paired_networks": paired,
        }
        evaluations.append(record)
    eligible = [row for row in evaluations if row["passes_all_fresh_selection_gates"]]
    eligible.sort(key=lambda row: (
        row["worst_delta_J14"], row["worst_delta_A"],
        row["mean_delta_J14"], row["mean_delta_A"],
        float(row["radius"]), row["candidate_id"],
    ))
    for rank, row in enumerate(eligible, start=1):
        row["fresh_selection_rank"] = rank
    return evaluations, eligible


def aggregate(
    config_path: Path = DEFAULT_CONFIG, root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config, manifest, _ = _load_contract(config_path.resolve(), root.resolve())
    provenance = atlas._analysis_provenance(manifest)
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    seeds = [int(seed) for seed in config["search"]["selection_network_seeds"]]
    worker_root = root / config["output_root"] / "workers"
    expected = len(candidates) * len(seeds)
    missing = [
        f"{candidate_id}:{seed}"
        for candidate_id in candidates for seed in seeds
        if not (worker_root / f"{candidate_id}_seed_{seed}.json").is_file()
    ]
    validated, invalid, scored_rows = [], [], []
    if not missing:
        positions, stage = atlas._reference_geometry(config, manifest, root.resolve())
        for candidate_id, candidate in candidates.items():
            for seed in seeds:
                path = (worker_root / f"{candidate_id}_seed_{seed}.json").resolve()
                try:
                    validated.append(atlas._validate_worker(
                        path, candidate, seed, config, manifest,
                        positions[seed], stage, root.resolve(),
                    ))
                except Exception as error:
                    invalid.append(
                        f"{candidate_id}:{seed}:{type(error).__name__}:{error}"
                    )
    status, error = "INCOMPLETE", None
    evaluations, eligible = [], []
    if not provenance["analysis_worktree_clean"]:
        status, error = "INVALID_PROVENANCE", "analysis worktree is dirty"
    elif not missing and not invalid and len(validated) == expected:
        try:
            j14 = json.loads(atlas._resolve(
                root, config["inputs"]["j14_config"]["path"]
            ).read_text())
            context = historical._patient_context(j14, root.resolve())
            support = canary._load_support_context(
                atlas._resolve(root, config["inputs"]["patient_support_config"]["path"]),
                root.resolve(), j14,
            )
            scored_rows = []
            for record in validated:
                candidate = candidates[record["candidate_id"]]
                row = flat._flat_row(
                    canary._score_worker(record, context, support), candidate,
                )
                coordinates = candidate.get("residual_coordinates") or {}
                row.update({
                    "family": coordinates.get("family"),
                    "radius": coordinates.get("joint_dual_field_radius"),
                })
                scored_rows.append(row)
            evaluations, eligible = evaluate_candidates(
                scored_rows, candidates, seeds, config["selection"],
            )
            status = STATUS
        except Exception as caught:
            status, error = "INVALID_INPUT", str(caught)
    inventory = {
        "expected_runs": expected, "present_validated": len(validated),
        "missing": missing, "invalid_artifact": invalid,
        "complete_cartesian_product": len(validated) == expected and not missing and not invalid,
    }
    output_root = root / config["output_root"] / "analysis"
    json_path = output_root / "fresh_selection_aggregate.json"
    runs_csv = output_root / "fresh_selection_scored_runs.csv"
    candidate_csv = output_root / "fresh_selection_candidates.csv"
    payload = {
        "schema_id": "topic4_rev17_dual_field_fresh_selection_aggregate_v1",
        "status": status, "input_error": error,
        "inventory": inventory, "provenance": provenance,
        "scored_runs": scored_rows, "candidate_evaluations": evaluations,
        "eligible_candidates": eligible,
        "selected_candidate_id": eligible[0]["candidate_id"] if eligible else None,
        "selection_contract": config["selection"],
        "boundaries": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off", "SNN_simulation_run_by_aggregator": False,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {
            "json": str(json_path), "runs_csv": str(runs_csv),
            "candidate_csv": str(candidate_csv),
        },
    }
    _atomic_json(json_path, payload)
    _atomic_csv(runs_csv, scored_rows)
    flat_evaluations = [
        {key: value for key, value in row.items() if key != "paired_networks"}
        for row in evaluations
    ]
    _atomic_csv(candidate_csv, flat_evaluations)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    payload = aggregate(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "expected_runs": payload["inventory"]["expected_runs"],
        "eligible_candidates": len(payload["eligible_candidates"]),
        "selected_candidate_id": payload["selected_candidate_id"],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
