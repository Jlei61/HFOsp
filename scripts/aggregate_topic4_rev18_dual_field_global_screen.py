#!/usr/bin/env python3
"""Score the rev18 global dual-field screen with KMeans-preserving fit loss."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev18_dual_field_global_screen.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev17_dual_field_residual_atlas as rev17  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.aggregate_topic4_rev12_node_canary import _strip_natural_arrays  # noqa: E402
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_rev18_dual_field_search import evaluate_candidate, nominate  # noqa: E402


STATUS = "REV18_DUAL_FIELD_GLOBAL_SCREEN_AGGREGATE_COMPLETE"
MANIFEST_STATUS = "REV18_DUAL_FIELD_GLOBAL_SCREEN_FROZEN"


def _load_contract(config_path: Path, root: Path) -> tuple[dict, dict, Path]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev18_dual_field_global_screen_v1":
        raise RuntimeError("rev18 config schema changed")
    for name, record in config["inputs"].items():
        path = rev17._resolve(root, record["path"])
        if not path.is_file() or rev17._sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev18 input changed: {name}")
    manifest_path = root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != MANIFEST_STATUS:
        raise RuntimeError("rev18 manifest is not frozen")
    if manifest.get("config_sha256") != rev17._sha256(config_path):
        raise RuntimeError("rev18 manifest/config hash mismatch")
    if not manifest.get("provenance", {}).get("formal_ready"):
        raise RuntimeError("rev18 manifest provenance is not formal")
    expected = int(config["dual_field_global_screen"][
        "expected_candidate_count_including_anchor"
    ])
    if len(manifest.get("candidates", [])) != expected:
        raise RuntimeError("rev18 manifest candidate count changed")
    return config, manifest, manifest_path


def _natural_summary(record: Mapping[str, Any], context: Mapping[str, Any],
                     seed: int, random_state: int) -> dict[str, Any]:
    arrays = record["arrays"]
    selection = historical.three_layer_event_selection(
        arrays, minimum_readable_contacts=int(context["minimum_readable_contacts"]),
    )
    primary = np.asarray(selection["contact_primary_indices"], dtype=np.int64)
    ranks = historical.exact._reorder_columns(
        np.asarray(arrays["ranks"], dtype=np.float64)[primary],
        arrays["contact_names"], context["patient"]["contact_names"],
    )
    onsets = historical.exact._reorder_columns(
        np.asarray(arrays["onsets"], dtype=np.float64)[primary],
        arrays["contact_names"], context["patient"]["contact_names"],
    )
    assignment = canary._assign_training_modes_from_full_timing(
        onsets, context["frozen_classifier"], context["groups"],
    )
    natural = natural_kmeans(
        ranks, np.asarray(assignment["labels"], dtype=int),
        random_state=int(random_state) + int(seed),
    )
    if natural.get("status") == "OK":
        readable = np.asarray(
            selection["fig4_kmeans_readable_within_contact"], dtype=bool,
        )
        if not np.array_equal(natural["valid_event_mask"], readable):
            raise RuntimeError("rev18 KMeans and Fig4 readable events differ")
    return _strip_natural_arrays(natural)


def _candidate_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    coordinates = candidate.get("residual_coordinates") or {}
    return {
        "family": coordinates.get("family"),
        "radius": coordinates.get("joint_dual_field_radius"),
        "mean_direction_l2": coordinates.get("mean_direction_l2"),
        "dispersion_direction_l2": coordinates.get("dispersion_direction_l2"),
    }


def aggregate(config_path: Path = DEFAULT_CONFIG,
              root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config, manifest, manifest_path = _load_contract(
        config_path.resolve(), root.resolve(),
    )
    provenance = rev17._analysis_provenance(manifest)
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    worker_root = root / config["output_root"] / "workers"
    missing = [
        f"{candidate}:{seed}" for candidate in candidates for seed in seeds
        if not (worker_root / f"{candidate}_seed_{seed}.json").is_file()
    ]
    validated, invalid, scored = [], [], []
    if not missing:
        positions, stage = rev17._reference_geometry(config, manifest, root.resolve())
        for candidate_id, candidate in candidates.items():
            for seed in seeds:
                path = (worker_root / f"{candidate_id}_seed_{seed}.json").resolve()
                try:
                    validated.append(rev17._validate_worker(
                        path, candidate, seed, config, manifest,
                        positions[seed], stage, root.resolve(),
                    ))
                except Exception as error:
                    invalid.append(
                        f"{candidate_id}:{seed}:{type(error).__name__}:{error}"
                    )
    status, input_error = "INCOMPLETE", None
    evaluations, nominees = [], []
    if not provenance["analysis_worktree_clean"]:
        status, input_error = "INVALID_PROVENANCE", "analysis worktree is dirty"
    elif not missing and not invalid and len(validated) == len(candidates) * len(seeds):
        try:
            j14_config = json.loads(rev17._resolve(
                root, config["inputs"]["j14_config"]["path"],
            ).read_text())
            context = historical._patient_context(j14_config, root.resolve())
            support = canary._load_support_context(
                rev17._resolve(root, config["inputs"]["patient_support_config"]["path"]),
                root.resolve(), j14_config,
            )
            random_state = int(config["robust_objective"]["natural_kmeans_seed"])
            for record in validated:
                candidate = candidates[record["candidate_id"]]
                row = rev17._flat(
                    canary._score_worker(record, context, support), candidate,
                )
                row["natural_kmeans"] = _natural_summary(
                    record, context, int(record["seed"]), random_state,
                )
                row.update(_candidate_metadata(candidate))
                scored.append(row)
            anchors = [row for row in scored if row["candidate_id"] == "exact_dual_anchor"]
            if len(anchors) != len(seeds):
                raise RuntimeError("rev18 exact anchor is incomplete")
            contract = config["robust_objective"]
            for candidate_id, candidate in candidates.items():
                rows = [row for row in scored if row["candidate_id"] == candidate_id]
                evaluation = evaluate_candidate(rows, anchors, contract)
                evaluation.update({"candidate_id": candidate_id, **_candidate_metadata(candidate)})
                evaluations.append(evaluation)
            anchor_eval = next(
                row for row in evaluations if row["candidate_id"] == "exact_dual_anchor"
            )
            for row in evaluations:
                row["delta_robust_loss_from_anchor"] = (
                    float(row["robust_loss"]) - float(anchor_eval["robust_loss"])
                )
            nominees = nominate(
                [row for row in evaluations if row["candidate_id"] != "exact_dual_anchor"],
                maximum_candidates=int(contract["maximum_nominees"]),
            )
            status = STATUS
        except Exception as error:
            status, input_error = "INVALID_INPUT", f"{type(error).__name__}: {error}"
    payload = {
        "schema_id": "topic4_rev18_dual_field_global_screen_aggregate_v1",
        "status": status,
        "input_error": input_error,
        "inventory": {
            "expected_runs": len(candidates) * len(seeds),
            "present_validated": len(validated),
            "missing": missing,
            "invalid_artifact": invalid,
            "complete_cartesian_product": bool(
                len(validated) == len(candidates) * len(seeds)
                and not missing and not invalid
            ),
        },
        "provenance": provenance,
        "scored_runs": scored,
        "candidate_evaluations": evaluations,
        "nominated_candidates": nominees,
        "robust_objective_contract": config["robust_objective"],
        "boundaries": config["boundaries"],
        "claim_boundary": config["claim_boundary"],
        "inputs": {
            "config": str(config_path),
            "config_sha256": rev17._sha256(config_path),
            "manifest": str(manifest_path),
            "manifest_sha256": rev17._sha256(manifest_path),
        },
    }
    output = root / config["output_root"] / "analysis"
    rev17._atomic_json(output / "global_screen_aggregate.json", payload)
    rev17._atomic_csv(output / "global_screen_scored_runs.csv", scored)
    flat = [{
        key: value for key, value in row.items() if key != "per_network"
    } for row in evaluations]
    rev17._atomic_csv(output / "global_screen_candidates.csv", flat)
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
        "nominated_candidates": [
            row["candidate_id"] for row in payload["nominated_candidates"]
        ],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
