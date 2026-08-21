#!/usr/bin/env python3
"""Aggregate rev12-ND Node-only workers without pooling away network identity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
    score_candidate,
)
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
    topology_network_reproducibility,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _strip_natural_arrays(result: dict) -> dict:
    return {
        key: value for key, value in result.items()
        if key not in {"valid_event_mask", "cluster_labels"}
    }


def _source_bundle(npz_path: Path, worker: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as loaded:
        returned = np.asarray(loaded["event_returned"], bool)
        maps = np.asarray(loaded["source_onset_maps_ms"], float)[returned]
        evaluable = np.asarray(loaded["source_onset_evaluable"], bool)[returned]
    if len(maps) != len(worker["labels"]):
        raise RuntimeError(f"source maps and returned events differ: {npz_path}")
    return maps[evaluable], np.asarray(worker["labels"], int)[evaluable]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_rev12_nd_node_dualmode_refit.json",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--seed-pool", choices=("canary", "fit", "selection", "confirmation"),
        default="canary",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = artifact_root / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("candidate manifest is stale")

    cohort_path = artifact_root / config["inputs"]["cohort_config"]["path"]
    classifier_path = artifact_root / config["inputs"]["classifier_config"]["path"]
    cohort = json.loads(cohort_path.read_text())
    classifier_config = json.loads(classifier_path.read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]), n_directions=64, seed=20260821,
    )
    calibration = calibrate_component_scales(
        patient["train_ranks"], patient["train_labels"], patient["train_blocks"],
        patient["contact_names"], projections, sample_size=6, draws=256,
        seed=20260821,
    )
    seeds = [
        int(seed) for seed in config["search"][f"{args.seed_pool}_network_seeds"]
    ]
    output_root = artifact_root / config["output_root"]
    rows = []
    for candidate in manifest["candidates"]:
        workers, source_maps, source_labels, per_seed = [], [], [], []
        for seed in seeds:
            stem = f"{candidate['candidate_id']}_seed_{seed}"
            npz_path = output_root / "workers" / f"{stem}.npz"
            if not npz_path.exists() or not npz_path.with_suffix(".json").exists():
                continue
            worker = _load_network_worker(
                npz_path, patient["contact_names"], classifier,
                semantics["raw_to_patient"],
            )
            maps, labels = _source_bundle(npz_path, worker)
            workers.append(worker)
            source_maps.append(maps)
            source_labels.append(labels)
            natural = natural_kmeans(
                worker["ranks"], worker["labels"], random_state=seed,
            )
            per_seed.append({
                "seed": seed,
                "n_returned_events": worker["n_returned"],
                "patient_mode_counts": np.bincount(
                    worker["labels"], minlength=2,
                ),
                "ood_fraction": (
                    float(np.mean(worker["ood"])) if len(worker["ood"]) else 1.0
                ),
                "natural_kmeans": _strip_natural_arrays(natural),
                "n_source_maps": int(len(maps)),
                "worker_npz": str(npz_path),
                "worker_npz_sha256": _sha256(npz_path),
            })
        if not workers:
            continue
        score = score_candidate(
            candidate, workers, patient, projections, calibration,
        )
        topology = topology_network_reproducibility(source_maps, source_labels)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "field_sha256": candidate["node_field"]["field_sha256"],
            "n_networks": len(workers),
            "score": score,
            "source_topology": topology,
            "per_seed": per_seed,
        })
    if not rows:
        raise RuntimeError(f"no completed {args.seed_pool} workers")
    rows.sort(key=lambda row: (
        row["score"]["mean_patient_objective"],
        -row["score"]["model_prototype_r2_on_heldout"],
        row["candidate_id"],
    ))
    payload = {
        "schema_id": "topic4_rev12_nd_node_aggregate_v1",
        "status": "REV12ND_NODE_AGGREGATE_COMPLETE",
        "scientific_role": "development_only_node_dualmode_refit",
        "seed_pool": args.seed_pool,
        "requested_seeds": seeds,
        "rows": rows,
        "component_calibration": calibration,
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": _sha256(manifest_path),
        },
        "claim_boundary": (
            "Patient assignment and natural KMeans are separate. Complete returned "
            "events are primary. Source topology is summarized per network and then "
            "equal-network; a single canary network cannot establish cross-network "
            "reproducibility."
        ),
    }
    aggregate_dir = output_root / "aggregate"
    _atomic_json(aggregate_dir / f"{args.seed_pool}_summary.json", payload)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    with (aggregate_dir / f"{args.seed_pool}_summary.csv").open("w", newline="") as handle:
        fields = [
            "candidate_id", "field_sha256", "n_networks", "mean_events",
            "same_network_both_fraction", "patient_objective", "mode_0_loss",
            "mode_1_loss", "heldout_r2", "topology_within_network",
            "topology_across_network", "topology_between_mode_distance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            score = row["score"]
            topology = row["source_topology"]
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "field_sha256": row["field_sha256"],
                "n_networks": row["n_networks"],
                "mean_events": score["mean_returned_events"],
                "same_network_both_fraction": score["same_network_both_fraction"],
                "patient_objective": score["mean_patient_objective"],
                "mode_0_loss": score["mean_mode_0_loss"],
                "mode_1_loss": score["mean_mode_1_loss"],
                "heldout_r2": score["model_prototype_r2_on_heldout"],
                "topology_within_network": topology[
                    "mean_within_network_split_half_cosine"
                ],
                "topology_across_network": topology[
                    "mean_across_network_template_cosine"
                ],
                "topology_between_mode_distance": topology[
                    "equal_network_between_mode_distance"
                ],
            })
    print(json.dumps({
        "status": payload["status"], "seed_pool": args.seed_pool,
        "n_candidates": len(rows), "output": str(aggregate_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
