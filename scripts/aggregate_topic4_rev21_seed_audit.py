#!/usr/bin/env python3
"""Score and decompose the rev21 topology x dynamics seed audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev20_dual_core_endpoint import (  # noqa: E402
    score_complete_distribution, score_validation_endpoints,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, path: str) -> Path:
    local = ROOT / path
    return local if local.exists() else artifact_root / path


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]).copy() for key in handle.files}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def crossed_variance(values) -> dict:
    """Two-way balanced decomposition; residual includes interaction and noise."""
    matrix = np.asarray(values, float)
    if matrix.ndim != 2 or matrix.size < 4 or not np.isfinite(matrix).all():
        return {"status": "NOT_ESTIMABLE"}
    grand = float(np.mean(matrix))
    topology_mean = np.mean(matrix, axis=1)
    dynamics_mean = np.mean(matrix, axis=0)
    ss_total = float(np.sum((matrix - grand) ** 2))
    ss_topology = float(matrix.shape[1] * np.sum((topology_mean - grand) ** 2))
    ss_dynamics = float(matrix.shape[0] * np.sum((dynamics_mean - grand) ** 2))
    ss_residual = max(0.0, ss_total - ss_topology - ss_dynamics)
    denominator = max(ss_total, np.finfo(float).eps)
    return {
        "status": "OK",
        "grand_mean": grand,
        "topology_means": topology_mean.tolist(),
        "dynamics_means": dynamics_mean.tolist(),
        "ss_total": ss_total,
        "ss_topology": ss_topology,
        "ss_dynamics": ss_dynamics,
        "ss_residual_interaction": ss_residual,
        "variance_share_topology": ss_topology / denominator,
        "variance_share_dynamics": ss_dynamics / denominator,
        "variance_share_residual_interaction": ss_residual / denominator,
        "range": [float(np.min(matrix)), float(np.max(matrix))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    output_root = artifact_root / config["output_root"]
    status = json.loads((output_root / "seed_audit/status/controller.json").read_text())
    if status.get("status") != "COMPLETE":
        raise RuntimeError("seed audit controller is not complete")

    training = _load_npz(_resolve(
        artifact_root, config["inputs"]["patient_training_target"]["path"],
    ))
    support_config = json.loads(_resolve(
        artifact_root, config["inputs"]["patient_support_config"]["path"],
    ).read_text())
    contract_path = _resolve(
        artifact_root, support_config["inputs"]["contact_contract"]["path"],
    )
    if _sha256(contract_path) != support_config["inputs"]["contact_contract"]["sha256"]:
        raise RuntimeError("contact contract changed")
    contract = json.loads(contract_path.read_text())
    classifier_path = _resolve(
        artifact_root,
        support_config["inputs"]["old_ab_train_only_classifier"]["path"],
    )
    if _sha256(classifier_path) != support_config["inputs"][
            "old_ab_train_only_classifier"]["sha256"]:
        raise RuntimeError("direction classifier changed")
    classifier = json.loads(classifier_path.read_text())["direction_classifier"]
    topology_seeds = [int(v) for v in config["search"]["canary_network_seeds"]]
    dynamics_seeds = [int(v) for v in config["search"]["seed_audit_dynamics_seeds"]]
    rows = []
    for topology in topology_seeds:
        for dynamics in dynamics_seeds:
            stem = (output_root / "seed_audit/workers"
                    / f"rev21_zm_off_topology_{topology}_dynamics_{dynamics}")
            payload = json.loads(stem.with_suffix(".json").read_text())
            arrays = _load_npz(stem.with_suffix(".npz"))
            if (payload.get("topology_seed") != topology
                    or payload.get("dynamics_seed") != dynamics):
                raise RuntimeError("worker seed identity changed")
            selection = score_complete_distribution(
                arrays["onsets"], arrays["event_returned"],
                contract=contract, training_arrays=training,
            )
            validation = score_validation_endpoints(
                arrays["onsets"], arrays["ranks"], arrays["event_returned"],
                contract=contract, training_arrays=training,
                classifier=classifier,
                kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
            )
            rows.append({
                "topology_seed": topology, "dynamics_seed": dynamics,
                "worker_json": str(stem.with_suffix(".json")),
                "worker_npz_sha256": _sha256(stem.with_suffix(".npz")),
                "selection": selection, "validation": validation,
            })

    def matrix(accessor):
        lookup = {(row["topology_seed"], row["dynamics_seed"]): accessor(row)
                  for row in rows}
        return [[lookup[(topology, dynamics)] for dynamics in dynamics_seeds]
                for topology in topology_seeds]

    endpoints = {
        "training_complete_distribution": matrix(
            lambda row: row["selection"]["complete_distribution_distance_training"]),
        "two_template_alignment": matrix(
            lambda row: row["validation"].get("direction_balanced_alignment")),
        "ood_all_returned": matrix(
            lambda row: row["validation"]["ood_all_returned"]),
        "returned_family_count": matrix(
            lambda row: row["selection"]["n_returned_families"]),
    }
    decomposition = {name: crossed_variance(values)
                     for name, values in endpoints.items()}
    natural_ok = sum(
        row["validation"]["natural_kmeans_status"] == "OK" for row in rows
    )
    both_clusters = sum(bool(row["validation"].get("two_clusters_present"))
                        for row in rows)
    payload = {
        "schema_id": "topic4_rev21_seed_factorization_audit_v1",
        "status": "REV21_SEED_FACTORIZATION_COMPLETE",
        "config_sha256": _sha256(config_path),
        "topology_seeds": topology_seeds,
        "dynamics_seeds": dynamics_seeds,
        "matrix_shape": [len(topology_seeds), len(dynamics_seeds)],
        "per_cell": rows,
        "endpoint_matrices": endpoints,
        "variance_decomposition": decomposition,
        "natural_kmeans_ok_cells": natural_ok,
        "both_clusters_present_cells": both_clusters,
        "patient_ictal_inputs_read": False,
        "patient_heldout_opened": False,
        "interpretation_boundary": (
            "topology and dynamics are crossed fixed audit factors; residual contains "
            "their interaction because there is one run per cell; patient held-out "
            "events remain sealed until the Z/M work point is frozen"
        ),
    }
    output = output_root / "seed_audit/seed_factorization_audit.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "natural_kmeans_ok": natural_ok,
                      "both_clusters": both_clusters}, indent=2))


if __name__ == "__main__":
    main()
