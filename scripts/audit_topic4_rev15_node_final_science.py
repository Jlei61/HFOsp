#!/usr/bin/env python3
"""Run the one-time held-out and source-topology audit for a frozen Node field."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
    score_candidate,
)
from scripts.run_topic4_rev12_node_intervention import (  # noqa: E402
    _source_bundle,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
)
from src.topic4_node_final_science import (  # noqa: E402
    final_zero_simulation_decision,
    score_workers_against_patient_endpoint,
    topology_label_permutation_test,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_node_final_science.json"
EXPECTED_CONFIG_SCHEMA = "topic4_rev15_node_final_science_v1"
EXPECTED_WORKER_STATUS = "REV15_M3_ROBUST_CANDIDATE_WORKER_COMPLETE"
OUTPUT_SCHEMA = "topic4_rev15_node_final_science_audit_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def _resolve(record: Mapping[str, Any], artifact_root: Path) -> Path:
    for root in (artifact_root, ROOT):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"final-science input changed: {record['path']}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _provenance() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    return {"analysis_commit": head, "worktree_status": status,
            "formal_ready": not status, "SNN_simulation_run": False}


def _candidate_workers(
    candidate_id: str, *, robust_config: Mapping[str, Any],
    seeds: list[int], patient: Mapping[str, Any], classifier: Mapping[str, Any],
    label_map: np.ndarray, artifact_root: Path,
    expected_coefficients_sha256: str | None,
    expected_mapping_sha256: str | None = None,
    expected_worker_status: str | None = None,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray], list[dict]]:
    worker_root = artifact_root / robust_config["output_root"] / "workers"
    workers, maps, labels, inputs = [], [], [], []
    for seed in seeds:
        npz_path = worker_root / f"{candidate_id}_seed_{seed}.npz"
        json_path = npz_path.with_suffix(".json")
        if not npz_path.is_file() or not json_path.is_file():
            raise RuntimeError(f"final-science worker is missing: {candidate_id}:{seed}")
        payload = json.loads(json_path.read_text())
        worker_status = expected_worker_status or EXPECTED_WORKER_STATUS
        if payload.get("status") != worker_status:
            raise RuntimeError(f"final-science worker is incomplete: {candidate_id}:{seed}")
        if payload.get("candidate_id") != candidate_id or int(payload.get("seed")) != seed:
            raise RuntimeError("final-science worker identity changed")
        mechanism = payload.get("mechanism_freeze", {})
        if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
            raise RuntimeError("final-science worker activated another mechanism")
        simulation = payload.get("simulation", {})
        if float(simulation.get("duration_ms", np.nan)) != 20000.0:
            raise RuntimeError("final-science worker duration changed")
        if simulation.get("runaway_early_stop_ms") is not None:
            raise RuntimeError("final-science worker ended in runaway")
        provenance = payload.get("provenance", {})
        if (int(provenance.get("runtime_modules_dirty", 1)) != 0
                or int(provenance.get("runtime_modules_match_expected_commit", 0)) != 1):
            raise RuntimeError("final-science worker provenance is not clean")
        array_record = payload.get("arrays", {})
        if _sha256(npz_path) != array_record.get("sha256"):
            raise RuntimeError("final-science worker array hash changed")
        coordinate = payload.get("fourier_field")
        observed_coefficients = (
            None if coordinate is None else coordinate.get("coefficients_sha256")
        )
        if expected_mapping_sha256 is None:
            if observed_coefficients != expected_coefficients_sha256:
                raise RuntimeError("final-science worker field coordinate changed")
        else:
            if expected_coefficients_sha256 is not None:
                raise ValueError(
                    "final-science worker identity must use one field contract"
                )
            if payload.get("node_mapping", {}).get(
                "mapping_sha256"
            ) != expected_mapping_sha256:
                raise RuntimeError("final-science worker dual mapping changed")
        worker = _load_network_worker(
            npz_path, patient["contact_names"], classifier, label_map,
        )
        source_maps, source_labels = _source_bundle(npz_path, worker)
        workers.append(worker)
        maps.append(source_maps)
        labels.append(source_labels)
        inputs.append({
            "seed": int(seed),
            "json": {"path": str(json_path), "sha256": _sha256(json_path)},
            "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        })
    return workers, maps, labels, inputs


def _paired_deltas(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict]:
    candidate_rows = {int(row["seed"]): row for row in candidate["network_scores"]}
    reference_rows = {int(row["seed"]): row for row in reference["network_scores"]}
    output = []
    for seed in sorted(candidate_rows):
        left, right = candidate_rows[seed], reference_rows[seed]
        output.append({
            "network_seed": seed,
            "delta_weakest_mode_loss": float(left["weakest_mode_lse"] - right["weakest_mode_lse"]),
            "delta_mode_0_loss": float(left["modes"]["0"]["mean"] - right["modes"]["0"]["mean"]),
            "delta_mode_1_loss": float(left["modes"]["1"]["mean"] - right["modes"]["1"]["mean"]),
            "delta_weakest_mode_cloud_loss": float(
                left["weakest_mode_cloud_lse"] - right["weakest_mode_cloud_lse"]
            ),
            "delta_mode_0_cloud_loss": float(
                left["modes"]["0"]["cloud"] - right["modes"]["0"]["cloud"]
            ),
            "delta_mode_1_cloud_loss": float(
                left["modes"]["1"]["cloud"] - right["modes"]["1"]["cloud"]
            ),
            "candidate_mode_counts": np.asarray(left["mode_counts"], int).tolist(),
            "reference_mode_counts": np.asarray(right["mode_counts"], int).tolist(),
        })
    return output


def audit(config_path: Path = DEFAULT_CONFIG,
          artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != EXPECTED_CONFIG_SCHEMA:
        raise RuntimeError("final-science config schema changed")
    if config["boundaries"].get("field_reranking_allowed") is not False:
        raise RuntimeError("final-science config allows field reranking")
    if config["boundaries"].get("EE_EtoI_ZM") != "off":
        raise RuntimeError("final-science config activates another mechanism")
    paths = {name: _resolve(record, artifact_root)
             for name, record in config["inputs"].items()}
    postselection = json.loads(paths["postselection_audit"].read_text())
    if postselection.get("status") != "NODE_POSTSELECTION_ACCEPTED":
        raise RuntimeError("post-selection acceptance changed")
    candidate_id = str(config["selected_candidate"]["candidate_id"])
    if postselection.get("candidate_id") != candidate_id:
        raise RuntimeError("final-science candidate identity changed")
    robust_config = json.loads(paths["robust_config"].read_text())
    manifest = json.loads(paths["robust_manifest"].read_text())
    robust_aggregate = json.loads(paths["robust_aggregate"].read_text())
    postselection_config = json.loads(paths["postselection_config"].read_text())
    rev12_config = json.loads(paths["rev12_config"].read_text())
    if robust_aggregate.get("status") != "COMPLETE":
        raise RuntimeError("robust aggregate changed or became incomplete")
    if robust_aggregate.get("best_usable_anchor") != candidate_id:
        raise RuntimeError("robust ranking candidate changed")
    if postselection_config["selected_candidate"]["candidate_id"] != candidate_id:
        raise RuntimeError("post-selection config candidate changed")
    if manifest.get("config_sha256") != _sha256(paths["robust_config"]):
        raise RuntimeError("robust manifest/config identity changed")
    if postselection.get("inputs", {}).get("config", {}).get("sha256") != _sha256(
        paths["postselection_config"]
    ):
        raise RuntimeError("post-selection audit/config identity changed")
    for name in ("cohort_config", "classifier_config"):
        if rev12_config["inputs"][name]["sha256"] != config["inputs"][name]["sha256"]:
            raise RuntimeError(f"rev12 {name} identity changed")
    if [int(seed) for seed in robust_config["search"]["active_network_seeds"]] != [
        int(seed) for seed in config["network_seeds"]
    ]:
        raise RuntimeError("final-science network seeds changed")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or "exact_off" not in candidates:
        raise RuntimeError("selected or exact field is absent")
    if candidates[candidate_id]["fourier_coordinate"]["coefficients_sha256"] != (
        config["selected_candidate"]["field_coefficients_sha256"]
    ):
        raise RuntimeError("selected field coefficient hash changed")

    cohort = json.loads(paths["cohort_config"].read_text())
    classifier_config = json.loads(paths["classifier_config"].read_text())
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
    seeds = [int(seed) for seed in config["network_seeds"]]
    candidate_workers, maps, labels, candidate_inputs = _candidate_workers(
        candidate_id, robust_config=robust_config, seeds=seeds,
        patient=patient, classifier=classifier,
        label_map=semantics["raw_to_patient"], artifact_root=artifact_root,
        expected_coefficients_sha256=candidates[candidate_id][
            "fourier_coordinate"
        ]["coefficients_sha256"],
    )
    reference_workers, _, _, reference_inputs = _candidate_workers(
        "exact_off", robust_config=robust_config, seeds=seeds,
        patient=patient, classifier=classifier,
        label_map=semantics["raw_to_patient"], artifact_root=artifact_root,
        expected_coefficients_sha256=None,
    )
    postselection_workers = {
        int(row["network_seed"]): row
        for row in postselection["inputs"]["workers"]
    }
    for row in candidate_inputs:
        expected = postselection_workers[int(row["seed"])]["npz"]["sha256"]
        if row["npz"]["sha256"] != expected:
            raise RuntimeError("final-science worker differs from KMeans post-selection")
    candidate_training_score = score_candidate(
        candidates[candidate_id], candidate_workers, patient, projections, calibration,
    )
    reference_training_score = score_candidate(
        candidates["exact_off"], reference_workers, patient, projections, calibration,
    )
    candidate_heldout_score = score_workers_against_patient_endpoint(
        candidate_workers,
        patient_ranks=patient["heldout_ranks"],
        patient_labels=patient["heldout_labels"],
        contact_names=patient["contact_names"],
        projections=projections,
        calibration=calibration,
    )
    reference_heldout_score = score_workers_against_patient_endpoint(
        reference_workers,
        patient_ranks=patient["heldout_ranks"],
        patient_labels=patient["heldout_labels"],
        contact_names=patient["contact_names"],
        projections=projections,
        calibration=calibration,
    )
    candidate_endpoint = {
        "heldout_eventwise_prototype_r2": candidate_training_score[
            "model_prototype_r2_on_heldout"
        ],
        **{
            key: candidate_heldout_score[key]
            for key in (
                "mean_weakest_mode_lse", "mean_mode_0_loss", "mean_mode_1_loss",
                "mean_weakest_mode_cloud_lse", "mean_mode_0_cloud_loss",
                "mean_mode_1_cloud_loss",
            )
        },
    }
    reference_endpoint = {
        "heldout_eventwise_prototype_r2": reference_training_score[
            "model_prototype_r2_on_heldout"
        ],
        **{
            key: reference_heldout_score[key]
            for key in (
                "mean_weakest_mode_lse", "mean_mode_0_loss", "mean_mode_1_loss",
                "mean_weakest_mode_cloud_lse", "mean_mode_0_cloud_loss",
                "mean_mode_1_cloud_loss",
            )
        },
    }
    topology = topology_label_permutation_test(
        maps, labels,
        draws=int(config["source_topology"]["permutation_draws"]),
        seed=int(config["source_topology"]["permutation_seed"]),
    )
    decision = final_zero_simulation_decision(
        candidate_endpoint, reference_endpoint, topology,
    )
    paired = _paired_deltas(candidate_heldout_score, reference_heldout_score)
    provenance = _provenance()
    status = (
        "INVALID_PROVENANCE" if not provenance["formal_ready"]
        else "NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
        if decision["accepted_for_same_checkpoint_intervention"]
        else "NODE_FINAL_SCIENCE_ZERO_SIMULATION_REJECTED"
    )
    output_root = artifact_root / config["output_root"] / "analysis"
    output_json = output_root / "node_final_science_audit.json"
    output_csv = output_root / "node_final_science_paired_networks.csv"
    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": status,
        "candidate_id": candidate_id,
        "paired_reference_candidate_id": "exact_off",
        "candidate_score": {
            "patient_training_target": candidate_training_score,
            "patient_heldout_target": candidate_heldout_score,
            "final_endpoint": candidate_endpoint,
        },
        "reference_score": {
            "patient_training_target": reference_training_score,
            "patient_heldout_target": reference_heldout_score,
            "final_endpoint": reference_endpoint,
        },
        "paired_network_deltas": paired,
        "source_topology_permutation": topology,
        "decision": decision,
        "patient_data_identity": {
            "contact_names_sha256": _array_sha256(patient["contact_names"]),
            "train_indices_sha256": _array_sha256(patient["train_event_indices"]),
            "heldout_indices_sha256": _array_sha256(patient["heldout_event_indices"]),
            "train_ranks_sha256": _array_sha256(patient["train_ranks"]),
            "heldout_ranks_sha256": _array_sha256(patient["heldout_ranks"]),
            "heldout_event_count": int(len(patient["heldout_ranks"])),
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "hashed_inputs": config["inputs"],
            "candidate_workers": candidate_inputs,
            "reference_workers": reference_inputs,
        },
        "provenance": provenance,
        "boundaries": {
            **config["boundaries"],
            "field_reranking_performed": False,
            "SNN_simulation_run": False,
            "same_checkpoint_intervention_run": False,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_json), "paired_csv": str(output_csv)},
    }
    _atomic_json(output_json, payload)
    output_root.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "advances_to_intervention": payload["decision"][
            "accepted_for_same_checkpoint_intervention"
        ],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
