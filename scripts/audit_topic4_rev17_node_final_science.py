#!/usr/bin/env python3
"""Run the one-time rev17 held-out and source-topology audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev17_node_final_science.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_topic4_rev15_node_final_science as base  # noqa: E402
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract, _old_to_patient_label_map, _patient_data,
    _reorder_patient_contract, score_candidate,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales, fixed_projection_matrix,
)
from src.topic4_node_final_science import (  # noqa: E402
    final_zero_simulation_decision, score_workers_against_patient_endpoint,
    topology_label_permutation_test,
)


WORKER_STATUS = "REV12ND_NODE_WORKER_COMPLETE"


def audit(config_path: Path = DEFAULT_CONFIG,
          root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config_path = config_path.resolve(); root = root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_node_final_science_v1":
        raise RuntimeError("rev17 final-science schema changed")
    if config["boundaries"].get("field_reranking_allowed") is not False or config[
        "boundaries"
    ].get("EE_EtoI_ZM") != "off":
        raise RuntimeError("rev17 final-science crossed a boundary")
    paths = {name: base._resolve(record, root) for name, record in config["inputs"].items()}
    confirmation = json.loads(paths["confirmation_config"].read_text())
    manifest = json.loads(paths["confirmation_manifest"].read_text())
    confirmation_audit = json.loads(paths["confirmation_audit"].read_text())
    postselection_config = json.loads(paths["postselection_config"].read_text())
    postselection = json.loads(paths["postselection_audit"].read_text())
    rev12 = json.loads(paths["rev12_config"].read_text())
    candidate_id = config["selected_candidate"]["candidate_id"]
    reference_id = config["selected_candidate"]["paired_reference_candidate_id"]
    if (
        confirmation_audit.get("scientific_confirmation", {}).get("accepted") is not True
        or postselection.get("status") != "REV17_NODE_POSTSELECTION_ACCEPTED"
        or postselection.get("candidate_id") != candidate_id
        or postselection_config["selected_candidate"]["candidate_id"] != candidate_id
    ):
        raise RuntimeError("rev17 final-science prerequisites changed")
    if manifest.get("config_sha256") != base._sha256(paths["confirmation_config"]):
        raise RuntimeError("rev17 confirmation manifest/config identity changed")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or reference_id not in candidates:
        raise RuntimeError("rev17 final-science candidates are absent")
    for name in ("cohort_config", "classifier_config"):
        if rev12["inputs"][name]["sha256"] != config["inputs"][name]["sha256"]:
            raise RuntimeError(f"rev17 {name} identity changed")
    cohort = json.loads(paths["cohort_config"].read_text())
    classifier_config = json.loads(paths["classifier_config"].read_text())
    patient = _patient_data(cohort, root)
    classifier = _classifier_contract(classifier_config, root)
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
    candidate_workers, maps, labels, candidate_inputs = base._candidate_workers(
        candidate_id, robust_config=confirmation, seeds=seeds,
        patient=patient, classifier=classifier,
        label_map=semantics["raw_to_patient"], artifact_root=root,
        expected_coefficients_sha256=None,
        expected_mapping_sha256=config["selected_candidate"]["mapping_sha256"],
        expected_worker_status=WORKER_STATUS,
    )
    reference_workers, reference_maps, reference_labels, reference_inputs = (
        base._candidate_workers(
        reference_id, robust_config=confirmation, seeds=seeds,
        patient=patient, classifier=classifier,
        label_map=semantics["raw_to_patient"], artifact_root=root,
        expected_coefficients_sha256=None,
        expected_mapping_sha256=config["selected_candidate"][
            "reference_mapping_sha256"
        ],
        expected_worker_status=WORKER_STATUS,
    ))
    postselection_workers = {
        int(row["network_seed"]): row for row in postselection["inputs"]["workers"]
    }
    for row in candidate_inputs:
        if row["npz"]["sha256"] != postselection_workers[int(row["seed"])][
            "npz"
        ]["sha256"]:
            raise RuntimeError("rev17 held-out worker differs from KMeans audit")
    candidate_training = score_candidate(
        candidates[candidate_id], candidate_workers, patient, projections, calibration,
    )
    reference_training = score_candidate(
        candidates[reference_id], reference_workers, patient, projections, calibration,
    )
    candidate_heldout = score_workers_against_patient_endpoint(
        candidate_workers, patient_ranks=patient["heldout_ranks"],
        patient_labels=patient["heldout_labels"],
        contact_names=patient["contact_names"], projections=projections,
        calibration=calibration,
    )
    reference_heldout = score_workers_against_patient_endpoint(
        reference_workers, patient_ranks=patient["heldout_ranks"],
        patient_labels=patient["heldout_labels"],
        contact_names=patient["contact_names"], projections=projections,
        calibration=calibration,
    )
    endpoint_keys = (
        "mean_weakest_mode_lse", "mean_mode_0_loss", "mean_mode_1_loss",
        "mean_weakest_mode_cloud_lse", "mean_mode_0_cloud_loss",
        "mean_mode_1_cloud_loss",
    )
    candidate_endpoint = {
        "heldout_eventwise_prototype_r2": candidate_training[
            "model_prototype_r2_on_heldout"
        ],
        **{key: candidate_heldout[key] for key in endpoint_keys},
    }
    reference_endpoint = {
        "heldout_eventwise_prototype_r2": reference_training[
            "model_prototype_r2_on_heldout"
        ],
        **{key: reference_heldout[key] for key in endpoint_keys},
    }
    topology = topology_label_permutation_test(
        maps, labels, draws=int(config["source_topology"]["permutation_draws"]),
        seed=int(config["source_topology"]["permutation_seed"]),
    )
    reference_topology = topology_label_permutation_test(
        reference_maps, reference_labels,
        draws=int(config["source_topology"]["permutation_draws"]),
        seed=int(config["source_topology"]["permutation_seed"]),
    )
    decision = final_zero_simulation_decision(
        candidate_endpoint, reference_endpoint, topology,
        reference_topology_test=reference_topology,
    )
    paired = base._paired_deltas(candidate_heldout, reference_heldout)
    provenance = base._provenance()
    status = (
        "INVALID_PROVENANCE" if not provenance["formal_ready"]
        else "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
        if decision["accepted_for_same_checkpoint_intervention"]
        else "REV17_NODE_FINAL_SCIENCE_REJECTED"
    )
    output_root = root / config["output_root"] / "analysis"
    output_json = output_root / "node_final_science_audit.json"
    output_csv = output_root / "node_final_science_paired_networks.csv"
    payload = {
        "schema_id": "topic4_rev17_node_final_science_audit_v1",
        "status": status, "candidate_id": candidate_id,
        "paired_reference_candidate_id": reference_id,
        "candidate_score": {
            "patient_training_target": candidate_training,
            "patient_heldout_target": candidate_heldout,
            "final_endpoint": candidate_endpoint,
        },
        "reference_score": {
            "patient_training_target": reference_training,
            "patient_heldout_target": reference_heldout,
            "final_endpoint": reference_endpoint,
        },
        "paired_network_deltas": paired,
        "source_topology_permutation": topology,
        "reference_source_topology_permutation": reference_topology,
        "decision": decision,
        "patient_data_identity": {
            "contact_names_sha256": base._array_sha256(patient["contact_names"]),
            "train_indices_sha256": base._array_sha256(patient["train_event_indices"]),
            "heldout_indices_sha256": base._array_sha256(patient["heldout_event_indices"]),
            "heldout_event_count": int(len(patient["heldout_ranks"])),
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": base._sha256(config_path)},
            "hashed_inputs": config["inputs"],
            "candidate_workers": candidate_inputs,
            "reference_workers": reference_inputs,
        },
        "provenance": provenance,
        "boundaries": {
            **config["boundaries"], "field_reranking_performed": False,
            "SNN_simulation_run": False, "same_checkpoint_intervention_run": False,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_json), "paired_csv": str(output_csv)},
    }
    base._atomic_json(output_json, payload)
    output_root.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"], "candidate_id": payload["candidate_id"],
        "advances_to_intervention": payload["decision"][
            "accepted_for_same_checkpoint_intervention"
        ],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
