#!/usr/bin/env python3
"""Aggregate fresh-seed rev21 confirmation and Z/M mechanism controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_topic4_rev21_zm_screen import (  # noqa: E402
    _atomic_json, _load_npz, _resolve, _sha256, model_ictal_or_control,
    reference_support, summarize_candidate,
)
from src.topic4_rev20_dual_core_endpoint import (  # noqa: E402
    score_complete_distribution, score_validation_endpoints,
)


MIN_ELIGIBLE_CELLS = 8
MIN_ROBUST_TOPOLOGIES = 2
MIN_ELIGIBLE_DYNAMICS_PER_TOPOLOGY = 2
JOINT_ID = "rev21_confirm_z_plus_m"
OFF_ID = "rev21_confirm_zm_off"


def confirmation_robustness(cells: list[dict]) -> dict:
    by_topology: dict[int, list[bool]] = {}
    for row in cells:
        by_topology.setdefault(int(row["topology_seed"]), []).append(
            row["model_ictal"].get("status") == "MODEL_ICTAL_ELIGIBLE_REV21"
        )
    counts = {str(seed): int(np.sum(values))
              for seed, values in sorted(by_topology.items())}
    robust = sum(
        count >= MIN_ELIGIBLE_DYNAMICS_PER_TOPOLOGY
        for count in counts.values()
    )
    total = sum(counts.values())
    return {
        "eligible_cells": int(total),
        "required_eligible_cells": MIN_ELIGIBLE_CELLS,
        "eligible_dynamics_by_topology": counts,
        "topologies_with_at_least_two_of_four": int(robust),
        "required_robust_topologies": MIN_ROBUST_TOPOLOGIES,
        "pass": bool(
            total >= MIN_ELIGIBLE_CELLS
            and robust >= MIN_ROBUST_TOPOLOGIES
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest", type=Path,
        help="Defaults to <output_root>/confirmation/candidate_manifest.json",
    )
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.read_text())
    output_root = artifact_root / config["output_root"]
    controller = json.loads((
        output_root / "confirmation/status/controller.json"
    ).read_text())
    if controller.get("status") != "COMPLETE":
        raise RuntimeError("confirmation controller is not complete")
    seed_audit = json.loads((
        output_root / "seed_audit/seed_factorization_audit.json"
    ).read_text())
    support = reference_support(seed_audit["endpoint_matrices"])

    training = _load_npz(_resolve(
        artifact_root, config["inputs"]["patient_training_target"]["path"],
    ))
    support_config = json.loads(_resolve(
        artifact_root, config["inputs"]["patient_support_config"]["path"],
    ).read_text())
    contract_path = _resolve(
        artifact_root, support_config["inputs"]["contact_contract"]["path"],
    )
    classifier_path = _resolve(
        artifact_root,
        support_config["inputs"]["old_ab_train_only_classifier"]["path"],
    )
    if (_sha256(contract_path)
            != support_config["inputs"]["contact_contract"]["sha256"]
            or _sha256(classifier_path)
            != support_config["inputs"]["old_ab_train_only_classifier"][
                "sha256"]):
        raise RuntimeError("training contact/classifier contract changed")
    contract = json.loads(contract_path.read_text())
    classifier = json.loads(classifier_path.read_text())["direction_classifier"]
    manifest_path = (
        args.candidate_manifest.resolve() if args.candidate_manifest
        else output_root / "confirmation/candidate_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(args.config.resolve()):
        raise RuntimeError("confirmation candidate manifest is stale")
    expected_ids = {row["candidate_id"] for row in manifest["candidates"]}
    if expected_ids != {
        JOINT_ID, "rev21_confirm_z_only", "rev21_confirm_m_only", OFF_ID,
    }:
        raise RuntimeError("confirmation arm set changed")

    cells = []
    arrays_by_candidate: dict[str, list[dict]] = {}
    for path in sorted((output_root / "confirmation/workers").glob("*.json")):
        worker = json.loads(path.read_text())
        arrays = _load_npz(Path(worker["arrays"]["path"]))
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
        cells.append({
            "candidate_id": worker["candidate_id"],
            "topology_seed": int(worker["topology_seed"]),
            "dynamics_seed": int(worker["dynamics_seed"]),
            "operational_onset_ms": worker["simulation"]["runaway_early_stop_ms"],
            "model_ictal": model_ictal_or_control(worker),
            "selection": selection,
            "validation": validation,
            "worker_json": str(path),
        })
        arrays_by_candidate.setdefault(worker["candidate_id"], []).append(arrays)
    grouped: dict[str, list[dict]] = {}
    for row in cells:
        grouped.setdefault(row["candidate_id"], []).append(row)
    if set(grouped) != expected_ids or any(len(rows) != 12 for rows in grouped.values()):
        raise RuntimeError("confirmation does not contain four complete 12-cell arms")

    off_lookup = {
        (row["topology_seed"], row["dynamics_seed"]): row
        for row in grouped[OFF_ID]
    }
    summaries = []
    for candidate_id, rows in grouped.items():
        group_arrays = arrays_by_candidate[candidate_id]
        pooled_onsets = np.concatenate([row["onsets"] for row in group_arrays])
        pooled_ranks = np.concatenate([row["ranks"] for row in group_arrays])
        pooled_returned = np.concatenate([
            row["event_returned"] for row in group_arrays
        ])
        pooled = {
            "selection": score_complete_distribution(
                pooled_onsets, pooled_returned, contract=contract,
                training_arrays=training,
            ),
            "validation": score_validation_endpoints(
                pooled_onsets, pooled_ranks, pooled_returned,
                contract=contract, training_arrays=training,
                classifier=classifier,
                kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
            ),
        }
        summaries.append(summarize_candidate(
            candidate_id, rows, off_lookup, support, pooled,
            candidate_id.removeprefix("rev21_confirm_"),
        ))
    joint = next(row for row in summaries if row["candidate_id"] == JOINT_ID)
    robustness = confirmation_robustness(grouped[JOINT_ID])
    established = bool(
        robustness["pass"] and joint["interictal_substrate_retained"]
    )
    payload = {
        "schema_id": "topic4_rev21_zm_confirmation_aggregate_v1",
        "status": ("CROSS_STATE_CONFIRMATION_ESTABLISHED_DEVELOPMENT_ONLY"
                   if established else
                   "CROSS_STATE_CONFIRMATION_NOT_ESTABLISHED"),
        "reference_support": support,
        "per_cell": cells,
        "candidate_summaries": summaries,
        "joint_confirmation_robustness": robustness,
        "joint_interictal_substrate_retained": bool(
            joint["interictal_substrate_retained"]
        ),
        "patient_heldout_opened": False,
        "patient_ictal_inputs_read": False,
        "selection_unit": "topology_by_dynamics_seed_cell",
        "pooled_role": "two-cluster presence diagnostic only",
    }
    output = output_root / "confirmation/aggregate.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "robustness": robustness,
    }, indent=2))


if __name__ == "__main__":
    main()
