#!/usr/bin/env python3
"""Aggregate the frozen rev21 Z/M timescale grid without patient ictal data."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_topic4_rev21_zm_screen import (  # noqa: E402
    _atomic_json, _load_npz, _resolve, _sha256, matched_interictal_retention,
    reference_support, replace_retention_with_matched, summarize_candidate,
)
from src.topic4_rev20_dual_core_endpoint import (  # noqa: E402
    score_complete_distribution, score_validation_endpoints,
)


def timescale_log_distance(level: dict, reference_level: dict) -> float:
    return float(math.hypot(
        math.log(float(level["tau_z_ms"]) / float(reference_level["tau_z_ms"])),
        math.log(float(level["tau_adp_ms"])
                 / float(reference_level["tau_adp_ms"])),
    ))


def annotate_timescale_neighbors(rows: list[dict]) -> None:
    tau_z = sorted({float(row["level"]["tau_z_ms"]) for row in rows})
    tau_adp = sorted({float(row["level"]["tau_adp_ms"]) for row in rows})
    for row in rows:
        zi = tau_z.index(float(row["level"]["tau_z_ms"]))
        ai = tau_adp.index(float(row["level"]["tau_adp_ms"]))
        neighbors = [
            other for other in rows if other is not row
            and abs(tau_z.index(float(other["level"]["tau_z_ms"])) - zi)
            + abs(tau_adp.index(float(other["level"]["tau_adp_ms"])) - ai)
            == 1
        ]
        row["neighbor_eligible_fraction"] = (
            float(np.mean([
                other["model_ictal_eligible_fraction"] for other in neighbors
            ])) if neighbors else 0.0
        )


def rank_timescale_candidates(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (
        -row["model_ictal_eligible_fraction"],
        -int(row["interictal_substrate_retained"]),
        (float("inf") if row["worst_standardized_deterioration"] is None
         else row["worst_standardized_deterioration"]),
        -row.get("neighbor_eligible_fraction", 0.0),
        row["log_distance_from_reference"],
        row["candidate_id"],
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest", type=Path,
        help="Defaults to <output_root>/timescale/candidate_manifest.json",
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
        output_root / "timescale/status/controller.json"
    ).read_text())
    if controller.get("status") != "COMPLETE":
        raise RuntimeError("timescale controller is not complete")
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
        else output_root / "timescale/candidate_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(args.config.resolve()):
        raise RuntimeError("timescale candidate manifest is stale")
    levels = {row["candidate_id"]: row["level"]
              for row in manifest["candidates"]}
    references = [row for row in manifest["candidates"] if row["is_reference"]]
    if len(references) != 1:
        raise RuntimeError("timescale manifest must contain one reference cell")
    reference_level = references[0]["level"]

    coarse_off = {}
    coarse_off_arrays = {}
    for path in sorted((output_root / "coarse/workers").glob(
            "rev21_zm_off_topology_*_dynamics_*.json")):
        worker = json.loads(path.read_text())
        arrays = _load_npz(Path(worker["arrays"]["path"]))
        key = (int(worker["topology_seed"]), int(worker["dynamics_seed"]))
        coarse_off_arrays[key] = arrays
        coarse_off[key] = {
            "selection": score_complete_distribution(
                arrays["onsets"], arrays["event_returned"],
                contract=contract, training_arrays=training,
            ),
            "validation": score_validation_endpoints(
                arrays["onsets"], arrays["ranks"], arrays["event_returned"],
                contract=contract, training_arrays=training,
                classifier=classifier,
                kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
            ),
        }
    if len(coarse_off) != 4:
        raise RuntimeError("timescale aggregation requires four matched coarse off cells")

    cells = []
    arrays_by_candidate: dict[str, list[dict]] = {}
    arrays_by_candidate_cell: dict[str, dict[tuple[int, int], dict]] = {}
    for path in sorted((output_root / "timescale/workers").glob("*.json")):
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
            "model_ictal": worker["model_ictal_rev21"],
            "selection": selection,
            "validation": validation,
            "worker_json": str(path),
        })
        arrays_by_candidate.setdefault(worker["candidate_id"], []).append(arrays)
        arrays_by_candidate_cell.setdefault(worker["candidate_id"], {})[
            (int(worker["topology_seed"]), int(worker["dynamics_seed"]))
        ] = arrays
    grouped: dict[str, list[dict]] = {}
    for row in cells:
        grouped.setdefault(row["candidate_id"], []).append(row)
    if set(grouped) != set(levels):
        raise RuntimeError("timescale worker candidate set is incomplete")

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
        summary = summarize_candidate(
            candidate_id, rows, coarse_off, support, pooled,
            levels[candidate_id],
        )
        matched = matched_interictal_retention(
            arrays_by_candidate_cell[candidate_id], coarse_off_arrays,
            contract=contract, training_arrays=training,
            classifier=classifier,
            kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
        )
        replace_retention_with_matched(summary, matched)
        summary["log_distance_from_reference"] = timescale_log_distance(
            levels[candidate_id], reference_level,
        )
        summaries.append(summary)
    annotate_timescale_neighbors(summaries)
    ranked = rank_timescale_candidates(summaries)
    eligible_retained = [
        row for row in ranked
        if row["model_ictal_eligible_cells"] > 0
        and row["interictal_substrate_retained"]
    ]
    payload = {
        "schema_id": "topic4_rev21_zm_timescale_aggregate_v2",
        "status": ("REV21_TIMESCALE_HAS_CONFIRMATION_FINALIST"
                   if eligible_retained else
                   "NO_CROSS_STATE_WORKPOINT_IN_FROZEN_TIMESCALE_GRID"),
        "reference_support": support,
        "per_cell": cells,
        "candidate_summaries": summaries,
        "ranked_candidate_ids": [row["candidate_id"] for row in ranked],
        "confirmation_finalist_candidate_id": (
            eligible_retained[0]["candidate_id"] if eligible_retained else None
        ),
        "patient_heldout_opened": False,
        "patient_ictal_inputs_read": False,
        "selection_unit": "topology_by_dynamics_seed_cell",
        "pooled_role": (
            "event-count-matched development retention screen; not same-network "
            "two-mode confirmation"
        ),
        "retention_calibration": (
            "patient matched-N absolute distribution floor plus per-cell "
            "event-count-matched paired Z/M-off retention null"
        ),
    }
    output = output_root / "timescale/aggregate.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "candidate": payload["confirmation_finalist_candidate_id"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
