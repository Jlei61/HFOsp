#!/usr/bin/env python3
"""Zero-simulation audit and rescore of the continuous patient-mode target."""
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

from scripts.aggregate_topic4_rev12_cascade_fit import (  # noqa: E402
    patient_direction_contract,
)
from scripts.aggregate_topic4_rev12_node_canary import _strip_natural_arrays  # noqa: E402
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
    soft_causal_direction_alignment,
    soft_causal_wave_monotonicity_alignment,
    soft_dual_mode_objective,
    soft_topology_network_reproducibility,
)


DEFAULT_CONFIG = ROOT / "config/topic4_rev12_nd_soft_mode_target_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


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


def _frozen_patient_reference(patient: dict, *, per_mode: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    selected = []
    for mode in (0, 1):
        available = np.flatnonzero(patient["train_labels"] == mode)
        if len(available) < int(per_mode):
            raise RuntimeError("patient mode is smaller than frozen reference sample")
        selected.extend(rng.choice(available, size=int(per_mode), replace=False).tolist())
    selected = np.asarray(selected, int)
    return {
        "ranks": np.asarray(patient["train_ranks"])[selected],
        "labels": np.asarray(patient["train_labels"])[selected],
        "indices": selected,
    }


def _soft_score(ranks: np.ndarray, probability_b: np.ndarray, patient: dict,
                projections: np.ndarray, calibration: dict,
                objective: dict) -> dict:
    return soft_dual_mode_objective(
        ranks, probability_b, patient["ranks"], patient["labels"],
        patient["contact_names"], projections=projections,
        calibration=calibration, tau=float(objective["tau"]),
        occupancy_weight=float(objective["occupancy_weight"]),
        ambiguity_weight=float(objective["ambiguity_weight"]),
        contrast_weight=float(objective["contrast_weight"]),
    )


def _source_bundle(npz_path: Path, worker: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as loaded:
        returned = np.asarray(loaded["event_returned"], bool)
        maps = np.asarray(loaded["source_onset_maps_ms"], float)[returned]
        evaluable = np.asarray(loaded["source_onset_evaluable"], bool)[returned]
    if len(maps) != len(worker["probability_B"]):
        raise RuntimeError("source maps and returned-event probabilities differ")
    return maps[evaluable], np.asarray(worker["probability_B"], float)[evaluable]


def _worker_diagnostics(worker: dict, npz_path: Path, patient_reference: dict,
                        projections: np.ndarray, calibration: dict,
                        objective: dict, direction: dict, *, seed: int) -> dict:
    probabilities = np.asarray(worker["probability_B"], float)
    score = _soft_score(
        worker["ranks"], probabilities, patient_reference,
        projections, calibration, objective,
    )
    maps, map_probabilities = _source_bundle(npz_path, worker)
    direction_score = soft_causal_direction_alignment(
        maps, map_probabilities,
        axis_unit=np.asarray(direction["axis_unit_xy"], float),
        expected_mode_signs=np.asarray(direction["expected_mode_signs"], float),
        bin_mm=1.0, tail_fraction=0.2,
    )
    monotonicity = soft_causal_wave_monotonicity_alignment(
        maps, map_probabilities,
        axis_unit=np.asarray(direction["axis_unit_xy"], float),
        expected_mode_signs=np.asarray(direction["expected_mode_signs"], float),
        bin_mm=1.0,
    )
    natural = _strip_natural_arrays(natural_kmeans(
        worker["ranks"], worker["labels"], random_state=int(seed),
    ))
    return {
        "seed": int(worker["seed"]),
        "n_events": int(len(probabilities)),
        "soft_objective": score,
        "probability_B": {
            "mean": float(np.mean(probabilities)),
            "q05_q25_q50_q75_q95": np.quantile(
                probabilities, [0.05, 0.25, 0.5, 0.75, 0.95],
            ),
            "boundary_fraction_0p4_0p6": float(np.mean(
                (probabilities >= 0.4) & (probabilities <= 0.6)
            )),
            "confident_fraction_le0p1_or_ge0p9": float(np.mean(
                (probabilities <= 0.1) | (probabilities >= 0.9)
            )),
        },
        "soft_causal_direction": direction_score,
        "soft_causal_monotonicity": monotonicity,
        "natural_kmeans_final_validation_diagnostic": natural,
        "npz": str(npz_path),
        "npz_sha256": _sha256(npz_path),
    }


def _mean(values) -> float:
    values = np.asarray(values, float)
    return float(np.mean(values)) if len(values) else float("nan")


def _candidate_summary(candidate: dict, workers: list[dict]) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "role": candidate.get("role"),
        "field_sha256": candidate["node_field"]["field_sha256"],
        "n_networks": int(len(workers)),
        "mean_events": _mean([row["n_events"] for row in workers]),
        "mean_soft_objective": _mean([
            row["soft_objective"]["objective"] for row in workers
        ]),
        "mean_soft_weakest_mode": _mean([
            row["soft_objective"]["weakest_mode_lse"] for row in workers
        ]),
        "mean_soft_mode_0": _mean([
            row["soft_objective"]["modes"]["0"]["mean"] for row in workers
        ]),
        "mean_soft_mode_1": _mean([
            row["soft_objective"]["modes"]["1"]["mean"] for row in workers
        ]),
        "mean_ambiguity": _mean([
            row["soft_objective"]["ambiguity"] for row in workers
        ]),
        "mean_contrast_alignment": _mean([
            row["soft_objective"]["contrast"]["alignment"] for row in workers
        ]),
        "mean_boundary_fraction_0p4_0p6": _mean([
            row["probability_B"]["boundary_fraction_0p4_0p6"] for row in workers
        ]),
        "mean_confident_fraction": _mean([
            row["probability_B"]["confident_fraction_le0p1_or_ge0p9"]
            for row in workers
        ]),
        "mean_soft_causal_direction": _mean([
            row["soft_causal_direction"]["score"] for row in workers
        ]),
        "mean_soft_causal_monotonicity": _mean([
            row["soft_causal_monotonicity"]["score"] for row in workers
        ]),
        "per_network": workers,
    }


def _synthetic_controls(patient_reference: dict, projections: np.ndarray,
                        calibration: dict, objective: dict,
                        decision: dict) -> dict:
    ranks = np.asarray(patient_reference["ranks"], float)
    labels = np.asarray(patient_reference["labels"], int)
    exact = _soft_score(ranks, labels.astype(float), patient_reference,
                        projections, calibration, objective)
    ambiguous = _soft_score(ranks, np.full(len(ranks), 0.5), patient_reference,
                            projections, calibration, objective)
    mode_zero = labels == 0
    one_mode = _soft_score(
        ranks[mode_zero], np.zeros(np.sum(mode_zero)), patient_reference,
        projections, calibration, objective,
    )
    permutation = np.roll(np.arange(ranks.shape[1]), 3)
    permuted = _soft_score(
        ranks[:, permutation], labels.astype(float), patient_reference,
        projections, calibration, objective,
    )
    duplicated = _soft_score(
        np.repeat(ranks, 3, axis=0), np.repeat(labels.astype(float), 3),
        patient_reference, projections, calibration, objective,
    )
    margin = float(decision["control_margin"])
    tolerance = float(decision["replication_tolerance"])
    checks = {
        "exact_beats_ambiguous_continuous_cloud": (
            exact["objective"] + margin < ambiguous["objective"]
        ),
        "exact_beats_one_mode": exact["objective"] + margin < one_mode["objective"],
        "exact_beats_contact_permutation": (
            exact["objective"] + margin < permuted["objective"]
        ),
        "detector_fragment_duplication_invariant": (
            abs(exact["objective"] - duplicated["objective"]) <= tolerance
        ),
    }
    return {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "scores": {
            "exact_two_mode": exact,
            "ambiguous_continuous_cloud": ambiguous,
            "one_mode": one_mode,
            "contact_permuted": permuted,
            "exact_duplicated_threefold": duplicated,
        },
        "contact_permutation": permutation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    for library in config["libraries"]:
        for key in ("config", "manifest"):
            path = _resolve(artifact_root, library[key]["path"])
            if _sha256(path) != library[key]["sha256"]:
                raise RuntimeError(f"library input hash changed: {library[key]['path']}")

    cohort = json.loads(_resolve(
        artifact_root, config["inputs"]["cohort_config"]["path"],
    ).read_text())
    classifier_config = json.loads(_resolve(
        artifact_root, config["inputs"]["classifier_config"]["path"],
    ).read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    objective = config["soft_objective"]
    patient_reference = _frozen_patient_reference(
        patient, per_mode=int(objective["patient_reference_events_per_mode"]),
        seed=int(objective["projection_seed"]),
    )
    patient_reference["contact_names"] = patient["contact_names"]
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]),
        n_directions=int(objective["projection_count"]),
        seed=int(objective["projection_seed"]),
    )
    calibration = calibrate_component_scales(
        patient["train_ranks"], patient["train_labels"], patient["train_blocks"],
        patient["contact_names"], projections,
        sample_size=int(objective["calibration_sample_size"]),
        draws=int(objective["calibration_draws"]),
        seed=int(objective["calibration_seed"]),
    )

    first_library = config["libraries"][0]
    first_manifest = json.loads(_resolve(
        artifact_root, first_library["manifest"]["path"],
    ).read_text())
    first_config = json.loads(_resolve(
        artifact_root, first_library["config"]["path"],
    ).read_text())
    first_candidate = first_manifest["candidates"][0]["candidate_id"]
    first_npz = next((
        artifact_root / first_config["output_root"] / "workers"
    ).glob(f"{first_candidate}_seed_*.npz"))
    with np.load(first_npz, allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        xy = np.asarray(loaded["contact_xy_mm"], float)
    order = [int(np.flatnonzero(names == name)[0]) for name in patient["contact_names"]]
    direction = patient_direction_contract(
        patient["train_ranks"], patient["train_labels"], xy[order],
    )

    rows = []
    for library in config["libraries"]:
        library_config = json.loads(_resolve(
            artifact_root, library["config"]["path"],
        ).read_text())
        manifest = json.loads(_resolve(
            artifact_root, library["manifest"]["path"],
        ).read_text())
        worker_root = artifact_root / library_config["output_root"] / "workers"
        for candidate in manifest["candidates"]:
            worker_rows = []
            topology_maps, topology_probabilities = [], []
            for npz_path in sorted(worker_root.glob(
                    f"{candidate['candidate_id']}_seed_*.npz")):
                worker = _load_network_worker(
                    npz_path, patient["contact_names"], classifier,
                    semantics["raw_to_patient"],
                )
                if not len(worker["ranks"]):
                    continue
                maps, map_probabilities = _source_bundle(npz_path, worker)
                topology_maps.append(maps)
                topology_probabilities.append(map_probabilities)
                worker_rows.append(_worker_diagnostics(
                    worker, npz_path, patient_reference, projections,
                    calibration, objective, direction,
                    seed=int(objective["projection_seed"]) + int(worker["seed"]),
                ))
            if not worker_rows:
                continue
            row = _candidate_summary(candidate, worker_rows)
            row["library_id"] = library["library_id"]
            topology = soft_topology_network_reproducibility(
                topology_maps, topology_probabilities,
            )
            row["soft_source_topology"] = topology
            row["soft_topology_across_network"] = topology[
                "mean_across_network_template_cosine"
            ]
            row["soft_topology_mode_separation"] = topology[
                "equal_network_between_mode_distance"
            ]
            rows.append(row)
    rows.sort(key=lambda row: (row["mean_soft_objective"], row["library_id"],
                               row["candidate_id"]))
    controls = _synthetic_controls(
        patient_reference, projections, calibration, objective, config["decision"],
    )
    stage_x = {
        row["candidate_id"]: row for row in rows if row["library_id"] == "stage_x"
    }
    paired_stage_x = None
    if {"stage_x_anchor", "stage_x_f14p03"}.issubset(stage_x):
        anchor = stage_x["stage_x_anchor"]
        candidate = stage_x["stage_x_f14p03"]
        anchor_by_seed = {row["seed"]: row for row in anchor["per_network"]}
        candidate_by_seed = {row["seed"]: row for row in candidate["per_network"]}
        common = sorted(set(anchor_by_seed) & set(candidate_by_seed))
        deltas = [
            anchor_by_seed[seed]["soft_objective"]["objective"]
            - candidate_by_seed[seed]["soft_objective"]["objective"]
            for seed in common
        ]
        paired_stage_x = {
            "utility_definition": "anchor objective minus candidate objective",
            "seeds": common,
            "per_seed_utility": deltas,
            "mean_utility": _mean(deltas),
            "positive_networks": int(np.sum(np.asarray(deltas) > 0.0)),
            "negative_networks": int(np.sum(np.asarray(deltas) < 0.0)),
        }

    payload = {
        "schema_id": config["schema_id"],
        "status": (
            "SOFT_MODE_TARGET_CONTROLS_PASS_ZERO_SIM_RESCORE_COMPLETE"
            if controls["all_pass"]
            else "SOFT_MODE_TARGET_CONTROL_FAIL"
        ),
        "scientific_role": config["scientific_role"],
        "synthetic_controls": controls,
        "rows": rows,
        "stage_x_paired_soft_rescore": paired_stage_x,
        "patient_training_reference": {
            "events_per_mode": int(objective["patient_reference_events_per_mode"]),
            "event_indices": patient_reference["indices"],
            "old_classifier_to_patient_mode": semantics,
            "direction_contract": direction,
        },
        "scoring_contract": {
            "continuous_membership": "patient-mapped frozen classifier probability_B",
            "natural_kmeans_role": "final validation diagnostic only; absent from fit score",
            "patient_data": "training events only for every score and direction contract",
            "model_event_unit": "frozen causal-family returned event",
            "network_weighting": "equal network mean",
            "component_calibration": calibration,
            "objective": objective,
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "libraries": config["libraries"],
        },
        "claim_boundary": config["claim_boundary"],
    }
    output_root = artifact_root / config["output_root"]
    _atomic_json(output_root / "soft_mode_target_audit.json", payload)
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "library_id", "candidate_id", "role", "n_networks", "mean_events",
        "mean_soft_objective", "mean_soft_weakest_mode", "mean_soft_mode_0",
        "mean_soft_mode_1", "mean_ambiguity", "mean_contrast_alignment",
        "mean_boundary_fraction_0p4_0p6", "mean_confident_fraction",
        "mean_soft_causal_direction", "mean_soft_causal_monotonicity",
        "soft_topology_across_network", "soft_topology_mode_separation",
    ]
    with (output_root / "soft_mode_candidate_rescore.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(rows),
        "best": rows[0]["candidate_id"],
        "stage_x": paired_stage_x,
        "output": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
