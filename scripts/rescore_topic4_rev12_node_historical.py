#!/usr/bin/env python3
"""Zero-simulation rescore of compatible historical continuous Node fields."""
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

from scripts.build_topic4_data_driven_snn_cohort_targets import _target_config  # noqa: E402
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.topic4_data_driven_cohort import (  # noqa: E402
    build_crossfit_patient_target,
    canonical_pair_contract,
    subject_raw_root,
    subset_pair_contract,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    contrast_r2,
    dual_mode_objective,
    event_features,
    fixed_projection_matrix,
    normalize_event_ranks,
    shaft_balanced_feature_weights,
    weighted_r2,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_COHORT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_v1.json"
DEFAULT_REFERENCE_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "historical_rescore"
)
DEFAULT_LEGACY_VARIANCE_AUDIT = DEFAULT_ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "interictal_variance_capture/variance_capture.json"
)
LIBRARY_CONFIGS = (
    "config/topic4_rev10_sa_observation_invariant_field_v3.json",
    "config/topic4_rev10_sa_observation_invariant_field_v4_1.json",
    "config/topic4_rev10_sa_observation_invariant_field_v5_1.json",
    "config/topic4_rev10_sa_observation_invariant_field_v5_2.json",
    "config/topic4_rev10_sa_observation_invariant_field_v6_1.json",
    "config/topic4_rev10_sa_observation_invariant_field_v6_2.json",
    "config/topic4_rev10_d6_1_natural_kmeans_closeout.json",
    "config/topic4_rev10_d6_2_joint_continuous_field_surface.json",
    "config/topic4_rev10_d6_3_joint_field_replication.json",
    "config/topic4_rev11_nlc_frozen_substrate_confirmation.json",
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
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
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


def _patient_data(cohort_config: dict, artifact_root: Path) -> dict:
    subject_id = "epilepsiae_1146"
    inputs = cohort_config["inputs"]
    rank_path = artifact_root / inputs["rank_displacement_root"] / f"{subject_id}.json"
    geometry_path = artifact_root / inputs["gradient_geometry_root"] / f"{subject_id}.json"
    pair = canonical_pair_contract(json.loads(rank_path.read_text()))
    geometry = json.loads(geometry_path.read_text())
    field = geometry.get("interictal_field") or {}
    if field.get("contact_order"):
        pair = subset_pair_contract(pair, [str(item) for item in field["contact_order"]])
    raw_root = subject_raw_root(
        subject_id,
        epilepsiae_root=inputs["epilepsiae_raw_root"],
        yuquan_root=inputs["yuquan_raw_root"],
    )
    raw = load_subject_propagation_events(raw_root)
    target = build_crossfit_patient_target(raw, pair, config=_target_config(cohort_config))
    lookup = {str(name): index for index, name in enumerate(raw["channel_names"])}
    rows = np.asarray([lookup[str(name)] for name in target["contact_order"]], int)
    normalized = mask_phantom_ranks(
        np.asarray(raw["ranks"], float)[rows],
        np.asarray(raw["bools"], bool)[rows], normalize=True,
    ).T
    train_index = np.asarray(target["train_event_indices"], int)
    heldout_index = np.asarray(target["heldout_event_indices"], int)
    train_labels = np.asarray(target["train_labels"], int)
    heldout_labels = np.asarray(target["heldout_labels"], int)
    train_ranks = normalized[train_index]
    heldout_ranks = normalized[heldout_index]
    train_features = event_features(train_ranks)
    heldout_features = event_features(heldout_ranks)
    train_prototypes = np.asarray([
        train_features[train_labels == mode].mean(axis=0) for mode in (0, 1)
    ])
    heldout_prototypes = np.asarray([
        heldout_features[heldout_labels == mode].mean(axis=0) for mode in (0, 1)
    ])
    return {
        "contact_names": np.asarray(target["contact_order"]).astype(str),
        "train_ranks": train_ranks,
        "heldout_ranks": heldout_ranks,
        "train_labels": train_labels,
        "heldout_labels": heldout_labels,
        "train_features": train_features,
        "heldout_features": heldout_features,
        "train_prototypes": train_prototypes,
        "heldout_prototypes": heldout_prototypes,
        "global_mean": train_features.mean(axis=0),
        "heldout_blocks": np.asarray(raw["block_ids"])[heldout_index],
        "train_blocks": np.asarray(raw["block_ids"])[train_index],
        "train_event_indices": train_index,
    }


def _classifier_contract(reference_config: dict, artifact_root: Path) -> dict:
    output_root = artifact_root / reference_config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    classifier = dict(manifest["direction_classifier"])
    for key in ("coef", "class_centers", "class_precisions", "ood_distance_thresholds"):
        classifier[key] = np.asarray(classifier[key], float)
    contract_path = artifact_root / reference_config["inputs"]["contact_contract"]["path"]
    target_path = artifact_root / reference_config["inputs"]["shaft_aware_target_npz"]["path"]
    floors_path = artifact_root / reference_config["inputs"]["shaft_aware_floors"]["path"]
    contract = json.loads(contract_path.read_text())
    names, embedding, _, _ = load_scoring_contract(
        target_path, floors_path, "FULL_TIMING", fixed_events_per_mode=6,
    )
    with np.load(target_path, allow_pickle=False) as target:
        old_event_indices = np.asarray(target["patient_train_event_indices"], int)
        old_labels = np.asarray(target["patient_train_old_labels"], int)
    return {
        "names": np.asarray(names).astype(str),
        "embedding": embedding,
        "groups": contract_groups(contract),
        "classifier": classifier,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "old_event_indices": old_event_indices,
        "old_labels": old_labels,
    }


def _reorder_patient_contract(patient: dict, target_names: np.ndarray) -> dict:
    """Rebuild every patient feature after an exact contact-name reorder."""
    source_names = np.asarray(patient["contact_names"]).astype(str)
    target_names = np.asarray(target_names).astype(str)
    if set(source_names) != set(target_names):
        raise RuntimeError("rev12 patient and classifier contact sets differ")
    order = np.asarray([
        int(np.flatnonzero(source_names == name)[0]) for name in target_names
    ])
    output = dict(patient)
    output["contact_names"] = target_names
    output["train_ranks"] = np.asarray(patient["train_ranks"])[:, order]
    output["heldout_ranks"] = np.asarray(patient["heldout_ranks"])[:, order]
    output["train_features"] = event_features(output["train_ranks"])
    output["heldout_features"] = event_features(output["heldout_ranks"])
    output["train_prototypes"] = np.asarray([
        output["train_features"][output["train_labels"] == mode].mean(axis=0)
        for mode in (0, 1)
    ])
    output["heldout_prototypes"] = np.asarray([
        output["heldout_features"][output["heldout_labels"] == mode].mean(axis=0)
        for mode in (0, 1)
    ])
    output["global_mean"] = output["train_features"].mean(axis=0)
    return output


def _old_to_patient_label_map(patient: dict, classifier_contract: dict) -> dict:
    """Freeze old classifier labels onto current TA/TB semantics using patient data only."""
    current = {
        int(index): int(label) for index, label in zip(
            patient["train_event_indices"], patient["train_labels"]
        )
    }
    contingency = np.zeros((2, 2), int)
    for index, old_label in zip(
            classifier_contract["old_event_indices"], classifier_contract["old_labels"]):
        if int(index) in current:
            contingency[int(old_label), current[int(index)]] += 1
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    order = np.asarray([0, 1] if identity >= swapped else [1, 0], int)
    return {
        "raw_to_patient": order,
        "contingency_old_by_patient": contingency,
        "n_shared_events": int(contingency.sum()),
        "identity_matches": identity,
        "swapped_matches": swapped,
        "selection_data": "overlapping patient training events only",
    }


def _candidate_is_node_only(candidate: dict, config_name: str) -> bool:
    if config_name == "topic4_rev11_nlc_frozen_substrate_confirmation.json":
        return candidate.get("candidate_id") == "node_baseline"
    arm = candidate.get("arm")
    return arm in (None, "Node", "Node_only")


def _worker_paths(output_root: Path, candidate_id: str) -> list[Path]:
    return sorted((output_root / "workers").glob(f"{candidate_id}_seed_*.npz"))


def _reorder(values: np.ndarray, source_names: np.ndarray,
             target_names: np.ndarray) -> np.ndarray:
    source_names = np.asarray(source_names).astype(str)
    target_names = np.asarray(target_names).astype(str)
    if set(source_names) != set(target_names):
        raise RuntimeError("historical worker contact set differs from rev12 target")
    order = np.asarray([int(np.flatnonzero(source_names == name)[0]) for name in target_names])
    return np.asarray(values)[:, order]


def _formal_clean_mask(onsets: np.ndarray, ood: np.ndarray,
                       groups: dict) -> np.ndarray:
    """Historical Fig.4 support mask; returned filtering happens upstream."""
    onsets = np.asarray(onsets, float)
    ood = np.asarray(ood, bool)
    if onsets.ndim != 2 or ood.shape != (len(onsets),):
        raise ValueError("onsets and OOD mask must align")
    return (
        np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
        & np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
        & ~ood
    )


def _load_network_worker(npz_path: Path, target_names: np.ndarray,
                         classifier_contract: dict,
                         label_map: np.ndarray) -> dict:
    json_path = npz_path.with_suffix(".json")
    payload = json.loads(json_path.read_text())
    if payload.get("arrays", {}).get("sha256") not in (None, _sha256(npz_path)):
        raise RuntimeError(f"worker array hash changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        onsets = _reorder(np.asarray(loaded["onsets"], float), names, target_names)
        ranks = _reorder(np.asarray(loaded["ranks"], float), names, target_names)
        returned = np.asarray(loaded["event_returned"], bool)
    assigned = assign_direction_modes(
        onsets,
        groups=classifier_contract["groups"],
        embedding=classifier_contract["embedding"],
        classifier=classifier_contract["classifier"],
    )
    selected = returned
    seed = int(payload.get("seed", int(npz_path.stem.rsplit("_", 1)[-1])))
    raw_labels = np.asarray(assigned["labels"], int)
    label_map = np.asarray(label_map, int)
    if label_map.shape != (2,) or set(label_map.tolist()) != {0, 1}:
        raise ValueError("old-to-patient label map must be a permutation")
    returned_onsets = onsets[selected]
    raw_probability_b = np.asarray(assigned["probability_B"], float)
    patient_probability_b = (
        raw_probability_b if int(label_map[1]) == 1 else 1.0 - raw_probability_b
    )
    formal_clean = _formal_clean_mask(
        returned_onsets, np.asarray(assigned["ood"], bool)[selected],
        classifier_contract["groups"],
    )
    return {
        "seed": seed,
        "ranks": ranks[selected],
        "onsets": returned_onsets,
        "labels": label_map[raw_labels][selected],
        "probability_B": patient_probability_b[selected],
        "ood": np.asarray(assigned["ood"], bool)[selected],
        "formal_clean": formal_clean,
        "n_detected": int(len(returned)),
        "n_returned": int(np.sum(returned)),
        "duration_ms": float(payload.get("simulation", {}).get("duration_ms", np.nan)),
        "npz": str(npz_path),
        "npz_sha256": _sha256(npz_path),
    }


def _network_prototype(worker: dict, selected: np.ndarray) -> np.ndarray | None:
    selected = np.asarray(selected, bool)
    if selected.shape != (len(worker["labels"]),):
        raise ValueError("event subset must align with worker events")
    labels = worker["labels"][selected]
    if any(not np.any(labels == mode) for mode in (0, 1)):
        return None
    features = event_features(normalize_event_ranks(worker["ranks"][selected]))
    return np.asarray([features[labels == mode].mean(axis=0) for mode in (0, 1)])


def _finite_mean(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float("nan") if not len(values) else float(np.mean(values))


def _substrate_stratum(config: dict) -> str:
    spatial = config.get("fixed_spatial_ou") or {}
    if spatial.get("mode") == "local" and np.isclose(
            float(spatial.get("sigma_rate_per_ms", np.nan)), 0.1):
        return "current_spatial_ou_node_only"
    return "legacy_without_current_spatial_ou"


def _deduplicate_workers_by_seed(workers: list[dict]) -> list[dict]:
    """A network seed is one independent unit; retain its longest trajectory."""
    selected = {}
    for worker in workers:
        seed = int(worker["seed"])
        current = selected.get(seed)
        if current is None or worker["duration_ms"] > current["duration_ms"]:
            selected[seed] = worker
        elif (worker["duration_ms"] == current["duration_ms"]
              and worker["npz_sha256"] != current["npz_sha256"]):
            raise RuntimeError(
                f"seed {seed} has two non-identical workers at the same duration"
            )
    return [selected[seed] for seed in sorted(selected)]


def _score_estimand(workers: list[dict], patient: dict, projections: np.ndarray,
                    calibration: dict, *, subset_key: str | None) -> dict:
    network_scores, prototypes = [], []
    for worker in workers:
        selected = (
            np.ones(len(worker["labels"]), bool) if subset_key is None
            else np.asarray(worker[subset_key], bool)
        )
        score = dual_mode_objective(
            worker["ranks"][selected], worker["labels"][selected],
            patient["train_ranks"], patient["train_labels"],
            patient["contact_names"], missing_mode_penalty=1.0,
            projections=projections, calibration=calibration,
        )
        score.update({
            "seed": worker["seed"],
            "n_events": int(np.sum(selected)),
            "mode_counts": np.bincount(worker["labels"][selected], minlength=2),
        })
        network_scores.append(score)
        prototype = _network_prototype(worker, selected)
        if prototype is not None:
            prototypes.append(prototype)
    mean_prototype = None if not prototypes else np.mean(prototypes, axis=0)
    weights = shaft_balanced_feature_weights(patient["contact_names"])
    patient_reference_r2 = weighted_r2(
        patient["heldout_features"], patient["heldout_labels"],
        patient["train_prototypes"], patient["global_mean"], weights,
    )
    if mean_prototype is None:
        heldout_r2 = float("nan")
        contrast = {
            "heldout_raw_r2": float("nan"),
            "heldout_train_scaled_r2": float("nan"),
            "train_fitted_nonnegative_scale": float("nan"),
            "heldout_weighted_cosine": float("nan"),
        }
    else:
        heldout_r2 = weighted_r2(
            patient["heldout_features"], patient["heldout_labels"],
            mean_prototype, patient["global_mean"], weights,
        )
        contrast = contrast_r2(
            mean_prototype, patient["train_prototypes"],
            patient["heldout_prototypes"], weights,
        )
    same_network_both = sum(
        int(np.all(np.asarray(score["mode_counts"]) > 0)) for score in network_scores
    )
    return {
        "n_networks": int(len(workers)),
        "same_network_both_modes": int(same_network_both),
        "same_network_both_fraction": float(same_network_both / max(1, len(workers))),
        "mean_events": _finite_mean([score["n_events"] for score in network_scores]),
        "mean_patient_objective": _finite_mean([score["objective"] for score in network_scores]),
        "mean_weakest_mode_lse": _finite_mean([score["weakest_mode_lse"] for score in network_scores]),
        "mean_mode_0_loss": _finite_mean([score["modes"]["0"]["mean"] for score in network_scores]),
        "mean_mode_1_loss": _finite_mean([score["modes"]["1"]["mean"] for score in network_scores]),
        "patient_train_k2_r2_on_heldout": patient_reference_r2,
        "model_prototype_r2_on_heldout": heldout_r2,
        "contrast": contrast,
        "network_scores": network_scores,
    }


def score_candidate(candidate: dict, workers: list[dict], patient: dict,
                    projections: np.ndarray, calibration: dict) -> dict:
    complete = _score_estimand(
        workers, patient, projections, calibration, subset_key=None,
    )
    formal_clean = _score_estimand(
        workers, patient, projections, calibration, subset_key="formal_clean",
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "field_type": candidate.get("field_type") or candidate.get("node_field", {}).get("field_type"),
        "field_sha256": candidate.get("field_sha256") or candidate.get("node_field", {}).get("field_sha256"),
        "roughness": candidate.get("roughness") or candidate.get("node_field", {}).get("roughness"),
        "n_networks": complete["n_networks"],
        "same_network_both_modes": complete["same_network_both_modes"],
        "same_network_both_fraction": complete["same_network_both_fraction"],
        "mean_returned_events": _finite_mean([worker["n_returned"] for worker in workers]),
        "mean_formal_clean_events": formal_clean["mean_events"],
        "mean_ood_fraction": _finite_mean([
            float(np.mean(worker["ood"])) if len(worker["ood"]) else 1.0
            for worker in workers
        ]),
        # Flat fields remain aliases for the rev12 primary complete-returned estimand.
        "mean_patient_objective": complete["mean_patient_objective"],
        "mean_weakest_mode_lse": complete["mean_weakest_mode_lse"],
        "mean_mode_0_loss": complete["mean_mode_0_loss"],
        "mean_mode_1_loss": complete["mean_mode_1_loss"],
        "patient_train_k2_r2_on_heldout": complete["patient_train_k2_r2_on_heldout"],
        "model_prototype_r2_on_heldout": complete["model_prototype_r2_on_heldout"],
        "contrast": complete["contrast"],
        "estimands": {
            "complete_returned": complete,
            "formal_clean": formal_clean,
        },
        "source_topology": "NOT_RECORDED_IN_HISTORICAL_ARTIFACT",
        "network_scores": complete["network_scores"],
        "worker_inputs": [{
            "seed": worker["seed"], "npz": worker["npz"],
            "npz_sha256": worker["npz_sha256"],
        } for worker in workers],
    }


def _legacy_variance_parity(rows: list[dict], path: Path) -> dict:
    """Verify the formal-clean estimator against the accepted rev11 audit."""
    if not path.exists():
        return {"status": "REFERENCE_MISSING", "path": str(path)}
    reference = json.loads(path.read_text())
    expected = reference["arms"]["node_baseline"]
    candidates = [
        row for row in rows
        if row["library"] == "topic4_rev11_nlc_frozen_substrate_confirmation"
        and row["candidate_id"] == "node_baseline"
    ]
    if len(candidates) != 1:
        raise RuntimeError("rev11 Node baseline is not unique in historical rescore")
    observed = candidates[0]["estimands"]["formal_clean"]
    comparisons = {
        "model_prototype_r2_on_heldout": {
            "observed": observed["model_prototype_r2_on_heldout"],
            "expected": expected["components"]["all"]["model_r2_on_patient_heldout"],
        },
        "heldout_train_scaled_contrast_r2": {
            "observed": observed["contrast"]["heldout_train_scaled_r2"],
            "expected": expected["between_mode_contrast"][
                "heldout_scale_calibrated_contrast_r2"
            ],
        },
    }
    for comparison in comparisons.values():
        comparison["absolute_error"] = abs(
            float(comparison["observed"]) - float(comparison["expected"])
        )
    maximum = max(item["absolute_error"] for item in comparisons.values())
    if maximum > 1e-12:
        raise RuntimeError(f"formal-clean parity drifted by {maximum:.3g}")
    return {
        "status": "EXACT_WITHIN_1E_12",
        "path": str(path),
        "sha256": _sha256(path),
        "comparisons": comparisons,
        "maximum_absolute_error": maximum,
    }


def pareto_front(rows: list[dict]) -> list[str]:
    valid = [row for row in rows if all(np.isfinite(row[key]) for key in (
        "mean_patient_objective", "model_prototype_r2_on_heldout",
        "same_network_both_fraction",
    ))]
    selected = []
    for row in valid:
        dominated = False
        for other in valid:
            if other is row:
                continue
            no_worse = (
                other["mean_patient_objective"] <= row["mean_patient_objective"]
                and other["model_prototype_r2_on_heldout"] >= row["model_prototype_r2_on_heldout"]
                and other["same_network_both_fraction"] >= row["same_network_both_fraction"]
            )
            strict = (
                other["mean_patient_objective"] < row["mean_patient_objective"]
                or other["model_prototype_r2_on_heldout"] > row["model_prototype_r2_on_heldout"]
                or other["same_network_both_fraction"] > row["same_network_both_fraction"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            selected.append(row["library_candidate_id"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    cohort_path = args.cohort_config.resolve()
    reference_path = args.reference_config.resolve()
    cohort = json.loads(cohort_path.read_text())
    reference = json.loads(reference_path.read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(reference, artifact_root)
    label_semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    if not np.array_equal(classifier["names"], patient["contact_names"]):
        raise RuntimeError("rev12 patient contact reorder failed")
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]), n_directions=64, seed=20260821,
    )
    calibration = calibrate_component_scales(
        patient["train_ranks"], patient["train_labels"], patient["train_blocks"],
        patient["contact_names"], projections, sample_size=6, draws=256,
        seed=20260821,
    )

    rows, library_inputs, field_groups = [], [], {}
    for relative_config in LIBRARY_CONFIGS:
        config_path = ROOT / relative_config
        config = json.loads(config_path.read_text())
        stratum = _substrate_stratum(config)
        output_root = artifact_root / config["output_root"]
        manifest_path = output_root / "candidate_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        library = Path(relative_config).stem
        for candidate in manifest.get("candidate_set", {}).get("candidates", []):
            if not _candidate_is_node_only(candidate, config_path.name):
                continue
            paths = _worker_paths(output_root, candidate["candidate_id"])
            if not paths:
                continue
            workers = [
                _load_network_worker(
                    path, patient["contact_names"], classifier,
                    label_semantics["raw_to_patient"],
                )
                for path in paths
            ]
            row = score_candidate(candidate, workers, patient, projections, calibration)
            row["library"] = library
            row["library_candidate_id"] = f"{library}::{candidate['candidate_id']}"
            row["config_path"] = str(config_path)
            row["config_sha256"] = _sha256(config_path)
            row["substrate_stratum"] = stratum
            rows.append(row)
            field_hash = row.get("field_sha256")
            if field_hash:
                group = field_groups.setdefault((stratum, field_hash), {
                    "candidate": candidate, "workers": [], "source_records": [],
                })
                group["workers"].extend(workers)
                group["source_records"].append(row["library_candidate_id"])
        library_inputs.append({
            "config": str(config_path), "config_sha256": _sha256(config_path),
            "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        })
    rows.sort(key=lambda row: (
        row["mean_patient_objective"], -row["model_prototype_r2_on_heldout"],
        row["library_candidate_id"],
    ))
    record_front = pareto_front(rows)
    for row in rows:
        row["historical_record_pareto"] = row["library_candidate_id"] in record_front
    legacy_parity = _legacy_variance_parity(rows, DEFAULT_LEGACY_VARIANCE_AUDIT)

    field_rows = []
    for (stratum, field_hash), group in field_groups.items():
        workers = _deduplicate_workers_by_seed(group["workers"])
        row = score_candidate(
            group["candidate"], workers, patient, projections, calibration,
        )
        row.update({
            "field_id": f"{stratum}::{field_hash[:12]}",
            "library_candidate_id": f"{stratum}::{field_hash[:12]}",
            "substrate_stratum": stratum,
            "field_sha256": field_hash,
            "source_records": sorted(group["source_records"]),
            "n_source_records": int(len(group["source_records"])),
        })
        field_rows.append(row)
    field_rows.sort(key=lambda row: (
        row["mean_patient_objective"], -row["model_prototype_r2_on_heldout"],
        row["field_id"],
    ))
    current_fields = [
        row for row in field_rows
        if row["substrate_stratum"] == "current_spatial_ou_node_only"
    ]
    field_front = pareto_front(current_fields)
    for row in field_rows:
        row["historical_field_pareto"] = row["field_id"] in field_front
    baseline_records = [
        row for row in rows
        if row["library"] == "topic4_rev11_nlc_frozen_substrate_confirmation"
        and row["candidate_id"] == "node_baseline"
    ]
    if len(baseline_records) != 1:
        raise RuntimeError("current frozen Node baseline is not unique")
    anchor_hash = baseline_records[0]["field_sha256"]
    distinct = [row for row in current_fields if row["field_sha256"] != anchor_hash]
    best_distinct = None if not distinct else distinct[0]["field_id"]

    payload = {
        "schema_id": "topic4_rev12_node_historical_rescore_v1",
        "status": "REV12ND_HISTORICAL_ZERO_SIMULATION_RESCORE_COMPLETE",
        "scientific_role": "development_only_node_initialization_search",
        "n_candidate_records": int(len(rows)),
        "record_level_pareto_ids_not_for_selection": record_front,
        "field_level_pareto_ids": field_front,
        "anchor_field_sha256": anchor_hash,
        "best_distinct_field_id": best_distinct,
        "patient": {
            "subject_id": "epilepsiae_1146",
            "n_train_events": int(len(patient["train_ranks"])),
            "n_heldout_events": int(len(patient["heldout_ranks"])),
            "n_heldout_blocks": int(len(np.unique(patient["heldout_blocks"]))),
        },
        "scoring_contract": {
            "primary_estimand": "complete_returned",
            "complete_returned": (
                "all returned events; OOD and single-shaft events remain negative evidence"
            ),
            "formal_clean": (
                "returned and joint-shaft and inside frozen patient support; historical "
                "Fig.4 comparability diagnostic only"
            ),
            "patient_distance": "four-component, weakest-mode LSE, patient training events",
            "heldout_r2": "equal-network mode prototypes evaluated on previously inspected held-out blocks",
            "source_topology": "not imputed when historical spikes are absent",
            "component_calibration": calibration,
            "old_classifier_to_patient_mode": _jsonable(label_semantics),
        },
        "legacy_formal_clean_parity": legacy_parity,
        "rows": rows,
        "field_rows": field_rows,
        "inputs": {
            "cohort_config": str(cohort_path),
            "cohort_config_sha256": _sha256(cohort_path),
            "reference_config": str(reference_path),
            "reference_config_sha256": _sha256(reference_path),
            "classifier_manifest": str(classifier["manifest_path"]),
            "classifier_manifest_sha256": classifier["manifest_sha256"],
            "libraries": library_inputs,
        },
        "claim_boundary": (
            "Complete-returned is the rev12 optimization estimand. Formal-clean only "
            "reproduces the historical Fig.4 diagnostic and cannot hide OOD or single-shaft "
            "events. Field-level ranking pools matching current-spatial-OU runs by field "
            "hash and counts a network seed once; record-level Pareto positions are not "
            "field evidence. Historical artifacts can nominate initializations but cannot satisfy "
            "the rev12 source-topology endpoint because spike-level source maps were not stored."
        ),
    }
    output = args.output.resolve()
    _atomic_json(output / "historical_rescore.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "library_candidate_id", "library", "candidate_id", "field_type",
        "n_networks", "same_network_both_modes", "same_network_both_fraction",
        "mean_returned_events", "mean_formal_clean_events", "mean_ood_fraction",
        "mean_patient_objective",
        "mean_weakest_mode_lse", "mean_mode_0_loss", "mean_mode_1_loss",
        "patient_train_k2_r2_on_heldout", "model_prototype_r2_on_heldout",
        "formal_clean_patient_objective", "formal_clean_model_prototype_r2",
        "formal_clean_scaled_contrast_r2",
        "historical_record_pareto",
    ]
    with (output / "historical_rescore.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            clean = row["estimands"]["formal_clean"]
            flat.update({
                "formal_clean_patient_objective": clean["mean_patient_objective"],
                "formal_clean_model_prototype_r2": clean[
                    "model_prototype_r2_on_heldout"
                ],
                "formal_clean_scaled_contrast_r2": clean["contrast"][
                    "heldout_train_scaled_r2"
                ],
            })
            writer.writerow({key: _jsonable(flat.get(key)) for key in fields})
    field_fields = [
        "field_id", "field_sha256", "substrate_stratum", "n_source_records",
        "n_networks", "same_network_both_fraction", "mean_returned_events",
        "mean_formal_clean_events", "mean_ood_fraction", "mean_patient_objective",
        "mean_mode_0_loss", "mean_mode_1_loss", "model_prototype_r2_on_heldout",
        "historical_field_pareto",
    ]
    with (output / "historical_field_rescore.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_fields)
        writer.writeheader()
        for row in field_rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in field_fields})
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(rows),
        "n_unique_fields": len(field_rows), "field_pareto": field_front,
        "best_distinct_field": best_distinct, "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
