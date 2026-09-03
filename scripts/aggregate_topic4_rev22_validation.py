#!/usr/bin/env python3
"""Strict rev22 Task 11 selection-blind validation aggregate.

This producer never runs an SNN and rejects patient ictal inputs.  It inventories the
complete frozen candidate x topology-unit grid before scoring.  Missing, corrupt,
runaway, or non-finite units make the candidate-level six-endpoint vector not
estimable; surviving-unit diagnostics remain explicit sidecars only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment,
    natural_kmeans,
    normalize_event_ranks,
    patient_profiles,
)
from src.topic4_rev20_dual_core_endpoint import embedding_from_training_arrays  # noqa: E402
from src.topic4_rev22_interictal_objective import (  # noqa: E402
    PatientBlockViews,
    block_split_floors,
    clipping_fractions,
    component_vector,
    embedding_features,
)
from src.topic4_rev22_validation import (  # noqa: E402
    C2ST_NOT_ESTIMABLE,
    RECALL_OK,
    classifier_two_sample_auc,
    fixed_budget_recall,
)
from src.topic4_shaft_aware import (  # noqa: E402
    contract_groups,
    contract_pairs,
    transform_patient_embedding,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402
from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402


DEFAULT_STAGE = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_interictal_identifiability"
)
DEFAULT_CONFIG = ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json"
PRIMARY = ("D_support", "D_order", "D_time_ms", "recall", "kmeans_alignment", "ood")
PHASES = ("qualification", "confirmation")
PRIMARY_NOT_ESTIMABLE = "PRIMARY_ENDPOINT_NOT_ESTIMABLE"
SELECTION_BLIND = "selection-blind"
FORBIDDEN_INPUT_MARKERS = ("patient_ictal", "seizure", "fig3", "fig5", "early_ictal")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _guard_nonictal(path: Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    normalized = str(resolved).lower().replace("interictal", "")
    if any(marker in normalized for marker in FORBIDDEN_INPUT_MARKERS):
        raise RuntimeError(f"patient-ictal boundary rejected {role}: {resolved}")
    return resolved


def _read_json(path: Path, role: str) -> tuple[dict, str]:
    path = _guard_nonictal(path, role)
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    return json.loads(path.read_text(encoding="utf-8")), _sha256(path)


def _load_npz(path: Path, role: str, required: Sequence[str]) -> tuple[dict, str]:
    path = _guard_nonictal(path, role)
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    digest = _sha256(path)
    with np.load(path, allow_pickle=False) as loaded:
        missing = sorted(set(required) - set(loaded.files))
        if missing:
            raise RuntimeError(f"{role} missing arrays: {missing}")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    return arrays, digest


def _classifier(payload: Mapping) -> dict:
    classifier = payload.get("direction_classifier")
    if not isinstance(classifier, Mapping):
        raise RuntimeError("classifier manifest lacks direction_classifier")
    required = (
        "coef", "intercept", "class_centers", "class_precisions",
        "ood_distance_thresholds",
    )
    missing = [key for key in required if key not in classifier]
    if missing:
        raise RuntimeError(f"direction classifier missing fields: {missing}")
    output = dict(classifier)
    for key in ("coef", "class_centers", "class_precisions", "ood_distance_thresholds"):
        output[key] = np.asarray(output[key], float)
        if not np.isfinite(output[key]).all():
            raise RuntimeError(f"direction classifier contains non-finite {key}")
    return output


def _phase_units(seed_manifest: Mapping, phase: str) -> list[dict]:
    block = seed_manifest.get(phase) or {}
    units = block.get("units") if isinstance(block, Mapping) else None
    if not isinstance(units, list) or not units:
        raise RuntimeError(f"seed manifest has no frozen {phase} units")
    output = []
    seen = set()
    for row in units:
        topology = int(row["topology_seed"])
        dynamics = int(row.get("dynamics_seed", topology))
        if topology in seen:
            raise RuntimeError(f"duplicate {phase} topology seed: {topology}")
        seen.add(topology)
        output.append({"topology_seed": topology, "dynamics_seed": dynamics})
    expected = 6 if phase == "qualification" else 12
    if len(output) != expected:
        raise RuntimeError(f"{phase} requires exactly {expected} topology units")
    return output


def _candidate_index(manifest: Mapping, frozen: Mapping) -> tuple[list[str], dict]:
    rows = {str(row["candidate_id"]): row for row in manifest.get("candidates", [])}
    ids = [str(value) for value in frozen.get("candidate_ids", [])]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("frozen candidate ids are empty or duplicated")
    missing = sorted(set(ids) - set(rows))
    if missing:
        raise RuntimeError(f"frozen candidates absent from execution manifest: {missing}")
    return ids, rows


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _invalid_unit(candidate_id: str, topology: int, dynamics: int, reason: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "topology_seed": int(topology),
        "dynamics_seed": int(dynamics),
        "artifact_status": reason,
        "primary_eligible": False,
        "failure_reasons": [reason],
        "runaway_early_stop_ms": None,
        "n_returned_families": 0,
        "arrays": None,
    }


def _validate_worker(worker_dir: Path, candidate: Mapping, topology: int, dynamics: int,
                     *, manifest_hash: str | None, seed_hash: str | None,
                     contact_names: Sequence[str]) -> dict:
    candidate_id = str(candidate["candidate_id"])
    stem = f"{candidate_id}_topo_{topology}_dyn_{dynamics}"
    json_path = worker_dir / f"{stem}.json"
    if not json_path.is_file():
        return _invalid_unit(candidate_id, topology, dynamics, "MISSING_WORKER")
    row = _invalid_unit(candidate_id, topology, dynamics, "INVALID_ARTIFACT")
    failures: list[str] = []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as error:
        row["failure_reasons"] = [f"JSON_READ:{error}"]
        return row
    if not str(payload.get("status", "")).endswith("_WORKER_COMPLETE"):
        failures.append("WORKER_STATUS_NOT_COMPLETE")
    if str(payload.get("candidate_id")) != candidate_id:
        failures.append("CANDIDATE_ID_MISMATCH")
    expected_mechanisms = candidate.get("mechanisms") or {}
    observed_mechanisms = payload.get("mechanism_freeze") or {}
    if expected_mechanisms:
        for key in ("g_EE", "g_EtoI", "ellipse_angle_deg", "ellipse_aspect_ratio"):
            if key in expected_mechanisms and not np.isclose(
                    float(expected_mechanisms[key]),
                    float(observed_mechanisms.get(key, np.nan)), rtol=0.0, atol=1e-12):
                failures.append("CANDIDATE_MECHANISM_MISMATCH")
                break
        if expected_mechanisms.get("Z_M") != "off" or observed_mechanisms.get("Z_M") != "off":
            failures.append("ZM_NOT_FROZEN_OFF")
    field_hash = (candidate.get("node_field") or {}).get("field_sha256")
    if field_hash is not None and payload.get("field_sha256") != field_hash:
        failures.append("NODE_FIELD_HASH_MISMATCH")
    if int(payload.get("topology_seed", -1)) != topology:
        failures.append("TOPOLOGY_SEED_MISMATCH")
    if int(payload.get("dynamics_seed", -1)) != dynamics:
        failures.append("DYNAMICS_SEED_MISMATCH")
    if manifest_hash is not None and payload.get("response_design_manifest_sha256") != manifest_hash:
        failures.append("EXECUTION_MANIFEST_HASH_MISMATCH")
    if seed_hash is not None and payload.get("seed_manifest_sha256") != seed_hash:
        failures.append("SEED_MANIFEST_HASH_MISMATCH")
    provenance = payload.get("provenance") or {}
    frozen_flag = provenance.get("runtime_modules_match_expected_commit")
    dirty_flag = provenance.get("runtime_modules_dirty")
    if type(frozen_flag) not in (bool, int) or frozen_flag != 1:
        failures.append("RUNTIME_MODULES_NOT_FROZEN")
    if type(dirty_flag) not in (bool, int) or dirty_flag != 0:
        failures.append("RUNTIME_MODULES_DIRTY_OR_UNKNOWN")
    expected_commit = str(provenance.get("expected_git_commit") or "")
    runtime_commit = str(provenance.get("git_commit") or "")
    if not _git_is_ancestor(expected_commit, runtime_commit):
        failures.append("WORKER_COMMIT_ANCESTRY_FAIL")
    if (not provenance.get("config_sha256") or
            provenance.get("config_sha256") != provenance.get("config_sha256_at_expected_commit")):
        failures.append("WORKER_CONFIG_NOT_AT_EXPECTED_COMMIT")
    simulation = payload.get("simulation") or {}
    if simulation.get("duration_ms") is None or not np.isclose(
            float(simulation["duration_ms"]), 20000.0):
        failures.append("SIMULATION_DURATION_NOT_20S")
    runaway = simulation.get("runaway_early_stop_ms")
    if runaway is not None:
        failures.append("RUNAWAY")
    arrays_record = payload.get("arrays") or {}
    npz_path = Path(arrays_record.get("path") or json_path.with_suffix(".npz"))
    if not npz_path.is_absolute():
        npz_path = worker_dir / npz_path
    arrays = None
    try:
        if npz_path.resolve().parent != worker_dir.resolve():
            raise RuntimeError("worker NPZ outside worker directory")
        if not npz_path.is_file() or arrays_record.get("sha256") != _sha256(npz_path):
            raise RuntimeError("worker NPZ missing or hash mismatch")
        with np.load(npz_path, allow_pickle=False) as loaded:
            required = {
                "contact_names", "onsets", "ranks", "event_returned",
                "topology_seed", "dynamics_seed", "active_fraction",
                "contact_envelope", "mechanism_parameters",
            }
            missing = sorted(required - set(loaded.files))
            if missing:
                raise RuntimeError(f"missing arrays {missing}")
            onsets = np.asarray(loaded["onsets"], float)
            ranks = np.asarray(loaded["ranks"], float)
            returned = np.asarray(loaded["event_returned"], bool)
            names = [str(value) for value in loaded["contact_names"]]
            if onsets.ndim != 2 or ranks.shape != onsets.shape or returned.shape != (len(onsets),):
                raise RuntimeError("onsets/ranks/returned shape mismatch")
            if onsets.shape[1] != len(contact_names) or names != list(contact_names):
                raise RuntimeError("contact order mismatch")
            if np.isinf(onsets).any() or np.isinf(ranks).any():
                raise RuntimeError("onsets or ranks contain infinity")
            for key in ("active_fraction", "contact_envelope", "mechanism_parameters"):
                if not np.isfinite(np.asarray(loaded[key], float)).all():
                    raise RuntimeError(f"non-finite required output {key}")
            if int(np.asarray(loaded["topology_seed"]).item()) != topology:
                raise RuntimeError("NPZ topology seed mismatch")
            if int(np.asarray(loaded["dynamics_seed"]).item()) != dynamics:
                raise RuntimeError("NPZ dynamics seed mismatch")
            onsets, ranks = onsets[returned], ranks[returned]
            if len(onsets) and not np.isfinite(onsets).any(axis=1).all():
                raise RuntimeError("returned family without recruited contact")
            arrays = {"onsets": onsets, "ranks": ranks}
    except Exception as error:
        failures.append(f"NPZ_INVALID:{error}")
    row.update(
        artifact_status="OK" if not failures else "INVALID_ARTIFACT",
        primary_eligible=not failures,
        failure_reasons=failures,
        runaway_early_stop_ms=runaway,
        n_returned_families=0 if arrays is None else int(len(arrays["onsets"])),
        arrays=arrays,
        worker_json={"path": str(json_path), "sha256": _sha256(json_path)},
        worker_npz=(None if arrays is None else {"path": str(npz_path), "sha256": _sha256(npz_path)}),
    )
    return row


def _old_embedding(training: Mapping) -> dict:
    return embedding_from_training_arrays(training)


def _mode_sidecars(onsets: np.ndarray, ranks: np.ndarray, mapped: np.ndarray) -> dict:
    normalized = normalize_event_ranks(ranks)
    output = {}
    for mode in (0, 1):
        selected = mapped == mode
        values = onsets[selected]
        rr = normalized[selected]
        output[str(mode)] = {
            "n_events": int(np.sum(selected)),
            "recruitment_probability": (
                np.isfinite(values).mean(axis=0).tolist() if len(values) else None
            ),
            "mean_rank_profile": (
                np.nanmean(rr, axis=0).tolist() if len(rr) else None
            ),
        }
    return output


def _masked_rank_features(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(ranks, float)
    valid = np.sum(np.isfinite(values), axis=1) >= 3
    selected = values[valid]
    if not len(selected):
        return np.zeros((0, values.shape[1]), float), valid
    return build_masked_kmeans_features(
        selected.T, np.isfinite(selected.T), impute="event_median"
    ), valid


def grouped_k2_vs_k1(ranks: np.ndarray, groups: np.ndarray, *, seed: int) -> dict:
    """Held-out K=2 versus K=1 likelihood with groups kept outside training folds."""
    features, valid = _masked_rank_features(ranks)
    group_values = np.asarray(groups)[valid]
    unique = np.unique(group_values)
    if len(features) < 12 or len(unique) < 2:
        return {"status": "NOT_ESTIMABLE_GROUPS", "delta_loglik_per_event": None}
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    deltas = []
    fold_rows = []
    for fold, (train, test) in enumerate(splitter.split(features, groups=group_values)):
        scores = []
        for components in (1, 2):
            model = GaussianMixture(
                n_components=components, covariance_type="diag", reg_covar=1e-5,
                n_init=5, random_state=int(seed) + fold,
            ).fit(features[train])
            scores.append(float(model.score(features[test])))
        delta = scores[1] - scores[0]
        deltas.append(delta)
        fold_rows.append({"fold": fold, "n_test": int(len(test)),
                          "n_test_groups": int(len(np.unique(group_values[test]))),
                          "delta_loglik_per_event": float(delta)})
    return {"status": "OK", "delta_loglik_per_event": float(np.mean(deltas)),
            "folds": fold_rows, "group_separated": True}


def grouped_patient_kmeans_benchmark(ranks: np.ndarray, labels: np.ndarray,
                                     groups: np.ndarray, *, seed: int) -> dict:
    """Patient KMeans benchmark with recording blocks held out from fitting."""
    features, valid = _masked_rank_features(ranks)
    labels = np.asarray(labels, int)[valid]
    group_values = np.asarray(groups)[valid]
    unique = np.unique(group_values)
    if len(features) < 12 or len(unique) < 2 or set(np.unique(labels)) != {0, 1}:
        return {"status": "NOT_ESTIMABLE_GROUPS", "balanced_alignment": None}
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    fold_rows = []
    for fold, (train, test) in enumerate(splitter.split(features, labels, group_values)):
        model = KMeans(n_clusters=2, n_init=50, random_state=int(seed) + fold).fit(features[train])
        train_alignment = best_binary_alignment(model.labels_, labels[train])
        swap = bool(np.mean(train_alignment["mapped_labels"] != model.labels_) > 0.5)
        predicted = model.predict(features[test])
        mapped = 1 - predicted if swap else predicted
        recalls = [float(np.mean(mapped[labels[test] == mode] == mode))
                   if np.any(labels[test] == mode) else np.nan for mode in (0, 1)]
        fold_rows.append({
            "fold": fold, "n_test": int(len(test)),
            "n_test_blocks": int(len(np.unique(group_values[test]))),
            "balanced_alignment": float(np.nanmean(recalls)),
        })
    return {
        "status": "OK",
        "balanced_alignment": float(np.mean([row["balanced_alignment"] for row in fold_rows])),
        "folds": fold_rows, "group_separated": True,
    }


def alignment_label_permutation(cluster_labels: np.ndarray, direction_labels: np.ndarray,
                                groups: np.ndarray | None, *, seed: int,
                                permutations: int = 512) -> dict:
    """Selection-blind alignment null preserving each topology's event count."""
    clusters = np.asarray(cluster_labels, int)
    labels = np.asarray(direction_labels, int)
    if len(clusters) != len(labels) or len(clusters) < 2:
        return {"status": "NOT_ESTIMABLE", "q95": None}
    observed = best_binary_alignment(clusters, labels)["balanced_alignment"]
    rng = np.random.default_rng(int(seed))
    null = []
    group_values = np.zeros(len(labels), int) if groups is None else np.asarray(groups)
    for _ in range(int(permutations)):
        shuffled = labels.copy()
        for group in np.unique(group_values):
            selected = np.flatnonzero(group_values == group)
            shuffled[selected] = rng.permutation(shuffled[selected])
        null.append(best_binary_alignment(clusters, shuffled)["balanced_alignment"])
    null = np.asarray(null, float)
    return {
        "status": "OK", "observed": observed, "q95": float(np.quantile(null, 0.95)),
        "p_upper": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "permutations": int(permutations), "within_group_shuffle": groups is not None,
    }


def network_identity_audit(cluster_labels: np.ndarray, network_groups: np.ndarray, *,
                           seed: int, permutations: int = 512) -> dict:
    """Test whether cluster membership is primarily explained by network identity."""
    labels = np.asarray(cluster_labels, int)
    groups = np.asarray(network_groups)
    if len(labels) != len(groups) or len(labels) < 2 or len(np.unique(groups)) < 2:
        return {"status": "NOT_ESTIMABLE_GROUPS", "nmi": None}
    observed = float(normalized_mutual_info_score(groups, labels))
    rng = np.random.default_rng(int(seed))
    null = np.asarray([
        normalized_mutual_info_score(groups, rng.permutation(labels))
        for _ in range(int(permutations))
    ], float)
    return {
        "status": "OK", "nmi": observed,
        "permutation_q95": float(np.quantile(null, 0.95)),
        "permutation_p_upper": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "network_identity_driven": bool(observed > np.quantile(null, 0.95)),
        "permutations": int(permutations), "seed": int(seed),
    }


def _score_kmeans_ood(onsets: np.ndarray, ranks: np.ndarray, *, groups: Mapping,
                      old_embedding: Mapping, classifier: Mapping, seed: int,
                      training_ranks: np.ndarray, training_labels: np.ndarray,
                      network_groups: np.ndarray | None = None,
                      diagnostics: bool = True) -> dict:
    assigned = assign_direction_modes(onsets, groups=groups, embedding=old_embedding,
                                      classifier=classifier)
    readable = np.sum(np.isfinite(onsets), axis=1) >= 3
    all_ood = ~readable | np.asarray(assigned["ood"], bool)
    natural = natural_kmeans(ranks, np.asarray(assigned["labels"], int), random_state=int(seed))
    output = {
        "terminology": SELECTION_BLIND,
        "n_returned": int(len(onsets)),
        "n_readable": int(np.sum(readable)),
        "unreadable_fraction": float(np.mean(~readable)) if len(onsets) else 1.0,
        "ood_fraction": float(np.mean(all_ood)) if len(onsets) else 1.0,
        "precision_like": float(1.0 - np.mean(all_ood)) if len(onsets) else 0.0,
        "natural_kmeans_status": natural["status"],
    }
    if natural["status"] != "OK":
        output.update(kmeans_alignment=None, mapped_labels=None, mode_sidecars=None)
        return output
    valid = np.asarray(natural["valid_event_mask"], bool)
    clusters = np.asarray(natural["cluster_labels"], int)
    alignment = best_binary_alignment(clusters, np.asarray(assigned["labels"], int)[valid])
    mapped = np.asarray(alignment["mapped_labels"], int)
    matrix = np.full((2, 2), np.nan)
    model_profiles = np.full((2, ranks.shape[1]), np.nan)
    normalized = normalize_event_ranks(ranks[valid])
    patient = patient_profiles(training_ranks, training_labels)
    for mode in (0, 1):
        selected = mapped == mode
        if np.any(selected):
            model_profiles[mode] = np.nanmean(normalized[selected], axis=0)
        for target in (0, 1):
            finite = np.isfinite(model_profiles[mode]) & np.isfinite(patient[target])
            if np.sum(finite) >= 3:
                matrix[mode, target] = float(spearmanr(
                    model_profiles[mode, finite], patient[target, finite]
                ).statistic)
    output.update(
        kmeans_alignment=natural["direction_balanced_alignment"],
        kmeans_direction_purity=natural["direction_purity"],
        kmeans_seed_ami_median=natural["kmeans_seed_ami_median"],
        heldout_gmm_k2_minus_k1_loglik_per_event=natural[
            "heldout_gmm_k2_minus_k1_loglik_per_event"
        ],
        cluster_counts=np.bincount(mapped, minlength=2).tolist(),
        minority_proportion=float(np.min(np.bincount(mapped, minlength=2)) / len(mapped)),
        prototype_spearman_matrix=matrix.tolist(),
        mapped_labels=mapped.tolist(),
        valid_event_mask=valid.tolist(),
        mode_sidecars=_mode_sidecars(onsets[valid], ranks[valid], mapped),
    )
    if diagnostics:
        output["alignment_label_permutation"] = alignment_label_permutation(
            clusters, np.asarray(assigned["labels"], int)[valid],
            None if network_groups is None else np.asarray(network_groups)[valid],
            seed=seed + 600,
        )
    if diagnostics and network_groups is not None:
        network_groups = np.asarray(network_groups)
        output["network_identity_audit"] = network_identity_audit(
            mapped, network_groups[valid], seed=seed + 700,
        )
        output["grouped_k2_vs_k1"] = grouped_k2_vs_k1(
            ranks, network_groups, seed=seed + 800,
        )
    return output


def _score_candidate(units: Sequence[Mapping], context: Mapping, *, phase: str,
                     candidate_id: str, seed: int,
                     recall_subsamples: int, c2st_resamples: int) -> dict:
    expected = len(units)
    valid = [unit for unit in units if unit["primary_eligible"]]
    base = {
        "phase": phase,
        "candidate_id": candidate_id,
        "expected_unit_count": expected,
        "valid_unit_count": len(valid),
        "units": [{key: value for key, value in unit.items() if key != "arrays"} for unit in units],
        "primary_status": "OK" if len(valid) == expected else PRIMARY_NOT_ESTIMABLE,
        "primary_endpoints": {name: None for name in PRIMARY},
        "sidecars_only_due_to_incomplete_grid": len(valid) != expected,
    }
    if not valid:
        return base
    all_onsets = np.concatenate([unit["arrays"]["onsets"] for unit in valid])
    all_ranks = np.concatenate([unit["arrays"]["ranks"] for unit in valid])
    model_groups = np.concatenate([
        np.full(len(unit["arrays"]["onsets"]), unit["topology_seed"], dtype=int)
        for unit in valid
    ])
    vector = component_vector(
        all_onsets, context["held_reference"], context["groups"], context["pairs"],
        context["new_embedding"], n_pair_min=context["n_pair_min"],
        lag_cap_ms=context["lag_cap_ms"], cover_quantile=context["cover_quantile"],
    )
    kmeans = _score_kmeans_ood(
        all_onsets, all_ranks, groups=context["groups"], old_embedding=context["old_embedding"],
        classifier=context["classifier"], seed=context["kmeans_seed"],
        training_ranks=context["training_ranks"], training_labels=context["training_labels"],
        network_groups=model_groups,
    )
    recalls = []
    recall_rows = []
    for index, unit in enumerate(valid):
        model_z = transform_patient_embedding(
            embedding_features(unit["arrays"]["onsets"], context["groups"],
                               lag_cap_ms=context["lag_cap_ms"]),
            context["new_embedding"],
        )
        recall = fixed_budget_recall(
            model_z, context["held_z"], n_cov=context["n_cov"], r_cov=context["r_cov"],
            subsamples=recall_subsamples, seed=seed + index,
        )
        recall_rows.append({"topology_seed": unit["topology_seed"], **recall})
        if recall["status"] == RECALL_OK:
            recalls.append(recall["recall"])
    recall_ok = len(recalls) == len(valid)
    endpoints = {
        "D_support": vector["D_support"]["value"],
        "D_order": vector["D_order"]["value"],
        "D_time_ms": vector["D_lag"]["value"],
        "recall": float(np.mean(recalls)) if recall_ok else None,
        "kmeans_alignment": kmeans.get("kmeans_alignment"),
        "ood": kmeans["ood_fraction"],
    }
    component_status = {
        "D_support": vector["D_support"]["status"],
        "D_order": vector["D_order"]["status"],
        "D_time_ms": vector["D_lag"]["status"],
        "recall": "OK" if recall_ok else "NOT_ESTIMABLE_LOW_YIELD",
        "kmeans_alignment": kmeans["natural_kmeans_status"],
        "ood": "OK",
    }
    all_primary_ok = (
        len(valid) == expected
        and all(component_status[name] == "OK" for name in PRIMARY)
        and all(value is not None and np.isfinite(value) for value in endpoints.values())
    )
    floor = block_split_floors(
        context["held_views"], [{"key": "candidate", "n": len(all_onsets),
                                 "components": ["D_support", "D_order", "D_lag"]}],
        draws=context["floor_draws"], seed=seed,
        n_pair_min=context["n_pair_min"], cover_quantile=context["cover_quantile"],
    )["candidate"]
    lag_floor = floor["D_lag"]
    lag_scale = None
    if lag_floor["q50"] is not None and lag_floor["q95"] is not None:
        lag_scale = max(float(lag_floor["q95"] - lag_floor["q50"]), 1e-9)
    lag_normalized = (
        None if endpoints["D_time_ms"] is None or lag_scale is None
        else max(0.0, (float(endpoints["D_time_ms"]) - float(lag_floor["q50"])) / lag_scale)
    )
    model_z = transform_patient_embedding(
        embedding_features(all_onsets, context["groups"], lag_cap_ms=context["lag_cap_ms"]),
        context["new_embedding"],
    )
    c2st = classifier_two_sample_auc(
        context["held_z"], context["held_groups"], model_z, model_groups,
        seed=seed, resamples=c2st_resamples, permutations=c2st_resamples,
    )
    per_network_modes = []
    for index, unit in enumerate(valid):
        mode = _score_kmeans_ood(
            unit["arrays"]["onsets"], unit["arrays"]["ranks"], groups=context["groups"],
            old_embedding=context["old_embedding"], classifier=context["classifier"],
            seed=context["kmeans_seed"], training_ranks=context["training_ranks"],
            training_labels=context["training_labels"], diagnostics=False,
        )
        counts = mode.get("cluster_counts")
        per_network_modes.append({
            "topology_seed": unit["topology_seed"],
            "status": mode["natural_kmeans_status"],
            "mode_counts": counts,
            "mode_proportions": (None if not counts else (np.asarray(counts) / sum(counts)).tolist()),
            "both_modes": bool(counts and min(counts) > 0),
        })
    equal_props = [row["mode_proportions"] for row in per_network_modes
                   if row["mode_proportions"] is not None]
    clip_units = [clipping_fractions(unit["arrays"]["onsets"],
                                    lag_cap_ms=context["lag_cap_ms"]) for unit in valid]
    base.update(
        primary_status="OK" if all_primary_ok else PRIMARY_NOT_ESTIMABLE,
        primary_endpoints=endpoints if all_primary_ok else {name: None for name in PRIMARY},
        conditional_survivor_endpoints=endpoints,
        endpoint_status=component_status,
        heldout_timing={"raw_ms": endpoints["D_time_ms"],
                        "floor_normalized_excess": lag_normalized, "floor": lag_floor},
        heldout_count_matched_floors=floor,
        secondary={"D_cloud_composite": vector["D_cloud_composite"],
                   "yield_total": int(len(all_onsets)), "yield_per_unit": [
                       int(unit["n_returned_families"]) for unit in valid],
                   "c2st": {**c2st, "formal_endpoint": "separability",
                            "raw_auc_role": "diagnostic_only"}},
        recall_units=recall_rows,
        kmeans_ood=kmeans,
        clipping={
            "pooled": clipping_fractions(all_onsets, lag_cap_ms=context["lag_cap_ms"]),
            "per_unit": clip_units,
        },
        mode_proportions={
            "pooled": (None if kmeans.get("cluster_counts") is None else
                       (np.asarray(kmeans["cluster_counts"]) /
                        sum(kmeans["cluster_counts"])).tolist()),
            "equal_network_weighted": (None if not equal_props else
                                       np.mean(equal_props, axis=0).tolist()),
            "networks_expressing_both_modes": int(sum(row["both_modes"] for row in per_network_modes)),
            "networks_evaluable": int(len(equal_props)),
            "network_dual_mode_fraction": (None if not equal_props else float(
                sum(row["both_modes"] for row in per_network_modes) / len(equal_props)
            )),
            "per_network": per_network_modes,
        },
    )
    return base


def _primary_vector_from_units(units: Sequence[Mapping], context: Mapping, *, seed: int,
                               recall_subsamples: int) -> dict | None:
    """Re-pool a topology resample and recompute every nonlinear primary endpoint."""
    if not units or any(not unit["primary_eligible"] for unit in units):
        return None
    onsets = np.concatenate([unit["arrays"]["onsets"] for unit in units])
    ranks = np.concatenate([unit["arrays"]["ranks"] for unit in units])
    vector = component_vector(
        onsets, context["held_reference"], context["groups"], context["pairs"],
        context["new_embedding"], n_pair_min=context["n_pair_min"],
        lag_cap_ms=context["lag_cap_ms"], cover_quantile=context["cover_quantile"],
    )
    ko = _score_kmeans_ood(
        onsets, ranks, groups=context["groups"], old_embedding=context["old_embedding"],
        classifier=context["classifier"], seed=context["kmeans_seed"],
        training_ranks=context["training_ranks"], training_labels=context["training_labels"],
        diagnostics=False,
    )
    recalls = []
    for index, unit in enumerate(units):
        z = transform_patient_embedding(
            embedding_features(unit["arrays"]["onsets"], context["groups"],
                               lag_cap_ms=context["lag_cap_ms"]), context["new_embedding"],
        )
        recall = fixed_budget_recall(
            z, context["held_z"], n_cov=context["n_cov"], r_cov=context["r_cov"],
            subsamples=recall_subsamples, seed=seed + index,
        )
        if recall["status"] != RECALL_OK:
            return None
        recalls.append(recall["recall"])
    output = {
        "D_support": vector["D_support"]["value"],
        "D_order": vector["D_order"]["value"],
        "D_time_ms": vector["D_lag"]["value"],
        "recall": float(np.mean(recalls)),
        "kmeans_alignment": ko.get("kmeans_alignment"),
        "ood": ko["ood_fraction"],
    }
    if any(value is None or not np.isfinite(value) for value in output.values()):
        return None
    return output


def paired_nonlinear_bootstrap(full_units: Sequence[Mapping], locked_units: Sequence[Mapping],
                               context: Mapping, *, draws: int, seed: int,
                               recall_subsamples: int) -> dict:
    """Paired topology bootstrap that re-pools events and re-scores every draw."""
    full_by_seed = {int(unit["topology_seed"]): unit for unit in full_units}
    locked_by_seed = {int(unit["topology_seed"]): unit for unit in locked_units}
    if set(full_by_seed) != set(locked_by_seed) or len(full_by_seed) < 2:
        return {"status": PRIMARY_NOT_ESTIMABLE, "reason": "UNPAIRED_TOPOLOGY_GRID"}
    topologies = np.asarray(sorted(full_by_seed), int)
    if any(not unit["primary_eligible"] for unit in full_units + locked_units):
        return {"status": PRIMARY_NOT_ESTIMABLE, "reason": "INVALID_EXPECTED_UNIT"}
    point_full = _primary_vector_from_units(full_units, context, seed=seed,
                                            recall_subsamples=recall_subsamples)
    point_locked = _primary_vector_from_units(locked_units, context, seed=seed,
                                              recall_subsamples=recall_subsamples)
    if point_full is None or point_locked is None:
        return {"status": PRIMARY_NOT_ESTIMABLE, "reason": "PRIMARY_ENDPOINT_NOT_ESTIMABLE"}
    point = {
        name: ((point_full[name] - point_locked[name]) if name in ("recall", "kmeans_alignment")
               else (point_locked[name] - point_full[name]))
        for name in PRIMARY
    }
    rng = np.random.default_rng(int(seed))
    samples = {name: [] for name in PRIMARY}
    failed_draws = 0
    for draw in range(int(draws)):
        selected = rng.choice(topologies, size=len(topologies), replace=True)
        f_units = [full_by_seed[int(value)] for value in selected]
        l_units = [locked_by_seed[int(value)] for value in selected]
        f_score = _primary_vector_from_units(
            f_units, context, seed=seed + 10 + draw * 2,
            recall_subsamples=recall_subsamples,
        )
        l_score = _primary_vector_from_units(
            l_units, context, seed=seed + 10 + draw * 2,
            recall_subsamples=recall_subsamples,
        )
        if f_score is None or l_score is None:
            failed_draws += 1
            continue
        for name in PRIMARY:
            delta = (f_score[name] - l_score[name]) if name in ("recall", "kmeans_alignment") else (
                l_score[name] - f_score[name]
            )
            samples[name].append(float(delta))
    if failed_draws or any(len(values) != int(draws) for values in samples.values()):
        return {"status": PRIMARY_NOT_ESTIMABLE, "reason": "BOOTSTRAP_DRAW_NOT_ESTIMABLE",
                "failed_draws": int(failed_draws), "draws": int(draws)}
    endpoints = {
        name: {"status": "OK", "delta": float(point[name]),
               "lo": float(np.quantile(samples[name], 0.05)),
               "hi": float(np.quantile(samples[name], 0.95)),
               "draws": int(draws), "positive_is_better": True}
        for name in PRIMARY
    }
    points = np.asarray([point[name] for name in PRIMARY], float)
    intervals = [(endpoints[name]["lo"], endpoints[name]["hi"]) for name in PRIMARY]
    if np.any(points < 0):
        status = "TRADEOFF"
    elif any(lo > 0 for lo, _ in intervals) and not any(hi < 0 for _, hi in intervals):
        status = "PARETO_SUPPORTED"
    elif all(lo <= 0 <= hi for lo, hi in intervals):
        status = "NON_IDENTIFIABLE_AT_CURRENT_SEEDS"
    else:
        status = "TRADEOFF"
    return {
        "status": status, "endpoints": endpoints,
        "bootstrap_method": "paired_topology_resample_then_repool_and_rescore_all_nonlinear_endpoints",
        "draws": int(draws), "seed": int(seed),
    }


def _unit_endpoint_row(unit: Mapping, context: Mapping, *, seed: int,
                       recall_subsamples: int) -> dict:
    if not unit["primary_eligible"]:
        return {"topology_seed": unit["topology_seed"], **{name: None for name in PRIMARY}}
    onsets, ranks = unit["arrays"]["onsets"], unit["arrays"]["ranks"]
    vector = component_vector(
        onsets, context["held_reference"], context["groups"], context["pairs"],
        context["new_embedding"], n_pair_min=context["n_pair_min"],
        lag_cap_ms=context["lag_cap_ms"], cover_quantile=context["cover_quantile"],
    )
    z = transform_patient_embedding(
        embedding_features(onsets, context["groups"], lag_cap_ms=context["lag_cap_ms"]),
        context["new_embedding"],
    )
    recall = fixed_budget_recall(
        z, context["held_z"], n_cov=context["n_cov"], r_cov=context["r_cov"],
        subsamples=recall_subsamples, seed=seed,
    )
    ko = _score_kmeans_ood(
        onsets, ranks, groups=context["groups"], old_embedding=context["old_embedding"],
        classifier=context["classifier"], seed=context["kmeans_seed"],
        training_ranks=context["training_ranks"], training_labels=context["training_labels"],
        diagnostics=False,
    )
    return {
        "topology_seed": unit["topology_seed"],
        "D_support": vector["D_support"]["value"],
        "D_order": vector["D_order"]["value"],
        "D_time_ms": vector["D_lag"]["value"],
        "recall": recall.get("recall"),
        "kmeans_alignment": ko.get("kmeans_alignment"),
        "ood": ko.get("ood_fraction"),
    }


def aggregate_validation(*, frozen_candidates_path: Path, candidate_manifest_path: Path,
                         seed_manifest_path: Path, qualification_worker_dir: Path,
                         confirmation_worker_dir: Path, training_contract_path: Path,
                         patient_training_target_path: Path, heldout_path: Path,
                         contact_contract_path: Path, classifier_manifest_path: Path,
                         output_dir: Path, n_cov: int, r_cov: float,
                         floor_draws: int = 64, bootstrap_draws: int = 512,
                         recall_subsamples: int = 200, c2st_resamples: int = 20,
                         lag_cap_ms: float = 180.0, n_pair_min: int = 5,
                         cover_quantile: float = 0.9, kmeans_seed: int = 20260902) -> dict:
    frozen, frozen_hash = _read_json(frozen_candidates_path, "frozen candidates")
    manifest, manifest_hash = _read_json(candidate_manifest_path, "execution candidate manifest")
    seeds, seed_hash = _read_json(seed_manifest_path, "seed manifest")
    contract, contract_hash = _read_json(contact_contract_path, "contact contract")
    classifier_payload, classifier_hash = _read_json(classifier_manifest_path, "classifier manifest")
    if frozen.get("schema_id") != "topic4_rev22_dci_frozen_candidates_v1":
        raise RuntimeError("unexpected frozen candidate schema")
    if manifest.get("schema_id") != "topic4_rev22_dci_execution_candidate_manifest_v1":
        raise RuntimeError("unexpected execution candidate manifest schema")
    if seeds.get("schema_id") != "topic4_rev22_dci_seed_manifest_v1":
        raise RuntimeError("unexpected seed manifest schema")
    frozen_manifest_hash = frozen.get("execution_candidate_manifest_sha256")
    if not frozen_manifest_hash or frozen_manifest_hash != manifest_hash:
        raise RuntimeError("execution candidate manifest changed after candidate freeze")
    frozen_seed_hash = frozen.get("seed_manifest_sha256")
    if not frozen_seed_hash or frozen_seed_hash != seed_hash:
        raise RuntimeError("seed manifest changed after candidate freeze")
    if not frozen.get("response_design_manifest_sha256"):
        raise RuntimeError("frozen candidates lack response-design binding")
    ids, candidates = _candidate_index(manifest, frozen)
    training_contract, training_contract_hash = _load_npz(
        training_contract_path, "patient training objective contract",
        ("feature_center", "feature_scale", "pca_components", "sw_directions",
         "reference_z", "contact_names", "patient_train_onsets_ms"),
    )
    training, training_hash = _load_npz(
        patient_training_target_path, "patient training KMeans/OOD target",
        ("feature_center", "feature_scale", "pca_components", "sw_directions",
         "global_reference_z", "contact_names", "patient_train_ranks",
         "patient_train_old_labels"),
    )
    heldout, heldout_hash = _load_npz(
        heldout_path, "patient interictal heldout contract",
        ("contact_names", "heldout_onsets", "heldout_ranks", "heldout_old_labels",
         "heldout_block_ids"),
    )
    names = [str(value) for value in training_contract["contact_names"]]
    if names != [str(value) for value in training["contact_names"]] or names != [str(value) for value in heldout["contact_names"]]:
        raise RuntimeError("training, heldout and worker contact orders differ")
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    new_embedding = {
        "center": training_contract["feature_center"], "scale": training_contract["feature_scale"],
        "components": training_contract["pca_components"], "directions": training_contract["sw_directions"],
        "reference_z": training_contract["reference_z"],
    }
    old_embedding = _old_embedding(training)
    held_ms = np.asarray(heldout["heldout_onsets"], float) * 1000.0
    held_groups = np.asarray(heldout["heldout_block_ids"])
    held_views = PatientBlockViews(held_ms, held_groups, groups, pairs, new_embedding,
                                   lag_cap_ms=lag_cap_ms)
    context = {
        "groups": groups, "pairs": pairs, "new_embedding": new_embedding,
        "old_embedding": old_embedding, "classifier": _classifier(classifier_payload),
        "training_ranks": np.asarray(training["patient_train_ranks"], float),
        "training_labels": np.asarray(training["patient_train_old_labels"], int),
        "held_views": held_views, "held_reference": held_views.full_reference(),
        "held_z": held_views.z, "held_groups": held_groups,
        "n_cov": int(n_cov), "r_cov": float(r_cov), "floor_draws": int(floor_draws),
        "n_pair_min": int(n_pair_min), "lag_cap_ms": float(lag_cap_ms),
        "cover_quantile": float(cover_quantile),
        "kmeans_seed": int(kmeans_seed),
    }
    patient_natural = natural_kmeans(
        np.asarray(heldout["heldout_ranks"], float),
        np.asarray(heldout["heldout_old_labels"], int), random_state=int(kmeans_seed),
    )
    patient_benchmark = {
        "grouping_unit": "recording_block",
        "grouped_k2_vs_k1": grouped_k2_vs_k1(
            np.asarray(heldout["heldout_ranks"], float), held_groups,
            seed=int(kmeans_seed) + 900,
        ),
        "natural_kmeans_status": patient_natural["status"],
        "balanced_alignment": patient_natural.get("direction_balanced_alignment"),
        "cluster_counts": patient_natural.get("cluster_counts"),
        "grouped_alignment_benchmark": grouped_patient_kmeans_benchmark(
            np.asarray(heldout["heldout_ranks"], float),
            np.asarray(heldout["heldout_old_labels"], int), held_groups,
            seed=int(kmeans_seed) + 1000,
        ),
    }
    phase_results = {}
    raw_units_by_phase: dict[str, dict[str, list[dict]]] = {}
    inventory = []
    for phase, worker_dir in (("qualification", qualification_worker_dir),
                              ("confirmation", confirmation_worker_dir)):
        units_contract = _phase_units(seeds, phase)
        phase_rows = []
        raw_units_by_phase[phase] = {}
        for candidate_index, candidate_id in enumerate(ids):
            units = [
                _validate_worker(
                    Path(worker_dir), candidates[candidate_id], unit["topology_seed"],
                    unit["dynamics_seed"],
                    manifest_hash=frozen.get("response_design_manifest_sha256"),
                    seed_hash=seed_hash, contact_names=names,
                ) for unit in units_contract
            ]
            inventory.extend({"phase": phase, **{k: v for k, v in unit.items() if k != "arrays"}}
                             for unit in units)
            raw_units_by_phase[phase][candidate_id] = units
            scored = _score_candidate(
                units, context, phase=phase, candidate_id=candidate_id,
                seed=kmeans_seed + candidate_index * 1000,
                recall_subsamples=recall_subsamples, c2st_resamples=c2st_resamples,
            )
            scored["family_membership"] = candidates[candidate_id].get("family_membership", [])
            scored["unit_endpoints"] = [
                _unit_endpoint_row(unit, context, seed=kmeans_seed + candidate_index * 1000 + i,
                                   recall_subsamples=recall_subsamples)
                for i, unit in enumerate(units)
            ]
            phase_rows.append(scored)
        phase_results[phase] = phase_rows
    confirmation = {row["candidate_id"]: row for row in phase_results["confirmation"]}
    mask_map = frozen.get("mask_to_candidates") or {}
    branch = str(frozen.get("branch", "primary"))
    full_mask = "M1100" if "fallback" in branch.lower() else "M1111"
    locked_masks = ("M1000", "M0100") if full_mask == "M1100" else (
        "M0111", "M1011", "M1101", "M1110"
    )
    contrasts = []
    full_ids = [value for value in mask_map.get(full_mask, []) if value in confirmation]
    for full_id in full_ids:
        for mask in locked_masks:
            for locked_id in [value for value in mask_map.get(mask, []) if value in confirmation]:
                contrasts.append({
                    "full_mask": full_mask, "full_candidate_id": full_id,
                    "locked_mask": mask, "locked_candidate_id": locked_id,
                    **paired_nonlinear_bootstrap(
                        raw_units_by_phase["confirmation"][full_id],
                        raw_units_by_phase["confirmation"][locked_id], context,
                        draws=bootstrap_draws, seed=kmeans_seed,
                        recall_subsamples=recall_subsamples,
                    ),
                })
    reference_contrasts = []
    reference_ids = [value for value in mask_map.get("M0000", []) if value in confirmation]
    for reference_id in reference_ids:
        for candidate_id in ids:
            if candidate_id == reference_id:
                continue
            reference_contrasts.append({
                "candidate_id": candidate_id, "reference_candidate_id": reference_id,
                **paired_nonlinear_bootstrap(
                    raw_units_by_phase["confirmation"][candidate_id],
                    raw_units_by_phase["confirmation"][reference_id], context,
                    draws=bootstrap_draws, seed=kmeans_seed + 20000,
                    recall_subsamples=recall_subsamples,
                ),
            })
    payload = {
        "schema_id": "topic4_rev22_dci_validation_aggregate_v1",
        "status": "VALIDATION_AGGREGATE_COMPLETE",
        "scientific_role": "development_only_selection_blind_interictal_validation",
        "terminology": {
            "kmeans_and_ood": SELECTION_BLIND,
            "heldout_views": "recording-block held-out interictal",
            "independent_claim_for_kmeans_or_ood": False,
        },
        "snn_simulation_run": False,
        "patient_ictal_input_read": False,
        "frozen_support_budget": {"n_cov": int(n_cov), "r_cov": float(r_cov)},
        "phases": phase_results,
        "paired_pareto_contrasts": contrasts,
        "paired_reference_contrasts": reference_contrasts,
        "patient_recording_block_benchmark": patient_benchmark,
        "unit_inventory": inventory,
        "input_hashes": {
            "frozen_candidates": frozen_hash, "candidate_manifest": manifest_hash,
            "seed_manifest": seed_hash, "training_contract": training_contract_hash,
            "patient_training_target": training_hash, "heldout_contract": heldout_hash,
            "contact_contract": contract_hash, "classifier_manifest": classifier_hash,
        },
        "claim_boundary": (
            "Development-only interictal validation. KMeans and OOD are selection-blind, "
            "not independent; no patient ictal input or SNN simulation is used."
        ),
    }
    _atomic_json(Path(output_dir) / "validation_aggregate.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frozen-candidates", type=Path,
                        default=DEFAULT_STAGE / "response_fit/frozen_candidates.json")
    parser.add_argument("--candidate-manifest", type=Path,
                        default=DEFAULT_STAGE / "response_design/execution_candidate_manifest.json")
    parser.add_argument("--seed-manifest", type=Path,
                        default=DEFAULT_STAGE / "response_design/seed_manifest.json")
    parser.add_argument("--qualification-workers", type=Path,
                        default=DEFAULT_STAGE / "qualification/workers")
    parser.add_argument("--confirmation-workers", type=Path,
                        default=DEFAULT_STAGE / "confirmation/workers")
    parser.add_argument("--training-contract", type=Path,
                        default=DEFAULT_STAGE / "objective_qualification/patient_training_contract_v1.npz")
    parser.add_argument("--patient-training-target", type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--contact-contract", type=Path)
    parser.add_argument("--classifier-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE / "validation")
    parser.add_argument("--floor-draws", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=4096)
    parser.add_argument("--recall-subsamples", type=int, default=200)
    parser.add_argument("--c2st-resamples", type=int, default=20)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path("/home/honglab/leijiaxin/HFOsp")
    def config_path(name: str) -> Path:
        record = config["inputs"][name]
        path = root / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"configured input changed: {name}")
        return path
    support_config_path = config_path("patient_support_config")
    support = json.loads(support_config_path.read_text(encoding="utf-8"))
    classifier_record = support["inputs"]["old_ab_train_only_classifier"]
    classifier_path = root / classifier_record["path"]
    if not classifier_path.is_file() or _sha256(classifier_path) != classifier_record["sha256"]:
        raise RuntimeError("configured direction classifier changed")
    response_fit = json.loads((DEFAULT_STAGE / "response_fit/response_fit.json").read_text())
    budget = response_fit["recall_support_budget"]
    if budget.get("status") != "OK" or budget.get("n_cov") is None or budget.get("r_cov") is None:
        raise RuntimeError("frozen recall support budget is not estimable")
    payload = aggregate_validation(
        frozen_candidates_path=args.frozen_candidates,
        candidate_manifest_path=args.candidate_manifest,
        seed_manifest_path=args.seed_manifest,
        qualification_worker_dir=args.qualification_workers,
        confirmation_worker_dir=args.confirmation_workers,
        training_contract_path=args.training_contract,
        patient_training_target_path=args.patient_training_target or config_path("patient_training_target"),
        heldout_path=args.heldout or config_path("patient_heldout_npz"),
        contact_contract_path=args.contact_contract or config_path("contact_contract"),
        classifier_manifest_path=args.classifier_manifest or classifier_path,
        output_dir=args.output_dir, n_cov=int(budget["n_cov"]), r_cov=float(budget["r_cov"]),
        floor_draws=args.floor_draws, bootstrap_draws=args.bootstrap_draws,
        recall_subsamples=args.recall_subsamples, c2st_resamples=args.c2st_resamples,
        lag_cap_ms=float(config["objective"]["lag_cap_ms"]),
        n_pair_min=int(config["objective"]["n_pair_min"]),
        cover_quantile=float(config["objective"]["cover_quantile"]),
        kmeans_seed=20260902,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output_dir),
                      "candidates": len(payload["phases"]["confirmation"])}, indent=2))


if __name__ == "__main__":
    main()
