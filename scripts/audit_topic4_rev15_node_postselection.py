#!/usr/bin/env python3
"""Audit same-network natural KMeans after a robust Node field is selected."""
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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import rescore_topic4_rev13_exact_off_static_node as exact  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.paper_figures import plot_fig4_spatial_edge_flow_validation as fig4  # noqa: E402
from scripts.run_topic4_rev15_m3_robust_candidate_worker import WORKER_STATUS  # noqa: E402
from src.topic4_node_dualmode import normalize_event_ranks  # noqa: E402
from src.topic4_shaft_aware_direction import all_event_shaft_participation  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_node_postselection.json"
OUTPUT_SCHEMA = "topic4_rev15_node_postselection_audit_v1"
SEMANTIC_MODE_ORDER = (1, 0)
SEMANTIC_MODE_NAMES = ("MTA", "MTB")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_record(
    record: Mapping[str, Any], *, artifact_root: Path,
    repository_root: Path = ROOT,
) -> Path:
    value = Path(str(record["path"]))
    candidates = (value,) if value.is_absolute() else (
        artifact_root / value, repository_root / value,
    )
    for path in candidates:
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"post-selection input changed: {record['path']}")


def _provenance() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    return {
        "analysis_commit": head,
        "worktree_status": status,
        "formal_ready": not status,
        "snn_simulation_run": False,
    }


def clean_event_mask(
    *, readable: np.ndarray, onsets: np.ndarray, ood: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    readable = np.asarray(readable, dtype=bool)
    onsets = np.asarray(onsets, dtype=float)
    ood = np.asarray(ood, dtype=bool)
    if onsets.ndim != 2 or onsets.shape[0] != len(readable):
        raise ValueError("primary onsets and readable mask do not align")
    if ood.shape != readable.shape:
        raise ValueError("primary OOD mask does not align")
    icl = np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
    scl = np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
    dual_shaft = icl & scl
    clean = readable & dual_shaft & ~ood
    return clean, {
        "n_primary": int(len(clean)),
        "n_readable": int(np.sum(readable)),
        "n_dual_shaft": int(np.sum(dual_shaft)),
        "n_in_support": int(np.sum(~ood)),
        "n_formal_clean": int(np.sum(clean)),
        "definition": "Fig4-readable AND both shafts AND frozen-classifier in-support",
    }


def _mean_profiles(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = normalize_event_ranks(np.asarray(ranks, dtype=float))
    labels = np.asarray(labels, dtype=int)
    output = np.full((2, values.shape[1]), np.nan, dtype=float)
    for row, mode in enumerate(SEMANTIC_MODE_ORDER):
        selected = values[labels == mode]
        if len(selected):
            count = np.sum(np.isfinite(selected), axis=0)
            total = np.nansum(selected, axis=0)
            output[row] = np.divide(
                total, count, out=np.full(count.shape, np.nan, dtype=float),
                where=count > 0,
            )
    return output


def _similarity(model_profiles: np.ndarray,
                patient_profiles: np.ndarray) -> np.ndarray:
    model_profiles = np.asarray(model_profiles, dtype=float)
    patient_profiles = np.asarray(patient_profiles, dtype=float)
    matrix = np.full((2, 2), np.nan, dtype=float)
    for row in range(2):
        for column in range(2):
            finite = np.isfinite(model_profiles[row]) & np.isfinite(
                patient_profiles[column]
            )
            if np.sum(finite) >= 3:
                matrix[row, column] = float(spearmanr(
                    model_profiles[row, finite],
                    patient_profiles[column, finite],
                ).statistic)
    return matrix


def _map_clusters(labels: np.ndarray, supervised: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=int)
    supervised = np.asarray(supervised, dtype=int)
    if labels.shape != supervised.shape:
        raise ValueError("KMeans and supervised labels do not align")
    contingency = np.zeros((2, 2), dtype=int)
    for cluster, mode in zip(labels, supervised):
        contingency[int(cluster), int(mode)] += 1
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    cluster_to_mode = np.asarray([0, 1] if identity >= swapped else [1, 0], int)
    mapped = cluster_to_mode[labels]
    return mapped, {
        "contingency_cluster_by_numeric_mode_0_1": contingency,
        "cluster_to_numeric_mode": cluster_to_mode,
        "direction_purity": float(max(identity, swapped) / max(1, len(labels))),
        "ami_with_supervised_direction": float(
            adjusted_mutual_info_score(supervised, labels)
        ),
    }


def patient_semantic_audit(
    *, patient: Mapping[str, Any], figure2_field: Mapping[str, Any],
) -> dict[str, Any]:
    field = figure2_field["interictal_field"]
    field_names = [str(value) for value in field["contact_order"]]
    patient_names = np.asarray(patient["contact_names"]).astype(str).tolist()
    if set(field_names) != set(patient_names):
        raise RuntimeError("Figure-2 and patient-training contact sets differ")
    reorder = np.asarray([patient_names.index(name) for name in field_names], int)
    ranks = normalize_event_ranks(patient["all_ranks"])
    labels = np.asarray(patient["all_labels"], dtype=int)
    numeric_profiles = np.asarray([
        _mean_profiles(ranks[labels == mode], np.full(np.sum(labels == mode), mode))[
            SEMANTIC_MODE_ORDER.index(mode)
        ][reorder]
        for mode in (0, 1)
    ])
    field_profiles = np.asarray([field["rank_a"], field["rank_b"]], float)
    matrix = _similarity(numeric_profiles, field_profiles)
    passed = bool(matrix[1, 0] > matrix[1, 1] and matrix[0, 1] > matrix[0, 0])
    if not passed:
        raise RuntimeError("numeric labels no longer map to MTA=1 and MTB=0")
    return {
        "status": "PASS",
        "numeric_label_to_semantic_model_mode": {"0": "MTB", "1": "MTA"},
        "field_columns": ["TA", "TB"],
        "old_numeric_mode_0_1_vs_figure2_field_spearman": matrix,
        "patient_training_counts": {
            "MTA": int(np.sum(labels == 1)),
            "MTB": int(np.sum(labels == 0)),
        },
    }


def acceptance_decision(
    *, network_rows: Sequence[Mapping[str, Any]], pooled_matrix: np.ndarray,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    minimum_events = int(acceptance["minimum_supervised_events_per_mode_per_network"])
    minimum_cluster = int(acceptance["minimum_kmeans_events_per_cluster_per_network"])
    minimum_ami = float(
        acceptance["minimum_per_network_kmeans_ami_with_supervised_direction"]
    )
    support = [
        min(row["supervised_counts_MTA_MTB"]) >= minimum_events
        and min(row["kmeans_counts_MTA_MTB"]) >= minimum_cluster
        for row in network_rows
    ]
    ami = [float(row["kmeans_ami_with_supervised_direction"]) >= minimum_ami
           for row in network_rows]
    matrix = np.asarray(pooled_matrix, dtype=float)
    matrix_signs = bool(
        matrix.shape == (2, 2) and np.isfinite(matrix).all()
        and matrix[0, 0] > 0 and matrix[1, 1] > 0
        and matrix[0, 1] < 0 and matrix[1, 0] < 0
    )
    support_count = int(sum(support))
    ami_count = int(sum(ami))
    support_required = int(acceptance["same_networks_with_both_modes_required"])
    ami_required = int(acceptance["networks_meeting_kmeans_ami_required"])
    clauses = {
        "same_network_dual_mode_and_cluster_support": {
            "passed_networks": support_count,
            "required_networks": support_required,
            "pass": support_count >= support_required,
        },
        "same_network_natural_kmeans_alignment": {
            "passed_networks": ami_count,
            "required_networks": ami_required,
            "minimum_ami": minimum_ami,
            "pass": ami_count >= ami_required,
        },
        "pooled_patient_profile_matrix_signs": {
            "rule": "positive diagonal and negative crossed cells",
            "pass": matrix_signs,
        },
    }
    return {
        "accepted": bool(all(row["pass"] for row in clauses.values())),
        "clauses": clauses,
    }


def _worker_arrays(
    *, config: Mapping[str, Any], robust_config: Mapping[str, Any],
    robust_manifest: Mapping[str, Any], candidate_id: str, seed: int,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    path = artifact_root / robust_config["output_root"] / "workers" / (
        f"{candidate_id}_seed_{seed}.json"
    )
    if not path.is_file():
        raise RuntimeError(f"selected robust worker is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != WORKER_STATUS:
        raise RuntimeError(f"selected robust worker is incomplete: {path}")
    if payload.get("candidate_id") != candidate_id or int(payload.get("seed")) != seed:
        raise RuntimeError("selected robust worker identity changed")
    mechanism = payload.get("mechanism_freeze", {})
    if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
        raise RuntimeError("post-selection worker activated EE, E-to-I, or Z/M")
    simulation = payload.get("simulation", {})
    if float(simulation.get("duration_ms", np.nan)) != 20000.0:
        raise RuntimeError("post-selection worker duration changed")
    if simulation.get("runaway_early_stop_ms") is not None:
        raise RuntimeError("post-selection worker ended in runaway")
    provenance = payload.get("provenance", {})
    if (int(provenance.get("runtime_modules_dirty", 1)) != 0
            or int(provenance.get("runtime_modules_match_expected_commit", 0)) != 1):
        raise RuntimeError("post-selection worker provenance is not clean")
    arrays_record = payload.get("arrays", {})
    npz_path = artifact_root / str(arrays_record.get("path", ""))
    if not npz_path.is_file() or _sha256(npz_path) != arrays_record.get("sha256"):
        raise RuntimeError("post-selection worker arrays changed")
    arrays = exact._load_npz_keys(npz_path, exact.WORKER_ARRAY_KEYS)
    candidates = {
        row["candidate_id"]: row for row in robust_manifest["candidates"]
    }
    expected_coefficients = candidates[candidate_id]["fourier_coordinate"][
        "coefficients_sha256"
    ]
    observed_coordinate = payload.get("fourier_field") or {}
    if observed_coordinate.get("coefficients_sha256") != expected_coefficients:
        raise RuntimeError("selected robust worker field coordinate changed")
    return payload, arrays, {
        "json": {"path": str(path), "sha256": _sha256(path)},
        "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
    }


def _score_network(
    *, arrays: Mapping[str, np.ndarray], seed: int,
    context: Mapping[str, Any], minimum_contacts: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    selection = historical.three_layer_event_selection(
        arrays, minimum_readable_contacts=minimum_contacts,
    )
    primary = np.asarray(selection["contact_primary_indices"], dtype=int)
    ranks = exact._reorder_columns(
        np.asarray(arrays["ranks"], dtype=float)[primary],
        arrays["contact_names"], context["patient"]["contact_names"],
    )
    onsets = exact._reorder_columns(
        np.asarray(arrays["onsets"], dtype=float)[primary],
        arrays["contact_names"], context["patient"]["contact_names"],
    )
    assignment = exact._assign_training_modes(
        ranks, context["frozen_classifier"], context["groups"],
    )
    clean, clean_audit = clean_event_mask(
        readable=np.asarray(
            selection["fig4_kmeans_readable_within_contact"], bool,
        ),
        onsets=onsets, ood=assignment["ood"], groups=context["groups"],
    )
    bundle = {
        "clean": clean,
        "ranks": ranks,
        "labels": assignment["labels"],
        "static": {"contact_names": np.asarray(context["patient"]["contact_names"])},
    }
    selected = np.flatnonzero(clean)
    supervised = np.asarray(assignment["labels"], dtype=int)[selected]
    natural = None
    if len(selected) >= 6:
        try:
            natural = fig4._canonical_rank_kmeans(
                bundle, min_shared_contacts=minimum_contacts,
            )
        except (RuntimeError, ValueError):
            natural = None
    if natural is not None:
        natural_selected = np.asarray(natural["clean_global_index"], dtype=int)
        if not np.array_equal(natural_selected, selected):
            raise RuntimeError("natural KMeans changed the formal clean event set")
        kmeans = np.asarray(natural["labels"], dtype=int)
        mapped, mapping = _map_clusters(kmeans, supervised)
        kmeans_status = "OK"
    else:
        mapped = np.full(len(selected), -1, dtype=int)
        mapping = {
            "contingency_cluster_by_numeric_mode_0_1": np.zeros((2, 2), int),
            "cluster_to_numeric_mode": np.asarray([-1, -1], int),
            "direction_purity": 0.0,
            "ami_with_supervised_direction": -1.0,
        }
        kmeans_status = "NOT_EVALUABLE_INSUFFICIENT_OR_DEGENERATE_EVENTS"
    selected_ranks = ranks[selected]
    supervised_profiles = _mean_profiles(selected_ranks, supervised)
    kmeans_profiles = _mean_profiles(selected_ranks, mapped)
    patient_profiles = _mean_profiles(
        context["patient"]["all_ranks"], context["patient"]["all_labels"],
    )
    supervised_counts = [int(np.sum(supervised == mode)) for mode in SEMANTIC_MODE_ORDER]
    kmeans_counts = [int(np.sum(mapped == mode)) for mode in SEMANTIC_MODE_ORDER]
    support = all_event_shaft_participation(onsets[selected], context["groups"])
    row = {
        "network_seed": int(seed),
        "event_selection": {**selection, **clean_audit},
        "formal_clean_global_event_indices": primary[selected],
        "kmeans_status": kmeans_status,
        "supervised_counts_MTA_MTB": supervised_counts,
        "kmeans_counts_MTA_MTB": kmeans_counts,
        "kmeans_ami_with_supervised_direction": mapping[
            "ami_with_supervised_direction"
        ],
        "kmeans_direction_purity": mapping["direction_purity"],
        "kmeans_contingency_cluster_by_numeric_mode_0_1": mapping[
            "contingency_cluster_by_numeric_mode_0_1"
        ],
        "kmeans_cluster_to_numeric_mode": mapping["cluster_to_numeric_mode"],
        "kmeans_stability_ami_median": (
            None if natural is None else natural["stability_ami_median"]
        ),
        "kmeans_silhouette_median": (
            None if natural is None else natural["silhouette_median"]
        ),
        "within_cluster_tau_mean": (
            None if natural is None else natural["within_cluster_tau_mean"]
        ),
        "shaft_support": support,
        "supervised_patient_matrix_MTA_MTB_by_TA_TB": _similarity(
            supervised_profiles, patient_profiles,
        ),
        "kmeans_patient_matrix_MTA_MTB_by_TA_TB": _similarity(
            kmeans_profiles, patient_profiles,
        ),
    }
    reusable = {
        "ranks": selected_ranks,
        "onsets": onsets[selected],
        "supervised": supervised,
        "kmeans_mapped": mapped,
        "supervised_profiles": supervised_profiles,
        "kmeans_profiles": kmeans_profiles,
    }
    return row, reusable


def audit(
    *, config_path: Path = DEFAULT_CONFIG,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev15_node_postselection_v1":
        raise RuntimeError("post-selection config schema changed")
    if config["boundaries"].get("patient_heldout_used") is not False:
        raise RuntimeError("post-selection config allows patient held-out access")
    if config["boundaries"].get("EE_EtoI_ZM") != "off":
        raise RuntimeError("post-selection config allows another mechanism")
    paths = {
        key: _resolve_record(record, artifact_root=artifact_root)
        for key, record in config["inputs"].items()
    }
    robust_config = json.loads(paths["robust_config"].read_text())
    robust_manifest = json.loads(paths["robust_manifest"].read_text())
    robust_aggregate = json.loads(paths["robust_aggregate"].read_text())
    if robust_config.get("pathways") != {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off",
        "Z_M": "off",
    }:
        raise RuntimeError("robust candidate config activated another mechanism")
    if [int(seed) for seed in robust_config["search"]["active_network_seeds"]] != [
        int(seed) for seed in config["network_seeds"]
    ]:
        raise RuntimeError("post-selection network seeds differ from robust replication")
    if robust_aggregate.get("status") != "COMPLETE":
        raise RuntimeError("robust aggregate is incomplete")
    if not robust_aggregate.get("inventory", {}).get("complete_cartesian_product"):
        raise RuntimeError("robust aggregate Cartesian product is incomplete")
    candidate_id = str(config["selected_candidate"]["candidate_id"])
    if robust_aggregate.get("best_usable_anchor") != candidate_id:
        raise RuntimeError("post-selection candidate differs from robust ranking")
    if candidate_id not in robust_aggregate.get("usable_two_mode_anchor_ids", []):
        raise RuntimeError("post-selection candidate is not a usable robust anchor")
    ranking = robust_aggregate.get("ranking_contract", {})
    if any(ranking.get(key) is not False for key in (
        "natural_kmeans_used", "patient_heldout_used", "ictal_data_used",
        "figure_used",
    )) or ranking.get("EE_EtoI_ZM") != "off":
        raise RuntimeError("robust candidate ranking crossed a forbidden boundary")
    if robust_manifest.get("config_sha256") != _sha256(paths["robust_config"]):
        raise RuntimeError("robust manifest/config identity changed")
    candidate_rows = {
        row["candidate_id"]: row for row in robust_manifest["candidates"]
    }
    if candidate_id not in candidate_rows or not candidate_rows[candidate_id].get(
        "selection_eligible"
    ):
        raise RuntimeError("selected robust candidate is not manifest-selectable")
    if candidate_rows[candidate_id]["fourier_coordinate"][
        "coefficients_sha256"
    ] != config["selected_candidate"]["field_coefficients_sha256"]:
        raise RuntimeError("post-selection selected-field hash changed")
    j14_config = json.loads(paths["j14_config"].read_text())
    context = historical._patient_context(j14_config, artifact_root)
    semantic = patient_semantic_audit(
        patient=context["patient"],
        figure2_field=json.loads(paths["figure2_template_field"].read_text()),
    )
    network_rows, reusable, worker_inputs = [], [], []
    for seed in config["network_seeds"]:
        _, arrays, inputs = _worker_arrays(
            config=config, robust_config=robust_config,
            robust_manifest=robust_manifest, candidate_id=candidate_id,
            seed=int(seed), artifact_root=artifact_root,
        )
        row, data = _score_network(
            arrays=arrays, seed=int(seed), context=context,
            minimum_contacts=int(config["event_contract"][
                "minimum_readable_contacts"
            ]),
        )
        network_rows.append(row)
        reusable.append(data)
        worker_inputs.append({"network_seed": int(seed), **inputs})
    pooled_ranks = np.concatenate([row["ranks"] for row in reusable], axis=0)
    pooled_supervised = np.concatenate(
        [row["supervised"] for row in reusable], axis=0,
    )
    pooled_bundle = {
        "clean": np.ones(len(pooled_ranks), dtype=bool),
        "ranks": pooled_ranks,
        "labels": pooled_supervised,
        "static": {"contact_names": np.asarray(context["patient"]["contact_names"])},
    }
    pooled_natural = None
    if len(pooled_ranks) >= 6:
        try:
            pooled_natural = fig4._canonical_rank_kmeans(
                pooled_bundle,
                min_shared_contacts=int(
                    config["event_contract"]["minimum_readable_contacts"]
                ),
            )
        except (RuntimeError, ValueError):
            pooled_natural = None
    if pooled_natural is not None:
        pooled_selected = np.asarray(pooled_natural["clean_global_index"], int)
        pooled_kmeans = np.asarray(pooled_natural["labels"], int)
        pooled_direction = pooled_supervised[pooled_selected]
        pooled_mapped, pooled_mapping = _map_clusters(
            pooled_kmeans, pooled_direction,
        )
        pooled_kmeans_status = "OK"
    else:
        pooled_selected = np.arange(len(pooled_ranks), dtype=int)
        pooled_direction = pooled_supervised.copy()
        pooled_mapped = np.full(len(pooled_ranks), -1, dtype=int)
        pooled_mapping = {
            "direction_purity": 0.0,
            "ami_with_supervised_direction": -1.0,
        }
        pooled_kmeans_status = "NOT_EVALUABLE_INSUFFICIENT_OR_DEGENERATE_EVENTS"
    patient_profiles = _mean_profiles(
        context["patient"]["all_ranks"], context["patient"]["all_labels"],
    )
    pooled_supervised_profiles = _mean_profiles(
        pooled_ranks[pooled_selected], pooled_direction,
    )
    pooled_kmeans_profiles = _mean_profiles(
        pooled_ranks[pooled_selected], pooled_mapped,
    )
    pooled_matrix = _similarity(pooled_kmeans_profiles, patient_profiles)
    decision = acceptance_decision(
        network_rows=network_rows, pooled_matrix=pooled_matrix,
        acceptance=config["acceptance"],
    )
    provenance = _provenance()
    status = (
        "INVALID_PROVENANCE" if not provenance["formal_ready"]
        else "NODE_POSTSELECTION_ACCEPTED" if decision["accepted"]
        else "NODE_POSTSELECTION_REJECTED"
    )
    output_root = artifact_root / config["output_root"] / "analysis"
    output_json = output_root / "node_postselection_audit.json"
    output_csv = output_root / "node_postselection_network_summary.csv"
    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": status,
        "candidate_id": candidate_id,
        "provenance": provenance,
        "semantic_mode_audit": semantic,
        "network_results": network_rows,
        "pooled": {
            "kmeans_status": pooled_kmeans_status,
            "formal_clean_events": int(len(pooled_selected)),
            "supervised_counts_MTA_MTB": [
                int(np.sum(pooled_direction == mode)) for mode in SEMANTIC_MODE_ORDER
            ],
            "kmeans_counts_MTA_MTB": [
                int(np.sum(pooled_mapped == mode)) for mode in SEMANTIC_MODE_ORDER
            ],
            "kmeans_ami_with_supervised_direction": pooled_mapping[
                "ami_with_supervised_direction"
            ],
            "kmeans_direction_purity": pooled_mapping["direction_purity"],
            "kmeans_stability_ami_median": (
                None if pooled_natural is None
                else pooled_natural["stability_ami_median"]
            ),
            "within_cluster_tau_mean": (
                None if pooled_natural is None
                else pooled_natural["within_cluster_tau_mean"]
            ),
            "supervised_patient_matrix_MTA_MTB_by_TA_TB": _similarity(
                pooled_supervised_profiles, patient_profiles,
            ),
            "kmeans_patient_matrix_MTA_MTB_by_TA_TB": pooled_matrix,
            "equal_network_supervised_patient_matrix_MTA_MTB_by_TA_TB": _similarity(
                np.nanmean(np.asarray([
                    row["supervised_profiles"] for row in reusable
                ]), axis=0), patient_profiles,
            ),
            "equal_network_kmeans_patient_matrix_MTA_MTB_by_TA_TB": _similarity(
                np.nanmean(np.asarray([
                    row["kmeans_profiles"] for row in reusable
                ]), axis=0), patient_profiles,
            ),
        },
        "acceptance": decision,
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "hashed_inputs": config["inputs"],
            "workers": worker_inputs,
        },
        "boundaries": {
            **config["boundaries"],
            "patient_heldout_loaded": False,
            "field_reranking_performed": False,
            "SNN_simulation_run": False,
        },
        "claim_boundary": (
            "Training-semantic post-selection only. Acceptance permits the already "
            "selected robust Node field to advance to Fig.4 inspection and one-time "
            "held-out evaluation; it is not patient-blind confirmation and cannot "
            "activate EE, E-to-I, or Z/M."
        ),
        "outputs": {"json": str(output_json), "network_csv": str(output_csv)},
    }
    _atomic_json(output_json, payload)
    _atomic_csv(output_csv, [{
        "network_seed": row["network_seed"],
        "formal_clean_events": row["event_selection"]["n_formal_clean"],
        "supervised_MTA": row["supervised_counts_MTA_MTB"][0],
        "supervised_MTB": row["supervised_counts_MTA_MTB"][1],
        "kmeans_MTA": row["kmeans_counts_MTA_MTB"][0],
        "kmeans_MTB": row["kmeans_counts_MTA_MTB"][1],
        "kmeans_ami": row["kmeans_ami_with_supervised_direction"],
        "kmeans_purity": row["kmeans_direction_purity"],
    } for row in network_rows])
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = audit(
        config_path=args.config.resolve(), artifact_root=args.artifact_root.resolve(),
    )
    print(json.dumps({
        "status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "accepted": payload["acceptance"]["accepted"],
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
