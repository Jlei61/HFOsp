#!/usr/bin/env python3
"""Zero-simulation rescore of the frozen rev12 static-Node libraries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import rescore_topic4_rev13_exact_off_static_node as exact  # noqa: E402
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    contact_split_folds,
    crossfit_patient_readout,
    natural_kmeans,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
    soft_dual_mode_objective,
)
from src.topic4_rev14_static_node_objective import rev14_objective  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import (  # noqa: E402
    all_event_shaft_participation,
)


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev14_static_node_historical_rescore.json"
HISTORICAL_ARRAY_KEYS = (
    "contact_names",
    "onsets",
    "ranks",
    "event_t_on_ms",
    "event_trigger_t_on_ms",
    "event_t_off_ms",
    "event_returned",
    "event_fragment_count",
    "event_directed_root_id",
    "event_root_count",
    "source_onset_maps_ms",
    "source_onset_evaluable",
    "source_bin_mm",
    "positions_E",
    "delta_vtheta",
)
_PROCESS_CONTEXT: Mapping[str, Any] | None = None


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    exact._atomic_json(path, payload)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]],
                columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in columns})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _runtime_paths(config_path: Path) -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        config_path.resolve(),
        ROOT / "scripts/rescore_topic4_rev13_exact_off_static_node.py",
        ROOT / "scripts/aggregate_topic4_rev13_node_zero_sum_canary.py",
        ROOT / "src/topic4_node_dualmode.py",
        ROOT / "src/topic4_rev14_static_node_objective.py",
        ROOT / "src/topic4_d6_natural_kmeans.py",
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/topic4_shaft_aware_direction.py",
    )


def runtime_provenance(config_path: Path) -> dict[str, Any]:
    paths = _runtime_paths(config_path)
    relative, repo_relative = [], []
    for path in paths:
        try:
            value = str(path.relative_to(ROOT))
            repo_relative.append(value)
        except ValueError:
            value = str(path)
        relative.append(value)
    dirty = _git("status", "--porcelain", "--", *repo_relative)
    return {
        "git_commit_at_analysis": _git("rev-parse", "HEAD"),
        "tracked_runtime_paths": relative,
        "runtime_paths_dirty": bool(dirty),
        "runtime_dirty_porcelain": dirty.splitlines(),
        "runtime_path_sha256": {
            str(path): exact._sha256(path) for path in paths
        },
    }


def _stable_candidate_offset(candidate_id: str) -> int:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1_000_000


def _inventory_paths(output_root: Path) -> tuple[Path, Path]:
    return (
        output_root / "historical_static_node_inventory.json",
        output_root / "historical_static_node_inventory.csv",
    )


def _formal_paths(output_root: Path) -> tuple[Path, Path]:
    return (
        output_root / "historical_static_node_rescore.json",
        output_root / "historical_static_node_rescore.csv",
    )


def _read_manifest(stage: Mapping[str, Any], artifact_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _resolve(artifact_root, Path(stage["root"]) / "candidate_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if exact._sha256(manifest_path) != str(stage["manifest_sha256"]):
        raise RuntimeError(f"manifest hash changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != stage["manifest_status"]:
        raise RuntimeError(f"manifest status changed: {manifest_path}")
    if len(manifest.get("candidates", [])) != int(stage["candidate_count"]):
        raise RuntimeError(f"candidate count changed: {manifest_path}")
    return manifest_path, manifest


def _npz_key_inventory(path: Path) -> tuple[list[str], list[str]]:
    with np.load(path, allow_pickle=False) as loaded:
        available = sorted(loaded.files)
    missing = sorted(set(HISTORICAL_ARRAY_KEYS).difference(available))
    return available, missing


def _effective_eligibility(stage: Mapping[str, Any], original: bool) -> bool:
    policy = str(stage["selection_policy"])
    if policy == "diagnostic_only_never_selectable":
        return False
    if policy != "preserve_manifest_selection_eligibility":
        raise RuntimeError(f"unknown selection policy: {policy}")
    return bool(original)


def build_inventory(config_path: Path, artifact_root: Path) -> dict[str, Any]:
    """Audit all manifests and frozen workers without loading event arrays."""
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    invariants = config["invariants"]
    expected_status = str(invariants["expected_worker_status"])
    expected_duration = float(invariants["expected_duration_ms"])
    records: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    for stage_order, stage in enumerate(config["inputs"]["stages"]):
        manifest_path, manifest = _read_manifest(stage, artifact_root)
        candidate_rows = list(manifest["candidates"])
        candidate_by_id = {
            str(row["candidate_id"]): row for row in candidate_rows
        }
        candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(f"duplicate candidate ID in {stage['stage_id']}")
        original_eligibility = {
            str(row["candidate_id"]): bool(row.get("selection_eligible", False))
            for row in candidate_rows
        }
        roles = {
            str(row["candidate_id"]): str(row.get("role", ""))
            for row in candidate_rows
        }
        worker_root = _resolve(artifact_root, Path(stage["root"]) / "workers")
        json_paths = sorted(worker_root.glob("*.json"))
        if len(json_paths) != int(stage["run_count"]):
            raise RuntimeError(f"worker JSON count changed: {stage['stage_id']}")
        by_key: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
        for json_path in json_paths:
            payload = json.loads(json_path.read_text())
            key = (str(payload.get("candidate_id")), int(payload.get("seed")))
            if key in by_key:
                raise RuntimeError(f"duplicate worker identity: {key}")
            by_key[key] = (json_path, payload)
        expected_keys = {
            (candidate_id, int(seed))
            for candidate_id in candidate_ids for seed in stage["seeds"]
        }
        if set(by_key) != expected_keys:
            missing = sorted(expected_keys.difference(by_key))
            extra = sorted(set(by_key).difference(expected_keys))
            raise RuntimeError(
                f"worker Cartesian product changed in {stage['stage_id']}: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )

        for candidate_order, candidate_id in enumerate(candidate_ids):
            for seed in stage["seeds"]:
                json_path, payload = by_key[(candidate_id, int(seed))]
                if payload.get("status") != expected_status:
                    raise RuntimeError(f"worker incomplete: {json_path}")
                simulation = payload.get("simulation", {})
                if float(simulation.get("duration_ms", np.nan)) != expected_duration:
                    raise RuntimeError(f"worker duration changed: {json_path}")
                if simulation.get("runaway_early_stop_ms") is not None:
                    raise RuntimeError(f"historical worker ended in runaway: {json_path}")
                mechanism = payload.get("mechanism_freeze", {})
                if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
                    raise RuntimeError(f"open mechanism in historical worker: {json_path}")
                if int(mechanism.get("edge_coefficients_all_zero", 0)) != 1:
                    raise RuntimeError(f"nonzero edge coefficient in: {json_path}")
                candidate = candidate_by_id[candidate_id]
                expected_field = str(candidate["node_field"]["field_sha256"])
                if str(payload.get("field_sha256")) != expected_field:
                    raise RuntimeError(f"worker field identity changed: {json_path}")
                expected_mapping = candidate.get("node_mapping", {}).get("mapping_sha256")
                observed_mapping = payload.get("node_mapping", {}).get("mapping_sha256")
                if expected_mapping != observed_mapping:
                    raise RuntimeError(f"worker node mapping changed: {json_path}")
                provenance = payload.get("provenance", {})
                if (int(provenance.get("runtime_modules_dirty", 1)) != 0
                        or int(provenance.get("runtime_modules_match_expected_commit", 0)) != 1
                        or provenance.get("git_commit") != provenance.get("expected_git_commit")):
                    raise RuntimeError(f"worker runtime provenance is not clean: {json_path}")
                arrays_record = payload.get("arrays", {})
                npz_path = _resolve(artifact_root, arrays_record.get("path", ""))
                expected_sibling = json_path.with_suffix(".npz").resolve()
                if npz_path != expected_sibling:
                    raise RuntimeError(f"worker array path is not its sibling: {json_path}")
                if not npz_path.exists():
                    raise FileNotFoundError(npz_path)
                npz_sha = exact._sha256(npz_path)
                if npz_sha != str(arrays_record.get("sha256")):
                    raise RuntimeError(f"worker NPZ hash changed: {npz_path}")
                _, missing_arrays = _npz_key_inventory(npz_path)
                if missing_arrays:
                    raise RuntimeError(
                        f"worker NPZ lacks required arrays: {npz_path}: {missing_arrays}"
                    )
                original = original_eligibility[candidate_id]
                records.append({
                    "stage_order": stage_order,
                    "candidate_order": candidate_order,
                    "stage_id": str(stage["stage_id"]),
                    "seed_pool_id": str(stage["seed_pool_id"]),
                    "candidate_id": candidate_id,
                    "candidate_role": roles[candidate_id],
                    "seed": int(seed),
                    "original_selection_eligible": original,
                    "effective_selection_eligible": _effective_eligibility(stage, original),
                    "diagnostic_only": not _effective_eligibility(stage, original),
                    "worker_json": str(json_path),
                    "worker_json_sha256": exact._sha256(json_path),
                    "worker_npz": str(npz_path),
                    "worker_npz_sha256": npz_sha,
                    "field_sha256": payload.get("field_sha256"),
                    "node_mapping_sha256": observed_mapping,
                    "worker_git_commit": provenance.get("git_commit"),
                    "worker_runtime_clean": True,
                    "duration_ms": float(simulation["duration_ms"]),
                    "mechanisms_off": True,
                    "required_arrays_present": True,
                })
        stages.append({
            "stage_id": str(stage["stage_id"]),
            "seed_pool_id": str(stage["seed_pool_id"]),
            "manifest": str(manifest_path),
            "manifest_sha256": exact._sha256(manifest_path),
            "manifest_status": manifest["status"],
            "candidate_count": len(candidate_ids),
            "run_count": len(expected_keys),
            "seeds": [int(seed) for seed in stage["seeds"]],
            "anchor_candidate_id": str(stage["anchor_candidate_id"]),
            "anchor_semantics": str(stage["anchor_semantics"]),
            "selection_policy": str(stage["selection_policy"]),
            "original_selection_eligible_count": int(sum(original_eligibility.values())),
            "effective_selection_eligible_count": int(sum(
                _effective_eligibility(stage, value)
                for value in original_eligibility.values()
            )),
        })

    expected_candidates = int(invariants["expected_total_candidates"])
    expected_runs = int(invariants["expected_total_runs"])
    total_candidates = sum(row["candidate_count"] for row in stages)
    if total_candidates != expected_candidates or len(records) != expected_runs:
        raise RuntimeError(
            f"historical inventory changed: candidates={total_candidates}, runs={len(records)}"
        )
    if any(row["stage_id"] == "stage_ak" and row["effective_selection_eligible"]
           for row in records):
        raise RuntimeError("Stage-AK was promoted despite diagnostic-only contract")
    return {
        "schema_id": "topic4_rev14_static_node_historical_inventory_v1",
        "status": "REV14_HISTORICAL_STATIC_NODE_DRY_RUN_COMPLETE",
        "config": {"path": str(config_path), "sha256": exact._sha256(config_path)},
        "counts": {
            "stages": len(stages),
            "candidates": total_candidates,
            "runs": len(records),
        },
        "stages": stages,
        "runs": records,
        "invariants": {
            "complete_candidate_seed_cartesian_products": True,
            "worker_npz_hashes_verified": True,
            "worker_field_and_mapping_identity_verified": True,
            "worker_runtime_provenance_clean": True,
            "required_arrays_present": True,
            "all_runs_20s_without_runaway": True,
            "all_runs_node_only_ee_etoi_zm_off": True,
            "manifest_selection_eligibility_preserved": True,
            "stage_ak_diagnostic_only": True,
            "cross_seed_pool_raw_ranking_performed": False,
            "snn_simulation_run": False,
        },
        "provenance": runtime_provenance(config_path),
        "claim_boundary": config["claim_boundary"],
    }


def write_inventory(inventory: Mapping[str, Any], output_root: Path) -> tuple[Path, Path]:
    json_path, csv_path = _inventory_paths(output_root)
    _atomic_json(json_path, inventory)
    columns = (
        "stage_id", "seed_pool_id", "candidate_id", "candidate_role", "seed",
        "original_selection_eligible", "effective_selection_eligible",
        "diagnostic_only", "duration_ms", "mechanisms_off",
        "required_arrays_present", "field_sha256", "worker_json",
        "worker_json_sha256", "worker_npz", "worker_npz_sha256",
    )
    _atomic_csv(csv_path, inventory["runs"], columns)
    return json_path, csv_path


def _empty_score(zero_event_objective: float) -> dict[str, float]:
    value = float(zero_event_objective)
    return {
        "objective": value,
        "weakest_mode_lse": value,
        "occupancy_js": float(np.log(2.0)),
        "ambiguity": 1.0,
        "contrast_loss": 1.0,
        "contrast_alignment": 0.0,
        "mode_0_mean": value,
        "mode_1_mean": value,
    }


def _not_evaluable(status: str, n_events: int = 0) -> dict[str, Any]:
    return {"status": status, "n_events": int(n_events)}


def temporal_overlap_connected_audit(
        event_t_on_ms: np.ndarray, event_t_off_ms: np.ndarray,
        original_event_indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Exclude every member of each strict interval-overlap component."""
    t_on = np.asarray(event_t_on_ms, dtype=np.float64)
    t_off = np.asarray(event_t_off_ms, dtype=np.float64)
    original = np.asarray(original_event_indices, dtype=np.int64)
    if t_on.ndim != 1 or t_off.shape != t_on.shape or original.shape != t_on.shape:
        raise ValueError("temporal overlap arrays must align")
    if (not np.all(np.isfinite(t_on)) or not np.all(np.isfinite(t_off))
            or np.any(t_off < t_on)):
        raise ValueError("temporal overlap audit requires finite ordered intervals")
    if np.any(np.diff(t_on) < 0.0):
        raise ValueError("temporal overlap audit requires chronological families")

    pairs: list[dict[str, Any]] = []
    for left in range(len(t_on)):
        right = left + 1
        while right < len(t_on) and t_on[right] < t_off[left]:
            pairs.append({
                "within_returned_interval_indices": [int(left), int(right)],
                "original_event_indices": [int(original[left]), int(original[right])],
            })
            right += 1
    components: list[list[int]] = []
    if len(t_on):
        current = [0]
        current_right = float(t_off[0])
        for index in range(1, len(t_on)):
            if float(t_on[index]) < current_right:
                current.append(index)
                current_right = max(current_right, float(t_off[index]))
            else:
                components.append(current)
                current = [index]
                current_right = float(t_off[index])
        components.append(current)
    excluded = np.zeros(len(t_on), dtype=bool)
    overlap_components = []
    for component in components:
        if len(component) < 2:
            continue
        excluded[component] = True
        overlap_components.append({
            "within_returned_interval_indices": [int(value) for value in component],
            "original_event_indices": [int(original[value]) for value in component],
            "n_families": int(len(component)),
        })
    return ~excluded, {
        "formal_action": "EXCLUDE_ALL_MEMBERS_OF_OVERLAP_CONNECTED_EPISODES",
        "strict_overlap_definition": "later_t_on_ms < earlier_component_max_t_off_ms",
        "source_map_or_displacement_used": False,
        "n_input_returned_valid_interval_families": int(len(t_on)),
        "n_overlap_pairs": int(len(pairs)),
        "n_overlap_components": int(len(overlap_components)),
        "n_excluded_families": int(np.sum(excluded)),
        "n_contact_primary_families": int(np.sum(~excluded)),
        "overlap_pairs": pairs,
        "overlap_components": overlap_components,
    }


def three_layer_event_selection(arrays: Mapping[str, np.ndarray], *,
                                minimum_readable_contacts: int) -> dict[str, Any]:
    """Build contact-loss, topology, and Fig4 masks without conflating them."""
    returned = np.asarray(arrays["event_returned"], dtype=bool)
    n_events = len(returned)
    aligned = (
        "source_onset_evaluable", "event_t_on_ms", "event_trigger_t_on_ms",
        "event_t_off_ms", "event_fragment_count", "event_directed_root_id",
        "event_root_count",
    )
    if any(np.asarray(arrays[key]).shape != (n_events,) for key in aligned):
        raise RuntimeError("worker event arrays do not align")
    for key in ("onsets", "ranks"):
        if np.asarray(arrays[key]).shape[0] != n_events:
            raise RuntimeError("worker contact arrays do not align")
    source_maps = np.asarray(arrays["source_onset_maps_ms"], dtype=np.float64)
    if source_maps.shape[0] != n_events:
        raise RuntimeError("worker source maps do not align")

    t_on_all = np.asarray(arrays["event_t_on_ms"], dtype=np.float64)
    t_off_all = np.asarray(arrays["event_t_off_ms"], dtype=np.float64)
    valid_interval = np.isfinite(t_on_all) & np.isfinite(t_off_all) & (t_off_all >= t_on_all)
    order = np.argsort(t_on_all, kind="stable")
    returned_valid = order[(returned & valid_interval)[order]]
    isolated_within, overlap = temporal_overlap_connected_audit(
        t_on_all[returned_valid], t_off_all[returned_valid], returned_valid,
    )
    contact_primary = returned_valid[isolated_within]

    axis = exact.substrate_pca_axis(arrays["positions_E"], arrays["delta_vtheta"])
    displacements = exact.event_axis_displacements(
        source_maps[contact_primary], axis_unit=axis,
        bin_mm=float(np.asarray(arrays["source_bin_mm"]).item()),
    )
    topology_within = (
        np.asarray(arrays["source_onset_evaluable"], dtype=bool)[contact_primary]
        & np.isfinite(displacements)
    )
    finite_contacts = np.sum(
        np.isfinite(np.asarray(arrays["ranks"], dtype=np.float64)[contact_primary]),
        axis=1,
    )
    returned_finite_contacts = np.sum(
        np.isfinite(np.asarray(arrays["ranks"], dtype=np.float64)[returned]), axis=1,
    )
    fig4_within = finite_contacts >= int(minimum_readable_contacts)
    return {
        "contact_primary_indices": contact_primary,
        "topology_primary_within_contact": topology_within,
        "topology_primary_indices": contact_primary[topology_within],
        "topology_primary_displacements_mm": displacements[topology_within],
        "fig4_kmeans_readable_within_contact": fig4_within,
        "fig4_kmeans_readable_indices": contact_primary[fig4_within],
        "substrate_axis_xy": axis,
        "overlap_audit": overlap,
        "n_total": int(n_events),
        "n_returned": int(np.sum(returned)),
        "n_returned_invalid_interval": int(np.sum(returned & ~valid_interval)),
        "n_returned_valid_interval": int(len(returned_valid)),
        "n_contact_primary": int(len(contact_primary)),
        "n_topology_primary": int(np.sum(topology_within)),
        "n_fig4_kmeans_readable": int(np.sum(fig4_within)),
        "n_contact_primary_source_not_evaluable": int(np.sum(
            ~np.asarray(arrays["source_onset_evaluable"], dtype=bool)[contact_primary]
        )),
        "n_contact_primary_lt3_finite_contacts": int(np.sum(~fig4_within)),
        "n_returned_less_than_minimum_contacts": int(np.sum(
            returned_finite_contacts < int(minimum_readable_contacts)
        )),
        "minimum_fig4_readable_contacts": int(minimum_readable_contacts),
    }


def _score_run(record: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    arrays = exact._load_npz_keys(Path(record["worker_npz"]), HISTORICAL_ARRAY_KEYS)
    selection = three_layer_event_selection(
        arrays,
        minimum_readable_contacts=int(context["minimum_readable_contacts"]),
    )
    primary = np.asarray(selection["contact_primary_indices"], dtype=np.int64)
    ranks = exact._reorder_columns(
        np.asarray(arrays["ranks"], dtype=np.float64)[primary],
        arrays["contact_names"], context["patient"]["contact_names"],
    )
    onsets = exact._reorder_columns(
        np.asarray(arrays["onsets"], dtype=np.float64)[primary],
        arrays["contact_names"], context["patient"]["contact_names"],
    )
    readable = np.asarray(
        selection["fig4_kmeans_readable_within_contact"], dtype=bool,
    )
    objective = context["objective"]
    if len(primary):
        assignment = exact._assign_training_modes(
            ranks, context["frozen_classifier"], context["groups"],
        )
        mode_evidence = readable & ~np.asarray(assignment["ood"], dtype=bool)
        score = soft_dual_mode_objective(
            ranks, assignment["probability_B"],
            context["patient"]["reference_ranks"],
            context["patient"]["reference_labels"],
            context["patient"]["contact_names"],
            projections=context["projections"], calibration=context["calibration"],
            tau=float(objective["tau"]),
            occupancy_weight=float(objective["occupancy_weight"]),
            ambiguity_weight=float(objective["ambiguity_weight"]),
            contrast_weight=float(objective["contrast_weight"]),
        )
        score_summary = exact._score_summary(score)
        assignment_summary = {
            "primary_mode_definition": "frozen_old_A_B_direction_labels",
            "raw_old_A_B_mode_counts": np.bincount(
                assignment["labels"], minlength=2,
            ),
            "raw_frozen_classifier_probability_B_mean": float(
                np.mean(assignment["probability_B"])
            ),
            "shaft_aware_k2_mapping_applied": False,
            "ood_count": int(np.sum(assignment["ood"])),
            "ood_fraction": float(np.mean(assignment["ood"])),
            "ood_is_diagnostic_only_and_not_filtered": True,
            "mode_evidence_count": int(np.sum(mode_evidence)),
            "mode_evidence_definition": "fig4_readable AND frozen-classifier in-support",
        }
        support = all_event_shaft_participation(onsets, context["groups"])
        candidate_offset = _stable_candidate_offset(str(record["candidate_id"]))
        natural = natural_kmeans(
            ranks, assignment["labels"],
            random_state=(int(context["natural_kmeans_seed"])
                          + int(record["seed"]) + candidate_offset),
        )
        if natural.get("status") == "OK":
            if not np.array_equal(np.asarray(natural["valid_event_mask"], bool), readable):
                raise RuntimeError("Fig4 readable mask differs from natural KMeans input")
            natural_summary = exact._strip_natural(natural)
        else:
            natural_summary = exact._jsonable(natural)
        if np.sum(readable) >= 2:
            crossfit = crossfit_patient_readout(
                ranks[readable], context["patient"]["all_ranks"],
                context["patient"]["all_labels"], context["folds"],
            )
        else:
            crossfit = _not_evaluable("INSUFFICIENT_FIG4_READABLE_EVENTS", int(np.sum(readable)))
        labels = np.asarray(assignment["labels"], dtype=np.int8)
    else:
        score_summary = _empty_score(float(objective["zero_event_objective"]))
        assignment_summary = {
            "primary_mode_definition": "frozen_old_A_B_direction_labels",
            "raw_old_A_B_mode_counts": np.asarray([0, 0]),
            "raw_frozen_classifier_probability_B_mean": None,
            "shaft_aware_k2_mapping_applied": False,
            "ood_count": 0,
            "ood_fraction": None,
            "ood_is_diagnostic_only_and_not_filtered": True,
        }
        support = _not_evaluable("NO_CONTACT_PRIMARY_CAUSAL_FAMILY")
        natural_summary = _not_evaluable("NO_CONTACT_PRIMARY_CAUSAL_FAMILY")
        crossfit = _not_evaluable("NO_CONTACT_PRIMARY_CAUSAL_FAMILY")
        labels = np.empty(0, dtype=np.int8)
        assignment = {
            "probability_B": np.empty(0, dtype=np.float64),
            "labels": labels,
        }
        mode_evidence = np.empty(0, dtype=bool)

    formal = context["formal_objective"]
    j14 = rev14_objective(
        ranks, np.asarray(assignment["probability_B"], dtype=np.float64),
        context["patient"]["all_ranks"], context["patient"]["all_labels"],
        context["patient"]["all_blocks"], context["patient"]["contact_names"],
        projections=context["projections"], calibration=context["calibration"],
        returned_families=int(selection["n_returned"]),
        contact_evaluable_families=int(selection["n_returned_valid_interval"]),
        overlap_excluded_families=int(
            selection["overlap_audit"]["n_excluded_families"]
        ),
        less_than_three_contact_families=int(
            selection["n_returned_less_than_minimum_contacts"]
        ),
        mode_evidence_mask=mode_evidence,
        sample_size=int(formal["sample_size_per_side"]),
        draws=int(formal["draws_per_network"]),
        seed=int(formal["seed"]) + int(record["seed"]),
        tau=float(formal["tau"]),
    )

    return {
        "stage_order": int(record["stage_order"]),
        "candidate_order": int(record["candidate_order"]),
        "stage_id": str(record["stage_id"]),
        "seed_pool_id": str(record["seed_pool_id"]),
        "candidate_id": str(record["candidate_id"]),
        "seed": int(record["seed"]),
        "original_selection_eligible": bool(record["original_selection_eligible"]),
        "effective_selection_eligible": bool(record["effective_selection_eligible"]),
        "diagnostic_only": bool(record["diagnostic_only"]),
        "field_sha256": record.get("field_sha256"),
        "event_selection": {
            key: selection[key] for key in (
                "n_total", "n_returned", "n_returned_invalid_interval",
                "n_returned_valid_interval", "n_contact_primary",
                "n_topology_primary", "n_fig4_kmeans_readable",
                "n_contact_primary_source_not_evaluable",
                "n_contact_primary_lt3_finite_contacts",
                "n_returned_less_than_minimum_contacts",
                "minimum_fig4_readable_contacts",
            )
        },
        "contact_primary_original_event_indices": primary,
        "overlap_connected_episode_audit": selection["overlap_audit"],
        "j14_v1": j14,
        "j14_v1_summary": exact._j14_summary(j14),
        "phase0_legacy_soft_score_summary": score_summary,
        "patient_training_assignment": assignment_summary,
        "shaft_participation_all_contact_primary_events": support,
        "natural_kmeans_diagnostic_only": natural_summary,
        "contact_split_crossfit_diagnostic_only": crossfit,
        "substrate_axis_xy": selection["substrate_axis_xy"],
        "topology_primary_original_event_indices": selection[
            "topology_primary_indices"
        ],
        "topology_primary_displacements_mm": selection[
            "topology_primary_displacements_mm"
        ],
        "_pooled": {
            "ranks": ranks,
            "labels": labels,
            "readable": readable,
        },
    }


def _initialize_process_context(context: Mapping[str, Any]) -> None:
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = context


def _score_run_process(record: Mapping[str, Any]) -> dict[str, Any]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("historical rescore process context was not initialized")
    return _score_run(record, _PROCESS_CONTEXT)


def _mean_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate_candidate(runs: Sequence[Mapping[str, Any]],
                         context: Mapping[str, Any]) -> dict[str, Any]:
    if not runs:
        raise ValueError("candidate aggregation requires runs")
    field_hashes = {run.get("field_sha256") for run in runs}
    if len(field_hashes) != 1:
        raise RuntimeError(f"candidate field differs across seeds: {runs[0]['candidate_id']}")
    equal_j14 = exact.equal_network_mean([
        run["j14_v1_summary"] for run in runs
    ])
    equal_legacy = exact.equal_network_mean([
        run["phase0_legacy_soft_score_summary"] for run in runs
    ])
    primary_ranks = np.vstack([run["_pooled"]["ranks"] for run in runs])
    primary_labels = np.concatenate([run["_pooled"]["labels"] for run in runs])
    if len(primary_ranks):
        pooled = natural_kmeans(
            primary_ranks, primary_labels,
            random_state=(int(context["natural_kmeans_seed"])
                          + _stable_candidate_offset(str(runs[0]["candidate_id"]))),
        )
        pooled_summary = (
            exact._strip_natural(pooled) if pooled.get("status") == "OK"
            else exact._jsonable(pooled)
        )
    else:
        pooled_summary = _not_evaluable("NO_CONTACT_PRIMARY_CAUSAL_FAMILY")
    matrices = [
        np.asarray(run["contact_split_crossfit_diagnostic_only"]["matrix"], float)
        for run in runs
        if "matrix" in run["contact_split_crossfit_diagnostic_only"]
    ]
    crossfit_matrix = exact._mean_matrix(matrices) if matrices else None
    clean_runs = []
    for run in runs:
        clean = {key: value for key, value in run.items() if key != "_pooled"}
        clean_runs.append(clean)
    return {
        "stage_order": int(runs[0]["stage_order"]),
        "candidate_order": int(runs[0]["candidate_order"]),
        "stage_id": str(runs[0]["stage_id"]),
        "seed_pool_id": str(runs[0]["seed_pool_id"]),
        "candidate_id": str(runs[0]["candidate_id"]),
        "original_selection_eligible": bool(runs[0]["original_selection_eligible"]),
        "effective_selection_eligible": bool(runs[0]["effective_selection_eligible"]),
        "diagnostic_only": bool(runs[0]["diagnostic_only"]),
        "field_sha256": next(iter(field_hashes)),
        "n_networks": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
        "counts": {
            "contact_primary_families": int(sum(
                run["event_selection"]["n_contact_primary"] for run in runs
            )),
            "topology_primary_families": int(sum(
                run["event_selection"]["n_topology_primary"] for run in runs
            )),
            "fig4_kmeans_readable_families": int(sum(
                run["event_selection"]["n_fig4_kmeans_readable"] for run in runs
            )),
            "overlap_connected_excluded": int(sum(
                run["event_selection"]["n_returned_valid_interval"]
                - run["event_selection"]["n_contact_primary"] for run in runs
            )),
            "contact_primary_source_not_evaluable": int(sum(
                run["event_selection"]["n_contact_primary_source_not_evaluable"]
                for run in runs
            )),
            "contact_primary_lt3_finite_contacts": int(sum(
                run["event_selection"]["n_contact_primary_lt3_finite_contacts"]
                for run in runs
            )),
            "returned_less_than_minimum_contacts": int(sum(
                run["event_selection"]["n_returned_less_than_minimum_contacts"]
                for run in runs
            )),
        },
        "equal_network_j14_v1_summary": equal_j14,
        "equal_network_phase0_legacy_soft_score": equal_legacy,
        "mean_ood_fraction": _mean_optional([
            run["patient_training_assignment"]["ood_fraction"] for run in runs
        ]),
        "pooled_natural_kmeans_diagnostic_only": pooled_summary,
        "equal_network_contact_split_matrix_diagnostic_only": crossfit_matrix,
        "per_network": clean_runs,
    }


def apply_within_stage_ranks(candidates: Sequence[Mapping[str, Any]],
                            stage_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank eligible candidates inside each frozen seed pool, never across pools."""
    output = [dict(row) for row in candidates]
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in output:
        by_stage.setdefault(str(row["stage_id"]), []).append(row)
    stage_lookup = {str(stage["stage_id"]): stage for stage in stage_records}
    for stage_id, rows in by_stage.items():
        stage = stage_lookup[stage_id]
        anchor_id = str(stage["anchor_candidate_id"])
        anchor_rows = [row for row in rows if row["candidate_id"] == anchor_id]
        if len(anchor_rows) != 1:
            raise RuntimeError(f"stage anchor missing or duplicated: {stage_id}")
        anchor_objective = float(
            anchor_rows[0]["equal_network_j14_v1_summary"]["objective"]
        )
        eligible = sorted(
            (row for row in rows if row["effective_selection_eligible"]),
            key=lambda row: (
                float(row["equal_network_j14_v1_summary"]["objective"]),
                str(row["candidate_id"]),
            ),
        )
        ranks = {row["candidate_id"]: index + 1 for index, row in enumerate(eligible)}
        for row in rows:
            row["stage_anchor_candidate_id"] = anchor_id
            row["j14_v1_delta_from_stage_anchor"] = (
                float(row["equal_network_j14_v1_summary"]["objective"])
                - anchor_objective
            )
            row["within_stage_eligible_j14_v1_rank"] = ranks.get(row["candidate_id"])
            row["cross_seed_pool_rank"] = None
    return sorted(output, key=lambda row: (row["stage_order"], row["candidate_order"]))


def _patient_context(config: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    inputs = config["inputs"]
    target_path = exact._verify_record(artifact_root, inputs["patient_training_target"])
    classifier_path = exact._verify_record(
        artifact_root, inputs["frozen_direction_classifier_manifest"],
    )
    contract_path = exact._verify_record(artifact_root, inputs["contact_contract"])
    objective = config["soft_objective"]
    patient = exact.load_patient_training_target(
        target_path,
        events_per_mode=int(objective["patient_reference_events_per_mode"]),
        seed=int(objective["projection_seed"]),
    )
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]),
        n_directions=int(objective["projection_count"]),
        seed=int(objective["projection_seed"]),
    )
    calibration = calibrate_component_scales(
        patient["all_ranks"], patient["all_labels"], patient["all_blocks"],
        patient["contact_names"], projections,
        sample_size=int(objective["calibration_sample_size"]),
        draws=int(objective["calibration_draws"]),
        seed=int(objective["calibration_seed"]),
    )
    contact_contract = json.loads(contract_path.read_text())
    exact._validate_contact_contract(contact_contract, patient["contact_names"])
    frozen = exact.load_frozen_direction_classifier(
        classifier_path, patient, config["frozen_direction_classifier"],
    )
    formal = config["formal_objective"]
    if formal["schema_id"] != "topic4_rev14_j14_v1":
        raise RuntimeError("historical rescore requires topic4_rev14_j14_v1")
    shaft_aware_ami = float(adjusted_mutual_info_score(
        patient["all_labels"], patient["shaft_aware_k2_labels_diagnostic_only"],
    ))
    return {
        "patient": patient,
        "projections": projections,
        "calibration": calibration,
        "folds": contact_split_folds(contact_contract),
        "groups": contract_groups(contact_contract),
        "frozen_classifier": frozen,
        "objective": objective,
        "formal_objective": formal,
        "minimum_readable_contacts": int(
            config["diagnostics"]["minimum_fig4_readable_contacts"]
        ),
        "natural_kmeans_seed": int(config["diagnostics"]["natural_kmeans_seed"]),
        "shaft_aware_k2_diagnostic": {
            "role": "extent_diagnostic_only_not_patient_mode_identity",
            "adjusted_mutual_information_with_old_A_B": shaft_aware_ami,
            "old_by_shaft_aware_k2_contingency": patient[
                "old_by_shaft_aware_k2_contingency"
            ],
            "mapping_applied_to_classifier_or_objective": False,
        },
        "input_paths": {
            "patient_training_target": target_path,
            "frozen_direction_classifier_manifest": classifier_path,
            "contact_contract": contract_path,
        },
    }


def verify_j14_interface(config: Mapping[str, Any], artifact_root: Path,
                         context: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the frozen exact-off J14 interface without using it for ranking."""
    record = config["inputs"]["exact_off_j14_interface_reference"]
    path = exact._verify_record(artifact_root, record)
    payload = json.loads(path.read_text())
    formal = payload.get("formal_objective", {})
    observed_schema = formal.get("schema_id")
    observed_objective = formal.get("equal_network_summary", {}).get("objective")
    expected_objective = float(record["equal_network_objective"])
    if observed_schema != record["schema_id"]:
        raise RuntimeError("exact-off J14 interface schema changed")
    if observed_objective is None or not np.isclose(
            float(observed_objective), expected_objective, rtol=0.0, atol=1e-12):
        raise RuntimeError("exact-off J14 interface objective changed")
    artifact_primary_mode = payload.get("patient_training_contract", {}).get(
        "primary_mode_definition"
    )
    semantic_parity = artifact_primary_mode == "frozen_old_A_B_direction_labels"
    if not semantic_parity:
        raise RuntimeError("exact-off J14 patient-mode semantics are stale")
    expected_formal = config["formal_objective"]
    for key in (
        "sample_size_per_side", "draws_per_network", "seed", "tau",
        "model_sampling", "normalization",
    ):
        if formal.get(key) != expected_formal.get(key):
            raise RuntimeError(f"exact-off J14 formal setting changed: {key}")
    expected_projection_sha = exact._sha256_array(context["projections"])
    if formal.get("projection_sha256") != expected_projection_sha:
        raise RuntimeError("exact-off J14 projection changed")
    observed_calibration = payload.get("patient_training_contract", {}).get(
        "calibration"
    )
    if exact._jsonable(observed_calibration) != exact._jsonable(context["calibration"]):
        raise RuntimeError("exact-off J14 calibration changed")
    observed_inputs = payload.get("input_hashes", {})
    expected_input_hashes = {
        "patient_training_target": exact._sha256(
            context["input_paths"]["patient_training_target"]
        ),
        "frozen_direction_classifier_manifest": exact._sha256(
            context["input_paths"]["frozen_direction_classifier_manifest"]
        ),
        "contact_contract": exact._sha256(
            context["input_paths"]["contact_contract"]
        ),
    }
    for key, expected_hash in expected_input_hashes.items():
        if observed_inputs.get(key, {}).get("sha256") != expected_hash:
            raise RuntimeError(f"exact-off J14 input changed: {key}")
    objective_path = ROOT / "src/topic4_rev14_static_node_objective.py"
    runtime_hashes = payload.get("provenance", {}).get("runtime_path_sha256", {})
    if runtime_hashes.get(str(objective_path)) != exact._sha256(objective_path):
        raise RuntimeError("exact-off J14 objective module changed")
    return {
        "status": "PASS",
        "role": record["role"],
        "used_for_candidate_ranking": False,
        "current_patient_mode_semantic_parity": True,
        "artifact_primary_mode_definition": artifact_primary_mode,
        "limitation": "interface parity only; never a cross-seed-pool comparator",
        "path": str(path),
        "sha256": exact._sha256(path),
        "schema_id": observed_schema,
        "equal_network_objective": float(observed_objective),
        "formal_settings_verified": True,
        "projection_verified": True,
        "calibration_verified": True,
        "input_hashes_verified": True,
        "objective_module_verified": True,
    }


def produce(config_path: Path, artifact_root: Path,
            output_root: Path | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    destination = (
        _resolve(artifact_root, config["output_root"])
        if output_root is None else output_root.resolve()
    )
    inventory = build_inventory(config_path, artifact_root)
    inventory_json, inventory_csv = write_inventory(inventory, destination)
    if dry_run:
        return {
            "inventory": inventory,
            "inventory_json": inventory_json,
            "inventory_csv": inventory_csv,
            "formal_json": None,
            "formal_csv": None,
        }

    context = _patient_context(config, artifact_root)
    j14_interface = verify_j14_interface(config, artifact_root, context)
    workers = max(1, int(config["diagnostics"]["parallel_workers"]))
    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_process_context,
            initargs=(context,),
    ) as executor:
        run_scores = list(executor.map(
            _score_run_process, inventory["runs"], chunksize=1,
        ))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in run_scores:
        grouped.setdefault((row["stage_id"], row["candidate_id"]), []).append(row)
    candidates = [
        _aggregate_candidate(grouped[key], context)
        for key in sorted(
            grouped,
            key=lambda key: (
                grouped[key][0]["stage_order"], grouped[key][0]["candidate_order"],
            ),
        )
    ]
    candidates = apply_within_stage_ranks(candidates, inventory["stages"])
    stage_outputs = []
    for stage in inventory["stages"]:
        rows = [row for row in candidates if row["stage_id"] == stage["stage_id"]]
        stage_outputs.append({
            **stage,
            "candidates": rows,
            "eligible_ranking_j14_v1": [
                row["candidate_id"] for row in sorted(
                    (
                        item for item in rows
                        if item["within_stage_eligible_j14_v1_rank"] is not None
                    ),
                    key=lambda item: item["within_stage_eligible_j14_v1_rank"],
                )
            ],
            "raw_scores_comparable_scope": "within_this_stage_seed_pool_only",
        })
    contact_primary_count = int(sum(
        row["counts"]["contact_primary_families"] for row in candidates
    ))
    topology_primary_count = int(sum(
        row["counts"]["topology_primary_families"] for row in candidates
    ))
    readable_count = int(sum(
        row["counts"]["fig4_kmeans_readable_families"] for row in candidates
    ))
    source_not_evaluable_count = int(sum(
        row["counts"]["contact_primary_source_not_evaluable"]
        for row in candidates
    ))
    lt3_count = int(sum(
        row["counts"]["contact_primary_lt3_finite_contacts"]
        for row in candidates
    ))
    overlap_count = int(sum(
        row["counts"]["overlap_connected_excluded"]
        for row in candidates
    ))
    returned_lt3_count = int(sum(
        row["counts"]["returned_less_than_minimum_contacts"]
        for row in candidates
    ))
    frozen = context["frozen_classifier"]
    patient = context["patient"]
    formal_payload = {
        "schema_id": config["schema_id"],
        "status": "REV14_HISTORICAL_STATIC_NODE_ZERO_SIM_RESCORE_COMPLETE",
        "scientific_role": config["scientific_role"],
        "counts": {
            "stages": len(stage_outputs),
            "candidates": len(candidates),
            "runs": len(run_scores),
            "contact_primary_families": contact_primary_count,
            "topology_primary_families": topology_primary_count,
            "fig4_kmeans_readable_families": readable_count,
            "contact_primary_source_not_evaluable": source_not_evaluable_count,
            "contact_primary_lt3_finite_contacts": lt3_count,
            "returned_less_than_minimum_contacts": returned_lt3_count,
            "overlap_connected_excluded": overlap_count,
        },
        "event_contract": {
            "contact_primary": (
                "all returned causal families with a finite ordered time interval; "
                "exclude every member of each temporal overlap-connected component; "
                "do not require source-map evaluability or contact count"
            ),
            "patient_loss_input": (
                "all contact_primary families, including 0-2 finite contacts and "
                "source-map non-evaluable families"
            ),
            "topology_primary": (
                "contact_primary AND source_onset_evaluable AND finite displacement; "
                "topology sidecar only"
            ),
            "fig4_kmeans_readable": (
                "contact_primary AND at least 3 finite contact ranks; KMeans/figure only"
            ),
            "ood_filter": "none",
            "joint_shaft_filter": "none",
            "missing_scl_filter": "none",
        },
        "patient_training_contract": {
            "source": str(context["input_paths"]["patient_training_target"]),
            "primary_mode_definition": patient["primary_mode_definition"],
            "primary_patient_label_key": "patient_train_old_labels",
            "model_mode_assignment": (
                "raw labels and probability_B from the frozen old-A/B classifier"
            ),
            "shaft_aware_k2": context["shaft_aware_k2_diagnostic"],
            "loaded_training_data_keys": patient["loaded_training_data_keys"],
            "loaded_frozen_embedding_keys": patient["loaded_frozen_embedding_keys"],
            "patient_heldout_loaded": False,
            "patient_heldout_used": False,
            "heldout_key_or_path_accessed": False,
            "frozen_direction_classifier": {
                "source": str(context["input_paths"]["frozen_direction_classifier_manifest"]),
                "accessed_classifier_keys": frozen["accessed_classifier_keys"],
                "classifier_refit_performed": False,
                "heldout_classifier_keys_accessed": False,
                "coef_sha256": exact._sha256_array(frozen["classifier"]["coef"]),
                "centers_sha256": exact._sha256_array(
                    frozen["classifier"]["class_centers"]
                ),
                "precisions_sha256": exact._sha256_array(
                    frozen["classifier"]["class_precisions"]
                ),
            },
            "calibration_patient_training_only": True,
            "calibration": context["calibration"],
        },
        "formal_objective": {
            **context["formal_objective"],
            "candidate_ranking_source": "equal_network_j14_v1_summary.objective",
            "event_rows": "all_contact_primary_families",
            "exact_off_interface_parity": j14_interface,
        },
        "phase0_legacy_soft_score": {
            "role": "diagnostic_only_never_used_for_candidate_ranking",
            "event_rows": "all_contact_primary_families",
        },
        "stages": stage_outputs,
        "ranking_contract": {
            "formal_score": "topic4_rev14_j14_v1",
            "legacy_soft_score_used_for_ranking": False,
            "cross_seed_pool_raw_ranking_performed": False,
            "only_within_stage_seed_pool_eligible_ranks_reported": True,
            "stage_ak_posthoc_promotion": False,
            "stage_ak_role": "diagnostic_only",
        },
        "inventory": {
            "json": str(inventory_json), "sha256": exact._sha256(inventory_json),
            "csv": str(inventory_csv), "csv_sha256": exact._sha256(inventory_csv),
        },
        "input_hashes": {
            "patient_training_target": exact._sha256(
                context["input_paths"]["patient_training_target"]
            ),
            "frozen_direction_classifier_manifest": exact._sha256(
                context["input_paths"]["frozen_direction_classifier_manifest"]
            ),
            "contact_contract": exact._sha256(
                context["input_paths"]["contact_contract"]
            ),
            "exact_off_j14_interface_reference": j14_interface["sha256"],
            "stage_manifests": {
                stage["stage_id"]: stage["manifest_sha256"]
                for stage in inventory["stages"]
            },
            "worker_json_and_npz_sha256_in_inventory": True,
        },
        "provenance": runtime_provenance(config_path),
        "execution": {
            "snn_simulation_run": False,
            "zero_simulation_only": True,
            "parallel_rescore_workers": workers,
            "network_weighting": "score each network first, then equal mean",
        },
        "claim_boundary": config["claim_boundary"],
    }
    formal_json, formal_csv = _formal_paths(destination)
    csv_rows = []
    for row in candidates:
        score = row["equal_network_j14_v1_summary"]
        legacy = row["equal_network_phase0_legacy_soft_score"]
        natural = row["pooled_natural_kmeans_diagnostic_only"]
        csv_rows.append({
            "stage_id": row["stage_id"],
            "seed_pool_id": row["seed_pool_id"],
            "candidate_id": row["candidate_id"],
            "original_selection_eligible": row["original_selection_eligible"],
            "effective_selection_eligible": row["effective_selection_eligible"],
            "diagnostic_only": row["diagnostic_only"],
            "within_stage_eligible_j14_v1_rank": row[
                "within_stage_eligible_j14_v1_rank"
            ],
            "cross_seed_pool_rank": None,
            "stage_anchor_candidate_id": row["stage_anchor_candidate_id"],
            "j14_v1_delta_from_stage_anchor": row[
                "j14_v1_delta_from_stage_anchor"
            ],
            "n_networks": row["n_networks"],
            "contact_primary_families": row["counts"]["contact_primary_families"],
            "topology_primary_families": row["counts"]["topology_primary_families"],
            "fig4_kmeans_readable_families": row["counts"]["fig4_kmeans_readable_families"],
            "contact_primary_source_not_evaluable": row["counts"]["contact_primary_source_not_evaluable"],
            "contact_primary_lt3_finite_contacts": row["counts"]["contact_primary_lt3_finite_contacts"],
            "returned_less_than_minimum_contacts": row["counts"]["returned_less_than_minimum_contacts"],
            "overlap_connected_excluded": row["counts"]["overlap_connected_excluded"],
            "j14_v1_objective": score["objective"],
            "j14_v1_weakest_mode_lse": score["weakest_mode_lse"],
            "j14_v1_mode_0_mean": score["mode_0_mean"],
            "j14_v1_mode_1_mean": score["mode_1_mean"],
            "j14_v1_occupancy_js": score["occupancy_js"],
            "j14_v1_ambiguity": score["ambiguity"],
            "j14_v1_contrast_loss": score["contrast_loss"],
            "j14_v1_overlap_fraction": score["overlap_fraction"],
            "j14_v1_support_loss": score["support_loss"],
            "j14_v1_mode_0_effective_events": score[
                "mode_0_effective_events"
            ],
            "j14_v1_mode_1_effective_events": score[
                "mode_1_effective_events"
            ],
            "phase0_legacy_objective": legacy["objective"],
            "phase0_legacy_weakest_mode_lse": legacy["weakest_mode_lse"],
            "phase0_legacy_mode_0_mean": legacy["mode_0_mean"],
            "phase0_legacy_mode_1_mean": legacy["mode_1_mean"],
            "phase0_legacy_occupancy_js": legacy["occupancy_js"],
            "phase0_legacy_ambiguity": legacy["ambiguity"],
            "phase0_legacy_contrast_alignment": legacy["contrast_alignment"],
            "mean_ood_fraction": row["mean_ood_fraction"],
            "natural_kmeans_status": natural.get("status"),
            "natural_kmeans_balanced_alignment": natural.get(
                "direction_balanced_alignment"
            ),
            "natural_kmeans_seed_ami_median": natural.get(
                "kmeans_seed_ami_median"
            ),
            "field_sha256": row["field_sha256"],
        })
    columns = tuple(csv_rows[0])
    _atomic_csv(formal_csv, csv_rows, columns)
    formal_payload["candidate_csv"] = {
        "path": str(formal_csv), "sha256": exact._sha256(formal_csv),
    }
    _atomic_json(formal_json, formal_payload)
    return {
        "inventory": inventory,
        "inventory_json": inventory_json,
        "inventory_csv": inventory_csv,
        "formal": formal_payload,
        "formal_json": formal_json,
        "formal_csv": formal_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = produce(
        args.config, args.artifact_root, args.output_root, dry_run=args.dry_run,
    )
    payload = result["inventory"] if args.dry_run else result["formal"]
    print(json.dumps({
        "status": payload["status"],
        "counts": payload["counts"],
        "inventory_json": str(result["inventory_json"]),
        "inventory_csv": str(result["inventory_csv"]),
        "formal_json": None if result["formal_json"] is None else str(result["formal_json"]),
        "formal_csv": None if result["formal_csv"] is None else str(result["formal_csv"]),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
