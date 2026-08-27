#!/usr/bin/env python3
"""Zero-simulation patient-training rescore of rev13 exact_off Node runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_topic4_rev13_node_zero_sum_canary import (  # noqa: E402
    event_axis_displacements,
    overlap_connected_episode_audit,
    substrate_pca_axis,
)
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    contact_split_folds,
    crossfit_patient_readout,
    natural_kmeans,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
    normalize_event_ranks,
    soft_dual_mode_objective,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import (  # noqa: E402
    all_event_shaft_participation,
    assign_direction_modes,
)


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev14_static_node_causal_family_diagnostic.json"
EXPECTED_WORKER_STATUS = "REV13_NODE_ZERO_SUM_WORKER_COMPLETE"
TRAINING_TARGET_KEYS = (
    "contact_names",
    "patient_train_block_ids",
    "patient_train_old_labels",
    "patient_train_ranks",
    "patient_train_shaft_aware_k2_labels",
)
FROZEN_EMBEDDING_KEYS = (
    "feature_center",
    "feature_scale",
    "pca_components",
)
WORKER_ARRAY_KEYS = (
    "contact_names",
    "shaft_ids",
    "contact_xy_mm",
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
    "h",
    "delta_vtheta",
    "contact_envelope",
    "contact_envelope_dt_ms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve(artifact_root: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else artifact_root / path


def _verify_record(artifact_root: Path, record: Mapping[str, Any]) -> Path:
    path = _resolve(artifact_root, record)
    if not path.exists():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != str(record["sha256"]):
        raise RuntimeError(f"input hash changed: {path}")
    return path


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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_npz_keys(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load only explicitly named arrays; callers never materialize held-out keys."""
    with np.load(path, allow_pickle=False) as loaded:
        missing = set(keys).difference(loaded.files)
        if missing:
            raise RuntimeError(f"NPZ lacks required arrays: {sorted(missing)}")
        return {key: np.asarray(loaded[key]).copy() for key in keys}


def _skip_json_string(text: str, start: int) -> int:
    if start >= len(text) or text[start] != '"':
        raise ValueError("JSON string does not start with a quote")
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == '"':
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated JSON string")


def _skip_json_value(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        raise ValueError("missing JSON value")
    if text[index] == '"':
        return _skip_json_string(text, index)
    if text[index] in "[{":
        stack = [text[index]]
        index += 1
        while index < len(text) and stack:
            character = text[index]
            if character == '"':
                index = _skip_json_string(text, index)
                continue
            if character in "[{":
                stack.append(character)
            elif character in "]}":
                opener = stack.pop()
                if (opener, character) not in (("[", "]"), ("{", "}")):
                    raise ValueError("mismatched JSON container")
            index += 1
        if stack:
            raise ValueError("unterminated JSON container")
        return index
    while index < len(text) and text[index] not in ",}":
        index += 1
    return index


def _json_object_members(text: str, start: int = 0):
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        raise ValueError("expected a JSON object")
    index += 1
    decoder = json.JSONDecoder()
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == "}":
            return
        key, key_end = decoder.raw_decode(text, index)
        if not isinstance(key, str):
            raise ValueError("JSON object key is not a string")
        index = key_end
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != ":":
            raise ValueError("JSON object key lacks a value")
        value_start = index + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        value_end = _skip_json_value(text, value_start)
        yield key, value_start, value_end
        index = value_end
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return
        raise ValueError("malformed JSON object")


def _load_frozen_classifier_keys(manifest_path: Path,
                                 accessed_keys: tuple[str, ...]) -> dict[str, Any]:
    """Decode only frozen inference values, never manifest held-out values."""
    text = manifest_path.read_text()
    direction_span = None
    for key, value_start, value_end in _json_object_members(text):
        if key == "direction_classifier":
            direction_span = (value_start, value_end)
            break
    if direction_span is None:
        raise RuntimeError("frozen manifest lacks direction_classifier")
    source_text = text[slice(*direction_span)]
    selected: dict[str, Any] = {}
    for key, value_start, value_end in _json_object_members(source_text):
        if key in accessed_keys:
            selected[key] = json.loads(source_text[value_start:value_end])
    missing = set(accessed_keys).difference(selected)
    if missing:
        raise RuntimeError(f"frozen classifier lacks keys: {sorted(missing)}")
    return selected


def load_patient_training_target(path: Path, *, events_per_mode: int,
                                 seed: int) -> dict[str, Any]:
    arrays = _load_npz_keys(
        path, TRAINING_TARGET_KEYS + FROZEN_EMBEDDING_KEYS,
    )
    names = np.asarray(arrays["contact_names"]).astype(str)
    ranks = np.asarray(arrays["patient_train_ranks"], dtype=np.float64)
    labels = np.asarray(
        arrays["patient_train_shaft_aware_k2_labels"], dtype=np.int8,
    )
    old_labels = np.asarray(arrays["patient_train_old_labels"], dtype=np.int8)
    blocks = np.asarray(arrays["patient_train_block_ids"])
    if ranks.ndim != 2 or ranks.shape[1] != len(names):
        raise RuntimeError("patient training ranks do not match contact order")
    if any(values.shape != (len(ranks),) for values in (
            labels, old_labels, blocks)):
        raise RuntimeError("patient training arrays do not align")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("shaft-aware training target is not K=2")
    rng = np.random.default_rng(int(seed))
    reference_rows = []
    reference_labels = []
    reference_indices = []
    for mode in (0, 1):
        available = np.flatnonzero(labels == mode)
        if len(available) < int(events_per_mode):
            raise RuntimeError("patient training mode is smaller than reference budget")
        selected = np.sort(rng.choice(
            available, size=int(events_per_mode), replace=False,
        ))
        reference_rows.append(ranks[selected])
        reference_labels.append(np.full(len(selected), mode, dtype=np.int8))
        reference_indices.append(selected)
    normalized = normalize_event_ranks(ranks)
    profiles = np.asarray([
        np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])
    contingency = np.zeros((2, 2), dtype=np.int64)
    for old_label, new_label in zip(old_labels, labels):
        contingency[int(old_label), int(new_label)] += 1
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    raw_to_k2 = np.asarray([0, 1] if identity >= swapped else [1, 0], dtype=np.int8)
    return {
        "contact_names": names,
        "all_ranks": ranks,
        "all_labels": labels,
        "all_blocks": blocks,
        "embedding": {
            "center": np.asarray(arrays["feature_center"], dtype=np.float64),
            "scale": np.asarray(arrays["feature_scale"], dtype=np.float64),
            "components": np.asarray(
                arrays["pca_components"], dtype=np.float64,
            ),
        },
        "reference_ranks": np.vstack(reference_rows),
        "reference_labels": np.concatenate(reference_labels),
        "reference_training_rows": reference_indices,
        "train_profiles": profiles,
        "full_train_mode_counts": np.bincount(labels, minlength=2),
        "old_to_shaft_aware_k2": raw_to_k2,
        "old_by_shaft_aware_k2_contingency": contingency,
        "loaded_training_data_keys": list(TRAINING_TARGET_KEYS),
        "loaded_frozen_embedding_keys": list(FROZEN_EMBEDDING_KEYS),
        "patient_heldout_loaded": False,
    }


def load_frozen_direction_classifier(manifest_path: Path,
                                     patient_training: Mapping[str, Any],
                                     contract: Mapping[str, Any]) -> dict[str, Any]:
    accessed_keys = tuple(str(key) for key in contract["accessed_keys"])
    if any("heldout" in key.lower() for key in accessed_keys):
        raise RuntimeError("frozen classifier access list contains heldout key")
    classifier = _load_frozen_classifier_keys(manifest_path, accessed_keys)
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        classifier[key] = np.asarray(classifier[key], dtype=np.float64)
    if str(contract["semantics"]) != "FULL_TIMING":
        raise RuntimeError("rev11 frozen direction classifier requires FULL_TIMING")
    return {
        "contact_names": np.asarray(patient_training["contact_names"]).astype(str),
        "embedding": patient_training["embedding"],
        "classifier": classifier,
        "accessed_classifier_keys": list(accessed_keys),
        "heldout_classifier_keys_accessed": False,
        "classifier_refit_performed": False,
        "manifest_non_classifier_fields_accessed": False,
    }


def _assign_training_modes(ranks: np.ndarray,
                           frozen: Mapping[str, Any], groups: Mapping[str, Any],
                           raw_to_k2: np.ndarray) -> dict[str, np.ndarray]:
    assigned = assign_direction_modes(
        ranks, groups=groups, embedding=frozen["embedding"],
        classifier=frozen["classifier"],
    )
    mapping = np.asarray(raw_to_k2, dtype=np.int8)
    if mapping.shape != (2,) or set(mapping.tolist()) != {0, 1}:
        raise ValueError("old-to-shaft-aware K2 mapping must be a permutation")
    raw_labels = np.asarray(assigned["labels"], dtype=np.int8)
    raw_probability = np.asarray(assigned["probability_B"], dtype=np.float64)
    probability = raw_probability if int(mapping[1]) == 1 else 1.0 - raw_probability
    return {
        "probability_B": probability,
        "labels": mapping[raw_labels],
        "ood_distance": np.asarray(assigned["ood_distance"], dtype=np.float64),
        "ood": np.asarray(assigned["ood"], dtype=bool),
        "raw_old_labels": raw_labels,
    }


def _load_worker(json_path: Path, npz_path: Path, *, expected_seed: int,
                 expected_npz_sha: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(json_path.read_text())
    if payload.get("status") != EXPECTED_WORKER_STATUS:
        raise RuntimeError(f"worker is incomplete: {json_path}")
    if payload.get("candidate_id") != "exact_off":
        raise RuntimeError("static diagnostic accepts exact_off only")
    if int(payload.get("seed")) != int(expected_seed):
        raise RuntimeError("worker seed differs from frozen config")
    mechanism = payload.get("mechanism_freeze", {})
    if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
        raise RuntimeError("static diagnostic received an open mechanism")
    node = payload.get("node_accessibility", {})
    if node.get("enabled") is not False or node.get("mode") != "exact_off":
        raise RuntimeError("worker is not exact_off")
    if str(payload.get("arrays", {}).get("sha256")) != str(expected_npz_sha):
        raise RuntimeError("worker JSON and config disagree on NPZ hash")
    arrays = _load_npz_keys(npz_path, WORKER_ARRAY_KEYS)
    return payload, arrays


def primary_family_selection(arrays: Mapping[str, np.ndarray], *,
                             minimum_readable_contacts: int) -> dict[str, Any]:
    n_events = len(np.asarray(arrays["event_returned"]))
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
    maps = np.asarray(arrays["source_onset_maps_ms"], dtype=np.float64)
    if maps.shape[0] != n_events:
        raise RuntimeError("worker source maps do not align")

    order = np.argsort(
        np.asarray(arrays["event_t_on_ms"], dtype=np.float64), kind="stable",
    )
    base = (
        np.asarray(arrays["event_returned"], dtype=bool)
        & np.asarray(arrays["source_onset_evaluable"], dtype=bool)
    )
    selected = order[base[order]]
    t_on = np.asarray(arrays["event_t_on_ms"], dtype=np.float64)[selected]
    t_off = np.asarray(arrays["event_t_off_ms"], dtype=np.float64)[selected]
    trigger = np.asarray(
        arrays["event_trigger_t_on_ms"], dtype=np.float64,
    )[selected]
    axis = substrate_pca_axis(arrays["positions_E"], arrays["delta_vtheta"])
    displacement = event_axis_displacements(
        maps[selected], axis_unit=axis,
        bin_mm=float(np.asarray(arrays["source_bin_mm"]).item()),
    )
    finite = (
        np.isfinite(displacement) & np.isfinite(t_on) & np.isfinite(t_off)
        & np.isfinite(trigger) & (t_off >= t_on)
    )
    directional_indices = selected[finite]
    isolated, overlap = overlap_connected_episode_audit(
        t_on[finite], t_off[finite], displacement[finite],
    )
    primary_indices = directional_indices[isolated]
    finite_ranks = np.sum(
        np.isfinite(np.asarray(arrays["ranks"], dtype=float)[primary_indices]),
        axis=1,
    )
    readable = finite_ranks >= int(minimum_readable_contacts)
    return {
        "primary_indices": primary_indices,
        "fig4_readable_within_primary": readable,
        "fig4_readable_indices": primary_indices[readable],
        "displacements_mm": displacement[finite][isolated],
        "substrate_axis_xy": axis,
        "overlap_audit": overlap,
        "n_total": int(n_events),
        "n_returned_source_evaluable": int(len(selected)),
        "n_directional_valid": int(len(directional_indices)),
        "n_primary_isolated": int(len(primary_indices)),
        "n_fig4_readable": int(np.sum(readable)),
        "minimum_fig4_readable_contacts": int(minimum_readable_contacts),
    }


def _reorder_columns(values: np.ndarray, source_names: np.ndarray,
                     target_names: np.ndarray) -> np.ndarray:
    source = np.asarray(source_names).astype(str)
    target = np.asarray(target_names).astype(str)
    if set(source) != set(target):
        raise RuntimeError("worker and patient contact sets differ")
    order = np.asarray([
        int(np.flatnonzero(source == name)[0]) for name in target
    ])
    return np.asarray(values)[:, order]


def _strip_natural(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if key not in {"valid_event_mask", "cluster_labels"}
    }


def _display_kmeans_labels(record: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(record["cluster_labels"], dtype=np.int8)
    contingency = np.asarray(record["direction_contingency"], dtype=np.int64)
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    return 1 - labels if swapped > identity else labels.copy()


def _groups_from_names(names: np.ndarray) -> dict[str, np.ndarray]:
    shafts = np.asarray([
        "".join(character for character in str(name) if not character.isdigit())
        for name in np.asarray(names).astype(str)
    ])
    groups = {shaft: np.flatnonzero(shafts == shaft) for shaft in ("ICL", "SCL")}
    if any(not len(indices) for indices in groups.values()):
        raise RuntimeError("patient contact target lacks a shaft")
    return groups


def _validate_contact_contract(contract: Mapping[str, Any], names: np.ndarray) -> None:
    rows = sorted(contract["contacts"], key=lambda row: int(row["contact_index"]))
    contract_names = np.asarray([str(row["contact_name"]) for row in rows])
    if not np.array_equal(contract_names, np.asarray(names).astype(str)):
        raise RuntimeError("contact split contract does not match patient target")


def _score_summary(score: Mapping[str, Any]) -> dict[str, float]:
    return {
        "objective": float(score["objective"]),
        "weakest_mode_lse": float(score["weakest_mode_lse"]),
        "occupancy_js": float(score["occupancy_js"]),
        "ambiguity": float(score["ambiguity"]),
        "contrast_loss": float(score["contrast"]["loss"]),
        "contrast_alignment": float(score["contrast"]["alignment"]),
        "mode_0_mean": float(score["modes"]["0"]["mean"]),
        "mode_1_mean": float(score["modes"]["1"]["mean"]),
    }


def equal_network_mean(records: list[Mapping[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("equal-network aggregation requires records")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records):
        raise ValueError("equal-network score records differ")
    return {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in keys
    }


def _mean_matrix(matrices: list[np.ndarray]) -> np.ndarray:
    stack = np.asarray(matrices, dtype=np.float64)
    count = np.sum(np.isfinite(stack), axis=0)
    return np.divide(
        np.nansum(stack, axis=0), count,
        out=np.full(stack.shape[1:], np.nan), where=count > 0,
    )


def _runtime_hashes() -> dict[str, str]:
    return {str(path): _sha256(path) for path in _runtime_paths(DEFAULT_CONFIG)}


def _runtime_paths(config_path: Path) -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        config_path.resolve(),
        ROOT / "scripts/aggregate_topic4_rev13_node_zero_sum_canary.py",
        ROOT / "src/topic4_node_dualmode.py",
        ROOT / "src/topic4_d6_natural_kmeans.py",
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/topic4_shaft_aware_direction.py",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True,
    ).strip()


def runtime_provenance(config_path: Path) -> dict[str, Any]:
    paths = _runtime_paths(config_path)
    relative_paths = [str(path.resolve().relative_to(ROOT)) for path in paths]
    dirty_output = _git("status", "--porcelain", "--", *relative_paths)
    return {
        "git_commit_at_analysis": _git("rev-parse", "HEAD"),
        "tracked_runtime_paths": relative_paths,
        "runtime_paths_dirty": bool(dirty_output),
        "runtime_dirty_porcelain": dirty_output.splitlines(),
        "runtime_path_sha256": {str(path): _sha256(path) for path in paths},
    }


def produce(config_path: Path, artifact_root: Path,
            output_root: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    inputs = config["inputs"]
    target_path = _verify_record(artifact_root, inputs["patient_training_target"])
    classifier_manifest_path = _verify_record(
        artifact_root, inputs["frozen_direction_classifier_manifest"],
    )
    contract_path = _verify_record(artifact_root, inputs["contact_contract"])
    objective = config["soft_objective"]
    patient = load_patient_training_target(
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
    _validate_contact_contract(contact_contract, patient["contact_names"])
    folds = contact_split_folds(contact_contract)
    groups = contract_groups(contact_contract)
    frozen_classifier = load_frozen_direction_classifier(
        classifier_manifest_path, patient, config["frozen_direction_classifier"],
    )
    if not np.array_equal(
            frozen_classifier["contact_names"], patient["contact_names"]):
        raise RuntimeError("frozen classifier and training target contacts differ")

    per_network = []
    bundles = []
    input_hashes: dict[str, Any] = {
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "patient_training_target": {
            "path": str(target_path), "sha256": _sha256(target_path),
        },
        "frozen_direction_classifier_manifest": {
            "path": str(classifier_manifest_path),
            "sha256": _sha256(classifier_manifest_path),
        },
        "contact_contract": {
            "path": str(contract_path), "sha256": _sha256(contract_path),
        },
        "workers": [],
    }
    minimum_contacts = int(config["diagnostics"]["minimum_fig4_readable_contacts"])
    for worker_record in inputs["workers"]:
        json_path = _verify_record(artifact_root, worker_record["json"])
        npz_path = _verify_record(artifact_root, worker_record["npz"])
        seed = int(worker_record["seed"])
        payload, arrays = _load_worker(
            json_path, npz_path, expected_seed=seed,
            expected_npz_sha=str(worker_record["npz"]["sha256"]),
        )
        selection = primary_family_selection(
            arrays, minimum_readable_contacts=minimum_contacts,
        )
        primary = np.asarray(selection["primary_indices"], dtype=np.int64)
        ranks = _reorder_columns(
            np.asarray(arrays["ranks"], dtype=np.float64)[primary],
            arrays["contact_names"], patient["contact_names"],
        )
        onsets = _reorder_columns(
            np.asarray(arrays["onsets"], dtype=np.float64)[primary],
            arrays["contact_names"], patient["contact_names"],
        )
        assignment = _assign_training_modes(
            ranks, frozen_classifier, groups,
            patient["old_to_shaft_aware_k2"],
        )
        score = soft_dual_mode_objective(
            ranks, assignment["probability_B"],
            patient["reference_ranks"], patient["reference_labels"],
            patient["contact_names"], projections=projections,
            calibration=calibration, tau=float(objective["tau"]),
            occupancy_weight=float(objective["occupancy_weight"]),
            ambiguity_weight=float(objective["ambiguity_weight"]),
            contrast_weight=float(objective["contrast_weight"]),
        )
        readable = np.asarray(
            selection["fig4_readable_within_primary"], dtype=bool,
        )
        natural = natural_kmeans(
            ranks, assignment["labels"],
            random_state=int(config["diagnostics"]["natural_kmeans_seed"]) + seed,
        )
        if natural.get("status") != "OK":
            raise RuntimeError(f"natural KMeans not evaluable for seed {seed}")
        if not np.array_equal(np.asarray(natural["valid_event_mask"], bool), readable):
            raise RuntimeError("Fig4 readable mask differs from natural KMeans input")
        crossfit = crossfit_patient_readout(
            ranks[readable], patient["all_ranks"], patient["all_labels"], folds,
        )
        support = all_event_shaft_participation(onsets, groups)
        summary = {
            "seed": seed,
            "worker_json": str(json_path),
            "worker_npz": str(npz_path),
            "event_selection": {
                key: selection[key] for key in (
                    "n_total", "n_returned_source_evaluable",
                    "n_directional_valid", "n_primary_isolated",
                    "n_fig4_readable", "minimum_fig4_readable_contacts",
                )
            },
            "primary_original_event_indices": primary,
            "overlap_connected_episode_audit": selection["overlap_audit"],
            "soft_patient_training_score": score,
            "soft_score_summary": _score_summary(score),
            "patient_training_assignment": {
                "mode_counts": np.bincount(
                    assignment["labels"], minlength=2,
                ),
                "probability_B_mean": float(np.mean(assignment["probability_B"])),
                "ood_fraction": float(np.mean(assignment["ood"])),
                "ood_is_diagnostic_only_and_not_filtered": True,
            },
            "shaft_participation_all_primary_events": support,
            "natural_kmeans_diagnostic_only": _strip_natural(natural),
            "contact_split_crossfit_diagnostic_only": crossfit,
            "substrate_axis_xy": selection["substrate_axis_xy"],
        }
        per_network.append(summary)
        bundles.append({
            "seed": seed,
            "primary_indices": primary,
            "ranks": ranks,
            "onsets": onsets,
            "readable": readable,
            "probability_B": assignment["probability_B"],
            "patient_labels": assignment["labels"],
            "frozen_old_direction_label": assignment["raw_old_labels"],
            "ood": assignment["ood"],
            "ood_distance": assignment["ood_distance"],
            "displacement": selection["displacements_mm"],
            "source_onset_maps": np.asarray(
                arrays["source_onset_maps_ms"], dtype=np.float32,
            )[primary],
            "event_t_on_ms": np.asarray(arrays["event_t_on_ms"], float)[primary],
            "event_t_off_ms": np.asarray(arrays["event_t_off_ms"], float)[primary],
            "kmeans_raw": np.asarray(natural["cluster_labels"], dtype=np.int8),
            "kmeans_display": _display_kmeans_labels(natural),
            "contact_envelope": np.asarray(arrays["contact_envelope"], np.float32),
            "contact_envelope_dt_ms": float(
                np.asarray(arrays["contact_envelope_dt_ms"]).item()
            ),
            "positions_E": np.asarray(arrays["positions_E"], np.float32),
            "h": np.asarray(arrays["h"], np.float32),
            "delta_vtheta": np.asarray(arrays["delta_vtheta"], np.float32),
        })
        input_hashes["workers"].append({
            "seed": seed,
            "json": {"path": str(json_path), "sha256": _sha256(json_path)},
            "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
            "field_sha256": payload.get("field_sha256"),
        })

    field_hashes = {row["field_sha256"] for row in input_hashes["workers"]}
    if len(field_hashes) != 1:
        raise RuntimeError("exact_off workers do not share one frozen Node field")
    equal_score = equal_network_mean([
        row["soft_score_summary"] for row in per_network
    ])
    crossfit_matrix = _mean_matrix([
        np.asarray(row["contact_split_crossfit_diagnostic_only"]["matrix"], float)
        for row in per_network
    ])

    primary_ranks = np.vstack([row["ranks"] for row in bundles])
    primary_labels = np.concatenate([row["patient_labels"] for row in bundles])
    pooled_natural = natural_kmeans(
        primary_ranks, primary_labels,
        random_state=int(config["diagnostics"]["natural_kmeans_seed"]),
    )
    if pooled_natural.get("status") != "OK":
        raise RuntimeError("pooled natural KMeans is not evaluable")
    pooled_valid = np.asarray(pooled_natural["valid_event_mask"], bool)
    pooled_crossfit = crossfit_patient_readout(
        primary_ranks[pooled_valid], patient["all_ranks"],
        patient["all_labels"], folds,
    )

    seed_array = np.concatenate([
        np.full(len(row["ranks"]), row["seed"], dtype=np.int32)
        for row in bundles
    ])
    output_arrays = {
        "contact_names": np.asarray(patient["contact_names"], dtype="U16"),
        "primary_network_seed": seed_array,
        "primary_original_event_index": np.concatenate([
            row["primary_indices"] for row in bundles
        ]),
        "primary_event_t_on_ms": np.concatenate([
            row["event_t_on_ms"] for row in bundles
        ]),
        "primary_event_t_off_ms": np.concatenate([
            row["event_t_off_ms"] for row in bundles
        ]),
        "primary_ranks": primary_ranks.astype(np.float32),
        "primary_onsets_ms": np.vstack([row["onsets"] for row in bundles]).astype(
            np.float32
        ),
        "primary_probability_B": np.concatenate([
            row["probability_B"] for row in bundles
        ]).astype(np.float32),
        "primary_patient_training_label": primary_labels.astype(np.int8),
        "primary_frozen_old_direction_label": np.concatenate([
            row["frozen_old_direction_label"] for row in bundles
        ]).astype(np.int8),
        "primary_ood": np.concatenate([row["ood"] for row in bundles]),
        "primary_ood_distance": np.concatenate([
            row["ood_distance"] for row in bundles
        ]).astype(np.float32),
        "primary_axis_displacement_mm": np.concatenate([
            row["displacement"] for row in bundles
        ]).astype(np.float32),
        "primary_fig4_readable": np.concatenate([
            row["readable"] for row in bundles
        ]),
        "primary_source_onset_maps_ms": np.concatenate([
            row["source_onset_maps"] for row in bundles
        ]),
        "fig4_kmeans_raw_label": np.asarray(
            pooled_natural["cluster_labels"], dtype=np.int8,
        ),
        "fig4_kmeans_display_label": _display_kmeans_labels(pooled_natural),
        "patient_training_reference_ranks": np.asarray(
            patient["reference_ranks"], dtype=np.float32,
        ),
        "patient_training_reference_labels": np.asarray(
            patient["reference_labels"], dtype=np.int8,
        ),
        "patient_training_profiles": np.asarray(
            patient["train_profiles"], dtype=np.float32,
        ),
        "patient_training_reference_row": np.concatenate(
            patient["reference_training_rows"]
        ).astype(np.int64),
        "old_to_shaft_aware_k2_label": np.asarray(
            patient["old_to_shaft_aware_k2"], dtype=np.int8,
        ),
        "contact_split_fold_0": np.asarray(folds[0], dtype=np.int8),
        "contact_split_fold_1": np.asarray(folds[1], dtype=np.int8),
        "contact_envelope": np.stack([
            row["contact_envelope"] for row in bundles
        ]),
        "contact_envelope_dt_ms": np.asarray(
            [row["contact_envelope_dt_ms"] for row in bundles], dtype=np.float64,
        ),
        "positions_E": np.stack([row["positions_E"] for row in bundles]),
        "h": np.stack([row["h"] for row in bundles]),
        "delta_vtheta": np.stack([row["delta_vtheta"] for row in bundles]),
    }
    destination = (
        artifact_root / config["output_root"]
        if output_root is None else output_root.resolve()
    )
    npz_output = destination / "exact_off_static_node_fig4_bundle.npz"
    json_output = destination / "exact_off_static_node_rescore.json"
    _atomic_npz(npz_output, output_arrays)
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV14_EXACT_OFF_STATIC_NODE_ZERO_SIM_RESCORE_COMPLETE",
        "scientific_role": config["scientific_role"],
        "candidate_id": "exact_off",
        "seeds": [int(row["seed"]) for row in per_network],
        "event_contract": {
            "primary": (
                "returned AND source_onset_evaluable AND finite displacement/time; "
                "exclude all members of overlap-connected episodes"
            ),
            "fig4_readable": "primary AND at least 3 finite contact ranks",
            "ood_filter": "none",
            "joint_shaft_filter": "none",
            "missing_scl_filter": "none",
        },
        "counts": {
            "primary_isolated_families": int(len(primary_ranks)),
            "fig4_readable_families": int(np.sum(pooled_valid)),
            "per_network_primary": [
                int(row["event_selection"]["n_primary_isolated"])
                for row in per_network
            ],
            "per_network_fig4_readable": [
                int(row["event_selection"]["n_fig4_readable"])
                for row in per_network
            ],
        },
        "patient_training_contract": {
            "source": str(target_path),
            "loaded_training_data_keys": patient["loaded_training_data_keys"],
            "loaded_frozen_embedding_keys": patient[
                "loaded_frozen_embedding_keys"
            ],
            "patient_heldout_loaded": False,
            "patient_heldout_used": False,
            "heldout_key_or_path_accessed": False,
            "full_train_mode_counts": patient["full_train_mode_counts"],
            "old_by_shaft_aware_k2_contingency": patient[
                "old_by_shaft_aware_k2_contingency"
            ],
            "old_to_shaft_aware_k2_label": patient[
                "old_to_shaft_aware_k2"
            ],
            "old_to_k2_mapping_source": (
                "patient_train_old_labels x "
                "patient_train_shaft_aware_k2_labels only"
            ),
            "reference_events_per_mode": int(
                objective["patient_reference_events_per_mode"]
            ),
            "frozen_direction_classifier": {
                "source": str(classifier_manifest_path),
                "accessed_classifier_keys": frozen_classifier[
                    "accessed_classifier_keys"
                ],
                "heldout_classifier_keys_accessed": False,
                "manifest_non_classifier_fields_accessed": False,
                "classifier_refit_performed": False,
                "n_train": int(frozen_classifier["classifier"]["n_train"]),
                "coef_sha256": _sha256_array(
                    frozen_classifier["classifier"]["coef"]
                ),
                "centers_sha256": _sha256_array(
                    frozen_classifier["classifier"]["class_centers"]
                ),
                "precisions_sha256": _sha256_array(
                    frozen_classifier["classifier"]["class_precisions"]
                ),
                "ood_is_diagnostic_only": True,
            },
            "calibration_source": str(target_path),
            "calibration_patient_training_only": True,
            "calibration": calibration,
        },
        "per_network": per_network,
        "equal_network_soft_score": equal_score,
        "equal_network_contact_split_matrix": crossfit_matrix,
        "pooled_natural_kmeans_diagnostic_only": _strip_natural(pooled_natural),
        "pooled_contact_split_crossfit_diagnostic_only": pooled_crossfit,
        "input_hashes": input_hashes,
        "provenance": runtime_provenance(config_path),
        "arrays": {
            "path": str(npz_output),
            "sha256": _sha256(npz_output),
        },
        "execution": {
            "snn_simulation_run": False,
            "zero_simulation_only": True,
            "network_weighting": "score each network first, then equal mean",
        },
        "claim_boundary": config["claim_boundary"],
    }
    _atomic_json(json_output, payload)
    return {
        "payload": payload,
        "json_output": json_output,
        "npz_output": npz_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = produce(args.config, args.artifact_root, args.output_root)
    payload = result["payload"]
    print(json.dumps({
        "status": payload["status"],
        "counts": payload["counts"],
        "equal_network_soft_score": payload["equal_network_soft_score"],
        "json": str(result["json_output"]),
        "npz": str(result["npz_output"]),
    }, indent=2))


if __name__ == "__main__":
    main()
