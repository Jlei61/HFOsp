#!/usr/bin/env python3
"""Aggregate the frozen rev14 M3 canary without running the simulator.

The aggregator is deliberately training-only.  It accepts exactly the frozen
34-candidate manifest Cartesian product on the currently active network seed,
reuses the rev14 event selector and J14 implementation, and treats natural
KMeans, patient held-out data, ictal data and rendered figures as forbidden
inputs.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import rescore_topic4_rev13_exact_off_static_node as exact  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from src.topic4_rev14_fourier_field import array_sha256  # noqa: E402
from src.topic4_rev14_patient_support import (  # noqa: E402
    ALL_ENDPOINTS,
    ModelContactPrimaryData,
    PatientSupportCalibration,
    evaluate_patient_support,
    patient_training_from_mapping,
)
from src.topic4_rev14_static_node_objective import rev14_objective  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev14_m3_canary.json"
DEFAULT_J14_CONFIG = ROOT / "config/topic4_rev14_static_node_historical_rescore.json"
DEFAULT_SUPPORT_CONFIG = ROOT / "config/topic4_rev14_patient_support_acceptance.json"
MANIFEST_SCHEMA = "topic4_rev14_m3_observation_free_canary_manifest_v1"
MANIFEST_STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_FROZEN"
WORKER_STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_WORKER_COMPLETE"
OUTPUT_SCHEMA = "topic4_rev14_m3_training_aggregate_v1"
SUPPORT_MANIFEST_SCHEMA = "topic4_rev14_patient_support_acceptance_manifest_v2"
SUPPORT_SCORE_SEED = 20260828
ANALYSIS_ONLY_ALLOWED_PATHS = frozenset({
    "scripts/aggregate_topic4_rev14_m3_canary.py",
    "tests/test_topic4_rev14_m3_aggregate.py",
})
FORBIDDEN_KEY_TOKENS = (
    "heldout", "kmeans", "ictal", "seizure", "figure", "rendered_image",
)
REQUIRED_ARRAY_KEYS = tuple(dict.fromkeys(
    historical.HISTORICAL_ARRAY_KEYS + (
        "shaft_ids", "h", "rev14_fourier_modes",
        "rev14_fourier_coefficients", "rev14_frozen_signed_depth",
        "rev14_projection_sha256",
    )
))


class AggregateContractError(RuntimeError):
    """A frozen artifact violates the aggregation contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _support_artifact_array_sha256(values: np.ndarray) -> str:
    """Match the array-byte contract used by the patient-support freezer."""
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(header + array.view(np.uint8).tobytes()).hexdigest()


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
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True,
                       allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "candidate_id", "seed", "selection_eligible", "field_kind",
        "inventory_status", "run_status", "j14_objective",
        "j14_delta_from_exact_off", "weakest_mode_lse",
        "mode_0_effective_events", "mode_1_effective_events",
        "patient_support_status", "patient_support_score",
        "patient_support_delta_from_exact_off", "contact_primary_events",
        "fig4_readable_events", "ood_events", "formal_rank",
        "worker_json", "worker_json_sha256", "worker_npz",
        "worker_npz_sha256", "error",
    )
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in columns})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _forbidden_key_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            path = f"{prefix}.{key}"
            if path == "$.provenance":
                # Provenance contains validated source-path registries whose
                # filenames can legitimately include "interictal" or "ictal".
                # It is audited structurally below and is not a scientific input.
                continue
            if any(token in lowered for token in FORBIDDEN_KEY_TOKENS):
                found.append(path)
            found.extend(_forbidden_key_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_key_paths(item, f"{prefix}[{index}]"))
    return found


def _forbidden_array_keys(keys: Sequence[str]) -> list[str]:
    return sorted(
        key for key in map(str, keys)
        if any(token in key.lower() for token in FORBIDDEN_KEY_TOKENS)
    )


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _load_manifest(config_path: Path, artifact_root: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    manifest_path = _resolve(artifact_root, config["candidate_manifest"])
    if not manifest_path.is_file():
        raise AggregateContractError(f"frozen M3 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_id") != MANIFEST_SCHEMA:
        raise AggregateContractError("M3 manifest schema changed")
    if manifest.get("status") != MANIFEST_STATUS:
        raise AggregateContractError("M3 manifest is not formally frozen")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise AggregateContractError("M3 manifest config hash changed")
    candidates = list(manifest.get("candidates", []))
    identifiers = [str(row.get("candidate_id")) for row in candidates]
    if len(candidates) != 34 or len(set(identifiers)) != 34:
        raise AggregateContractError("M3 manifest must contain 34 unique candidates")
    selectable = sum(bool(row.get("selection_eligible")) for row in candidates)
    if selectable != 32:
        raise AggregateContractError("M3 manifest must contain 32 selectable candidates")
    active = [int(seed) for seed in manifest.get("search", {}).get(
        "active_network_seeds", [])]
    if active != [int(seed) for seed in config["search"]["active_network_seeds"]]:
        raise AggregateContractError("manifest active seed pool differs from config")
    if len(active) != 1:
        raise AggregateContractError("seed-2321 aggregation accepts one active seed")
    if not manifest.get("search", {}).get("common_random_numbers_across_candidates"):
        raise AggregateContractError("M3 manifest does not freeze common random numbers")
    provenance = manifest.get("provenance", {})
    if (not provenance.get("formal_ready")
            or not provenance.get("all_explicit_paths_clean")
            or provenance.get("git_commit") != provenance.get("expected_git_commit")):
        raise AggregateContractError("M3 manifest provenance is not formal and clean")
    return config, {
        "path": manifest_path,
        "sha256": _sha256(manifest_path),
        "payload": manifest,
        "candidate_by_id": {str(row["candidate_id"]): row for row in candidates},
        "active_seed": active[0],
        "git_commit": str(provenance["git_commit"]),
    }


def _event_unit_parity(worker: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    checks = (
        "name", "minimum_active_neurons", "minimum_dominance",
        "movie_frame_ms", "movie_bin_mm", "causal_memory_method",
        "contact_geometry_used_for_boundary",
    )
    for key in checks:
        if key in frozen and worker.get(key) != frozen.get(key):
            raise AggregateContractError(f"worker event_unit changed: {key}")
    if frozen.get("name") == "edge_supported_causal_family_observation":
        edge = worker.get("edge_support", {})
        for key in (
                "minimum_parent_support", "minimum_parent_dominance",
                "edge_delay_rounding"):
            if edge.get(key) != frozen.get(key):
                raise AggregateContractError(
                    f"worker edge-supported event_unit changed: {key}"
                )


def _worker_identity(payload: Mapping[str, Any], path: Path) -> tuple[str, int]:
    candidate = payload.get("candidate_id")
    seed = payload.get("seed")
    if not isinstance(candidate, str) or isinstance(seed, bool):
        raise AggregateContractError(f"worker identity is malformed: {path}")
    try:
        return candidate, int(seed)
    except (TypeError, ValueError) as error:
        raise AggregateContractError(f"worker seed is malformed: {path}") from error


def _load_worker_arrays(npz_path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(npz_path, allow_pickle=False) as loaded:
        available = sorted(loaded.files)
        polluted = _forbidden_array_keys(available)
        if polluted:
            raise AggregateContractError(
                f"worker NPZ contains forbidden analysis arrays: {polluted}"
            )
        missing = sorted(set(REQUIRED_ARRAY_KEYS).difference(available))
        if missing:
            raise AggregateContractError(f"worker NPZ lacks required arrays: {missing}")
        arrays = {key: np.asarray(loaded[key]).copy() for key in REQUIRED_ARRAY_KEYS}
    return arrays, available


def _validate_worker(
        json_path: Path, payload: Mapping[str, Any], *, candidate: Mapping[str, Any],
        active_seed: int, manifest: Mapping[str, Any], manifest_sha256: str,
        manifest_commit: str, config: Mapping[str, Any], artifact_root: Path,
) -> dict[str, Any]:
    polluted = _forbidden_key_paths(payload)
    if polluted:
        raise AggregateContractError(
            f"worker JSON contains forbidden analysis fields: {polluted[:8]}"
        )
    candidate_id, seed = _worker_identity(payload, json_path)
    if candidate_id != str(candidate["candidate_id"]) or seed != int(active_seed):
        raise AggregateContractError("worker identity differs from frozen Cartesian product")
    if payload.get("status") != WORKER_STATUS:
        raise AggregateContractError(f"worker did not complete: {json_path}")
    if bool(payload.get("candidate_selection_eligible")) != bool(
            candidate["selection_eligible"]):
        raise AggregateContractError("worker selection eligibility changed")
    if not _same_json(payload.get("fourier_field"), candidate.get("fourier_coordinate")):
        raise AggregateContractError("worker Fourier coordinate differs from manifest")
    simulation = payload.get("simulation", {})
    duration = float(simulation.get("duration_ms", np.nan))
    expected_duration = float(config["search"]["simulation"]["duration_ms"])
    if not np.isclose(duration, expected_duration, rtol=0.0, atol=0.0):
        raise AggregateContractError("worker simulation duration changed")
    mechanism = payload.get("mechanism_freeze", {})
    if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
        raise AggregateContractError("worker activated a forbidden pathway")
    if mechanism.get("edge_coefficients_all_zero") is not True:
        raise AggregateContractError("worker edge coefficients are not exactly off")
    if mechanism.get("static_node_field") != candidate.get("field_kind"):
        raise AggregateContractError("worker field kind differs from manifest")
    _event_unit_parity(payload.get("event_unit", {}), manifest["event_unit"])

    provenance = payload.get("provenance", {})
    explicit = provenance.get("rev14_explicit_runtime_freeze", {})
    manifest_audit = provenance.get("rev14_manifest_audit", {})
    if (explicit.get("git_commit") != manifest_commit
            or explicit.get("expected_git_commit") != manifest_commit
            or explicit.get("formal_ready") is not True
            or explicit.get("all_explicit_paths_clean") is not True):
        raise AggregateContractError("worker rev14 runtime provenance changed")
    if manifest_audit.get("manifest_sha256") != manifest_sha256:
        raise AggregateContractError("worker used a different frozen manifest")
    if (provenance.get("git_commit") != manifest_commit
            or provenance.get("expected_git_commit") != manifest_commit
            or int(provenance.get("runtime_modules_dirty", 1)) != 0
            or int(provenance.get("runtime_modules_match_expected_commit", 0)) != 1):
        raise AggregateContractError("base worker runtime provenance is not clean")

    arrays_record = payload.get("arrays", {})
    npz_path = _resolve(artifact_root, arrays_record.get("path", ""))
    if npz_path != json_path.with_suffix(".npz").resolve():
        raise AggregateContractError("worker JSON and NPZ are not sibling artifacts")
    if not npz_path.is_file():
        raise AggregateContractError(f"worker NPZ is missing: {npz_path}")
    npz_hash = _sha256(npz_path)
    if npz_hash != str(arrays_record.get("sha256")):
        raise AggregateContractError("worker NPZ hash differs from JSON")
    arrays, available_keys = _load_worker_arrays(npz_path)

    projection = payload.get("field_projection", {}).get("hashes", {})
    npz_projection = str(np.asarray(arrays["rev14_projection_sha256"]).item())
    if npz_projection != projection.get("projection_sha256"):
        raise AggregateContractError("worker projection hash differs between JSON and NPZ")
    if projection.get("h_sha256") != payload.get("field_sha256"):
        raise AggregateContractError("worker JSON field hashes disagree")
    # The composed rev12 writer deliberately stores h and delta_vtheta as
    # float32, whereas the frozen projection hashes refer to float64 arrays.
    # Their byte hashes therefore cannot be compared.  Integrity is carried by
    # the whole-NPZ hash and the exact float64 projection/depth/coordinate
    # records below; the readout arrays must still be finite and aligned.
    if (not np.all(np.isfinite(arrays["h"]))
            or not np.all(np.isfinite(arrays["delta_vtheta"]))
            or np.asarray(arrays["h"]).shape != np.asarray(arrays["positions_E"]).shape[:1]):
        raise AggregateContractError("worker float32 field arrays are invalid")
    depth_hash = array_sha256(arrays["rev14_frozen_signed_depth"])
    expected_depth = config["node_mapping"]["signed_depth_contract"]["sha256"]
    if depth_hash != expected_depth or projection.get(
            "frozen_signed_depth_sha256") != expected_depth:
        raise AggregateContractError("worker frozen signed-depth hash changed")
    coordinate = candidate.get("fourier_coordinate")
    coefficients = np.asarray(arrays["rev14_fourier_coefficients"])
    if coordinate is None:
        if coefficients.size != 0:
            raise AggregateContractError("exact_off unexpectedly stores Fourier coefficients")
    else:
        expected_coefficients = np.asarray(coordinate["coefficients"], dtype=np.float64)
        if not np.array_equal(coefficients, expected_coefficients):
            raise AggregateContractError("worker Fourier coefficients changed")
        if array_sha256(coefficients) != coordinate["coefficients_sha256"]:
            raise AggregateContractError("worker Fourier coefficient hash changed")
        if not np.array_equal(
                np.asarray(arrays["rev14_fourier_modes"]),
                np.asarray(coordinate["modes"])):
            raise AggregateContractError("worker Fourier mode inventory changed")

    runaway = simulation.get("runaway_early_stop_ms")
    numerical_failure = simulation.get("numerical_failure")
    run_status = "VALID"
    if runaway is not None:
        run_status = "INVALID_RUNAWAY"
    elif numerical_failure not in (None, False, ""):
        run_status = "INVALID_NUMERICAL"
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "selection_eligible": bool(candidate["selection_eligible"]),
        "field_kind": str(candidate["field_kind"]),
        "inventory_status": "PRESENT_VALIDATED",
        "run_status": run_status,
        "worker_json": str(json_path),
        "worker_json_sha256": _sha256(json_path),
        "worker_npz": str(npz_path),
        "worker_npz_sha256": npz_hash,
        "npz_array_keys": available_keys,
        "arrays": arrays,
        "payload": dict(payload),
        "error": None,
    }


def inventory_workers(
        *, config_path: Path, artifact_root: Path, worker_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config, frozen = _load_manifest(config_path.resolve(), artifact_root.resolve())
    manifest = frozen["payload"]
    expected = {
        (str(row["candidate_id"]), int(frozen["active_seed"])): row
        for row in manifest["candidates"]
    }
    discovered: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    malformed: list[dict[str, str]] = []
    if worker_root.is_dir():
        for path in sorted(worker_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
                identity = _worker_identity(payload, path)
                discovered.setdefault(identity, []).append((path.resolve(), payload))
            except Exception as error:  # retained in the audit, never skipped
                malformed.append({"path": str(path), "error": str(error)})
    rows: list[dict[str, Any]] = []
    for identity, candidate in expected.items():
        matches = discovered.get(identity, [])
        base = {
            "candidate_id": identity[0], "seed": identity[1],
            "selection_eligible": bool(candidate["selection_eligible"]),
            "field_kind": str(candidate["field_kind"]),
            "formal_rank": None,
        }
        if not matches:
            rows.append({
                **base, "inventory_status": "MISSING", "run_status": None,
                "worker_json": None, "worker_json_sha256": None,
                "worker_npz": None, "worker_npz_sha256": None,
                "error": "missing worker JSON/NPZ pair",
            })
            continue
        if len(matches) != 1:
            rows.append({
                **base, "inventory_status": "DUPLICATE", "run_status": None,
                "worker_json": ";".join(str(path) for path, _ in matches),
                "worker_json_sha256": None, "worker_npz": None,
                "worker_npz_sha256": None,
                "error": f"duplicate worker identity ({len(matches)} JSON files)",
            })
            continue
        path, payload = matches[0]
        try:
            rows.append(_validate_worker(
                path, payload, candidate=candidate,
                active_seed=frozen["active_seed"], manifest=manifest,
                manifest_sha256=frozen["sha256"],
                manifest_commit=frozen["git_commit"], config=config,
                artifact_root=artifact_root.resolve(),
            ))
        except Exception as error:
            rows.append({
                **base, "inventory_status": "INVALID_ARTIFACT", "run_status": None,
                "worker_json": str(path), "worker_json_sha256": _sha256(path),
                "worker_npz": None, "worker_npz_sha256": None,
                "error": str(error),
            })
    extra = sorted(set(discovered).difference(expected))
    audit = {
        "expected_runs": len(expected),
        "present_validated": sum(
            row["inventory_status"] == "PRESENT_VALIDATED" for row in rows
        ),
        "missing": [row["candidate_id"] for row in rows if row["inventory_status"] == "MISSING"],
        "duplicate": [row["candidate_id"] for row in rows if row["inventory_status"] == "DUPLICATE"],
        "invalid_artifact": [
            row["candidate_id"] for row in rows
            if row["inventory_status"] == "INVALID_ARTIFACT"
        ],
        "extra_identities": [[candidate, seed] for candidate, seed in extra],
        "malformed_json": malformed,
        "complete_cartesian_product": (
            len(rows) == 34
            and all(row["inventory_status"] == "PRESENT_VALIDATED" for row in rows)
            and not extra and not malformed
        ),
    }
    return config, frozen, rows, audit


def _load_calibration_from_sidecar(
        manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
) -> PatientSupportCalibration:
    floor = manifest["floor_contract"]
    metadata = dict(floor["calibration_metadata"])
    keys = floor["all_calibration_array_keys"]
    values: dict[str, Any] = dict(metadata)
    for field in dataclasses.fields(PatientSupportCalibration):
        name = field.name
        if name in values:
            continue
        record = keys.get(name)
        if record is None:
            raise AggregateContractError(f"support sidecar lacks calibration field: {name}")
        if set(record) == {"__self__"}:
            values[name] = np.asarray(arrays[record["__self__"]]).copy()
        else:
            values[name] = {
                endpoint: np.asarray(arrays[array_key]).copy()
                for endpoint, array_key in record.items()
            }
    values["floor_q95"] = {
        str(key): float(value) for key, value in values["floor_q95"].items()
    }
    values["eligible_block_counts"] = {
        str(key): int(value) for key, value in values["eligible_block_counts"].items()
    }
    return PatientSupportCalibration(**{
        field.name: values[field.name]
        for field in dataclasses.fields(PatientSupportCalibration)
    })


def _load_support_context(
        support_config_path: Path, artifact_root: Path, j14_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = json.loads(support_config_path.read_text())
    output = config["output"]
    directory = _resolve(artifact_root, output["directory"])
    manifest_path = directory / output["manifest"]
    npz_path = directory / output["npz"]
    if not manifest_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError("frozen patient-support manifest/NPZ is not available")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_id") != SUPPORT_MANIFEST_SCHEMA:
        raise AggregateContractError("patient-support sidecar schema changed")
    if manifest.get("status") != "PATIENT_TRAINING_SUPPORT_ACCEPTANCE_FROZEN":
        raise AggregateContractError("patient-support sidecar is not frozen")
    provenance = manifest.get("provenance", {})
    provenance_config = provenance.get("config", {})
    if provenance_config.get("sha256") != _sha256(support_config_path):
        raise AggregateContractError("patient-support sidecar config hash changed")
    if provenance.get("runtime_sources_dirty_or_untracked") is not False:
        raise AggregateContractError("patient-support sidecar runtime was dirty")
    forbidden = manifest.get("forbidden_input_audit", {})
    expected_forbidden = {
        "only_three_scientific_inputs": True,
        "patient_evaluation_arrays_loaded": False,
        "seizure_arrays_loaded": False,
        "model_artifacts_loaded": False,
        "candidate_scores_loaded": False,
        "candidate_selection_performed": False,
        "simulation_performed": False,
    }
    if any(forbidden.get(key) is not value for key, value in expected_forbidden.items()):
        raise AggregateContractError("patient-support sidecar input boundary changed")
    if _sha256(npz_path) != manifest["artifact"]["npz_sha256"]:
        raise AggregateContractError("patient-support NPZ hash changed")
    with np.load(npz_path, allow_pickle=False) as loaded:
        support_arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    if set(support_arrays) != set(manifest["artifact"]["array_keys"]):
        raise AggregateContractError("patient-support sidecar array inventory changed")
    for key, expected in manifest["artifact"]["array_sha256"].items():
        if _support_artifact_array_sha256(support_arrays[key]) != expected:
            raise AggregateContractError(f"patient-support array hash changed: {key}")

    target_record = j14_config["inputs"]["patient_training_target"]
    support_record = config["inputs"]["patient_training_target"]
    if target_record != support_record:
        raise AggregateContractError("J14 and patient-support training targets differ")
    target_path = exact._verify_record(artifact_root, target_record)
    target_keys = (
        "contact_names", "shaft_ids", "patient_train_onsets",
        "patient_train_old_labels", "patient_train_block_ids",
    )
    target = exact._load_npz_keys(target_path, target_keys)
    patient = patient_training_from_mapping({
        "contact_names": target["contact_names"],
        "shaft_ids": target["shaft_ids"],
        "patient_train_onsets": target["patient_train_onsets"],
        "patient_train_old_labels": target["patient_train_old_labels"],
        "patient_train_classifier_labels": support_arrays[
            "patient_train_classifier_labels"
        ],
        "patient_train_block_ids": target["patient_train_block_ids"],
        "patient_train_ood": support_arrays["patient_train_ood"],
        "primary_label_key": "patient_train_old_labels",
        "source_sha256": support_record["sha256"],
    }, expected_source_sha256=support_record["sha256"])
    calibration = _load_calibration_from_sidecar(manifest, support_arrays)
    if set(calibration.floor_q95) != set(ALL_ENDPOINTS):
        raise AggregateContractError("patient-support endpoint set changed")
    return {
        "patient": patient,
        "calibration": calibration,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "npz_path": npz_path,
        "npz_sha256": _sha256(npz_path),
    }


def _score_worker(
        record: Mapping[str, Any], context: Mapping[str, Any],
        support_context: Mapping[str, Any],
) -> dict[str, Any]:
    if record["run_status"] != "VALID":
        return {**record, "j14_v1": None, "patient_support": None}
    arrays = record["arrays"]
    selection = historical.three_layer_event_selection(
        arrays, minimum_readable_contacts=int(context["minimum_readable_contacts"]),
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
    assignment = (
        exact._assign_training_modes(
            ranks, context["frozen_classifier"], context["groups"],
        )
        if len(ranks)
        else {
            "probability_B": np.empty(0, dtype=np.float64),
            "labels": np.empty(0, dtype=np.int8),
            "ood": np.empty(0, dtype=bool),
        }
    )
    readable = np.asarray(
        selection["fig4_kmeans_readable_within_contact"], dtype=bool,
    )
    ood = np.asarray(assignment["ood"], dtype=bool)
    mode_evidence = readable & ~ood
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
    support = evaluate_patient_support(
        support_context["patient"], support_context["calibration"],
        ModelContactPrimaryData(
            contact_names=tuple(np.asarray(context["patient"]["contact_names"]).astype(str)),
            onsets=onsets,
            classifier_labels=np.asarray(assignment["labels"], dtype=np.int8),
            ood=ood,
        ),
        seed=SUPPORT_SCORE_SEED,
    )
    if support.status == "INVALID":
        raise AggregateContractError(
            f"patient-support score is invalid: {support.invalid_reason}"
        )
    clean = {key: value for key, value in record.items() if key not in {"arrays", "payload"}}
    return {
        **clean,
        "event_selection": {
            key: selection[key] for key in (
                "n_total", "n_returned", "n_returned_invalid_interval",
                "n_returned_valid_interval", "n_contact_primary",
                "n_topology_primary", "n_fig4_kmeans_readable",
                "n_contact_primary_source_not_evaluable",
                "n_contact_primary_lt3_finite_contacts",
                "n_returned_less_than_minimum_contacts",
            )
        },
        "overlap_connected_episode_audit": selection["overlap_audit"],
        "patient_training_assignment": {
            "classifier_A": int(np.sum(np.asarray(assignment["labels"]) == 0)),
            "classifier_B": int(np.sum(np.asarray(assignment["labels"]) == 1)),
            "ood_count": int(np.sum(ood)),
            "all_contact_primary_in_distance": True,
            "ood_rows_retained_in_mode_distances": True,
            "mode_evidence_definition": (
                "fig4-readable AND frozen-classifier in-support; evidence only, "
                "never a distance-row filter"
            ),
        },
        "j14_v1": j14,
        "j14_v1_summary": exact._j14_summary(j14),
        "patient_support": dataclasses.asdict(support),
        "forbidden_analysis_audit": {
            "patient_heldout_loaded": False,
            "natural_kmeans_computed": False,
            "ictal_data_loaded": False,
            "figure_or_image_loaded": False,
        },
    }


def _csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    j14 = record.get("j14_v1_summary") or {}
    support = record.get("patient_support") or {}
    event = record.get("event_selection") or {}
    assignment = record.get("patient_training_assignment") or {}
    return {
        "candidate_id": record.get("candidate_id"),
        "seed": record.get("seed"),
        "selection_eligible": record.get("selection_eligible"),
        "field_kind": record.get("field_kind"),
        "inventory_status": record.get("inventory_status"),
        "run_status": record.get("run_status"),
        "j14_objective": j14.get("objective"),
        "j14_delta_from_exact_off": record.get("j14_delta_from_exact_off"),
        "weakest_mode_lse": j14.get("weakest_mode_lse"),
        "mode_0_effective_events": j14.get("mode_0_effective_events"),
        "mode_1_effective_events": j14.get("mode_1_effective_events"),
        "patient_support_status": support.get("status"),
        "patient_support_score": support.get("score"),
        "patient_support_delta_from_exact_off": record.get(
            "patient_support_delta_from_exact_off"
        ),
        "contact_primary_events": event.get("n_contact_primary"),
        "fig4_readable_events": event.get("n_fig4_kmeans_readable"),
        "ood_events": assignment.get("ood_count"),
        "formal_rank": record.get("formal_rank"),
        "worker_json": record.get("worker_json"),
        "worker_json_sha256": record.get("worker_json_sha256"),
        "worker_npz": record.get("worker_npz"),
        "worker_npz_sha256": record.get("worker_npz_sha256"),
        "error": record.get("error"),
    }


def _runtime_provenance(expected_commit: str) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        worktree_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT, text=True,
        ).splitlines()
        changed_paths = subprocess.check_output(
            ["git", "diff", "--name-only", f"{expected_commit}..HEAD"],
            cwd=ROOT, text=True,
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
        worktree_status = ["git provenance unavailable"]
        changed_paths = ["git provenance unavailable"]
    aggregator_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(Path(__file__).relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).splitlines() if commit is not None else ["git provenance unavailable"]
    analysis_only_commit = (
        commit is not None
        and set(changed_paths).issubset(ANALYSIS_ONLY_ALLOWED_PATHS)
    )
    return {
        "git_commit_at_analysis": commit,
        "expected_git_commit": expected_commit,
        "head_matches_frozen_manifest": commit == expected_commit,
        "analysis_only_commit_allowed": analysis_only_commit,
        "paths_changed_since_worker_freeze": changed_paths,
        "analysis_only_allowed_paths": sorted(ANALYSIS_ONLY_ALLOWED_PATHS),
        "aggregator_path": str(Path(__file__).resolve()),
        "aggregator_sha256": _sha256(Path(__file__).resolve()),
        "aggregator_dirty_or_untracked": bool(aggregator_status),
        "aggregator_git_status": aggregator_status,
        "worktree_dirty_or_untracked": bool(worktree_status),
        "worktree_git_status": worktree_status,
        "formal_ready": (
            analysis_only_commit
            and not aggregator_status
            and not worktree_status
        ),
        "snn_simulation_run": False,
    }


def aggregate(
        *, config_path: Path = DEFAULT_CONFIG,
        j14_config_path: Path = DEFAULT_J14_CONFIG,
        support_config_path: Path = DEFAULT_SUPPORT_CONFIG,
        artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
        worker_root: Path | None = None,
        output_root: Path | None = None,
        allow_incomplete: bool = False,
        context_override: Mapping[str, Any] | None = None,
        support_context_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit and aggregate one frozen active-seed Cartesian product."""
    config_path = config_path.resolve()
    j14_config_path = j14_config_path.resolve()
    support_config_path = support_config_path.resolve()
    artifact_root = artifact_root.resolve()
    config_hint = json.loads(config_path.read_text())
    base_output = _resolve(artifact_root, config_hint["output_root"])
    worker_root = (base_output / "workers" if worker_root is None else worker_root.resolve())
    output_root = (base_output / "analysis" if output_root is None else output_root.resolve())
    config, frozen, records, inventory = inventory_workers(
        config_path=config_path, artifact_root=artifact_root,
        worker_root=worker_root,
    )
    active_seed = int(frozen["active_seed"])
    provenance = _runtime_provenance(frozen["git_commit"])
    json_path = output_root / f"m3_seed_{active_seed}_training_aggregate.json"
    csv_path = output_root / f"m3_seed_{active_seed}_training_per_run.csv"

    blocking_inventory = (
        not inventory["complete_cartesian_product"] and not allow_incomplete
    )
    status = "INCOMPLETE"
    scored: list[dict[str, Any]] = []
    input_error = None
    if blocking_inventory:
        scored = [{key: value for key, value in row.items()
                   if key not in {"arrays", "payload"}} for row in records]
    elif not provenance["formal_ready"] and context_override is None:
        input_error = "aggregation runtime is not a clean analysis-only descendant"
        status = "INVALID_PROVENANCE"
        scored = [{key: value for key, value in row.items()
                   if key not in {"arrays", "payload"}} for row in records]
    else:
        try:
            j14_config = json.loads(j14_config_path.read_text())
            context = (
                dict(context_override) if context_override is not None
                else historical._patient_context(j14_config, artifact_root)
            )
            if context_override is None:
                historical.verify_j14_interface(j14_config, artifact_root, context)
            support_context = (
                dict(support_context_override)
                if support_context_override is not None
                else _load_support_context(
                    support_config_path, artifact_root, j14_config,
                )
            )
            for record in records:
                if record["inventory_status"] != "PRESENT_VALIDATED":
                    scored.append({key: value for key, value in record.items()
                                   if key not in {"arrays", "payload"}})
                    continue
                scored.append(_score_worker(record, context, support_context))
        except Exception as error:
            input_error = str(error)
            scored = [{key: value for key, value in row.items()
                       if key not in {"arrays", "payload"}} for row in records]
            status = "INVALID_INPUT"

    exact_rows = [row for row in scored if row.get("candidate_id") == "exact_off"]
    for row in scored:
        row.setdefault("formal_rank", None)
    reference_valid = (
        len(exact_rows) == 1 and exact_rows[0].get("run_status") == "VALID"
        and exact_rows[0].get("j14_v1_summary") is not None
    )
    if input_error is None and not blocking_inventory:
        if not reference_valid:
            status = "INVALID_EXACT_OFF_REFERENCE"
        else:
            exact_j14 = float(exact_rows[0]["j14_v1_summary"]["objective"])
            exact_support = float(exact_rows[0]["patient_support"]["score"])
            for row in scored:
                j14 = row.get("j14_v1_summary")
                support = row.get("patient_support")
                row["j14_delta_from_exact_off"] = (
                    None if j14 is None else float(j14["objective"]) - exact_j14
                )
                row["patient_support_delta_from_exact_off"] = (
                    None if support is None
                    else float(support["score"]) - exact_support
                )
            selectable = [
                row for row in scored
                if row.get("selection_eligible")
                and row.get("run_status") == "VALID"
                and row.get("j14_v1_summary") is not None
                and row.get("patient_support") is not None
            ]
            selectable.sort(key=lambda row: (
                float(row["j14_v1_summary"]["objective"]),
                str(row["candidate_id"]),
            ))
            for rank, row in enumerate(selectable, start=1):
                row["formal_rank"] = rank
            n_invalid = sum(
                row.get("run_status") in {"INVALID_RUNAWAY", "INVALID_NUMERICAL"}
                for row in scored
            )
            status = (
                "PARTIAL_DIAGNOSTIC" if not inventory["complete_cartesian_product"]
                else "COMPLETE_WITH_INVALID_RUNS" if n_invalid
                else "COMPLETE"
            )

    clean_records = sorted(scored, key=lambda row: (
        [str(candidate["candidate_id"]) for candidate in frozen["payload"]["candidates"]].index(
            str(row["candidate_id"])
        ) if str(row.get("candidate_id")) in {
            str(candidate["candidate_id"]) for candidate in frozen["payload"]["candidates"]
        } else 10_000,
        int(row.get("seed", active_seed)),
    ))
    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": status,
        "scientific_role": "training_only_rev14_m3_active_seed_ranking",
        "active_seed": active_seed,
        "complete_collection_required_by_default": True,
        "allow_incomplete": bool(allow_incomplete),
        "inventory": inventory,
        "input_error": input_error,
        "ranking_contract": {
            "eligible_pool": "32 manifest-selectable M3 fields only",
            "primary": "J14_v1 objective ascending",
            "tie_break_1": "candidate_id lexical",
            "patient_support_role": (
                "reported absolute acceptance diagnostic; never a canary ranking term"
            ),
            "paired_reference": "same-seed exact_off under common random numbers",
            "natural_kmeans_used": False,
            "patient_heldout_used": False,
            "ictal_data_used": False,
            "figure_or_image_used": False,
        },
        "event_contract": {
            "selector": "three_layer_event_selection from rev14 historical rescore",
            "distance_rows": "all contact_primary events including 0-2-contact and OOD rows",
            "mode_evidence_only": "fig4-readable AND in-support",
            "overlap_rule": "exclude every member of overlap-connected episodes",
        },
        "reference_valid": reference_valid,
        "formal_ranking": [
            row["candidate_id"] for row in sorted(
                (row for row in clean_records if row.get("formal_rank") is not None),
                key=lambda row: int(row["formal_rank"]),
            )
        ],
        "per_run": clean_records,
        "inputs": {
            "m3_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "m3_manifest": {"path": str(frozen["path"]), "sha256": frozen["sha256"]},
            "j14_config": {"path": str(j14_config_path), "sha256": _sha256(j14_config_path)},
            "patient_support_config": {
                "path": str(support_config_path),
                "sha256": _sha256(support_config_path),
            },
            "worker_root": str(worker_root),
        },
        "outputs": {"json": str(json_path), "csv": str(csv_path)},
        "provenance": provenance,
        "claim_boundary": (
            "Seed-active training-only M3 canary aggregation. No natural KMeans, "
            "patient held-out, ictal target or image enters ranking. A complete "
            "seed Cartesian product is required for formal ranking."
        ),
    }
    _atomic_json(json_path, payload)
    _atomic_csv(csv_path, [_csv_row(row) for row in clean_records])
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--j14-config", type=Path, default=DEFAULT_J14_CONFIG)
    parser.add_argument("--patient-support-config", type=Path,
                        default=DEFAULT_SUPPORT_CONFIG)
    parser.add_argument("--artifact-root", type=Path,
                        default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    payload = aggregate(
        config_path=args.config,
        j14_config_path=args.j14_config,
        support_config_path=args.patient_support_config,
        artifact_root=args.artifact_root,
        worker_root=args.worker_root,
        output_root=args.output_root,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps({
        "status": payload["status"],
        "active_seed": payload["active_seed"],
        "present_validated": payload["inventory"]["present_validated"],
        "formal_ranked": len(payload["formal_ranking"]),
        "json": payload["outputs"]["json"],
        "csv": payload["outputs"]["csv"],
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
