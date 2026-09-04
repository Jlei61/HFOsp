#!/usr/bin/env python3
"""Aggregate the rev22-DCI fit stage using patient-training inputs only.

The script inventories the exact candidate x fit-unit Cartesian product before it
computes any endpoint. Missing, unsafe, non-finite and low-yield units remain explicit
feasibility failures. Only candidates with four valid, conditionally estimable topology
units are emitted as continuous response-surface observations.

This module deliberately does not import the broader shaft-aware analysis module because
that module also owns validation-only clustering helpers. Instead, the already-frozen
training objective is loaded with a minimal numerical dependency shim containing only the
five embedding symbols that objective imports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_STAGE = DEFAULT_ROOT / "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"
DEFAULT_DESIGN = DEFAULT_STAGE / "response_design/response_design_manifest.json"
DEFAULT_SEEDS = DEFAULT_STAGE / "response_design/seed_manifest.json"
DEFAULT_QUALIFICATION = DEFAULT_STAGE / "objective_qualification/objective_qualification.json"
DEFAULT_WORKERS = DEFAULT_STAGE / "fit/workers"
DEFAULT_OUT = DEFAULT_STAGE / "fit/aggregate"

MIN_RETURNED_FAMILIES = 12
EXPECTED_FIT_UNITS = 4
MIN_POOLED_RETURNED_FAMILIES = MIN_RETURNED_FAMILIES * EXPECTED_FIT_UNITS
MIN_LOO_RETURNED_FAMILIES = MIN_RETURNED_FAMILIES * (EXPECTED_FIT_UNITS - 1)
COMPONENTS = ("D_support", "D_order", "D_lag", "D_cover")
CONDITIONAL_COMPONENTS = ("D_order", "D_lag")
UNCONDITIONAL_COMPONENTS = ("D_support", "D_cover")
SHAFT_ORDER = ("ICL", "SCL")
PAIR_CLASS_ORDER = ("ICL-ICL", "SCL-SCL", "ICL-SCL")
FORBIDDEN_PATH_MARKERS = (
    "heldout", "held_out", "held-out", "kmeans", "ood", "out_of_distribution",
    "fig5", "seizure", "patient_ictal",
)


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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(fd)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _guard_read_path(path: Path, role: str) -> Path:
    """Reject validation or ictal inputs before any file handle is opened."""
    resolved = Path(path).expanduser().resolve()
    normalized = str(resolved).lower().replace("interictal", "")
    if "ictal" in normalized or any(marker in normalized for marker in FORBIDDEN_PATH_MARKERS):
        raise RuntimeError(f"training-only boundary rejected {role}: {resolved}")
    return resolved


def _read_json(path: Path, role: str) -> tuple[dict, str]:
    path = _guard_read_path(path, role)
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    digest = _sha256(path)
    return json.loads(path.read_text(encoding="utf-8")), digest


def _verify_sha_sidecar(path: Path, digest: str, role: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar = _guard_read_path(sidecar, f"{role} SHA256 sidecar")
    if not sidecar.is_file():
        raise RuntimeError(f"frozen {role} has no SHA256 sidecar: {sidecar}")
    recorded = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if recorded != digest:
        raise RuntimeError(f"stale {role} SHA256 sidecar: {sidecar}")


def _fit_patient_embedding(features: np.ndarray, **_: Any) -> dict:
    raise RuntimeError("fit aggregation must use the frozen patient-training embedding")


def _transform_embedding(features: np.ndarray, embedding: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    center = np.asarray(embedding["center"], dtype=float)
    scale = np.asarray(embedding["scale"], dtype=float)
    components = np.asarray(embedding["components"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(center):
        raise ValueError("features do not match the frozen training embedding")
    return ((values - center) / scale) @ components.T


def _sliced_distance(features: np.ndarray, embedding: Mapping[str, Any], *,
                     reference_z: np.ndarray | None = None) -> float:
    z = _transform_embedding(features, embedding)
    reference = np.asarray(
        embedding["reference_z"] if reference_z is None else reference_z, dtype=float,
    )
    directions = np.asarray(embedding["directions"], dtype=float)
    if len(z) < 2:
        return float("nan")
    return float(np.mean([
        wasserstein_distance(z @ direction, reference @ direction)
        for direction in directions
    ]))


_OBJECTIVE_MODULE = None


def _load_training_objective():
    """Load the accepted objective without importing its validation-only sibling code."""
    global _OBJECTIVE_MODULE
    if _OBJECTIVE_MODULE is not None:
        return _OBJECTIVE_MODULE

    dependency_name = "src.topic4_shaft_aware"
    shim = types.ModuleType(dependency_name)
    shim.PAIR_CLASS_ORDER = PAIR_CLASS_ORDER
    shim.SHAFT_ORDER = SHAFT_ORDER
    shim.fit_patient_embedding = _fit_patient_embedding
    shim.sliced_event_cloud_distance = _sliced_distance
    shim.transform_patient_embedding = _transform_embedding

    objective_path = ROOT / "src/topic4_rev22_interictal_objective.py"
    spec = importlib.util.spec_from_file_location("_topic4_rev22_fit_objective", objective_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training objective: {objective_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(dependency_name)
    sys.modules[dependency_name] = shim
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(dependency_name, None)
        else:
            sys.modules[dependency_name] = previous
    _OBJECTIVE_MODULE = module
    return module


def _groups_and_pairs(contact_names: Sequence[str]) -> tuple[dict, dict]:
    names = [str(value) for value in contact_names]
    groups = {
        shaft: np.asarray([i for i, name in enumerate(names) if name.upper().startswith(shaft)], int)
        for shaft in SHAFT_ORDER
    }
    if any(len(groups[shaft]) == 0 for shaft in SHAFT_ORDER):
        raise RuntimeError("patient-training contact names must contain both ICL and SCL")
    if sum(len(indices) for indices in groups.values()) != len(names):
        raise RuntimeError("patient-training contact names contain an unsupported shaft")
    pairs = {
        "ICL-ICL": np.asarray(list(combinations(groups["ICL"], 2)), dtype=int).reshape((-1, 2)),
        "SCL-SCL": np.asarray(list(combinations(groups["SCL"], 2)), dtype=int).reshape((-1, 2)),
        "ICL-SCL": np.asarray(list(product(groups["ICL"], groups["SCL"])), dtype=int).reshape((-1, 2)),
    }
    if any(len(pairs[name]) == 0 for name in PAIR_CLASS_ORDER):
        raise RuntimeError("patient-training contract needs at least two contacts on each shaft")
    return groups, pairs


def _resolve_contract_path(qualification_path: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_absolute():
        return path
    return qualification_path.parent / path


def _load_training_contract(qualification_path: Path, qualification: Mapping[str, Any],
                            expected_hash: str) -> dict:
    recorded = qualification.get("patient_training_contract_npz")
    recorded_hash = qualification.get("patient_training_contract_sha256")
    if not recorded or recorded_hash != expected_hash:
        raise RuntimeError("objective qualification patient-training contract hash is inconsistent")
    path = _guard_read_path(_resolve_contract_path(qualification_path, str(recorded)),
                            "patient training contract")
    if not path.name.startswith("patient_training_contract"):
        raise RuntimeError("objective qualification did not point to a training-contract artifact")
    if not path.is_file() or _sha256(path) != expected_hash:
        raise RuntimeError("frozen patient-training contract is missing or changed")
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "feature_center", "feature_scale", "pca_components", "sw_directions",
            "reference_z", "reference_indices", "contact_names",
            "patient_train_onsets_ms", "patient_train_block_ids",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise RuntimeError(f"patient-training contract is missing arrays: {missing}")
        arrays = {name: np.asarray(loaded[name]) for name in required}
    if np.isinf(np.asarray(arrays["patient_train_onsets_ms"], float)).any():
        raise RuntimeError("patient-training onsets contain infinity")
    embedding = {
        "center": arrays["feature_center"],
        "scale": arrays["feature_scale"],
        "components": arrays["pca_components"],
        "directions": arrays["sw_directions"],
        "reference_z": arrays["reference_z"],
        "reference_indices": arrays["reference_indices"],
    }
    groups, pairs = _groups_and_pairs(arrays["contact_names"])
    return {
        "path": path,
        "sha256": expected_hash,
        "onsets_ms": np.asarray(arrays["patient_train_onsets_ms"], float),
        "block_ids": arrays["patient_train_block_ids"],
        "contact_names": [str(v) for v in arrays["contact_names"]],
        "embedding": embedding,
        "groups": groups,
        "pairs": pairs,
    }


def _git_is_ancestor(ancestor: str, descendant: str, repo_root: Path) -> bool:
    if not ancestor or not descendant:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(ancestor), str(descendant)],
        cwd=repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _declared_hash(payload: Mapping[str, Any], key: str) -> Any:
    alternatives = [
        payload.get(key),
        (payload.get("manifest") or {}).get(key),
        (payload.get("provenance") or {}).get(key),
    ]
    if key == "response_design_manifest_sha256":
        alternatives.append((payload.get("manifest") or {}).get("sha256"))
    present = [value for value in alternatives if value is not None]
    if not present:
        return None
    if len({str(value) for value in present}) != 1:
        return "CONFLICTING_DECLARATIONS"
    return str(present[0])


def _candidate_matches_worker(candidate: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    if payload.get("candidate_id") != candidate.get("candidate_id"):
        return False
    expected = candidate.get("mechanisms") or {}
    observed = payload.get("mechanism_freeze") or {}
    for key in ("g_EE", "g_EtoI", "ellipse_angle_deg", "ellipse_aspect_ratio"):
        if key in expected and not np.isclose(float(expected[key]), float(observed.get(key, np.nan)),
                                              rtol=0.0, atol=1e-12):
            return False
    field_hash = (candidate.get("node_field") or {}).get("field_sha256")
    if field_hash is not None and payload.get("field_sha256") != field_hash:
        return False
    return expected.get("Z_M") == "off" and observed.get("Z_M") == "off"


def _worker_paths(worker_dir: Path, candidate_id: str, topology_seed: int,
                  dynamics_seed: int) -> list[Path]:
    stems = [f"{candidate_id}_topo_{topology_seed}_dyn_{dynamics_seed}"]
    if topology_seed == dynamics_seed:
        stems.insert(0, f"{candidate_id}_seed_{topology_seed}")
    return [worker_dir / f"{stem}.json" for stem in stems if (worker_dir / f"{stem}.json").is_file()]


def _base_unit(candidate_id: str, topology_seed: int, dynamics_seed: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "topology_seed": int(topology_seed),
        "dynamics_seed": int(dynamics_seed),
        "inventory_status": "MISSING",
        "artifact_integrity": False,
        "safe": False,
        "event_yield_estimable": False,
        "conditional_estimable": False,
        "feasibility": False,
        "n_returned_families": 0,
        "failure_reasons": ["MISSING_WORKER_JSON"],
        "components": {name: None for name in COMPONENTS},
        "component_status": {name: None for name in COMPONENTS},
        "input_hashes": {},
    }


def _validate_and_score_unit(
    *, candidate: Mapping[str, Any], topology_seed: int, dynamics_seed: int,
    worker_dir: Path, response_hash: str, seed_hash: str,
    response_commit: str, objective_commit: str, training: Mapping[str, Any],
    objective: Any, ancestry_checker: Callable[[str, str, Path], bool], repo_root: Path,
) -> tuple[dict, np.ndarray | None]:
    cid = str(candidate["candidate_id"])
    row = _base_unit(cid, topology_seed, dynamics_seed)
    candidates = _worker_paths(worker_dir, cid, topology_seed, dynamics_seed)
    if not candidates:
        return row, None
    if len(candidates) != 1:
        row.update(inventory_status="DUPLICATE", failure_reasons=["DUPLICATE_WORKER_JSON"])
        return row, None

    json_path = _guard_read_path(candidates[0], "fit worker JSON")
    try:
        payload, json_hash = _read_json(json_path, "fit worker JSON")
    except Exception as error:
        row.update(inventory_status="INVALID_ARTIFACT", failure_reasons=[f"JSON_READ:{error}"])
        return row, None
    row["input_hashes"]["worker_json"] = {"path": str(json_path), "sha256": json_hash}
    failures = []
    if not str(payload.get("status", "")).endswith("_WORKER_COMPLETE"):
        failures.append("WORKER_STATUS_NOT_COMPLETE")
    if int(payload.get("topology_seed", -1)) != int(topology_seed):
        failures.append("TOPOLOGY_SEED_MISMATCH")
    if int(payload.get("dynamics_seed", -1)) != int(dynamics_seed):
        failures.append("DYNAMICS_SEED_MISMATCH")
    if not _candidate_matches_worker(candidate, payload):
        failures.append("CANDIDATE_OR_ZM_CONTRACT_MISMATCH")
    if _declared_hash(payload, "response_design_manifest_sha256") != response_hash:
        failures.append("RESPONSE_DESIGN_HASH_MISMATCH")
    if _declared_hash(payload, "seed_manifest_sha256") != seed_hash:
        failures.append("SEED_MANIFEST_HASH_MISMATCH")

    provenance = payload.get("provenance") or {}
    expected_commit = str(provenance.get("expected_git_commit") or "")
    runtime_commit = str(provenance.get("git_commit") or "")
    frozen_flag = provenance.get("runtime_modules_match_expected_commit")
    if type(frozen_flag) not in (bool, int) or frozen_flag != 1:
        failures.append("RUNTIME_MODULES_NOT_FROZEN")
    dirty_flag = provenance.get("runtime_modules_dirty")
    if type(dirty_flag) not in (bool, int) or dirty_flag != 0:
        failures.append("RUNTIME_MODULES_DIRTY_OR_UNKNOWN")
    if provenance.get("config_sha256") != provenance.get("config_sha256_at_expected_commit"):
        failures.append("WORKER_CONFIG_NOT_AT_EXPECTED_COMMIT")
    for ancestor, descendant, label in (
        (response_commit, expected_commit, "DESIGN_COMMIT_NOT_ANCESTOR"),
        (objective_commit, expected_commit, "OBJECTIVE_COMMIT_NOT_ANCESTOR"),
        (expected_commit, runtime_commit, "WORKER_COMMIT_ANCESTRY_FAIL"),
    ):
        if not ancestry_checker(ancestor, descendant, repo_root):
            failures.append(label)

    arrays = payload.get("arrays") or {}
    recorded_path = arrays.get("path")
    npz_path = Path(recorded_path) if recorded_path else json_path.with_suffix(".npz")
    if not npz_path.is_absolute():
        npz_path = json_path.parent / npz_path
    try:
        npz_path = _guard_read_path(npz_path, "fit worker NPZ")
        if npz_path.parent.resolve() != worker_dir.resolve():
            raise RuntimeError("worker NPZ is outside the frozen worker directory")
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        npz_hash = _sha256(npz_path)
        if arrays.get("sha256") != npz_hash:
            raise RuntimeError("worker NPZ hash mismatch")
        row["input_hashes"]["worker_npz"] = {"path": str(npz_path), "sha256": npz_hash}
        with np.load(npz_path, allow_pickle=False) as loaded:
            required = {
                "contact_names", "onsets", "event_returned", "topology_seed", "dynamics_seed",
                "active_fraction", "contact_envelope", "mechanism_parameters",
            }
            missing = sorted(required - set(loaded.files))
            if missing:
                raise RuntimeError(f"missing arrays {missing}")
            onsets = np.asarray(loaded["onsets"], float)
            returned = np.asarray(loaded["event_returned"], bool)
            contact_names = [str(value) for value in np.asarray(loaded["contact_names"])]
            npz_topology = int(np.asarray(loaded["topology_seed"]).item())
            npz_dynamics = int(np.asarray(loaded["dynamics_seed"]).item())
            for key in ("active_fraction", "contact_envelope", "mechanism_parameters"):
                if key in loaded.files and not np.isfinite(np.asarray(loaded[key], float)).all():
                    raise RuntimeError(f"non-finite required output {key}")
        if onsets.ndim != 2 or onsets.shape[1] != len(training["contact_names"]):
            raise RuntimeError("onset table does not match the training contact contract")
        if contact_names != training["contact_names"]:
            raise RuntimeError("worker contact order does not match the training contract")
        if returned.shape != (len(onsets),):
            raise RuntimeError("event_returned does not align with onsets")
        if np.isinf(onsets).any():
            raise RuntimeError("onsets contain infinity")
        if npz_topology != topology_seed or npz_dynamics != dynamics_seed:
            raise RuntimeError("NPZ seeds do not match the seed manifest")
        returned_onsets = onsets[returned]
        if len(returned_onsets) and not np.isfinite(returned_onsets).any(axis=1).all():
            raise RuntimeError("a returned family has no recruited contact onset")
    except Exception as error:
        failures.append(f"NPZ_INVALID:{error}")
        returned_onsets = None

    duration_ms = (payload.get("simulation") or {}).get("duration_ms")
    if duration_ms is None or not np.isclose(float(duration_ms), 20000.0):
        failures.append("SIMULATION_DURATION_NOT_20S")
    artifact_integrity = not failures
    runaway = (payload.get("simulation") or {}).get("runaway_early_stop_ms") is not None
    safe = bool(artifact_integrity and not runaway)
    n_returned = int(len(returned_onsets)) if returned_onsets is not None else 0
    yield_ok = bool(safe and n_returned >= MIN_RETURNED_FAMILIES)
    if runaway:
        failures.append("RUNAWAY")
    if safe and n_returned < MIN_RETURNED_FAMILIES:
        failures.append("LOW_RETURNED_FAMILY_YIELD")

    row.update(
        inventory_status="PRESENT_VALIDATED" if artifact_integrity else "INVALID_ARTIFACT",
        artifact_integrity=artifact_integrity,
        safe=safe,
        event_yield_estimable=yield_ok,
        n_returned_families=n_returned,
        failure_reasons=failures,
        worker_status=payload.get("status"),
        worker_expected_commit=expected_commit or None,
        worker_runtime_commit=runtime_commit or None,
        runaway_early_stop_ms=(payload.get("simulation") or {}).get("runaway_early_stop_ms"),
    )
    if not safe or returned_onsets is None:
        return row, None

    try:
        vector = objective.component_vector(
            returned_onsets, training["reference"], training["groups"], training["pairs"],
            training["embedding"], composite=False,
        )
    except Exception as error:
        row["failure_reasons"].append(f"PER_UNIT_COMPONENT_SCORING_FAILED:{error}")
        return row, returned_onsets
    statuses = {name: vector[name]["status"] for name in COMPONENTS}
    values = {name: vector[name]["value"] for name in COMPONENTS}
    conditional_ok = all(statuses[name] == objective.STATUS_OK for name in CONDITIONAL_COMPONENTS)
    row.update(
        conditional_estimable=conditional_ok,
        feasibility=bool(yield_ok and conditional_ok),
        components=values,
        component_status=statuses,
        clipping=vector["clipping"],
    )
    if not conditional_ok:
        row["failure_reasons"].append("CONDITIONAL_COMPONENT_NOT_ESTIMABLE")
    return row, returned_onsets


def _pooled_event_candidate(
    unit_rows: Sequence[Mapping[str, Any]],
    unit_onsets: Sequence[np.ndarray],
    *,
    training: Mapping[str, Any],
    objective: Any,
) -> tuple[dict, dict, list[dict]]:
    """Recompute nonlinear endpoints after pooling events over topology units.

    Per-unit endpoint means are not interchangeable with the registered candidate-level
    statistic.  Support probabilities, pair eligibility and coverage quantiles all change
    when event tables are concatenated, so both the point estimate and every jackknife
    replicate must be rescored from the corresponding pooled event table.
    """
    if len(unit_rows) != len(unit_onsets) or len(unit_rows) < 2:
        raise ValueError("pooled candidate needs aligned rows and at least two topology units")

    def score(tables: Sequence[np.ndarray], *, minimum_events: int) -> dict[str, float]:
        combined = np.concatenate([np.asarray(table, float) for table in tables], axis=0)
        if len(combined) < int(minimum_events):
            raise RuntimeError(
                f"pooled event yield {len(combined)} is below the registered minimum "
                f"{minimum_events}"
            )
        vector = objective.component_vector(
            combined,
            training["reference"],
            training["groups"],
            training["pairs"],
            training["embedding"],
            composite=False,
        )
        values = {}
        for component in COMPONENTS:
            if vector[component]["status"] != objective.STATUS_OK:
                raise RuntimeError(f"pooled {component} is not estimable")
            values[component] = float(vector[component]["value"])
        return values

    pooled = score(unit_onsets, minimum_events=MIN_POOLED_RETURNED_FAMILIES)
    leave_out = []
    for omitted, row in enumerate(unit_rows):
        record = {"omitted_topology_seed": int(row["topology_seed"])}
        record.update(score(
            [table for index, table in enumerate(unit_onsets) if index != omitted],
            minimum_events=MIN_LOO_RETURNED_FAMILIES,
        ))
        leave_out.append(record)
    jackknife = {}
    m = len(unit_rows)
    for component in COMPONENTS:
        array = np.asarray([row[component] for row in leave_out], float)
        jackknife[component] = float(np.sqrt((m - 1) / m * np.sum((array - array.mean()) ** 2)))
    return pooled, jackknife, leave_out


def _unit_csv_row(row: Mapping[str, Any]) -> dict:
    return {
        "candidate_id": row["candidate_id"],
        "topology_seed": row["topology_seed"],
        "dynamics_seed": row["dynamics_seed"],
        "inventory_status": row["inventory_status"],
        "artifact_integrity": row["artifact_integrity"],
        "safe": row["safe"],
        "event_yield_estimable": row["event_yield_estimable"],
        "conditional_estimable": row["conditional_estimable"],
        "feasibility": row["feasibility"],
        "n_returned_families": row["n_returned_families"],
        **{component: row["components"].get(component) for component in COMPONENTS},
        **{f"{component}_status": row["component_status"].get(component) for component in COMPONENTS},
        "failure_reasons": "|".join(row["failure_reasons"]),
    }


def aggregate_fit(
    *, response_design_path: Path, seed_manifest_path: Path,
    objective_qualification_path: Path, worker_dir: Path, output_dir: Path,
    repo_root: Path = ROOT,
    ancestry_checker: Callable[[str, str, Path], bool] = _git_is_ancestor,
) -> dict:
    response_design_path = _guard_read_path(response_design_path, "response design manifest")
    seed_manifest_path = _guard_read_path(seed_manifest_path, "seed manifest")
    objective_qualification_path = _guard_read_path(
        objective_qualification_path, "objective qualification",
    )
    worker_dir = _guard_read_path(worker_dir, "fit worker directory")
    output_dir = Path(output_dir).expanduser().resolve()

    design, design_hash = _read_json(response_design_path, "response design manifest")
    seeds, seed_hash = _read_json(seed_manifest_path, "seed manifest")
    qualification, qualification_hash = _read_json(
        objective_qualification_path, "objective qualification",
    )
    _verify_sha_sidecar(response_design_path, design_hash, "response design manifest")
    _verify_sha_sidecar(seed_manifest_path, seed_hash, "seed manifest")
    if seeds.get("response_design_manifest_sha256") != design_hash:
        raise RuntimeError("seed manifest does not bind the response design manifest")
    if qualification.get("status") != "OBJECTIVE_QUALIFIED" or qualification.get("smoke") is True:
        raise RuntimeError("fit aggregation requires the formal qualified training objective")
    if qualification.get("forbidden_inputs_loaded") is not False:
        raise RuntimeError("objective qualification crossed the training-only boundary")
    if tuple(qualification.get("components") or ()) != COMPONENTS:
        raise RuntimeError("objective qualification component contract drifted")

    candidates = list(design.get("candidates") or [])
    if len(candidates) != int(design.get("candidate_count", -1)):
        raise RuntimeError("response design candidate count is inconsistent")
    if len({row.get("candidate_id") for row in candidates}) != len(candidates):
        raise RuntimeError("response design contains duplicate candidate IDs")
    fit_units = list((seeds.get("fit") or {}).get("units") or [])
    fit_keys = [(int(row["topology_seed"]), int(row["dynamics_seed"])) for row in fit_units]
    if len(fit_keys) != EXPECTED_FIT_UNITS or len(set(fit_keys)) != EXPECTED_FIT_UNITS:
        raise RuntimeError("seed manifest must freeze exactly four unique fit units")
    if len({topology for topology, _ in fit_keys}) != EXPECTED_FIT_UNITS:
        raise RuntimeError("fit units must contain four distinct topology seeds")

    contract_hash = str(qualification.get("patient_training_contract_sha256") or "")
    training = _load_training_contract(objective_qualification_path, qualification, contract_hash)
    objective = _load_training_objective()
    training["reference"] = objective.patient_reference(
        training["onsets_ms"], training["groups"], training["pairs"], training["embedding"],
    )
    block_views = objective.PatientBlockViews(
        training["onsets_ms"], training["block_ids"], training["groups"], training["pairs"],
        training["embedding"], lag_cap_ms=float(objective.LAG_CAP_MS),
    )
    floor_contract = qualification.get("floors") or {}
    floor_draws = int(floor_contract.get("draws", 0))
    floor_seed = int(floor_contract.get("seed", -1))
    if floor_draws < 1 or floor_seed < 0:
        raise RuntimeError("objective qualification does not freeze a valid patient-floor contract")

    response_commit = str(design.get("git_commit") or "")
    objective_commit = str(qualification.get("git_commit") or "")
    if not response_commit or not objective_commit:
        raise RuntimeError("design and objective qualification must record git commits")

    all_units = []
    candidate_rows = []
    pooled_onsets_by_candidate = {}
    input_hashes = {
        "response_design_manifest": {"path": str(response_design_path), "sha256": design_hash},
        "seed_manifest": {"path": str(seed_manifest_path), "sha256": seed_hash},
        "objective_qualification": {
            "path": str(objective_qualification_path), "sha256": qualification_hash,
        },
        "patient_training_contract": {"path": str(training["path"]), "sha256": contract_hash},
        "training_objective_module": {
            "path": str(ROOT / "src/topic4_rev22_interictal_objective.py"),
            "sha256": _sha256(ROOT / "src/topic4_rev22_interictal_objective.py"),
        },
        "workers": {},
    }

    for candidate in candidates:
        unit_rows = []
        unit_onsets = []
        for topology_seed, dynamics_seed in fit_keys:
            row, returned_onsets = _validate_and_score_unit(
                candidate=candidate, topology_seed=topology_seed, dynamics_seed=dynamics_seed,
                worker_dir=worker_dir, response_hash=design_hash, seed_hash=seed_hash,
                response_commit=response_commit, objective_commit=objective_commit,
                training=training, objective=objective, ancestry_checker=ancestry_checker,
                repo_root=repo_root,
            )
            unit_rows.append(row)
            unit_onsets.append(returned_onsets)
            all_units.append(row)
            unit_key = f"{candidate['candidate_id']}::{topology_seed}::{dynamics_seed}"
            input_hashes["workers"][unit_key] = row["input_hashes"] or None

        all_four_safe = len(unit_rows) == EXPECTED_FIT_UNITS and all(
            row["artifact_integrity"] and row["safe"] for row in unit_rows
        ) and all(table is not None for table in unit_onsets)
        candidate_record = {
            "candidate_id": candidate["candidate_id"],
            "block": candidate.get("block"),
            "physical": candidate.get("physical"),
            "family_membership": candidate.get("family_membership"),
            "expected_fit_units": EXPECTED_FIT_UNITS,
            "observed_artifact_units": int(sum(row["artifact_integrity"] for row in unit_rows)),
            "safe_units": int(sum(row["safe"] for row in unit_rows)),
            "yield_estimable_units": int(sum(row["event_yield_estimable"] for row in unit_rows)),
            "conditional_estimable_units": int(sum(row["conditional_estimable"] for row in unit_rows)),
            "joint_feasibility": False,
            "continuous_surface_eligible": False,
            "pooled_candidate": None,
            "floor": {component: None for component in COMPONENTS},
            "standardized_Z": {component: None for component in COMPONENTS},
            "normalized_excess_E": {component: None for component in COMPONENTS},
            "jackknife_sd": {component: None for component in COMPONENTS},
            "standardized_jackknife_sd": {component: None for component in COMPONENTS},
            "leave_one_topology_out": [],
            "candidate_failure_reasons": [],
            "units": unit_rows,
        }
        if all_four_safe:
            try:
                pooled, jackknife, leave_out = _pooled_event_candidate(
                    unit_rows,
                    unit_onsets,
                    training=training,
                    objective=objective,
                )
            except RuntimeError as error:
                candidate_record["candidate_failure_reasons"].append(
                    f"POOLED_OR_LOO_NOT_ESTIMABLE:{error}"
                )
            else:
                candidate_record["joint_feasibility"] = True
                candidate_record["continuous_surface_eligible"] = True
                candidate_record["pooled_candidate"] = {
                    "method": (
                        "concatenate_four_topology_event_tables_then_recompute_components; "
                        "all leave-one-topology-out pooled replicates re-estimable"
                    ),
                    "n_pooled_events": int(sum(
                        len(table) for table in unit_onsets if table is not None
                    )),
                    "components": pooled,
                }
                candidate_record["jackknife_sd"] = jackknife
                candidate_record["leave_one_topology_out"] = leave_out
                pooled_table = np.concatenate(
                    [np.asarray(table, float) for table in unit_onsets if table is not None], axis=0,
                )
                pooled_onsets_by_candidate[str(candidate["candidate_id"])] = pooled_table
        else:
            candidate_record["candidate_failure_reasons"].append(
                "MISSING_UNSAFE_OR_NONFINITE_FIT_UNIT"
            )
        candidate_rows.append(candidate_record)

    # Patient floors are candidate specific. Support and coverage need only the pooled
    # event count, whereas order and physical lag also reproduce the candidate's
    # recruitment censoring before the patient split-half score is calculated.
    floor_requests = []
    unconditional_keys = set()
    for row in candidate_rows:
        if not row["joint_feasibility"]:
            continue
        candidate_id = str(row["candidate_id"])
        onsets = pooled_onsets_by_candidate[candidate_id]
        n_events = int(len(onsets))
        unconditional_key = f"n{n_events}"
        if unconditional_key not in unconditional_keys:
            floor_requests.append({
                "key": unconditional_key,
                "n": n_events,
                "components": UNCONDITIONAL_COMPONENTS,
                "thin_profile": None,
            })
            unconditional_keys.add(unconditional_key)
        floor_requests.append({
            "key": f"candidate::{candidate_id}",
            "n": n_events,
            "components": CONDITIONAL_COMPONENTS,
            "thin_profile": objective.recruitment_profile(onsets),
        })

    floors = objective.block_split_floors(
        block_views,
        floor_requests,
        draws=floor_draws,
        seed=floor_seed,
        n_pair_min=int(objective.N_PAIR_MIN),
        cover_quantile=0.90,
    ) if floor_requests else {}

    surface_rows = []
    for row in candidate_rows:
        if not row["joint_feasibility"]:
            continue
        candidate_id = str(row["candidate_id"])
        n_events = int(row["pooled_candidate"]["n_pooled_events"])
        raw = row["pooled_candidate"]["components"]
        normalization_ok = True
        for component in COMPONENTS:
            floor_key = (
                f"candidate::{candidate_id}"
                if component in CONDITIONAL_COMPONENTS else f"n{n_events}"
            )
            floor = (floors.get(floor_key) or {}).get(component)
            row["floor"][component] = floor
            finite_floor = bool(
                floor is not None
                and floor.get("q50") is not None
                and floor.get("q95") is not None
                and np.isfinite(float(floor["q50"]))
                and np.isfinite(float(floor["q95"]))
                and float(floor["q95"]) > float(floor["q50"])
            )
            if not finite_floor:
                normalization_ok = False
                row["candidate_failure_reasons"].append(
                    f"{component}_PATIENT_FLOOR_NOT_ESTIMABLE"
                )
                continue
            row["standardized_Z"][component] = objective.standardized_excess(
                raw[component], floor,
            )
            row["normalized_excess_E"][component] = objective.normalized_excess(
                raw[component], floor,
            )
            denominator = float(floor["q95"]) - float(floor["q50"]) + 1e-9
            row["standardized_jackknife_sd"][component] = float(
                row["jackknife_sd"][component] / denominator
            )
        row["continuous_surface_eligible"] = bool(normalization_ok)
        if not normalization_ok:
            continue
        for component in COMPONENTS:
            surface_rows.append({
                "candidate_id": candidate_id,
                "component": component,
                "raw_D": raw[component],
                "standardized_Z": row["standardized_Z"][component],
                "normalized_excess_E": row["normalized_excess_E"][component],
                "raw_jackknife_sd": row["jackknife_sd"][component],
                "standardized_jackknife_sd": row["standardized_jackknife_sd"][component],
                "floor": row["floor"][component],
                "physical": row.get("physical"),
                "family_membership": row.get("family_membership"),
            })

    expected_count = len(candidates) * EXPECTED_FIT_UNITS
    all_artifacts_valid = len(all_units) == expected_count and all(
        row["artifact_integrity"] for row in all_units
    )
    payload = {
        "schema_id": "topic4_rev22_dci_training_only_fit_aggregate_v2",
        "status": "FIT_AGGREGATE_COMPLETE" if all_artifacts_valid else "FIT_ARTIFACT_GRID_INCOMPLETE",
        "scientific_role": "training_only_component_response_fit_input",
        "training_only": True,
        "forbidden_inputs_loaded": False,
        "component_contract": list(COMPONENTS),
        "candidate_statistic": (
            "events pooled over the four topology units before nonlinear components are recomputed; "
            "candidate-specific count-matched and recruitment-thinned patient floors convert raw D "
            "to standardized Z and clipped normalized excess E; per-unit components are sidecars"
        ),
        "patient_floor_contract": {
            "method": floor_contract.get("kind"),
            "draws": floor_draws,
            "seed": floor_seed,
            "unconditional_components": list(UNCONDITIONAL_COMPONENTS),
            "conditional_components": list(CONDITIONAL_COMPONENTS),
            "conditional_thinning": "candidate_pooled_recruitment_profile",
            "cover_patient_query": "all_events_in_reference_block_half",
        },
        "eligibility_contract": {
            "expected_fit_units_per_candidate": EXPECTED_FIT_UNITS,
            "per_unit_low_yield_sidecar_threshold": MIN_RETURNED_FAMILIES,
            "min_pooled_returned_families": MIN_POOLED_RETURNED_FAMILIES,
            "min_leave_one_topology_out_returned_families": MIN_LOO_RETURNED_FAMILIES,
            "requires_safe_finite_nonrunaway": True,
            "requires_D_order_and_D_lag_estimable_in_pooled_and_every_leave_one_out": True,
            "failed_units_are_retained": True,
        },
        "inventory": {
            "candidate_count": len(candidates),
            "expected_unit_count": expected_count,
            "unit_rows": len(all_units),
            "artifact_integrity_count": int(sum(row["artifact_integrity"] for row in all_units)),
            "joint_feasibility_count": int(sum(row["joint_feasibility"] for row in candidate_rows)),
            "continuous_surface_candidate_count": int(sum(
                row["continuous_surface_eligible"] for row in candidate_rows
            )),
        },
        "input_hashes": input_hashes,
        "candidates": candidate_rows,
        "continuous_component_surface": surface_rows,
        "claim_boundary": (
            "Patient-training component aggregation only; no validation endpoint, model label, "
            "patient held-out artifact or patient seizure artifact is read."
        ),
    }

    _atomic_json(output_dir / "fit_aggregate.json", payload)
    unit_fields = [
        "candidate_id", "topology_seed", "dynamics_seed", "inventory_status",
        "artifact_integrity", "safe", "event_yield_estimable", "conditional_estimable",
        "feasibility", "n_returned_families", *COMPONENTS,
        *[f"{component}_status" for component in COMPONENTS], "failure_reasons",
    ]
    _atomic_csv(output_dir / "fit_unit_inventory.csv", [_unit_csv_row(row) for row in all_units], unit_fields)
    candidate_csv = []
    for row in candidate_rows:
        pooled = (row.get("pooled_candidate") or {}).get("components") or {}
        candidate_csv.append({
            "candidate_id": row["candidate_id"],
            "block": row["block"],
            "artifact_units": row["observed_artifact_units"],
            "safe_units": row["safe_units"],
            "yield_estimable_units": row["yield_estimable_units"],
            "conditional_estimable_units": row["conditional_estimable_units"],
            "joint_feasibility": row["joint_feasibility"],
            "continuous_surface_eligible": row["continuous_surface_eligible"],
            **{component: pooled.get(component) for component in COMPONENTS},
            **{f"Z_{component}": row["standardized_Z"].get(component) for component in COMPONENTS},
            **{f"E_{component}": row["normalized_excess_E"].get(component) for component in COMPONENTS},
            **{f"JK_{component}": row["jackknife_sd"].get(component) for component in COMPONENTS},
            **{
                f"JK_Z_{component}": row["standardized_jackknife_sd"].get(component)
                for component in COMPONENTS
            },
            "candidate_failure_reasons": "|".join(row["candidate_failure_reasons"]),
        })
    candidate_fields = [
        "candidate_id", "block", "artifact_units", "safe_units", "yield_estimable_units",
        "conditional_estimable_units", "joint_feasibility", "continuous_surface_eligible",
        *COMPONENTS, *[f"Z_{component}" for component in COMPONENTS],
        *[f"E_{component}" for component in COMPONENTS],
        *[f"JK_{component}" for component in COMPONENTS],
        *[f"JK_Z_{component}" for component in COMPONENTS], "candidate_failure_reasons",
    ]
    _atomic_csv(output_dir / "fit_candidate_components.csv", candidate_csv, candidate_fields)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-design-manifest", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--objective-qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = aggregate_fit(
        response_design_path=args.response_design_manifest,
        seed_manifest_path=args.seed_manifest,
        objective_qualification_path=args.objective_qualification,
        worker_dir=args.worker_dir,
        output_dir=args.out_dir,
    )
    print(json.dumps({
        "status": payload["status"],
        "candidates": payload["inventory"]["candidate_count"],
        "expected_units": payload["inventory"]["expected_unit_count"],
        "surface_candidates": payload["inventory"]["continuous_surface_candidate_count"],
        "output": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
