#!/usr/bin/env python3
"""Freeze rev14 training-only patient-support floors and OOD labels.

The producer opens exactly three scientific inputs: the frozen patient training
target, the old-A/B training-only direction classifier, and the contact
contract.  It does not import or run the SNN and cannot score/select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev14_patient_support import (  # noqa: E402
    PRIMARY_LABEL_KEY,
    build_patient_support_calibration,
    patient_training_from_mapping,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import (  # noqa: E402
    assign_direction_modes,
    fit_direction_classifier,
)


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev14_patient_support_acceptance.json"
ALLOWED_INPUTS = {
    "patient_training_target",
    "old_ab_train_only_classifier",
    "contact_contract",
}
TARGET_KEYS = (
    "contact_names",
    "shaft_ids",
    "patient_train_onsets",
    "patient_train_old_labels",
    "patient_train_block_ids",
    "feature_center",
    "feature_scale",
    "pca_components",
)
CLASSIFIER_KEYS = (
    "coef",
    "intercept",
    "class_centers",
    "class_precisions",
    "ood_distance_thresholds",
    "ood_quantile",
    "regularization_c",
    "n_train",
)
CLASSIFIER_PARAMETER_KEYS = (
    "coef",
    "intercept",
    "class_centers",
    "class_precisions",
    "ood_distance_thresholds",
    "ood_quantile",
    "regularization_c",
    "n_train",
)
TRAINING_CV_SUMMARY_KEYS_NOT_ACCESSED = {
    "heldout_balanced_accuracy",
    "heldout_roc_auc",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = _canonical_json_bytes({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    })
    return hashlib.sha256(header + array.view(np.uint8).tobytes()).hexdigest()


def _resolve(path: str | Path, artifact_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (artifact_root / value).resolve()


def _file_record(path: Path, *, role: str, display_path: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": display_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _assert_no_forbidden_name(name: str, *, context: str) -> None:
    lowered = str(name).lower().replace("-", "_")
    if "ictal" in lowered or "heldout" in lowered or "held_out" in lowered:
        raise RuntimeError(f"forbidden patient field in {context}: {name}")


def _skip_json_string(text: str, start: int) -> int:
    if text[start] != '"':
        raise ValueError("JSON string must start with a quote")
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
    if start >= len(text):
        raise ValueError("missing JSON value")
    character = text[start]
    if character == '"':
        return _skip_json_string(text, start)
    if character in "[{":
        pairs = {"[": "]", "{": "}"}
        stack = [pairs[character]]
        index = start + 1
        while index < len(text) and stack:
            current = text[index]
            if current == '"':
                index = _skip_json_string(text, index)
                continue
            if current in pairs:
                stack.append(pairs[current])
            elif current == stack[-1]:
                stack.pop()
            index += 1
        if stack:
            raise ValueError("unterminated JSON container")
        return index
    index = start
    while index < len(text) and text[index] not in ",}]\r\n\t ":
        index += 1
    return index


def _json_object_members(text: str):
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        raise ValueError("expected a JSON object")
    index += 1
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == "}":
            return
        key_end = _skip_json_string(text, index)
        key = json.loads(text[index:key_end])
        index = key_end
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != ":":
            raise ValueError("JSON object key lacks a value")
        value_start = index + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        value_end = _skip_json_value(text, value_start)
        yield str(key), value_start, value_end
        index = value_end
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return
        raise ValueError("malformed JSON object")


def _load_classifier_inference(path: Path, accessed_keys: tuple[str, ...]) -> dict:
    """Decode only inference parameters, never CV summaries or other payloads."""
    text = path.read_text()
    direction_text = None
    for key, start, end in _json_object_members(text):
        _assert_no_forbidden_name(key, context="classifier top level")
        if key == "direction_classifier":
            direction_text = text[start:end]
    if direction_text is None:
        raise RuntimeError("classifier manifest lacks direction_classifier")

    selected: dict[str, Any] = {}
    observed_keys: set[str] = set()
    for key, start, end in _json_object_members(direction_text):
        observed_keys.add(key)
        if key in TRAINING_CV_SUMMARY_KEYS_NOT_ACCESSED:
            continue
        _assert_no_forbidden_name(key, context="direction classifier")
        if key in accessed_keys:
            selected[key] = json.loads(direction_text[start:end])
    missing = set(accessed_keys).difference(selected)
    if missing:
        raise RuntimeError(f"classifier lacks inference keys: {sorted(missing)}")
    for key in ("coef", "class_centers", "class_precisions", "ood_distance_thresholds"):
        selected[key] = np.asarray(selected[key], dtype=np.float64)
    selected["_manifest_direction_keys"] = sorted(observed_keys)
    return selected


def _load_training_target(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        for key in loaded.files:
            _assert_no_forbidden_name(key, context="patient training target")
        missing = set(TARGET_KEYS).difference(loaded.files)
        if missing:
            raise RuntimeError(f"patient training target lacks keys: {sorted(missing)}")
        return {key: np.asarray(loaded[key]) for key in TARGET_KEYS}


def _load_contact_contract(path: Path) -> dict:
    payload = json.loads(path.read_text())

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                current = f"{prefix}.{key}" if prefix else str(key)
                _assert_no_forbidden_name(str(key), context=f"contact contract {current}")
                visit(item, current)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(payload)
    return payload


def _verify_contact_order(target: Mapping[str, np.ndarray], contract: Mapping) -> None:
    rows = sorted(contract["contacts"], key=lambda row: int(row["contact_index"]))
    contract_names = np.asarray([str(row["contact_name"]) for row in rows])
    contract_shafts = np.asarray([str(row["shaft_id"]).upper() for row in rows])
    if not np.array_equal(contract_names, np.asarray(target["contact_names"]).astype(str)):
        raise RuntimeError("patient target and contact contract contact order differ")
    if not np.array_equal(contract_shafts, np.asarray(target["shaft_ids"]).astype(str)):
        raise RuntimeError("patient target and contact contract shaft identities differ")


def _old_label_conditioned_ood(
        embedding: np.ndarray, old_labels: np.ndarray,
        classifier: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic only: condition distance on frozen old A/B rather than assignment."""
    values = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(old_labels, dtype=np.int8)
    centers = np.asarray(classifier["class_centers"], dtype=np.float64)
    precisions = np.asarray(classifier["class_precisions"], dtype=np.float64)
    thresholds = np.asarray(
        classifier["ood_distance_thresholds"], dtype=np.float64,
    )
    if values.ndim != 2 or labels.shape != (len(values),):
        raise RuntimeError("training embedding and old A/B labels do not align")
    if np.any(~np.isin(labels, (0, 1))):
        raise RuntimeError("training old A/B labels must remain binary")
    distances = np.empty(len(values), dtype=np.float64)
    for mode in (0, 1):
        selected = labels == mode
        delta = values[selected] - centers[mode]
        distances[selected] = np.einsum(
            "ni,ij,nj->n", delta, precisions[mode], delta,
        )
    return distances, distances > thresholds[labels]


def _refit_and_verify_classifier(
        target: Mapping[str, np.ndarray], contract: Mapping[str, Any],
        embedding: Mapping[str, np.ndarray], frozen: Mapping[str, Any],
        classifier_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the classifier from the frozen training split and compare all parameters."""
    n_splits = int(classifier_config["refit_n_splits"])
    atol = float(classifier_config["refit_atol"])
    rtol = float(classifier_config["refit_rtol"])
    if n_splits < 2:
        raise RuntimeError("classifier refit_n_splits must be at least two")
    if atol < 0.0 or atol > 1e-12 or rtol != 0.0:
        raise RuntimeError("classifier refit tolerance is not strict")
    refit = fit_direction_classifier(
        np.asarray(target["patient_train_onsets"], dtype=np.float64),
        np.asarray(target["patient_train_old_labels"], dtype=np.int8),
        np.asarray(target["patient_train_block_ids"]),
        groups=contract_groups(contract),
        embedding=embedding,
        n_splits=n_splits,
        regularization_c=float(frozen["regularization_c"]),
        ood_quantile=float(frozen["ood_quantile"]),
    )
    comparison: dict[str, Any] = {}
    for key in CLASSIFIER_PARAMETER_KEYS:
        observed = np.asarray(frozen[key])
        rebuilt = np.asarray(refit[key])
        if observed.shape != rebuilt.shape:
            raise RuntimeError(f"classifier refit parameter shape differs: {key}")
        if key == "n_train":
            matches = bool(np.array_equal(observed, rebuilt))
            maximum_error = float(abs(int(observed) - int(rebuilt)))
        else:
            observed_float = observed.astype(np.float64)
            rebuilt_float = rebuilt.astype(np.float64)
            finite = bool(
                np.all(np.isfinite(observed_float))
                and np.all(np.isfinite(rebuilt_float))
            )
            matches = finite and bool(np.allclose(
                observed_float, rebuilt_float, atol=atol, rtol=rtol,
            ))
            maximum_error = (
                float(np.max(np.abs(observed_float - rebuilt_float)))
                if observed_float.size else 0.0
            )
        comparison[key] = {
            "matches": matches,
            "max_abs_error": maximum_error,
            "frozen_sha256": _array_sha256(observed),
            "refit_sha256": _array_sha256(rebuilt),
        }
        if not matches:
            raise RuntimeError(f"classifier refit parameter differs: {key}")
    return refit, {
        "n_splits": n_splits,
        "atol": atol,
        "rtol": rtol,
        "parameters": comparison,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _calibration_public_mapping(calibration: Any) -> dict[str, Any]:
    """Prefer a calibration's public serializer; fall back to dataclass fields."""
    for name in ("to_manifest", "to_dict", "as_dict"):
        accessor = getattr(calibration, name, None)
        if callable(accessor):
            value = accessor()
            if not isinstance(value, Mapping):
                raise RuntimeError(f"calibration {name}() did not return a mapping")
            return dict(value)
    if not is_dataclass(calibration):
        raise RuntimeError("patient-support calibration has no public serialization contract")
    return {field.name: getattr(calibration, field.name) for field in fields(calibration)}


def _calibration_artifact_parts(
        calibration: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, str]]]:
    """Split public calibration metadata from distribution and block-audit arrays."""
    public = _calibration_public_mapping(calibration)
    candidate_names = set(public)
    if is_dataclass(calibration):
        candidate_names.update(field.name for field in fields(calibration))
    distribution_names = {
        name for name in candidate_names if name.endswith("_distributions")
    }
    arrays: dict[str, np.ndarray] = {}
    indices: dict[str, dict[str, str]] = {}
    for name in sorted(distribution_names):
        value = getattr(calibration, name, public.get(name))
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise RuntimeError(f"calibration distribution field is not a mapping: {name}")
        indices[name] = {}
        for endpoint, draws in sorted(value.items()):
            array = np.asarray(draws, dtype=np.float64)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise RuntimeError(
                    f"calibration distribution is not a finite vector: {name}.{endpoint}"
                )
            key = (
                _floor_array_key(str(endpoint))
                if name == "floor_distributions"
                else "calibration__" + name + "__" + str(endpoint).replace(
                    ".", "__"
                ).replace("-", "_")
            )
            if key in arrays:
                raise RuntimeError(f"duplicate calibration array key: {key}")
            arrays[key] = array
            indices[name][str(endpoint)] = key
    array_field_names: set[str] = set()
    for name in sorted(candidate_names.difference(distribution_names)):
        value = getattr(calibration, name, public.get(name))
        if not isinstance(value, np.ndarray):
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise RuntimeError(f"calibration array uses object dtype: {name}")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise RuntimeError(f"calibration array is non-finite: {name}")
        key = "calibration__" + name
        if key in arrays:
            raise RuntimeError(f"duplicate calibration array key: {key}")
        arrays[key] = array
        indices[name] = {"__self__": key}
        array_field_names.add(name)
    metadata = {
        str(key): _jsonable(value)
        for key, value in public.items()
        if key not in distribution_names and key not in array_field_names
    }
    return metadata, arrays, indices


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[key]), allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_deterministic_npz_bytes(arrays))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _runtime_provenance(config_path: Path) -> dict[str, Any]:
    import sklearn

    runtime_paths = [
        Path(__file__).resolve(),
        ROOT / "src/topic4_rev14_patient_support.py",
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/topic4_shaft_aware_direction.py",
    ]
    path_records = []
    for path in runtime_paths:
        relative = str(path.relative_to(ROOT))
        status = _git_output("status", "--short", "--", relative)
        path_records.append({
            **_file_record(path, role="runtime_source", display_path=relative),
            "git_status": status,
            "dirty_or_untracked": bool(status),
        })
    config_relative = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path)
    config_status = _git_output("status", "--short", "--", config_relative)
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "config": {
            **_file_record(config_path, role="execution_contract", display_path=config_relative),
            "git_status": config_status,
            "dirty_or_untracked": bool(config_status),
        },
        "runtime_sources": path_records,
        "runtime_sources_dirty_or_untracked": bool(
            config_status or any(row["dirty_or_untracked"] for row in path_records)
        ),
    }


def _execution_source_paths(config_path: Path) -> dict[str, Path]:
    return {
        "config": config_path.resolve(),
        "freezer": Path(__file__).resolve(),
        "patient_support": (ROOT / "src/topic4_rev14_patient_support.py").resolve(),
        "shaft_aware": (ROOT / "src/topic4_shaft_aware.py").resolve(),
        "direction_classifier": (
            ROOT / "src/topic4_shaft_aware_direction.py"
        ).resolve(),
    }


def _source_snapshot(
        scientific_paths: Mapping[str, Path], config_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        **{f"input:{role}": path.resolve() for role, path in scientific_paths.items()},
        **{
            f"runtime:{role}": path
            for role, path in _execution_source_paths(config_path).items()
        },
    }
    return {
        role: {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for role, path in sorted(paths.items())
    }


def _assert_source_snapshot_unchanged(
        snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    for role, record in snapshot.items():
        path = Path(str(record["path"]))
        if not path.is_file():
            raise RuntimeError(f"freeze source disappeared during build: {role}")
        if (int(path.stat().st_size) != int(record["size_bytes"])
                or _sha256(path) != str(record["sha256"])):
            raise RuntimeError(f"freeze source drifted during build: {role}")


def _floor_array_key(endpoint: str) -> str:
    return "floor_draws__" + endpoint.replace(".", "__").replace("-", "_")


def freeze_acceptance(
        *, artifact_root: Path, config_path: Path,
        output_dir_override: Path | None = None) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev14_patient_support_acceptance_config_v2":
        raise RuntimeError("patient-support config schema changed")
    if config.get("scientific_role") != "training_only_patient_support_acceptance":
        raise RuntimeError("patient-support scientific role changed")
    if set(config.get("inputs", {})) != ALLOWED_INPUTS:
        raise RuntimeError("patient-support freezer accepts exactly three scientific inputs")
    if config.get("classifier", {}).get("primary_label_key") != PRIMARY_LABEL_KEY:
        raise RuntimeError("patient-support primary label is not old A/B")
    if config.get("classifier", {}).get("ood_reference_label") != (
            "classifier_assigned_labels"):
        raise RuntimeError("patient-support OOD must use classifier-assigned class")
    if config.get("classifier", {}).get("replace_primary_labels") is not False:
        raise RuntimeError("classifier predictions must not replace old A/B labels")
    accessed_keys = tuple(str(key) for key in config["classifier"]["accessed_keys"])
    if accessed_keys != CLASSIFIER_KEYS:
        raise RuntimeError("classifier inference-key allowlist changed")

    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for role in sorted(ALLOWED_INPUTS):
        record = config["inputs"][role]
        path = _resolve(record["path"], artifact_root)
        if _sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"frozen input hash changed: {role}")
        paths[role] = path
        records[role] = _file_record(
            path, role=role, display_path=str(record["path"]),
        )
    source_snapshot = _source_snapshot(paths, config_path)

    target = _load_training_target(paths["patient_training_target"])
    contract = _load_contact_contract(paths["contact_contract"])
    _verify_contact_order(target, contract)
    classifier = _load_classifier_inference(
        paths["old_ab_train_only_classifier"], accessed_keys,
    )
    if config["classifier"].get("semantics") != "FULL_TIMING":
        raise RuntimeError("old A/B classifier must retain FULL_TIMING semantics")
    embedding = {
        "center": np.asarray(target["feature_center"], dtype=np.float64),
        "scale": np.asarray(target["feature_scale"], dtype=np.float64),
        "components": np.asarray(target["pca_components"], dtype=np.float64),
    }
    refit_classifier, refit_audit = _refit_and_verify_classifier(
        target, contract, embedding, classifier, config["classifier"],
    )
    assigned = assign_direction_modes(
        np.asarray(target["patient_train_onsets"], dtype=np.float64),
        groups=contract_groups(contract), embedding=embedding,
        classifier=refit_classifier,
    )
    old_labels = np.asarray(target["patient_train_old_labels"], dtype=np.int8)
    assigned_labels = np.asarray(assigned["labels"], dtype=np.int8)
    patient_train_ood_distance = np.asarray(
        assigned["ood_distance"], dtype=np.float64,
    )
    patient_train_ood = np.asarray(assigned["ood"], dtype=bool)
    old_label_ood_distance, old_label_ood = _old_label_conditioned_ood(
        np.asarray(assigned["embedding"], dtype=np.float64), old_labels,
        refit_classifier,
    )
    mapping = {
        "contact_names": np.asarray(target["contact_names"]).astype(str),
        "shaft_ids": np.asarray(target["shaft_ids"]).astype(str),
        "patient_train_onsets": np.asarray(
            target["patient_train_onsets"], dtype=np.float64,
        ),
        PRIMARY_LABEL_KEY: old_labels,
        "patient_train_classifier_labels": assigned_labels,
        "patient_train_block_ids": np.asarray(target["patient_train_block_ids"]),
        "patient_train_ood": patient_train_ood,
        "primary_label_key": PRIMARY_LABEL_KEY,
        "source_sha256": records["patient_training_target"]["sha256"],
    }
    patient = patient_training_from_mapping(
        mapping,
        expected_source_sha256=records["patient_training_target"]["sha256"],
    )
    floor = config["floor"]
    calibration = build_patient_support_calibration(
        patient,
        sample_size=int(floor["sample_size_per_side"]),
        floor_draws=int(floor["draws"]),
        joint_draws=int(floor["joint_draws"]),
        joint_inner_draws=int(floor["joint_inner_draws"]),
        seed=int(floor["seed"]),
        tie_tolerance=float(floor["tie_tolerance"]),
    )
    if int(floor["draws"]) != 4096:
        raise RuntimeError("formal patient-support floor must use 4096 draws")
    if float(floor["quantile"]) != 0.95 or floor["quantile_method"] != "linear":
        raise RuntimeError("patient-support q95 definition changed")

    calibration_metadata, calibration_arrays, calibration_array_keys = (
        _calibration_artifact_parts(calibration)
    )
    floor_keys = calibration_array_keys.get("floor_distributions", {})
    arrays: dict[str, np.ndarray] = {
        "contact_names": np.asarray(patient.contact_names),
        "shaft_ids": np.asarray(patient.shaft_ids),
        "patient_train_classifier_labels": assigned_labels,
        "patient_train_ood": patient_train_ood,
        "patient_train_ood_distance": patient_train_ood_distance,
        "patient_train_old_label_conditioned_ood_diagnostic": old_label_ood,
        "patient_train_old_label_conditioned_ood_distance_diagnostic": (
            old_label_ood_distance
        ),
        "patient_train_probability_B_diagnostic": np.asarray(
            assigned["probability_B"], dtype=np.float64,
        ),
    }
    arrays.update(calibration_arrays)

    output_config = config["output"]
    output_dir = (
        _resolve(output_dir_override, artifact_root)
        if output_dir_override is not None
        else _resolve(output_config["directory"], artifact_root)
    )
    manifest_path = output_dir / str(output_config["manifest"])
    npz_path = output_dir / str(output_config["npz"])
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError("patient-support output path is not a directory")
        observed_names = {path.name for path in output_dir.iterdir()}
        expected_names = {manifest_path.name, npz_path.name}
        if observed_names != expected_names:
            raise RuntimeError("patient-support freeze is partial or has unexpected files")
    npz_bytes = _deterministic_npz_bytes(arrays)
    npz_sha = hashlib.sha256(npz_bytes).hexdigest()

    confusion = np.zeros((2, 2), dtype=np.int64)
    for old, predicted in zip(old_labels, assigned_labels):
        confusion[int(old), int(predicted)] += 1
    manifest = {
        "schema_id": "topic4_rev14_patient_support_acceptance_manifest_v2",
        "status": "PATIENT_TRAINING_SUPPORT_ACCEPTANCE_FROZEN",
        "subject": config["subject"],
        "claim_boundary": (
            "Training-only interictal patient support calibration. It contains no "
            "patient evaluation split, seizure data, model artifact, candidate score, "
            "candidate selection, or simulation output."
        ),
        "input_snapshot": {
            "files": [records[role] for role in sorted(records)],
            "source_set_sha256": _canonical_sha256(
                [records[role] for role in sorted(records)]
            ),
            "allowed_roles": sorted(ALLOWED_INPUTS),
        },
        "patient_training_contract": {
            "n_events": int(len(old_labels)),
            "n_recording_blocks": int(len(np.unique(patient.block_ids))),
            "mode_counts": np.bincount(old_labels, minlength=2).tolist(),
            "primary_label_key": PRIMARY_LABEL_KEY,
            "primary_labels_replaced_by_classifier": False,
            "joint_null_label_contract": {
                "pseudo_model_side": "patient_train_classifier_labels",
                "patient_reference_side": PRIMARY_LABEL_KEY,
            },
            "classifier_label_agreement": float(np.mean(old_labels == assigned_labels)),
            "old_by_classifier_contingency": confusion.tolist(),
            "patient_train_ood_count": int(np.sum(patient_train_ood)),
            "patient_train_ood_fraction": float(np.mean(patient_train_ood)),
            "patient_train_old_label_conditioned_ood_count_diagnostic": int(
                np.sum(old_label_ood)
            ),
            "patient_train_old_label_conditioned_ood_fraction_diagnostic": float(
                np.mean(old_label_ood)
            ),
            "patient_data_sha256": patient.data_sha256,
            "contact_sha256": patient.contact_sha256,
        },
        "classifier_contract": {
            "semantics": "FULL_TIMING",
            "fit_data": "patient training recording blocks only",
            "refit_performed": True,
            "refit_all_parameters_match": True,
            "refit_audit": refit_audit,
            "accessed_keys": list(accessed_keys),
            "manifest_direction_keys": classifier["_manifest_direction_keys"],
            "training_cv_summary_keys_accessed": False,
            "prediction_role": "assign_formal_patient_train_ood_class",
            "patient_train_ood_semantics": (
                "class-conditional Mahalanobis distance and threshold selected by "
                "the frozen classifier-assigned class, exactly matching model OOD; "
                "old-label-conditioned OOD is diagnostic only"
            ),
            "parameter_hashes": {
                key: _array_sha256(classifier[key])
                for key in CLASSIFIER_PARAMETER_KEYS
            },
        },
        "floor_contract": {
            "sample_size_per_side": int(calibration.sample_size),
            "draws": int(calibration.floor_draws),
            "seed": int(calibration.seed),
            "tie_tolerance": float(floor["tie_tolerance"]),
            "quantile": 0.95,
            "quantile_method": "linear",
            "sampling": floor["sampling"],
            "primary_labels": PRIMARY_LABEL_KEY,
            "shaft_aware_k2_role": "not_loaded",
            "floor_q95": dict(calibration.floor_q95),
            "eligible_block_counts": dict(calibration.eligible_block_counts),
            "floor_distribution_array_keys": floor_keys,
            "all_calibration_array_keys": calibration_array_keys,
            "calibration_sha256": calibration.calibration_sha256,
            "calibration_metadata": calibration_metadata,
        },
        "artifact": {
            "manifest_name": manifest_path.name,
            "npz_name": npz_path.name,
            "npz_sha256": npz_sha,
            "array_keys": sorted(arrays),
            "array_sha256": {
                key: _array_sha256(value) for key, value in arrays.items()
            },
        },
        "forbidden_input_audit": {
            "only_three_scientific_inputs": True,
            "patient_evaluation_arrays_loaded": False,
            "seizure_arrays_loaded": False,
            "model_artifacts_loaded": False,
            "candidate_scores_loaded": False,
            "candidate_selection_performed": False,
            "simulation_performed": False,
        },
        "provenance": _runtime_provenance(config_path),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent,
    ))
    staging_manifest = staging_dir / manifest_path.name
    staging_npz = staging_dir / npz_path.name
    try:
        staging_npz.write_bytes(npz_bytes)
        _write_json(staging_manifest, manifest)
        if _sha256(staging_npz) != manifest["artifact"]["npz_sha256"]:
            raise RuntimeError("staged patient-support NPZ hash differs from manifest")
        if json.loads(staging_manifest.read_text()) != manifest:
            raise RuntimeError("staged patient-support manifest does not round-trip")
        _assert_source_snapshot_unchanged(source_snapshot)

        if output_dir.exists():
            existing = json.loads(manifest_path.read_text())
            if existing != manifest or _sha256(npz_path) != npz_sha:
                raise RuntimeError(
                    "deterministic patient-support rebuild differs from freeze"
                )
            return existing

        os.replace(staging_dir, output_dir)
        staging_dir = None
        try:
            if _sha256(npz_path) != manifest["artifact"]["npz_sha256"]:
                raise RuntimeError(
                    "published patient-support NPZ hash differs from manifest"
                )
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(
                    "published patient-support manifest differs from staging"
                )
        except Exception:
            shutil.rmtree(output_dir)
            raise
        return manifest
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = freeze_acceptance(
        artifact_root=args.artifact_root,
        config_path=args.config,
        output_dir_override=args.output_dir,
    )
    output_dir = (
        _resolve(args.output_dir, args.artifact_root)
        if args.output_dir is not None
        else _resolve(
            json.loads(args.config.read_text())["output"]["directory"],
            args.artifact_root,
        )
    )
    print(json.dumps({
        "status": manifest["status"],
        "draws": manifest["floor_contract"]["draws"],
        "patient_train_ood_fraction": manifest[
            "patient_training_contract"
        ]["patient_train_ood_fraction"],
        "manifest": str(output_dir / manifest["artifact"]["manifest_name"]),
        "npz": str(output_dir / manifest["artifact"]["npz_name"]),
    }, indent=2))


if __name__ == "__main__":
    main()
