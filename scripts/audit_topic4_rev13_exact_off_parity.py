#!/usr/bin/env python3
"""Fail-closed artifact parity audit for the rev13 exact-off overlay.

The audit compares a shorter rev13 exact-off run with a same-seed Stage-AK
run.  Only samples and causal-family events fully observable in their common
time prefix are compared.  Events too close to the right boundary are omitted
because the historical run has future causal context unavailable to the short
run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA_ID = "topic4_rev13_exact_off_artifact_parity_v1"
SIGNAL_FIELDS = {
    "active_fraction": "active_fraction_bin_ms",
    "contact_envelope": "contact_envelope_dt_ms",
    "sheet_activity_counts": "sheet_activity_frame_ms",
}
EVENT_FIELDS = (
    "event_t_on_ms",
    "event_t_off_ms",
    "event_returned",
    "onsets",
    "ranks",
    "source_onset_maps_ms",
    "source_onset_evaluable",
    "event_fragment_count",
    "event_root_count",
    "event_directed_root_id",
)
SUBSTRATE_ARRAY_FIELDS = (
    "positions_E",
    "h",
    "delta_vtheta",
    "edge_coefficients",
    "contact_names",
    "shaft_ids",
    "contact_xy_mm",
)


class AuditFailure(RuntimeError):
    """Raised for malformed inputs that cannot support a parity claim."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - exercised through CLI guard
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditFailure(f"JSON root is not an object: {path}")
    return payload


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise AuditFailure(f"missing {context}.{key}")
    return mapping[key]


def _nested(mapping: Mapping[str, Any], keys: tuple[str, ...], context: str) -> Any:
    value: Any = mapping
    traversed = context
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise AuditFailure(f"missing {traversed}.{key}")
        value = value[key]
        traversed = f"{traversed}.{key}"
    return value


def _exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _first_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    equal = np.equal(left, right)
    if left.dtype.kind in "fc":
        equal = equal | (np.isnan(left) & np.isnan(right))
    bad = np.argwhere(~equal)
    if not len(bad):
        return {}
    index = tuple(int(value) for value in bad[0])
    return {
        "index": list(index),
        "left": _json_scalar(left[index]),
        "right": _json_scalar(right[index]),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _record_check(
    checks: list[dict[str, Any]], name: str, passed: bool, **details: Any
) -> None:
    checks.append({
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    })


def _actual_coverage_ms(
    payload: Mapping[str, Any], arrays: Mapping[str, np.ndarray], label: str
) -> tuple[float, dict[str, float]]:
    declared = float(_nested(payload, ("simulation", "duration_ms"), label))
    if not np.isfinite(declared) or declared <= 0.0:
        raise AuditFailure(f"invalid {label}.simulation.duration_ms")
    coverage = {"declared_duration_ms": declared}
    for field, dt_field in SIGNAL_FIELDS.items():
        values = np.asarray(_required(arrays, field, f"{label}.npz"))
        dt_value = np.asarray(_required(arrays, dt_field, f"{label}.npz"))
        if dt_value.shape != ():
            raise AuditFailure(f"{label}.{dt_field} is not scalar")
        dt_ms = float(dt_value)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise AuditFailure(f"invalid {label}.{dt_field}")
        if values.ndim < 1:
            raise AuditFailure(f"{label}.{field} has no time axis")
        time_axis = 1 if field == "contact_envelope" else 0
        coverage[f"{field}_coverage_ms"] = float(values.shape[time_axis] * dt_ms)
    actual = min(coverage.values())
    if actual <= 0.0:
        raise AuditFailure(f"{label} has no common signal coverage")
    return float(actual), coverage


def _prefix(values: np.ndarray, field: str, n_time: int) -> np.ndarray:
    if field == "contact_envelope":
        return values[:, :n_time, ...]
    return values[:n_time, ...]


def _event_indices(
    arrays: Mapping[str, np.ndarray], boundary_ms: float, label: str
) -> np.ndarray:
    t_on = np.asarray(_required(arrays, "event_t_on_ms", f"{label}.npz"))
    t_off = np.asarray(_required(arrays, "event_t_off_ms", f"{label}.npz"))
    if t_on.ndim != 1 or t_off.shape != t_on.shape:
        raise AuditFailure(f"{label} event boundary arrays do not align")
    if not np.isfinite(t_on).all() or not np.isfinite(t_off).all():
        raise AuditFailure(f"{label} event boundaries contain non-finite values")
    if np.any(t_on < 0.0) or np.any(t_off < t_on):
        raise AuditFailure(f"{label} event boundaries are invalid")
    if len(t_on) > 1 and np.any(t_on[1:] < t_on[:-1]):
        raise AuditFailure(f"{label} events are not time ordered")
    return np.flatnonzero(t_off <= boundary_ms + 1e-9)


def _validate_event_alignment(
    payload: Mapping[str, Any], arrays: Mapping[str, np.ndarray], label: str
) -> None:
    t_on = np.asarray(_required(arrays, "event_t_on_ms", f"{label}.npz"))
    n_events = int(len(t_on))
    for field in EVENT_FIELDS:
        values = np.asarray(_required(arrays, field, f"{label}.npz"))
        if values.ndim < 1 or values.shape[0] != n_events:
            raise AuditFailure(f"{label}.{field} is not event aligned")
    events = _required(payload, "events", label)
    if not isinstance(events, list) or len(events) != n_events:
        raise AuditFailure(f"{label}.events does not align with NPZ events")
    for index, row in enumerate(events):
        if not isinstance(row, Mapping):
            raise AuditFailure(f"{label}.events[{index}] is not an object")
        expected = {
            "t_on_ms": np.asarray(arrays["event_t_on_ms"])[index],
            "t_off_ms": np.asarray(arrays["event_t_off_ms"])[index],
            "returned": np.asarray(arrays["event_returned"])[index],
            "n_detector_fragments": np.asarray(arrays["event_fragment_count"])[index],
            "root_count": np.asarray(arrays["event_root_count"])[index],
        }
        for key, value in expected.items():
            if key not in row:
                raise AuditFailure(f"missing {label}.events[{index}].{key}")
            if isinstance(value, (np.floating, float)):
                same = float(row[key]) == float(value)
            else:
                same = row[key] == _json_scalar(value)
            if not same:
                raise AuditFailure(
                    f"{label}.events[{index}].{key} disagrees with NPZ"
                )


def _validate_exact_off(
    payload: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> list[str]:
    failures: list[str] = []
    node = payload.get("node_accessibility")
    if not isinstance(node, Mapping):
        failures.append("JSON node_accessibility object is missing")
    else:
        if node.get("enabled") is not False:
            failures.append("JSON node_accessibility.enabled is not false")
        if node.get("mode") != "exact_off":
            failures.append("JSON node_accessibility.mode is not exact_off")
        if node.get("support") is not None:
            failures.append("JSON exact-off controller support is not empty")
        manifest = node.get("manifest")
        if not isinstance(manifest, Mapping) or manifest.get("mode") != "exact_off":
            failures.append("JSON exact-off controller manifest is invalid")
        elif manifest.get("enabled") is not False:
            failures.append("JSON exact-off controller manifest is enabled")
    mechanism = payload.get("mechanism_freeze")
    if not isinstance(mechanism, Mapping):
        failures.append("JSON mechanism_freeze object is missing")
    else:
        if mechanism.get("node_accessibility") != "exact_off":
            failures.append("mechanism_freeze.node_accessibility is not exact_off")
        if mechanism.get("node_accessibility_active") is not False:
            failures.append("mechanism_freeze.node_accessibility_active is not false")
    required = {"node_accessibility_enabled", "node_accessibility_mode"}
    missing = required.difference(arrays)
    if missing:
        failures.append("NPZ misses " + ", ".join(sorted(missing)))
        return failures
    enabled = np.asarray(arrays["node_accessibility_enabled"])
    mode = np.asarray(arrays["node_accessibility_mode"])
    if enabled.shape != () or bool(enabled):
        failures.append("NPZ node_accessibility_enabled is not scalar false")
    if mode.shape != () or str(mode.item()) != "exact_off":
        failures.append("NPZ node_accessibility_mode is not scalar exact_off")
    dynamic_keys = sorted(
        key for key in arrays
        if key.startswith("node_accessibility_") and key not in required
    )
    if not dynamic_keys:
        failures.append("NPZ exact-off dynamic-field contract is absent")
    for key in dynamic_keys:
        if np.asarray(arrays[key]).size != 0:
            failures.append(f"NPZ exact-off dynamic field is nonempty: {key}")
    return failures


def _artifact_hash_check(payload: Mapping[str, Any], npz_path: Path, label: str) -> str:
    expected = str(_nested(payload, ("arrays", "sha256"), label))
    observed = _sha256_file(npz_path)
    if expected != observed:
        raise AuditFailure(f"{label} NPZ sha256 does not match its JSON sidecar")
    return observed


def audit_exact_off_parity(
    rev13_json: Path,
    rev13_npz: Path,
    stage_ak_json: Path,
    stage_ak_npz: Path,
) -> dict[str, Any]:
    """Compare rev13 exact-off and Stage-AK artifacts on observable history."""
    paths = {
        "rev13_json": Path(rev13_json).resolve(),
        "rev13_npz": Path(rev13_npz).resolve(),
        "stage_ak_json": Path(stage_ak_json).resolve(),
        "stage_ak_npz": Path(stage_ak_npz).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise AuditFailure(f"missing input {name}: {path}")
    rev13_payload = _load_json(paths["rev13_json"])
    stage_payload = _load_json(paths["stage_ak_json"])
    rev13_hash = _artifact_hash_check(
        rev13_payload, paths["rev13_npz"], "rev13"
    )
    stage_hash = _artifact_hash_check(
        stage_payload, paths["stage_ak_npz"], "stage_ak"
    )

    checks: list[dict[str, Any]] = []
    with np.load(paths["rev13_npz"], allow_pickle=False) as rev13_loaded, np.load(
        paths["stage_ak_npz"], allow_pickle=False
    ) as stage_loaded:
        rev13_arrays = {key: np.asarray(rev13_loaded[key]) for key in rev13_loaded.files}
        stage_arrays = {key: np.asarray(stage_loaded[key]) for key in stage_loaded.files}

    controller_failures = _validate_exact_off(rev13_payload, rev13_arrays)
    _record_check(
        checks, "rev13_controller_exact_off", not controller_failures,
        failures=controller_failures,
    )

    seed_left = _required(rev13_payload, "seed", "rev13")
    seed_right = _required(stage_payload, "seed", "stage_ak")
    _record_check(
        checks, "same_network_seed", seed_left == seed_right,
        rev13=seed_left, stage_ak=seed_right,
    )

    identity_pairs = {
        "field_sha256": (
            _required(rev13_payload, "field_sha256", "rev13"),
            _required(stage_payload, "field_sha256", "stage_ak"),
        ),
        "mapping_sha256": (
            _nested(rev13_payload, ("node_mapping", "mapping_sha256"), "rev13"),
            _nested(stage_payload, ("node_mapping", "mapping_sha256"), "stage_ak"),
        ),
        "contact_readout": (
            _required(rev13_payload, "contact_readout", "rev13"),
            _required(stage_payload, "contact_readout", "stage_ak"),
        ),
    }
    for name, (left, right) in identity_pairs.items():
        _record_check(
            checks, f"identity_{name}", left == right,
            rev13=left, stage_ak=right,
        )

    for field in SUBSTRATE_ARRAY_FIELDS:
        left = np.asarray(_required(rev13_arrays, field, "rev13.npz"))
        right = np.asarray(_required(stage_arrays, field, "stage_ak.npz"))
        passed = _exact_equal(left, right)
        _record_check(
            checks, f"substrate_array_{field}", passed,
            rev13_sha256=_sha256_array(left),
            stage_ak_sha256=_sha256_array(right),
            first_difference={} if passed else _first_difference(left, right),
        )

    rev13_mechanism = _required(rev13_payload, "mechanism_freeze", "rev13")
    stage_mechanism = _required(stage_payload, "mechanism_freeze", "stage_ak")
    base_mechanism_keys = ("EE", "E_to_I", "Z_M", "edge_coefficients_all_zero")
    mechanism_match = all(
        rev13_mechanism.get(key) == stage_mechanism.get(key)
        for key in base_mechanism_keys
    ) and all(
        rev13_mechanism.get(key) in ("off", True, 1)
        for key in base_mechanism_keys
    )
    _record_check(
        checks, "base_mechanisms_frozen_off", mechanism_match,
        rev13={key: rev13_mechanism.get(key) for key in base_mechanism_keys},
        stage_ak={key: stage_mechanism.get(key) for key in base_mechanism_keys},
    )

    rev13_coverage, rev13_coverage_detail = _actual_coverage_ms(
        rev13_payload, rev13_arrays, "rev13"
    )
    stage_coverage, stage_coverage_detail = _actual_coverage_ms(
        stage_payload, stage_arrays, "stage_ak"
    )
    common_ms = min(rev13_coverage, stage_coverage)

    for field, dt_field in SIGNAL_FIELDS.items():
        left = np.asarray(_required(rev13_arrays, field, "rev13.npz"))
        right = np.asarray(_required(stage_arrays, field, "stage_ak.npz"))
        left_dt = float(np.asarray(rev13_arrays[dt_field]))
        right_dt = float(np.asarray(stage_arrays[dt_field]))
        if left_dt != right_dt:
            _record_check(
                checks, f"prefix_signal_{field}", False,
                reason="time_step_mismatch", rev13_dt_ms=left_dt,
                stage_ak_dt_ms=right_dt,
            )
            continue
        n_time = int(np.floor(common_ms / left_dt + 1e-9))
        left_prefix = _prefix(left, field, n_time)
        right_prefix = _prefix(right, field, n_time)
        passed = _exact_equal(left_prefix, right_prefix)
        _record_check(
            checks, f"prefix_signal_{field}", passed,
            compared_samples=n_time,
            dt_ms=left_dt,
            first_difference=(
                {} if passed else _first_difference(left_prefix, right_prefix)
            ),
        )

    memory_left = float(_nested(
        rev13_payload, ("event_unit", "causal_memory_ms"), "rev13"
    ))
    memory_right = float(_nested(
        stage_payload, ("event_unit", "causal_memory_ms"), "stage_ak"
    ))
    memory_match = (
        np.isfinite(memory_left) and np.isfinite(memory_right)
        and memory_left >= 0.0 and memory_left == memory_right
    )
    _record_check(
        checks, "causal_memory_contract", bool(memory_match),
        rev13_ms=memory_left, stage_ak_ms=memory_right,
    )
    causal_memory_ms = memory_left if memory_match else max(memory_left, memory_right)
    event_boundary_ms = common_ms - causal_memory_ms
    if event_boundary_ms < 0.0:
        raise AuditFailure("common prefix is shorter than causal memory")

    _validate_event_alignment(rev13_payload, rev13_arrays, "rev13")
    _validate_event_alignment(stage_payload, stage_arrays, "stage_ak")
    rev13_indices = _event_indices(rev13_arrays, event_boundary_ms, "rev13")
    stage_indices = _event_indices(stage_arrays, event_boundary_ms, "stage_ak")
    _record_check(
        checks, "comparable_event_count",
        np.array_equal(rev13_indices, stage_indices),
        rev13_count=int(len(rev13_indices)),
        stage_ak_count=int(len(stage_indices)),
        rev13_indices=rev13_indices.tolist(),
        stage_ak_indices=stage_indices.tolist(),
    )
    for field in EVENT_FIELDS:
        left = np.asarray(rev13_arrays[field])[rev13_indices]
        right = np.asarray(stage_arrays[field])[stage_indices]
        passed = _exact_equal(left, right)
        _record_check(
            checks, f"prefix_events_{field}", passed,
            first_difference={} if passed else _first_difference(left, right),
        )

    failures = [row["name"] for row in checks if row["status"] != "PASS"]
    return {
        "schema_id": SCHEMA_ID,
        "status": "PASS" if not failures else "FAIL",
        "passed": not failures,
        "claim": (
            "REV13_EXACT_OFF_COMMON_PREFIX_ARTIFACT_PARITY"
            if not failures else "REV13_EXACT_OFF_ARTIFACT_PARITY_NOT_ESTABLISHED"
        ),
        "inputs": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in paths.items()
        },
        "npz_sidecar_hashes": {
            "rev13": rev13_hash,
            "stage_ak": stage_hash,
        },
        "common_prefix_ms": common_ms,
        "causal_memory_ms": causal_memory_ms,
        "event_right_boundary_ms": event_boundary_ms,
        "coverage": {
            "rev13": rev13_coverage_detail,
            "stage_ak": stage_coverage_detail,
        },
        "n_comparable_events": {
            "rev13": int(len(rev13_indices)),
            "stage_ak": int(len(stage_indices)),
        },
        "checks": checks,
        "failed_checks": failures,
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _failure_payload(exc: Exception, inputs: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "status": "FAIL",
        "passed": False,
        "claim": "REV13_EXACT_OFF_ARTIFACT_PARITY_NOT_ESTABLISHED",
        "inputs": {key: str(path.resolve()) for key, path in inputs.items()},
        "checks": [],
        "failed_checks": ["audit_input_or_contract_error"],
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rev13-json", required=True, type=Path)
    parser.add_argument("--rev13-npz", required=True, type=Path)
    parser.add_argument("--stage-ak-json", required=True, type=Path)
    parser.add_argument("--stage-ak-npz", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    inputs = {
        "rev13_json": args.rev13_json,
        "rev13_npz": args.rev13_npz,
        "stage_ak_json": args.stage_ak_json,
        "stage_ak_npz": args.stage_ak_npz,
    }
    try:
        payload = audit_exact_off_parity(**inputs)
    except Exception as exc:
        payload = _failure_payload(exc, inputs)
    _atomic_json(payload, args.out_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
