"""Lossless Phase-C -> Phase-D counterfactual state migration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import topic4_zm_fast_carrier_contract as C
from src.topic4_zm_checkpoint import load_state_npz, state_hash


class StateMigrationError(RuntimeError):
    """Raised when a source state cannot be migrated without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateMigrationError(message)


def _copy_value(value):
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    return np.array(value, copy=True)


def _array_fingerprint(value: Any) -> dict:
    if isinstance(value, dict):
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return {"dtype": "json", "shape": [], "sha256": hashlib.sha256(raw).hexdigest()}
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def fingerprint_observables(
    result: Mapping[str, Any], *, excluded: tuple[str, ...] = ("wall_s",)
) -> dict:
    """Content fingerprints for exact continuation outputs.

    Wall-clock duration is deliberately excluded; every scientific observable,
    including the realised external-input trace when present, remains load
    bearing.
    """
    out = {}
    for key in sorted(set(result) - set(excluded)):
        value = result[key]
        if value is None:
            out[key] = {"kind": "none"}
        elif isinstance(value, (str, bool, int, float, np.generic)):
            out[key] = {
                "kind": "scalar",
                "value": value.item() if isinstance(value, np.generic) else value,
            }
        else:
            out[key] = {"kind": "array", **_array_fingerprint(value)}
    return out


def fingerprint_slow_traces(slow: Any) -> dict:
    """Hash all diagnostic trace buffers without prescribing their names."""
    inner = getattr(slow, "inner", slow)
    return {
        key: _array_fingerprint(value)
        for key, value in sorted(vars(inner).items())
        if key.startswith("trace_")
    }


def require_exact_continuation(
    reference: Mapping[str, Any], migrated: Mapping[str, Any], *, label: str
) -> None:
    """Fail closed if two continuation fingerprint trees differ anywhere."""
    _require(reference == migrated, f"{label} is not byte-identical after migration")


def _shallow_validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema") == C.INPUT_SCHEMA, "input schema drift")
    claimed = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(claimed == C.canonical_sha(body), "input manifest self-hash mismatch")
    _require(
        manifest.get("production_authorized") is False,
        "bootstrap input cannot authorize production",
    )


def migrate_state(
    source_state: Mapping[str, Any],
    source_row: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> tuple[dict, dict]:
    """Copy all classified fields and insert only the all-zero phi vector."""
    migration = input_manifest["state_migration"]
    _require(
        "slow.phi_increment" not in source_state,
        "source already contains phi; deterministic insertion is not valid",
    )
    expected = set(migration["carried_fields"])
    actual = set(source_state)
    _require(
        actual == expected,
        "source field inventory drift: "
        f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}",
    )
    sizes = migration["population_sizes"]
    N, NE, NI = (int(sizes[key]) for key in ("N", "NE", "NI"))
    _require(N == NE + NI, "population-size contract is inconsistent")
    _require(
        np.asarray(source_state["V"]).shape == (N,),
        "source V shape does not match the locked population",
    )

    migrated = {key: _copy_value(source_state[key]) for key in sorted(expected)}
    phi_spec = migration["inserted_fields"]["slow.phi_increment"]
    _require(
        phi_spec == {
            "dtype": "float64",
            "shape": [N],
            "fill": 0.0,
            "target": "E_active_I_exact_zero",
        },
        "inserted phi schema drift",
    )
    phi = np.zeros(N, dtype=np.float64)
    _require(np.count_nonzero(phi[NE:]) == 0, "inserted I-cell phi is nonzero")
    migrated["slow.phi_increment"] = phi

    for key in expected:
        left, right = source_state[key], migrated[key]
        if isinstance(left, dict):
            _require(left == right, f"carried field changed during migration: {key}")
        else:
            _require(
                np.array_equal(left, right)
                and np.asarray(left).dtype == np.asarray(right).dtype
                and np.asarray(left).shape == np.asarray(right).shape,
                f"carried field changed during migration: {key}",
            )

    fingerprints = {
        key: _array_fingerprint(source_state[key]) for key in sorted(expected)
    }
    source_hash = source_row["state_hash"]
    record_body = {
        "schema": "zm_fast_carrier_state_migration_v1.1_2026-07-31",
        "source_path": source_row["path"],
        "source_file_sha256": source_row["file_sha256"],
        "source_state_hash": source_hash,
        "source_config_sha": input_manifest["source"]["canonical_config_sha"],
        "source_engine_sha": source_row["source_state_manifest"]["engine_sha"],
        "phaseD_arm_config_sha256": input_manifest["phaseD_arm_config_sha256"],
        "carried_fields": sorted(expected),
        "inserted_fields": ["slow.phi_increment"],
        "carried_field_fingerprints": fingerprints,
        "migrated_state_hash": state_hash(migrated),
        "source_and_intervention_provenance_separate": True,
    }
    return migrated, {
        **record_body,
        "transformation_payload_sha256": C.canonical_sha(record_body),
    }


def load_and_migrate(
    root: Path | str,
    input_manifest: Mapping[str, Any],
    *,
    row_id: tuple[str, str],
    contract_already_validated: bool = False,
) -> tuple[dict, dict]:
    """Resolve one locked real checkpoint, verify it, then migrate it."""
    root = Path(root)
    if contract_already_validated:
        _shallow_validate_manifest(input_manifest)
    else:
        C.validate_input_manifest(input_manifest, root)
    matches = [
        row
        for row in input_manifest["source_panel"]
        if (row["bin_name"], row["fast_phase"]) == tuple(row_id)
    ]
    _require(len(matches) == 1, f"source row {row_id!r} is not locked exactly once")
    row = matches[0]
    path = root / row["path"]
    _require(path.is_file(), f"source state missing: {row['path']}")
    _require(
        C.sha256_file(path) == row["file_sha256"],
        f"source file hash drift: {row['path']}",
    )
    state, source_manifest = load_state_npz(
        path,
        expected_config_sha=input_manifest["source"]["canonical_config_sha"],
        expected_engine_sha=row["source_state_manifest"]["engine_sha"],
        expected_dt=input_manifest["source"]["dt_ms"],
    )
    _require(
        source_manifest == row["source_state_manifest"],
        "source embedded manifest drift",
    )
    return migrate_state(state, row, input_manifest)


__all__ = [
    "StateMigrationError",
    "fingerprint_observables",
    "fingerprint_slow_traces",
    "load_and_migrate",
    "migrate_state",
    "require_exact_continuation",
]
