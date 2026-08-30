#!/usr/bin/env python3
"""Freeze joint M3+M4 fields before fresh-network simulation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev14_m3_canary as base  # noqa: E402
from scripts import prepare_topic4_rev16_joint_candidate_config as prepare  # noqa: E402
from src.topic4_rev14_fourier_field import array_sha256, mode_inventory  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV16_JOINT_M3_M4_CANDIDATES_FROZEN"
PREPARE_STATUS = "REV16_JOINT_M3_M4_CANDIDATES_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev16_joint_m3_m4_candidates_v1"
MANIFEST_SCHEMA = "topic4_rev16_joint_m3_m4_candidates_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(base.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev16_joint_m3_m4_candidates.json",
    "scripts/prepare_topic4_rev16_joint_candidate_config.py",
    "scripts/freeze_topic4_rev16_joint_m3_m4_candidates.py",
    "scripts/run_topic4_rev16_joint_m3_m4_candidate_worker.py",
    "scripts/monitor_topic4_rev16_joint_m3_m4_candidates.py",
    "scripts/launch_topic4_rev16_joint_m3_m4_candidates.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py",
    "src/topic4_rev14_fourier_field.py",
    "src/topic4_rev14_field_projection.py",
)

_atomic_json = base._atomic_json
_jsonable = base._jsonable
_sha256 = base._sha256
_resolve = base._resolve


def runtime_provenance(
    config_path: Path, *, expected_commit: str | None, require_clean: bool,
) -> dict[str, Any]:
    previous = base.FORMAL_RUNTIME_PATHS
    base.FORMAL_RUNTIME_PATHS = FORMAL_RUNTIME_PATHS
    try:
        return base.runtime_provenance(
            config_path, expected_commit=expected_commit,
            require_clean=require_clean,
        )
    finally:
        base.FORMAL_RUNTIME_PATHS = previous


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != EXPECTED_SCHEMA:
        raise RuntimeError("rev16 joint-candidate schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev16 joint candidates must remain Node-only")
    expected_inputs = {
        "rev13_config", "rev13_exact_off_manifest", "source_shell_config",
        "source_shell_manifest", "joint_response_analysis_config",
        "joint_response_aggregate", "j14_config", "patient_support_config",
    }
    if set(config.get("inputs", {})) != expected_inputs:
        raise RuntimeError("rev16 joint-candidate input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev16 joint input is not hashed: {name}")
    design = config["field_design"]
    expected = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 4, "expected_modes": 24,
        "expected_real_coefficients": 48,
        "sheet_length_mm": 20.0, "quadrature_per_axis": 128,
        "coordinate_decimal_places": 13,
        "candidate_rms_levels": [0.4, 0.6, 0.8],
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"rev16 joint design changed: {key}")
    family_ids = design.get("robust_direction_ids", [])
    if not family_ids or any(name not in prepare.FAMILY_ORDER for name in family_ids):
        raise RuntimeError("rev16 joint direction inventory is invalid")
    if family_ids != [name for name in prepare.FAMILY_ORDER if name in family_ids]:
        raise RuntimeError("rev16 joint direction order changed")
    selectable = int(design.get("selectable_candidate_count", -1))
    if not 1 <= selectable <= len(prepare.FAMILY_ORDER) * len(prepare.RMS_LEVELS):
        raise RuntimeError("rev16 joint selectable count is invalid")
    if int(design.get("candidate_count", -1)) != selectable + 1:
        raise RuntimeError("rev16 joint exact-reference count is invalid")
    if len(design.get("candidate_ids", [])) != selectable:
        raise RuntimeError("rev16 joint candidate ID count changed")
    if [row.get("candidate_id") for row in design.get("candidate_blueprint", [])] != design["candidate_ids"]:
        raise RuntimeError("rev16 joint blueprint changed")
    if config["search"].get("active_network_seeds") != [2351, 2352, 2353]:
        raise RuntimeError("rev16 joint fresh-network pool changed")
    if float(config["search"]["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev16 joint duration changed")
    selection = config["selection"]
    expected_selection = {
        "fresh_J14_improvement_required_networks": 3,
        "fresh_A_improvement_required_networks": 3,
        "fresh_B_protection_required_networks": 3,
        "B_protection_ratio": 1.10,
        "equal_network_effective_support_minimum_per_mode": 6.0,
        "natural_kmeans_used": False, "patient_heldout_used": False,
        "ictal_data_used": False, "figure_used": False,
    }
    for key, value in expected_selection.items():
        if selection.get(key) != value:
            raise RuntimeError(f"rev16 joint selection boundary changed: {key}")


def _load_hashed(
    config: Mapping[str, Any], name: str, artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    record = config["inputs"][name]
    path = _resolve(artifact_root, record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise RuntimeError(f"rev16 joint-candidate input changed: {name}")
    return path, json.loads(path.read_text())


def build_candidates(
    config: Mapping[str, Any], artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _validate_config(config)
    _, aggregate = _load_hashed(config, "joint_response_aggregate", artifact_root)
    rows, audit = prepare.candidate_blueprint(
        aggregate, n_per_axis=int(config["field_design"]["quadrature_per_axis"]),
    )
    if json.dumps(rows, sort_keys=True) != json.dumps(
        config["field_design"]["candidate_blueprint"], sort_keys=True,
    ):
        raise RuntimeError("joint candidate blueprint no longer matches aggregate")
    if audit["candidate_ids"] != config["field_design"]["candidate_ids"]:
        raise RuntimeError("joint candidate IDs no longer match aggregate")
    modes = mode_inventory(4)
    decimal_places = int(config["field_design"]["coordinate_decimal_places"])
    candidates = [base._coordinate_record(
        candidate_id="exact_off", field_kind="stage_ak_exact_off_benchmark",
        selectable=False, modes=modes, coefficients=None,
        decimal_places=decimal_places,
    )]
    for row in rows:
        coefficients = np.asarray(row["coefficients"], dtype=np.float64)
        candidate = base._coordinate_record(
            candidate_id=row["candidate_id"],
            field_kind="absolute_paired_phase_fourier_joint_m3_m4",
            selectable=True, modes=modes, coefficients=coefficients,
            direction_index=None, sign=None, target_rms=float(row["target_rms"]),
            decimal_places=decimal_places,
        )
        candidate["joint_direction"] = {
            "family": row["family"],
            "raw_direction_sha256": row["raw_direction_sha256"],
            "m3_l2_fraction": row["m3_l2_fraction"],
            "m4_shell_l2_fraction": row["m4_shell_l2_fraction"],
        }
        if candidate["fourier_coordinate"]["coefficients_sha256"] != row["coefficients_sha256"]:
            raise RuntimeError("joint coefficient quantization changed")
        candidates.append(candidate)
    if len(candidates) != int(config["field_design"]["candidate_count"]):
        raise RuntimeError("rev16 joint candidate count changed")
    hashes = [
        row["fourier_coordinate"]["coefficients_sha256"]
        for row in candidates if row["fourier_coordinate"] is not None
    ]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("rev16 joint fields contain duplicate coefficients")
    return candidates, audit, aggregate


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    candidates, audit, aggregate = build_candidates(config, artifact_root)
    _, rev13_config = _load_hashed(config, "rev13_config", artifact_root)
    _, rev13_manifest = _load_hashed(
        config, "rev13_exact_off_manifest", artifact_root,
    )
    return {
        "schema_id": MANIFEST_SCHEMA, "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates, "direction_audit": audit,
        "signed_depth_audit": base.frozen_signed_depth_audit(config),
        "exact_off_reconstruction": base._exact_off_reconstruction(
            rev13_config, rev13_manifest,
        ),
        "event_unit": copy.deepcopy(rev13_config["event_unit"]),
        "source_topology": copy.deepcopy(rev13_config["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": {
            name: {"path": str(_load_hashed(config, name, artifact_root)[0]),
                   "sha256": record["sha256"], "verified": True}
            for name, record in config["inputs"].items()
        },
        "source_response": {
            "aggregate_sha256": config["inputs"]["joint_response_aggregate"]["sha256"],
            "aggregate_status": aggregate["status"],
            "joint_dimension": 48,
        },
        "provenance": copy.deepcopy(dict(provenance)),
        "claim_boundary": config["claim_boundary"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--prepare-only", "--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only and args.expected_commit is None:
        parser.error("formal freeze requires --expected-commit")
    config_path = args.config.resolve()
    provenance = runtime_provenance(
        config_path, expected_commit=args.expected_commit,
        require_clean=not args.prepare_only,
    )
    payload = build_manifest_payload(
        config_path, artifact_root=args.artifact_root.resolve(),
        provenance=provenance,
        status=PREPARE_STATUS if args.prepare_only else STATUS,
    )
    if not args.prepare_only:
        config = json.loads(config_path.read_text())
        _atomic_json(args.artifact_root.resolve() / config["candidate_manifest"], payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(payload["candidates"]),
        "n_jobs": len(payload["candidates"]) * 3,
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
