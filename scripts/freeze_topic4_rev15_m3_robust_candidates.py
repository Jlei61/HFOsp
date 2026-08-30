#!/usr/bin/env python3
"""Freeze complete-tensor robust M3 fields before fresh-network simulation."""
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
from scripts import freeze_topic4_rev15_m3_coordinate_atlas as atlas  # noqa: E402
from scripts import prepare_topic4_rev15_m3_robust_candidate_config as prepare  # noqa: E402
from src.topic4_rev14_fourier_field import array_sha256, mode_inventory  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV15_M3_ROBUST_CANDIDATES_FROZEN"
PREPARE_STATUS = "REV15_M3_ROBUST_CANDIDATES_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev15_m3_robust_candidates_v1"
MANIFEST_SCHEMA = "topic4_rev15_m3_robust_candidates_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(base.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev15_m3_robust_candidates.json",
    "scripts/prepare_topic4_rev15_m3_robust_candidate_config.py",
    "scripts/freeze_topic4_rev15_m3_robust_candidates.py",
    "scripts/run_topic4_rev15_m3_robust_candidate_worker.py",
    "scripts/monitor_topic4_rev15_m3_robust_candidates.py",
    "scripts/launch_topic4_rev15_m3_robust_candidates.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py",
    "src/topic4_rev14_fourier_field.py",
    "src/topic4_rev14_field_projection.py",
)

_atomic_json = base._atomic_json
_sha256 = base._sha256


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.is_file() else artifact_root / relative


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
        raise RuntimeError("rev15 robust-candidate schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev15 robust candidates must remain Node-only")
    expected_inputs = {
        "rev13_config", "rev13_exact_off_manifest", "source_atlas_config",
        "source_atlas_manifest", "multinetwork_response_analysis_config",
        "multinetwork_response_aggregate", "j14_config",
        "patient_support_config",
    }
    if set(config.get("inputs", {})) != expected_inputs:
        raise RuntimeError("rev15 robust-candidate input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev15 robust-candidate input is not hashed: {name}")
    design = config["m3_design"]
    expected = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 3,
        "expected_modes": 14,
        "expected_real_coefficients": 28,
        "sheet_length_mm": 20.0,
        "quadrature_per_axis": 128,
        "coordinate_decimal_places": 13,
        "candidate_rms_levels": [0.4, 0.6, 0.8],
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"rev15 robust-candidate design changed: {key}")
    family_ids = design.get("robust_direction_ids", [])
    if not family_ids or any(name not in prepare.FAMILY_ORDER for name in family_ids):
        raise RuntimeError("rev15 robust direction family inventory is invalid")
    if family_ids != [name for name in prepare.FAMILY_ORDER if name in family_ids]:
        raise RuntimeError("rev15 robust direction family order changed")
    selectable = int(design.get("selectable_candidate_count", -1))
    if not 1 <= selectable <= len(prepare.FAMILY_ORDER) * len(prepare.RMS_LEVELS):
        raise RuntimeError("rev15 robust selectable-candidate count is invalid")
    if int(design.get("candidate_count", -1)) != selectable + 1:
        raise RuntimeError("rev15 robust exact-reference count is invalid")
    if len(design.get("candidate_ids", [])) != selectable:
        raise RuntimeError("rev15 robust candidate ID count changed")
    blueprint = design.get("candidate_blueprint", [])
    if [row.get("candidate_id") for row in blueprint] != design["candidate_ids"]:
        raise RuntimeError("rev15 robust candidate blueprint changed")
    search = config["search"]
    if search.get("construction_network_seeds") != [2331, 2332, 2333]:
        raise RuntimeError("rev15 robust construction seed pool changed")
    if search.get("canary_network_seeds") != [2341, 2342, 2343]:
        raise RuntimeError("rev15 robust canary seed pool changed")
    if search.get("active_network_seeds") != [2341, 2342, 2343]:
        raise RuntimeError("rev15 robust active seed pool changed")
    if float(search["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev15 robust simulation duration changed")
    selection = config["selection"]
    expected_selection = {
        "fresh_A_improvement_required_networks": 3,
        "fresh_B_protection_required_networks": 3,
        "B_protection_ratio": 1.10,
        "equal_network_effective_support_minimum_per_mode": 6.0,
        "natural_kmeans_used": False,
        "patient_heldout_used": False,
        "ictal_data_used": False,
        "figure_used": False,
    }
    for key, value in expected_selection.items():
        if selection.get(key) != value:
            raise RuntimeError(f"rev15 robust selection boundary changed: {key}")


def _load_hashed(
    config: Mapping[str, Any], name: str, artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    record = config["inputs"][name]
    path = _resolve(artifact_root, str(record["path"]))
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise RuntimeError(f"rev15 robust-candidate input changed: {name}")
    return path, json.loads(path.read_text())


def _input_audit(config: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    return {
        name: {
            "path": str(_load_hashed(config, name, artifact_root)[0]),
            "sha256": record["sha256"], "verified": True,
        }
        for name, record in config["inputs"].items()
    }


def build_candidates(
    config: Mapping[str, Any], artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _validate_config(config)
    _, source_config = _load_hashed(config, "source_atlas_config", artifact_root)
    _, source_manifest = _load_hashed(config, "source_atlas_manifest", artifact_root)
    _, aggregate = _load_hashed(
        config, "multinetwork_response_aggregate", artifact_root,
    )
    if source_manifest.get("status") != atlas.STATUS:
        raise RuntimeError("source M3 atlas is not formally frozen")
    if aggregate.get("status") != "COMPLETE":
        raise RuntimeError("multinetwork response aggregate is incomplete")
    rows, audit = prepare.candidate_blueprint(
        aggregate, n_per_axis=int(config["m3_design"]["quadrature_per_axis"]),
    )
    expected_blueprint = config["m3_design"]["candidate_blueprint"]
    if json.dumps(rows, sort_keys=True) != json.dumps(expected_blueprint, sort_keys=True):
        raise RuntimeError("robust candidate blueprint no longer matches aggregate")
    if audit["feasible_direction_ids"] != config["m3_design"]["robust_direction_ids"]:
        raise RuntimeError("robust direction inventory no longer matches aggregate")
    if audit["candidate_ids"] != config["m3_design"]["candidate_ids"]:
        raise RuntimeError("robust candidate IDs no longer match aggregate")
    if audit["deduplicated"] != config["m3_design"]["deduplication_audit"]:
        raise RuntimeError("robust candidate deduplication audit changed")
    if config["node_mapping"] != source_config["node_mapping"]:
        raise RuntimeError("rev15 robust candidates changed the Node mapping")
    modes = mode_inventory(3)
    if [list(mode) for mode in modes] != source_manifest["direction_audit"]["modes"]:
        raise RuntimeError("rev15 robust candidates changed the M3 basis")
    candidates = [copy.deepcopy(next(
        row for row in source_manifest["candidates"]
        if row["candidate_id"] == "exact_off"
    ))]
    decimal_places = int(config["m3_design"]["coordinate_decimal_places"])
    for blueprint in rows:
        coefficients = np.asarray(blueprint["coefficients"], dtype=np.float64)
        if array_sha256(coefficients) != blueprint["coefficients_sha256"]:
            raise RuntimeError("robust candidate coefficient hash changed")
        row = base._coordinate_record(
            candidate_id=blueprint["candidate_id"],
            field_kind="absolute_paired_phase_fourier_m3",
            selectable=True, modes=modes, coefficients=coefficients,
            direction_index=None, sign=None,
            target_rms=float(blueprint["target_rms"]),
            decimal_places=decimal_places,
        )
        row["robust_response_construction"] = {
            "family": blueprint["family"],
            "target_centered_surface_rms": blueprint["target_rms"],
            "raw_direction_sha256": blueprint["raw_direction_sha256"],
            "construction_network_seeds": [2331, 2332, 2333],
        }
        candidates.append(row)
    if len(candidates) != int(config["m3_design"]["candidate_count"]):
        raise RuntimeError("rev15 robust-candidate count changed")
    hashes = [
        row["fourier_coordinate"]["coefficients_sha256"]
        for row in candidates if row["selection_eligible"]
    ]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("rev15 robust candidates contain duplicate fields")
    arrays = [
        np.asarray(row["fourier_coordinate"]["coefficients"], dtype=np.float64)
        for row in candidates if row["selection_eligible"]
    ]
    for index, left in enumerate(arrays):
        if any(np.array_equal(left, -right) for right in arrays[index + 1:]):
            raise RuntimeError("rev15 robust candidates contain sign-equivalent fields")
    direction_audit = {
        "modes": [list(mode) for mode in modes],
        "modes_sha256": array_sha256(np.asarray(modes, dtype=np.int64)),
        "feasible_direction_ids": audit["feasible_direction_ids"],
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "deduplicated": audit["deduplicated"],
        "patient_training_scores_used_for_construction": True,
        "natural_kmeans_used": False,
        "patient_heldout_used": False,
        "observation_geometry_used": False,
    }
    return candidates, direction_audit, source_manifest


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    candidates, direction_audit, source_manifest = build_candidates(
        config, artifact_root,
    )
    return {
        "schema_id": MANIFEST_SCHEMA,
        "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": direction_audit,
        "signed_depth_audit": copy.deepcopy(source_manifest["signed_depth_audit"]),
        "exact_off_reconstruction": copy.deepcopy(
            source_manifest["exact_off_reconstruction"]
        ),
        "event_unit": copy.deepcopy(source_manifest["event_unit"]),
        "source_topology": copy.deepcopy(source_manifest["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "m3_design": copy.deepcopy(config["m3_design"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "selection": copy.deepcopy(config["selection"]),
        "inputs": _input_audit(config, artifact_root),
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
        parser.error("formal robust-candidate freeze requires --expected-commit")
    config_path = args.config.resolve()
    provenance = runtime_provenance(
        config_path, expected_commit=args.expected_commit,
        require_clean=not args.prepare_only,
    )
    status = PREPARE_STATUS if args.prepare_only else STATUS
    payload = build_manifest_payload(
        config_path, artifact_root=args.artifact_root.resolve(),
        provenance=provenance, status=status,
    )
    config = json.loads(config_path.read_text())
    output = args.artifact_root.resolve() / config["candidate_manifest"]
    if not args.prepare_only:
        _atomic_json(output, payload)
    print(json.dumps({
        "status": status,
        "n_candidates": len(payload["candidates"]),
        "n_jobs": len(payload["candidates"]) * 3,
        "formal_ready": bool(provenance["formal_ready"]),
        "output_written": not args.prepare_only,
        "output": str(output),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
