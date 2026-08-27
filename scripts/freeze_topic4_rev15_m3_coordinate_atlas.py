#!/usr/bin/env python3
"""Freeze the observation-free complete-M3 coordinate response atlas."""
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

from scripts import freeze_topic4_rev14_m3_canary as base
from src.topic4_rev14_fourier_field import array_sha256, mode_inventory


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV15_M3_COORDINATE_ATLAS_FROZEN"
PREPARE_STATUS = "REV15_M3_COORDINATE_ATLAS_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev15_m3_coordinate_atlas_v1"
MANIFEST_SCHEMA = "topic4_rev15_m3_coordinate_atlas_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(base.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev15_m3_coordinate_atlas.json",
    "scripts/freeze_topic4_rev15_m3_coordinate_atlas.py",
    "scripts/run_topic4_rev15_m3_coordinate_atlas_worker.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py",
    "src/topic4_rev14_fourier_field.py",
    "src/topic4_rev14_field_projection.py",
)

_jsonable = base._jsonable
_atomic_json = base._atomic_json
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
        raise RuntimeError("rev15 coordinate-atlas config schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev15 coordinate atlas must keep EE, E-to-I and Z/M off")
    if set(config.get("inputs", {})) != {
            "rev13_config", "rev13_exact_off_manifest"}:
        raise RuntimeError("rev15 coordinate-atlas input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev15 input hash is not frozen: {name}")
    design = config["m3_design"]
    expected = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 3,
        "expected_modes": 14,
        "expected_real_coefficients": 28,
        "atlas_coordinate_count": 28,
        "surface_rms_levels": [0.8],
        "signs": [-1, 1],
        "sheet_length_mm": 20.0,
        "quadrature_per_axis": 128,
        "coordinate_decimal_places": 13,
        "candidate_count": 58,
        "selectable_candidate_count": 56,
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, expected_value in expected.items():
        if design.get(key) != expected_value:
            raise RuntimeError(f"rev15 coordinate-atlas design changed: {key}")
    forbidden = (
        "patient", "contact", "shaft", "electrode", "gaussian", "prototype",
        "classifier", "heldout", "manual_core",
    )
    generation = str(design.get("candidate_generation_inputs", "")).lower()
    if any(token in generation for token in forbidden):
        raise RuntimeError("rev15 atlas generation uses observation geometry")
    reference = json.loads(
        (ROOT / "config/topic4_rev14_m3_canary.json").read_text()
    )
    if config.get("node_mapping") != reference["node_mapping"]:
        raise RuntimeError("rev15 changed the frozen absolute Node mapping")
    search = config["search"]
    if search.get("active_network_seeds") != [2331]:
        raise RuntimeError("rev15 active atlas seed changed")
    if search.get("canary_network_seeds") != [2331]:
        raise RuntimeError("rev15 canary seed changed")
    if search.get("replication_network_seeds") != [2332, 2333]:
        raise RuntimeError("rev15 replication pool changed")
    if float(search["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev15 atlas duration changed")


def frozen_signed_depth_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return base.frozen_signed_depth_audit(config)


def build_candidates(config: Mapping[str, Any]) -> tuple[list[dict], dict]:
    _validate_config(config)
    design = config["m3_design"]
    modes = mode_inventory(int(design["maximum_order"]))
    dimension = 2 * len(modes)
    if len(modes) != int(design["expected_modes"]) or dimension != int(
            design["expected_real_coefficients"]):
        raise RuntimeError("rev15 M3 coordinate inventory changed")
    basis = base._deterministic_physical_basis(
        modes, n_per_axis=int(design["quadrature_per_axis"]),
        sheet_length_mm=float(design["sheet_length_mm"]),
    )
    decimal_places = int(design["coordinate_decimal_places"])
    candidates = [
        base._coordinate_record(
            candidate_id="exact_off", field_kind="stage_ak_exact_off_benchmark",
            selectable=False, modes=modes, coefficients=None,
            decimal_places=decimal_places,
        ),
        base._coordinate_record(
            candidate_id="uniform_node", field_kind="zero_fourier_uniform_benchmark",
            selectable=False, modes=modes,
            coefficients=np.zeros((len(modes), 2), dtype=np.float64),
            target_rms=0.0, decimal_places=decimal_places,
        ),
    ]
    sign_pair_hashes: dict[str, dict[str, str]] = {}
    for coordinate_index in range(dimension):
        vector = np.zeros(dimension, dtype=np.float64)
        vector[coordinate_index] = 1.0
        unit = base._deterministic_normalize_shell_rms(
            vector.reshape(len(modes), 2), basis, target_rms=1.0,
        )
        mode_index, phase_index = divmod(coordinate_index, 2)
        phase = "cos" if phase_index == 0 else "sin"
        nx, ny = modes[mode_index]
        pair_key = f"c{coordinate_index:02d}"
        sign_pair_hashes[pair_key] = {}
        for sign in map(int, design["signs"]):
            coefficients = sign * float(design["surface_rms_levels"][0]) * unit
            sign_tag = "p" if sign > 0 else "m"
            record = base._coordinate_record(
                candidate_id=f"m3_c{coordinate_index:02d}_{sign_tag}_r08",
                field_kind="absolute_paired_phase_fourier_m3",
                selectable=True, modes=modes, coefficients=coefficients,
                direction_index=coordinate_index, sign=sign, target_rms=0.8,
                decimal_places=decimal_places,
            )
            record["coordinate_atlas"] = {
                "coordinate_index": coordinate_index,
                "mode_index": mode_index,
                "mode": [int(nx), int(ny)],
                "phase": phase,
                "ordered_basis_only": True,
            }
            candidates.append(record)
            sign_pair_hashes[pair_key][sign_tag] = record[
                "fourier_coordinate"
            ]["coefficients_sha256"]
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("rev15 coordinate-atlas candidate count changed")
    selectable = [row for row in candidates if row["selection_eligible"]]
    if len(selectable) != int(design["selectable_candidate_count"]):
        raise RuntimeError("rev15 selectable coordinate count changed")
    hashes = [
        row["fourier_coordinate"]["coefficients_sha256"]
        for row in candidates if row["fourier_coordinate"] is not None
    ]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("rev15 coordinate atlas contains duplicate fields")
    for coordinate_index in range(dimension):
        rows = [
            row for row in selectable
            if row["coordinate_atlas"]["coordinate_index"] == coordinate_index
        ]
        if len(rows) != 2:
            raise RuntimeError("rev15 coordinate lacks a sign pair")
        coefficients = [
            np.asarray(row["fourier_coordinate"]["coefficients"], dtype=np.float64)
            for row in sorted(rows, key=lambda row: row["fourier_coordinate"]["sign"])
        ]
        if not np.array_equal(coefficients[0], -coefficients[1]):
            raise RuntimeError("rev15 coordinate sign pair is not exact")
    audit = {
        "modes": [list(mode) for mode in modes],
        "modes_sha256": array_sha256(np.asarray(modes, dtype=np.int64)),
        "ordered_coordinate_count": dimension,
        "sign_pair_hashes": sign_pair_hashes,
        "candidate_generation_inputs": design["candidate_generation_inputs"],
        "observation_geometry_used": False,
        "predeclared_object_basis_used": False,
    }
    return candidates, audit


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    rev13_config, rev13_manifest, input_audit = base._load_inputs(
        config, artifact_root,
    )
    reconstruction = base._exact_off_reconstruction(rev13_config, rev13_manifest)
    candidates, direction_audit = build_candidates(config)
    return {
        "schema_id": MANIFEST_SCHEMA,
        "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": direction_audit,
        "signed_depth_audit": frozen_signed_depth_audit(config),
        "exact_off_reconstruction": reconstruction,
        "event_unit": copy.deepcopy(rev13_config["event_unit"]),
        "source_topology": copy.deepcopy(rev13_config["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": input_audit,
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
    config_path = args.config.resolve()
    if not args.prepare_only and args.expected_commit is None:
        parser.error("formal freeze requires --expected-commit")
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
        "n_selectable": sum(row["selection_eligible"] for row in payload["candidates"]),
        "formal_ready": bool(provenance["formal_ready"]),
        "output_written": not args.prepare_only,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
