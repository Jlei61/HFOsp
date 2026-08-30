#!/usr/bin/env python3
"""Freeze the observation-free M4-minus-M3 coordinate response atlas."""
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
from src.topic4_rev14_fourier_field import (  # noqa: E402
    array_sha256,
    mode_inventory,
    mode_shell,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV16_M4_SHELL_COORDINATE_ATLAS_FROZEN"
PREPARE_STATUS = "REV16_M4_SHELL_COORDINATE_ATLAS_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev16_m4_shell_coordinate_atlas_v1"
MANIFEST_SCHEMA = "topic4_rev16_m4_shell_coordinate_atlas_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(base.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev16_m4_shell_coordinate_atlas.json",
    "scripts/freeze_topic4_rev16_m4_shell_coordinate_atlas.py",
    "scripts/run_topic4_rev16_m4_shell_coordinate_worker.py",
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
        raise RuntimeError("rev16 M4-shell config schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev16 M4 shell must keep EE, E-to-I and Z/M off")
    if set(config.get("inputs", {})) != {
        "rev13_config", "rev13_exact_off_manifest",
    }:
        raise RuntimeError("rev16 M4-shell input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev16 input hash is not frozen: {name}")
    design = config["field_design"]
    expected = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 4,
        "frozen_inner_order": 3,
        "expected_modes": 24,
        "expected_inner_modes": 14,
        "expected_shell_modes": 10,
        "expected_real_coefficients": 48,
        "shell_coordinate_count": 20,
        "surface_rms_levels": [0.8],
        "signs": [-1, 1],
        "sheet_length_mm": 20.0,
        "quadrature_per_axis": 128,
        "coordinate_decimal_places": 13,
        "candidate_count": 40,
        "selectable_candidate_count": 40,
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"rev16 M4-shell design changed: {key}")
    forbidden = (
        "patient", "contact", "shaft", "electrode", "gaussian", "prototype",
        "classifier", "heldout", "manual_core",
    )
    generation = str(design.get("candidate_generation_inputs", "")).lower()
    # The contract enumerates forbidden sources only to state that they are absent.
    if "only; no patient" not in generation or any(
        f"uses {token}" in generation for token in forbidden
    ):
        raise RuntimeError("rev16 M4-shell generation provenance is ambiguous")
    reference = json.loads((ROOT / "config/topic4_rev14_m3_canary.json").read_text())
    if config.get("node_mapping") != reference["node_mapping"]:
        raise RuntimeError("rev16 changed the frozen absolute Node mapping")
    search = config["search"]
    if search.get("active_network_seeds") != [2331, 2332, 2333]:
        raise RuntimeError("rev16 M4-shell construction networks changed")
    if float(search["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev16 M4-shell duration changed")


def frozen_signed_depth_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return base.frozen_signed_depth_audit(config)


def build_candidates(config: Mapping[str, Any]) -> tuple[list[dict], dict]:
    _validate_config(config)
    design = config["field_design"]
    modes = mode_inventory(int(design["maximum_order"]))
    inner_modes = mode_inventory(int(design["frozen_inner_order"]))
    shell_modes = mode_shell(
        int(design["frozen_inner_order"]), int(design["maximum_order"]),
    )
    if (
        len(modes) != int(design["expected_modes"])
        or len(inner_modes) != int(design["expected_inner_modes"])
        or len(shell_modes) != int(design["expected_shell_modes"])
        or 2 * len(modes) != int(design["expected_real_coefficients"])
    ):
        raise RuntimeError("rev16 Fourier inventory changed")
    if set(modes) != set(inner_modes).union(shell_modes):
        raise RuntimeError("M3 and M4 shell do not partition the M4 inventory")
    basis = base._deterministic_physical_basis(
        modes, n_per_axis=int(design["quadrature_per_axis"]),
        sheet_length_mm=float(design["sheet_length_mm"]),
    )
    decimal_places = int(design["coordinate_decimal_places"])
    mode_to_index = {mode: index for index, mode in enumerate(modes)}
    candidates: list[dict[str, Any]] = []
    sign_pair_hashes: dict[str, dict[str, str]] = {}
    for shell_mode_index, mode in enumerate(shell_modes):
        full_mode_index = mode_to_index[mode]
        for phase_index, phase in enumerate(("cos", "sin")):
            shell_coordinate_index = 2 * shell_mode_index + phase_index
            coefficients = np.zeros((len(modes), 2), dtype=np.float64)
            coefficients[full_mode_index, phase_index] = 1.0
            unit = base._deterministic_normalize_shell_rms(
                coefficients, basis, target_rms=1.0,
            )
            pair_key = f"c{shell_coordinate_index:02d}"
            sign_pair_hashes[pair_key] = {}
            for sign in map(int, design["signs"]):
                sign_tag = "p" if sign > 0 else "m"
                record = base._coordinate_record(
                    candidate_id=f"m4s_c{shell_coordinate_index:02d}_{sign_tag}_r08",
                    field_kind="absolute_paired_phase_fourier_m4_shell_coordinate",
                    selectable=True, modes=modes,
                    coefficients=sign * float(design["surface_rms_levels"][0]) * unit,
                    direction_index=shell_coordinate_index, sign=sign,
                    target_rms=0.8, decimal_places=decimal_places,
                )
                record["coordinate_atlas"] = {
                    "shell_coordinate_index": shell_coordinate_index,
                    "shell_mode_index": shell_mode_index,
                    "full_mode_index": full_mode_index,
                    "mode": [int(mode[0]), int(mode[1])],
                    "phase": phase,
                    "inner_m3_coefficients_all_zero": True,
                    "ordered_basis_only": True,
                }
                candidates.append(record)
                sign_pair_hashes[pair_key][sign_tag] = record[
                    "fourier_coordinate"
                ]["coefficients_sha256"]
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("rev16 M4-shell candidate count changed")
    hashes = [row["fourier_coordinate"]["coefficients_sha256"] for row in candidates]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("rev16 M4-shell atlas contains duplicate fields")
    for coordinate_index in range(int(design["shell_coordinate_count"])):
        rows = [
            row for row in candidates
            if row["coordinate_atlas"]["shell_coordinate_index"] == coordinate_index
        ]
        if len(rows) != 2:
            raise RuntimeError("rev16 M4-shell coordinate lacks a sign pair")
        rows.sort(key=lambda row: row["fourier_coordinate"]["sign"])
        negative = np.asarray(rows[0]["fourier_coordinate"]["coefficients"])
        positive = np.asarray(rows[1]["fourier_coordinate"]["coefficients"])
        if not np.array_equal(negative, -positive):
            raise RuntimeError("rev16 M4-shell sign pair is not exact")
        if np.count_nonzero(positive[:len(inner_modes)]) != 0:
            raise RuntimeError("rev16 shell coordinate contains an M3 coefficient")
    audit = {
        "full_modes": [list(mode) for mode in modes],
        "inner_modes": [list(mode) for mode in inner_modes],
        "shell_modes": [list(mode) for mode in shell_modes],
        "full_modes_sha256": array_sha256(np.asarray(modes, dtype=np.int64)),
        "shell_modes_sha256": array_sha256(np.asarray(shell_modes, dtype=np.int64)),
        "ordered_shell_coordinate_count": 2 * len(shell_modes),
        "sign_pair_hashes": sign_pair_hashes,
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
    rev13_config, rev13_manifest, input_audit = base._load_inputs(config, artifact_root)
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
    if args.prepare_only:
        print(json.dumps(payload, indent=2))
        return
    config = json.loads(config_path.read_text())
    output = args.artifact_root.resolve() / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "manifest": str(output),
        "n_candidates": len(payload["candidates"]),
        "n_jobs": len(payload["candidates"]) * len(
            config["search"]["active_network_seeds"]
        ),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
