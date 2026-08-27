#!/usr/bin/env python3
"""Freeze training-response-derived M3 fields before fresh-network runs."""
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
from src.topic4_rev14_fourier_field import array_sha256, mode_inventory  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV15_M3_RESPONSE_COMBINATIONS_FROZEN"
PREPARE_STATUS = "REV15_M3_RESPONSE_COMBINATIONS_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev15_m3_response_combinations_v1"
MANIFEST_SCHEMA = "topic4_rev15_m3_response_combinations_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(base.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev15_m3_response_combinations.json",
    "scripts/freeze_topic4_rev15_m3_response_combinations.py",
    "scripts/run_topic4_rev15_m3_response_combination_worker.py",
    "scripts/monitor_topic4_rev15_m3_response_combinations.py",
    "scripts/launch_topic4_rev15_m3_response_combinations.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py",
    "src/topic4_rev14_fourier_field.py",
    "src/topic4_rev14_field_projection.py",
)

_atomic_json = base._atomic_json
_jsonable = base._jsonable


def _sha256(path: Path) -> str:
    return base._sha256(path)


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
        raise RuntimeError("rev15 response-combination schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev15 response combinations must keep pathways off")
    if set(config.get("inputs", {})) != {
        "rev13_config", "rev13_exact_off_manifest", "coordinate_atlas_config",
        "coordinate_atlas_manifest", "coordinate_atlas_aggregate",
    }:
        raise RuntimeError("rev15 response-combination input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev15 combination input is not hashed: {name}")
    design = config["m3_design"]
    expected = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 3,
        "expected_modes": 14,
        "expected_real_coefficients": 28,
        "sheet_length_mm": 20.0,
        "quadrature_per_axis": 128,
        "coordinate_decimal_places": 13,
        "source_coordinate_rms": 0.8,
        "combination_rms_levels": [0.6, 0.8, 1.0],
        "combination_direction_ids": [
            "dense_a", "bprotected_a", "sparse4_a", "sparse8_a",
        ],
        "single_control_ids": ["m3_c09_m_r08", "m3_c07_p_r08"],
        "candidate_count": 15,
        "selectable_candidate_count": 14,
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"rev15 response-combination design changed: {key}")
    selection = config["selection"]
    if selection.get("best_overall_single") != "m3_c09_m_r08":
        raise RuntimeError("best atlas single changed before replication")
    if selection.get("best_A_and_B_single") != "m3_c07_p_r08":
        raise RuntimeError("best joint-improving single changed")
    for key in (
        "natural_kmeans_used", "patient_heldout_used", "ictal_data_used",
        "figure_or_image_used",
    ):
        if selection.get(key) is not False:
            raise RuntimeError(f"forbidden response-combination input used: {key}")
    if config["search"].get("source_atlas_network_seeds") != [2331]:
        raise RuntimeError("rev15 construction seed changed")
    if config["search"].get("active_network_seeds") != [2332, 2333]:
        raise RuntimeError("rev15 fresh replication seeds changed")
    if float(config["search"]["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev15 response-combination duration changed")


def _load_input(
    config: Mapping[str, Any], name: str, artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    record = config["inputs"][name]
    path = _resolve(artifact_root, str(record["path"]))
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise RuntimeError(f"rev15 response-combination input changed: {name}")
    return path, json.loads(path.read_text())


def _input_audit(config: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    audit = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, str(record["path"]))
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev15 response-combination input changed: {name}")
        audit[name] = {"path": str(path), "sha256": record["sha256"], "verified": True}
    return audit


def _response_vectors(aggregate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pairs = aggregate.get("signed_coordinate_pairs", [])
    if len(pairs) != 28:
        raise RuntimeError("rev15 signed coordinate atlas is incomplete")
    source_rms = 0.8
    g_a = np.zeros(28, dtype=np.float64)
    g_b = np.zeros(28, dtype=np.float64)
    for expected_index, row in enumerate(sorted(pairs, key=lambda item: item["coordinate_index"])):
        if int(row["coordinate_index"]) != expected_index:
            raise RuntimeError("rev15 coordinate response order changed")
        g_a[expected_index] = float(row["signed_A_response_half_difference"]) / source_rms
        g_b[expected_index] = float(row["signed_B_response_half_difference"]) / source_rms
    if not np.isfinite(g_a).all() or not np.isfinite(g_b).all():
        raise RuntimeError("rev15 response gradient is non-finite")
    return g_a, g_b


def _combination_directions(
    aggregate: Mapping[str, Any], atlas_manifest: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    g_a, g_b = _response_vectors(aggregate)
    dense = -g_a
    b_dot = float(np.dot(g_b, dense))
    b_norm2 = float(np.dot(g_b, g_b))
    if b_norm2 <= 0.0:
        raise RuntimeError("rev15 B response gradient has zero norm")
    bprotected = dense - max(0.0, b_dot) * g_b / b_norm2

    manifest_rows = {
        row["candidate_id"]: row for row in atlas_manifest.get("candidates", [])
    }
    eligible = list(aggregate.get("a_improving_b_preserving_candidate_ids", []))
    if len(eligible) < 8:
        raise RuntimeError("fewer than eight A-improving/B-preserving coordinates")

    def sparse(count: int) -> tuple[np.ndarray, list[str]]:
        selected = eligible[:count]
        vector = np.zeros(28, dtype=np.float64)
        seen = set()
        for candidate_id in selected:
            candidate = manifest_rows[candidate_id]
            coordinate_index = int(candidate["coordinate_atlas"]["coordinate_index"])
            if coordinate_index in seen:
                raise RuntimeError("sparse combination selected both signs of one coordinate")
            seen.add(coordinate_index)
            sign = int(candidate["fourier_coordinate"]["sign"])
            vector[coordinate_index] = sign * abs(float(g_a[coordinate_index]))
        return vector, selected

    sparse4, sparse4_ids = sparse(4)
    sparse8, sparse8_ids = sparse(8)
    directions = {
        "dense_a": dense,
        "bprotected_a": bprotected,
        "sparse4_a": sparse4,
        "sparse8_a": sparse8,
    }
    for name, vector in directions.items():
        if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0.0:
            raise RuntimeError(f"rev15 combination direction is invalid: {name}")
    audit = {
        "g_A": g_a.tolist(),
        "g_B": g_b.tolist(),
        "g_A_sha256": array_sha256(g_a),
        "g_B_sha256": array_sha256(g_b),
        "dense_predicted_first_order_B_change": b_dot,
        "bprotected_predicted_first_order_B_change": float(
            np.dot(g_b, bprotected)
        ),
        "harmful_B_component_removed": bool(b_dot > 0.0),
        "sparse4_source_candidates": sparse4_ids,
        "sparse8_source_candidates": sparse8_ids,
    }
    return directions, audit


def build_candidates(
    config: Mapping[str, Any], artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _validate_config(config)
    _, atlas_config = _load_input(config, "coordinate_atlas_config", artifact_root)
    _, atlas_manifest = _load_input(config, "coordinate_atlas_manifest", artifact_root)
    _, aggregate = _load_input(config, "coordinate_atlas_aggregate", artifact_root)
    if atlas_manifest.get("status") != atlas.STATUS:
        raise RuntimeError("source coordinate atlas is not formally frozen")
    if aggregate.get("status") != "COMPLETE":
        raise RuntimeError("source coordinate response aggregate is incomplete")
    if not aggregate.get("inventory", {}).get("complete_cartesian_product"):
        raise RuntimeError("source coordinate response inventory is incomplete")
    ranking = aggregate.get("ranking_contract", {})
    for key in ("natural_kmeans_used", "patient_heldout_used", "ictal_data_used"):
        if ranking.get(key) is not False:
            raise RuntimeError(f"source atlas crossed a forbidden boundary: {key}")
    if aggregate.get("best_coordinate_candidate") != config["selection"][
        "best_overall_single"
    ]:
        raise RuntimeError("frozen best-overall single differs from atlas ranking")
    ranked = aggregate.get("ranked_coordinate_responses", [])
    joint = sorted(
        (
            row for row in ranked
            if row.get("mode_A_improves") and float(row["delta_mode_B_vs_exact"]) < 0.0
        ),
        key=lambda row: (float(row["delta_mode_A_vs_exact"]), row["candidate_id"]),
    )
    if not joint or joint[0]["candidate_id"] != config["selection"][
        "best_A_and_B_single"
    ]:
        raise RuntimeError("frozen joint-improving single differs from atlas response")

    design = config["m3_design"]
    modes = mode_inventory(int(design["maximum_order"]))
    if [list(mode) for mode in modes] != atlas_manifest["direction_audit"]["modes"]:
        raise RuntimeError("rev15 response combinations changed the M3 basis")
    if config["node_mapping"] != atlas_config["node_mapping"]:
        raise RuntimeError("rev15 response combinations changed the Node mapping")
    basis = base._deterministic_physical_basis(
        modes, n_per_axis=int(design["quadrature_per_axis"]),
        sheet_length_mm=float(design["sheet_length_mm"]),
    )
    directions, response_audit = _combination_directions(aggregate, atlas_manifest)
    source_rows = {
        row["candidate_id"]: row for row in atlas_manifest["candidates"]
    }
    candidates = [copy.deepcopy(source_rows["exact_off"])]
    for candidate_id in design["single_control_ids"]:
        row = copy.deepcopy(source_rows[candidate_id])
        row["response_combination"] = {
            "family": "independent_single_coordinate_control",
            "source_candidate_id": candidate_id,
            "construction_seed_only": 2331,
        }
        candidates.append(row)
    decimal_places = int(design["coordinate_decimal_places"])
    for direction_id in design["combination_direction_ids"]:
        raw = directions[direction_id]
        for target_rms in map(float, design["combination_rms_levels"]):
            coefficients = base._deterministic_normalize_shell_rms(
                raw.reshape(len(modes), 2), basis, target_rms=target_rms,
            )
            rms_tag = f"r{int(round(10 * target_rms)):02d}"
            row = base._coordinate_record(
                candidate_id=f"combo_{direction_id}_{rms_tag}",
                field_kind="absolute_paired_phase_fourier_m3",
                selectable=True, modes=modes, coefficients=coefficients,
                direction_index=None, sign=None, target_rms=target_rms,
                decimal_places=decimal_places,
            )
            row["response_combination"] = {
                "family": direction_id,
                "target_centered_surface_rms": target_rms,
                "raw_direction_sha256": array_sha256(raw),
                "construction_seed_only": 2331,
            }
            candidates.append(row)
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("rev15 response-combination candidate count changed")
    selectable = [row for row in candidates if row["selection_eligible"]]
    if len(selectable) != int(design["selectable_candidate_count"]):
        raise RuntimeError("rev15 response-combination selectable count changed")
    hashes = [
        row["fourier_coordinate"]["coefficients_sha256"]
        for row in selectable
    ]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("rev15 response-combination fields contain duplicates")
    arrays = [
        np.asarray(row["fourier_coordinate"]["coefficients"], dtype=np.float64)
        for row in selectable
    ]
    for i, left in enumerate(arrays):
        for right in arrays[i + 1:]:
            if np.array_equal(left, -right):
                raise RuntimeError("rev15 response-combination fields are sign-equivalent")
    direction_audit = {
        **response_audit,
        "modes": [list(mode) for mode in modes],
        "modes_sha256": array_sha256(np.asarray(modes, dtype=np.int64)),
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "observation_geometry_used": False,
        "patient_training_scores_used_for_construction": True,
        "patient_heldout_used": False,
    }
    return candidates, direction_audit, atlas_manifest


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    candidates, direction_audit, atlas_manifest = build_candidates(
        config, artifact_root,
    )
    return {
        "schema_id": MANIFEST_SCHEMA,
        "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": direction_audit,
        "signed_depth_audit": copy.deepcopy(atlas_manifest["signed_depth_audit"]),
        "exact_off_reconstruction": copy.deepcopy(
            atlas_manifest["exact_off_reconstruction"]
        ),
        "event_unit": copy.deepcopy(atlas_manifest["event_unit"]),
        "source_topology": copy.deepcopy(atlas_manifest["source_topology"]),
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
        parser.error("formal combination freeze requires --expected-commit")
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
        "n_selectable": sum(row["selection_eligible"] for row in payload["candidates"]),
        "formal_ready": bool(provenance["formal_ready"]),
        "output_written": not args.prepare_only,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
