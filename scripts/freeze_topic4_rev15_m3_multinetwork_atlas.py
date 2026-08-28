#!/usr/bin/env python3
"""Freeze an exact candidate copy of the M3 atlas on fresh networks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev14_m3_canary as base  # noqa: E402
from scripts import freeze_topic4_rev15_m3_coordinate_atlas as source  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV15_M3_MULTINETWORK_ATLAS_FROZEN"
PREPARE_STATUS = "REV15_M3_MULTINETWORK_ATLAS_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev15_m3_multinetwork_atlas_v1"
MANIFEST_SCHEMA = "topic4_rev15_m3_multinetwork_atlas_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(base.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev15_m3_multinetwork_atlas.json",
    "scripts/freeze_topic4_rev15_m3_multinetwork_atlas.py",
    "scripts/run_topic4_rev15_m3_multinetwork_atlas_worker.py",
    "scripts/monitor_topic4_rev15_m3_multinetwork_atlas.py",
    "scripts/launch_topic4_rev15_m3_multinetwork_atlas.py",
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


def _load_hashed(
    config: Mapping[str, Any], name: str, artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    record = config["inputs"][name]
    path = _resolve(artifact_root, str(record["path"]))
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise RuntimeError(f"rev15 multinetwork input changed: {name}")
    return path, json.loads(path.read_text())


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != EXPECTED_SCHEMA:
        raise RuntimeError("rev15 multinetwork-atlas schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev15 multinetwork atlas must keep pathways off")
    if set(config.get("inputs", {})) != {
        "rev13_config", "rev13_exact_off_manifest",
        "source_atlas_config", "source_atlas_manifest",
    }:
        raise RuntimeError("rev15 multinetwork-atlas input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev15 multinetwork input is not frozen: {name}")
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
        "source_candidate_set": "exact_copy_of_seed2331_coordinate_atlas",
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"rev15 multinetwork design changed: {key}")
    if config.get("node_mapping") != json.loads(
        (ROOT / "config/topic4_rev15_m3_coordinate_atlas.json").read_text()
    )["node_mapping"]:
        raise RuntimeError("rev15 multinetwork atlas changed Node mapping")
    search = config["search"]
    if search.get("source_atlas_network_seeds") != [2331]:
        raise RuntimeError("rev15 source-atlas seed changed")
    if search.get("canary_network_seeds") != [2332, 2333]:
        raise RuntimeError("rev15 compatibility seed pool changed")
    if search.get("active_network_seeds") != [2332, 2333]:
        raise RuntimeError("rev15 multinetwork seed pool changed")
    if float(search["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev15 multinetwork duration changed")


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    _, source_config = _load_hashed(config, "source_atlas_config", artifact_root)
    source_path, source_manifest = _load_hashed(
        config, "source_atlas_manifest", artifact_root,
    )
    if source_manifest.get("status") != source.STATUS:
        raise RuntimeError("source M3 atlas is not formally frozen")
    if source_manifest.get("config_sha256") != config["inputs"][
        "source_atlas_config"
    ]["sha256"]:
        raise RuntimeError("source M3 atlas config hash changed")
    candidates = copy.deepcopy(source_manifest.get("candidates", []))
    if len(candidates) != 58 or sum(
        bool(row.get("selection_eligible")) for row in candidates
    ) != 56:
        raise RuntimeError("source M3 candidate inventory changed")
    return {
        "schema_id": MANIFEST_SCHEMA,
        "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": copy.deepcopy(source_manifest["direction_audit"]),
        "signed_depth_audit": copy.deepcopy(source_manifest["signed_depth_audit"]),
        "exact_off_reconstruction": copy.deepcopy(
            source_manifest["exact_off_reconstruction"]
        ),
        "event_unit": copy.deepcopy(source_manifest["event_unit"]),
        "source_topology": copy.deepcopy(source_manifest["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": {
            name: {
                "path": str(_load_hashed(config, name, artifact_root)[0]),
                "sha256": record["sha256"], "verified": True,
            }
            for name, record in config["inputs"].items()
        },
        "source_atlas": {
            "manifest": str(source_path),
            "manifest_sha256": config["inputs"]["source_atlas_manifest"]["sha256"],
            "candidate_payload_exact_copy": True,
            "source_network_seeds": source_config["search"]["active_network_seeds"],
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
    if args.prepare_only:
        print(json.dumps(payload, indent=2))
        return
    config = json.loads(config_path.read_text())
    output = args.artifact_root.resolve() / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "manifest": str(output),
        "n_candidates": len(payload["candidates"]),
        "n_jobs": len(payload["candidates"]) * 2,
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
