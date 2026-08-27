#!/usr/bin/env python3
"""Freeze the seed-2321 J14 M3 shortlist for network replication."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev14_m3_canary as canary  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV14_M3_NETWORK_REPLICATION_FROZEN"
PREPARE_STATUS = "REV14_M3_NETWORK_REPLICATION_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev14_m3_replication_v1"
MANIFEST_SCHEMA = "topic4_rev14_m3_replication_manifest_v1"
EXPECTED_PATHWAYS = canary.EXPECTED_PATHWAYS
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev14_m3_replication.json",
    "scripts/freeze_topic4_rev14_m3_replication.py",
    "scripts/run_topic4_rev14_m3_replication_worker.py",
    "scripts/monitor_topic4_rev14_m3_replication.py",
    "scripts/launch_topic4_rev14_m3_replication.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py",
    "src/topic4_rev14_fourier_field.py",
    "src/topic4_rev14_field_projection.py",
)

_jsonable = canary._jsonable
_atomic_json = canary._atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / str(relative)
    return local if local.exists() else artifact_root / str(relative)


def _git(arguments: list[str], *, text: bool = True):
    return subprocess.check_output(arguments, cwd=ROOT, text=text)


def _path_provenance(relative: str, expected_commit: str | None) -> dict[str, Any]:
    absolute = ROOT / relative
    if not absolute.is_file():
        raise RuntimeError(f"rev14 M3 replication runtime path is missing: {relative}")
    dirty = bool(_git([
        "git", "status", "--porcelain", "--untracked-files=all", "--", relative,
    ]).strip())
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    record = {
        "observed_sha256": _sha256(absolute),
        "tracked": tracked,
        "dirty": dirty,
        "expected_sha256": None,
        "matches_expected_commit": False,
    }
    if expected_commit is not None and tracked:
        try:
            committed = _git(
                ["git", "show", f"{expected_commit}:{relative}"], text=False,
            )
        except subprocess.CalledProcessError:
            committed = None
        if committed is not None:
            record["expected_sha256"] = hashlib.sha256(committed).hexdigest()
            record["matches_expected_commit"] = (
                record["observed_sha256"] == record["expected_sha256"]
            )
    return record


def runtime_provenance(
    config_path: Path, *, expected_commit: str | None, require_clean: bool,
) -> dict[str, Any]:
    relative_config = str(config_path.resolve().relative_to(ROOT))
    paths = tuple(dict.fromkeys((relative_config, *FORMAL_RUNTIME_PATHS)))
    current = _git(["git", "rev-parse", "HEAD"]).strip()
    expected = (
        _git(["git", "rev-parse", str(expected_commit)]).strip()
        if expected_commit is not None else None
    )
    records = {path: _path_provenance(path, expected) for path in paths}
    all_clean = all(not row["dirty"] for row in records.values())
    all_tracked = all(row["tracked"] for row in records.values())
    all_match = bool(expected) and all(
        row["matches_expected_commit"] for row in records.values()
    )
    commit_match = expected is not None and current == expected
    formal_ready = bool(commit_match and all_clean and all_tracked and all_match)
    if require_clean and not formal_ready:
        raise RuntimeError(
            "rev14 M3 replication provenance is not clean and frozen: "
            f"commit_match={commit_match}, all_clean={all_clean}, "
            f"all_tracked={all_tracked}, all_match={all_match}"
        )
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "commit_matches": commit_match,
        "all_explicit_paths_clean": all_clean,
        "all_explicit_paths_tracked": all_tracked,
        "all_explicit_paths_match_expected_commit": all_match,
        "formal_ready": formal_ready,
        "explicit_runtime_files": records,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != EXPECTED_SCHEMA:
        raise RuntimeError("rev14 M3 replication config schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev14 M3 replication must keep EE, E-to-I and Z/M off")
    if set(config.get("inputs", {})) != {
        "rev13_config", "rev13_exact_off_manifest", "m3_canary_config",
        "m3_canary_manifest", "m3_seed_2321_training_aggregate",
    }:
        raise RuntimeError("rev14 M3 replication input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"replication input hash is not frozen: {name}")
    selection = config.get("selection", {})
    selected = selection.get("selected_candidate_ids", [])
    if len(selected) != 8 or len(set(selected)) != 8:
        raise RuntimeError("rev14 M3 replication shortlist must contain eight fields")
    if selection.get("paired_nonselectable_benchmark") != "exact_off":
        raise RuntimeError("rev14 M3 replication requires exact_off paired references")
    if selection.get("patient_support_can_reorder") is not False:
        raise RuntimeError("patient support cannot reorder M3 replication")
    if selection.get("natural_kmeans_used") is not False:
        raise RuntimeError("natural KMeans is forbidden during M3 replication")
    if selection.get("patient_heldout_used") is not False:
        raise RuntimeError("patient held-out is forbidden during M3 replication")
    design = config.get("m3_design", {})
    expected_design = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 3,
        "expected_modes": 14,
        "expected_real_coefficients": 28,
        "sheet_length_mm": 20.0,
        "quadrature_per_axis": 128,
        "coordinate_decimal_places": 13,
        "candidate_count": 9,
        "selectable_candidate_count": 8,
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    if any(design.get(key) != value for key, value in expected_design.items()):
        raise RuntimeError("rev14 M3 replication design changed")
    if config.get("search", {}).get("active_network_seeds") != [2322, 2323]:
        raise RuntimeError("rev14 M3 replication seed pool changed")
    if float(config["search"]["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev14 M3 replication duration changed")


def _load_record(config: Mapping[str, Any], name: str, artifact_root: Path):
    record = config["inputs"][name]
    path = _resolve(artifact_root, record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise RuntimeError(f"rev14 M3 replication input changed: {name}")
    return path, json.loads(path.read_text())


def _input_audit(config: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    audit = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev14 M3 replication input changed: {name}")
        audit[name] = {
            "path": str(path), "sha256": record["sha256"], "verified": True,
        }
    return audit


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    _, canary_config = _load_record(config, "m3_canary_config", artifact_root)
    _, canary_manifest = _load_record(config, "m3_canary_manifest", artifact_root)
    _, aggregate = _load_record(
        config, "m3_seed_2321_training_aggregate", artifact_root,
    )
    if canary_manifest.get("status") != canary.STATUS:
        raise RuntimeError("source M3 canary manifest is not frozen")
    if canary_manifest.get("config_sha256") != config["inputs"][
            "m3_canary_config"]["sha256"]:
        raise RuntimeError("source M3 canary manifest/config mismatch")
    if aggregate.get("status") != "COMPLETE":
        raise RuntimeError("seed-2321 M3 aggregate is not complete")
    inventory = aggregate.get("inventory", {})
    if inventory.get("present_validated") != 34 or not inventory.get(
            "complete_cartesian_product"):
        raise RuntimeError("seed-2321 M3 aggregate inventory is incomplete")
    ranking_contract = aggregate.get("ranking_contract", {})
    forbidden = (
        ranking_contract.get("natural_kmeans_used"),
        ranking_contract.get("patient_heldout_used"),
        ranking_contract.get("ictal_data_used"),
        ranking_contract.get("figure_or_image_used"),
    )
    if any(value is not False for value in forbidden):
        raise RuntimeError("seed-2321 M3 ranking crossed a forbidden boundary")
    selected_ids = list(config["selection"]["selected_candidate_ids"])
    if aggregate.get("formal_ranking", [])[:8] != selected_ids:
        raise RuntimeError("replication shortlist differs from frozen J14 ranking")
    source_rows = {
        row["candidate_id"]: row for row in canary_manifest.get("candidates", [])
    }
    candidate_ids = ["exact_off", *selected_ids]
    if any(candidate_id not in source_rows for candidate_id in candidate_ids):
        raise RuntimeError("replication candidate is missing from M3 canary")
    candidates = [copy.deepcopy(source_rows[candidate_id]) for candidate_id in candidate_ids]
    if candidates[0].get("selection_eligible") is not False:
        raise RuntimeError("exact_off became selectable")
    if any(row.get("selection_eligible") is not True for row in candidates[1:]):
        raise RuntimeError("an M3 shortlist field is not selectable")
    source_design = canary_config["m3_design"]
    for key in (
        "basis_family", "maximum_order", "expected_modes",
        "expected_real_coefficients", "sheet_length_mm", "quadrature_per_axis",
        "coordinate_decimal_places", "basis_uses_observation_geometry",
        "basis_uses_predeclared_objects",
    ):
        if config["m3_design"][key] != source_design[key]:
            raise RuntimeError(f"M3 replication changed the field basis: {key}")
    if config["node_mapping"] != canary_config["node_mapping"]:
        raise RuntimeError("M3 replication changed the Node mapping")
    if config["inputs"]["rev13_config"] != canary_config["inputs"]["rev13_config"]:
        raise RuntimeError("M3 replication changed the rev13 config")
    if config["inputs"]["rev13_exact_off_manifest"] != canary_config[
            "inputs"]["rev13_exact_off_manifest"]:
        raise RuntimeError("M3 replication changed the exact_off source")
    return {
        "schema_id": MANIFEST_SCHEMA,
        "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": copy.deepcopy(canary_manifest["direction_audit"]),
        "signed_depth_audit": copy.deepcopy(canary_manifest["signed_depth_audit"]),
        "exact_off_reconstruction": copy.deepcopy(
            canary_manifest["exact_off_reconstruction"]
        ),
        "event_unit": copy.deepcopy(canary_manifest["event_unit"]),
        "source_topology": copy.deepcopy(canary_manifest["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": _input_audit(config, artifact_root),
        "selection": copy.deepcopy(config["selection"]),
        "seed_2321_reference": {
            "aggregate_sha256": config["inputs"][
                "m3_seed_2321_training_aggregate"
            ]["sha256"],
            "exact_off_j14": next(
                row["j14_v1_summary"]["objective"]
                for row in aggregate["per_run"]
                if row["candidate_id"] == "exact_off"
            ),
            "selected_j14": {
                row["candidate_id"]: row["j14_v1_summary"]["objective"]
                for row in aggregate["per_run"]
                if row["candidate_id"] in selected_ids
            },
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
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    _validate_config(config)
    provenance = runtime_provenance(
        config_path, expected_commit=args.expected_commit,
        require_clean=not args.prepare_only,
    )
    payload = build_manifest_payload(
        config_path, artifact_root=artifact_root, provenance=provenance,
        status=PREPARE_STATUS if args.prepare_only else STATUS,
    )
    output = artifact_root / config["candidate_manifest"]
    if not args.prepare_only:
        _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "manifest": str(output),
        "n_candidates": len(payload["candidates"]),
        "n_selectable": sum(
            bool(row["selection_eligible"]) for row in payload["candidates"]
        ),
        "active_network_seeds": payload["search"]["active_network_seeds"],
        "formal_ready": provenance["formal_ready"],
        "written": not args.prepare_only,
    }, indent=2))


if __name__ == "__main__":
    main()
