#!/usr/bin/env python3
"""Freeze selected plus exact_off before unseen-network confirmation."""
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
from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as selection  # noqa: E402
from scripts import prepare_topic4_rev16_node_confirmation_config as prepare  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV16_NODE_CONFIRMATION_FROZEN"
PREPARE_STATUS = "REV16_NODE_CONFIRMATION_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = prepare.OUTPUT_SCHEMA
MANIFEST_SCHEMA = "topic4_rev16_node_confirmation_manifest_v1"
EXPECTED_PATHWAYS = copy.deepcopy(selection.EXPECTED_PATHWAYS)
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev16_node_confirmation.json",
    "scripts/prepare_topic4_rev16_node_confirmation_config.py",
    "scripts/freeze_topic4_rev16_node_confirmation.py",
    "scripts/run_topic4_rev16_node_confirmation_worker.py",
    "scripts/monitor_topic4_rev16_node_confirmation.py",
    "scripts/launch_topic4_rev16_node_confirmation.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py", "src/topic4_rev14_fourier_field.py",
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
        raise RuntimeError("rev16 confirmation schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev16 confirmation is not Node-only")
    if config.get("search", {}).get("active_network_seeds") != prepare.CONFIRMATION_SEEDS:
        raise RuntimeError("rev16 confirmation network pool changed")
    if config.get("field_design", {}).get("candidate_count") != 2:
        raise RuntimeError("rev16 confirmation candidate count changed")
    if config["field_design"].get("candidate_ids") != [
        "exact_off", config["selected_candidate"]["candidate_id"],
    ]:
        raise RuntimeError("rev16 confirmation candidate identity changed")
    if any(config.get("boundaries", {}).get(key) is not False for key in (
        "field_reranking_allowed", "patient_heldout_used", "natural_kmeans_used",
        "figure_used",
    )) or config["boundaries"].get("EE_EtoI_ZM") != "off":
        raise RuntimeError("rev16 confirmation crossed a scientific boundary")
    acceptance = config.get("confirmation_acceptance", {})
    expected_acceptance = {
        "J14_improvement_required_networks": 3,
        "A_improvement_required_networks": 3,
        "B_protection_required_networks": 3,
        "B_protection_ratio": 1.10,
        "equal_network_effective_support_minimum_per_mode": 6.0,
    }
    if acceptance != expected_acceptance:
        raise RuntimeError("rev16 confirmation acceptance contract changed")
    expected_inputs = {
        "selection_config", "selection_manifest", "selection_aggregate",
        "rev13_config", "rev13_exact_off_manifest", "j14_config",
        "patient_support_config",
    }
    if set(config.get("inputs", {})) != expected_inputs:
        raise RuntimeError("rev16 confirmation input set changed")
    if any(
        not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64
        for record in config["inputs"].values()
    ):
        raise RuntimeError("rev16 confirmation contains an unhashed input")
    simulation = config["search"].get("simulation", {})
    if (
        float(simulation.get("duration_ms", -1.0)) != 20000.0
        or simulation.get("early_stop_runaway") is not True
        or simulation.get("late_runaway_is_invalid") is not True
    ):
        raise RuntimeError("rev16 confirmation simulation contract changed")


def _load_hashed(
    config: Mapping[str, Any], name: str, artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    record = config["inputs"][name]
    path = _resolve(artifact_root, record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise RuntimeError(f"rev16 confirmation input changed: {name}")
    return path, json.loads(path.read_text())


def build_manifest_payload(
    config_path: Path, *, artifact_root: Path,
    provenance: Mapping[str, Any], status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    selection_config_path, selection_config = _load_hashed(
        config, "selection_config", artifact_root,
    )
    _, selection_manifest = _load_hashed(config, "selection_manifest", artifact_root)
    _, selection_aggregate = _load_hashed(config, "selection_aggregate", artifact_root)
    selection._validate_config(selection_config)
    if (
        selection_manifest.get("status") != selection.STATUS
        or selection_manifest.get("config_sha256") != _sha256(selection_config_path)
    ):
        raise RuntimeError("rev16 source selection manifest is not frozen")
    candidate_id = config["selected_candidate"]["candidate_id"]
    if (
        selection_aggregate.get("status") != "COMPLETE"
        or selection_aggregate.get("best_usable_anchor") != candidate_id
        or selection_aggregate.get("ranking_contract", {}).get(
            "J14_improvement"
        ) != "3/3 fresh networks"
    ):
        raise RuntimeError("rev16 source selection aggregate changed")
    source = {row["candidate_id"]: row for row in selection_manifest["candidates"]}
    candidates = [copy.deepcopy(source["exact_off"]), copy.deepcopy(source[candidate_id])]
    if candidates[1]["fourier_coordinate"]["coefficients_sha256"] != (
        config["selected_candidate"]["coefficients_sha256"]
    ):
        raise RuntimeError("rev16 selected coefficient hash changed")
    _, rev13_config = _load_hashed(config, "rev13_config", artifact_root)
    _, rev13_manifest = _load_hashed(
        config, "rev13_exact_off_manifest", artifact_root,
    )
    return {
        "schema_id": MANIFEST_SCHEMA, "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": {
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "field_reranking_performed": False,
            "selection_aggregate_sha256": config["inputs"][
                "selection_aggregate"
            ]["sha256"],
        },
        "signed_depth_audit": base.frozen_signed_depth_audit(config),
        "exact_off_reconstruction": base._exact_off_reconstruction(
            rev13_config, rev13_manifest,
        ),
        "event_unit": copy.deepcopy(selection_manifest["event_unit"]),
        "source_topology": copy.deepcopy(selection_manifest["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "confirmation_acceptance": copy.deepcopy(
            config["confirmation_acceptance"]
        ),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": {
            name: {"path": str(_load_hashed(config, name, artifact_root)[0]),
                   "sha256": record["sha256"], "verified": True}
            for name, record in config["inputs"].items()
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
        "status": payload["status"], "n_candidates": 2, "n_jobs": 6,
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
