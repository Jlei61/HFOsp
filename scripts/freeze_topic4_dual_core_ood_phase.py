#!/usr/bin/env python3
"""Freeze one phase of the OOD-guided dual-core experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_dual_core_ood import generate_sobol_candidates  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"
ROLE = (
    "development_only_ood_guided_dual_core_node_recovery_then_"
    "frozen_pathway_factorization"
)


def _source(config, key):
    record = config["inputs"][key]
    path = ROOT / record["path"]
    if _sha256(path) != record["sha256"]:
        raise RuntimeError(f"input hash changed: {record['path']}")
    return json.loads(path.read_text())


def _base_node(config):
    source = _source(config, "source_confirmation_manifest")
    row = next(
        copy.deepcopy(item) for item in source["candidate_set"]["candidates"]
        if item["candidate_id"] == "node_baseline"
    )
    if np.any(np.asarray(row["coefficients"], float) != 0.0):
        raise RuntimeError("source Node candidate has nonzero edge coefficients")
    return source, row


def _candidate_from_field(base, field, *, seeds, duration_ms, save_grid):
    row = copy.deepcopy(base)
    row.update({
        "candidate_id": field["candidate_id"],
        "arm": "Dual-core Node",
        "node_field": {
            **copy.deepcopy(field),
            "role": "strict binary two-core Node field",
            "component_count": 2,
            "peak_count_constraint": 2,
        },
        "allowed_network_seeds": list(map(int, seeds)),
        "simulation_duration_ms": float(duration_ms),
        "save_activity_grid": bool(save_grid),
        "search_coordinates": {
            "two_core_only": True,
            "edge_coefficients": "exact zero",
            "patient_ictal_target": "not read",
        },
    })
    return row


def _read_previous(config, phase):
    path = ROOT / config["output_root"] / phase / "candidate_manifest.json"
    return json.loads(path.read_text()), path


def _read_aggregate(config, phase):
    path = ROOT / config["output_root"] / phase / "aggregate.json"
    payload = json.loads(path.read_text())
    if payload.get("status") != "DUAL_CORE_OOD_PHASE_COMPLETE":
        raise RuntimeError(f"{phase} aggregate is incomplete")
    return payload, path


def _phase_fields(config, phase):
    search = config["search"]
    if phase == "fit":
        return generate_sobol_candidates(config["two_core_family"]), []
    previous_phase = "fit" if phase == "selection" else "selection"
    previous, manifest_path = _read_previous(config, previous_phase)
    aggregate, aggregate_path = _read_aggregate(config, previous_phase)
    n = (
        int(search["selection"]["n_screen_shortlist"])
        if phase == "selection"
        else int(search["selection"]["n_confirmation_candidates"])
    )
    selected_ids = [row["candidate_id"] for row in aggregate["ranking"][:n]]
    fields = {
        row["candidate_id"]: copy.deepcopy(row["node_field"])
        for row in previous["candidate_set"]["candidates"]
    }
    return [fields[candidate_id] for candidate_id in selected_ids], [
        {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
        {"path": str(aggregate_path.relative_to(ROOT)), "sha256": _sha256(aggregate_path)},
    ]


def build_manifest(config_path: Path, phase: str, expected_commit: str) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != ROLE:
        raise RuntimeError("dual-core OOD scientific role changed")
    for record in config["inputs"].values():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source, base = _base_node(config)
    search = config["search"]
    if phase in {"fit", "selection", "confirmation"}:
        fields, parent_inputs = _phase_fields(config, phase)
        seed_key = f"{phase}_network_seeds"
        seeds = search[seed_key]
        duration = search["duration_ms"][phase]
        candidates = [
            _candidate_from_field(
                base, field, seeds=seeds, duration_ms=duration,
                save_grid=(phase == "confirmation"),
            )
            for field in fields
        ]
    elif phase == "pathway":
        confirmation, confirmation_path = _read_previous(config, "confirmation")
        if len(confirmation["candidate_set"]["candidates"]) != 1:
            raise RuntimeError("confirmation did not freeze exactly one Node field")
        field = copy.deepcopy(
            confirmation["candidate_set"]["candidates"][0]["node_field"]
        )
        source_pathways = _source(config, "source_pathway_manifest")
        source_by_id = {
            row["candidate_id"]: row
            for row in source_pathways["candidate_set"]["candidates"]
        }
        candidates = []
        for source_id, label in zip(
            config["pathway_arms"]["source_candidate_ids"],
            config["pathway_arms"]["labels"],
        ):
            row = copy.deepcopy(source_by_id[source_id])
            suffix = {
                "node_baseline": "node",
                "joint_04_ee_only": "ee",
                "joint_04_etoi_only": "etoi",
                "joint_04_control": "both",
            }[source_id]
            row.update({
                "candidate_id": f"frozen_dualcore_{suffix}",
                "arm": label,
                "node_field": copy.deepcopy(field),
                "allowed_network_seeds": list(map(int, search["pathway_network_seeds"])),
                "simulation_duration_ms": float(search["duration_ms"]["pathway"]),
                "save_activity_grid": False,
                "search_coordinates": {
                    "frozen_dual_core": True,
                    "source_pathway_candidate_id": source_id,
                    "pathway_refit": False,
                },
            })
            coefficients = np.asarray(row["coefficients"], float)
            if array_sha256(coefficients) != row["coefficients_sha256"]:
                raise RuntimeError(f"pathway coefficient hash changed: {source_id}")
            candidates.append(row)
        parent_inputs = [{
            "path": str(confirmation_path.relative_to(ROOT)),
            "sha256": _sha256(confirmation_path),
        }]
        seeds = search["pathway_network_seeds"]
        duration = search["duration_ms"]["pathway"]
    else:
        raise ValueError(f"unknown phase: {phase}")

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if dirty or provenance["runtime_modules_dirty"] or not provenance[
        "runtime_modules_match_expected_commit"
    ]:
        raise RuntimeError("dual-core freezer runtime or config is not frozen")
    return {
        "status": "REV16_DUAL_CORE_OOD_PHASE_FROZEN",
        "phase": phase,
        "scientific_role": ROLE,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
        },
        "direction_classifier": copy.deepcopy(source["direction_classifier"]),
        "direction_classifier_source": copy.deepcopy(
            source["direction_classifier_source"]
        ),
        "fixed_contract": {
            "network_seeds": list(map(int, seeds)),
            "duration_ms": float(duration),
            "detector": copy.deepcopy(search["detector"]),
            "topology": "frozen",
            "delays": "frozen",
            "Z_M": "off",
            "OOD_primary": "all returned events; unreadable counts as OOD",
        },
        "parent_inputs": parent_inputs,
        "provenance": {**provenance, "config_dirty": dirty},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=("fit", "selection", "confirmation", "pathway"), required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT / config["output_root"] / args.phase / "candidate_manifest.json"
    if any((output.parent / "workers").glob("*.json")):
        raise RuntimeError(f"{args.phase} workers exist before manifest freeze")
    payload = build_manifest(Path(args.config), args.phase, args.expected_commit)
    atomic_write_json(payload, output)
    print(json.dumps({
        "phase": args.phase,
        "candidates": payload["candidate_set"]["n_candidates"],
        "seeds": payload["fixed_contract"]["network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
