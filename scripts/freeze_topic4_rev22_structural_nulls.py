#!/usr/bin/env python3
"""Freeze rev22 Task 10b structural controls after candidate selection.

This producer reads only the already-frozen candidate identities and structural
source manifests.  It does not read held-out, KMeans, OOD, or patient-ictal
endpoints.  The output is a dedicated execution config plus an 18-candidate
manifest consumed by the existing rev12 Node worker.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE = ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"
sys.path.insert(0, str(ROOT))

from src.topic4_rev20_dual_core_mechanism import dual_core_field_sha256  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, expected: str | None = None) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing frozen input: {path}")
    if expected is not None and _sha256(path) != expected:
        raise RuntimeError(f"frozen input hash changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"frozen input is not an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return _sha256(path)


def _candidates(manifest: Mapping[str, Any]) -> list[dict]:
    rows = manifest.get("candidates")
    if rows is None:
        rows = (manifest.get("candidate_set") or {}).get("candidates")
    if not isinstance(rows, list):
        raise RuntimeError("candidate manifest has no candidate list")
    return rows


def _one(rows: list[dict], candidate_id: str) -> dict:
    matches = [row for row in rows if str(row.get("candidate_id")) == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen candidate {candidate_id}, found {len(matches)}")
    return copy.deepcopy(matches[0])


def matched_norm_rows(coefficients: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two independent row-specific controls with exact source L2 norm."""
    source = np.asarray(coefficients, dtype=np.float64)
    if source.shape != (2, 6) or not np.isfinite(source).all():
        raise RuntimeError("learned coefficient matrix must be finite with shape (2, 6)")
    rng = np.random.default_rng(int(seed))
    controls = []
    for row_index in range(2):
        random_row = rng.standard_normal(source.shape[1])
        random_row *= np.linalg.norm(source[row_index]) / np.linalg.norm(random_row)
        replacement = source.copy()
        replacement[row_index] = random_row
        controls.append(replacement)
    return controls[0], controls[1]


def random_two_core_centers(seed: int) -> list[list[float]]:
    """Patient-blind placement draw with boundary and separation constraints."""
    rng = np.random.default_rng(int(seed))
    for _ in range(10000):
        centers = rng.uniform(2.0, 18.0, size=(2, 2))
        if np.linalg.norm(centers[0] - centers[1]) >= 10.0:
            order = np.lexsort((centers[:, 1], centers[:, 0]))
            return centers[order].tolist()
    raise RuntimeError("failed to draw separated random core centers")


def _manual_field(centers: list[list[float]], target_count: int) -> dict:
    return {
        "field_type": "manual_dual_core_budget_matched",
        "centers_mm": centers,
        "target_count": int(target_count),
        "field_sha256": dual_core_field_sha256(centers, int(target_count)),
    }


def build_structural_candidates(*, analysis: Mapping[str, Any], execution: Mapping[str, Any],
                                frozen: Mapping[str, Any], execution_manifest: Mapping[str, Any],
                                substrate_manifest: Mapping[str, Any],
                                node_manifests: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    contract = analysis["structural_nulls"]
    execution_rows = _candidates(execution_manifest)
    reference = _one(execution_rows, "dci_p000")
    full_ids = (frozen.get("mask_to_candidates") or {}).get("M1111") or []
    if not full_ids:
        full_ids = (frozen.get("mask_to_candidates") or {}).get("M1100") or []
    if not full_ids:
        raise RuntimeError("Task 10b requires a frozen full-model proposal")
    full_options = [_one(execution_rows, str(candidate_id)) for candidate_id in full_ids]
    primary = [row for row in full_options if row.get("proposal_origin") == "gp"]
    if not primary:
        primary = [row for row in full_options if row.get("proposal_origin") == "observed"]
    full = copy.deepcopy(primary[0] if primary else full_options[0])
    full["structural_null_primary_selection"] = {
        "rule": "gp_then_observed_then_frozen_manifest_order",
        "candidate_id": full["candidate_id"],
        "all_full_model_candidates": [row["candidate_id"] for row in full_options],
        "validation_endpoints_used": False,
    }
    conditions = (("M0000", reference), ("full", full))

    substrate = _one(_candidates(substrate_manifest),
                     str(execution["reference"]["base_substrate_candidate_id"]))
    coefficients = np.asarray(substrate["coefficients"], dtype=np.float64)
    random_rows = matched_norm_rows(coefficients, int(contract["random_learned_row_seed"]))
    target_count = int(analysis["dual_core_anchor"]["target_count"])
    original_centers = np.asarray(analysis["dual_core_anchor"]["centers_mm"], dtype=float)
    midpoint = np.mean(original_centers, axis=0).tolist()
    placements = {
        "merged_midpoint_core": _manual_field([midpoint, midpoint], target_count),
        "random_two_core_centers": _manual_field(
            random_two_core_centers(int(contract["random_two_core_seed"])), target_count),
    }

    rows: list[dict] = []
    variants = list(contract["fixed_topology_variants"])
    if variants != ["r180", "r90", "matched_norm_row_1", "matched_norm_row_2",
                    "merged_midpoint_core", "random_two_core_centers"]:
        raise RuntimeError("fixed topology null list changed")
    for variant in variants:
        for condition, source in conditions:
            row = copy.deepcopy(source)
            row["candidate_id"] = f"sn_{variant}_{condition}"
            row["selection_eligible"] = False
            row["structural_null"] = {
                "family": "fixed_topology", "variant": variant, "condition": condition,
                "pairing": "paired_by_topology_seed",
            }
            if variant in {"r180", "r90"}:
                row["field_transform"] = variant
            elif variant.startswith("matched_norm_row_"):
                index = int(variant.rsplit("_", 1)[1]) - 1
                values = random_rows[index]
                row["edge_coefficients_override"] = {
                    "values": values.tolist(), "sha256": array_sha256(values),
                    "source_row_replaced": index,
                    "source_row_l2_norm": float(np.linalg.norm(coefficients[index])),
                }
            else:
                row["node_field"] = placements[variant]
            rows.append(row)

    for condition, source in conditions:
        row = copy.deepcopy(source)
        row["candidate_id"] = f"sn_isotropic_graph_{condition}"
        row["selection_eligible"] = False
        row["topology_override"] = {
            "graph_aspect_ratio": 1.0,
            "pairing": "unpaired_rebuilt_topology_control",
        }
        row["structural_null"] = {
            "family": "isotropic_graph", "variant": "isotropic_graph",
            "condition": condition, "pairing": "unpaired_rebuilt_topology_control",
            "interpretation": ("M0000 leaves the AR=1 sampled graph unreweighted; full applies "
                               "the same frozen fixed-topology operator used by the selected model"),
        }
        rows.append(row)

    for record in contract["node_blocking_factors"]:
        source_manifest = node_manifests[str(record["candidate_id"])]
        node_source = _one(_candidates(source_manifest), str(record["candidate_id"]))
        node_field = copy.deepcopy(node_source.get("node_field") or node_source)
        dispersion = copy.deepcopy(node_source.get("node_dispersion_field"))
        mapping = copy.deepcopy(node_source.get("node_mapping") or {
            "node_gain": 1.0, "signed_depth_shrinkage": 1.0,
        })
        for condition, source in conditions:
            row = copy.deepcopy(source)
            row["candidate_id"] = f"sn_node_{record['candidate_id']}_{condition}"
            row["selection_eligible"] = False
            row["node_field"] = node_field
            row["node_mapping"] = mapping
            if dispersion is not None:
                row["node_dispersion_field"] = dispersion
            else:
                row.pop("node_dispersion_field", None)
            row["structural_null"] = {
                "family": "node_blocking_factor", "variant": record["candidate_id"],
                "condition": condition, "pairing": "paired_by_topology_seed",
                "historical_status": record["historical_status"],
            }
            rows.append(row)
    if len(rows) != 18 or len({r["candidate_id"] for r in rows}) != 18:
        raise RuntimeError("Task 10b must freeze exactly 18 unique candidates")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json")
    parser.add_argument("--base-execution-config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_response_execution.json")
    parser.add_argument("--execution-manifest", type=Path,
                        default=STAGE / "response_fit/final_execution_candidate_manifest.json")
    parser.add_argument("--frozen-candidates", type=Path,
                        default=STAGE / "response_fit/frozen_candidates.json")
    parser.add_argument("--out-config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_structural_null_execution.json")
    parser.add_argument("--out-manifest", type=Path,
                        default=STAGE / "structural_nulls/candidate_manifest.json")
    args = parser.parse_args()

    analysis = _read(args.analysis_config)
    execution = _read(args.base_execution_config)
    frozen = _read(args.frozen_candidates)
    execution_manifest = _read(args.execution_manifest)
    transition_path = ROOT / execution["inputs"]["transition_config"]["path"]
    transition = _read(transition_path, execution["inputs"]["transition_config"]["sha256"])
    substrate_record = transition["inputs"]["frozen_substrate_manifest"]
    substrate_path = ARTIFACT_ROOT / substrate_record["path"]
    substrate_manifest = _read(substrate_path, substrate_record["sha256"])
    node_manifests = {}
    for record in analysis["structural_nulls"]["node_blocking_factors"]:
        path = ARTIFACT_ROOT / record["source_manifest"]
        node_manifests[record["candidate_id"]] = _read(path, record["source_manifest_sha256"])

    rows = build_structural_candidates(
        analysis=analysis, execution=execution, frozen=frozen,
        execution_manifest=execution_manifest, substrate_manifest=substrate_manifest,
        node_manifests=node_manifests,
    )
    config = copy.deepcopy(execution)
    config["candidate_manifest"] = str(args.out_manifest.resolve().relative_to(ARTIFACT_ROOT))
    # Keep the worker's already-audited scientific role; this is a validation
    # phase of the same rev22 experiment, not a new modelling family.
    config["scientific_role"] = (
        "development_only_frozen_dual_core_interictal_connectivity_identifiability"
    )
    config["structural_null_execution"] = {
        "candidate_count": 18, "seed_count": 6, "expected_trajectories": 108,
        "selection_eligible": False,
    }
    config["frozen_contracts"]["frozen_candidates"] = {
        "path": str(args.frozen_candidates.resolve()), "sha256": _sha256(args.frozen_candidates),
    }
    config["frozen_contracts"]["final_execution_manifest"] = {
        "path": str(args.execution_manifest.resolve()), "sha256": _sha256(args.execution_manifest),
    }
    for record in analysis["structural_nulls"]["node_blocking_factors"]:
        key = f"node_factor_{record['candidate_id']}"
        path = ARTIFACT_ROOT / record["source_manifest"]
        config["frozen_contracts"][key] = {"path": str(path), "sha256": _sha256(path)}
    config_hash = _write(args.out_config, config)
    seed_hash = str(config["frozen_contracts"]["seed_manifest"]["sha256"])
    response_hash = str(execution_manifest["response_design_manifest_sha256"])
    manifest = {
        "schema_id": "topic4_rev22_dci_execution_candidate_manifest_v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip(),
        "config_sha256": config_hash,
        "seed_manifest_sha256": seed_hash,
        "response_design_manifest_sha256": response_hash,
        "branch": frozen.get("branch"),
        "candidate_count": len(rows),
        "candidates": rows,
        "source_hashes": {
            "analysis_config": _sha256(args.analysis_config),
            "base_execution_config": _sha256(args.base_execution_config),
            "frozen_candidates": _sha256(args.frozen_candidates),
            "final_execution_manifest": _sha256(args.execution_manifest),
            "frozen_substrate_manifest": _sha256(substrate_path),
        },
        "claim_boundary": ("Selection-blind structural sensitivity controls. Fixed-topology "
                           "arms are paired; rebuilt AR=1 topology is unpaired; Node factors "
                           "are historical controls, not alternative winners."),
    }
    manifest_hash = _write(args.out_manifest, manifest)
    print(json.dumps({"status": "STRUCTURAL_NULLS_FROZEN", "candidates": len(rows),
                      "trajectories": len(rows) * 6, "config_sha256": config_hash,
                      "manifest_sha256": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()
