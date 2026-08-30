#!/usr/bin/env python3
"""Prepare frozen joint-M3+M4 candidate config from a complete response tensor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev14_m3_canary as base  # noqa: E402
from src.topic4_rev14_fourier_field import array_sha256, mode_inventory  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_ANALYSIS_CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_response_analysis.json"
DEFAULT_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "m4_shell_coordinate_atlas/analysis/joint_m3_m4_response_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_joint_m3_m4_candidates.json"
FAMILY_ORDER = (
    "mean_a", "maximin_bprotected", "maximin_supportprotected",
    "consensus_sparse",
)
RMS_LEVELS = (0.4, 0.6, 0.8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feasible_directions(aggregate: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if aggregate.get("status") != "COMPLETE":
        raise RuntimeError("joint response aggregate is incomplete")
    if not aggregate.get("inventory", {}).get("complete_cartesian_product"):
        raise RuntimeError("M4-shell Cartesian product is incomplete")
    ranking = aggregate.get("ranking_contract", {})
    if any(ranking.get(key) is not False for key in (
        "natural_kmeans_used", "patient_heldout_used", "figure_used",
    )) or ranking.get("EE_EtoI_ZM") != "off":
        raise RuntimeError("joint candidate construction crossed a forbidden boundary")
    directions = {}
    for family in FAMILY_ORDER:
        row = (aggregate.get("robust_directions") or {}).get(family) or {}
        if family == "mean_a":
            feasible = row.get("direction") is not None
        elif family.startswith("maximin"):
            feasible = row.get("feasible_positive_margin") is True
        else:
            feasible = row.get("feasible") is True
        if not feasible:
            continue
        vector = np.asarray(row.get("direction"), dtype=np.float64)
        if vector.shape != (48,) or not np.isfinite(vector).all():
            raise RuntimeError(f"joint direction is malformed: {family}")
        if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"joint direction is not unit norm: {family}")
        directions[family] = vector
    if not directions:
        raise RuntimeError("no joint response direction is feasible")
    return directions


def candidate_blueprint(
    aggregate: Mapping[str, Any], *, n_per_axis: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directions = feasible_directions(aggregate)
    modes = mode_inventory(4)
    basis = base._deterministic_physical_basis(
        modes, n_per_axis=n_per_axis, sheet_length_mm=20.0,
    )
    rows, seen, duplicate = [], {}, []
    for family in FAMILY_ORDER:
        if family not in directions:
            continue
        raw = directions[family]
        for target_rms in RMS_LEVELS:
            coefficients = base._deterministic_normalize_shell_rms(
                raw.reshape(len(modes), 2), basis, target_rms=target_rms,
            )
            digest = array_sha256(coefficients)
            opposite = array_sha256(-coefficients)
            if digest in seen or opposite in seen:
                duplicate.append({
                    "family": family, "target_rms": target_rms,
                    "duplicate_of": seen.get(digest, seen.get(opposite)),
                    "sign_equivalent": opposite in seen,
                })
                continue
            candidate_id = f"joint_{family}_r{int(round(10 * target_rms)):02d}"
            seen[digest] = candidate_id
            rows.append({
                "candidate_id": candidate_id, "family": family,
                "target_rms": target_rms,
                "raw_direction": raw.tolist(),
                "raw_direction_sha256": array_sha256(raw),
                "coefficients": coefficients.tolist(),
                "coefficients_sha256": digest,
                "m3_l2_fraction": float(
                    np.linalg.norm(raw[:28]) / np.linalg.norm(raw)
                ),
                "m4_shell_l2_fraction": float(
                    np.linalg.norm(raw[28:]) / np.linalg.norm(raw)
                ),
            })
    return rows, {
        "feasible_direction_ids": list(directions),
        "candidate_ids": [row["candidate_id"] for row in rows],
        "deduplicated": duplicate,
        "candidate_count_including_exact": 1 + len(rows),
        "selectable_candidate_count": len(rows),
    }


def build_config(
    *, aggregate_path: Path, analysis_config_path: Path, artifact_root: Path,
) -> dict[str, Any]:
    aggregate = json.loads(aggregate_path.read_text())
    rows, audit = candidate_blueprint(aggregate)
    source_config = ROOT / "config/topic4_rev16_m4_shell_coordinate_atlas.json"
    source_manifest = artifact_root / (
        "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
        "m4_shell_coordinate_atlas/candidate_manifest.json"
    )
    rev13_config = ROOT / "config/topic4_rev13_node_zero_sum_recovery.json"
    rev13_manifest = artifact_root / (
        "results/topic4_sef_hfo/data_driven_node_dualmode_rev13/"
        "node_zero_sum_recovery_canary/candidate_manifest.json"
    )
    j14_config = ROOT / "config/topic4_rev14_static_node_historical_rescore.json"
    support_config = ROOT / "config/topic4_rev14_patient_support_acceptance.json"
    for path in (
        source_config, source_manifest, rev13_config, rev13_manifest,
        j14_config, support_config, analysis_config_path, aggregate_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"joint candidate source is missing: {path}")
    node_mapping = json.loads(source_config.read_text())["node_mapping"]
    return {
        "schema_id": "topic4_rev16_joint_m3_m4_candidates_v1",
        "scientific_role": "development_only_training_blind_joint_m3_m4_fresh_network_selection",
        "output_root": "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/joint_m3_m4_candidates",
        "candidate_manifest": "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/joint_m3_m4_candidates/candidate_manifest.json",
        "network_cache": "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache",
        "inputs": {
            "rev13_config": {"path": str(rev13_config.relative_to(ROOT)), "sha256": _sha256(rev13_config)},
            "rev13_exact_off_manifest": {"path": str(rev13_manifest.relative_to(artifact_root)), "sha256": _sha256(rev13_manifest)},
            "source_shell_config": {"path": str(source_config.relative_to(ROOT)), "sha256": _sha256(source_config)},
            "source_shell_manifest": {"path": str(source_manifest.relative_to(artifact_root)), "sha256": _sha256(source_manifest)},
            "joint_response_analysis_config": {"path": str(analysis_config_path.relative_to(ROOT)), "sha256": _sha256(analysis_config_path)},
            "joint_response_aggregate": {"path": str(aggregate_path.relative_to(artifact_root)), "sha256": _sha256(aggregate_path)},
            "j14_config": {"path": str(j14_config.relative_to(ROOT)), "sha256": _sha256(j14_config)},
            "patient_support_config": {"path": str(support_config.relative_to(ROOT)), "sha256": _sha256(support_config)},
        },
        "field_design": {
            "basis_family": "absolute_paired_phase_whole_sheet_fourier",
            "maximum_order": 4, "expected_modes": 24,
            "expected_real_coefficients": 48,
            "sheet_length_mm": 20.0, "quadrature_per_axis": 128,
            "coordinate_decimal_places": 13,
            "robust_direction_ids": audit["feasible_direction_ids"],
            "candidate_rms_levels": list(RMS_LEVELS),
            "candidate_ids": audit["candidate_ids"],
            "candidate_count": audit["candidate_count_including_exact"],
            "selectable_candidate_count": audit["selectable_candidate_count"],
            "deduplication_audit": audit["deduplicated"],
            "candidate_blueprint": rows,
            "basis_uses_observation_geometry": False,
            "basis_uses_predeclared_objects": False,
        },
        "node_mapping": node_mapping,
        "search": {
            "construction_network_seeds": [2331, 2332, 2333],
            "canary_network_seeds": [2351, 2352, 2353],
            "active_network_seeds": [2351, 2352, 2353],
            "common_random_numbers_across_candidates": True,
            "simulation": {
                "duration_ms": 20000.0, "early_stop_runaway": True,
                "late_runaway_is_invalid": True,
            },
        },
        "selection": {
            "fresh_A_improvement_required_networks": 3,
            "fresh_B_protection_required_networks": 3,
            "B_protection_ratio": 1.10,
            "equal_network_effective_support_minimum_per_mode": 6.0,
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
        },
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off", "Z_M": "off",
        },
        "resources": {
            "numerical_threads_per_worker": 1, "maximum_workers": 8,
            "recommended_workers": 6,
            "measured_canary_peak_rss_kib": 13356224,
            "worker_rss_safety_multiplier": 1.2,
            "stop_launching_below_available_memory_gib": 80,
            "emergency_stop_below_available_memory_gib": 64,
            "minimum_free_disk_gib": 40, "monitor_interval_seconds": 600,
            "long_run_launcher": "systemd-run --user plus nohup",
        },
        "claim_boundary": (
            "All feasible frozen joint M3+M4 directions are tested on three fresh "
            "development networks. Natural KMeans, held-out data, figures, EE, "
            "E-to-I and Z/M remain unavailable and this stage cannot itself freeze Node."
        ),
        "prepared_from": {
            "aggregate_status": aggregate["status"],
            "aggregate_sha256": _sha256(aggregate_path),
            "family_order": list(FAMILY_ORDER),
            "candidate_rms_levels": list(RMS_LEVELS),
            "blueprint_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True).encode()
            ).hexdigest(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_config(
        aggregate_path=args.aggregate.resolve(),
        analysis_config_path=args.analysis_config.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV16_JOINT_CANDIDATE_CONFIG_PREPARED",
        "output": str(args.output.resolve()),
        "n_candidates": payload["field_design"]["candidate_count"],
        "n_jobs": payload["field_design"]["candidate_count"] * 3,
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
