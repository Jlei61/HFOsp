#!/usr/bin/env python3
"""Mechanically prepare the robust-field config after the response tensor closes."""
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
DEFAULT_ANALYSIS_CONFIG = ROOT / "config/topic4_rev15_m3_multinetwork_response_analysis.json"
DEFAULT_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_multinetwork_atlas/analysis/m3_multinetwork_response_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev15_m3_robust_candidates.json"
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
        raise RuntimeError("multinetwork response aggregate is incomplete")
    if not aggregate.get("inventory", {}).get("complete_cartesian_product"):
        raise RuntimeError("multinetwork response Cartesian product is incomplete")
    ranking = aggregate.get("ranking_contract", {})
    if any(ranking.get(key) is not False for key in (
        "natural_kmeans_used", "patient_heldout_used", "figure_used",
    )):
        raise RuntimeError("robust candidate construction crossed a forbidden boundary")
    if ranking.get("EE_EtoI_ZM") != "off":
        raise RuntimeError("robust candidate construction activated another mechanism")
    source = aggregate.get("robust_directions") or {}
    directions = {}
    for family in FAMILY_ORDER:
        row = source.get(family) or {}
        feasible = (
            row.get("feasible_positive_margin") is True
            if family.startswith("maximin") else row.get("feasible") is True
        )
        if family == "mean_a":
            feasible = row.get("direction") is not None
        if not feasible:
            continue
        vector = np.asarray(row.get("direction"), dtype=np.float64)
        if vector.shape != (28,) or not np.isfinite(vector).all():
            raise RuntimeError(f"robust direction is malformed: {family}")
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"robust direction is not unit norm: {family}")
        directions[family] = vector
    if not directions:
        raise RuntimeError("no robust direction is feasible")
    return directions


def candidate_blueprint(
    aggregate: Mapping[str, Any], *, n_per_axis: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directions = feasible_directions(aggregate)
    modes = mode_inventory(3)
    basis = base._deterministic_physical_basis(
        modes, n_per_axis=n_per_axis, sheet_length_mm=20.0,
    )
    rows, seen, signed = [], {}, []
    for family in FAMILY_ORDER:
        if family not in directions:
            continue
        raw = directions[family]
        for target_rms in RMS_LEVELS:
            coefficients = base._deterministic_normalize_shell_rms(
                raw.reshape(len(modes), 2), basis, target_rms=target_rms,
            )
            digest = array_sha256(coefficients)
            signed_digest = array_sha256(-coefficients)
            if digest in seen or signed_digest in seen:
                signed.append({
                    "family": family, "target_rms": target_rms,
                    "duplicate_of": seen.get(digest, seen.get(signed_digest)),
                    "sign_equivalent": signed_digest in seen,
                })
                continue
            candidate_id = f"robust_{family}_r{int(round(10 * target_rms)):02d}"
            seen[digest] = candidate_id
            rows.append({
                "candidate_id": candidate_id, "family": family,
                "target_rms": target_rms,
                "raw_direction": raw.tolist(),
                "raw_direction_sha256": array_sha256(raw),
                "coefficients": coefficients.tolist(),
                "coefficients_sha256": digest,
            })
    return rows, {
        "feasible_direction_ids": list(directions),
        "candidate_ids": [row["candidate_id"] for row in rows],
        "deduplicated": signed,
        "candidate_count_including_exact": 1 + len(rows),
        "selectable_candidate_count": len(rows),
    }


def build_config(
    *, aggregate_path: Path, analysis_config_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    aggregate = json.loads(aggregate_path.read_text())
    analysis_config = json.loads(analysis_config_path.read_text())
    rows, audit = candidate_blueprint(aggregate)
    source_config = ROOT / "config/topic4_rev15_m3_coordinate_atlas.json"
    source_manifest = artifact_root / (
        "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
        "m3_coordinate_atlas/candidate_manifest.json"
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
        j14_config, support_config,
    ):
        if not path.is_file():
            raise RuntimeError(f"robust candidate source is missing: {path}")
    node_mapping = json.loads(source_config.read_text())["node_mapping"]
    return {
        "schema_id": "topic4_rev15_m3_robust_candidates_v1",
        "scientific_role": "development_only_training_blind_robust_m3_candidate_replication",
        "output_root": "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/m3_robust_candidates",
        "candidate_manifest": "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/m3_robust_candidates/candidate_manifest.json",
        "network_cache": "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache",
        "inputs": {
            "rev13_config": {"path": str(rev13_config.relative_to(ROOT)), "sha256": _sha256(rev13_config)},
            "rev13_exact_off_manifest": {"path": str(rev13_manifest.relative_to(artifact_root)), "sha256": _sha256(rev13_manifest)},
            "source_atlas_config": {"path": str(source_config.relative_to(ROOT)), "sha256": _sha256(source_config)},
            "source_atlas_manifest": {"path": str(source_manifest.relative_to(artifact_root)), "sha256": _sha256(source_manifest)},
            "multinetwork_response_analysis_config": {"path": str(analysis_config_path.relative_to(ROOT)), "sha256": _sha256(analysis_config_path)},
            "multinetwork_response_aggregate": {"path": str(aggregate_path.relative_to(artifact_root)), "sha256": _sha256(aggregate_path)},
            "j14_config": {"path": str(j14_config.relative_to(ROOT)), "sha256": _sha256(j14_config)},
            "patient_support_config": {"path": str(support_config.relative_to(ROOT)), "sha256": _sha256(support_config)},
        },
        "m3_design": {
            "basis_family": "absolute_paired_phase_whole_sheet_fourier",
            "maximum_order": 3, "expected_modes": 14,
            "expected_real_coefficients": 28,
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
            "canary_network_seeds": [2341, 2342, 2343],
            "active_network_seeds": [2341, 2342, 2343],
            "common_random_numbers_across_candidates": True,
            "simulation": {"duration_ms": 20000.0, "early_stop_runaway": True, "late_runaway_is_invalid": True},
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
            "numerical_threads_per_worker": 1, "maximum_workers": 10,
            "recommended_workers": 9, "measured_canary_peak_rss_kib": 13356224,
            "worker_rss_safety_multiplier": 1.2,
            "stop_launching_below_available_memory_gib": 64,
            "emergency_stop_below_available_memory_gib": 48,
            "minimum_free_disk_gib": 40, "monitor_interval_seconds": 600,
            "long_run_launcher": "systemd-run --user plus nohup",
        },
        "claim_boundary": (
            "All feasible pre-registered robust directions are evaluated on three "
            "fresh development networks. This stage cannot use natural KMeans, held-out "
            "data, figures, EE, E-to-I or Z/M and cannot itself freeze Node."
        ),
        "prepared_from": {
            "aggregate_status": aggregate["status"],
            "aggregate_sha256": _sha256(aggregate_path),
            "analysis_family_order": list(FAMILY_ORDER),
            "analysis_candidate_rms_levels": list(RMS_LEVELS),
            "blueprint_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
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
        "status": "REV15_M3_ROBUST_CANDIDATE_CONFIG_PREPARED",
        "output": str(args.output.resolve()),
        "n_candidates": payload["m3_design"]["candidate_count"],
        "n_jobs": payload["m3_design"]["candidate_count"] * 3,
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
