#!/usr/bin/env python3
"""Prepare an unseen-network confirmation of one frozen rev16 Node field."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as selection_freezer  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_SELECTION_CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_candidates.json"
DEFAULT_SELECTION_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/analysis/joint_candidates_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_node_confirmation.json"
OUTPUT_SCHEMA = "topic4_rev16_node_confirmation_v1"
EXPECTED_AGGREGATE_SCHEMA = "topic4_rev16_joint_m3_m4_candidates_aggregate_v1"
SELECTION_SEEDS = [2351, 2352, 2353]
CONFIRMATION_SEEDS = [2361, 2362, 2363]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"rev16 confirmation input is missing: {path}")
    return {"path": str(path.relative_to(root.resolve())), "sha256": _sha256(path)}


def _selected_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = aggregate.get("best_usable_anchor")
    rows = [
        row for row in aggregate.get("candidate_summaries", [])
        if row.get("candidate_id") == candidate_id
    ]
    if len(rows) != 1 or rows[0].get("usable_two_mode_anchor") is not True:
        raise RuntimeError("rev16 confirmation lacks one training-qualified anchor")
    row = rows[0]
    if any(int(row.get(key, -1)) != 3 for key in (
        "fresh_J14_improvement_count", "fresh_A_improvement_count",
        "fresh_B_protection_count",
    )):
        raise RuntimeError("rev16 selected anchor lost its 3/3 training clauses")
    return row


def build_config(
    *, selection_config_path: Path, selection_aggregate_path: Path,
    artifact_root: Path = ARTIFACT_ROOT, repository_root: Path = ROOT,
) -> dict[str, Any]:
    selection_config = json.loads(selection_config_path.read_text())
    selection_freezer._validate_config(selection_config)
    aggregate = json.loads(selection_aggregate_path.read_text())
    if aggregate.get("schema_id") != EXPECTED_AGGREGATE_SCHEMA:
        raise RuntimeError("rev16 selection aggregate schema changed")
    if aggregate.get("status") != "COMPLETE" or not aggregate.get(
        "inventory", {}
    ).get("complete_cartesian_product"):
        raise RuntimeError("rev16 selection aggregate is incomplete")
    ranking = aggregate.get("ranking_contract", {})
    if ranking.get("J14_improvement") != "3/3 fresh networks":
        raise RuntimeError("rev16 selection predates the full-J14 contract")
    if any(ranking.get(key) is not False for key in (
        "natural_kmeans_used", "patient_heldout_used", "ictal_data_used",
        "figure_used",
    )) or ranking.get("EE_EtoI_ZM") != "off":
        raise RuntimeError("rev16 confirmation source crossed a selection boundary")
    selected = _selected_summary(aggregate)
    candidate_id = str(selected["candidate_id"])
    manifest_path = artifact_root / selection_config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != selection_freezer.STATUS
        or manifest.get("config_sha256") != _sha256(selection_config_path)
    ):
        raise RuntimeError("rev16 selection manifest is not frozen")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or "exact_off" not in candidates:
        raise RuntimeError("rev16 selected or exact candidate is absent")
    selected_candidate = candidates[candidate_id]
    if selected_candidate.get("selection_eligible") is not True:
        raise RuntimeError("rev16 selected candidate is not selectable")
    inputs = {
        "selection_config": _record(selection_config_path, repository_root),
        "selection_manifest": _record(manifest_path, artifact_root),
        "selection_aggregate": _record(selection_aggregate_path, artifact_root),
    }
    for name in (
        "rev13_config", "rev13_exact_off_manifest", "j14_config",
        "patient_support_config",
    ):
        record = selection_config["inputs"][name]
        path = selection_freezer._resolve(artifact_root, record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev16 inherited input changed: {name}")
        inputs[name] = dict(record)
    source_design = selection_config["field_design"]
    return {
        "schema_id": OUTPUT_SCHEMA,
        "scientific_role": (
            "development_only_unseen_network_confirmation_of_one_training_selected_node"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
            "node_confirmation"
        ),
        "candidate_manifest": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
            "node_confirmation/candidate_manifest.json"
        ),
        "network_cache": selection_config["network_cache"],
        "inputs": inputs,
        "selected_candidate": {
            "candidate_id": candidate_id,
            "coefficients_sha256": selected_candidate["fourier_coordinate"][
                "coefficients_sha256"
            ],
            "selection_network_seeds": SELECTION_SEEDS,
            "selection_summary": selected,
        },
        "field_design": {
            "basis_family": source_design["basis_family"],
            "maximum_order": source_design["maximum_order"],
            "expected_modes": source_design["expected_modes"],
            "expected_real_coefficients": source_design[
                "expected_real_coefficients"
            ],
            "sheet_length_mm": source_design["sheet_length_mm"],
            "quadrature_per_axis": source_design["quadrature_per_axis"],
            "coordinate_decimal_places": source_design[
                "coordinate_decimal_places"
            ],
            "candidate_ids": ["exact_off", candidate_id],
            "candidate_count": 2,
            "selectable_candidate_count": 1,
        },
        "node_mapping": selection_config["node_mapping"],
        "search": {
            "selection_network_seeds": SELECTION_SEEDS,
            "confirmation_network_seeds": CONFIRMATION_SEEDS,
            "active_network_seeds": CONFIRMATION_SEEDS,
            "common_random_numbers_across_candidates": True,
            "simulation": {
                "duration_ms": 20000.0, "early_stop_runaway": True,
                "late_runaway_is_invalid": True,
            },
        },
        "pathways": selection_config["pathways"],
        "resources": {
            "numerical_threads_per_worker": 1,
            "maximum_workers": 6, "recommended_workers": 6,
            "measured_canary_peak_rss_kib": 13356224,
            "worker_rss_safety_multiplier": 1.2,
            "stop_launching_below_available_memory_gib": 80,
            "emergency_stop_below_available_memory_gib": 64,
            "minimum_free_disk_gib": 40, "monitor_interval_seconds": 600,
            "long_run_launcher": "systemd-run --user plus nohup",
        },
        "boundaries": {
            "field_reranking_allowed": False,
            "patient_heldout_used": False,
            "natural_kmeans_used": False,
            "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "claim_boundary": (
            "Only the training-selected field and paired exact_off are copied to "
            "three unseen network seeds. This stage cannot rerank fields, access "
            "KMeans or patient held-out events, or activate EE, E-to-I or Z/M."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-config", type=Path, default=DEFAULT_SELECTION_CONFIG)
    parser.add_argument(
        "--selection-aggregate", type=Path, default=DEFAULT_SELECTION_AGGREGATE,
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_config(
        selection_config_path=args.selection_config.resolve(),
        selection_aggregate_path=args.selection_aggregate.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV16_NODE_CONFIRMATION_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "n_jobs": 6, "output": str(args.output.resolve()),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
