#!/usr/bin/env python3
"""Verify and training-score the unseen-network rev16 Node confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev16_joint_candidates as joint  # noqa: E402
from scripts import freeze_topic4_rev16_node_confirmation as freezer  # noqa: E402
from scripts import (  # noqa: E402
    rescore_topic4_rev14_static_node_historical_libraries as historical,
)
from scripts import run_topic4_rev16_node_confirmation_worker as worker  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_node_confirmation.json"
OUTPUT_SCHEMA = "topic4_rev16_node_confirmation_audit_v1"
COMPLETE_STATUS = "REV16_NODE_CONFIRMATION_SCORED_COMPLETE"


def confirmation_decision(
    scored_rows: list[dict[str, Any]], manifest: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    selection = {
        "fresh_J14_improvement_required_networks": int(
            acceptance["J14_improvement_required_networks"]
        ),
        "fresh_A_improvement_required_networks": int(
            acceptance["A_improvement_required_networks"]
        ),
        "fresh_B_protection_required_networks": int(
            acceptance["B_protection_required_networks"]
        ),
        "B_protection_ratio": float(acceptance["B_protection_ratio"]),
        "equal_network_effective_support_minimum_per_mode": float(
            acceptance["equal_network_effective_support_minimum_per_mode"]
        ),
    }
    summaries = joint.summaries(
        scored_rows, {"selection": selection, "candidates": manifest["candidates"]},
    )
    if len(summaries) != 1:
        raise RuntimeError("confirmation must score exactly one selected field")
    row = summaries[0]
    return {
        "accepted": bool(row["usable_two_mode_anchor"]),
        "candidate_id": row["candidate_id"],
        "J14_improvement_count": int(row["fresh_J14_improvement_count"]),
        "A_improvement_count": int(row["fresh_A_improvement_count"]),
        "B_protection_count": int(row["fresh_B_protection_count"]),
        "A_support_count": int(row["fresh_A_support_count"]),
        "B_support_count": int(row["fresh_B_support_count"]),
        "equal_network_A_effective_support": float(
            row["equal_network_A_effective_support"]
        ),
        "equal_network_B_effective_support": float(
            row["equal_network_B_effective_support"]
        ),
        "minimum_network_A_effective_support": float(
            row["minimum_network_A_effective_support"]
        ),
        "minimum_network_B_effective_support": float(
            row["minimum_network_B_effective_support"]
        ),
        "worst_delta_J14": float(row["worst_delta_J14"]),
        "mean_delta_J14": float(row["mean_delta_J14"]),
        "worst_delta_A": float(row["worst_delta_A"]),
        "worst_B_ratio": float(row["worst_B_ratio"]),
        "per_network": row["per_network"],
        "failure_does_not_trigger_reranking": True,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    config_path: Path = DEFAULT_CONFIG, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(); artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != freezer.STATUS
        or manifest.get("schema_id") != freezer.MANIFEST_SCHEMA
        or manifest.get("config_sha256") != _sha256(config_path)
    ):
        raise RuntimeError("rev16 confirmation manifest is not frozen")
    expected_commit = manifest["provenance"]["git_commit"]
    records, validated_records, missing, invalid = [], [], [], []
    worker_root = artifact_root / config["output_root"] / "workers"
    manifest_hash = _sha256(manifest_path)
    previous_status = canary.WORKER_STATUS
    canary.WORKER_STATUS = worker.WORKER_STATUS
    try:
        for candidate in manifest["candidates"]:
            candidate_id = candidate["candidate_id"]
            for seed in config["search"]["active_network_seeds"]:
                stem = worker_root / f"{candidate_id}_seed_{seed}"
                json_path = stem.with_suffix(".json")
                npz_path = stem.with_suffix(".npz")
                key = f"{candidate_id}:{seed}"
                if not json_path.is_file() or not npz_path.is_file():
                    missing.append(key)
                    continue
                try:
                    record = canary._validate_worker(
                        json_path.resolve(), json.loads(json_path.read_text()),
                        candidate=candidate, active_seed=int(seed), manifest=manifest,
                        manifest_sha256=manifest_hash,
                        manifest_commit=expected_commit, config=config,
                        artifact_root=artifact_root,
                    )
                    if record["run_status"] != "VALID":
                        raise RuntimeError(f"worker status is {record['run_status']}")
                    validated_records.append(record)
                    records.append({
                        "candidate_id": candidate_id,
                        "seed": int(seed),
                        "json": {
                            "path": str(json_path), "sha256": _sha256(json_path),
                        },
                        "npz": {
                            "path": str(npz_path), "sha256": _sha256(npz_path),
                        },
                    })
                except Exception as error:
                    invalid.append(f"{key}:{error}")
    finally:
        canary.WORKER_STATUS = previous_status
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    complete = len(records) == 6 and not missing and not invalid
    score_error = None
    scientific_confirmation = None
    scored_rows: list[dict[str, Any]] = []
    if complete:
        try:
            _, j14_config = freezer._load_hashed(
                config, "j14_config", artifact_root,
            )
            support_path, _ = freezer._load_hashed(
                config, "patient_support_config", artifact_root,
            )
            context = historical._patient_context(j14_config, artifact_root)
            support = canary._load_support_context(
                support_path, artifact_root, j14_config,
            )
            candidates = {
                row["candidate_id"]: row for row in manifest["candidates"]
            }
            scored_rows = [
                joint._flat(
                    canary._score_worker(record, context, support),
                    candidates[record["candidate_id"]],
                )
                for record in validated_records
            ]
            scientific_confirmation = confirmation_decision(
                scored_rows, manifest, config["confirmation_acceptance"],
            )
        except Exception as error:
            score_error = str(error)
    status = (
        "INVALID_PROVENANCE" if status_lines else
        COMPLETE_STATUS if complete and scientific_confirmation is not None else
        "INVALID_SCIENTIFIC_SCORE" if complete and score_error else
        "INCOMPLETE"
    )
    output_path = artifact_root / config["output_root"] / "analysis/confirmation_audit.json"
    payload = {
        "schema_id": OUTPUT_SCHEMA, "status": status,
        "candidate_id": config["selected_candidate"]["candidate_id"],
        "network_seeds": config["search"]["active_network_seeds"],
        "inventory": {
            "expected_runs": 6, "present_validated": len(records),
            "missing": missing, "invalid_artifact": invalid,
            "complete_cartesian_product": complete,
        },
        "workers": records,
        "scientific_confirmation": scientific_confirmation,
        "score_error": score_error,
        "per_confirmation_run": scored_rows,
        "selection_source": config["inputs"]["selection_aggregate"],
        "boundaries": {**config["boundaries"], "SNN_simulation_run": False},
        "provenance": {
            "worker_commit": expected_commit, "analysis_commit": head,
            "worktree_status": status_lines, "formal_ready": not status_lines,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_path)},
    }
    freezer._atomic_json(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "scientifically_confirmed": (
            payload.get("scientific_confirmation") or {}
        ).get("accepted"),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
