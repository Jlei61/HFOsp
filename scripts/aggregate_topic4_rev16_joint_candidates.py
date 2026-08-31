#!/usr/bin/env python3
"""Aggregate fresh-network joint M3+M4 Node candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as atlas  # noqa: E402
from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as freezer  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.run_topic4_rev16_joint_m3_m4_candidate_worker import WORKER_STATUS  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_candidates.json"
OUTPUT_SCHEMA = "topic4_rev16_joint_m3_m4_candidates_aggregate_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(canary._jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_manifest(
    config_path: Path, root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    path = root / config["candidate_manifest"]
    if not path.is_file():
        raise RuntimeError("rev16 joint manifest is missing")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev16 joint manifest is not frozen")
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev16 joint manifest schema changed")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev16 joint manifest/config mismatch")
    return config, manifest, path


def _provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    worker_commit = manifest["provenance"]["git_commit"]
    return {
        "worker_freeze_commit": worker_commit, "analysis_commit": head,
        "worktree_status": status, "same_commit_as_worker": head == worker_commit,
        "analysis_worktree_clean": not status,
        "formal_ready": not status,
        "snn_simulation_run": False,
    }


def _inventory(
    config: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_root = root / config["output_root"] / "workers"
    manifest_hash = _sha256(manifest_path)
    commit = manifest["provenance"]["git_commit"]
    rows, missing, invalid = [], [], []
    previous = canary.WORKER_STATUS
    canary.WORKER_STATUS = WORKER_STATUS
    try:
        for candidate in manifest["candidates"]:
            for seed in config["search"]["active_network_seeds"]:
                key = f"{candidate['candidate_id']}:{seed}"
                path = worker_root / f"{candidate['candidate_id']}_seed_{seed}.json"
                if not path.is_file():
                    missing.append(key)
                    continue
                try:
                    rows.append(canary._validate_worker(
                        path.resolve(), json.loads(path.read_text()),
                        candidate=candidate, active_seed=int(seed), manifest=manifest,
                        manifest_sha256=manifest_hash, manifest_commit=commit,
                        config=config, artifact_root=root,
                    ))
                except Exception as error:
                    invalid.append(f"{key}:{error}")
    finally:
        canary.WORKER_STATUS = previous
    expected = len(manifest["candidates"]) * 3
    return rows, {
        "expected_runs": expected, "present_validated": len(rows),
        "missing": missing, "invalid_artifact": invalid,
        "complete_cartesian_product": len(rows) == expected and not missing and not invalid,
    }


def _flat(scored: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    row = atlas._flat_row(scored, candidate)
    metadata = candidate.get("joint_direction") or {}
    row.update({
        "family": metadata.get("family"),
        "m3_l2_fraction": metadata.get("m3_l2_fraction"),
        "m4_shell_l2_fraction": metadata.get("m4_shell_l2_fraction"),
        "target_rms": (candidate.get("fourier_coordinate") or {}).get(
            "target_centered_surface_rms"
        ),
    })
    return row


def summaries(
    rows: list[dict[str, Any]], manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["candidate_id"], []).append(row)
    exact = {int(row["seed"]): row for row in by_id["exact_off"]}
    required_j14 = int(selection["fresh_J14_improvement_required_networks"])
    required_a = int(selection["fresh_A_improvement_required_networks"])
    required_b = int(selection["fresh_B_protection_required_networks"])
    b_ratio = float(selection["B_protection_ratio"])
    support_minimum = float(selection["equal_network_effective_support_minimum_per_mode"])
    output = []
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id == "exact_off":
            continue
        paired = []
        for row in sorted(by_id[candidate_id], key=lambda item: int(item["seed"])):
            reference = exact[int(row["seed"])]
            reference_b = float(reference["mode_1_mean"])
            candidate_b = float(row["mode_1_mean"])
            paired.append({
                **row,
                "delta_A_vs_exact": float(row["mode_0_mean"]) - float(reference["mode_0_mean"]),
                "delta_B_vs_exact": candidate_b - reference_b,
                "delta_J14_vs_exact": float(row["j14"]) - float(reference["j14"]),
                "B_ratio_vs_exact": candidate_b / reference_b,
                "A_improves": float(row["mode_0_mean"]) < float(reference["mode_0_mean"]),
                "J14_improves": float(row["j14"]) < float(reference["j14"]),
                "B_within_ratio": candidate_b <= b_ratio * reference_b,
            })
        j14_count = sum(row["J14_improves"] for row in paired)
        a_count = sum(row["A_improves"] for row in paired)
        b_count = sum(row["B_within_ratio"] for row in paired)
        support_a_count = sum(
            float(row["mode_0_effective_events"]) >= support_minimum
            for row in paired
        )
        support_b_count = sum(
            float(row["mode_1_effective_events"]) >= support_minimum
            for row in paired
        )
        support_a = float(np.mean([row["mode_0_effective_events"] for row in paired]))
        support_b = float(np.mean([row["mode_1_effective_events"] for row in paired]))
        usable = bool(
            j14_count >= required_j14
            and a_count >= required_a and b_count >= required_b
            and support_a_count == len(paired)
            and support_b_count == len(paired)
        )
        output.append({
            "candidate_id": candidate_id, "family": paired[0]["family"],
            "target_rms": paired[0]["target_rms"],
            "m3_l2_fraction": paired[0]["m3_l2_fraction"],
            "m4_shell_l2_fraction": paired[0]["m4_shell_l2_fraction"],
            "evaluation_network_count": len(paired),
            "fresh_J14_improvement_count": j14_count,
            "fresh_A_improvement_count": a_count,
            "fresh_B_protection_count": b_count,
            "fresh_A_support_count": support_a_count,
            "fresh_B_support_count": support_b_count,
            "mean_delta_A": float(np.mean([row["delta_A_vs_exact"] for row in paired])),
            "worst_delta_A": float(np.max([row["delta_A_vs_exact"] for row in paired])),
            "mean_delta_B": float(np.mean([row["delta_B_vs_exact"] for row in paired])),
            "worst_B_ratio": float(np.max([row["B_ratio_vs_exact"] for row in paired])),
            "mean_delta_J14": float(np.mean([row["delta_J14_vs_exact"] for row in paired])),
            "worst_delta_J14": float(np.max([row["delta_J14_vs_exact"] for row in paired])),
            "equal_network_A_effective_support": support_a,
            "equal_network_B_effective_support": support_b,
            "minimum_network_A_effective_support": float(min(
                row["mode_0_effective_events"] for row in paired
            )),
            "minimum_network_B_effective_support": float(min(
                row["mode_1_effective_events"] for row in paired
            )),
            "usable_two_mode_anchor": usable, "per_network": paired,
        })
    output.sort(key=lambda row: (
        not row["usable_two_mode_anchor"], -row["fresh_J14_improvement_count"],
        row["worst_delta_J14"], row["mean_delta_J14"],
        -row["fresh_A_improvement_count"],
        -row["fresh_B_protection_count"], row["worst_delta_A"],
        row["worst_B_ratio"], row["mean_delta_A"], row["candidate_id"],
    ))
    for rank, row in enumerate(output, 1):
        row["replication_rank"] = rank
    return output


def aggregate(
    config_path: Path = DEFAULT_CONFIG, root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config, manifest, manifest_path = _load_manifest(config_path.resolve(), root.resolve())
    provenance = _provenance(manifest)
    records, inventory = _inventory(config, manifest, manifest_path, root.resolve())
    status, error, rows, summary = "INCOMPLETE", None, [], []
    if not provenance["formal_ready"]:
        status, error = "INVALID_PROVENANCE", "aggregator is not on clean worker commit"
    elif inventory["complete_cartesian_product"]:
        try:
            _, j14_config = freezer._load_hashed(config, "j14_config", root.resolve())
            support_path, _ = freezer._load_hashed(
                config, "patient_support_config", root.resolve(),
            )
            context = historical._patient_context(j14_config, root.resolve())
            support = canary._load_support_context(
                support_path, root.resolve(), j14_config,
            )
            candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
            rows = [
                _flat(canary._score_worker(record, context, support),
                      candidates[record["candidate_id"]])
                for record in records
            ]
            summary = summaries(rows, manifest, config["selection"])
            status = "COMPLETE"
        except Exception as caught:
            status, error = "INVALID_INPUT", str(caught)
    output_root = root.resolve() / config["output_root"] / "analysis"
    json_path = output_root / "joint_candidates_aggregate.json"
    csv_path = output_root / "joint_candidates_summary.csv"
    usable = [row for row in summary if row["usable_two_mode_anchor"]]
    payload = {
        "schema_id": OUTPUT_SCHEMA, "status": status, "input_error": error,
        "inventory": inventory, "provenance": provenance,
        "ranking_contract": {
            "J14_improvement": "3/3 fresh networks",
            "A_improvement": "3/3 fresh networks",
            "B_protection": "3/3 fresh networks at <=110% paired exact",
            "per_network_effective_support": "A>=6 and B>=6 on 3/3 networks",
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "per_fresh_run": rows, "candidate_summaries": summary,
        "usable_two_mode_anchor_ids": [row["candidate_id"] for row in usable],
        "best_usable_anchor": usable[0]["candidate_id"] if usable else None,
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(json_path), "summary_csv": str(csv_path)},
    }
    _atomic_json(json_path, payload)
    _atomic_csv(csv_path, [
        {key: value for key, value in row.items() if key != "per_network"}
        for row in summary
    ])
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = aggregate(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "best_usable_anchor": payload["best_usable_anchor"],
        "n_usable": len(payload["usable_two_mode_anchor_ids"]),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
