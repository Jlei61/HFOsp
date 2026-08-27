#!/usr/bin/env python3
"""Aggregate fresh-network rev15 M3 response-combination runs."""
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
from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as atlas_aggregate  # noqa: E402
from scripts import freeze_topic4_rev15_m3_response_combinations as freezer  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.run_topic4_rev15_m3_response_combination_worker import WORKER_STATUS  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_m3_response_combinations.json"
DEFAULT_J14_CONFIG = ROOT / "config/topic4_rev14_static_node_historical_rescore.json"
DEFAULT_SUPPORT_CONFIG = ROOT / "config/topic4_rev14_patient_support_acceptance.json"
OUTPUT_SCHEMA = "topic4_rev15_m3_response_combinations_aggregate_v1"


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
    config_path: Path, artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    path = artifact_root / config["candidate_manifest"]
    if not path.is_file():
        raise RuntimeError("rev15 response-combination manifest is missing")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev15 response-combination manifest is not frozen")
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev15 response-combination manifest schema changed")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev15 response-combination manifest/config mismatch")
    return config, manifest, path


def _provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    worker_commit = str(manifest["provenance"]["git_commit"])
    return {
        "worker_freeze_commit": worker_commit,
        "analysis_commit": head,
        "worktree_status": status,
        "same_commit_as_worker": head == worker_commit,
        "formal_ready": bool(head == worker_commit and not status),
        "snn_simulation_run": False,
    }


def _inventory(
    *, config: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_root = artifact_root / config["output_root"] / "workers"
    manifest_hash = _sha256(manifest_path)
    commit = str(manifest["provenance"]["git_commit"])
    rows = []
    missing = []
    invalid = []
    previous = canary.WORKER_STATUS
    canary.WORKER_STATUS = WORKER_STATUS
    try:
        for candidate in manifest["candidates"]:
            for seed in config["search"]["active_network_seeds"]:
                candidate_id = candidate["candidate_id"]
                path = worker_root / f"{candidate_id}_seed_{seed}.json"
                if not path.is_file():
                    missing.append(f"{candidate_id}:{seed}")
                    continue
                try:
                    payload = json.loads(path.read_text())
                    rows.append(canary._validate_worker(
                        path.resolve(), payload, candidate=candidate,
                        active_seed=int(seed), manifest=manifest,
                        manifest_sha256=manifest_hash, manifest_commit=commit,
                        config=config, artifact_root=artifact_root,
                    ))
                except Exception as error:
                    invalid.append(f"{candidate_id}:{seed}:{error}")
    finally:
        canary.WORKER_STATUS = previous
    expected = len(manifest["candidates"]) * len(
        config["search"]["active_network_seeds"]
    )
    audit = {
        "expected_runs": expected,
        "present_validated": len(rows),
        "missing": missing,
        "invalid_artifact": invalid,
        "complete_cartesian_product": len(rows) == expected and not missing and not invalid,
    }
    return rows, audit


def _flat(scored: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    row = atlas_aggregate._flat_row(scored, candidate)
    metadata = candidate.get("response_combination") or {}
    row.update({
        "family": metadata.get("family"),
        "target_rms": (candidate.get("fourier_coordinate") or {}).get(
            "target_centered_surface_rms"
        ),
    })
    return row


def _summaries(
    rows: list[dict[str, Any]], manifest: Mapping[str, Any],
    atlas_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["candidate_id"], []).append(row)
    exact = {int(row["seed"]): row for row in by_id["exact_off"]}
    atlas_rows = {row["candidate_id"]: row for row in atlas_payload["per_candidate"]}
    single_ids = set(manifest["m3_design"]["single_control_ids"])
    summaries = []
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id == "exact_off":
            continue
        fresh = sorted(by_id[candidate_id], key=lambda row: int(row["seed"]))
        paired = []
        for row in fresh:
            reference = exact[int(row["seed"])]
            paired.append({
                **row,
                "delta_A_vs_exact": float(row["mode_0_mean"]) - float(reference["mode_0_mean"]),
                "delta_B_vs_exact": float(row["mode_1_mean"]) - float(reference["mode_1_mean"]),
                "delta_J14_vs_exact": float(row["j14"]) - float(reference["j14"]),
                "A_improves": float(row["mode_0_mean"]) < float(reference["mode_0_mean"]),
                "B_within_10pct": float(row["mode_1_mean"]) <= 1.1 * float(reference["mode_1_mean"]),
            })
        evaluation = list(paired)
        if candidate_id in single_ids:
            atlas_row = atlas_rows[candidate_id]
            atlas_exact = atlas_rows["exact_off"]
            evaluation.append({
                **atlas_row,
                "seed": 2331,
                "delta_A_vs_exact": float(atlas_row["mode_0_mean"]) - float(atlas_exact["mode_0_mean"]),
                "delta_B_vs_exact": float(atlas_row["mode_1_mean"]) - float(atlas_exact["mode_1_mean"]),
                "delta_J14_vs_exact": float(atlas_row["j14"]) - float(atlas_exact["j14"]),
                "A_improves": float(atlas_row["mode_0_mean"]) < float(atlas_exact["mode_0_mean"]),
                "B_within_10pct": float(atlas_row["mode_1_mean"]) <= 1.1 * float(atlas_exact["mode_1_mean"]),
            })
        n_required = 2
        a_count = sum(bool(row["A_improves"]) for row in evaluation)
        b_count = sum(bool(row["B_within_10pct"]) for row in evaluation)
        support_a = float(np.mean([row["mode_0_effective_events"] for row in evaluation]))
        support_b = float(np.mean([row["mode_1_effective_events"] for row in evaluation]))
        usable = bool(
            a_count >= n_required and b_count >= n_required
            and support_a >= 6.0 and support_b >= 6.0
        )
        summaries.append({
            "candidate_id": candidate_id,
            "family": paired[0]["family"],
            "target_rms": paired[0]["target_rms"],
            "evaluation_network_count": len(evaluation),
            "fresh_A_improvement_count": sum(bool(row["A_improves"]) for row in paired),
            "fresh_B_protection_count": sum(bool(row["B_within_10pct"]) for row in paired),
            "A_improvement_count": a_count,
            "B_protection_count": b_count,
            "mean_delta_A": float(np.mean([row["delta_A_vs_exact"] for row in evaluation])),
            "mean_delta_B": float(np.mean([row["delta_B_vs_exact"] for row in evaluation])),
            "mean_delta_J14": float(np.mean([row["delta_J14_vs_exact"] for row in evaluation])),
            "equal_network_A_effective_support": support_a,
            "equal_network_B_effective_support": support_b,
            "usable_two_mode_anchor": usable,
            "per_network": evaluation,
        })
    summaries.sort(key=lambda row: (
        not row["usable_two_mode_anchor"],
        -row["fresh_A_improvement_count"],
        -row["fresh_B_protection_count"],
        row["mean_delta_A"],
        row["mean_delta_B"],
        row["mean_delta_J14"],
        row["candidate_id"],
    ))
    for rank, row in enumerate(summaries, start=1):
        row["replication_rank"] = rank
    return summaries


def aggregate(
    *, config_path: Path = DEFAULT_CONFIG,
    j14_config_path: Path = DEFAULT_J14_CONFIG,
    support_config_path: Path = DEFAULT_SUPPORT_CONFIG,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config, manifest, manifest_path = _load_manifest(config_path, artifact_root)
    provenance = _provenance(manifest)
    records, inventory = _inventory(
        config=config, manifest=manifest, manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    status = "INCOMPLETE"
    error = None
    rows = []
    summaries = []
    if not provenance["formal_ready"]:
        status = "INVALID_PROVENANCE"
        error = "aggregator is not running from the clean worker-freeze commit"
    elif inventory["complete_cartesian_product"]:
        try:
            j14_config = json.loads(j14_config_path.read_text())
            context = historical._patient_context(j14_config, artifact_root)
            support_context = canary._load_support_context(
                support_config_path, artifact_root, j14_config,
            )
            candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
            rows = [
                _flat(canary._score_worker(record, context, support_context),
                      candidates[record["candidate_id"]])
                for record in records
            ]
            _, atlas_payload = freezer._load_input(
                config, "coordinate_atlas_aggregate", artifact_root,
            )
            summaries = _summaries(rows, manifest, atlas_payload)
            status = "COMPLETE"
        except Exception as caught:
            status = "INVALID_INPUT"
            error = str(caught)
    output_root = artifact_root / config["output_root"] / "analysis"
    json_path = output_root / "m3_response_combinations_aggregate.json"
    csv_path = output_root / "m3_response_combinations_summary.csv"
    usable = [row for row in summaries if row["usable_two_mode_anchor"]]
    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": status,
        "inventory": inventory,
        "input_error": error,
        "provenance": provenance,
        "ranking_contract": {
            "fresh_combination_requires_A_improvement": "2/2 networks",
            "single_control_requires_A_improvement": "at least 2/3 networks",
            "B_protection": "at least 2 networks at <=110% paired exact",
            "equal_network_effective_support": "A>=6 and B>=6",
            "natural_kmeans_used": False,
            "patient_heldout_used": False,
            "ictal_data_used": False,
        },
        "per_fresh_run": rows,
        "candidate_summaries": summaries,
        "usable_two_mode_anchor_ids": [row["candidate_id"] for row in usable],
        "best_usable_anchor": usable[0]["candidate_id"] if usable else None,
        "claim_boundary": (
            "Fresh-network training-only replication. A usable anchor can advance "
            "to Node freeze and post-freeze natural KMeans, but is not itself a "
            "Fig.4 or held-out acceptance result."
        ),
        "outputs": {"json": str(json_path), "summary_csv": str(csv_path)},
    }
    _atomic_json(json_path, payload)
    _atomic_csv(csv_path, [
        {key: value for key, value in row.items() if key != "per_network"}
        for row in summaries
    ])
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--j14-config", type=Path, default=DEFAULT_J14_CONFIG)
    parser.add_argument("--support-config", type=Path, default=DEFAULT_SUPPORT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = aggregate(
        config_path=args.config, j14_config_path=args.j14_config,
        support_config_path=args.support_config,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "best_usable_anchor": payload["best_usable_anchor"],
        "n_usable": len(payload["usable_two_mode_anchor_ids"]),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
