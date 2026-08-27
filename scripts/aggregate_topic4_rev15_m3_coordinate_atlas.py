#!/usr/bin/env python3
"""Aggregate the frozen rev15 M3 coordinate atlas using training data only."""
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
from scripts import freeze_topic4_rev15_m3_coordinate_atlas as freezer  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.run_topic4_rev15_m3_coordinate_atlas_worker import WORKER_STATUS  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_m3_coordinate_atlas.json"
DEFAULT_J14_CONFIG = ROOT / "config/topic4_rev14_static_node_historical_rescore.json"
DEFAULT_SUPPORT_CONFIG = ROOT / "config/topic4_rev14_patient_support_acceptance.json"
OUTPUT_SCHEMA = "topic4_rev15_m3_coordinate_atlas_aggregate_v1"
ANALYSIS_ONLY_ALLOWED_PATHS = frozenset({
    "scripts/aggregate_topic4_rev15_m3_coordinate_atlas.py",
    "tests/test_topic4_rev15_m3_coordinate_atlas_aggregate.py",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def _runtime_provenance(worker_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{worker_commit}..HEAD"],
        cwd=ROOT, text=True,
    ).splitlines()
    allowed = set(changed).issubset(ANALYSIS_ONLY_ALLOWED_PATHS)
    return {
        "worker_freeze_commit": worker_commit,
        "analysis_commit": commit,
        "paths_changed_since_worker_freeze": changed,
        "analysis_only_allowed_paths": sorted(ANALYSIS_ONLY_ALLOWED_PATHS),
        "analysis_only_commit_allowed": allowed,
        "worktree_status": status,
        "formal_ready": bool(allowed and not status),
        "snn_simulation_run": False,
    }


def _load_manifest(
    config_path: Path, artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    path = artifact_root / str(config["candidate_manifest"])
    if not path.is_file():
        raise RuntimeError("rev15 coordinate-atlas manifest is missing")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev15 coordinate-atlas manifest schema changed")
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev15 coordinate-atlas manifest is not frozen")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev15 coordinate-atlas manifest/config mismatch")
    candidates = manifest.get("candidates", [])
    if len(candidates) != 58 or sum(row["selection_eligible"] for row in candidates) != 56:
        raise RuntimeError("rev15 coordinate-atlas Cartesian product changed")
    provenance = manifest.get("provenance", {})
    if not provenance.get("formal_ready"):
        raise RuntimeError("rev15 coordinate-atlas manifest provenance is invalid")
    return config, manifest, path


def _inventory(
    *, config: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_root = artifact_root / str(config["output_root"]) / "workers"
    seed = int(config["search"]["active_network_seeds"][0])
    manifest_hash = _sha256(manifest_path)
    commit = str(manifest["provenance"]["git_commit"])
    rows = []
    missing = []
    invalid = []
    previous = canary.WORKER_STATUS
    canary.WORKER_STATUS = WORKER_STATUS
    try:
        for candidate in manifest["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            path = worker_root / f"{candidate_id}_seed_{seed}.json"
            if not path.is_file():
                missing.append(candidate_id)
                rows.append({
                    "candidate_id": candidate_id, "seed": seed,
                    "inventory_status": "MISSING", "run_status": None,
                    "error": "worker JSON is missing",
                })
                continue
            try:
                payload = json.loads(path.read_text())
                rows.append(canary._validate_worker(
                    path.resolve(), payload, candidate=candidate,
                    active_seed=seed, manifest=manifest,
                    manifest_sha256=manifest_hash, manifest_commit=commit,
                    config=config, artifact_root=artifact_root,
                ))
            except Exception as error:
                invalid.append(candidate_id)
                rows.append({
                    "candidate_id": candidate_id, "seed": seed,
                    "inventory_status": "INVALID_ARTIFACT", "run_status": None,
                    "error": str(error),
                })
    finally:
        canary.WORKER_STATUS = previous
    audit = {
        "expected_runs": 58,
        "present_validated": sum(
            row["inventory_status"] == "PRESENT_VALIDATED" for row in rows
        ),
        "missing": missing,
        "invalid_artifact": invalid,
    }
    audit["complete_cartesian_product"] = (
        audit["present_validated"] == 58 and not missing and not invalid
    )
    return rows, audit


def _mode_components(score: Mapping[str, Any], mode: int) -> dict[str, Any]:
    values = (score.get("j14_v1") or {}).get("modes", {}).get(str(mode), {})
    return {
        f"mode_{mode}_{key}": values.get(key)
        for key in ("recruitment", "precedence", "profile", "cloud", "mean")
    }


def _flat_row(
    scored: Mapping[str, Any], candidate: Mapping[str, Any],
) -> dict[str, Any]:
    summary = scored.get("j14_v1_summary") or {}
    assignment = scored.get("patient_training_assignment") or {}
    support = scored.get("patient_support") or {}
    support_detail = support.get("support") or {}
    event = scored.get("event_selection") or {}
    coordinate = candidate.get("coordinate_atlas") or {}
    objective = scored.get("j14_v1") or {}
    contrast = objective.get("contrast") or {}
    return {
        "candidate_id": scored["candidate_id"],
        "seed": scored["seed"],
        "selection_eligible": bool(candidate["selection_eligible"]),
        "run_status": scored.get("run_status"),
        "coordinate_index": coordinate.get("coordinate_index"),
        "mode_nx": (coordinate.get("mode") or [None, None])[0],
        "mode_ny": (coordinate.get("mode") or [None, None])[1],
        "phase": coordinate.get("phase"),
        "sign": (candidate.get("fourier_coordinate") or {}).get("sign"),
        "j14": summary.get("objective"),
        **_mode_components(scored, 0),
        **_mode_components(scored, 1),
        "mode_0_effective_events": summary.get("mode_0_effective_events"),
        "mode_1_effective_events": summary.get("mode_1_effective_events"),
        "contrast_alignment": contrast.get("alignment"),
        "classifier_A": assignment.get("classifier_A"),
        "classifier_B": assignment.get("classifier_B"),
        "ood_count": assignment.get("ood_count"),
        "in_support_A": support_detail.get("n_in_support_classifier_A"),
        "in_support_B": support_detail.get("n_in_support_classifier_B"),
        "n_returned": event.get("n_returned"),
        "n_contact_primary": event.get("n_contact_primary"),
        "n_fig4_kmeans_readable": event.get("n_fig4_kmeans_readable"),
        "patient_support_status": support.get("status"),
        "patient_support_score": support.get("score"),
    }


def rank_coordinate_responses(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact = next(row for row in rows if row["candidate_id"] == "exact_off")
    exact_a = float(exact["mode_0_mean"])
    exact_b = float(exact["mode_1_mean"])
    ranked = []
    for row in rows:
        if not row["selection_eligible"]:
            continue
        delta_a = float(row["mode_0_mean"]) - exact_a
        delta_b = float(row["mode_1_mean"]) - exact_b
        response = {
            **row,
            "delta_mode_A_vs_exact": delta_a,
            "delta_mode_B_vs_exact": delta_b,
            "mode_A_improves": bool(delta_a < 0.0),
            "mode_B_within_10pct": bool(
                float(row["mode_1_mean"]) <= 1.1 * exact_b
            ),
        }
        ranked.append(response)
    ranked.sort(key=lambda row: (
        not row["mode_A_improves"],
        not row["mode_B_within_10pct"],
        row["delta_mode_A_vs_exact"],
        row["delta_mode_B_vs_exact"],
        row["j14"],
        row["candidate_id"],
    ))
    for index, row in enumerate(ranked, start=1):
        row["atlas_rank"] = index

    pairs = []
    for coordinate_index in range(28):
        pair = [
            row for row in ranked if row["coordinate_index"] == coordinate_index
        ]
        if len(pair) != 2:
            raise RuntimeError("coordinate response lacks an exact sign pair")
        negative = next(row for row in pair if row["sign"] == -1)
        positive = next(row for row in pair if row["sign"] == 1)
        pairs.append({
            "coordinate_index": coordinate_index,
            "mode_nx": positive["mode_nx"],
            "mode_ny": positive["mode_ny"],
            "phase": positive["phase"],
            "signed_A_response_half_difference": 0.5 * (
                positive["mode_0_mean"] - negative["mode_0_mean"]
            ),
            "signed_B_response_half_difference": 0.5 * (
                positive["mode_1_mean"] - negative["mode_1_mean"]
            ),
            "signed_J14_response_half_difference": 0.5 * (
                positive["j14"] - negative["j14"]
            ),
            "best_sign_candidate_id": min(
                pair, key=lambda row: (
                    not row["mode_A_improves"],
                    not row["mode_B_within_10pct"],
                    row["delta_mode_A_vs_exact"], row["candidate_id"],
                )
            )["candidate_id"],
        })
    return ranked, pairs


def aggregate(
    *, config_path: Path = DEFAULT_CONFIG,
    j14_config_path: Path = DEFAULT_J14_CONFIG,
    support_config_path: Path = DEFAULT_SUPPORT_CONFIG,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config, manifest, manifest_path = _load_manifest(config_path, artifact_root)
    provenance = _runtime_provenance(str(manifest["provenance"]["git_commit"]))
    records, inventory = _inventory(
        config=config, manifest=manifest, manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    status = "INCOMPLETE"
    input_error = None
    flat_rows: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    if not provenance["formal_ready"]:
        status = "INVALID_PROVENANCE"
        input_error = "analysis is not a clean analysis-only descendant"
    elif inventory["complete_cartesian_product"]:
        try:
            j14_config = json.loads(j14_config_path.read_text())
            context = historical._patient_context(j14_config, artifact_root)
            support_context = canary._load_support_context(
                support_config_path, artifact_root, j14_config,
            )
            scored = [
                canary._score_worker(row, context, support_context) for row in records
            ]
            candidates = {
                row["candidate_id"]: row for row in manifest["candidates"]
            }
            flat_rows = [
                _flat_row(row, candidates[row["candidate_id"]]) for row in scored
            ]
            ranked, pairs = rank_coordinate_responses(flat_rows)
            status = "COMPLETE"
        except Exception as error:
            status = "INVALID_INPUT"
            input_error = str(error)
    output_root = artifact_root / str(config["output_root"]) / "analysis"
    json_path = output_root / "m3_coordinate_atlas_aggregate.json"
    rows_path = output_root / "m3_coordinate_atlas_per_candidate.csv"
    pairs_path = output_root / "m3_coordinate_atlas_signed_pairs.csv"
    eligible = [
        row for row in ranked
        if row["mode_A_improves"] and row["mode_B_within_10pct"]
    ]
    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": status,
        "scientific_role": "training_only_complete_m3_coordinate_response_atlas",
        "inventory": inventory,
        "input_error": input_error,
        "provenance": provenance,
        "ranking_contract": {
            "primary_mode_identity": "frozen_patient_training_old_A_B",
            "mode_A_must_improve_vs_same_seed_exact_off": True,
            "mode_B_tolerance": "candidate D_B <= 1.10 * exact_off D_B",
            "natural_kmeans_used": False,
            "patient_heldout_used": False,
            "ictal_data_used": False,
            "snn_simulation_run_by_aggregator": False,
        },
        "per_candidate": flat_rows,
        "ranked_coordinate_responses": ranked,
        "signed_coordinate_pairs": pairs,
        "a_improving_b_preserving_candidate_ids": [
            row["candidate_id"] for row in eligible
        ],
        "best_coordinate_candidate": eligible[0]["candidate_id"] if eligible else None,
        "claim_boundary": (
            "One-network training-only coordinate response. It can construct a "
            "fresh-seed combination test but cannot establish natural K=2, "
            "held-out agreement, convergence or continuous-field failure."
        ),
        "outputs": {
            "json": str(json_path), "per_candidate_csv": str(rows_path),
            "signed_pairs_csv": str(pairs_path),
        },
    }
    _atomic_json(json_path, payload)
    _atomic_csv(rows_path, ranked if ranked else flat_rows)
    _atomic_csv(pairs_path, pairs)
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
        "best_coordinate_candidate": payload["best_coordinate_candidate"],
        "n_a_improving_b_preserving": len(
            payload["a_improving_b_preserving_candidate_ids"]
        ),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()

