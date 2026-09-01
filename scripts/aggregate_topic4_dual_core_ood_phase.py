#!/usr/bin/env python3
"""Aggregate one dual-core OOD phase with network seed as the unit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_dual_core_ood import (  # noqa: E402
    candidate_sort_key,
    load_embedding,
    score_returned_event_support,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _score_worker(npz_path, contract, embedding, classifier):
    with np.load(npz_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["onsets"], float)
        returned = np.asarray(loaded["event_returned"], bool)
    scored = score_returned_event_support(
        onsets, returned, contract=contract, embedding=embedding,
        classifier=classifier,
    )
    groups = contract_groups(contract)
    event_rows = []
    for index in range(len(onsets)):
        event_rows.append({
            "event_index": index,
            "returned": bool(returned[index]),
            "readable": bool(scored["readable"][index]),
            "in_support": bool(scored["in_support"][index]),
            "mode": int(scored["labels"][index]),
            "ood": bool(scored["ood"][index]),
            "normalized_support_distance": float(
                scored["normalized_support_distance"][index]
            ),
            "ICL_recruited": int(np.sum(np.isfinite(
                onsets[index, groups["ICL"]]
            ))),
            "SCL_recruited": int(np.sum(np.isfinite(
                onsets[index, groups["SCL"]]
            ))),
        })
    return {
        key: value for key, value in scored.items()
        if key not in {
            "labels", "ood", "readable", "in_support",
            "normalized_support_distance",
        }
    }, event_rows


def aggregate(config_path: Path, phase: str):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"] / phase
    manifest_path = root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "REV16_DUAL_CORE_OOD_PHASE_FROZEN":
        raise RuntimeError("phase manifest is not frozen")
    contract_record = config["inputs"]["contact_contract"]
    contract_path = ROOT / contract_record["path"]
    if _sha256(contract_path) != contract_record["sha256"]:
        raise RuntimeError("contact contract changed")
    contract = json.loads(contract_path.read_text())
    target_record = config["inputs"]["shaft_aware_target_npz"]
    target_path = ROOT / target_record["path"]
    if _sha256(target_path) != target_record["sha256"]:
        raise RuntimeError("shaft-aware embedding changed")
    embedding = load_embedding(str(target_path))
    classifier = manifest["direction_classifier"]
    seeds = list(map(int, manifest["fixed_contract"]["network_seeds"]))
    summaries, per_network = [], []
    for candidate in manifest["candidate_set"]["candidates"]:
        rows = []
        for seed in seeds:
            stem = root / "workers" / f"{candidate['candidate_id']}_seed_{seed}"
            json_path = stem.with_suffix(".json")
            npz_path = stem.with_suffix(".npz")
            if not json_path.is_file() or not npz_path.is_file():
                raise RuntimeError(f"missing worker: {stem.name}")
            payload = json.loads(json_path.read_text())
            if payload.get("status") != "REV10R_EDGE_FLOW_WORKER_COMPLETE":
                raise RuntimeError(f"worker incomplete: {json_path}")
            if payload["arrays"]["sha256"] != _sha256(npz_path):
                raise RuntimeError(f"worker array hash changed: {npz_path}")
            score, events = _score_worker(
                npz_path, contract, embedding, classifier,
            )
            row = {
                "candidate_id": candidate["candidate_id"],
                "seed": seed,
                **score,
                "events": events,
                "worker_json": str(json_path.relative_to(ROOT)),
                "worker_json_sha256": _sha256(json_path),
                "worker_npz": str(npz_path.relative_to(ROOT)),
                "worker_npz_sha256": _sha256(npz_path),
            }
            rows.append(row)
            per_network.append(row)
        mode_distance = []
        for mode in (0, 1):
            values = [
                row["mean_normalized_support_distance_by_mode"][mode]
                for row in rows
                if row["mean_normalized_support_distance_by_mode"][mode] is not None
            ]
            mode_distance.append(float(np.mean(values)) if values else None)
        summary = {
            "candidate_id": candidate["candidate_id"],
            "node_field": candidate["node_field"],
            "n_networks": len(rows),
            "networks_with_both_modes": int(sum(
                row["both_modes_in_support"] for row in rows
            )),
            "equal_network_ood_all_returned": float(np.mean([
                row["ood_all_returned"] for row in rows
            ])),
            "equal_network_ood_returned_readable": float(np.mean([
                row["ood_returned_readable"] for row in rows
            ])),
            "equal_network_unreadable_returned_fraction": float(np.mean([
                row["unreadable_returned_fraction"] for row in rows
            ])),
            "equal_network_returned_events": float(np.mean([
                row["n_returned"] for row in rows
            ])),
            "pooled_mode_counts_in_support": np.sum([
                row["mode_counts_in_support"] for row in rows
            ], axis=0),
            "mean_normalized_support_distance_by_mode": mode_distance,
            "weakest_mode_normalized_support_distance": (
                float(max(mode_distance))
                if all(value is not None for value in mode_distance) else None
            ),
            "per_network": rows,
        }
        summaries.append(summary)
    ranking = sorted(summaries, key=candidate_sort_key)
    output = {
        "status": "DUAL_CORE_OOD_PHASE_COMPLETE",
        "phase": phase,
        "primary_metric": config["search"]["primary_metric"],
        "ranking_rule": [
            "networks_with_both_modes descending",
            "equal_network_ood_all_returned ascending",
            "weakest_mode_normalized_support_distance ascending",
            "equal_network_returned_events descending",
        ],
        "n_candidates": len(summaries),
        "n_networks_per_candidate": len(seeds),
        "ranking": ranking,
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": _sha256(manifest_path),
        },
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    _atomic_json(root / "aggregate.json", output)
    csv_path = root / "per_network_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "seed", "n_returned", "n_returned_readable",
            "n_in_support", "mode_A_in_support", "mode_B_in_support",
            "both_modes_in_support", "ood_all_returned",
            "ood_returned_readable", "unreadable_returned_fraction",
        ])
        writer.writeheader()
        for row in per_network:
            writer.writerow({
                "candidate_id": row["candidate_id"], "seed": row["seed"],
                "n_returned": row["n_returned"],
                "n_returned_readable": row["n_returned_readable"],
                "n_in_support": row["n_in_support"],
                "mode_A_in_support": row["mode_counts_in_support"][0],
                "mode_B_in_support": row["mode_counts_in_support"][1],
                "both_modes_in_support": row["both_modes_in_support"],
                "ood_all_returned": row["ood_all_returned"],
                "ood_returned_readable": row["ood_returned_readable"],
                "unreadable_returned_fraction": row[
                    "unreadable_returned_fraction"
                ],
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=("fit", "selection", "confirmation", "pathway"), required=True)
    args = parser.parse_args()
    output = aggregate(Path(args.config), args.phase)
    print(json.dumps({
        "status": output["status"], "phase": args.phase,
        "top": output["ranking"][0]["candidate_id"],
        "top_ood": output["ranking"][0]["equal_network_ood_all_returned"],
    }, indent=2))


if __name__ == "__main__":
    main()
