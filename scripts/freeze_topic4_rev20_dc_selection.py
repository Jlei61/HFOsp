#!/usr/bin/env python3
"""Freeze one rev20-DC non-reference level per family from selection data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _summaries(aggregate: dict, manifest: dict, expected_seeds: list[int]) -> dict:
    rows = defaultdict(list)
    for row in aggregate["per_network"]:
        rows[row["candidate_id"]].append(row)
    output = {}
    for candidate in manifest["candidates"]:
        identifier = candidate["candidate_id"]
        values = rows.get(identifier, [])
        by_seed = {int(row["seed"]): row for row in values}
        distances = []
        valid = set(by_seed) == set(expected_seeds)
        invalid_reasons = []
        if not valid:
            invalid_reasons.append("INCOMPLETE_PAIRED_SEED_SET")
        for seed in expected_seeds:
            row = by_seed.get(seed)
            if row is None:
                continue
            if row.get("runaway_early_stop_ms") is not None:
                valid = False
                invalid_reasons.append(f"RUNAWAY_SEED_{seed}")
            distance = row.get("selection", {}).get(
                "complete_distribution_distance_training"
            )
            if distance is None or not np.isfinite(float(distance)):
                valid = False
                invalid_reasons.append(f"UNESTIMABLE_SEED_{seed}")
            else:
                distances.append(float(distance))
        output[identifier] = {
            "candidate_id": identifier,
            "family": candidate["family"],
            "level": candidate["level"],
            "is_reference": bool(candidate["is_reference"]),
            "selection_eligible": bool(valid),
            "invalid_reasons": sorted(set(invalid_reasons)),
            "network_count": len(by_seed),
            "mean_training_distance": (
                float(np.mean(distances)) if distances else None
            ),
            "median_training_distance": (
                float(np.median(distances)) if distances else None
            ),
            "worst_seed_training_distance": (
                float(np.max(distances)) if distances else None
            ),
            "per_seed_training_distance": {
                str(seed): (
                    None if seed not in by_seed else by_seed[seed]["selection"].get(
                        "complete_distribution_distance_training"
                    )
                ) for seed in expected_seeds
            },
        }
    return output


def freeze_selection(config: dict, aggregate: dict, manifest: dict) -> dict:
    if aggregate.get("phase") != "screen":
        raise RuntimeError("selection requires the screen aggregate")
    if aggregate.get("heldout_opened"):
        raise RuntimeError("held-out was opened before selection")
    if aggregate.get("validation_endpoint_status") != "SEALED_UNTIL_SELECTION_FREEZE":
        raise RuntimeError("validation endpoints were not sealed during selection")
    for row in aggregate["per_network"]:
        if "validation" in row or "validation_diagnostic" in row:
            raise RuntimeError("screen aggregate contains forbidden validation fields")
        selection = row.get("selection", {})
        if any(bool(selection.get(key)) for key in (
                "selection_used_labels", "selection_used_ood",
                "selection_used_heldout")):
            raise RuntimeError("selection endpoint consumed a forbidden target")

    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    summaries = _summaries(aggregate, manifest, seeds)
    reference = next(
        row for row in summaries.values() if row["is_reference"]
    )
    if not reference["selection_eligible"]:
        raise RuntimeError("reference candidate is not selection-estimable")

    selected = [reference["candidate_id"]]
    family_winners = {}
    families = sorted({
        row["family"] for row in summaries.values() if not row["is_reference"]
    })
    for family in families:
        eligible = [
            row for row in summaries.values()
            if row["family"] == family and row["selection_eligible"]
        ]
        if not eligible:
            family_winners[family] = None
            continue
        winner = min(eligible, key=lambda row: (
            row["mean_training_distance"],
            row["worst_seed_training_distance"],
            str(row["candidate_id"]),
        ))
        winner = dict(winner)
        winner["paired_mean_delta_vs_reference"] = float(np.mean([
            float(winner["per_seed_training_distance"][str(seed)])
            - float(reference["per_seed_training_distance"][str(seed)])
            for seed in seeds
        ]))
        winner["better_than_reference_on_training_mean"] = bool(
            winner["mean_training_distance"]
            < reference["mean_training_distance"]
        )
        family_winners[family] = winner
        selected.append(winner["candidate_id"])

    return {
        "schema_id": "topic4_rev20_dc_frozen_selection_v1",
        "status": "REV20_DC_SELECTION_FROZEN",
        "candidate_ids": selected,
        "reference_candidate_id": reference["candidate_id"],
        "family_winners": family_winners,
        "candidate_summaries": summaries,
        "selection_rule": (
            "within each family: complete paired seed set, no runaway, finite "
            "training complete-distribution distance; minimize equal-network mean, "
            "then worst-seed distance, then candidate id"
        ),
        "validation_used_for_selection": False,
        "heldout_used_for_selection": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    aggregate_path = args.aggregate.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("candidate manifest is stale")
    payload = freeze_selection(config, aggregate, manifest)
    payload.update({
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(manifest_path),
        "screen_aggregate_sha256": _sha256(aggregate_path),
    })
    _atomic_json(args.out.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "candidate_ids": payload["candidate_ids"],
        "output": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
