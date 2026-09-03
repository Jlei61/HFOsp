#!/usr/bin/env python3
"""Pair low/high initial-condition arms and aggregate the spatial Z/M screen."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_phase_diagram import (  # noqa: E402
    adjudicate_seed_family,
    classify_paired_initial_states,
)


def _coordinate_key(record):
    coordinate = record["coordinates"]
    return (
        float(coordinate["q_clamp"]),
        float(coordinate["eta_m"]),
        int(coordinate["noise_seed"]),
    )


def pair_records(records):
    """Pair arms fail-closed and audit the matched-future-noise contract."""
    grouped = defaultdict(dict)
    config_hashes = set()
    scientific_contract_hashes = set()
    for record in records:
        if record.get("status") != "SPATIAL_ZM_PHASE_POINT_COMPLETE":
            continue
        coordinate = record["coordinates"]
        arm = str(coordinate["initial_state"])
        if arm not in {"low", "high"}:
            raise ValueError(f"invalid initial-state arm: {arm}")
        key = _coordinate_key(record)
        if arm in grouped[key]:
            raise ValueError(f"duplicate {arm} arm at {key}")
        grouped[key][arm] = record
        config_hashes.add(record["phase_config"]["sha256"])
        scientific_contract_hashes.add(record.get(
            "scientific_contract_sha256", record["phase_config"]["sha256"]))
    if not grouped:
        raise ValueError("no completed spatial Z/M phase points found")
    if len(scientific_contract_hashes) != 1:
        raise ValueError("phase points mix different scientific contract hashes")

    pairs = []
    for (q_clamp, eta_m, noise_seed), arms in sorted(grouped.items()):
        missing = {"low", "high"} - set(arms)
        if missing:
            raise ValueError(
                f"incomplete initial-state pair at {(q_clamp, eta_m, noise_seed)}: "
                f"missing {sorted(missing)}")
        low, high = arms["low"], arms["high"]
        low_noise = low["paired_noise_contract"]["future_noise_sha256"]
        high_noise = high["paired_noise_contract"]["future_noise_sha256"]
        if low_noise != high_noise:
            raise ValueError(
                f"future-noise mismatch at {(q_clamp, eta_m, noise_seed)}")
        low_label = low["classification"]["label"]
        high_label = high["classification"]["label"]
        pair_label = classify_paired_initial_states(low_label, high_label)
        pairs.append({
            "q_clamp": q_clamp,
            "eta_m": eta_m,
            "noise_seed": noise_seed,
            "low_start_label": low_label,
            "high_start_label": high_label,
            "pair_label": pair_label,
            "future_noise_sha256": low_noise,
            "low_median_rate_hz": low["stationary_metrics"]["median_rate_hz"],
            "high_median_rate_hz": high["stationary_metrics"]["median_rate_hz"],
            "low_active_fraction": low["stationary_metrics"][
                "median_active_E_fraction_20ms"],
            "high_active_fraction": high["stationary_metrics"][
                "median_active_E_fraction_20ms"],
            "low_sheet_fraction": low["stationary_metrics"][
                "median_recruited_sheet_fraction_1mm"],
            "high_sheet_fraction": high["stationary_metrics"][
                "median_recruited_sheet_fraction_1mm"],
            "low_json": low["_input_path"],
            "high_json": high["_input_path"],
        })
    return pairs, {
        "scientific_contract_sha256": next(iter(scientific_contract_hashes)),
        "phase_config_sha256": sorted(config_hashes),
    }


def aggregate_pairs(pairs, *, minimum_seeds=3):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair["q_clamp"], pair["eta_m"])].append(pair)
    families = []
    for (q_clamp, eta_m), rows in sorted(grouped.items()):
        labels = [row["pair_label"] for row in rows]
        adjudication = adjudicate_seed_family(
            labels, minimum_seeds=minimum_seeds)
        families.append({
            "q_clamp": q_clamp,
            "eta_m": eta_m,
            "noise_seeds": sorted(row["noise_seed"] for row in rows),
            "adjudication": adjudication,
        })
    return families


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", action="append", required=True,
        help=("Directory containing phase-point JSONs; repeat to aggregate "
              "only explicitly selected complete stages."))
    parser.add_argument("--out", required=True, help="Aggregate JSON path")
    parser.add_argument("--minimum-seeds", type=int, default=3)
    args = parser.parse_args()
    if args.minimum_seeds < 1:
        raise SystemExit("--minimum-seeds must be positive")

    records = []
    input_paths = sorted({
        path.resolve()
        for directory in args.input_dir
        for path in Path(directory).rglob("*.json")
    })
    for path in input_paths:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") == "SPATIAL_ZM_PHASE_POINT_COMPLETE":
            record["_input_path"] = str(path.resolve())
            records.append(record)
    pairs, contract_identity = pair_records(records)
    families = aggregate_pairs(pairs, minimum_seeds=args.minimum_seeds)
    output = Path(args.out).resolve()
    payload = {
        "status": "SPATIAL_ZM_PHASE_DIAGRAM_AGGREGATED",
        **contract_identity,
        "n_arms": len(records),
        "n_pairs": len(pairs),
        "minimum_seeds_for_robust_label": int(args.minimum_seeds),
        "pairs": pairs,
        "families": families,
        "claim_boundary": (
            "Finite stochastic SNN branch evidence only; mathematical "
            "bifurcation requires deterministic continuation and Jacobian evidence."),
    }
    _atomic_json(payload, output)
    _atomic_csv(pairs, output.with_suffix(".csv"))
    print(json.dumps({
        "status": payload["status"],
        "n_pairs": len(pairs),
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
