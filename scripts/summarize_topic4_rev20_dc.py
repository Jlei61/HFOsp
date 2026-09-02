#!/usr/bin/env python3
"""Summarize the frozen rev20-DC screen and confirmation response atlas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REFERENCE_LEVELS = {
    "node_gain": 1.0,
    "core_budget_scale": 1.0,
    "signed_depth_shrinkage": 1.0,
    "g_EE": 0.5,
    "g_EtoI": 1.0,
    "both_scale": 1.0,
    "ellipse_angle_deg": 45.0,
    "ellipse_aspect_ratio": 2.0,
}
METRICS = (
    "training_complete_distribution",
    "heldout_complete_distribution",
    "kmeans_balanced_alignment",
    "ood_all_returned",
    "unreadable_fraction",
    "returned_family_count",
)


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


def _metric(row: dict, name: str):
    selection = row.get("selection", {})
    validation = row.get("validation", {})
    if name == "training_complete_distribution":
        return selection.get("complete_distribution_distance_training")
    if name == "heldout_complete_distribution":
        return validation.get("complete_distribution_distance_reference")
    if name == "kmeans_balanced_alignment":
        return validation.get("direction_balanced_alignment")
    if name == "ood_all_returned":
        return validation.get("ood_all_returned")
    if name == "unreadable_fraction":
        return validation.get("unreadable_fraction")
    if name == "returned_family_count":
        return validation.get("n_returned_families")
    raise KeyError(name)


def _bootstrap_mean(values, *, draws: int, seed: int) -> dict | None:
    values = np.asarray([value for value in values if value is not None], float)
    if not len(values):
        return None
    rng = np.random.default_rng(int(seed))
    means = np.mean(
        rng.choice(values, size=(int(draws), len(values)), replace=True), axis=1,
    )
    return {
        "n_networks": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "bootstrap_q05": float(np.quantile(means, 0.05)),
        "bootstrap_q95": float(np.quantile(means, 0.95)),
    }


def _candidate_rows(aggregate: dict) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in aggregate["per_network"]:
        grouped[str(row["candidate_id"])].append(row)
    return dict(grouped)


def _candidate_summary(rows: list[dict], *, reference_rows: list[dict],
                       draws: int, seed: int) -> dict:
    by_seed = {int(row["seed"]): row for row in rows}
    reference = {int(row["seed"]): row for row in reference_rows}
    metrics, paired = {}, {}
    for offset, name in enumerate(METRICS):
        values = [_metric(row, name) for row in rows]
        metrics[name] = _bootstrap_mean(
            values, draws=draws, seed=seed + offset,
        )
        deltas = []
        for network_seed in sorted(set(by_seed).intersection(reference)):
            value = _metric(by_seed[network_seed], name)
            base = _metric(reference[network_seed], name)
            if value is not None and base is not None:
                deltas.append(float(value) - float(base))
        paired[name] = _bootstrap_mean(
            deltas, draws=draws, seed=seed + 100 + offset,
        )
    floor_keys = (
        ("training_patient_floor", "training_floor"),
        ("heldout_patient_floor", "heldout_floor"),
    )
    floors = {}
    for source, target in floor_keys:
        entries = [row[source] for row in rows if source in row]
        floors[target] = {
            quantile: _bootstrap_mean(
                [entry[quantile] for entry in entries],
                draws=draws, seed=seed + 200 + index,
            )
            for index, quantile in enumerate(("q05", "q50", "q95"))
        } if entries else None
    return {
        "network_seeds": sorted(by_seed),
        "metrics": metrics,
        "paired_delta_vs_reference": paired,
        **floors,
    }


def summarize(config: dict, manifest: dict, selection: dict,
              screen: dict, confirmation: dict) -> dict:
    if screen.get("validation_endpoint_status") != "FROZEN_VALIDATION_OPENED":
        raise RuntimeError("screen validation response surface is not open")
    if confirmation.get("validation_endpoint_status") != "FROZEN_VALIDATION_OPENED":
        raise RuntimeError("confirmation validation endpoints are not open")
    if not screen.get("heldout_opened") or not confirmation.get("heldout_opened"):
        raise RuntimeError("held-out complete-distribution endpoint is missing")
    screen_rows = _candidate_rows(screen)
    confirmation_rows = _candidate_rows(confirmation)
    reference_id = selection["reference_candidate_id"]
    if reference_id not in screen_rows or reference_id not in confirmation_rows:
        raise RuntimeError("reference candidate missing from response atlas")
    draws = int(config["validation"]["paired_bootstrap_draws"])
    seed = int(config["validation"]["paired_bootstrap_seed"])
    candidates = {}
    for index, candidate in enumerate(manifest["candidates"]):
        identifier = candidate["candidate_id"]
        if identifier not in screen_rows:
            raise RuntimeError(f"screen candidate missing: {identifier}")
        selection_audit = selection.get("candidate_summaries", {}).get(
            identifier, {}
        )
        candidates[identifier] = {
            "candidate_id": identifier,
            "family": candidate["family"],
            "level": candidate["level"],
            "is_reference": bool(candidate["is_reference"]),
            "screen_eligible": bool(
                selection_audit.get("selection_eligible", True)
            ),
            "screen_invalid_reasons": list(
                selection_audit.get("invalid_reasons", [])
            ),
            "selected_for_confirmation": identifier in selection["candidate_ids"],
            "screen": _candidate_summary(
                screen_rows[identifier], reference_rows=screen_rows[reference_id],
                draws=draws, seed=seed + index * 1000,
            ),
            "confirmation": (
                _candidate_summary(
                    confirmation_rows[identifier],
                    reference_rows=confirmation_rows[reference_id],
                    draws=draws, seed=seed + 50000 + index * 1000,
                ) if identifier in confirmation_rows else None
            ),
        }

    reference = candidates[reference_id]
    family_curves = {}
    for family, reference_level in REFERENCE_LEVELS.items():
        rows = [
            row for row in candidates.values() if row["family"] == family
        ]
        reference_copy = {
            **reference, "family": family, "level": reference_level,
            "candidate_id": reference_id,
        }
        rows.append(reference_copy)
        rows.sort(key=lambda row: float(row["level"]))
        monotonic = {}
        levels = np.asarray([float(row["level"]) for row in rows])
        for name in METRICS[:4]:
            values = np.asarray([
                row["screen"]["metrics"][name]["mean"]
                if row["screen"]["metrics"][name] is not None else np.nan
                for row in rows
            ])
            finite = np.isfinite(values)
            monotonic[name] = (
                float(spearmanr(levels[finite], values[finite]).statistic)
                if np.sum(finite) >= 3 else None
            )
        family_curves[family] = {
            "reference_level": reference_level,
            "selected_candidate_id": (
                None if selection["family_winners"].get(family) is None
                else selection["family_winners"][family]["candidate_id"]
            ),
            "rows": rows,
            "screen_level_spearman": monotonic,
        }
    return {
        "schema_id": "topic4_rev20_dc_response_atlas_v1",
        "status": "REV20_DC_RESPONSE_ATLAS_COMPLETE",
        "reference_candidate_id": reference_id,
        "selected_candidate_ids": selection["candidate_ids"],
        "candidates": candidates,
        "family_curves": family_curves,
        "statistical_unit": "paired network seed",
        "bootstrap_interval": "90% paired network bootstrap",
        "selection_boundary": (
            "candidate levels frozen from training complete-distribution distance "
            "before KMeans, OOD or held-out endpoint opening"
        ),
        "claim_boundary": config["claim_boundary"],
    }


def _write_csv(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "candidate_id", "family", "level", "selected_for_confirmation",
            "phase", "metric", "mean", "q05", "q95", "paired_delta_mean",
            "paired_delta_q05", "paired_delta_q95",
        ))
        writer.writeheader()
        for candidate in payload["candidates"].values():
            for phase in ("screen", "confirmation"):
                values = candidate.get(phase)
                if values is None:
                    continue
                for metric in METRICS:
                    summary = values["metrics"][metric]
                    paired = values["paired_delta_vs_reference"][metric]
                    writer.writerow({
                        "candidate_id": candidate["candidate_id"],
                        "family": candidate["family"],
                        "level": candidate["level"],
                        "selected_for_confirmation": candidate[
                            "selected_for_confirmation"
                        ],
                        "phase": phase,
                        "metric": metric,
                        "mean": None if summary is None else summary["mean"],
                        "q05": None if summary is None else summary["bootstrap_q05"],
                        "q95": None if summary is None else summary["bootstrap_q95"],
                        "paired_delta_mean": None if paired is None else paired["mean"],
                        "paired_delta_q05": (
                            None if paired is None else paired["bootstrap_q05"]
                        ),
                        "paired_delta_q95": (
                            None if paired is None else paired["bootstrap_q95"]
                        ),
                    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--confirmation", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name).resolve()
        for name in ("config", "manifest", "selection", "screen", "confirmation")
    }
    loaded = {name: json.loads(path.read_text()) for name, path in paths.items()}
    payload = summarize(
        loaded["config"], loaded["manifest"], loaded["selection"],
        loaded["screen"], loaded["confirmation"],
    )
    payload["input_sha256"] = {name: _sha256(path) for name, path in paths.items()}
    _atomic_json(args.out_json.resolve(), payload)
    _write_csv(args.out_csv.resolve(), payload)
    print(json.dumps({
        "status": payload["status"], "out_json": str(args.out_json.resolve()),
        "out_csv": str(args.out_csv.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
