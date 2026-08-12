"""Freeze a small, diverse rev10-R2 library for fresh-network selection."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _unit_direction(candidate):
    values = np.asarray(candidate.get(
        "latent_whitened_direction", candidate["coefficients"],
    ), float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"nonzero candidate has invalid direction: {candidate['candidate_id']}")
    return values / norm


def select_diverse_pareto(rows, candidates, maximum):
    """Select fit-Pareto candidates without collapsing onto one direction."""
    by_id = {row["candidate_id"]: row for row in rows}
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    pool = [
        row for row in rows
        if row["candidate_id"] != "edge_noop"
        and row["n_runaway_networks"] == 0
        and row["pareto_nondominated"]
    ]
    if not pool:
        return []
    missing = sorted(
        row["candidate_id"] for row in pool
        if row["candidate_id"] not in candidate_by_id
    )
    if missing:
        raise KeyError(f"fit rows absent from source manifest: {missing}")
    pool.sort(key=lambda row: (
        row["selection_score_equal_network"], row["candidate_id"],
    ))
    selected = [pool.pop(0)]
    selected_vectors = [_unit_direction(
        candidate_by_id[selected[0]["candidate_id"]]
    )]
    while pool and len(selected) < int(maximum):
        ranked = []
        for row in pool:
            vector = _unit_direction(candidate_by_id[row["candidate_id"]])
            min_distance = min(
                1.0 - float(np.clip(vector @ prior, -1.0, 1.0))
                for prior in selected_vectors
            )
            ranked.append((
                -min_distance,
                row["selection_score_equal_network"],
                row["candidate_id"], row, vector,
            ))
        _, _, _, chosen, vector = min(ranked, key=lambda item: item[:3])
        selected.append(chosen)
        selected_vectors.append(vector)
        pool = [row for row in pool if row is not chosen]
    return [row["candidate_id"] for row in selected]


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    expected_role = (
        "development_only_observation_invariant_spatial_route_selection"
    )
    if config.get("scientific_role") != expected_role:
        raise RuntimeError("rev10-R2 selection scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    fit_summary_path = ROOT / config["inputs"]["fit_screen_summary"]["path"]
    source_manifest_path = ROOT / config["inputs"]["fit_candidate_manifest"]["path"]
    fit_summary = json.loads(fit_summary_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    if fit_summary.get("status") != "REV10R_FIT_SCREEN_COMPLETE":
        raise RuntimeError("fit screen is incomplete")
    if source_manifest.get("status") != "REV10R2_SPATIAL_EDGE_LIBRARY_FROZEN":
        raise RuntimeError("source candidate library is not rev10-R2")
    if fit_summary["manifest"]["sha256"] != _sha256(source_manifest_path):
        raise RuntimeError("fit summary and source candidate manifest differ")

    candidates = source_manifest["candidate_set"]["candidates"]
    maximum = int(config["search"]["objective"][
        "maximum_fit_pareto_candidates"
    ])
    selected_ids = select_diverse_pareto(
        fit_summary["candidate_rows"], candidates, maximum,
    )
    if not selected_ids:
        raise RuntimeError("fit screen has no safe nonzero Pareto candidate")
    by_id = {row["candidate_id"]: row for row in candidates}
    frozen = [by_id["edge_noop"], *[by_id[key] for key in selected_ids]]

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    provenance["config_dirty"] = config_dirty
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("selection freezer runtime is not frozen")

    return {
        "status": "REV10R2_SPATIAL_EDGE_SELECTION_LIBRARY_FROZEN",
        "scientific_role": expected_role,
        "candidate_set": {
            "candidates": frozen,
            "n_nonzero": len(selected_ids),
            "n_exact_noop": 1,
        },
        "fit_selection": {
            "rule": (
                "safe fit-Pareto candidates; first by equal-network score, "
                "then greedy maximum minimum angular distance in the frozen "
                "whitened coefficient coordinates"
            ),
            "selected_nonzero_candidate_ids": selected_ids,
            "maximum_nonzero_candidates": maximum,
            "source_fit_summary": {
                "path": str(fit_summary_path.relative_to(ROOT)),
                "sha256": _sha256(fit_summary_path),
            },
            "source_candidate_manifest": {
                "path": str(source_manifest_path.relative_to(ROOT)),
                "sha256": _sha256(source_manifest_path),
            },
            "selection_networks_were_read": False,
        },
        "direction_classifier": source_manifest["direction_classifier"],
        "fixed_contract": {
            **source_manifest["fixed_contract"],
            "fit_network_seeds": config["search"]["fit_network_seeds"],
            "selection_network_seeds": config["search"]["selection_network_seeds"],
        },
        "inputs": config["inputs"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.out or ROOT / config["output_root"] / "candidate_manifest.json")
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "selected": payload["fit_selection"]["selected_nonzero_candidate_ids"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
