#!/usr/bin/env python3
"""Analyze the nine-network Stage-X paired field replication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from scripts.analyze_topic4_rev12_orthogonal_anchor_relative import (  # noqa: E402
    MODE_ENDPOINTS,
    all_aggregate_utilities,
    all_network_utilities,
)
from scripts.analyze_topic4_rev12_orthogonal_free_field_screen import (  # noqa: E402
    NETWORK_ENDPOINTS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def paired_bootstrap(values: dict[str, float], *, draws: int,
                     quantiles: tuple[float, float], seed: int) -> dict:
    ordered = np.asarray([values[key] for key in sorted(values)], float)
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(ordered, size=(int(draws), len(ordered)), replace=True)
    means = np.mean(sampled, axis=1)
    return {
        "n_networks": int(len(ordered)),
        "mean": float(np.mean(ordered)),
        "median": float(np.median(ordered)),
        "n_positive": int(np.sum(ordered > 0.0)),
        "n_negative": int(np.sum(ordered < 0.0)),
        "q_low": float(np.quantile(means, quantiles[0])),
        "q_high": float(np.quantile(means, quantiles[1])),
        "quantiles": list(quantiles),
        "draws": int(draws),
    }


def analyze_replication(rows: dict[str, dict], *, decision: dict) -> dict:
    anchor_id = str(decision["anchor_candidate_id"])
    candidate_id = str(decision["candidate_id"])
    if set((anchor_id, candidate_id)) - rows.keys():
        raise RuntimeError("paired replication aggregate lacks one field")
    anchor_aggregate = all_aggregate_utilities(rows[anchor_id])
    candidate_aggregate = all_aggregate_utilities(rows[candidate_id])
    aggregate_delta = {
        endpoint: float(candidate_aggregate[endpoint] - anchor_aggregate[endpoint])
        for endpoint in anchor_aggregate
    }
    anchor_network = all_network_utilities(rows[anchor_id])
    candidate_network = all_network_utilities(rows[candidate_id])
    if anchor_network.keys() != candidate_network.keys():
        raise RuntimeError("paired replication network pools differ")
    network_delta = {
        endpoint: {
            str(seed): float(
                candidate_network[seed][endpoint] - anchor_network[seed][endpoint]
            )
            for seed in sorted(anchor_network)
        }
        for endpoint in NETWORK_ENDPOINTS + MODE_ENDPOINTS
    }
    quantiles = tuple(float(value) for value in decision["bootstrap_interval"])
    bootstrap = {
        endpoint: paired_bootstrap(
            values, draws=int(decision["bootstrap_draws"]),
            quantiles=quantiles, seed=int(decision["bootstrap_seed"]) + index,
        )
        for index, (endpoint, values) in enumerate(network_delta.items())
    }
    required_aggregate = list(decision["required_aggregate_endpoints"])
    required_sign = list(decision["required_sign_endpoints"])
    minimum_positive = int(decision["minimum_positive_networks"])
    failed_aggregate = [
        endpoint for endpoint in required_aggregate
        if aggregate_delta[endpoint] <= 0.0
    ]
    failed_sign = [
        endpoint for endpoint in required_sign
        if bootstrap[endpoint]["n_positive"] < minimum_positive
    ]
    topology_anchor = rows[anchor_id]["source_topology"]
    topology_candidate = rows[candidate_id]["source_topology"]
    topology_delta = {
        key: float(topology_candidate[key] - topology_anchor[key])
        for key in (
            "mean_within_network_split_half_cosine",
            "mean_across_network_template_cosine",
            "equal_network_between_mode_distance",
        )
    }
    advances = not failed_aggregate and not failed_sign
    return {
        "anchor_candidate_id": anchor_id,
        "candidate_id": candidate_id,
        "aggregate_delta_from_anchor": aggregate_delta,
        "network_delta_from_anchor": network_delta,
        "paired_network_bootstrap": bootstrap,
        "source_topology_delta": topology_delta,
        "failed_aggregate_endpoints": failed_aggregate,
        "failed_sign_endpoints": failed_sign,
        "fit_replication_advances_to_selection_review": bool(advances),
        "decision": (
            "OPEN_FRESH_SELECTION_REVIEW" if advances
            else "CLOSE_CURRENT_ANCHOR_LOCAL_BASIN"
        ),
        "patient_heldout_used_for_decision": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/analyze_topic4_rev12_orthogonal_anchor_relative.py",
        "scripts/analyze_topic4_rev12_orthogonal_free_field_screen.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("paired replication analysis paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"paired replication analysis path drifted: {relative}")
    artifact_root = args.artifact_root.resolve()
    input_audit = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"paired replication input changed: {record['path']}")
        input_audit[name] = {"path": str(path), "sha256": observed}
    aggregate_path = artifact_root / config["aggregate"]
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "REV12ND_CASCADE_FIT_AGGREGATE_COMPLETE":
        raise RuntimeError("paired replication aggregate is incomplete")
    rows = {row["candidate_id"]: row for row in aggregate["rows"]}
    result = analyze_replication(rows, decision=config["decision"])
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV12ND_PAIRED_FIELD_REPLICATION_ANALYSIS_COMPLETE",
        "scientific_role": config["scientific_role"],
        **result,
        "inputs": {
            **input_audit,
            "aggregate": {"path": str(aggregate_path),
                          "sha256": _sha256(aggregate_path)},
        },
        "provenance": {
            "git_commit": expected,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path), "dirty": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "decision": payload["decision"],
        "failed_aggregate_endpoints": payload["failed_aggregate_endpoints"],
        "failed_sign_endpoints": payload["failed_sign_endpoints"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
