#!/usr/bin/env python3
"""Compare directly simulated Stage-W local canaries with their anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


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
    stable_sign,
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


def analyze_direct_results(rows: dict[str, dict], *, minimum_same_sign: int,
                           required_aggregate: list[str],
                           required_network: list[str]) -> dict:
    if "stage_w_anchor" not in rows:
        raise RuntimeError("Stage-W anchor is absent")
    anchor_aggregate = all_aggregate_utilities(rows["stage_w_anchor"])
    anchor_network = all_network_utilities(rows["stage_w_anchor"])
    output = []
    for candidate_id, row in sorted(rows.items()):
        if candidate_id == "stage_w_anchor":
            continue
        candidate_aggregate = all_aggregate_utilities(row)
        candidate_network = all_network_utilities(row)
        if candidate_network.keys() != anchor_network.keys():
            raise RuntimeError("Stage-W candidate and anchor network pools differ")
        aggregate_delta = {
            endpoint: float(candidate_aggregate[endpoint] - anchor_aggregate[endpoint])
            for endpoint in anchor_aggregate
        }
        network_delta = {
            endpoint: {
                str(seed): float(
                    candidate_network[seed][endpoint]
                    - anchor_network[seed][endpoint]
                )
                for seed in sorted(anchor_network)
            }
            for endpoint in NETWORK_ENDPOINTS + MODE_ENDPOINTS
        }
        support = {
            endpoint: stable_sign(
                list(network_delta[endpoint].values()),
                minimum_same_sign=minimum_same_sign,
            )
            for endpoint in required_network
        }
        aggregate_ok = all(
            aggregate_delta[endpoint] > 0.0 for endpoint in required_aggregate
        )
        network_ok = all(record["stable_sign"] == 1 for record in support.values())
        output.append({
            "candidate_id": candidate_id,
            "aggregate_delta_from_anchor": aggregate_delta,
            "network_delta_from_anchor": network_delta,
            "network_improvement_support": support,
            "fit_advancement_eligible": bool(aggregate_ok and network_ok),
            "failed_aggregate_endpoints": [
                endpoint for endpoint in required_aggregate
                if aggregate_delta[endpoint] <= 0.0
            ],
            "failed_network_support_endpoints": [
                endpoint for endpoint, record in support.items()
                if record["stable_sign"] != 1
            ],
        })
    output.sort(key=lambda row: (
        not row["fit_advancement_eligible"],
        -row["aggregate_delta_from_anchor"]["objective_utility"],
        row["candidate_id"],
    ))
    eligible = [row["candidate_id"] for row in output
                if row["fit_advancement_eligible"]]
    return {
        "anchor": {
            "candidate_id": "stage_w_anchor",
            "aggregate_utilities": anchor_aggregate,
            "per_network_utilities": anchor_network,
        },
        "candidate_results": output,
        "fit_advancement_candidates": eligible,
        "decision": (
            "OPEN_FRESH_SELECTION_REVIEW" if eligible
            else "STOP_NO_LOCAL_CANARY_JOINTLY_IMPROVED"
        ),
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
        raise RuntimeError("Stage-W result analysis paths are dirty")
    for relative in tracked:
        expected_content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(expected_content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"Stage-W result analysis path drifted: {relative}")
    artifact_root = args.artifact_root.resolve()
    input_audit = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Stage-W result input changed: {record['path']}")
        input_audit[name] = {"path": str(path), "sha256": observed}
    aggregate_path = artifact_root / config["aggregate"]
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "REV12ND_CASCADE_FIT_AGGREGATE_COMPLETE":
        raise RuntimeError("Stage-W aggregate is incomplete")
    rows = {row["candidate_id"]: row for row in aggregate["rows"]}
    result = analyze_direct_results(
        rows,
        minimum_same_sign=int(config["decision"]["minimum_same_sign_networks"]),
        required_aggregate=list(config["decision"]["required_aggregate_endpoints"]),
        required_network=list(config["decision"]["required_network_endpoints"]),
    )
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV12ND_LOCAL_CURVATURE_DIRECT_ANALYSIS_COMPLETE",
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
        "fit_advancement_candidates": payload["fit_advancement_candidates"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
