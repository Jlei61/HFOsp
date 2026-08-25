#!/usr/bin/env python3
"""Check whether Stage-U mode responses persist at the frozen outer amplitude."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from scripts.analyze_topic4_rev12_orthogonal_free_field_screen import (
    AGGREGATE_ENDPOINTS,
    NETWORK_ENDPOINTS,
    aggregate_utilities,
    per_network_utilities,
    stable_sign,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


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


def outer_response_rows(stage_u_analysis: dict, stage_v_manifest: dict,
                        stage_v_rows: dict[str, dict], *,
                        minimum_same_sign: int) -> list[dict]:
    """Compare each selected outer candidate with the paired anchor."""
    if "stage_v_anchor" not in stage_v_rows:
        raise RuntimeError("Stage-V aggregate lacks its anchor")
    anchor_aggregate = aggregate_utilities(stage_v_rows["stage_v_anchor"])
    anchor_network = per_network_utilities(stage_v_rows["stage_v_anchor"])
    inner = {
        int(row["mode_index"]): row for row in stage_u_analysis["mode_responses"]
    }
    selected = {
        (int(row["mode_index"]), int(row["orientation"])): row
        for row in stage_u_analysis["outer_followup"][
            "selected_for_outer_amplitude"
        ]
    }
    audit = {
        row["candidate_id"]: row
        for row in stage_v_manifest["outer_amplitude_audit"]
    }
    output = []
    for candidate_id, record in sorted(audit.items()):
        if candidate_id not in stage_v_rows:
            raise RuntimeError("Stage-V candidate lacks a complete aggregate row")
        mode = int(record["mode_index"])
        orientation = int(record["orientation"])
        key = (mode, orientation)
        if key not in selected or mode not in inner:
            raise RuntimeError("Stage-V candidate was not selected by Stage-U")
        candidate_aggregate = aggregate_utilities(stage_v_rows[candidate_id])
        candidate_network = per_network_utilities(stage_v_rows[candidate_id])
        if candidate_network.keys() != anchor_network.keys():
            raise RuntimeError("Stage-V candidate and anchor networks differ")
        amplitude = abs(float(stage_u_analysis["outer_followup"]["outer_amplitude"]))
        expected = {
            endpoint: float(
                orientation * amplitude
                * inner[mode]["aggregate_slopes"][endpoint]
            )
            for endpoint in AGGREGATE_ENDPOINTS
        }
        observed = {
            endpoint: float(candidate_aggregate[endpoint] - anchor_aggregate[endpoint])
            for endpoint in AGGREGATE_ENDPOINTS
        }
        network_deltas = {
            endpoint: {
                str(seed): float(
                    candidate_network[seed][endpoint]
                    - anchor_network[seed][endpoint]
                )
                for seed in sorted(anchor_network)
            }
            for endpoint in NETWORK_ENDPOINTS
        }
        network_support = {
            endpoint: stable_sign(
                list(values.values()), minimum_same_sign=minimum_same_sign,
            )
            for endpoint, values in network_deltas.items()
        }
        endpoint_checks = {}
        for endpoint in AGGREGATE_ENDPOINTS:
            predicted = expected[endpoint]
            actual = observed[endpoint]
            support = network_support.get(endpoint)
            expected_sign = int(np.sign(predicted))
            endpoint_checks[endpoint] = {
                "expected_delta": predicted,
                "observed_delta": actual,
                "expected_sign": expected_sign,
                "aggregate_sign_preserved": bool(
                    expected_sign != 0 and int(np.sign(actual)) == expected_sign
                ),
                "network_sign_preserved": (
                    None if support is None else bool(
                        expected_sign != 0
                        and support["stable_sign"] == expected_sign
                    )
                ),
                "outer_to_linear_ratio": (
                    None if abs(predicted) <= 1e-12
                    else float(actual / predicted)
                ),
            }
        retained = [
            endpoint for endpoint, was_stable in selected[key][
                "stable_improvement"
            ].items()
            if was_stable
            and endpoint_checks[endpoint]["aggregate_sign_preserved"]
            and endpoint_checks[endpoint]["network_sign_preserved"]
        ]
        output.append({
            "candidate_id": candidate_id,
            "mode_index": mode, "kx": int(record["kx"]),
            "ky": int(record["ky"]), "orientation": orientation,
            "selection_reasons": list(record["selection_reasons"]),
            "endpoint_checks": endpoint_checks,
            "network_deltas": network_deltas,
            "network_sign_support": network_support,
            "retained_improvement_endpoints": retained,
            "response_status": (
                "OUTER_RESPONSE_RETAINED" if retained
                else "OUTER_RESPONSE_NOT_RETAINED"
            ),
        })
    if len(output) != len(selected):
        raise RuntimeError("Stage-V manifest does not cover every selected orientation")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/analyze_topic4_rev12_orthogonal_free_field_screen.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("outer-amplitude analysis paths are dirty")
    for relative in tracked:
        expected = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(expected).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"outer-amplitude analysis path drifted: {relative}")
    artifact_root = args.artifact_root.resolve()
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"outer-amplitude analysis input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    aggregate_path = _resolve(artifact_root, config["aggregate"])
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "REV12ND_CASCADE_FIT_AGGREGATE_COMPLETE":
        raise RuntimeError("Stage-V aggregate is incomplete")
    rows = {row["candidate_id"]: row for row in aggregate["rows"]}
    response = outer_response_rows(
        loaded["stage_u_analysis"], loaded["stage_v_manifest"], rows,
        minimum_same_sign=int(config["minimum_same_sign_networks"]),
    )
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV12ND_ORTHOGONAL_OUTER_AMPLITUDE_ANALYSIS_COMPLETE",
        "scientific_role": config["scientific_role"],
        "responses": response,
        "n_retained": int(sum(
            row["response_status"] == "OUTER_RESPONSE_RETAINED"
            for row in response
        )),
        "inputs": {
            **inputs,
            "aggregate": {"path": str(aggregate_path), "sha256": _sha256(aggregate_path)},
        },
        "provenance": {
            "git_commit": commit, "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path), "dirty": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_retained": payload["n_retained"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
