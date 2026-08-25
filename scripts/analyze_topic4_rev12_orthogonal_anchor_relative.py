#!/usr/bin/env python3
"""Audit Stage-U against its anchor and propose bounded local canaries."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
MODE_ID = re.compile(r"^stage_u_f(?P<mode>\d+)_(?P<sign>[mp])$")

from scripts.analyze_topic4_rev12_orthogonal_free_field_screen import (  # noqa: E402
    AGGREGATE_ENDPOINTS,
    NETWORK_ENDPOINTS,
    aggregate_utilities,
    per_network_utilities,
    stable_sign,
)


MODE_ENDPOINTS = ("patient_mode_0_utility", "patient_mode_1_utility")
ALL_ENDPOINTS = AGGREGATE_ENDPOINTS + MODE_ENDPOINTS


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


def patient_mode_utilities(row: dict) -> dict[int, dict[str, float]]:
    seeds = [int(record["seed"]) for record in row["per_seed"]]
    scores = row["matched_network_scores"]
    if len(seeds) != len(scores):
        raise RuntimeError("patient mode scores do not align with network seeds")
    return {
        seed: {
            "patient_mode_0_utility": -float(scores[index]["modes"]["0"]["mean"]),
            "patient_mode_1_utility": -float(scores[index]["modes"]["1"]["mean"]),
        }
        for index, seed in enumerate(seeds)
    }


def all_network_utilities(row: dict) -> dict[int, dict[str, float]]:
    base = per_network_utilities(row)
    modes = patient_mode_utilities(row)
    if base.keys() != modes.keys():
        raise RuntimeError("network utility families do not align")
    return {seed: {**base[seed], **modes[seed]} for seed in base}


def all_aggregate_utilities(row: dict) -> dict[str, float]:
    output = aggregate_utilities(row)
    network = patient_mode_utilities(row)
    for endpoint in MODE_ENDPOINTS:
        output[endpoint] = float(np.mean([
            values[endpoint] for values in network.values()
        ]))
    return output


def _difference(candidate: dict[str, float], anchor: dict[str, float]) -> dict:
    return {
        endpoint: float(candidate[endpoint] - anchor[endpoint])
        for endpoint in ALL_ENDPOINTS
    }


def anchor_relative_rows(rows: dict[str, dict], *, minimum_same_sign: int) -> list[dict]:
    anchor_aggregate = all_aggregate_utilities(rows["stage_u_anchor"])
    anchor_network = all_network_utilities(rows["stage_u_anchor"])
    output = []
    for candidate_id, row in sorted(rows.items()):
        match = MODE_ID.match(candidate_id)
        if match is None:
            continue
        aggregate_delta = _difference(
            all_aggregate_utilities(row), anchor_aggregate,
        )
        candidate_network = all_network_utilities(row)
        if candidate_network.keys() != anchor_network.keys():
            raise RuntimeError("candidate and anchor network pools differ")
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
        output.append({
            "candidate_id": candidate_id,
            "mode_index": int(match.group("mode")),
            "orientation": 1 if match.group("sign") == "p" else -1,
            "aggregate_delta_from_anchor": aggregate_delta,
            "network_delta_from_anchor": network_delta,
            "network_improvement_support": {
                endpoint: stable_sign(
                    list(values.values()),
                    minimum_same_sign=minimum_same_sign,
                )
                for endpoint, values in network_delta.items()
            },
            "both_patient_modes_improve_aggregate": bool(
                aggregate_delta["patient_mode_0_utility"] > 0.0
                and aggregate_delta["patient_mode_1_utility"] > 0.0
            ),
        })
    return output


def diagonal_quadratic(rows: dict[str, dict]) -> list[dict]:
    anchor_aggregate = all_aggregate_utilities(rows["stage_u_anchor"])
    anchor_network = all_network_utilities(rows["stage_u_anchor"])
    output = []
    for mode in sorted({
        int(match.group("mode"))
        for candidate_id in rows
        if (match := MODE_ID.match(candidate_id)) is not None
    }):
        minus = rows[f"stage_u_f{mode:02d}_m"]
        plus = rows[f"stage_u_f{mode:02d}_p"]
        minus_aggregate = all_aggregate_utilities(minus)
        plus_aggregate = all_aggregate_utilities(plus)
        minus_network = all_network_utilities(minus)
        plus_network = all_network_utilities(plus)
        amplitude = 0.08
        aggregate = {}
        for endpoint in ALL_ENDPOINTS:
            y0 = anchor_aggregate[endpoint]
            ym, yp = minus_aggregate[endpoint], plus_aggregate[endpoint]
            aggregate[endpoint] = {
                "linear": float((yp - ym) / (2.0 * amplitude)),
                "quadratic": float(
                    (yp + ym - 2.0 * y0) / (2.0 * amplitude ** 2)
                ),
            }
        network = {}
        for endpoint in NETWORK_ENDPOINTS + MODE_ENDPOINTS:
            network[endpoint] = {}
            for seed in sorted(anchor_network):
                y0 = anchor_network[seed][endpoint]
                ym, yp = minus_network[seed][endpoint], plus_network[seed][endpoint]
                network[endpoint][str(seed)] = {
                    "linear": float((yp - ym) / (2.0 * amplitude)),
                    "quadratic": float(
                        (yp + ym - 2.0 * y0) / (2.0 * amplitude ** 2)
                    ),
                }
        output.append({
            "mode_index": mode,
            "symmetric_observation_amplitude": amplitude,
            "aggregate": aggregate,
            "network": network,
        })
    return output


def _predicted_delta(coefficients: list[dict], composition: tuple[tuple[int, float], ...]) -> dict:
    by_mode = {row["mode_index"]: row for row in coefficients}
    aggregate = {endpoint: 0.0 for endpoint in ALL_ENDPOINTS}
    seeds = sorted(next(iter(by_mode.values()))["network"]["objective_utility"])
    network = {
        endpoint: {seed: 0.0 for seed in seeds}
        for endpoint in NETWORK_ENDPOINTS + MODE_ENDPOINTS
    }
    for mode, amplitude in composition:
        row = by_mode[mode]
        for endpoint in ALL_ENDPOINTS:
            fit = row["aggregate"][endpoint]
            aggregate[endpoint] += (
                fit["linear"] * amplitude + fit["quadratic"] * amplitude ** 2
            )
        for endpoint in NETWORK_ENDPOINTS + MODE_ENDPOINTS:
            for seed in seeds:
                fit = row["network"][endpoint][seed]
                network[endpoint][seed] += (
                    fit["linear"] * amplitude + fit["quadratic"] * amplitude ** 2
                )
    return {"aggregate": aggregate, "network": network}


def local_proposals(coefficients: list[dict], *, screen: dict) -> dict:
    amplitudes = tuple(float(value) for value in screen["amplitude_grid"])
    if any(value == 0.0 for value in amplitudes):
        raise RuntimeError("local amplitude grid must omit zero")
    modes = tuple(row["mode_index"] for row in coefficients)
    proposals = []
    for n_nonzero in range(1, int(screen["maximum_nonzero_modes"]) + 1):
        for selected_modes in itertools.combinations(modes, n_nonzero):
            for selected_amplitudes in itertools.product(amplitudes, repeat=n_nonzero):
                if math.sqrt(sum(value ** 2 for value in selected_amplitudes)) > (
                        float(screen["maximum_l2_amplitude"]) + 1e-12):
                    continue
                composition = tuple(zip(selected_modes, selected_amplitudes))
                predicted = _predicted_delta(coefficients, composition)
                aggregate = predicted["aggregate"]
                required = tuple(screen["required_positive_aggregate_endpoints"])
                if any(aggregate[endpoint] <= 0.0 for endpoint in required):
                    continue
                support = {}
                for endpoint in screen["required_network_support_endpoints"]:
                    support[endpoint] = stable_sign(
                        list(predicted["network"][endpoint].values()),
                        minimum_same_sign=int(screen["minimum_same_sign_networks"]),
                    )
                if any(record["stable_sign"] != 1 for record in support.values()):
                    continue
                if aggregate["monotonicity_utility"] < float(
                        screen["minimum_monotonicity_delta"]):
                    continue
                protected = min(aggregate[endpoint] for endpoint in required)
                network_objective = np.asarray(list(
                    predicted["network"]["objective_utility"].values()
                ), float)
                score = (
                    aggregate["objective_utility"]
                    + float(screen["protected_endpoint_weight"]) * protected
                    - float(screen["network_dispersion_weight"])
                    * float(np.std(network_objective))
                )
                proposals.append({
                    "composition": [
                        {"mode_index": mode, "amplitude": amplitude}
                        for mode, amplitude in composition
                    ],
                    "l2_amplitude": float(math.sqrt(sum(
                        value ** 2 for value in selected_amplitudes
                    ))),
                    "predicted_delta": predicted,
                    "network_improvement_support": support,
                    "surrogate_rank_score": float(score),
                    "scientific_role": (
                        "diagonal-quadratic fit-only canary; interactions unobserved"
                    ),
                })
    proposals.sort(key=lambda row: (
        -row["surrogate_rank_score"],
        len(row["composition"]),
        tuple(item["mode_index"] for item in row["composition"]),
    ))
    selected = []
    seen_support = set()
    for row in proposals:
        support_key = tuple(item["mode_index"] for item in row["composition"])
        if support_key in seen_support:
            continue
        selected.append(row)
        seen_support.add(support_key)
        if len(selected) >= int(screen["maximum_proposals"]):
            break
    return {
        "n_feasible_grid_proposals": len(proposals),
        "selected": selected,
        "assumption": (
            "Axis-wise quadratic effects add locally; cross-mode interactions are "
            "unknown and require direct SNN validation."
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
        raise RuntimeError("anchor-relative analysis paths are dirty")
    for relative in tracked:
        expected = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(expected).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"anchor-relative analysis path drifted: {relative}")

    artifact_root = args.artifact_root.resolve()
    loaded, input_audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"anchor-relative input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    rows = {row["candidate_id"]: row for row in loaded["aggregate"]["rows"]}
    expected_ids = {
        row["candidate_id"] for row in loaded["candidate_manifest"]["candidates"]
    }
    if rows.keys() != expected_ids or "stage_u_anchor" not in rows:
        raise RuntimeError("Stage-U candidate set drifted")
    relative = anchor_relative_rows(
        rows, minimum_same_sign=int(config["screen"]["minimum_same_sign_networks"]),
    )
    objective_improvers = [
        row["candidate_id"] for row in relative
        if row["aggregate_delta_from_anchor"]["objective_utility"] > 0.0
        and row["network_improvement_support"]["objective_utility"]["stable_sign"] == 1
    ]
    coefficients = diagonal_quadratic(rows)
    proposals = local_proposals(coefficients, screen=config["screen"])
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV12ND_ANCHOR_RELATIVE_LOCAL_AUDIT_COMPLETE",
        "scientific_role": config["scientific_role"],
        "anchor": {
            "candidate_id": "stage_u_anchor",
            "aggregate_utilities": all_aggregate_utilities(rows["stage_u_anchor"]),
            "per_network_utilities": all_network_utilities(rows["stage_u_anchor"]),
        },
        "anchor_relative_candidates": relative,
        "outer_amplitude_decision": {
            "status": (
                "NOT_OPENED_NO_ANCHOR_OBJECTIVE_IMPROVEMENT"
                if not objective_improvers else "REQUIRES_SEPARATE_REVIEW"
            ),
            "anchor_objective_improvers": objective_improvers,
            "reason": (
                "A symmetric +/- slope is not evidence that either endpoint "
                "improves on the current anchor."
            ),
        },
        "diagonal_local_quadratic": coefficients,
        "local_canary_proposals": proposals,
        "inputs": input_audit,
        "provenance": {
            "git_commit": commit,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
            "dirty": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "outer_amplitude_status": payload["outer_amplitude_decision"]["status"],
        "n_feasible_local_proposals": proposals["n_feasible_grid_proposals"],
        "n_selected_local_proposals": len(proposals["selected"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
