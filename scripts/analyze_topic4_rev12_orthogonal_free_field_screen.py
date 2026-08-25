#!/usr/bin/env python3
"""Analyze cross-network responses of the Stage-U orthogonal Node screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
MODE_ID = re.compile(r"^stage_u_f(?P<mode>\d+)_(?P<sign>[mp])$")
NETWORK_ENDPOINTS = (
    "objective_utility", "patient_utility", "kmeans_utility",
    "ood_utility", "compound_utility", "direction_utility",
    "monotonicity_utility",
)
AGGREGATE_ENDPOINTS = NETWORK_ENDPOINTS + (
    "topology_reliability_utility", "topology_separation_utility",
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


def _base_utilities(*, patient: float, kmeans: float, ood: float,
                    compound: float, direction: float,
                    monotonicity: float) -> dict[str, float]:
    return {
        "objective_utility": (
            -float(patient) + 0.5 * float(kmeans) - 0.25 * float(ood)
            - 0.25 * float(compound) + 0.5 * float(direction)
        ),
        "patient_utility": -float(patient),
        "kmeans_utility": float(kmeans),
        "ood_utility": -float(ood),
        "compound_utility": -float(compound),
        "direction_utility": float(direction),
        "monotonicity_utility": float(monotonicity),
    }


def aggregate_utilities(row: dict) -> dict[str, float]:
    selection = row["selection_objective"]
    topology = row["source_topology"]
    reliability_values = [
        float(topology["mean_within_network_split_half_cosine"]),
        float(topology["mean_across_network_template_cosine"]),
    ]
    finite_reliability = [value for value in reliability_values if np.isfinite(value)]
    reliability = min(finite_reliability) if finite_reliability else 0.0
    output = _base_utilities(
        patient=selection["matched_patient_loss"],
        kmeans=1.0 - float(selection["kmeans_direction_loss"]),
        ood=selection["ood_fraction"], compound=selection["compound_fraction"],
        direction=selection["causal_direction_score"],
        monotonicity=row["causal_wave_monotonicity"]["score"],
    )
    output["objective_utility"] = -float(selection["objective"])
    output.update({
        "topology_reliability_utility": float(np.clip(reliability, 0.0, 1.0)),
        "topology_separation_utility": float(np.clip(
            topology["equal_network_between_mode_distance"], 0.0, 1.0,
        )),
    })
    return output


def per_network_utilities(row: dict) -> dict[int, dict[str, float]]:
    seeds = [int(record["seed"]) for record in row["per_seed"]]
    arrays = (
        row["matched_network_scores"],
        row["causal_direction_alignment"]["per_network"],
        row["causal_wave_monotonicity"]["per_network"],
    )
    if any(len(array) != len(seeds) for array in arrays):
        raise RuntimeError("Stage-U per-network arrays do not align")
    kmeans = {
        int(record["seed"]): float(record["direction_balanced_alignment"])
        for record in row["per_network_natural_kmeans"]["rows"]
    }
    output = {}
    for index, seed in enumerate(seeds):
        output[seed] = _base_utilities(
            patient=row["matched_network_scores"][index]["objective"],
            kmeans=kmeans[seed],
            ood=row["per_seed"][index]["ood_fraction"],
            compound=row["per_seed"][index]["compound_fraction"],
            direction=row["causal_direction_alignment"]["per_network"][index]["score"],
            monotonicity=row["causal_wave_monotonicity"]["per_network"][index]["score"],
        )
    return output


def stable_sign(values: list[float], *, minimum_same_sign: int) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    positive = int(np.sum(values > 0.0))
    negative = int(np.sum(values < 0.0))
    if positive >= int(minimum_same_sign) and positive > negative:
        sign = 1
    elif negative >= int(minimum_same_sign) and negative > positive:
        sign = -1
    else:
        sign = 0
    return {
        "n_evaluable": int(len(values)),
        "n_positive": positive, "n_negative": negative,
        "stable_sign": int(sign),
        "median_slope": float(np.median(values)) if len(values) else None,
        "mean_slope": float(np.mean(values)) if len(values) else None,
    }


def mode_response_rows(rows: dict[str, dict], manifest: dict, *,
                       minimum_same_sign: int) -> list[dict]:
    metadata = {
        row["candidate_id"]: {
            "mode_index": int(
                row["node_field"]["residual_coordinates"]["residual_index"]
            ),
            "kx": int(row["node_field"]["residual_coordinates"]["kx"]),
            "ky": int(row["node_field"]["residual_coordinates"]["ky"]),
            "signed_amplitude": float(
                row["node_field"]["residual_coordinates"]["signed_log_surface_rms"]
            ),
        }
        for row in manifest["candidates"]
        if MODE_ID.match(row["candidate_id"])
    }
    paired = {}
    for candidate_id, row in rows.items():
        match = MODE_ID.match(candidate_id)
        if match is not None:
            paired.setdefault(int(match.group("mode")), {})[
                match.group("sign")
            ] = row
    output = []
    for mode, pair in sorted(paired.items()):
        if set(pair) != {"m", "p"}:
            raise RuntimeError("Stage-U mode lacks one symmetric arm")
        minus_aggregate = aggregate_utilities(pair["m"])
        plus_aggregate = aggregate_utilities(pair["p"])
        minus_network = per_network_utilities(pair["m"])
        plus_network = per_network_utilities(pair["p"])
        if minus_network.keys() != plus_network.keys():
            raise RuntimeError("Stage-U symmetric arms have different networks")
        minus_id = f"stage_u_f{mode:02d}_m"
        plus_id = f"stage_u_f{mode:02d}_p"
        if minus_id not in metadata or plus_id not in metadata:
            raise RuntimeError("Stage-U manifest lacks symmetric mode metadata")
        minus_meta, plus_meta = metadata[minus_id], metadata[plus_id]
        if ((minus_meta["kx"], minus_meta["ky"])
                != (plus_meta["kx"], plus_meta["ky"])):
            raise RuntimeError("Stage-U symmetric arms have different basis modes")
        minus_amplitude = float(minus_meta["signed_amplitude"])
        plus_amplitude = float(plus_meta["signed_amplitude"])
        if (minus_amplitude >= 0.0 or plus_amplitude <= 0.0
                or not np.isclose(-minus_amplitude, plus_amplitude,
                                  rtol=0.0, atol=1e-12)):
            raise RuntimeError("Stage-U manifest amplitudes are not symmetric")
        amplitude = plus_amplitude
        aggregate_slopes = {
            endpoint: float(
                (plus_aggregate[endpoint] - minus_aggregate[endpoint])
                / (2.0 * amplitude)
            )
            for endpoint in AGGREGATE_ENDPOINTS
        }
        network_slopes = {
            endpoint: {
                str(seed): float(
                    (plus_network[seed][endpoint] - minus_network[seed][endpoint])
                    / (2.0 * amplitude)
                )
                for seed in sorted(minus_network)
            }
            for endpoint in NETWORK_ENDPOINTS
        }
        output.append({
            "mode_index": int(mode),
            "kx": plus_meta["kx"], "ky": plus_meta["ky"],
            "symmetric_amplitude": float(amplitude),
            "aggregate_slopes": aggregate_slopes,
            "network_slopes": network_slopes,
            "network_sign_support": {
                endpoint: stable_sign(
                    list(seed_values.values()),
                    minimum_same_sign=minimum_same_sign,
                )
                for endpoint, seed_values in network_slopes.items()
            },
        })
    return output


def _robust_scales(mode_rows: list[dict]) -> dict[str, float]:
    output = {}
    for endpoint in AGGREGATE_ENDPOINTS:
        values = np.asarray([
            row["aggregate_slopes"][endpoint] for row in mode_rows
        ], float)
        scale = float(np.median(np.abs(values - np.median(values))))
        output[endpoint] = max(scale, 1e-6)
    return output


def select_outer_followup(mode_rows: list[dict], *, screen: dict) -> dict:
    scales = _robust_scales(mode_rows)
    orientations = []
    for row in mode_rows:
        for sign in (-1, 1):
            stable = {
                endpoint: bool(
                    row["network_sign_support"][endpoint]["stable_sign"] == sign
                )
                for endpoint in NETWORK_ENDPOINTS
            }
            normalized = {
                endpoint: float(sign * row["aggregate_slopes"][endpoint] / scales[endpoint])
                for endpoint in AGGREGATE_ENDPOINTS
            }
            orientations.append({
                "mode_index": row["mode_index"], "kx": row["kx"], "ky": row["ky"],
                "orientation": int(sign),
                "stable_improvement": stable,
                "normalized_effects": normalized,
                "balanced_score": float(sum(
                    float(weight) * normalized[endpoint]
                    for endpoint, weight in screen["balanced_weights"].items()
                )),
            })
    chosen, reasons = [], {}
    champion_endpoints = list(screen["primary_endpoints"])
    for endpoint in champion_endpoints:
        eligible = [row for row in orientations if row["stable_improvement"][endpoint]]
        if eligible:
            winner = max(eligible, key=lambda row: (
                row["normalized_effects"][endpoint], row["balanced_score"],
                -row["mode_index"], row["orientation"],
            ))
            key = (winner["mode_index"], winner["orientation"])
            if key not in chosen:
                chosen.append(key)
            reasons.setdefault(key, []).append(f"champion:{endpoint}")
    balanced = [
        row for row in orientations
        if any(row["stable_improvement"][endpoint]
               for endpoint in screen["primary_endpoints"])
    ]
    for winner in sorted(
            balanced, key=lambda row: (-row["balanced_score"], row["mode_index"])):
        key = (winner["mode_index"], winner["orientation"])
        if key not in chosen:
            chosen.append(key)
            reasons.setdefault(key, []).append("balanced_response")
        if len(chosen) >= int(screen["maximum_outer_followup_modes"]):
            break
    selected = []
    by_key = {(row["mode_index"], row["orientation"]): row for row in orientations}
    for key in chosen[:int(screen["maximum_outer_followup_modes"])]:
        selected.append({**by_key[key], "selection_reasons": reasons[key]})
    return {
        "robust_endpoint_scales": scales,
        "n_oriented_modes": len(orientations),
        "selected_for_outer_amplitude": selected,
        "outer_amplitude": float(screen["outer_amplitude"]),
        "selection_role": "fit-only scale validation, not Node field selection",
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
    tracked = [str(config_path.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT))]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("Stage-U analysis runtime paths are dirty")
    for relative in tracked:
        expected = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(expected).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(
                f"Stage-U analysis path differs from expected commit: {relative}"
            )
    artifact_root = args.artifact_root.resolve()
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Stage-U analysis input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    aggregate_path = _resolve(artifact_root, config["aggregate"])
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "REV12ND_CASCADE_FIT_AGGREGATE_COMPLETE":
        raise RuntimeError("Stage-U aggregate is incomplete")
    rows = {row["candidate_id"]: row for row in aggregate["rows"]}
    expected_ids = {row["candidate_id"] for row in loaded["candidate_manifest"]["candidates"]}
    if rows.keys() != expected_ids:
        raise RuntimeError("Stage-U aggregate candidate set drifted")
    mode_rows = mode_response_rows(
        rows, loaded["candidate_manifest"],
        minimum_same_sign=int(config["screen"]["minimum_same_sign_networks"]),
    )
    followup = select_outer_followup(mode_rows, screen=config["screen"])
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV12ND_ORTHOGONAL_MODE_RESPONSE_ANALYSIS_COMPLETE",
        "scientific_role": config["scientific_role"],
        "anchor": {
            "candidate_id": "stage_u_anchor",
            "aggregate_utilities": aggregate_utilities(rows["stage_u_anchor"]),
            "per_network_utilities": per_network_utilities(rows["stage_u_anchor"]),
        },
        "mode_responses": mode_rows,
        "outer_followup": followup,
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
        "status": payload["status"],
        "n_modes": len(mode_rows),
        "n_outer_followup": len(followup["selected_for_outer_amplitude"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
