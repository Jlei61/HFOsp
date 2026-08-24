#!/usr/bin/env python3
"""Audit whether the frozen four-direction Node search has usable gradients."""
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
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from src.topic4_continuous_field import tensor_basis  # noqa: E402
from src.topic4_node_field_search import uniform_sheet_grid  # noqa: E402


PERTURBATION = re.compile(
    r"^stage_i_a(?P<anchor>\d+)_d(?P<direction>\d+)_s(?P<scale>\d+)_(?P<sign>[mp])$"
)
ENDPOINTS = (
    "objective_utility", "patient_utility", "kmeans_utility",
    "ood_utility", "compound_utility", "direction_utility",
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


def aggregate_utilities(row: dict) -> dict[str, float]:
    score = row["selection_objective"]
    return {
        "objective_utility": -float(score["objective"]),
        "patient_utility": -float(score["matched_patient_loss"]),
        "kmeans_utility": 1.0 - float(score["kmeans_direction_loss"]),
        "ood_utility": -float(score["ood_fraction"]),
        "compound_utility": -float(score["compound_fraction"]),
        "direction_utility": float(score["causal_direction_score"]),
    }


def per_network_utilities(row: dict) -> dict[int, dict[str, float]]:
    seeds = [int(record["seed"]) for record in row["per_seed"]]
    if not (len(seeds) == len(row["matched_network_scores"])
            == len(row["causal_direction_alignment"]["per_network"])):
        raise RuntimeError("per-network endpoint arrays do not align")
    kmeans = {
        int(record["seed"]): float(record["direction_balanced_alignment"])
        for record in row["per_network_natural_kmeans"]["rows"]
    }
    output = {}
    for index, seed in enumerate(seeds):
        patient = -float(row["matched_network_scores"][index]["objective"])
        km = kmeans[seed]
        ood = -float(row["per_seed"][index]["ood_fraction"])
        compound = -float(row["per_seed"][index]["compound_fraction"])
        direction = float(
            row["causal_direction_alignment"]["per_network"][index]["score"]
        )
        output[seed] = {
            "objective_utility": (
                patient + 0.5 * km + 0.25 * ood
                + 0.25 * compound + 0.5 * direction
            ),
            "patient_utility": patient,
            "kmeans_utility": km,
            "ood_utility": ood,
            "compound_utility": compound,
            "direction_utility": direction,
        }
    return output


def agreement_summary(left: np.ndarray, right: np.ndarray, *,
                      zero_tolerance: float = 1e-10) -> dict:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = left[valid], right[valid]
    directional = (np.abs(x) > zero_tolerance) & (np.abs(y) > zero_tolerance)
    sign_count = int(np.sum(np.sign(x[directional]) == np.sign(y[directional])))
    sign_total = int(np.sum(directional))
    if len(x) >= 2 and np.ptp(x) > 0.0 and np.ptp(y) > 0.0:
        rho = float(spearmanr(x, y).statistic)
    else:
        rho = None
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    cosine = float(np.dot(x, y) / denominator) if denominator > 0.0 else None
    return {
        "n_total": int(len(x)),
        "n_directional": sign_total,
        "sign_agreement_count": sign_count,
        "sign_agreement_fraction": (
            float(sign_count / sign_total) if sign_total else None
        ),
        "sign_agreement_binomial_p_greater_than_half": (
            float(binomtest(sign_count, sign_total, 0.5, alternative="greater").pvalue)
            if sign_total else None
        ),
        "spearman": rho,
        "cosine": cosine,
    }


def _perturbation_records(manifest: dict) -> list[dict]:
    output = []
    for candidate in manifest["candidates"]:
        match = PERTURBATION.match(candidate["candidate_id"])
        if match is None:
            continue
        coordinates = candidate["node_field"]["residual_coordinates"]
        amplitude = abs(float(coordinates["signed_log_surface_rms"]))
        output.append({
            "candidate_id": candidate["candidate_id"],
            "anchor": int(match.group("anchor")),
            "direction": int(match.group("direction")),
            "scale": int(match.group("scale")),
            "sign": match.group("sign"),
            "amplitude": amplitude,
        })
    return output


def symmetric_slope_records(manifest: dict, rows: dict[str, dict], *,
                            network_seed: int | None = None) -> list[dict]:
    records = _perturbation_records(manifest)
    grouped = {}
    for record in records:
        key = (record["anchor"], record["direction"], record["scale"])
        grouped.setdefault(key, {})[record["sign"]] = record
    output = []
    for (anchor, direction, scale), pair in sorted(grouped.items()):
        if set(pair) != {"m", "p"}:
            raise RuntimeError("symmetric field perturbation is incomplete")
        minus_row = rows[pair["m"]["candidate_id"]]
        plus_row = rows[pair["p"]["candidate_id"]]
        if network_seed is None:
            minus = aggregate_utilities(minus_row)
            plus = aggregate_utilities(plus_row)
        else:
            minus = per_network_utilities(minus_row)[int(network_seed)]
            plus = per_network_utilities(plus_row)[int(network_seed)]
        amplitude = float(pair["p"]["amplitude"])
        if not np.isclose(amplitude, pair["m"]["amplitude"]):
            raise RuntimeError("positive and negative amplitudes differ")
        output.append({
            "anchor": anchor, "direction": direction, "scale": scale,
            "amplitude": amplitude,
            "slopes": {
                endpoint: float((plus[endpoint] - minus[endpoint]) / (2.0 * amplitude))
                for endpoint in ENDPOINTS
            },
        })
    return output


def scale_predictivity(records: list[dict]) -> dict:
    grouped = {}
    for record in records:
        grouped.setdefault((record["anchor"], record["direction"]), []).append(record)
    output = {endpoint: {"inner": [], "outer": []} for endpoint in ENDPOINTS}
    details = []
    for key, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["amplitude"])
        if len(values) != 2 or not values[0]["amplitude"] < values[1]["amplitude"]:
            raise RuntimeError("scale predictivity requires exactly two amplitudes")
        details.append({
            "anchor": key[0], "direction": key[1],
            "inner_amplitude": values[0]["amplitude"],
            "outer_amplitude": values[1]["amplitude"],
        })
        for endpoint in ENDPOINTS:
            output[endpoint]["inner"].append(values[0]["slopes"][endpoint])
            output[endpoint]["outer"].append(values[1]["slopes"][endpoint])
    return {
        "n_anchor_direction_pairs": len(grouped),
        "endpoints": {
            endpoint: {
                **agreement_summary(values["inner"], values["outer"]),
                "inner_slopes": values["inner"],
                "outer_slopes": values["outer"],
            }
            for endpoint, values in output.items()
        },
        "pairs": details,
    }


def seed_predictivity(left: list[dict], right: list[dict], *,
                      left_seed: int, right_seed: int) -> dict:
    def index(records):
        return {
            (row["anchor"], row["direction"], row["scale"]): row
            for row in records
        }
    left_by, right_by = index(left), index(right)
    if left_by.keys() != right_by.keys():
        raise RuntimeError("network slope records do not align")
    return {
        "left_seed": int(left_seed), "right_seed": int(right_seed),
        "n_paired_slopes": len(left_by),
        "endpoints": {
            endpoint: agreement_summary(
                [left_by[key]["slopes"][endpoint] for key in sorted(left_by)],
                [right_by[key]["slopes"][endpoint] for key in sorted(right_by)],
            )
            for endpoint in ENDPOINTS
        },
    }


def _surface_basis(manifest: dict, *, grid_per_axis: int) -> np.ndarray:
    first = manifest["candidates"][0]["node_field"]
    grid = uniform_sheet_grid(int(grid_per_axis), sheet_mm=20.0)
    return tensor_basis(
        grid, int(first["n_basis"]), degree=int(first["degree"]), L=20.0,
    )


def _field_surfaces(manifest: dict, basis: np.ndarray) -> dict[str, np.ndarray]:
    return {
        candidate["candidate_id"]: basis @ np.asarray(
            candidate["node_field"]["coefficients"], float,
        ).ravel()
        for candidate in manifest["candidates"]
    }


def project_coordinates(target: np.ndarray, basis_vectors: np.ndarray) -> dict:
    target = np.asarray(target, float)
    vectors = np.asarray(basis_vectors, float)
    if target.ndim != 1 or vectors.ndim != 2 or vectors.shape[0] != len(target):
        raise ValueError("target and basis vectors do not align")
    coordinates, *_ = np.linalg.lstsq(vectors, target, rcond=None)
    fitted = vectors @ coordinates
    denominator = float(np.linalg.norm(target))
    return {
        "coordinates": coordinates,
        "relative_residual_norm": (
            float(np.linalg.norm(target - fitted) / denominator)
            if denominator > 0.0 else 0.0
        ),
        "projected_norm_fraction": (
            float(np.linalg.norm(fitted) / denominator) if denominator > 0.0 else 0.0
        ),
        "projected_energy_fraction": (
            float(np.dot(fitted, fitted) / np.dot(target, target))
            if denominator > 0.0 else 0.0
        ),
    }


def response_surface_validation(stage_s_manifest: dict, stage_s_rows: dict[str, dict],
                                stage_t_manifest: dict, stage_t_rows: dict[str, dict],
                                *, grid_per_axis: int) -> tuple[dict, dict]:
    basis = _surface_basis(stage_s_manifest, grid_per_axis=grid_per_axis)
    s_surfaces = _field_surfaces(stage_s_manifest, basis)
    t_surfaces = _field_surfaces(stage_t_manifest, basis)
    slope_records = symmetric_slope_records(stage_s_manifest, stage_s_rows)
    slopes = {}
    for record in slope_records:
        key = (record["anchor"], record["direction"])
        slopes.setdefault(key, {endpoint: [] for endpoint in ENDPOINTS})
        for endpoint in ENDPOINTS:
            slopes[key][endpoint].append(record["slopes"][endpoint])
    anchor_ids = sorted(
        candidate["candidate_id"] for candidate in stage_s_manifest["candidates"]
        if candidate["candidate_id"].startswith("stage_i_anchor_")
    )
    residuals = {}
    for anchor_index, anchor_id in enumerate(anchor_ids):
        columns = []
        for direction in range(4):
            plus_id = f"stage_i_a{anchor_index:02d}_d{direction:02d}_s00_p"
            columns.append((s_surfaces[plus_id] - s_surfaces[anchor_id]) / 0.08)
        residuals[anchor_id] = np.column_stack(columns)
    endpoint_values = {endpoint: {"predicted": [], "observed": []}
                       for endpoint in ENDPOINTS}
    proposal_rows = []
    for candidate in stage_t_manifest["candidates"]:
        if not bool(candidate.get("selection_eligible", False)):
            continue
        candidate_id = candidate["candidate_id"]
        anchor_id = str(candidate["source_candidate_ids"][0])
        anchor_index = int(anchor_id.rsplit("_", 1)[1])
        projection = project_coordinates(
            t_surfaces[candidate_id] - s_surfaces[anchor_id], residuals[anchor_id],
        )
        anchor_utility = aggregate_utilities(stage_s_rows[anchor_id])
        candidate_utility = aggregate_utilities(stage_t_rows[candidate_id])
        row = {
            "candidate_id": candidate_id,
            "anchor_id": anchor_id,
            "coordinates": projection["coordinates"].tolist(),
            "relative_surface_projection_residual": projection["relative_residual_norm"],
            "endpoints": {},
        }
        for endpoint in ENDPOINTS:
            gradient = np.asarray([
                np.median(slopes[(anchor_index, direction)][endpoint])
                for direction in range(4)
            ], float)
            predicted = float(np.dot(gradient, projection["coordinates"]))
            observed = float(candidate_utility[endpoint] - anchor_utility[endpoint])
            row["endpoints"][endpoint] = {
                "predicted_delta": predicted, "observed_delta": observed,
            }
            endpoint_values[endpoint]["predicted"].append(predicted)
            endpoint_values[endpoint]["observed"].append(observed)
        proposal_rows.append(row)
    validation = {
        "n_proposals": len(proposal_rows),
        "endpoints": {
            endpoint: {
                **agreement_summary(values["predicted"], values["observed"]),
                "predicted_deltas": values["predicted"],
                "observed_deltas": values["observed"],
            }
            for endpoint, values in endpoint_values.items()
        },
        "proposals": proposal_rows,
    }
    manual_id = "stage_t_manual_smooth_capacity"
    manual_span = {}
    for anchor_id in anchor_ids:
        projection = project_coordinates(
            t_surfaces[manual_id] - s_surfaces[anchor_id], residuals[anchor_id],
        )
        manual_span[anchor_id] = {
            "projected_norm_fraction": projection["projected_norm_fraction"],
            "projected_energy_fraction": projection["projected_energy_fraction"],
            "relative_residual_norm": projection["relative_residual_norm"],
        }
    return validation, {
        "role": "capacity_span_diagnostic_only",
        "used_for_candidate_generation": False,
        "used_for_selection": False,
        "per_anchor": manual_span,
    }


def advisory_status(scale: dict, seed: dict, stage_t: dict,
                    thresholds: dict) -> dict:
    failures = []
    for endpoint in thresholds["required_endpoints"]:
        scale_sign = scale["endpoints"][endpoint]["sign_agreement_fraction"]
        stage_t_rho = stage_t["endpoints"][endpoint]["spearman"]
        seed_sign = seed["endpoints"][endpoint]["sign_agreement_fraction"]
        if scale_sign is None or scale_sign < float(thresholds["scale_sign_agreement_min"]):
            failures.append(f"{endpoint}:scale_sign")
        if stage_t_rho is None or stage_t_rho < float(thresholds["stage_t_spearman_min"]):
            failures.append(f"{endpoint}:stage_t_prediction")
        if seed_sign is None or seed_sign < float(thresholds["seed_sign_agreement_min"]):
            failures.append(f"{endpoint}:seed_sign")
    return {
        "status": (
            "LOCAL_FOUR_DIRECTION_GRADIENT_ACTIONABLE"
            if not failures else "LOCAL_FOUR_DIRECTION_GRADIENT_NOT_ACTIONABLE"
        ),
        "failed_advisory_checks": failures,
        "thresholds": thresholds,
        "role": "simulation-allocation advice, not a scientific acceptance gate",
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
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != commit:
        raise RuntimeError("search-coordinate diagnostic is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_continuous_field.py", "src/topic4_node_field_search.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("search-coordinate diagnostic runtime paths are dirty")
    loaded, input_audit = {}, {}
    artifact_root = args.artifact_root.resolve()
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    s_rows = {row["candidate_id"]: row for row in loaded["stage_s_summary"]["rows"]}
    t_rows = {row["candidate_id"]: row for row in loaded["stage_t_summary"]["rows"]}
    aggregate_slopes = symmetric_slope_records(loaded["stage_s_manifest"], s_rows)
    scale = scale_predictivity(aggregate_slopes)
    seeds = sorted(per_network_utilities(next(iter(s_rows.values()))))
    if len(seeds) != 2:
        raise RuntimeError("diagnostic requires the frozen two-network fit pool")
    seed_left = symmetric_slope_records(
        loaded["stage_s_manifest"], s_rows, network_seed=seeds[0],
    )
    seed_right = symmetric_slope_records(
        loaded["stage_s_manifest"], s_rows, network_seed=seeds[1],
    )
    seed = seed_predictivity(
        seed_left, seed_right, left_seed=seeds[0], right_seed=seeds[1],
    )
    stage_t, manual_span = response_surface_validation(
        loaded["stage_s_manifest"], s_rows,
        loaded["stage_t_manifest"], t_rows,
        grid_per_axis=int(config["surface_grid_per_axis"]),
    )
    decision = advisory_status(
        scale, seed, stage_t, config["advisory_actionability"],
    )
    payload = {
        "schema_id": config["schema_id"],
        "status": "REV12ND_SEARCH_COORDINATE_DIAGNOSTIC_COMPLETE",
        "scientific_role": config["scientific_role"],
        "advisory_decision": decision,
        "scale_predictivity": scale,
        "network_seed_predictivity": seed,
        "stage_t_out_of_surface_prediction": stage_t,
        "manual_capacity_span": manual_span,
        "inputs": input_audit,
        "provenance": {
            "git_commit": commit,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
            "tracked_modules": tracked,
            "dirty": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "advisory_decision": decision["status"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
