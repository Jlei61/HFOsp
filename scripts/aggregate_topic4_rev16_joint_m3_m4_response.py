#!/usr/bin/env python3
"""Aggregate M4-shell workers and construct the joint 3x48 Node response tensor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as atlas  # noqa: E402
from scripts import freeze_topic4_rev16_m4_shell_coordinate_atlas as freezer  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.run_topic4_rev16_m4_shell_coordinate_worker import WORKER_STATUS  # noqa: E402
from src.topic4_rev14_fourier_field import mode_inventory, mode_shell  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_response_analysis.json"
OUTPUT_SCHEMA = "topic4_rev16_joint_m3_m4_response_aggregate_v1"
ALLOWED_ANALYSIS_PATHS = frozenset({
    "config/topic4_rev16_joint_m3_m4_response_analysis.json",
    "scripts/aggregate_topic4_rev16_joint_m3_m4_response.py",
    "tests/test_topic4_rev16_joint_m3_m4_response.py",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.is_file() else root / relative


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(canary._jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_inputs(
    config_path: Path, root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev16_joint_m3_m4_response_analysis_v1":
        raise RuntimeError("rev16 joint-response analysis schema changed")
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, record in config["inputs"].items():
        path = _resolve(root, record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev16 joint-response input changed: {name}")
        loaded[name] = (path, json.loads(path.read_text()))
    source = loaded["m4_shell_config"][1]
    freezer._validate_config(source)
    manifest = loaded["m4_shell_manifest"][1]
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("M4-shell manifest is not formally frozen")
    if manifest.get("config_sha256") != config["inputs"]["m4_shell_config"]["sha256"]:
        raise RuntimeError("M4-shell manifest/config mismatch")
    m3 = loaded["m3_response_aggregate"][1]
    if m3.get("status") != "COMPLETE" or not m3.get(
        "inventory", {}
    ).get("complete_cartesian_product"):
        raise RuntimeError("frozen M3 response tensor is incomplete")
    ranking = m3.get("ranking_contract", {})
    if any(ranking.get(key) is not False for key in (
        "natural_kmeans_used", "patient_heldout_used", "figure_used",
    )) or ranking.get("EE_EtoI_ZM") != "off":
        raise RuntimeError("frozen M3 tensor crossed a construction boundary")
    tensor = config["response_tensor"]
    if tensor != {
        "network_seeds": [2331, 2332, 2333],
        "m3_real_coordinates": 28,
        "m4_shell_real_coordinates": 20,
        "joint_real_coordinates": 48,
        "source_coordinate_rms": 0.8,
        "central_difference_denominator": 1.6,
        "network_weighting": "equal",
        "frozen_direction_classifier_input": "full_contact_onset_timing",
        "natural_kmeans_input": "not_used_for_construction",
    }:
        raise RuntimeError("rev16 joint response-tensor contract changed")
    progression = config["progression_rule"]
    if any(progression[key] for key in (
        "natural_kmeans_used_for_construction",
        "patient_heldout_used_for_construction",
        "figure_used_for_construction",
    )) or progression["EE_EtoI_ZM"] != "off":
        raise RuntimeError("rev16 joint construction opened a forbidden input")
    return config, loaded


def _provenance(worker_commit: str) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{worker_commit}..HEAD"],
        cwd=ROOT, text=True,
    ).splitlines()
    allowed = set(changed).issubset(ALLOWED_ANALYSIS_PATHS)
    return {
        "worker_freeze_commit": worker_commit,
        "analysis_commit": head,
        "paths_changed_since_worker_freeze": changed,
        "analysis_only_commit_allowed": allowed,
        "worktree_status": status,
        "formal_ready": bool(allowed and not status),
        "snn_simulation_run": False,
    }


def _inventory(
    source: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_root = root / source["output_root"] / "workers"
    manifest_hash = _sha256(manifest_path)
    commit = manifest["provenance"]["git_commit"]
    rows, missing, invalid = [], [], []
    previous = canary.WORKER_STATUS
    canary.WORKER_STATUS = WORKER_STATUS
    try:
        for candidate in manifest["candidates"]:
            for seed in source["search"]["active_network_seeds"]:
                path = worker_root / f"{candidate['candidate_id']}_seed_{seed}.json"
                key = f"{candidate['candidate_id']}:{seed}"
                if not path.is_file():
                    missing.append(key)
                    continue
                try:
                    rows.append(canary._validate_worker(
                        path.resolve(), json.loads(path.read_text()),
                        candidate=candidate, active_seed=int(seed), manifest=manifest,
                        manifest_sha256=manifest_hash, manifest_commit=commit,
                        config=source, artifact_root=root,
                    ))
                except Exception as error:
                    invalid.append(f"{key}:{error}")
    finally:
        canary.WORKER_STATUS = previous
    expected = len(manifest["candidates"]) * len(
        source["search"]["active_network_seeds"]
    )
    return rows, {
        "expected_runs": expected,
        "present_validated": len(rows),
        "missing": missing,
        "invalid_artifact": invalid,
        "complete_cartesian_product": len(rows) == expected and not missing and not invalid,
    }


def _score_records(
    records: list[dict[str, Any]], manifest: Mapping[str, Any],
    j14_config: Mapping[str, Any], support_path: Path, root: Path,
) -> list[dict[str, Any]]:
    context = historical._patient_context(j14_config, root)
    support = canary._load_support_context(support_path, root, j14_config)
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    rows = []
    for record in records:
        candidate = candidates[record["candidate_id"]]
        adapted = json.loads(json.dumps(candidate))
        coordinate = adapted["coordinate_atlas"]
        coordinate["coordinate_index"] = coordinate["shell_coordinate_index"]
        row = atlas._flat_row(
            canary._score_worker(record, context, support), adapted,
        )
        row["shell_coordinate_index"] = row.pop("coordinate_index")
        row["full_mode_index"] = coordinate["full_mode_index"]
        rows.append(row)
    return rows


def response_tensor(
    rows: list[dict[str, Any]], *, denominator: float = 1.6,
) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != [2331, 2332, 2333]:
        raise RuntimeError("M4-shell response tensor lacks a frozen network")
    metrics = {
        "A": "mode_0_mean", "B": "mode_1_mean", "J14": "j14",
        "support_A": "mode_0_effective_events",
        "support_B": "mode_1_effective_events",
    }
    gradients = {name: np.zeros((3, 20), dtype=float) for name in metrics}
    pairs = []
    for network_index, seed in enumerate(seeds):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for coordinate in range(20):
            pair = [
                row for row in seed_rows
                if row["shell_coordinate_index"] == coordinate
            ]
            if len(pair) != 2:
                raise RuntimeError(
                    f"missing M4-shell sign pair: seed={seed}, coordinate={coordinate}"
                )
            negative = next(row for row in pair if int(row["sign"]) == -1)
            positive = next(row for row in pair if int(row["sign"]) == 1)
            record = {
                "seed": seed, "shell_coordinate_index": coordinate,
                "full_mode_index": positive["full_mode_index"],
                "mode_nx": positive["mode_nx"], "mode_ny": positive["mode_ny"],
                "phase": positive["phase"],
            }
            for name, key in metrics.items():
                value = (float(positive[key]) - float(negative[key])) / denominator
                gradients[name][network_index, coordinate] = value
                record[f"gradient_{name}"] = value
            pairs.append(record)
    return {
        "network_seeds": seeds,
        "coordinate_pairs": pairs,
        "gradients": {name: values.tolist() for name, values in gradients.items()},
    }


def joint_tensor(
    m3_aggregate: Mapping[str, Any], m4_tensor: Mapping[str, Any],
) -> dict[str, Any]:
    m3 = m3_aggregate["response_tensor"]
    if m3["network_seeds"] != m4_tensor["network_seeds"]:
        raise RuntimeError("M3 and M4-shell response networks differ")
    m3_modes = mode_inventory(3)
    m4_modes = mode_inventory(4)
    shell = mode_shell(3, 4)
    if m4_modes[:len(m3_modes)] != m3_modes or m4_modes[len(m3_modes):] != shell:
        raise RuntimeError("M3 and M4-shell coefficient order is not concatenable")
    gradients = {}
    for name in ("A", "B", "J14", "support_A", "support_B"):
        left = np.asarray(m3["gradients"][name], dtype=float)
        right = np.asarray(m4_tensor["gradients"][name], dtype=float)
        if left.shape != (3, 28) or right.shape != (3, 20):
            raise RuntimeError(f"joint response shape changed: {name}")
        gradients[name] = np.concatenate([left, right], axis=1).tolist()
    return {
        "network_seeds": m4_tensor["network_seeds"],
        "full_modes": [list(mode) for mode in m4_modes],
        "m3_coordinate_slice": [0, 28],
        "m4_shell_coordinate_slice": [28, 48],
        "gradients": gradients,
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("joint direction has zero or nonfinite norm")
    return values / norm


def maximin_direction(
    g_a: np.ndarray, g_b: np.ndarray, *,
    support_a: np.ndarray | None = None,
    support_b: np.ndarray | None = None,
    ftol: float = 1e-10, maxiter: int = 4000,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    g_a = np.asarray(g_a, dtype=float)
    g_b = np.asarray(g_b, dtype=float)
    if g_a.ndim != 2 or g_a.shape[0] != 3 or g_b.shape != g_a.shape:
        raise ValueError("joint maximin gradients must be aligned 3xd matrices")
    dimension = g_a.shape[1]
    initial_v = _unit(-np.mean(g_a, axis=0))
    initial_q = min(0.0, float(np.min(-g_a @ initial_v)))
    x0 = np.r_[initial_v, initial_q]
    constraints = [{
        "type": "ineq",
        "fun": lambda x: 1.0 - float(np.dot(x[:-1], x[:-1])),
    }]
    for row in g_a:
        constraints.append({
            "type": "ineq",
            "fun": lambda x, row=row: float(-np.dot(row, x[:-1]) - x[-1]),
        })
    for row in g_b:
        constraints.append({
            "type": "ineq",
            "fun": lambda x, row=row: float(-np.dot(row, x[:-1])),
        })
    for matrix in (support_a, support_b):
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != g_a.shape:
                raise ValueError("joint support gradients are misaligned")
            for row in matrix:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, row=row: float(np.dot(row, x[:-1])),
                })
    result = minimize(
        lambda x: -float(x[-1]), x0, method="SLSQP",
        bounds=[(-1.0, 1.0)] * dimension + [(-100.0, 100.0)],
        constraints=constraints,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )
    vector = np.asarray(result.x[:-1], dtype=float)
    q = float(result.x[-1])
    residuals = np.asarray([constraint["fun"](result.x) for constraint in constraints])
    feasible = bool(result.success and np.min(residuals) >= -tolerance and q > 1e-8)
    normalized = _unit(vector) if feasible else None
    return {
        "success": bool(result.success),
        "feasible_positive_margin": feasible,
        "message": str(result.message), "iterations": int(result.nit),
        "worst_A_improvement_margin": q,
        "minimum_constraint_residual": float(np.min(residuals)),
        "direction": None if normalized is None else normalized.tolist(),
        "predicted_A_changes": None if normalized is None else (g_a @ normalized).tolist(),
        "predicted_B_changes": None if normalized is None else (g_b @ normalized).tolist(),
    }


def consensus_sparse_direction(
    g_a: np.ndarray, g_b: np.ndarray, support_a: np.ndarray, *,
    max_coordinates: int,
) -> dict[str, Any]:
    dimension = g_a.shape[1]
    vector = np.zeros(dimension, dtype=float)
    eligible = []
    for coordinate in range(dimension):
        median = float(np.median(g_a[:, coordinate]))
        if median == 0.0:
            continue
        sign = -float(np.sign(median))
        a_changes = sign * g_a[:, coordinate]
        b_changes = sign * g_b[:, coordinate]
        support_changes = sign * support_a[:, coordinate]
        if (
            np.sum(a_changes < 0.0) == 3
            and np.sum(b_changes <= 0.0) >= 2
            and np.sum(support_changes >= 0.0) >= 2
        ):
            eligible.append((float(np.median(-a_changes)), coordinate, sign))
    eligible.sort(key=lambda row: (-row[0], row[1]))
    selected = eligible[:max_coordinates]
    for _, coordinate, sign in selected:
        vector[coordinate] = sign * float(np.median(np.abs(g_a[:, coordinate])))
    return {
        "feasible": bool(selected),
        "selected_coordinates": [int(row[1]) for row in selected],
        "direction": _unit(vector).tolist() if selected else None,
    }


def construct_directions(
    tensor: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    gradients = {
        key: np.asarray(value, dtype=float)
        for key, value in tensor["gradients"].items()
    }
    optimizer = contract["optimizer"]
    kwargs = {
        "ftol": float(optimizer["ftol"]),
        "maxiter": int(optimizer["maxiter"]),
        "tolerance": float(optimizer["constraint_tolerance"]),
    }
    mean = _unit(-np.mean(gradients["A"], axis=0))
    return {
        "mean_a": {"feasible": True, "direction": mean.tolist()},
        "maximin_bprotected": maximin_direction(
            gradients["A"], gradients["B"], **kwargs,
        ),
        "maximin_supportprotected": maximin_direction(
            gradients["A"], gradients["B"],
            support_a=gradients["support_A"],
            support_b=gradients["support_B"], **kwargs,
        ),
        "consensus_sparse": consensus_sparse_direction(
            gradients["A"], gradients["B"], gradients["support_A"],
            max_coordinates=int(contract["consensus_sparse_maximum_coordinates"]),
        ),
    }


def aggregate(config_path: Path, root: Path) -> dict[str, Any]:
    config, loaded = _load_inputs(config_path.resolve(), root.resolve())
    source = loaded["m4_shell_config"][1]
    manifest_path, manifest = loaded["m4_shell_manifest"]
    provenance = _provenance(manifest["provenance"]["git_commit"])
    records, inventory = _inventory(source, manifest, manifest_path, root.resolve())
    status, error = "INCOMPLETE", None
    shell_tensor = joint = directions = None
    if not provenance["formal_ready"]:
        status, error = "INVALID_PROVENANCE", "analysis worktree is not a clean allowed descendant"
    elif inventory["complete_cartesian_product"]:
        try:
            rows = _score_records(
                records, manifest, loaded["j14_config"][1],
                loaded["support_config"][0], root.resolve(),
            )
            shell_tensor = response_tensor(
                rows, denominator=float(
                    config["response_tensor"]["central_difference_denominator"]
                ),
            )
            joint = joint_tensor(loaded["m3_response_aggregate"][1], shell_tensor)
            directions = construct_directions(
                joint, config["robust_direction_construction"],
            )
            status = "COMPLETE"
        except Exception as caught:
            status, error = "INVALID_INPUT", str(caught)
    output_root = root.resolve() / config["output_root"]
    json_path = output_root / "joint_m3_m4_response_aggregate.json"
    csv_path = output_root / "m4_shell_response_pairs.csv"
    payload = {
        "schema_id": OUTPUT_SCHEMA, "status": status, "input_error": error,
        "inventory": inventory, "provenance": provenance,
        "m4_shell_response_tensor": shell_tensor,
        "joint_response_tensor": joint,
        "robust_directions": directions,
        "construction_contract": config["robust_direction_construction"],
        "progression_rule": config["progression_rule"],
        "ranking_contract": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "figure_used": False, "EE_EtoI_ZM": "off",
            "m3_numeric_source": "frozen_complete_rev15_response_tensor",
            "m4_numeric_source": "raw_worker_npz_rescored_in_this_aggregate",
            "frozen_direction_classifier_input": "full_contact_onset_timing",
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(json_path), "pair_csv": str(csv_path)},
    }
    _atomic_json(json_path, payload)
    _atomic_csv(
        csv_path, [] if shell_tensor is None else shell_tensor["coordinate_pairs"],
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    result = aggregate(args.config, args.artifact_root)
    print(json.dumps({
        "status": result["status"],
        "present_validated": result["inventory"]["present_validated"],
        "robust_direction_families": [] if result["robust_directions"] is None
        else list(result["robust_directions"]),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
