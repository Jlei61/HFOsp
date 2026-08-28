#!/usr/bin/env python3
"""Aggregate the three-network M3 response tensor and construct robust directions."""
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
from scripts import freeze_topic4_rev15_m3_multinetwork_atlas as freezer  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.run_topic4_rev15_m3_multinetwork_atlas_worker import WORKER_STATUS  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_m3_multinetwork_response_analysis.json"
OUTPUT_SCHEMA = "topic4_rev15_m3_multinetwork_response_aggregate_v1"
ALLOWED_ANALYSIS_PATHS = frozenset({
    "config/topic4_rev15_m3_multinetwork_response_analysis.json",
    "scripts/aggregate_topic4_rev15_m3_multinetwork_response.py",
    "tests/test_topic4_rev15_m3_multinetwork_response.py",
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


def _load_inputs(config_path: Path, root: Path) -> tuple[dict, dict[str, tuple[Path, dict]]]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev15_m3_multinetwork_response_analysis_v1":
        raise RuntimeError("multinetwork response-analysis schema changed")
    loaded = {}
    for name, record in config["inputs"].items():
        path = _resolve(root, record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"multinetwork response input changed: {name}")
        loaded[name] = (path, json.loads(path.read_text()))
    source = loaded["multinetwork_config"][1]
    manifest = loaded["multinetwork_manifest"][1]
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("multinetwork worker manifest is not frozen")
    if manifest.get("config_sha256") != config["inputs"]["multinetwork_config"]["sha256"]:
        raise RuntimeError("multinetwork manifest/config mismatch")
    if source["search"]["active_network_seeds"] != [2332, 2333]:
        raise RuntimeError("multinetwork active seed pool changed")
    if config["response_tensor"]["network_seeds"] != [2331, 2332, 2333]:
        raise RuntimeError("three-network response tensor changed")
    forbidden = config["progression_rule"]
    if any(forbidden[key] for key in (
        "natural_kmeans_used_for_construction",
        "patient_heldout_used_for_construction",
        "figure_used_for_construction",
    )):
        raise RuntimeError("forbidden construction input was enabled")
    if forbidden["EE_EtoI_ZM"] != "off":
        raise RuntimeError("connections or slow variables entered Node construction")
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
        "worker_freeze_commit": worker_commit, "analysis_commit": head,
        "paths_changed_since_worker_freeze": changed,
        "analysis_only_commit_allowed": allowed,
        "worktree_status": status,
        "formal_ready": bool(allowed and not status),
        "snn_simulation_run": False,
    }


def _inventory(
    source: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, root: Path,
) -> tuple[list[dict], dict]:
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
    expected = len(manifest["candidates"]) * 2
    return rows, {
        "expected_runs": expected, "present_validated": len(rows),
        "missing": missing, "invalid_artifact": invalid,
        "complete_cartesian_product": len(rows) == expected and not missing and not invalid,
    }


def _score_fresh(
    records: list[dict], manifest: Mapping[str, Any],
    j14_config: Mapping[str, Any], support_path: Path, root: Path,
) -> list[dict]:
    context = historical._patient_context(j14_config, root)
    support = canary._load_support_context(support_path, root, j14_config)
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    return [
        atlas._flat_row(
            canary._score_worker(record, context, support),
            candidates[record["candidate_id"]],
        )
        for record in records
    ]


def _paired_rows(rows: list[dict]) -> list[dict]:
    exact = {int(row["seed"]): row for row in rows if row["candidate_id"] == "exact_off"}
    output = []
    for row in rows:
        if not row["selection_eligible"]:
            continue
        reference = exact[int(row["seed"])]
        output.append({
            **row,
            "delta_A_vs_exact": float(row["mode_0_mean"]) - float(reference["mode_0_mean"]),
            "delta_B_vs_exact": float(row["mode_1_mean"]) - float(reference["mode_1_mean"]),
            "delta_J14_vs_exact": float(row["j14"]) - float(reference["j14"]),
            "delta_A_support_vs_exact": float(row["mode_0_effective_events"]) - float(reference["mode_0_effective_events"]),
            "delta_B_support_vs_exact": float(row["mode_1_effective_events"]) - float(reference["mode_1_effective_events"]),
        })
    return output


def response_tensor(rows: list[dict], denominator: float = 1.6) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != [2331, 2332, 2333]:
        raise RuntimeError("response tensor lacks a frozen network")
    metrics = {
        "A": "mode_0_mean", "B": "mode_1_mean", "J14": "j14",
        "support_A": "mode_0_effective_events",
        "support_B": "mode_1_effective_events",
    }
    gradients = {name: np.zeros((3, 28), dtype=float) for name in metrics}
    pairs = []
    for ni, seed in enumerate(seeds):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for coordinate in range(28):
            pair = [row for row in seed_rows if row["coordinate_index"] == coordinate]
            if len(pair) != 2:
                raise RuntimeError(f"missing sign pair: seed={seed}, coordinate={coordinate}")
            negative = next(row for row in pair if int(row["sign"]) == -1)
            positive = next(row for row in pair if int(row["sign"]) == 1)
            record = {"seed": seed, "coordinate_index": coordinate}
            for name, key in metrics.items():
                value = (float(positive[key]) - float(negative[key])) / denominator
                gradients[name][ni, coordinate] = value
                record[f"gradient_{name}"] = value
            pairs.append(record)
    return {
        "network_seeds": seeds,
        "coordinate_pairs": pairs,
        "gradients": {name: values.tolist() for name, values in gradients.items()},
        "gradient_sign_consistency": {
            name: [
                int(max(np.sum(values[:, j] > 0), np.sum(values[:, j] < 0)))
                for j in range(28)
            ] for name, values in gradients.items()
        },
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("robust direction has zero or nonfinite norm")
    return vector / norm


def maximin_direction(
    g_a: np.ndarray, g_b: np.ndarray,
    *, support_a: np.ndarray | None = None,
    support_b: np.ndarray | None = None,
    ftol: float = 1e-10, maxiter: int = 4000,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    g_a = np.asarray(g_a, dtype=float)
    g_b = np.asarray(g_b, dtype=float)
    if g_a.shape != (3, 28) or g_b.shape != (3, 28):
        raise ValueError("maximin gradients must be 3x28")
    initial_v = _unit(-np.mean(g_a, axis=0))
    initial_q = min(0.0, float(np.min(-g_a @ initial_v)))
    x0 = np.r_[initial_v, initial_q]
    constraints = [
        {"type": "ineq", "fun": lambda x: 1.0 - float(np.dot(x[:-1], x[:-1]))},
    ]
    for row in g_a:
        constraints.append({
            "type": "ineq", "fun": lambda x, row=row: float(-np.dot(row, x[:-1]) - x[-1]),
        })
    for row in g_b:
        constraints.append({
            "type": "ineq", "fun": lambda x, row=row: float(-np.dot(row, x[:-1])),
        })
    for matrix in (support_a, support_b):
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (3, 28):
                raise ValueError("support gradients must be 3x28")
            for row in matrix:
                constraints.append({
                    "type": "ineq", "fun": lambda x, row=row: float(np.dot(row, x[:-1])),
                })
    result = minimize(
        lambda x: -float(x[-1]), x0, method="SLSQP",
        bounds=[(-1.0, 1.0)] * 28 + [(-100.0, 100.0)],
        constraints=constraints,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )
    vector = np.asarray(result.x[:-1], dtype=float)
    q = float(result.x[-1])
    residuals = np.asarray([constraint["fun"](result.x) for constraint in constraints])
    feasible = bool(result.success and np.min(residuals) >= -tolerance and q > 1e-8)
    return {
        "success": bool(result.success), "feasible_positive_margin": feasible,
        "message": str(result.message), "iterations": int(result.nit),
        "worst_A_improvement_margin": q,
        "minimum_constraint_residual": float(np.min(residuals)),
        "direction": _unit(vector).tolist() if feasible else None,
        "predicted_A_changes": (g_a @ _unit(vector)).tolist() if feasible else None,
        "predicted_B_changes": (g_b @ _unit(vector)).tolist() if feasible else None,
    }


def consensus_sparse_direction(
    g_a: np.ndarray, g_b: np.ndarray, support_a: np.ndarray,
    *, max_coordinates: int = 8,
) -> dict[str, Any]:
    vector = np.zeros(28, dtype=float)
    eligible = []
    for j in range(28):
        median = float(np.median(g_a[:, j]))
        if median == 0.0:
            continue
        sign = -float(np.sign(median))
        a_changes = sign * g_a[:, j]
        b_changes = sign * g_b[:, j]
        support_changes = sign * support_a[:, j]
        if (
            np.sum(a_changes < 0.0) == 3
            and np.sum(b_changes <= 0.0) >= 2
            and np.sum(support_changes >= 0.0) >= 2
        ):
            eligible.append((float(np.median(-a_changes)), j, sign, a_changes, b_changes))
    eligible.sort(key=lambda row: (-row[0], row[1]))
    selected = eligible[:max_coordinates]
    for _, j, sign, _, _ in selected:
        vector[j] = sign * float(np.median(np.abs(g_a[:, j])))
    return {
        "feasible": bool(selected),
        "selected_coordinates": [int(row[1]) for row in selected],
        "direction": _unit(vector).tolist() if selected else None,
    }


def construct_directions(tensor: Mapping[str, Any], contract: Mapping[str, Any]) -> dict:
    gradients = {key: np.asarray(value, dtype=float) for key, value in tensor["gradients"].items()}
    optimizer = contract["optimizer"]
    mean = _unit(-np.mean(gradients["A"], axis=0))
    return {
        "mean_a": {
            "feasible": True, "direction": mean.tolist(),
            "predicted_A_changes": (gradients["A"] @ mean).tolist(),
            "predicted_B_changes": (gradients["B"] @ mean).tolist(),
        },
        "maximin_bprotected": maximin_direction(
            gradients["A"], gradients["B"],
            ftol=float(optimizer["ftol"]), maxiter=int(optimizer["maxiter"]),
            tolerance=float(optimizer["constraint_tolerance"]),
        ),
        "maximin_supportprotected": maximin_direction(
            gradients["A"], gradients["B"],
            support_a=gradients["support_A"], support_b=gradients["support_B"],
            ftol=float(optimizer["ftol"]), maxiter=int(optimizer["maxiter"]),
            tolerance=float(optimizer["constraint_tolerance"]),
        ),
        "consensus_sparse": consensus_sparse_direction(
            gradients["A"], gradients["B"], gradients["support_A"],
            max_coordinates=int(contract["consensus_sparse"]["maximum_coordinates"]),
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(canary._jsonable(payload), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def aggregate(config_path: Path, root: Path) -> dict[str, Any]:
    config, loaded = _load_inputs(config_path.resolve(), root.resolve())
    source_path, source = loaded["multinetwork_config"]
    manifest_path, manifest = loaded["multinetwork_manifest"]
    provenance = _provenance(manifest["provenance"]["git_commit"])
    records, inventory = _inventory(source, manifest, manifest_path, root.resolve())
    status, error, fresh_rows, tensor, directions = "INCOMPLETE", None, [], None, None
    if not provenance["formal_ready"]:
        status, error = "INVALID_PROVENANCE", "analysis worktree is not a clean allowed descendant"
    elif inventory["complete_cartesian_product"]:
        try:
            fresh_rows = _score_fresh(
                records, manifest, loaded["j14_config"][1],
                loaded["support_config"][0], root.resolve(),
            )
            seed2331 = loaded["seed2331_atlas_aggregate"][1]
            if seed2331.get("status") != "COMPLETE" or seed2331["inventory"]["present_validated"] != 58:
                raise RuntimeError("seed2331 source atlas is incomplete")
            all_rows = list(seed2331["per_candidate"]) + fresh_rows
            tensor = response_tensor(all_rows, float(config["response_tensor"]["central_difference_denominator"]))
            directions = construct_directions(tensor, config["robust_direction_construction"])
            status = "COMPLETE"
        except Exception as caught:
            status, error = "INVALID_INPUT", str(caught)
    output_root = root.resolve() / config["output_root"]
    json_path = output_root / "m3_multinetwork_response_aggregate.json"
    csv_path = output_root / "m3_multinetwork_response_pairs.csv"
    payload = {
        "schema_id": OUTPUT_SCHEMA, "status": status, "input_error": error,
        "inventory": inventory, "provenance": provenance,
        "response_tensor": tensor, "robust_directions": directions,
        "construction_contract": config["robust_direction_construction"],
        "progression_rule": config["progression_rule"],
        "ranking_contract": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "figure_used": False, "EE_EtoI_ZM": "off",
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(json_path), "pair_csv": str(csv_path)},
    }
    _atomic_json(json_path, payload)
    _atomic_csv(csv_path, [] if tensor is None else tensor["coordinate_pairs"])
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
        "robust_direction_families": [] if result["robust_directions"] is None else list(result["robust_directions"]),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
