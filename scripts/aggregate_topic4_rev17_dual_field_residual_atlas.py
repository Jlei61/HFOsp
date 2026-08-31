#!/usr/bin/env python3
"""Validate and score the rev17 exact-dual-field local response atlas."""
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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev17_dual_field_residual_atlas.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as flat  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from src.topic4_continuous_field import continuous_field_h  # noqa: E402
from src.topic4_core_field_rev9 import reconstruct_node_from_dual_fields  # noqa: E402
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402


STATUS = "REV17_DUAL_FIELD_RESIDUAL_ATLAS_AGGREGATE_COMPLETE"
WORKER_STATUS = "REV12ND_NODE_WORKER_COMPLETE"
REQUIRED_ARRAYS = tuple(dict.fromkeys(
    historical.HISTORICAL_ARRAY_KEYS + ("h", "edge_coefficients",)
))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    local = ROOT / path
    return local if local.exists() else root / path


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


def _load_contract(config_path: Path, root: Path) -> tuple[dict, dict, Path]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_dual_field_residual_atlas_v1":
        raise RuntimeError("rev17 config schema changed")
    for name, record in config["inputs"].items():
        path = _resolve(root, record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev17 input changed: {name}")
    manifest_path = root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "REV17_DUAL_FIELD_RESIDUAL_ATLAS_FROZEN":
        raise RuntimeError("rev17 manifest is not formally frozen")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev17 manifest/config hash mismatch")
    if len(manifest.get("candidates", [])) != 61:
        raise RuntimeError("rev17 candidate count changed")
    if not manifest.get("provenance", {}).get("formal_ready"):
        raise RuntimeError("rev17 worker manifest provenance is not formal")
    return config, manifest, manifest_path


def _analysis_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    return {
        "worker_commit": manifest["provenance"]["git_commit"],
        "analysis_commit": head,
        "analysis_worktree_clean": not dirty,
        "analysis_worktree_status": dirty,
        "same_commit_as_worker": head == manifest["provenance"]["git_commit"],
        "SNN_simulation_run": False,
    }


def _reference_geometry(config: Mapping[str, Any], manifest: Mapping[str, Any],
                        root: Path) -> tuple[dict[int, np.ndarray], dict]:
    transition_path = _resolve(root, config["inputs"]["transition_config"]["path"])
    transition = load_round_config(transition_path)
    anchor = next(
        row for row in manifest["candidates"]
        if row["candidate_id"] == "exact_dual_anchor"
    )
    positions = {}
    for seed in config["search"]["fit_network_seeds"]:
        substrate = build_substrate(
            transition, "node_baseline", int(seed),
            cache_dir=str(root / config["network_cache"]),
            ee_dose=0.0, etoi_dose=0.0,
            node_candidate_override=anchor["node_field"],
            node_dispersion_candidate_override=anchor["node_dispersion_field"],
            artifact_root=root,
        )
        positions[int(seed)] = np.asarray(substrate.positions_e, np.float64)
    stage_record = transition["inputs"]["stage_config"]
    stage_path = _resolve(root, stage_record["path"])
    if _sha256(stage_path) != stage_record["sha256"]:
        raise RuntimeError("rev17 stage config changed")
    return positions, json.loads(stage_path.read_text())


def _expected_arrays(candidate: Mapping[str, Any], positions: np.ndarray,
                     stage: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    engine = stage["engine"]
    target = float(stage["N_core_manual"])
    mean = candidate["node_field"]
    dispersion = candidate["node_dispersion_field"]
    h_mean, _ = continuous_field_h(
        mean["coefficients"], positions,
        n_basis=int(mean["n_basis"]), degree=int(mean["degree"]),
        target_count=target, L=float(engine["L"]),
    )
    h_dispersion, _ = continuous_field_h(
        dispersion["coefficients"], positions,
        n_basis=int(dispersion["n_basis"]), degree=int(dispersion["degree"]),
        target_count=target, L=float(engine["L"]),
    )
    node = reconstruct_node_from_dual_fields(
        h_mean, h_dispersion, n_total=len(h_mean),
        quantile_seed=int(stage["quantile_seed"]),
        core_mean=float(engine["core_mean"]),
        core_std=float(engine["core_std"]),
        v_base=float(engine["v_base"]),
    )
    return np.asarray(h_mean, np.float32), np.asarray(node["delta_vtheta"], np.float32)


def _validate_worker(path: Path, candidate: Mapping[str, Any], seed: int,
                     config: Mapping[str, Any], manifest: Mapping[str, Any],
                     positions: np.ndarray, stage: Mapping[str, Any],
                     root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != WORKER_STATUS:
        raise RuntimeError("rev17 worker did not complete")
    if payload.get("candidate_id") != candidate["candidate_id"] or int(payload.get("seed")) != seed:
        raise RuntimeError("rev17 worker identity changed")
    if payload.get("scientific_role") != config["scientific_role"]:
        raise RuntimeError("rev17 worker scientific role changed")
    if payload.get("field_sha256") != candidate["node_field"]["field_sha256"]:
        raise RuntimeError("rev17 worker mean-field hash changed")
    if payload.get("node_mapping", {}).get("mapping_sha256") != candidate["node_mapping"]["mapping_sha256"]:
        raise RuntimeError("rev17 worker dual mapping hash changed")
    mechanism = payload.get("mechanism_freeze", {})
    if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
        raise RuntimeError("rev17 worker activated a forbidden pathway")
    if mechanism.get("edge_coefficients_all_zero") not in (True, 1):
        raise RuntimeError("rev17 worker edge coefficients are not off")
    provenance = payload.get("provenance", {})
    worker_commit = manifest["provenance"]["git_commit"]
    if (
        provenance.get("expected_git_commit") != worker_commit
        or provenance.get("runtime_modules_match_expected_commit") not in (True, 1)
        or provenance.get("runtime_modules_dirty") not in (False, 0)
        or provenance.get("config_sha256") != manifest["config_sha256"]
        or provenance.get("config_sha256_at_expected_commit") != manifest["config_sha256"]
    ):
        raise RuntimeError("rev17 worker provenance changed")
    npz_path = _resolve(root, payload.get("arrays", {}).get("path", ""))
    if npz_path != path.with_suffix(".npz").resolve() or not npz_path.is_file():
        raise RuntimeError("rev17 worker NPZ is missing or not adjacent")
    if _sha256(npz_path) != payload["arrays"].get("sha256"):
        raise RuntimeError("rev17 worker NPZ hash changed")
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(REQUIRED_ARRAYS).difference(loaded.files)
        if missing:
            raise RuntimeError(f"rev17 worker arrays are incomplete: {sorted(missing)}")
        arrays = {key: np.asarray(loaded[key]).copy() for key in REQUIRED_ARRAYS}
    if not np.array_equal(arrays["positions_E"], np.asarray(positions, np.float32)):
        raise RuntimeError("rev17 worker network positions changed")
    expected_h, expected_delta = _expected_arrays(candidate, positions, stage)
    if not np.array_equal(arrays["h"], expected_h):
        raise RuntimeError("rev17 worker mean field differs from frozen candidate")
    if not np.array_equal(arrays["delta_vtheta"], expected_delta):
        raise RuntimeError("rev17 worker dual-field threshold mapping changed")
    if not np.array_equal(arrays["edge_coefficients"], np.zeros_like(arrays["edge_coefficients"])):
        raise RuntimeError("rev17 worker NPZ contains nonzero edge coefficients")
    simulation = payload.get("simulation", {})
    runaway = simulation.get("runaway_early_stop_ms")
    return {
        "candidate_id": candidate["candidate_id"], "seed": int(seed),
        "selection_eligible": bool(candidate["selection_eligible"]),
        "inventory_status": "PRESENT_VALIDATED",
        "run_status": "INVALID_RUNAWAY" if runaway is not None else "VALID",
        "worker_json": str(path), "worker_json_sha256": _sha256(path),
        "worker_npz": str(npz_path), "worker_npz_sha256": _sha256(npz_path),
        "arrays": arrays, "payload": payload,
    }


def _flat(scored: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    row = flat._flat_row(scored, candidate)
    coordinates = candidate.get("residual_coordinates") or {}
    row.update({
        "channel": coordinates.get("channel"),
        "mode_index": coordinates.get("mode_index"),
        "kx": coordinates.get("kx"), "ky": coordinates.get("ky"),
        "orientation": coordinates.get("orientation"),
        "signed_log_surface_rms": coordinates.get("signed_log_surface_rms"),
    })
    return row


def finite_differences(rows: list[dict[str, Any]], amplitude: float) -> list[dict[str, Any]]:
    by_key = {(row["candidate_id"], int(row["seed"])): row for row in rows}
    anchors = {int(row["seed"]): row for row in rows if row["candidate_id"] == "exact_dual_anchor"}
    endpoints = ("j14", "mode_0_mean", "mode_1_mean",
                 "mode_0_effective_events", "mode_1_effective_events")
    grouped: dict[tuple[str, int], dict[int, str]] = {}
    for row in rows:
        if row["channel"] is None:
            continue
        signs = grouped.setdefault(
            (str(row["channel"]), int(row["mode_index"])), {}
        )
        orientation = int(row["orientation"])
        candidate_id = str(row["candidate_id"])
        existing = signs.get(orientation)
        if existing is not None and existing != candidate_id:
            raise RuntimeError("multiple rev17 candidates occupy one residual coordinate")
        signs[orientation] = candidate_id
    output = []
    for (channel, mode), signs in sorted(grouped.items()):
        if set(signs) != {-1, 1}:
            raise RuntimeError("rev17 residual coordinate is missing an antithetic sign")
        for seed in sorted(anchors):
            negative = by_key[(signs[-1], seed)]
            positive = by_key[(signs[1], seed)]
            anchor = anchors[seed]
            record = {"channel": channel, "mode_index": mode, "seed": seed}
            for endpoint in endpoints:
                left, center, right = map(float, (
                    negative[endpoint], anchor[endpoint], positive[endpoint],
                ))
                record[f"{endpoint}_derivative"] = (right - left) / (2.0 * amplitude)
                record[f"{endpoint}_curvature"] = (right + left - 2.0 * center) / (amplitude ** 2)
            output.append(record)
    return output


def summarize_differences(rows: list[dict[str, Any]], amplitude: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["channel"], int(row["mode_index"])), []).append(row)
    output = []
    for (channel, mode), values in sorted(grouped.items()):
        record: dict[str, Any] = {"channel": channel, "mode_index": mode}
        derivative_keys = [key for key in values[0] if key.endswith("_derivative")]
        for key in derivative_keys:
            numbers = np.asarray([row[key] for row in values], float)
            record[f"mean_{key}"] = float(np.mean(numbers))
            record[f"positive_networks_{key}"] = int(np.sum(numbers > 0.0))
            record[f"negative_networks_{key}"] = int(np.sum(numbers < 0.0))
        curvature = np.asarray([row["j14_curvature"] for row in values], float)
        derivative = np.asarray([row["j14_derivative"] for row in values], float)
        record["mean_abs_j14_curvature"] = float(np.mean(np.abs(curvature)))
        record["j14_linearity_ratio"] = float(
            np.mean(np.abs(curvature)) * amplitude
            / max(np.mean(np.abs(derivative)), 1e-12)
        )
        output.append(record)
    output.sort(key=lambda row: (
        -max(row["negative_networks_mode_0_mean_derivative"],
             row["positive_networks_mode_0_mean_derivative"]),
        abs(row["mean_mode_0_mean_derivative"]), row["channel"], row["mode_index"],
    ))
    return output


def aggregate(config_path: Path = DEFAULT_CONFIG,
              root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config, manifest, manifest_path = _load_contract(config_path.resolve(), root.resolve())
    provenance = _analysis_provenance(manifest)
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    worker_root = root / config["output_root"] / "workers"
    expected = len(candidates) * len(seeds)
    missing = [
        f"{candidate}:{seed}" for candidate in candidates for seed in seeds
        if not (worker_root / f"{candidate}_seed_{seed}.json").is_file()
    ]
    validated, invalid, scored_rows = [], [], []
    if not missing:
        positions, stage = _reference_geometry(config, manifest, root.resolve())
        for candidate_id, candidate in candidates.items():
            for seed in seeds:
                path = (worker_root / f"{candidate_id}_seed_{seed}.json").resolve()
                try:
                    validated.append(_validate_worker(
                        path, candidate, seed, config, manifest,
                        positions[seed], stage, root.resolve(),
                    ))
                except Exception as error:
                    invalid.append(f"{candidate_id}:{seed}:{error}")
    status, input_error = "INCOMPLETE", None
    differences, response_summary = [], []
    if not provenance["analysis_worktree_clean"]:
        status, input_error = "INVALID_PROVENANCE", "analysis worktree is dirty"
    elif not missing and not invalid and len(validated) == expected:
        try:
            j14_config = json.loads(_resolve(
                root, config["inputs"]["j14_config"]["path"]
            ).read_text())
            context = historical._patient_context(j14_config, root.resolve())
            support = canary._load_support_context(
                _resolve(root, config["inputs"]["patient_support_config"]["path"]),
                root.resolve(), j14_config,
            )
            scored_rows = [
                _flat(canary._score_worker(record, context, support),
                      candidates[record["candidate_id"]])
                for record in validated
            ]
            amplitude = float(config["dual_field_residual"]["amplitude"])
            differences = finite_differences(scored_rows, amplitude)
            response_summary = summarize_differences(differences, amplitude)
            status = STATUS
        except Exception as error:
            status, input_error = "INVALID_INPUT", str(error)
    inventory = {
        "expected_runs": expected, "present_validated": len(validated),
        "missing": missing, "invalid_artifact": invalid,
        "complete_cartesian_product": len(validated) == expected and not missing and not invalid,
    }
    output_root = root / config["output_root"] / "analysis"
    json_path = output_root / "dual_field_response_aggregate.json"
    run_csv = output_root / "dual_field_scored_runs.csv"
    response_csv = output_root / "dual_field_response_summary.csv"
    payload = {
        "schema_id": "topic4_rev17_dual_field_residual_aggregate_v1",
        "status": status, "input_error": input_error,
        "inventory": inventory, "provenance": provenance,
        "scored_runs": scored_rows,
        "finite_differences": differences,
        "response_summary": response_summary,
        "boundaries": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off", "SNN_simulation_run_by_aggregator": False,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(json_path), "runs_csv": str(run_csv),
                    "response_csv": str(response_csv)},
    }
    _atomic_json(json_path, payload)
    _atomic_csv(run_csv, scored_rows)
    _atomic_csv(response_csv, response_summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    payload = aggregate(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "expected_runs": payload["inventory"]["expected_runs"],
        "missing": len(payload["inventory"]["missing"]),
        "invalid": len(payload["inventory"]["invalid_artifact"]),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
