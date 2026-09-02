#!/usr/bin/env python3
"""Freeze geometry-only Gaussian tissue patches and exact N0 support checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv, atomic_write_json, load_frozen_cell, parameter_state_sha256,
    sha256_file,
)
from src.topic5_latent_pass1_v0_2 import build_future_field_data, interpolate_phase_vectors, observable_design  # noqa: E402
from src.topic5_latent_perturbation_v0_2 import DOSES, residual_covariance_direction_sd  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import SPATIAL  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import (  # noqa: E402
    SUPPORT_K, SUPPORT_LANDMARKS_PER_PHASE, reference_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT, cell_dir, replay_states  # noqa: E402


PATCH = SPATIAL / "patch_freeze"
PATCH_FREEZE_REVISION = "PATCH_FREEZE_R0_ALL_GRID_GAUSSIAN_EXACT_N0"
PATCH_WIDTH_MULTIPLIER = 2.0
CENTER_CHUNK = 8
_FIELD_CACHE: dict[str, object] = {}


def patch_dir(row: pd.Series) -> Path:
    return PATCH / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def patch_directions(nodes: np.ndarray) -> tuple[np.ndarray, float, float]:
    xy = np.asarray(nodes, float); distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    local_spacing = float(np.median(np.min(distance, axis=1)))
    width = PATCH_WIDTH_MULTIPLIER * local_spacing
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    directions = np.exp(-(distance ** 2) / max(2.0 * width ** 2, 1e-12))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return directions, local_spacing, width


def run_cell(row: pd.Series, sample: pd.DataFrame, eligibility: pd.Series, device: torch.device, batch_size: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash, decoder_hash = parameter_state_sha256(model), parameter_state_sha256(decoder)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as source:
        ranks = np.asarray(source["ranks"]); split = np.asarray(source["split"]); labels = np.asarray(source["full_train_mode"])
    fit_id = str(row.fit_id)
    if fit_id not in _FIELD_CACHE:
        _FIELD_CACHE[fit_id] = build_future_field_data(
            ranks, split, labels, positive_mode=int(eligibility.positive_mode),
            negative_mode=int(eligibility.negative_mode), tier=str(eligibility.status), shuffle_key=fit_id,
        )
    field_data = _FIELD_CACHE[fit_id]
    states = replay_states(model, ranks, split, sample, device, batch_size)
    with np.load(reference_dir(row) / "reference_contract.npz", allow_pickle=False) as source:
        q = {name: np.asarray(source[name]) for name in source.files}
    pass1 = cell_dir(row)
    with np.load(pass1 / "geometry_arrays.npz", allow_pickle=False) as source:
        geometry = {name: np.asarray(source[name]) for name in source.files}
    with np.load(pass1 / "conditional_manifold_arrays.npz", allow_pickle=False) as source:
        branch_grid = np.asarray(source["field_direction_raw"], float)
    grid = geometry["phase_grid"].astype(float)
    gamma = interpolate_phase_vectors(grid, geometry["gamma_raw"], states["phase"])
    branch = interpolate_phase_vectors(grid, branch_grid, states["phase"])
    event_u = field_data.event_coordinate_z[states["event_index"]]
    conditional = gamma + event_u[:, None] * branch
    residual = states["hidden"].astype(float) - conditional
    progress = interpolate_phase_vectors(grid, geometry["progress_axes_raw"], states["phase"])
    field = interpolate_phase_vectors(grid, geometry["field_axes_raw"], states["phase"])
    observables = observable_design(states["step"], int(row.n_contacts), states["x"], states["recruited"])[:, 1:]
    feature_raw = np.column_stack([
        np.einsum("ij,ij->i", residual, progress), np.einsum("ij,ij->i", residual, field), observables,
    ])
    neighbor_models = {}
    for phase_bin in range(5):
        train_rows = np.flatnonzero((states["split"] == 0) & (states["phase_bin"] == phase_bin))
        ordered = sorted(train_rows.tolist(), key=lambda state_row: (
            hashlib.sha256(
                f"{row.fit_id}\0{int(states['event_index'][state_row])}\0{int(states['step'][state_row])}".encode()
            ).hexdigest(), state_row,
        ))[:SUPPORT_LANDMARKS_PER_PHASE]
        center = q["feature_center_by_phase"][phase_bin].astype(float)
        scale = q["feature_scale_by_phase"][phase_bin].astype(float)
        landmarks = (feature_raw[np.asarray(ordered, int)] - center[None]) / scale[None]
        neighbors = NearestNeighbors(n_neighbors=min(SUPPORT_K, len(landmarks)), algorithm="auto")
        neighbors.fit(landmarks); neighbor_models[phase_bin] = neighbors

    lookup = {(int(event), int(step)): index for index, (event, step) in enumerate(zip(states["event_index"], states["step"]))}
    state_rows = np.asarray([
        lookup[(int(event), int(step))] for event, step in zip(q["reference_event_index"], q["step"])
    ], int)
    ref_feature_raw = feature_raw[state_rows]
    ref_h = q["hidden"].astype(float); ref_conditional = q["conditional_center"].astype(float)
    ref_progress = q["progress_axis"].astype(float); ref_field = q["field_axis"].astype(float)
    ref_bins = q["phase_bin"].astype(int); hidden_dim = ref_h.shape[1]
    with np.load(cache / "plane.npz", allow_pickle=False) as source: nodes = np.asarray(source["nodes_xy_mm"], float)
    if len(nodes) != hidden_dim:
        raise RuntimeError(f"patch hidden/node mismatch {len(nodes)} != {hidden_dim}")
    directions, local_spacing, width = patch_directions(nodes)
    n_ref, n_centers = len(ref_h), len(nodes)
    local_sd = np.full((n_ref, n_centers), np.nan, np.float32)
    for reference in range(n_ref):
        phase_bin = ref_bins[reference]
        for center_index, direction in enumerate(directions):
            local_sd[reference, center_index] = residual_covariance_direction_sd(
                direction, geometry["local_residual_eigenvalues"][phase_bin],
                geometry["local_residual_components"][phase_bin], geometry["local_residual_diagonal"][phase_bin],
            )
    checks = np.zeros((n_ref, n_centers, len(DOSES), 2, 3), np.uint8)
    knn_distance = np.full((n_ref, n_centers, len(DOSES), 2), np.nan, np.float32)
    signs = np.asarray([-1.0, 1.0])
    for start in range(0, n_centers, CENTER_CHUNK):
        stop = min(start + CENTER_CHUNK, n_centers); direction = directions[start:stop]
        sd = local_sd[:, start:stop].astype(float)
        candidate = (
            ref_h[:, None, None, None, :] + direction[None, :, None, None, :]
            * sd[:, :, None, None, None] * DOSES[None, None, :, None, None]
            * signs[None, None, None, :, None]
        )
        shape = candidate.shape[:-1]; flat = candidate.reshape(-1, hidden_dim)
        reference_index = np.broadcast_to(np.arange(n_ref)[:, None, None, None], shape).reshape(-1)
        center_index = np.broadcast_to(np.arange(start, stop)[None, :, None, None], shape).reshape(-1)
        delta = flat - ref_h[reference_index]
        raw = ref_feature_raw[reference_index].copy()
        raw[:, 0] += np.einsum("ij,ij->i", delta, ref_progress[reference_index])
        raw[:, 1] += np.einsum("ij,ij->i", delta, ref_field[reference_index])
        bins = ref_bins[reference_index]
        feature = (
            raw - q["feature_center_by_phase"][bins].astype(float)
        ) / q["feature_scale_by_phase"][bins].astype(float)
        residual_norm = np.linalg.norm(flat - ref_conditional[reference_index], axis=1) / np.sqrt(hidden_dim)
        local_checks = np.zeros((len(flat), 3), np.uint8)
        finite = np.isfinite(flat).all(axis=1) & np.isfinite(feature).all(axis=1)
        local_checks[:, 0] = (
            np.all(flat >= q["node_lower"][None].astype(float), axis=1)
            & np.all(flat <= q["node_upper"][None].astype(float), axis=1)
        )
        local_checks[:, 2] = residual_norm <= q["manifold_residual_q95_by_phase"][bins]
        distances = np.full(len(flat), np.nan)
        for phase_bin in range(5):
            use = np.flatnonzero((bins == phase_bin) & finite)
            if len(use): distances[use] = neighbor_models[phase_bin].kneighbors(feature[use], return_distance=True)[0][:, -1]
        local_checks[:, 1] = distances <= q["knn_q95_by_phase"][bins]
        local_checks[~finite] = 0
        checks[:, start:stop] = local_checks.reshape(*shape, 3)
        knn_distance[:, start:stop] = distances.reshape(shape)
    arrays = {
        "node_xy_mm": nodes.astype(np.float32), "patch_directions": directions.astype(np.float32),
        "local_node_spacing_mm": np.asarray([local_spacing], np.float32),
        "patch_width_mm": np.asarray([width], np.float32), "patch_width_multiplier": np.asarray([PATCH_WIDTH_MULTIPLIER], np.float32),
        "doses": DOSES.astype(np.float32), "patch_local_sd": local_sd,
        "support_checks": checks, "knn_distance": knn_distance,
        "reference_event_index": q["reference_event_index"].astype(np.int64),
        "step": q["step"].astype(np.int16), "phase_target": q["phase_target"].astype(np.float32),
    }
    primary = checks[:, :, 1].all(axis=(2, 3))
    metrics = {
        "contract": "topic5_spatial_patch_freeze_cell_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_freeze_revision": PATCH_FREEZE_REVISION, "status": "PASS",
        "patient": str(row.patient), "fit_id": fit_id, "public_arm": str(row.public_arm), "seed": int(row.seed),
        "n_reference_states": n_ref, "n_patch_centers": n_centers,
        "primary_supported_state_centers": int(primary.sum()), "primary_total_state_centers": int(primary.size),
        "local_node_spacing_mm": local_spacing, "patch_width_mm": width,
        "model_hash_unchanged": model_hash == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_hash == parameter_state_sha256(decoder),
        "reference_contract_sha256": sha256_file(reference_dir(row) / "reference_contract.npz"),
        "response_values_read_before_freeze": False, "target_values_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return arrays, metrics


def write_cell(row: pd.Series, arrays: dict[str, np.ndarray], metrics: dict[str, object]) -> None:
    target = patch_dir(row); target.mkdir(parents=True, exist_ok=True)
    write_npz(target / "patch_contract.npz", arrays); atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True, "patch_freeze_revision": PATCH_FREEZE_REVISION,
        "patch_contract_sha256": sha256_file(target / "patch_contract.npz"),
        "metrics_sha256": sha256_file(target / "metrics.json"), "target_values_read": False,
    })


def aggregate(manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict()); target = patch_dir(row)
        if not (target / "DONE.json").is_file(): missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
        else: rows.append(json.loads((target / "metrics.json").read_text()))
    if rows: atomic_write_csv(PATCH / "PATCH_FREEZE_CELL_SUMMARY.csv", pd.DataFrame(rows))
    payload = {
        "contract": "topic5_spatial_patch_freeze_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_freeze_revision": PATCH_FREEZE_REVISION,
        "status": "PASS" if len(rows) == 630 and not missing else "INCOMPLETE",
        "scheduled_cells": 630, "completed_cells": len(rows),
        "reference_states": int(sum(row["n_reference_states"] for row in rows)),
        "state_center_pairs": int(sum(row["primary_total_state_centers"] for row in rows)),
        "primary_supported_state_centers": int(sum(row["primary_supported_state_centers"] for row in rows)),
        "missing_count": len(missing), "missing_first20": missing[:20],
        "response_values_read_before_freeze": False, "target_values_read": False,
    }
    atomic_write_json(PATCH / "PATCH_FREEZE_STATUS.json", payload)
    if payload["status"] == "PASS":
        atomic_write_json(PATCH / "PATCH_FREEZE_SEAL.json", {
            "sealed": True, "patch_freeze_revision": PATCH_FREEZE_REVISION,
            "summary_sha256": sha256_file(PATCH / "PATCH_FREEZE_CELL_SUMMARY.csv"),
            "status_sha256": sha256_file(PATCH / "PATCH_FREEZE_STATUS.json"),
            "response_values_read_before_freeze": False, "target_values_read": False,
        })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--cell-key")
    parser.add_argument("--limit", type=int); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv"); manifest = manifest_all.copy()
    if args.cell_key:
        fit, arm, seed_text = args.cell_key.split("/"); manifest = manifest[
            manifest.fit_id.eq(fit) & manifest.public_arm.eq(arm) & manifest.seed.eq(int(seed_text.removeprefix("seed")))
        ]
    elif args.limit is not None: manifest = manifest.iloc[:args.limit]
    samples = pd.read_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id")
    device = torch.device(args.device); failures = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = patch_dir(row)
        if (target / "DONE.json").is_file() and not args.force: continue
        try:
            arrays, metrics = run_cell(row, samples[samples.fit_id.eq(row.fit_id)].copy(), eligibility.loc[row.fit_id], device, args.batch_size)
            write_cell(row, arrays, metrics)
            print(f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} support={metrics['primary_supported_state_centers']}/{metrics['primary_total_state_centers']} {metrics['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures.append({"fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed), "error_type": type(error).__name__, "error": str(error)})
            atomic_write_json(target / "FAILURE.json", failures[-1]); print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate(manifest_all); print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
