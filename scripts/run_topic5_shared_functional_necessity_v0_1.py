#!/usr/bin/env python3
"""Train-side operator extraction, cross-topology freeze and heldout lesion.

This runner never changes model parameters.  It has three explicit stages:

1. ``train-operator`` extracts a Gaussian-patch to future-contact derivative
   from axis-train events only;
2. ``freeze-directions`` constructs leave-one-real-topology-out components;
3. ``lesion`` deletes those components in heldout-test reference states and
   measures delayed next-contact likelihood against matched displacements.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    load_frozen_cell,
    parameter_state_sha256,
    sha256_file,
)
from src.topic5_latent_pass1_v0_2 import interpolate_phase_vectors, observable_design  # noqa: E402
from src.topic5_latent_perturbation_v0_2 import (  # noqa: E402
    PHASE_TARGETS,
    PRIMARY_DOSE,
    SUPPORT_K,
    residual_covariance_direction_sd,
)
from src.topic5_latent_response_v0_2 import raw_logits_stop  # noqa: E402
from src.topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    CONTROL_FAMILIES,
    LESION_DOSES,
    REAL_ARMS,
    drop_heldout_outcome_fields,
    equal_norm_subspace_toward_center,
    equal_norm_toward_center,
    leave_one_topology_component,
    orthogonalize_rows,
    projection_erasure,
    rank_set_nll,
    single_operator_component,
    subspace_projection_erasure,
    unit_vector,
)
from scripts.freeze_topic5_latent_reference_states_v0_2 import (  # noqa: E402
    SUPPORT_LANDMARKS_PER_PHASE,
    reference_dir,
    robust_columns,
    select_references,
)
from scripts.freeze_topic5_spatial_patch_contract_v0_2 import (  # noqa: E402
    patch_dir,
    patch_directions,
)
from scripts.run_topic5_axis_perturbations_v0_2 import (  # noqa: E402
    HORIZON,
    future_input,
    open_loop_pair,
)
from scripts.run_topic5_latent_pass1_v0_2 import (  # noqa: E402
    OUT,
    PARENT,
    cell_dir,
    replay_states,
)


NECESSITY = OUT / "shared_functional_computation_necessity_v0_2"
TRAIN_OPERATOR = NECESSITY / "train_operator"
DIRECTIONS = NECESSITY / "direction_freeze"
LESION = NECESSITY / "heldout_lesion"
SUBSPACE = NECESSITY / "heldout_subspace_sensitivity"
TRAIN_EVENT_CAP = 64
TRAIN_OPERATOR_DOSE = float(PRIMARY_DOSE)
FUTURE_TAU = (1, 2, 3)
TRAIN_OPERATOR_REVISION = "TRAIN_SPLIT_PATCH_OPERATOR_R1_TARGET_FREE_PHASE_CENTER"
DIRECTION_REVISION = "LEAVE_ONE_REAL_TOPOLOGY_OUT_SVD_R0"
LESION_REVISION = "HELDOUT_DELAYED_NEXT_CONTACT_NLL_R1_TARGET_FREE_PHASE_CENTER"
SUBSPACE_REVISION = "HELDOUT_CUMULATIVE_RANK123_SENSITIVITY_R1_TARGET_FREE_PHASE_CENTER"
LOCAL_NORMAL_INDICES = tuple(range(8))
PCA_INDICES = (10, 11, 12)
PAIR_CHUNK = 4096
FLOAT32_REFERENCE_REPLAY_TOLERANCE = 1.0e-5


def train_operator_dir(row: pd.Series) -> Path:
    return TRAIN_OPERATOR / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def direction_dir(fit_id: str, heldout_arm: str) -> Path:
    return DIRECTIONS / "per_fit" / str(fit_id) / str(heldout_arm)


def lesion_dir(row: pd.Series) -> Path:
    return LESION / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def subspace_dir(row: pd.Series) -> Path:
    return SUBSPACE / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


@dataclass
class StateContext:
    model: torch.nn.Module
    decoder: torch.nn.Module
    ranks: np.ndarray
    split: np.ndarray
    states: dict[str, np.ndarray]
    geometry: dict[str, np.ndarray]
    q: dict[str, np.ndarray]
    conditional: np.ndarray
    progress: np.ndarray
    field: np.ndarray
    feature_raw: np.ndarray
    neighbor_models: dict[int, NearestNeighbors]
    feature_center_by_phase: np.ndarray
    feature_scale_by_phase: np.ndarray
    knn_q95_by_phase: np.ndarray
    manifold_residual_q95_by_phase: np.ndarray
    parameter_hash_before: str
    decoder_hash_before: str

    def state_rows(self, events: np.ndarray, steps: np.ndarray) -> np.ndarray:
        lookup = {
            (int(event), int(step)): index
            for index, (event, step) in enumerate(zip(self.states["event_index"], self.states["step"]))
        }
        return np.asarray([lookup[(int(event), int(step))] for event, step in zip(events, steps)], int)

    def support(self, state_rows: np.ndarray, candidate_hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(state_rows, dtype=int)
        candidate = np.asarray(candidate_hidden, dtype=np.float64)
        base = self.states["hidden"][rows].astype(np.float64)
        delta = candidate - base
        raw = self.feature_raw[rows].copy()
        raw[:, 0] += np.einsum("ij,ij->i", delta, self.progress[rows])
        raw[:, 1] += np.einsum("ij,ij->i", delta, self.field[rows])
        bins = self.states["phase_bin"][rows].astype(int)
        feature = (
            raw - self.feature_center_by_phase[bins]
        ) / self.feature_scale_by_phase[bins]
        residual_norm = np.linalg.norm(candidate - self.conditional[rows], axis=1) / np.sqrt(candidate.shape[1])
        checks = np.zeros((len(candidate), 3), dtype=np.uint8)
        finite = np.isfinite(candidate).all(axis=1) & np.isfinite(feature).all(axis=1)
        checks[:, 0] = (
            np.all(candidate >= self.q["node_lower"][None].astype(float), axis=1)
            & np.all(candidate <= self.q["node_upper"][None].astype(float), axis=1)
        )
        checks[:, 2] = residual_norm <= self.manifold_residual_q95_by_phase[bins]
        distance = np.full(len(candidate), np.nan, dtype=np.float64)
        for phase_bin in range(5):
            use = np.flatnonzero((bins == phase_bin) & finite)
            if len(use):
                distance[use] = self.neighbor_models[phase_bin].kneighbors(
                    feature[use], return_distance=True
                )[0][:, -1]
        checks[:, 1] = distance <= self.knn_q95_by_phase[bins]
        checks[~finite] = 0
        return checks, distance


def build_context(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> StateContext:
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash = parameter_state_sha256(model)
    decoder_hash = parameter_state_sha256(decoder)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as source:
        ranks = np.asarray(source["ranks"])
        split = np.asarray(source["split"])
    states = replay_states(model, ranks, split, sample, device, batch_size)
    pass1 = cell_dir(row)
    with np.load(pass1 / "geometry_arrays.npz", allow_pickle=False) as source:
        geometry = {name: np.asarray(source[name]) for name in source.files}
    with np.load(reference_dir(row) / "reference_contract.npz", allow_pickle=False) as source:
        q = {name: np.asarray(source[name]) for name in source.files}
    # The frozen reference archive predates this necessity experiment and also
    # contains quantities calculated from each event's completed suffix.  They
    # are intentionally removed here so neither the lesion centre nor the
    # support gate can accidentally consume the held-out outcome.
    q = drop_heldout_outcome_fields(q)
    grid = geometry["phase_grid"].astype(float)
    gamma = interpolate_phase_vectors(grid, geometry["gamma_raw"], states["phase"])
    # P0 fix: the support centre and deletion reference must not use the full
    # heldout event's future-field coordinate.  The train-fitted phase curve is
    # available before any heldout suffix is read and is therefore the only
    # admissible centre for this necessity experiment.
    conditional = gamma
    residual = states["hidden"].astype(float) - conditional
    progress = interpolate_phase_vectors(grid, geometry["progress_axes_raw"], states["phase"])
    field = interpolate_phase_vectors(grid, geometry["field_axes_raw"], states["phase"])
    observables = observable_design(
        states["step"], int(row.n_contacts), states["x"], states["recruited"]
    )[:, 1:]
    feature_raw = np.column_stack([
        np.einsum("ij,ij->i", residual, progress),
        np.einsum("ij,ij->i", residual, field),
        observables,
    ])
    neighbor_models: dict[int, NearestNeighbors] = {}
    feature_centers: list[np.ndarray] = []
    feature_scales: list[np.ndarray] = []
    knn_q95: list[float] = []
    residual_q95: list[float] = []
    train_validation = states["split"] <= 1
    for phase_bin in range(5):
        local = feature_raw[train_validation & (states["phase_bin"] == phase_bin)]
        feature_center, feature_scale = robust_columns(local)
        train_rows = np.flatnonzero((states["split"] == 0) & (states["phase_bin"] == phase_bin))
        validation_rows = np.flatnonzero(
            (states["split"] == 1) & (states["phase_bin"] == phase_bin)
        )
        ordered = sorted(
            train_rows.tolist(),
            key=lambda state_row: (
                hashlib.sha256(
                    f"{row.fit_id}\0{int(states['event_index'][state_row])}\0{int(states['step'][state_row])}".encode()
                ).hexdigest(),
                state_row,
            ),
        )[:SUPPORT_LANDMARKS_PER_PHASE]
        if not ordered:
            raise RuntimeError(f"no train support landmarks in phase {phase_bin}")
        landmarks = (
            feature_raw[np.asarray(ordered, int)] - feature_center[None]
        ) / feature_scale[None]
        validation_features = (
            feature_raw[validation_rows] - feature_center[None]
        ) / feature_scale[None]
        neighbors = NearestNeighbors(n_neighbors=min(SUPPORT_K, len(landmarks)), algorithm="auto")
        neighbors.fit(landmarks)
        neighbor_models[phase_bin] = neighbors
        validation_distance = neighbors.kneighbors(
            validation_features, return_distance=True
        )[0][:, -1]
        feature_centers.append(feature_center)
        feature_scales.append(feature_scale)
        knn_q95.append(float(np.quantile(validation_distance, 0.95)))
        residual_q95.append(float(np.quantile(
            np.linalg.norm(residual[train_validation & (states["phase_bin"] == phase_bin)], axis=1)
            / np.sqrt(residual.shape[1]),
            0.95,
        )))
    return StateContext(
        model=model,
        decoder=decoder,
        ranks=ranks,
        split=split,
        states=states,
        geometry=geometry,
        q=q,
        conditional=conditional,
        progress=progress,
        field=field,
        feature_raw=feature_raw,
        neighbor_models=neighbor_models,
        feature_center_by_phase=np.stack(feature_centers),
        feature_scale_by_phase=np.stack(feature_scales),
        knn_q95_by_phase=np.asarray(knn_q95, float),
        manifold_residual_q95_by_phase=np.asarray(residual_q95, float),
        parameter_hash_before=model_hash,
        decoder_hash_before=decoder_hash,
    )


def select_train_reference_rows(context: StateContext, sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train = sample[sample["split"].eq(0)].sort_values(
        ["identity_sha256", "event_array_index"], kind="mergesort"
    )
    events = train["event_array_index"].to_numpy(int)[:TRAIN_EVENT_CAP]
    rows = select_references(context.states, events)
    if not len(rows):
        raise RuntimeError("no train reference states")
    phase_target = np.tile(PHASE_TARGETS, len(events))[: len(rows)]
    return rows, phase_target


def directional_sd(
    directions: np.ndarray,
    phase_bins: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> np.ndarray:
    values = np.full((len(phase_bins), len(directions)), np.nan, dtype=np.float64)
    for reference, phase_bin in enumerate(np.asarray(phase_bins, int)):
        for center, direction in enumerate(directions):
            values[reference, center] = residual_covariance_direction_sd(
                direction,
                geometry["local_residual_eigenvalues"][phase_bin],
                geometry["local_residual_components"][phase_bin],
                geometry["local_residual_diagonal"][phase_bin],
            )
    return values


@torch.no_grad()
def run_train_operator_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    context = build_context(row, sample, eligibility, device, batch_size)
    reference_rows, phase_target = select_train_reference_rows(context, sample)
    h = context.states["hidden"][reference_rows].astype(float)
    recruited = context.states["recruited"][reference_rows].astype(np.uint8)
    step = context.states["step"][reference_rows].astype(int)
    event = context.states["event_index"][reference_rows].astype(int)
    phase_bins = context.states["phase_bin"][reference_rows].astype(int)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "plane.npz", allow_pickle=False) as source:
        nodes = np.asarray(source["nodes_xy_mm"], float)
    directions, local_spacing, patch_width = patch_directions(nodes)
    if directions.shape[1] != h.shape[1]:
        raise RuntimeError("patch basis and hidden state do not align")
    local_sd = directional_sd(directions, phase_bins, context.geometry)
    ref_grid, center_grid = np.indices(local_sd.shape)
    ref_flat, center_flat = ref_grid.ravel(), center_grid.ravel()
    magnitude = TRAIN_OPERATOR_DOSE * local_sd[ref_flat, center_flat]
    delta = directions[center_flat] * magnitude[:, None]
    minus_checks, _ = context.support(reference_rows[ref_flat], h[ref_flat] - delta)
    plus_checks, _ = context.support(reference_rows[ref_flat], h[ref_flat] + delta)
    eligible = (
        minus_checks.all(axis=1)
        & plus_checks.all(axis=1)
        & np.isfinite(magnitude)
        & (magnitude > 1e-8)
    )
    ref_flat, center_flat, magnitude = ref_flat[eligible], center_flat[eligible], magnitude[eligible]
    n_phase, n_center, n_contact = len(PHASE_TARGETS), len(directions), int(row.n_contacts)
    total = np.zeros((n_phase, n_center, HORIZON + 1, n_contact), np.float64)
    counts = np.zeros((n_phase, n_center, HORIZON + 1), np.int32)
    phase_index = np.searchsorted(PHASE_TARGETS, phase_target)
    for start in range(0, len(ref_flat), PAIR_CHUNK):
        use = slice(start, min(start + PAIR_CHUNK, len(ref_flat)))
        ref = ref_flat[use]
        center = center_flat[use]
        mag = magnitude[use]
        pair_delta = directions[center] * mag[:, None]
        result = open_loop_pair(
            context.model,
            h[ref] - pair_delta,
            h[ref] + pair_delta,
            recruited[ref],
            step[ref],
            event[ref],
            context.ranks,
            2.0 * mag,
            context.q["contact_progress_axis"].astype(float),
            context.q["contact_future_field_axis"].astype(float),
            device,
        )
        for local in range(len(ref)):
            p = int(phase_index[ref[local]])
            c = int(center[local])
            valid = result["valid"][local].astype(bool)
            total[p, c, valid] += result["contact_response"][local, valid]
            counts[p, c, valid] += 1
    mean_operator = np.divide(
        total,
        counts[..., None],
        out=np.full_like(total, np.nan),
        where=counts[..., None] > 0,
    )
    selected_operator = mean_operator[:, :, FUTURE_TAU, :]
    finite_count = np.isfinite(selected_operator).sum(axis=(0, 2))
    pooled = np.divide(
        np.nansum(selected_operator, axis=(0, 2)),
        finite_count,
        out=np.full((n_center, n_contact), np.nan, dtype=np.float64),
        where=finite_count > 0,
    ).T
    arrays = {
        "phase_targets": PHASE_TARGETS.astype(np.float32),
        "future_tau": np.asarray(FUTURE_TAU, np.int8),
        "node_xy_mm": nodes.astype(np.float32),
        "patch_directions": directions.astype(np.float32),
        "operator_dose_local_sd": np.asarray(TRAIN_OPERATOR_DOSE, np.float32),
        "mean_contact_operator": mean_operator.astype(np.float32),
        "valid_counts": counts,
        "pooled_contact_by_patch_operator": pooled.astype(np.float32),
        "train_reference_event_index": event.astype(np.int64),
        "train_reference_step": step.astype(np.int16),
        "train_reference_phase_target": phase_target.astype(np.float32),
    }
    metrics = {
        "contract": "topic5_train_split_patch_operator_cell_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": TRAIN_OPERATOR_REVISION,
        "status": "PASS",
        "patient": str(row.patient),
        "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "n_train_reference_events": int(np.unique(event).size),
        "n_train_reference_states": int(len(reference_rows)),
        "eligible_state_patch_pairs": int(len(ref_flat)),
        "possible_state_patch_pairs": int(local_sd.size),
        "finite_operator_fraction": float(np.isfinite(pooled).mean()),
        "local_node_spacing_mm": float(local_spacing),
        "patch_width_mm": float(patch_width),
        "train_future_inputs_used": True,
        "heldout_target_values_read": False,
        "model_hash_unchanged": context.parameter_hash_before == parameter_state_sha256(context.model),
        "decoder_hash_unchanged": context.decoder_hash_before == parameter_state_sha256(context.decoder),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return arrays, metrics


def write_train_operator(row: pd.Series, arrays: dict[str, np.ndarray], metrics: dict[str, object]) -> None:
    target = train_operator_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    write_npz(target / "train_operator.npz", arrays)
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True,
        "revision": TRAIN_OPERATOR_REVISION,
        "operator_sha256": sha256_file(target / "train_operator.npz"),
        "metrics_sha256": sha256_file(target / "metrics.json"),
        "heldout_target_values_read": False,
    })


def aggregate_train_operator(manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = train_operator_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
        else:
            rows.append(json.loads((target / "metrics.json").read_text()))
    if rows:
        atomic_write_csv(TRAIN_OPERATOR / "TRAIN_OPERATOR_CELL_SUMMARY.csv", pd.DataFrame(rows))
    payload = {
        "contract": "topic5_train_split_patch_operator_execution_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": TRAIN_OPERATOR_REVISION,
        "status": "PASS" if len(rows) == len(manifest) and not missing else "INCOMPLETE",
        "scheduled_cells": int(len(manifest)),
        "completed_cells": int(len(rows)),
        "eligible_state_patch_pairs": int(sum(row["eligible_state_patch_pairs"] for row in rows)),
        "possible_state_patch_pairs": int(sum(row["possible_state_patch_pairs"] for row in rows)),
        "model_hash_unchanged_cells": int(sum(bool(row["model_hash_unchanged"]) for row in rows)),
        "decoder_hash_unchanged_cells": int(sum(bool(row["decoder_hash_unchanged"]) for row in rows)),
        "missing_count": len(missing),
        "missing_first20": missing[:20],
        "heldout_target_values_read": False,
    }
    atomic_write_json(TRAIN_OPERATOR / "TRAIN_OPERATOR_STATUS.json", payload)
    return payload


def pooled_arm_operator(manifest: pd.DataFrame, fit_id: str, arm: str) -> tuple[np.ndarray, list[str]]:
    rows = manifest[manifest.fit_id.eq(fit_id) & manifest.public_arm.eq(arm)].sort_values("seed")
    operators, hashes = [], []
    for _, row in rows.iterrows():
        path = train_operator_dir(row) / "train_operator.npz"
        with np.load(path, allow_pickle=False) as source:
            operators.append(np.asarray(source["pooled_contact_by_patch_operator"], float))
        hashes.append(sha256_file(path))
    if len(operators) != 3:
        raise RuntimeError(f"expected 3 seed operators for {fit_id}/{arm}, found {len(operators)}")
    stacked = np.stack(operators)
    finite_count = np.isfinite(stacked).sum(axis=0)
    pooled = np.divide(
        np.nansum(stacked, axis=0),
        finite_count,
        out=np.full(stacked.shape[1:], np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    return pooled, hashes


def freeze_directions(manifest: pd.DataFrame) -> dict[str, object]:
    rows = []
    for fit_id, fit_rows in manifest.groupby("fit_id", sort=True):
        fit_id = str(fit_id)
        operators: dict[str, np.ndarray] = {}
        hashes: dict[str, list[str]] = {}
        for arm in (*REAL_ARMS, "C-suffix"):
            operators[arm], hashes[arm] = pooled_arm_operator(manifest, fit_id, arm)
        first = fit_rows.iloc[0]
        with np.load(patch_dir(first) / "patch_contract.npz", allow_pickle=False) as source:
            patch_basis = np.asarray(source["patch_directions"], float)
            node_xy = np.asarray(source["node_xy_mm"], float)
        c_suffix = single_operator_component(operators["C-suffix"], patch_basis, rank=3)
        for heldout_arm in REAL_ARMS:
            shared = leave_one_topology_component(operators, heldout_arm, patch_basis, rank=3)
            target = direction_dir(fit_id, heldout_arm)
            target.mkdir(parents=True, exist_ok=True)
            arrays = {
                "node_xy_mm": node_xy.astype(np.float32),
                "source_arms": shared["source_arms"],
                "shared_consensus_operator": shared["consensus_operator"].astype(np.float32),
                "shared_patch_components": shared["patch_components"].astype(np.float32),
                "shared_hidden_components": shared["hidden_components"].astype(np.float32),
                "shared_singular_values": shared["singular_values"].astype(np.float32),
                "shared_explained_fraction": shared["explained_fraction"].astype(np.float32),
                "c_suffix_patch_components": c_suffix["patch_components"].astype(np.float32),
                "c_suffix_hidden_components": c_suffix["hidden_components"].astype(np.float32),
                "c_suffix_singular_values": c_suffix["singular_values"].astype(np.float32),
                "c_suffix_explained_fraction": c_suffix["explained_fraction"].astype(np.float32),
            }
            write_npz(target / "direction_contract.npz", arrays)
            metadata = {
                "contract": "topic5_leave_one_topology_out_direction_v0_1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "revision": DIRECTION_REVISION,
                "status": "PASS",
                "fit_id": fit_id,
                "heldout_arm": heldout_arm,
                "source_real_arms": shared["source_arms"].tolist(),
                "heldout_arm_operator_read": False,
                "c_suffix_used_only_as_control": True,
                "source_operator_sha256": {
                    arm: hashes[arm] for arm in shared["source_arms"].tolist()
                },
                "c_suffix_operator_sha256": hashes["C-suffix"],
                "shared_first_component_explained_fraction": float(shared["explained_fraction"][0]),
                "c_suffix_first_component_explained_fraction": float(c_suffix["explained_fraction"][0]),
                "heldout_target_values_read": False,
            }
            atomic_write_json(target / "metadata.json", metadata)
            atomic_write_json(target / "DONE.json", {
                "ok": True,
                "revision": DIRECTION_REVISION,
                "direction_sha256": sha256_file(target / "direction_contract.npz"),
                "metadata_sha256": sha256_file(target / "metadata.json"),
                "heldout_arm_operator_read": False,
            })
            rows.append(metadata)
    frame = pd.DataFrame(rows)
    atomic_write_csv(DIRECTIONS / "DIRECTION_SUMMARY.csv", frame)
    payload = {
        "contract": "topic5_leave_one_topology_out_direction_freeze_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": DIRECTION_REVISION,
        "status": "PASS" if len(frame) == 42 * 4 else "INCOMPLETE",
        "fits": int(frame.fit_id.nunique()) if len(frame) else 0,
        "directions": int(len(frame)),
        "heldout_arm_operator_read": False,
        "heldout_target_values_read": False,
    }
    atomic_write_json(DIRECTIONS / "DIRECTION_FREEZE_STATUS.json", payload)
    if payload["status"] == "PASS":
        atomic_write_json(DIRECTIONS / "DIRECTION_FREEZE_SEAL.json", {
            "sealed": True,
            "revision": DIRECTION_REVISION,
            "summary_sha256": sha256_file(DIRECTIONS / "DIRECTION_SUMMARY.csv"),
            "status_sha256": sha256_file(DIRECTIONS / "DIRECTION_FREEZE_STATUS.json"),
            "heldout_target_values_read": False,
        })
    return payload


@torch.no_grad()
def branch_nll(
    model: torch.nn.Module,
    base_hidden: np.ndarray,
    branch_hidden: np.ndarray,
    recruited: np.ndarray,
    step: np.ndarray,
    event: np.ndarray,
    ranks: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = len(base_hidden)
    base_nll = np.full((count, HORIZON + 1), np.nan, np.float32)
    branch_nll_values = np.full_like(base_nll, np.nan)
    stop_delta = np.full_like(base_nll, np.nan)
    logit_delta_norm = np.full_like(base_nll, np.nan)
    valid = np.zeros((count, HORIZON + 1), np.uint8)
    left = torch.as_tensor(base_hidden, dtype=torch.float32, device=device)
    right = torch.as_tensor(branch_hidden, dtype=torch.float32, device=device)
    current_recruited = np.asarray(recruited, dtype=np.uint8)
    for tau in range(HORIZON + 1):
        current_step = np.asarray(step, int) + tau
        if tau > 0:
            x, current_recruited, input_valid = future_input(ranks, event, current_step)
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            next_left = model._step(left, x_tensor)
            next_right = model._step(right, x_tensor)
            use_tensor = torch.as_tensor(input_valid[:, None], dtype=torch.bool, device=device)
            left = torch.where(use_tensor, next_left, left)
            right = torch.where(use_tensor, next_right, right)
        else:
            input_valid = np.ones(count, dtype=bool)
        target, _, target_valid = future_input(ranks, event, current_step + 1)
        use = input_valid & target_valid & (target.sum(axis=1) > 0)
        recruited_tensor = torch.as_tensor(current_recruited, dtype=torch.bool, device=device)
        step_tensor = torch.as_tensor(current_step, dtype=torch.long, device=device)
        left_logits, left_stop, _ = raw_logits_stop(model, left, step_tensor, recruited_tensor)
        right_logits, right_stop, _ = raw_logits_stop(model, right, step_tensor, recruited_tensor)
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        available = ~recruited_tensor
        left_value = rank_set_nll(left_logits, target_tensor, available).detach().cpu().numpy()
        right_value = rank_set_nll(right_logits, target_tensor, available).detach().cpu().numpy()
        difference = (right_logits - left_logits).detach().cpu().numpy()
        base_nll[use, tau] = left_value[use]
        branch_nll_values[use, tau] = right_value[use]
        stop_delta[use, tau] = (
            torch.sigmoid(right_stop) - torch.sigmoid(left_stop)
        ).detach().cpu().numpy()[use]
        logit_delta_norm[use, tau] = np.linalg.norm(difference[use], axis=1)
        valid[use, tau] = 1
    return {
        "base_nll": base_nll,
        "branch_nll": branch_nll_values,
        "delta_nll": branch_nll_values - base_nll,
        "stop_probability_delta": stop_delta,
        "logit_delta_norm": logit_delta_norm,
        "valid": valid,
    }


@torch.no_grad()
def response_blind_match(
    model: torch.nn.Module,
    hidden: np.ndarray,
    target_hidden: np.ndarray,
    candidates: np.ndarray,
    center: np.ndarray,
    displacement_norm: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose equal-norm directions matching immediate readout and zero-input gain."""
    h = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_hidden, dtype=torch.float32, device=device)
    zero = torch.zeros((len(hidden), model.n_contacts), dtype=torch.float32, device=device)
    base_logits = model._readout(h)
    base_step = model._step(h, zero)
    target_logits = model._readout(target)
    target_step = model._step(target, zero)
    target_output = torch.linalg.vector_norm(target_logits - base_logits, dim=1).cpu().numpy()
    target_gain = torch.linalg.vector_norm(target_step - base_step, dim=1).cpu().numpy()
    score = np.full((len(hidden), candidates.shape[1]), np.inf, dtype=np.float64)
    candidate_hidden = np.full((len(hidden), candidates.shape[1], hidden.shape[1]), np.nan, np.float64)
    for index in range(candidates.shape[1]):
        direction = candidates[:, index]
        defined = np.isfinite(direction).all(axis=1)
        if not defined.any():
            continue
        moved, _ = equal_norm_toward_center(
            hidden[defined], center[defined], direction[defined], displacement_norm[defined]
        )
        candidate_hidden[defined, index] = moved
        moved_tensor = torch.as_tensor(moved, dtype=torch.float32, device=device)
        defined_tensor = torch.as_tensor(defined, dtype=torch.bool, device=device)
        output = torch.linalg.vector_norm(
            model._readout(moved_tensor) - base_logits[defined_tensor], dim=1
        ).cpu().numpy()
        gain = torch.linalg.vector_norm(
            model._step(moved_tensor, zero[defined_tensor]) - base_step[defined_tensor], dim=1
        ).cpu().numpy()
        epsilon = 1e-8
        score[defined, index] = (
            np.abs(np.log((output + epsilon) / (target_output[defined] + epsilon)))
            + np.abs(np.log((gain + epsilon) / (target_gain[defined] + epsilon)))
        )
    selected = np.argmin(score, axis=1)
    best = candidate_hidden[np.arange(len(hidden)), selected]
    best_score = score[np.arange(len(hidden)), selected]
    invalid = ~np.isfinite(best_score)
    best[invalid] = np.nan
    selected = selected.astype(np.int16)
    selected[invalid] = -1
    return best, selected, best_score


@torch.no_grad()
def run_lesion_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    context = build_context(row, sample, eligibility, device, batch_size)
    if str(row.public_arm) not in REAL_ARMS:
        raise ValueError("lesion primary runs on real-order arms only")
    with np.load(direction_dir(str(row.fit_id), str(row.public_arm)) / "direction_contract.npz", allow_pickle=False) as source:
        direction = {name: np.asarray(source[name]) for name in source.files}
    shared = unit_vector(direction["shared_hidden_components"][0])
    c_suffix = unit_vector(direction["c_suffix_hidden_components"][0])
    q = context.q
    h = q["hidden"].astype(float)
    recruited = q["recruited"].astype(np.uint8)
    step = q["step"].astype(int)
    event = q["reference_event_index"].astype(int)
    state_rows = context.state_rows(event, step)
    center = context.conditional[state_rows].astype(float)
    replay_error = float(np.max(np.abs(h - context.states["hidden"][state_rows].astype(float))))
    if replay_error > FLOAT32_REFERENCE_REPLAY_TOLERANCE:
        raise RuntimeError(f"heldout reference replay mismatch: {replay_error}")
    n_ref = len(h)
    n_family = len(CONTROL_FAMILIES)
    n_dose = len(LESION_DOSES)
    delta_nll = np.full((n_ref, n_family, n_dose, HORIZON + 1), np.nan, np.float32)
    base_nll = np.full_like(delta_nll, np.nan)
    stop_delta = np.full_like(delta_nll, np.nan)
    logit_norm = np.full_like(delta_nll, np.nan)
    valid = np.zeros_like(delta_nll, np.uint8)
    support_checks = np.zeros((n_ref, n_family, n_dose, 3), np.uint8)
    support_distance = np.full((n_ref, n_family, n_dose), np.nan, np.float32)
    displacement_norm = np.full((n_ref, n_dose), np.nan, np.float32)
    actual_displacement_norm = np.full((n_ref, n_family, n_dose), np.nan, np.float32)
    selected_orthogonal = np.full((n_ref, n_dose), -1, np.int16)
    selected_pca = np.full((n_ref, n_dose), -1, np.int16)
    orthogonal_match_score = np.full((n_ref, n_dose), np.nan, np.float32)
    pca_match_score = np.full((n_ref, n_dose), np.nan, np.float32)
    raw_normal = q["control_directions"][:, LOCAL_NORMAL_INDICES].astype(float)
    raw_pca = q["control_directions"][:, PCA_INDICES].astype(float)
    normal = np.stack([
        orthogonalize_rows(raw_normal[:, index], shared) for index in range(raw_normal.shape[1])
    ], axis=1)
    pca = np.stack([
        orthogonalize_rows(raw_pca[:, index], shared) for index in range(raw_pca.shape[1])
    ], axis=1)
    for dose_index, dose in enumerate(LESION_DOSES):
        shared_hidden, shared_delta, projection = projection_erasure(h, center, shared, float(dose))
        norm = np.linalg.norm(shared_delta, axis=1)
        displacement_norm[:, dose_index] = norm.astype(np.float32)
        orth_hidden, orth_index, orth_score = response_blind_match(
            context.model, h, shared_hidden, normal, center, norm, device
        )
        pca_hidden, pca_index, pca_score = response_blind_match(
            context.model, h, shared_hidden, pca, center, norm, device
        )
        c_suffix_hidden, _ = equal_norm_toward_center(h, center, c_suffix, norm)
        selected_orthogonal[:, dose_index] = orth_index
        selected_pca[:, dose_index] = pca_index
        orthogonal_match_score[:, dose_index] = orth_score.astype(np.float32)
        pca_match_score[:, dose_index] = pca_score.astype(np.float32)
        branches = (shared_hidden, orth_hidden, pca_hidden, c_suffix_hidden)
        for family_index, branch in enumerate(branches):
            actual_displacement_norm[:, family_index, dose_index] = np.linalg.norm(
                branch - h, axis=1
            ).astype(np.float32)
            checks, distance = context.support(state_rows, branch)
            eligible = (
                checks.all(axis=1)
                & np.isfinite(branch).all(axis=1)
                & np.isfinite(norm)
                & (norm > 1e-8)
            )
            support_checks[:, family_index, dose_index] = checks
            support_distance[:, family_index, dose_index] = distance.astype(np.float32)
            indices = np.flatnonzero(eligible)
            if not len(indices):
                continue
            result = branch_nll(
                context.model,
                h[indices],
                branch[indices],
                recruited[indices],
                step[indices],
                event[indices],
                context.ranks,
                device,
            )
            delta_nll[indices, family_index, dose_index] = result["delta_nll"]
            base_nll[indices, family_index, dose_index] = result["base_nll"]
            stop_delta[indices, family_index, dose_index] = result["stop_probability_delta"]
            logit_norm[indices, family_index, dose_index] = result["logit_delta_norm"]
            valid[indices, family_index, dose_index] = result["valid"]
    arrays = {
        "family_names": np.asarray(CONTROL_FAMILIES),
        "doses": LESION_DOSES.astype(np.float32),
        "future_tau": np.arange(HORIZON + 1, dtype=np.int8),
        "event_index": event.astype(np.int64),
        "step": step.astype(np.int16),
        "phase_target": q["phase_target"].astype(np.float32),
        "shared_projection": ((h - center) @ shared).astype(np.float32),
        "displacement_norm": displacement_norm,
        "actual_displacement_norm": actual_displacement_norm,
        "delta_nll": delta_nll,
        "base_nll": base_nll,
        "stop_probability_delta": stop_delta,
        "logit_delta_norm": logit_norm,
        "valid": valid,
        "support_checks": support_checks,
        "support_knn_distance": support_distance,
        "selected_orthogonal_index": selected_orthogonal,
        "selected_pca_index": selected_pca,
        "orthogonal_match_score": orthogonal_match_score,
        "pca_match_score": pca_match_score,
    }
    delayed_decisions = valid[:, :, :, FUTURE_TAU]
    delayed = delayed_decisions.any(axis=3)
    norm_error = np.abs(actual_displacement_norm - displacement_norm[:, None, :])
    metrics = {
        "contract": "topic5_shared_functional_component_heldout_lesion_cell_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": LESION_REVISION,
        "status": "PASS",
        "patient": str(row.patient),
        "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "n_reference_states": int(n_ref),
        "delayed_valid_state_family_dose": int(delayed.sum()),
        "delayed_possible_state_family_dose": int(delayed.size),
        "delayed_valid_decisions": int(delayed_decisions.sum()),
        "delayed_possible_decisions": int(delayed_decisions.size),
        "shared_full_dose_delayed_valid_states": int(delayed[:, 0, -1].sum()),
        "max_control_displacement_norm_error": float(np.nanmax(norm_error)),
        "max_reference_replay_error": replay_error,
        "reference_replay_tolerance": FLOAT32_REFERENCE_REPLAY_TOLERANCE,
        "direction_contract_sha256": sha256_file(
            direction_dir(str(row.fit_id), str(row.public_arm)) / "direction_contract.npz"
        ),
        "model_hash_unchanged": context.parameter_hash_before == parameter_state_sha256(context.model),
        "decoder_hash_unchanged": context.decoder_hash_before == parameter_state_sha256(context.decoder),
        "heldout_target_read_after_direction_freeze": True,
        "state_center_definition": "TRAIN_FITTED_PHASE_CURVE_GAMMA",
        "heldout_future_field_used_in_state_center": False,
        "heldout_future_field_used_in_support_gate": False,
        "heldout_outcome_keys_dropped_before_lesion": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return arrays, metrics


def write_lesion(row: pd.Series, arrays: dict[str, np.ndarray], metrics: dict[str, object]) -> None:
    target = lesion_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    write_npz(target / "lesion_response.npz", arrays)
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True,
        "revision": LESION_REVISION,
        "response_sha256": sha256_file(target / "lesion_response.npz"),
        "metrics_sha256": sha256_file(target / "metrics.json"),
        "direction_contract_sha256": metrics["direction_contract_sha256"],
    })


def aggregate_lesion(real_manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in real_manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = lesion_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
        else:
            rows.append(json.loads((target / "metrics.json").read_text()))
    if rows:
        atomic_write_csv(LESION / "LESION_CELL_SUMMARY.csv", pd.DataFrame(rows))
    payload = {
        "contract": "topic5_shared_functional_component_heldout_lesion_execution_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": LESION_REVISION,
        "status": "PASS" if len(rows) == len(real_manifest) and not missing else "INCOMPLETE",
        "scheduled_cells": int(len(real_manifest)),
        "completed_cells": int(len(rows)),
        "delayed_valid_state_family_dose": int(sum(row["delayed_valid_state_family_dose"] for row in rows)),
        "delayed_possible_state_family_dose": int(sum(row["delayed_possible_state_family_dose"] for row in rows)),
        "delayed_valid_decisions": int(sum(row.get("delayed_valid_decisions", 0) for row in rows)),
        "delayed_possible_decisions": int(sum(row.get("delayed_possible_decisions", 0) for row in rows)),
        "model_hash_unchanged_cells": int(sum(bool(row["model_hash_unchanged"]) for row in rows)),
        "decoder_hash_unchanged_cells": int(sum(bool(row["decoder_hash_unchanged"]) for row in rows)),
        "state_center_definition": "TRAIN_FITTED_PHASE_CURVE_GAMMA",
        "heldout_future_field_used_in_state_center": False,
        "heldout_future_field_used_in_support_gate": False,
        "heldout_outcome_keys_dropped_before_lesion": True,
        "missing_count": len(missing),
        "missing_first20": missing[:20],
    }
    atomic_write_json(LESION / "LESION_EXECUTION_STATUS.json", payload)
    return payload


@torch.no_grad()
def run_subspace_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Pre-registered rank-1/2/3 cumulative sensitivity; never changes primary."""
    started = time.perf_counter()
    context = build_context(row, sample, eligibility, device, batch_size)
    if str(row.public_arm) not in REAL_ARMS:
        raise ValueError("subspace sensitivity runs on real-order arms only")
    with np.load(direction_dir(str(row.fit_id), str(row.public_arm)) / "direction_contract.npz", allow_pickle=False) as source:
        direction = {name: np.asarray(source[name]) for name in source.files}
    shared_components = direction["shared_hidden_components"].astype(float)
    c_suffix_components = direction["c_suffix_hidden_components"].astype(float)
    pca_components = context.geometry["pca_components"].astype(float).T[:3]
    q = context.q
    h = q["hidden"].astype(float)
    recruited = q["recruited"].astype(np.uint8)
    step = q["step"].astype(int)
    event = q["reference_event_index"].astype(int)
    state_rows = context.state_rows(event, step)
    center = context.conditional[state_rows].astype(float)
    ranks_kept = np.asarray([1, 2, 3], np.int8)
    family_names = np.asarray(["SHARED", "C_SUFFIX", "PCA"])
    shape = (len(h), len(family_names), len(ranks_kept), len(LESION_DOSES), HORIZON + 1)
    delta_nll = np.full(shape, np.nan, np.float32)
    valid = np.zeros(shape, np.uint8)
    support_checks = np.zeros(shape[:-1] + (3,), np.uint8)
    displacement_norm = np.full((len(h), len(ranks_kept), len(LESION_DOSES)), np.nan, np.float32)
    actual_norm = np.full(shape[:-1], np.nan, np.float32)
    for rank_index, rank in enumerate(ranks_kept):
        shared_basis = shared_components[: int(rank)]
        cs_basis = c_suffix_components[: int(rank)]
        pca_basis = pca_components[: int(rank)]
        for dose_index, dose in enumerate(LESION_DOSES):
            shared_hidden, shared_delta = subspace_projection_erasure(
                h, center, shared_basis, float(dose)
            )
            norm = np.linalg.norm(shared_delta, axis=1)
            displacement_norm[:, rank_index, dose_index] = norm.astype(np.float32)
            cs_hidden, _ = equal_norm_subspace_toward_center(h, center, cs_basis, norm)
            pca_hidden, _ = equal_norm_subspace_toward_center(h, center, pca_basis, norm)
            branches = (shared_hidden, cs_hidden, pca_hidden)
            for family_index, branch in enumerate(branches):
                actual_norm[:, family_index, rank_index, dose_index] = np.linalg.norm(
                    branch - h, axis=1
                ).astype(np.float32)
                checks, _ = context.support(state_rows, branch)
                eligible = (
                    checks.all(axis=1)
                    & np.isfinite(branch).all(axis=1)
                    & np.isfinite(norm)
                    & (norm > 1e-8)
                )
                support_checks[:, family_index, rank_index, dose_index] = checks
                indices = np.flatnonzero(eligible)
                if not len(indices):
                    continue
                result = branch_nll(
                    context.model,
                    h[indices],
                    branch[indices],
                    recruited[indices],
                    step[indices],
                    event[indices],
                    context.ranks,
                    device,
                )
                delta_nll[indices, family_index, rank_index, dose_index] = result["delta_nll"]
                valid[indices, family_index, rank_index, dose_index] = result["valid"]
    norm_error = np.abs(actual_norm - displacement_norm[:, None])
    arrays = {
        "family_names": family_names,
        "ranks": ranks_kept,
        "doses": LESION_DOSES.astype(np.float32),
        "future_tau": np.arange(HORIZON + 1, dtype=np.int8),
        "event_index": event.astype(np.int64),
        "step": step.astype(np.int16),
        "phase_target": q["phase_target"].astype(np.float32),
        "displacement_norm": displacement_norm,
        "actual_displacement_norm": actual_norm,
        "delta_nll": delta_nll,
        "valid": valid,
        "support_checks": support_checks,
    }
    delayed = np.take(valid, FUTURE_TAU, axis=-1)
    metrics = {
        "contract": "topic5_shared_subspace_sensitivity_cell_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": SUBSPACE_REVISION,
        "status": "PASS",
        "patient": str(row.patient),
        "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "n_reference_states": int(len(h)),
        "delayed_valid_decisions": int(delayed.sum()),
        "delayed_possible_decisions": int(delayed.size),
        "max_control_displacement_norm_error": float(np.nanmax(norm_error)),
        "model_hash_unchanged": context.parameter_hash_before == parameter_state_sha256(context.model),
        "decoder_hash_unchanged": context.decoder_hash_before == parameter_state_sha256(context.decoder),
        "direction_contract_sha256": sha256_file(
            direction_dir(str(row.fit_id), str(row.public_arm)) / "direction_contract.npz"
        ),
        "state_center_definition": "TRAIN_FITTED_PHASE_CURVE_GAMMA",
        "heldout_future_field_used_in_state_center": False,
        "heldout_future_field_used_in_support_gate": False,
        "heldout_outcome_keys_dropped_before_lesion": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return arrays, metrics


def write_subspace(row: pd.Series, arrays: dict[str, np.ndarray], metrics: dict[str, object]) -> None:
    target = subspace_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    write_npz(target / "subspace_response.npz", arrays)
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True,
        "revision": SUBSPACE_REVISION,
        "response_sha256": sha256_file(target / "subspace_response.npz"),
        "metrics_sha256": sha256_file(target / "metrics.json"),
        "direction_contract_sha256": metrics["direction_contract_sha256"],
    })


def aggregate_subspace(real_manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in real_manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = subspace_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
        else:
            rows.append(json.loads((target / "metrics.json").read_text()))
    if rows:
        atomic_write_csv(SUBSPACE / "SUBSPACE_CELL_SUMMARY.csv", pd.DataFrame(rows))
    payload = {
        "contract": "topic5_shared_subspace_sensitivity_execution_v0_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": SUBSPACE_REVISION,
        "status": "PASS" if len(rows) == len(real_manifest) and not missing else "INCOMPLETE",
        "scheduled_cells": int(len(real_manifest)),
        "completed_cells": int(len(rows)),
        "delayed_valid_decisions": int(sum(row["delayed_valid_decisions"] for row in rows)),
        "delayed_possible_decisions": int(sum(row["delayed_possible_decisions"] for row in rows)),
        "model_hash_unchanged_cells": int(sum(bool(row["model_hash_unchanged"]) for row in rows)),
        "decoder_hash_unchanged_cells": int(sum(bool(row["decoder_hash_unchanged"]) for row in rows)),
        "state_center_definition": "TRAIN_FITTED_PHASE_CURVE_GAMMA",
        "heldout_future_field_used_in_state_center": False,
        "heldout_future_field_used_in_support_gate": False,
        "heldout_outcome_keys_dropped_before_lesion": True,
        "missing_count": len(missing),
        "missing_first20": missing[:20],
    }
    atomic_write_json(SUBSPACE / "SUBSPACE_EXECUTION_STATUS.json", payload)
    return payload


def select_manifest(
    manifest: pd.DataFrame,
    cell_key: str | None,
    limit: int | None,
    shard_index: int | None = None,
    n_shards: int | None = None,
) -> pd.DataFrame:
    selected = manifest.copy()
    if cell_key:
        fit, arm, seed_text = cell_key.split("/")
        selected = selected[
            selected.fit_id.eq(fit)
            & selected.public_arm.eq(arm)
            & selected.seed.eq(int(seed_text.removeprefix("seed")))
        ]
    elif limit is not None:
        selected = selected.iloc[:limit]
    if shard_index is not None or n_shards is not None:
        if shard_index is None or n_shards is None:
            raise ValueError("--shard-index and --n-shards must be provided together")
        if n_shards <= 0 or not 0 <= shard_index < n_shards:
            raise ValueError("invalid shard specification")
        selected = selected.iloc[shard_index::n_shards]
    return selected


def run_cells(
    selected: pd.DataFrame,
    sample: pd.DataFrame,
    eligibility: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    force: bool,
    done_path: Callable[[pd.Series], Path],
    runner: Callable,
    writer: Callable,
) -> list[dict[str, object]]:
    failures = []
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        target = done_path(row)
        if (target / "DONE.json").is_file() and not force:
            print(f"skip {position}/{len(selected)} {row.fit_id}/{row.public_arm}/seed{row.seed}", flush=True)
            continue
        try:
            arrays, metrics = runner(
                row,
                sample[sample.fit_id.eq(row.fit_id)].copy(),
                eligibility.loc[row.fit_id],
                device,
                batch_size,
            )
            writer(row, arrays, metrics)
            print(
                f"done {position}/{len(selected)} {row.fit_id}/{row.public_arm}/seed{row.seed} "
                f"{metrics['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:
            failure = {
                "fit_id": str(row.fit_id),
                "public_arm": str(row.public_arm),
                "seed": int(row.seed),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            target.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target / "FAILURE.json", failure)
            print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("train-operator", "freeze-directions", "lesion", "subspace-lesion")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cell-key")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--n-shards", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sharded = args.shard_index is not None or args.n_shards is not None
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    sample = pd.read_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id")
    if args.stage == "freeze-directions":
        status = freeze_directions(manifest)
        print(json.dumps(status, indent=2))
        if status["status"] != "PASS":
            raise SystemExit(1)
        return
    device = torch.device(args.device)
    if args.stage == "train-operator":
        selected = select_manifest(
            manifest, args.cell_key, args.limit, args.shard_index, args.n_shards
        )
        failures = run_cells(
            selected, sample, eligibility, device, args.batch_size, args.force,
            train_operator_dir, run_train_operator_cell, write_train_operator,
        )
        status = (
            {"status": "SHARD_COMPLETE", "completed_in_this_process": int(len(selected))}
            if sharded else aggregate_train_operator(manifest)
        )
    elif args.stage == "lesion":
        real_manifest = manifest[manifest.public_arm.isin(REAL_ARMS)].copy()
        selected = select_manifest(
            real_manifest, args.cell_key, args.limit, args.shard_index, args.n_shards
        )
        failures = run_cells(
            selected, sample, eligibility, device, args.batch_size, args.force,
            lesion_dir, run_lesion_cell, write_lesion,
        )
        status = (
            {"status": "SHARD_COMPLETE", "completed_in_this_process": int(len(selected))}
            if sharded else aggregate_lesion(real_manifest)
        )
    else:
        real_manifest = manifest[manifest.public_arm.isin(REAL_ARMS)].copy()
        selected = select_manifest(
            real_manifest, args.cell_key, args.limit, args.shard_index, args.n_shards
        )
        failures = run_cells(
            selected, sample, eligibility, device, args.batch_size, args.force,
            subspace_dir, run_subspace_cell, write_subspace,
        )
        status = (
            {"status": "SHARD_COMPLETE", "completed_in_this_process": int(len(selected))}
            if sharded else aggregate_subspace(real_manifest)
        )
    print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
