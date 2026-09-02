#!/usr/bin/env python3
"""Compare unperturbed closed-loop and teacher-forced latent transitions for C2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv, atomic_write_json, load_frozen_cell,
    parameter_state_sha256, sha256_file,
)
from src.topic5_latent_pass1_v0_2 import interpolate_phase_vectors  # noqa: E402
from src.topic5_latent_response_v0_2 import deterministic_sets, raw_logits_stop  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import (  # noqa: E402
    FREEZE_REVISION, REFERENCE, reference_dir,
)
from scripts.run_topic5_axis_perturbations_v0_2 import future_input  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import (  # noqa: E402
    ANALYSIS_REVISION, OUT, PARENT, SYSTEM, cell_dir,
)


TRANSITION = OUT / "dynamical_transport" / "closed_loop_transition"
TRANSITION_REVISION = "C2_CLOSED_LOOP_R0_FROZEN_DECODER_PROJECTED_TRANSITION"
HORIZON = 3


def transition_dir(row: pd.Series) -> Path:
    return TRANSITION / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def latent_coordinates(
    hidden: np.ndarray,
    phase: np.ndarray,
    event_u: np.ndarray,
    grid: np.ndarray,
    gamma_grid: np.ndarray,
    branch_grid: np.ndarray,
    progress_grid: np.ndarray,
    field_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gamma = interpolate_phase_vectors(grid, gamma_grid, phase)
    branch = interpolate_phase_vectors(grid, branch_grid, phase)
    center = gamma + event_u[:, None] * branch
    progress = interpolate_phase_vectors(grid, progress_grid, phase)
    field = interpolate_phase_vectors(grid, field_grid, phase)
    residual = hidden - center
    z = np.stack([
        np.einsum("ij,ij->i", residual, progress),
        np.einsum("ij,ij->i", residual, field),
    ], axis=-1)
    distance = np.linalg.norm(residual, axis=1) / np.sqrt(hidden.shape[1])
    return z, distance


@torch.no_grad()
def run_cell(row: pd.Series, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash, decoder_hash = parameter_state_sha256(model), parameter_state_sha256(decoder)
    frozen = reference_dir(row) / "reference_contract.npz"
    with np.load(frozen, allow_pickle=False) as source:
        q = {key: np.asarray(source[key]) for key in source.files}
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as source:
        ranks = np.asarray(source["ranks"])
    pass1 = cell_dir(row)
    with np.load(pass1 / "geometry_arrays.npz", allow_pickle=False) as source:
        geometry = {key: np.asarray(source[key]) for key in source.files}
    with np.load(pass1 / "conditional_manifold_arrays.npz", allow_pickle=False) as source:
        branch_grid = np.asarray(source["field_direction_raw"], dtype=float)
    event = q["reference_event_index"].astype(int)
    step = q["step"].astype(int)
    h0 = q["hidden"].astype(np.float64)
    recruited0 = q["recruited"].astype(np.uint8)
    event_u = q["event_u"].astype(np.float64)
    rows = ranks[event]
    event_length = np.max(np.where(rows >= 0, rows, -1), axis=1) + 1
    phase_future = np.stack([
        np.minimum((step + tau) / np.maximum(event_length - 1, 1), 1.0)
        for tau in range(HORIZON + 1)
    ], axis=1)
    z0, distance0 = latent_coordinates(
        h0, q["phase"].astype(float), event_u,
        geometry["phase_grid"], geometry["gamma_raw"], branch_grid,
        geometry["progress_axes_raw"], geometry["field_axes_raw"],
    )

    tf_hidden = torch.as_tensor(h0, dtype=torch.float32, device=device)
    cl_hidden = torch.as_tensor(h0, dtype=torch.float32, device=device)
    cl_recruited = torch.as_tensor(recruited0, dtype=torch.bool, device=device)
    cl_step = torch.as_tensor(step, dtype=torch.long, device=device)
    cl_active = torch.ones(len(h0), dtype=torch.bool, device=device)
    tf_z = np.full((len(h0), HORIZON, 2), np.nan, np.float32)
    cl_z = np.full_like(tf_z, np.nan)
    tf_distance = np.full((len(h0), HORIZON), np.nan, np.float32)
    cl_distance = np.full_like(tf_distance, np.nan)
    tf_valid = np.zeros((len(h0), HORIZON), np.uint8)
    cl_valid = np.zeros_like(tf_valid)
    generated_sets = np.zeros((len(h0), HORIZON, int(row.n_contacts)), np.uint8)
    for tau in range(1, HORIZON + 1):
        x_true, _, valid_true = future_input(ranks, event, step + tau)
        x_true_tensor = torch.as_tensor(x_true, dtype=torch.float32, device=device)
        next_tf = model._step(tf_hidden, x_true_tensor)
        tf_hidden = torch.where(
            torch.as_tensor(valid_true[:, None], dtype=torch.bool, device=device),
            next_tf, tf_hidden,
        )
        logits, stop, features = raw_logits_stop(model, cl_hidden, cl_step, cl_recruited)
        continuing = cl_active & (stop < 0.5) & ~cl_recruited.all(-1)
        sizes = decoder(features).argmax(-1) + 1
        next_set = deterministic_sets(
            logits.detach().cpu().numpy(), cl_recruited.detach().cpu().numpy(),
            sizes.detach().cpu().numpy(), continuing.detach().cpu().numpy(),
        )
        generated_sets[:, tau - 1] = next_set
        next_set_tensor = torch.as_tensor(next_set, dtype=torch.float32, device=device)
        next_cl = model._step(cl_hidden, next_set_tensor)
        cl_hidden = torch.where(continuing[:, None], next_cl, cl_hidden)
        cl_recruited = cl_recruited | next_set_tensor.bool()
        cl_step = torch.where(continuing, cl_step + 1, cl_step)
        cl_active = continuing
        tf_values = tf_hidden.detach().cpu().numpy().astype(np.float64)
        cl_values = cl_hidden.detach().cpu().numpy().astype(np.float64)
        tf_coordinate, tf_dist = latent_coordinates(
            tf_values, phase_future[:, tau], event_u,
            geometry["phase_grid"], geometry["gamma_raw"], branch_grid,
            geometry["progress_axes_raw"], geometry["field_axes_raw"],
        )
        cl_coordinate, cl_dist = latent_coordinates(
            cl_values, phase_future[:, tau], event_u,
            geometry["phase_grid"], geometry["gamma_raw"], branch_grid,
            geometry["progress_axes_raw"], geometry["field_axes_raw"],
        )
        active_np = cl_active.detach().cpu().numpy()
        tf_z[valid_true, tau - 1] = tf_coordinate[valid_true] - z0[valid_true]
        tf_distance[valid_true, tau - 1] = tf_dist[valid_true] - distance0[valid_true]
        tf_valid[valid_true, tau - 1] = 1
        cl_z[active_np, tau - 1] = cl_coordinate[active_np] - z0[active_np]
        cl_distance[active_np, tau - 1] = cl_dist[active_np] - distance0[active_np]
        cl_valid[active_np, tau - 1] = 1
    arrays = {
        "event_index": event.astype(np.int64), "step": step.astype(np.int16),
        "phase": q["phase"].astype(np.float32), "phase_target": q["phase_target"].astype(np.float32),
        "teacher_forced_delta_z": tf_z, "closed_loop_delta_z": cl_z,
        "teacher_forced_delta_manifold_distance": tf_distance,
        "closed_loop_delta_manifold_distance": cl_distance,
        "teacher_forced_valid": tf_valid, "closed_loop_valid": cl_valid,
        "generated_sets": generated_sets,
    }
    metrics = {
        "contract": "topic5_closed_loop_transition_cell_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION, "freeze_revision": FREEZE_REVISION,
        "transition_revision": TRANSITION_REVISION, "status": "PASS",
        "patient": str(row.patient), "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm), "seed": int(row.seed),
        "n_reference_states": int(len(h0)),
        "teacher_forced_valid_transitions": int(tf_valid.sum()),
        "closed_loop_valid_transitions": int(cl_valid.sum()),
        "model_hash_unchanged": model_hash == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_hash == parameter_state_sha256(decoder),
        "reference_contract_sha256": sha256_file(frozen),
        "target_values_read": False, "elapsed_seconds": time.perf_counter() - started,
    }
    return arrays, metrics


def write_cell(row: pd.Series, arrays: dict[str, np.ndarray], metrics: dict[str, object]) -> None:
    target = transition_dir(row); target.mkdir(parents=True, exist_ok=True)
    write_npz(target / "transition.npz", arrays)
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True, "transition_revision": TRANSITION_REVISION,
        "transition_sha256": sha256_file(target / "transition.npz"),
        "metrics_sha256": sha256_file(target / "metrics.json"), "target_values_read": False,
    })


def aggregate(manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict()); target = transition_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}"); continue
        rows.append(json.loads((target / "metrics.json").read_text()))
    if rows: atomic_write_csv(TRANSITION / "CLOSED_LOOP_TRANSITION_CELL_SUMMARY.csv", pd.DataFrame(rows))
    payload = {
        "contract": "topic5_closed_loop_transition_execution_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transition_revision": TRANSITION_REVISION,
        "status": "PASS" if len(rows) == 630 and not missing else "INCOMPLETE",
        "scheduled_cells": 630, "completed_cells": len(rows),
        "teacher_forced_valid_transitions": int(sum(row["teacher_forced_valid_transitions"] for row in rows)),
        "closed_loop_valid_transitions": int(sum(row["closed_loop_valid_transitions"] for row in rows)),
        "missing_count": len(missing), "missing_first20": missing[:20], "target_values_read": False,
    }
    atomic_write_json(TRANSITION / "CLOSED_LOOP_TRANSITION_STATUS.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    parser.add_argument("--cell-key"); parser.add_argument("--limit", type=int); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if json.loads((REFERENCE / "REFERENCE_FREEZE_AUDIT.json").read_text()).get("status") != "PASS":
        raise RuntimeError("reference freeze audit must pass")
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv"); manifest = manifest_all.copy()
    if args.cell_key:
        fit, arm, seed_text = args.cell_key.split("/")
        manifest = manifest[manifest["fit_id"].eq(fit) & manifest["public_arm"].eq(arm) & manifest["seed"].eq(int(seed_text.removeprefix("seed")))]
    elif args.limit is not None: manifest = manifest.iloc[:args.limit]
    device = torch.device(args.device); failures = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = transition_dir(row)
        if (target / "DONE.json").is_file() and not args.force: continue
        try:
            arrays, metrics = run_cell(row, device); write_cell(row, arrays, metrics)
            print(f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} {metrics['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures.append({"fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed), "error_type": type(error).__name__, "error": str(error)})
            atomic_write_json(target / "FAILURE.json", failures[-1]); print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate(manifest_all); print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
