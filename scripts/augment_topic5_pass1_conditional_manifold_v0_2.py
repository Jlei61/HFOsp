#!/usr/bin/env python3
"""Recover the frozen PF branch magnitude b(s) for conditional convergence."""
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
    atomic_write_json,
    load_frozen_cell,
    parameter_state_sha256,
    sha256_file,
)
from src.topic5_latent_pass1_v0_2 import (  # noqa: E402
    build_future_field_data,
    event_first_phase_balanced_weights,
    spline_basis,
    weighted_ridge,
)
from scripts.run_topic5_latent_pass1_v0_2 import (  # noqa: E402
    ANALYSIS_REVISION,
    OUT,
    PARENT,
    SYSTEM,
    cell_dir,
    replay_states,
)


_FIELD_CACHE: dict[str, object] = {}


def augment_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_before = parameter_state_sha256(model)
    decoder_before = parameter_state_sha256(decoder)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"])
        split_event = np.asarray(events["split"])
        full_train_mode = np.asarray(events["full_train_mode"])
    if str(row.fit_id) not in _FIELD_CACHE:
        _FIELD_CACHE[str(row.fit_id)] = build_future_field_data(
            ranks, split_event, full_train_mode,
            positive_mode=int(eligibility["positive_mode"]),
            negative_mode=int(eligibility["negative_mode"]),
            tier=str(eligibility["status"]),
            shuffle_key=str(row.fit_id),
        )
    field = _FIELD_CACHE[str(row.fit_id)]
    train_sample = sample[sample["split"].eq(0)].copy()
    states = replay_states(model, ranks, split_event, train_sample, device, batch_size)
    weights = event_first_phase_balanced_weights(
        states["event_index"], states["split"], states["phase_bin"]
    )
    target = cell_dir(row)
    metrics = json.loads((target / "metrics.json").read_text())
    if metrics.get("analysis_revision") != ANALYSIS_REVISION:
        raise RuntimeError("Pass 1 revision mismatch")
    selection = metrics["model_selection"]["PF"]
    knots = tuple(float(value) for value in selection["knots"])
    alpha = float(selection["alpha"])
    with np.load(target / "geometry_arrays.npz", allow_pickle=False) as arrays:
        center = np.asarray(arrays["robust_center"], dtype=float)
        scale = np.asarray(arrays["robust_scale"], dtype=float)
        grid = np.asarray(arrays["phase_grid"], dtype=float)
        gamma_saved = np.asarray(arrays["gamma_raw"], dtype=float)
    y = (states["hidden"].astype(float) - center[None, :]) / scale[None, :]
    u = field.event_coordinate_z[states["event_index"]]
    basis = spline_basis(states["phase"], knots)
    design = np.column_stack([basis, u[:, None] * basis])
    train = states["split"] == 0
    coefficient = weighted_ridge(design[train], y[train], weights[train], alpha)
    grid_basis = spline_basis(grid, knots)
    n_basis = grid_basis.shape[1]
    gamma = center[None, :] + (grid_basis @ coefficient[:n_basis]) * scale[None, :]
    field_direction = (grid_basis @ coefficient[n_basis:]) * scale[None, :]
    gamma_max_abs = float(np.max(np.abs(gamma - gamma_saved)))
    if gamma_max_abs > 5e-5:
        raise RuntimeError(f"reconstructed PF gamma drift: {gamma_max_abs}")
    temporary = target / "conditional_manifold_arrays.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            phase_grid=grid.astype(np.float32),
            field_direction_raw=field_direction.astype(np.float32),
            field_coordinate_train_mean=np.asarray([field.train_coordinate_mean], np.float64),
            field_coordinate_train_scale=np.asarray([field.train_coordinate_scale], np.float64),
        )
    temporary.replace(target / "conditional_manifold_arrays.npz")
    payload = {
        "contract": "topic5_pass1_conditional_manifold_augmentation_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "status": "PASS",
        "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "PF_knots": list(knots),
        "PF_alpha": alpha,
        "gamma_reconstruction_max_abs": gamma_max_abs,
        "arrays_sha256": sha256_file(target / "conditional_manifold_arrays.npz"),
        "model_hash_unchanged": model_before == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_before == parameter_state_sha256(decoder),
        "target_values_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(target / "CONDITIONAL_MANIFOLD_DONE.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cell-key")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    audit = json.loads((SYSTEM / "PASS1_AUDIT.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("Pass 1 audit must pass")
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    manifest = manifest_all.copy()
    if args.cell_key:
        fit, arm, seed_text = args.cell_key.split("/")
        seed = int(seed_text.removeprefix("seed"))
        manifest = manifest[
            manifest["fit_id"].eq(fit)
            & manifest["public_arm"].eq(arm)
            & manifest["seed"].eq(seed)
        ]
    elif args.limit is not None:
        manifest = manifest.iloc[:args.limit]
    samples = pd.read_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id")
    device = torch.device(args.device)
    failures = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = cell_dir(row)
        if (target / "CONDITIONAL_MANIFOLD_DONE.json").is_file() and not args.force:
            print(f"skip {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed}", flush=True)
            continue
        try:
            payload = augment_cell(
                row,
                samples[samples["fit_id"].eq(row.fit_id)].copy(),
                eligibility.loc[row.fit_id],
                device,
                args.batch_size,
            )
            print(
                f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} "
                f"{payload['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:
            failures.append({
                "fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed),
                "error_type": type(error).__name__, "error": str(error),
            })
            print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    completed = sum(
        (cell_dir(pd.Series(item._asdict())) / "CONDITIONAL_MANIFOLD_DONE.json").is_file()
        for item in manifest_all.itertuples(index=False)
    )
    status = {
        "contract": "topic5_pass1_conditional_manifold_execution_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "status": "PASS" if completed == 630 and not failures else "INCOMPLETE",
        "completed_cells": completed,
        "scheduled_cells": 630,
        "run_failures": failures,
        "target_values_read": False,
    }
    atomic_write_json(SYSTEM / "CONDITIONAL_MANIFOLD_EXECUTION_STATUS.json", status)
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
