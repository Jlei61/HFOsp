#!/usr/bin/env python3
"""Run teacher-forced tangent transport and transverse contraction for C2."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    load_frozen_cell,
    parse_bool,
    parameter_state_sha256,
    sha256_file,
)
from src.topic5_latent_pass1_v0_2 import (  # noqa: E402
    build_future_field_data,
    interpolate_phase_vectors,
    leaky_rnn_jvp,
)
from scripts.run_topic5_latent_pass1_v0_2 import (  # noqa: E402
    ANALYSIS_REVISION,
    OUT,
    PARENT,
    SYSTEM,
    cell_dir,
    replay_states,
)


TRANSPORT = OUT / "dynamical_transport"
TRANSPORT_REVISION = "TRANSPORT_R1_CONDITIONAL_PF_MANIFOLD"
PHASE_TARGETS = (0.25, 0.50, 0.75)
NORMAL_DIRECTIONS = 8
_FIELD_CACHE: dict[str, object] = {}


def stable_normals(
    *, key: str, progress: np.ndarray, field: np.ndarray, count: int
) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    basis: list[np.ndarray] = []
    for candidate in (progress, field):
        if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1e-10:
            value = candidate.astype(float).copy()
            for existing in basis:
                value -= existing * float(np.dot(existing, value))
            norm = float(np.linalg.norm(value))
            if norm > 1e-10:
                basis.append(value / norm)
    normals: list[np.ndarray] = []
    attempts = 0
    while len(normals) < count and attempts < count * 20:
        attempts += 1
        value = rng.normal(size=len(progress))
        for existing in (*basis, *normals):
            value -= existing * float(np.dot(existing, value))
        norm = float(np.linalg.norm(value))
        if norm > 1e-8:
            normals.append(value / norm)
    if len(normals) != count:
        raise RuntimeError("could not construct local-normal directions")
    return np.stack(normals)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    output = values / np.maximum(norms, 1e-12)
    output[norms[:, 0] <= 1e-10] = np.nan
    return output


def run_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_before = parameter_state_sha256(model)
    decoder_before = parameter_state_sha256(decoder)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"])
        split = np.asarray(events["split"])
        full_train_mode = np.asarray(events["full_train_mode"])
    if str(row.fit_id) not in _FIELD_CACHE:
        _FIELD_CACHE[str(row.fit_id)] = build_future_field_data(
            ranks, split, full_train_mode,
            positive_mode=int(eligibility["positive_mode"]),
            negative_mode=int(eligibility["negative_mode"]),
            tier=str(eligibility["status"]),
            shuffle_key=str(row.fit_id),
        )
    field_data = _FIELD_CACHE[str(row.fit_id)]
    selected = sample[sample["pass2_reference_event"].map(parse_bool)].copy()
    states = replay_states(model, ranks, split, selected, device, batch_size)
    lookup = {
        (int(event), int(step)): index
        for index, (event, step) in enumerate(zip(states["event_index"], states["step"]))
    }
    pass1 = cell_dir(row)
    pass1_metrics = json.loads((pass1 / "metrics.json").read_text())
    if pass1_metrics.get("analysis_revision") != ANALYSIS_REVISION:
        raise RuntimeError("Pass 1 analysis revision mismatch")
    with np.load(pass1 / "geometry_arrays.npz", allow_pickle=False) as arrays:
        grid = np.asarray(arrays["phase_grid"], dtype=float)
        progress_grid = np.asarray(arrays["progress_axes_raw"], dtype=float)
        field_grid = np.asarray(arrays["field_axes_raw"], dtype=float)
        gamma_grid = np.asarray(arrays["gamma_raw"], dtype=float)
    conditional_path = pass1 / "conditional_manifold_arrays.npz"
    if not conditional_path.is_file():
        raise RuntimeError("conditional manifold augmentation missing")
    with np.load(conditional_path, allow_pickle=False) as conditional:
        field_direction_grid = np.asarray(conditional["field_direction_raw"], dtype=float)

    references: list[dict[str, object]] = []
    for event in selected["event_array_index"].to_numpy(dtype=int):
        length = int(np.max(ranks[event][ranks[event] >= 0])) + 1
        if length < 2:
            continue
        allowed = np.arange(length - 1)
        phases = allowed / (length - 1)
        for target_phase in PHASE_TARGETS:
            step = int(allowed[np.argmin(np.abs(phases - target_phase))])
            current = lookup[(int(event), step)]
            nxt = lookup[(int(event), step + 1)]
            references.append({
                "event_array_index": int(event),
                "phase_target": float(target_phase),
                "step": step,
                "current_row": current,
                "next_row": nxt,
                "phase": float(step / (length - 1)),
                "phase_next": float((step + 1) / (length - 1)),
                "phase_abs_error": float(abs(step / (length - 1) - target_phase)),
            })
    if not references:
        raise RuntimeError("no teacher-forced transport references")
    current_rows = np.asarray([item["current_row"] for item in references], dtype=int)
    next_rows = np.asarray([item["next_row"] for item in references], dtype=int)
    phases = np.asarray([item["phase"] for item in references], dtype=float)
    phases_next = np.asarray([item["phase_next"] for item in references], dtype=float)
    progress = unit_rows(interpolate_phase_vectors(grid, progress_grid, phases))
    progress_next = unit_rows(interpolate_phase_vectors(grid, progress_grid, phases_next))
    field = unit_rows(interpolate_phase_vectors(grid, field_grid, phases))
    field_next = unit_rows(interpolate_phase_vectors(grid, field_grid, phases_next))
    field_valid = np.isfinite(field).all(axis=1) & np.isfinite(field_next).all(axis=1)
    field_for_jvp = field.copy()
    field_for_jvp[~field_valid] = 0.0
    gamma = interpolate_phase_vectors(grid, gamma_grid, phases)
    gamma_next = interpolate_phase_vectors(grid, gamma_grid, phases_next)
    branch = interpolate_phase_vectors(grid, field_direction_grid, phases)
    branch_next = interpolate_phase_vectors(grid, field_direction_grid, phases_next)
    event_u = field_data.event_coordinate_z[np.asarray([
        int(item["event_array_index"]) for item in references
    ])]
    conditional_gamma = gamma + event_u[:, None] * branch
    conditional_gamma_next = gamma_next + event_u[:, None] * branch_next
    h = states["hidden"][current_rows].astype(np.float64)
    h_next = states["hidden"][next_rows].astype(np.float64)
    x_next = states["x"][next_rows].astype(np.float32)
    normals = np.stack([
        stable_normals(
            key=(
                f"{row.fit_id}/{row.public_arm}/{int(row.seed)}/"
                f"{references[index]['event_array_index']}/{references[index]['phase_target']}"
            ),
            progress=progress[index],
            field=field[index],
            count=NORMAL_DIRECTIONS,
        )
        for index in range(len(references))
    ])
    directions = np.concatenate([
        progress[:, None, :], field_for_jvp[:, None, :], normals
    ], axis=1)
    with torch.no_grad():
        transported = leaky_rnn_jvp(
            model,
            torch.from_numpy(h.astype(np.float32)).to(device),
            torch.from_numpy(x_next).to(device),
            torch.from_numpy(directions.astype(np.float32)).to(device),
        ).cpu().numpy().astype(np.float64)

    rows: list[dict[str, object]] = []
    for index, reference in enumerate(references):
        jp = transported[index, 0]
        jf = transported[index, 1]
        normal_gain = np.linalg.norm(transported[index, 2:], axis=1)
        progress_gain = float(np.linalg.norm(jp))
        field_gain = float(np.linalg.norm(jf)) if field_valid[index] else float("nan")
        median_normal = float(np.median(normal_gain))
        current_distance = float(np.linalg.norm(h[index] - gamma[index]) / np.sqrt(h.shape[1]))
        next_distance = float(
            np.linalg.norm(h_next[index] - gamma_next[index]) / np.sqrt(h.shape[1])
        )
        conditional_distance = float(
            np.linalg.norm(h[index] - conditional_gamma[index]) / np.sqrt(h.shape[1])
        )
        conditional_next_distance = float(
            np.linalg.norm(h_next[index] - conditional_gamma_next[index]) / np.sqrt(h.shape[1])
        )
        rows.append({
            "patient": str(row.patient),
            "fit_id": str(row.fit_id),
            "geometry_view": str(row.geometry_view),
            "public_arm": str(row.public_arm),
            "seed": int(row.seed),
            "field_axis_tier": str(pass1_metrics["field_axis_tier"]),
            "canonical_ab": bool(pass1_metrics["canonical_ab"]),
            **reference,
            "progress_transport_cosine": cosine(jp, progress_next[index]),
            "field_transport_cosine": (
                cosine(jf, field_next[index]) if field_valid[index] else float("nan")
            ),
            "progress_gain": progress_gain,
            "field_gain": field_gain,
            "normal_gain_median": median_normal,
            "transverse_contraction": 1.0 - median_normal,
            "progress_gain_minus_normal": progress_gain - median_normal,
            "field_gain_minus_normal": field_gain - median_normal,
            "distance_to_progress_curve": current_distance,
            "next_distance_to_progress_curve": next_distance,
            "event_to_curve_convergence": current_distance - next_distance,
            "distance_to_PF_manifold": conditional_distance,
            "next_distance_to_PF_manifold": conditional_next_distance,
            "event_to_PF_manifold_convergence": conditional_distance - conditional_next_distance,
            "finite": bool(
                np.isfinite(jp).all()
                and (not field_valid[index] or np.isfinite(jf).all())
                and np.isfinite(normal_gain).all()
            ),
            "target_values_read": False,
        })
    frame = pd.DataFrame(rows).drop(columns=["current_row", "next_row"])
    if not frame["finite"].all():
        raise RuntimeError("nonfinite transport branch")
    metrics = {
        "contract": "topic5_latent_transport_cell_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "transport_revision": TRANSPORT_REVISION,
        "status": "PASS",
        "patient": str(row.patient),
        "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "n_reference_events": int(frame["event_array_index"].nunique()),
        "n_reference_states": int(len(frame)),
        "max_phase_abs_error": float(frame["phase_abs_error"].max()),
        "model_hash_unchanged": model_before == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_before == parameter_state_sha256(decoder),
        "pass1_arrays_sha256": sha256_file(pass1 / "geometry_arrays.npz"),
        "conditional_manifold_sha256": sha256_file(conditional_path),
        "target_values_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return frame, metrics


def transport_dir(row: pd.Series) -> Path:
    return TRANSPORT / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_cell(row: pd.Series, frame: pd.DataFrame, metrics: dict[str, object]) -> None:
    target = transport_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(target / "transport.csv", frame)
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True,
        "transport_sha256": sha256_file(target / "transport.csv"),
        "metrics_sha256": sha256_file(target / "metrics.json"),
        "target_values_read": False,
    })


def aggregate(manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = transport_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
            continue
        frame = pd.read_csv(target / "transport.csv")
        summary = frame.groupby("phase_target", as_index=False)[[
            "progress_transport_cosine", "field_transport_cosine", "progress_gain",
            "field_gain", "normal_gain_median", "transverse_contraction",
            "progress_gain_minus_normal", "field_gain_minus_normal",
            "event_to_curve_convergence",
            "event_to_PF_manifold_convergence",
        ]].median(numeric_only=True)
        summary.insert(0, "patient", item.patient)
        summary.insert(1, "fit_id", item.fit_id)
        summary.insert(2, "geometry_view", item.geometry_view)
        summary.insert(3, "public_arm", item.public_arm)
        summary.insert(4, "seed", int(item.seed))
        metrics = json.loads((target / "metrics.json").read_text())
        summary["field_axis_tier"] = pd.read_csv(target / "transport.csv", nrows=1)[
            "field_axis_tier"
        ].iloc[0]
        summary["canonical_ab"] = bool(
            pd.read_csv(target / "transport.csv", nrows=1)["canonical_ab"].iloc[0]
        )
        rows.append(summary)
    if rows:
        atomic_write_csv(TRANSPORT / "TRANSPORT_CELL_PHASE_SUMMARY.csv", pd.concat(rows, ignore_index=True))
    payload = {
        "contract": "topic5_latent_transport_execution_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "transport_revision": TRANSPORT_REVISION,
        "status": "PASS" if len(rows) == 630 and not missing else "INCOMPLETE",
        "completed_cells": len(rows),
        "scheduled_cells": 630,
        "missing_count": len(missing),
        "missing_first20": missing[:20],
        "closed_loop_consistency": "PENDING_PASS2",
        "target_values_read": False,
    }
    atomic_write_json(TRANSPORT / "TRANSPORT_EXECUTION_STATUS.json", payload)
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
        raise RuntimeError("Pass 1 audit must pass before transport")
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
        target = transport_dir(row)
        if (target / "DONE.json").is_file() and not args.force:
            print(f"skip {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed}", flush=True)
            continue
        try:
            frame, metrics = run_cell(
                row,
                samples[samples["fit_id"].eq(row.fit_id)].copy(),
                eligibility.loc[row.fit_id],
                device,
                args.batch_size,
            )
            write_cell(row, frame, metrics)
            print(
                f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} "
                f"refs={len(frame)} {metrics['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:
            failures.append({
                "fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed),
                "error_type": type(error).__name__, "error": str(error),
            })
            atomic_write_json(target / "FAILURE.json", failures[-1])
            print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate(manifest_all)
    print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
