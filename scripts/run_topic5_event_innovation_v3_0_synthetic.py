#!/usr/bin/env python3
"""Synthetic acceptance for Topic 5 v3.0 impulse and accumulation.

The synthetic clock is event indexed: one step is one complete event.  This
runner never loads human data and is intentionally small enough for one CPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_event_innovation_v3_0 import (  # noqa: E402
    RankStateBasis,
    fit_local_projection,
    observable_impulse,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "results/topic5_event_innovation_impulse_response/v3_0/synthetic_calibration"
)


def simulate_event_indexed_state(
    n_events: int,
    transition: np.ndarray,
    impulse: np.ndarray,
    *,
    state_noise: float,
    measurement_noise: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate an event-indexed state with a known innovation update."""

    a = np.asarray(transition, dtype=float)
    b = np.asarray(impulse, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("transition must be square")
    if b.ndim != 2 or b.shape[0] != a.shape[0]:
        raise ValueError("impulse/state dimension mismatch")
    rng = np.random.default_rng(int(seed))
    n = int(n_events)
    if n < 100:
        raise ValueError("synthetic calibration requires at least 100 events")
    state = np.zeros((n + 1, a.shape[0]), dtype=float)
    innovation = rng.normal(size=(n, b.shape[1]))
    for event in range(n):
        state[event + 1] = a @ state[event] + b @ innovation[event]
        state[event + 1] += rng.normal(scale=float(state_noise), size=a.shape[0])
    return {
        "pre": state[:-1]
        + rng.normal(scale=float(measurement_noise), size=state[:-1].shape),
        "future": state[1:]
        + rng.normal(scale=float(measurement_noise), size=state[1:].shape),
        "innovation": innovation
        + rng.normal(scale=float(measurement_noise), size=innovation.shape),
        "latent_pre": state[:-1],
        "latent_future": state[1:],
        "true_innovation": innovation,
    }


def split_rows(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_stop = int(np.floor(0.6 * n_rows))
    validation_stop = int(np.floor(0.8 * n_rows))
    return (
        np.arange(0, train_stop),
        np.arange(train_stop, validation_stop),
        np.arange(validation_stop, n_rows),
    )


def mse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2))


def fit_and_score_local_projection(
    data: dict[str, np.ndarray], *, alpha: float
) -> tuple[object, dict[str, float]]:
    train, validation, test = split_rows(len(data["pre"]))
    candidates = [0.0, float(alpha), 10.0 * float(alpha)]
    fitted = []
    for candidate in candidates:
        model = fit_local_projection(
            data["pre"][train],
            data["future"][train],
            data["innovation"][train],
            alpha=candidate,
        )
        full = model.predict(data["pre"][validation], data["innovation"][validation])
        fitted.append((mse(data["future"][validation], full), candidate, model))
    _, selected_alpha, model = min(fitted, key=lambda item: (item[0], item[1]))
    full = model.predict(data["pre"][test], data["innovation"][test])
    autonomous = model.predict(
        data["pre"][test], np.zeros_like(data["innovation"][test])
    )
    return model, {
        "selected_alpha": float(selected_alpha),
        "test_full_mse": mse(data["future"][test], full),
        "test_autonomous_mse": mse(data["future"][test], autonomous),
        "test_gain": mse(data["future"][test], autonomous)
        - mse(data["future"][test], full),
    }


def block_permute(values: np.ndarray, block_size: int, seed: int) -> np.ndarray:
    array = np.asarray(values)
    blocks = [
        array[start : start + int(block_size)]
        for start in range(0, len(array), int(block_size))
    ]
    order = np.random.default_rng(int(seed)).permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order], axis=0)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def controlled_accumulation(
    transition: np.ndarray,
    impulse: np.ndarray,
    window: int,
) -> dict[str, float]:
    """Compare aligned and cancelling innovations under the same dynamics."""

    a = np.asarray(transition, dtype=float)
    b = np.asarray(impulse, dtype=float)
    direction = np.zeros(b.shape[1], dtype=float)
    direction[0] = 1.0

    def rollout(signs: np.ndarray) -> tuple[float, float]:
        state = np.zeros(a.shape[0], dtype=float)
        for sign in signs:
            state = a @ state + b @ (float(sign) * direction)
        immediate = float(np.linalg.norm(state))
        for _ in range(20):
            state = a @ state
        return immediate, float(np.linalg.norm(state))

    aligned = np.ones(int(window), dtype=float)
    cancelling = np.where(np.arange(int(window)) % 2 == 0, 1.0, -1.0)
    aligned_now, aligned_later = rollout(aligned)
    cancelling_now, cancelling_later = rollout(cancelling)
    return {
        "window": int(window),
        "aligned_displacement": aligned_now,
        "cancelling_displacement": cancelling_now,
        "aligned_minus_cancelling": aligned_now - cancelling_now,
        "persistent_displacement_h20": aligned_later,
    }


def run_calibration(output: Path, seed: int = 7103) -> dict:
    pin_thread_environment(1, disable_cuda=True)
    output.mkdir(parents=True, exist_ok=True)
    transition = np.diag([0.94, 0.86, 0.76])
    truth = np.array(
        [[0.35, 0.08, 0.00], [0.04, -0.28, 0.06], [0.02, 0.00, 0.20]]
    )
    rng = np.random.default_rng(int(seed))
    loadings, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    basis = RankStateBasis(
        backbone=np.linspace(0.1, 0.9, 12),
        loadings=loadings,
        singular_values=np.ones(3),
    )

    driven = simulate_event_indexed_state(
        8000,
        transition,
        truth,
        state_noise=0.04,
        measurement_noise=0.03,
        seed=seed,
    )
    fitted, driven_score = fit_and_score_local_projection(driven, alpha=1.0)
    observable_truth = basis.loadings @ truth
    observable_fit = observable_impulse(basis, fitted.impulse)
    driven_score["observable_impulse_cosine"] = cosine_similarity(
        observable_truth, observable_fit
    )

    null = simulate_event_indexed_state(
        8000,
        transition,
        np.zeros_like(truth),
        state_noise=0.04,
        measurement_noise=0.03,
        seed=seed + 1,
    )
    null_fit, null_score = fit_and_score_local_projection(null, alpha=1.0)
    null_score["fitted_impulse_norm"] = float(np.linalg.norm(null_fit.impulse))

    shifted = dict(driven)
    shifted["innovation"] = block_permute(
        driven["innovation"], block_size=20, seed=seed + 2
    )
    _, shifted_score = fit_and_score_local_projection(shifted, alpha=1.0)

    accumulation = [
        controlled_accumulation(transition, truth, window)
        for window in (5, 10, 20, 40)
    ]
    pd.DataFrame(accumulation).to_csv(
        output / "synthetic_accumulation_recovery.csv", index=False
    )
    pd.DataFrame(
        [
            {"condition": "event_driven", **driven_score},
            {"condition": "autonomous_null", **null_score},
            {"condition": "block_permuted_innovation", **shifted_score},
        ]
    ).to_csv(output / "synthetic_impulse_recovery.csv", index=False)

    checks = {
        "event_driven_gain_positive": driven_score["test_gain"] > 0.05,
        "observable_impulse_recovered": driven_score["observable_impulse_cosine"] > 0.95,
        "autonomous_null_no_material_gain": null_score["test_gain"] < 0.002,
        "autonomous_null_small_impulse": null_score["fitted_impulse_norm"] < 0.05,
        "true_pairing_beats_block_permutation": driven_score["test_gain"]
        > shifted_score["test_gain"] + 0.03,
        "alignment_exceeds_cancellation_all_windows": all(
            row["aligned_minus_cancelling"] > 0 for row in accumulation
        ),
        "aligned_dose_is_monotone": all(
            accumulation[index + 1]["aligned_displacement"]
            > accumulation[index]["aligned_displacement"]
            for index in range(len(accumulation) - 1)
        ),
        "persistence_detected": all(
            row["persistent_displacement_h20"] > 0 for row in accumulation
        ),
    }
    state = {
        "contract": "topic5_event_innovation_impulse_response_v3_0_synthetic",
        "status": "SYNTHETIC_IDENTIFIABILITY_COMPLETE"
        if all(checks.values())
        else "SYNTHETIC_IDENTIFIABILITY_FAILED",
        "one_step_is_one_complete_event": True,
        "human_data_read": False,
        "within_event_next_rank_model_fit": False,
        "checks": checks,
        "event_driven": driven_score,
        "autonomous_null": null_score,
        "block_permuted": shifted_score,
        "accumulation": accumulation,
        "seed": int(seed),
    }
    atomic_write_json(output / "synthetic_identifiability_state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7103)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state = run_calibration(output, seed=args.seed)
    print(json.dumps(state, indent=2, sort_keys=True))
    if state["status"] != "SYNTHETIC_IDENTIFIABILITY_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
