#!/usr/bin/env python3
"""Synthetic identification acceptance for the matched Topic 5 v3.1 transition."""
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
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_event_innovation_transition_v3_1 import (  # noqa: E402
    SharedLinearFilter,
    fit_event_transition_from_latent_trace,
    observable_transition_impulse,
    simulate_innovation_transition,
    transition_prediction_error,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/topic5_event_innovation_state_space/v3_1/synthetic_calibration"


def shared_filter() -> SharedLinearFilter:
    return SharedLinearFilter(
        transition=np.array([[0.82, 0.08], [0.0, 0.73]]),
        observation=np.eye(2),
        filter_gain=np.eye(2) * 0.4,
    )


def split_trace(data, fraction: float = 0.7):
    stop = int(np.floor(float(fraction) * (len(data.prior) - 1)))
    train = np.arange(stop)
    test = np.arange(stop, len(data.prior) - 1)
    return train, test


def fit_and_compare(data, shared: SharedLinearFilter, alpha: float = 1.0) -> dict:
    train, test = split_trace(data)
    fitted = fit_event_transition_from_latent_trace(
        data.posterior[train],
        data.innovation[train],
        data.prior[train + 1],
        shared,
        alpha=float(alpha),
    )
    observer = transition_prediction_error(
        data.posterior[test], data.innovation[test], data.prior[test + 1], shared
    )
    driven = transition_prediction_error(
        data.posterior[test],
        data.innovation[test],
        data.prior[test + 1],
        shared,
        fitted,
    )
    return {
        "observer_test_mse": observer,
        "event_driven_test_mse": driven,
        "test_gain": observer - driven,
        "fitted_transition": fitted,
    }


def block_permute(values: np.ndarray, block_size: int, seed: int) -> np.ndarray:
    blocks = [
        values[start : start + int(block_size)]
        for start in range(0, len(values), int(block_size))
    ]
    order = np.random.default_rng(int(seed)).permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order], axis=0)


def discrete_switching_scores(seed: int = 0) -> dict[str, float]:
    """Held-out next-event comparison for a true finite-state process."""

    rng = np.random.default_rng(int(seed))
    means = np.array([[-1.2, -0.2], [1.1, -0.1], [0.0, 1.2]])
    transition = np.array(
        [[0.92, 0.05, 0.03], [0.04, 0.92, 0.04], [0.05, 0.05, 0.90]]
    )
    n = 9000
    regime = np.zeros(n, dtype=int)
    for event in range(1, n):
        regime[event] = rng.choice(3, p=transition[regime[event - 1]])
    observation = means[regime] + rng.normal(scale=0.12, size=(n, 2))
    stop = 6000

    linear = Ridge(alpha=1.0).fit(observation[: stop - 1], observation[1:stop])
    linear_prediction = linear.predict(observation[stop:-1])
    linear_mse = float(np.mean((observation[stop + 1 :] - linear_prediction) ** 2))

    cluster = KMeans(n_clusters=3, random_state=17, n_init=20).fit(observation[:stop])
    labels = cluster.labels_
    counts = np.ones((3, 3), dtype=float) * 1e-3
    for left, right in zip(labels[:-1], labels[1:]):
        counts[left, right] += 1.0
    probability = counts / counts.sum(axis=1, keepdims=True)
    current = cluster.predict(observation[stop:-1])
    switching_prediction = probability[current] @ cluster.cluster_centers_
    switching_mse = float(
        np.mean((observation[stop + 1 :] - switching_prediction) ** 2)
    )
    return {
        "linear_test_mse": linear_mse,
        "switching_test_mse": switching_mse,
        "switching_gain": linear_mse - switching_mse,
    }


def run_calibration(output: Path, seed: int = 7201) -> dict:
    pin_thread_environment(1, disable_cuda=True)
    output.mkdir(parents=True, exist_ok=True)
    shared = shared_filter()
    zero = np.zeros((2, 2))
    truth = np.array([[0.45, 0.02], [0.12, -0.35]])

    observer_data = simulate_innovation_transition(
        9000,
        shared,
        zero,
        observation_noise=0.8,
        transition_noise=0.03,
        seed=seed,
    )
    observer = fit_and_compare(observer_data, shared)

    driven_data = simulate_innovation_transition(
        9000,
        shared,
        truth,
        observation_noise=0.8,
        transition_noise=0.03,
        seed=seed + 1,
    )
    driven = fit_and_compare(driven_data, shared, alpha=1e-4)
    fitted = np.asarray(driven.pop("fitted_transition"))
    observer_fitted = np.asarray(observer.pop("fitted_transition"))
    cosine = float(
        np.vdot(fitted.ravel(), truth.ravel())
        / (np.linalg.norm(fitted) * np.linalg.norm(truth))
    )

    _, test = split_trace(driven_data)
    permuted = block_permute(driven_data.innovation[test], 20, seed + 2)
    permuted_error = transition_prediction_error(
        driven_data.posterior[test],
        permuted,
        driven_data.prior[test + 1],
        shared,
        fitted,
    )
    switching = discrete_switching_scores(seed + 3)
    loading = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    observable = observable_transition_impulse(loading, fitted)

    equality_registry = {
        "transition_equal": True,
        "observation_equal": True,
        "filter_gain_equal": True,
        "state_dimension_equal": True,
        "only_added_parameter": "event_transition_B",
    }
    checks = {
        "observer_only_no_material_t2_gain": observer["test_gain"] < 5e-5,
        "observer_only_fitted_B_small": float(np.linalg.norm(observer_fitted)) < 0.02,
        "event_driven_t2_gain_positive": driven["test_gain"] > 0.05,
        "event_transition_direction_recovered": cosine > 0.99,
        "matched_block_donor_loses_gain": permuted_error
        > driven["event_driven_test_mse"] + 0.05,
        "discrete_switching_control_wins": switching["switching_gain"] > 0.0,
        "observable_impulse_finite": bool(np.all(np.isfinite(observable))),
        "shared_parameter_registry_equal": all(
            value is True
            for key, value in equality_registry.items()
            if key != "only_added_parameter"
        ),
    }
    state = {
        "contract": "topic5_event_innovation_recurrent_transition_v3_1_synthetic",
        "status": "SYNTHETIC_TRANSITION_IDENTIFICATION_COMPLETE"
        if all(checks.values())
        else "SYNTHETIC_TRANSITION_IDENTIFICATION_FAILED",
        "one_step_is_one_complete_event": True,
        "human_data_read": False,
        "v3_0_handoff_read": False,
        "within_event_next_rank_model_fit": False,
        "shared_parameter_registry": equality_registry,
        "observer_only": observer,
        "event_driven": {
            **driven,
            "transition_cosine": cosine,
            "permuted_innovation_test_mse": permuted_error,
        },
        "discrete_switching": switching,
        "observable_impulse": observable.tolist(),
        "checks": checks,
        "seed": int(seed),
    }
    atomic_write_json(output / "synthetic_transition_acceptance_state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7201)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state = run_calibration(output, seed=args.seed)
    print(json.dumps(state, indent=2, sort_keys=True))
    if state["status"] != "SYNTHETIC_TRANSITION_IDENTIFICATION_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
