#!/usr/bin/env python3
"""Synthetic recovery for the long-window cumulative-exposure instrument."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_long_total import (
    LONG_TOTAL_REVISION,
    build_long_window_design,
    decoder_readout,
    fit_decoder_space_edge,
    intercept_operator,
    metric_contrast,
    occurrence_block_variation,
    predict_state,
    state_prediction_metrics,
)


def _linear(weight: np.ndarray):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
    return layer


def _model(dim: int, rng: np.random.Generator):
    return SimpleNamespace(
        state_timing=_linear(rng.normal(size=(1, dim))),
        state_size=_linear(rng.normal(size=(5, dim))),
        state_contact=_linear(rng.normal(size=(4, dim))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=contract.RESULT_ROOT / "t2_long_total_effect/synthetic/recovery.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(20260826)
    dim = 4
    n_event = 26000
    interval = rng.lognormal(mean=np.log(2.1), sigma=0.45, size=n_event)
    event_time = np.cumsum(interval).astype(np.float64)
    event_split = np.where(np.arange(n_event) < 18000, 0, 1).astype(np.int8)
    segment = np.zeros(n_event, dtype=np.int64)
    innovation = rng.normal(size=n_event)
    # Stable flow with both decay and rotation.
    matrix = np.diag([-0.020, -0.010, -0.005, -0.002])
    matrix[0, 1] = 0.018; matrix[1, 0] = -0.018
    mu = np.zeros(dim)
    state = np.zeros((n_event, dim), dtype=np.float64)
    for index in range(1, n_event):
        state[index] = 0.996 * state[index - 1] + rng.normal(scale=0.03, size=dim)
    design = build_long_window_design(
        event_time, event_split, segment, state, innovation, matrix, mu,
        window_kind="event_count_10000", scale_events=10000,
        delay_events=1000, coverage_start=np.asarray([event_time[0] - 1.0]),
    )
    model = _model(dim, rng)
    validation = np.flatnonzero(design.split == 1)
    train = design.split == 0
    offset = intercept_operator(design)
    scenarios = {}
    # Exposure-driven truths use the real operator.  The exposure-free truths
    # do not: they carry a state offset and a slow drift that no IED sequence
    # produced.  They are the calibration that a white-noise null cannot give,
    # because the exposure arms own a free state-space intercept.
    exposure_truth = {
        "mixed_true_edge": np.asarray([
            0.22, -0.16, 0.10, 0.05, 0.18, 0.12, -0.20, 0.08
        ]),
        "occurrence_only": np.asarray([
            0.22, -0.16, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0
        ]),
        "null": np.zeros(2 * dim),
    }
    n_window = len(design.split)
    ramp = np.linspace(0.0, 1.0, n_window)[:, None]
    exposure_free_truth = {
        "null_with_state_offset": np.broadcast_to(
            np.asarray([0.5, -0.3, 0.2, 0.1]), (n_window, dim)
        ).copy(),
        "null_with_slow_drift": ramp * np.asarray([0.5, -0.3, 0.2, 0.1]),
    }
    plan = [
        (label, value, True) for label, value in exposure_truth.items()
    ] + [
        (label, value, False) for label, value in exposure_free_truth.items()
    ]
    for label, truth_value, uses_exposure in plan:
        noise = rng.normal(scale=0.025, size=design.natural_state.shape)
        if uses_exposure:
            theta_true = np.asarray(truth_value, dtype=np.float64)
            target_delta = np.einsum(
                "ndp,p->nd", design.real_operator, theta_true
            ) + noise
        else:
            theta_true = None
            target_delta = np.asarray(truth_value, dtype=np.float64) + noise
        target = design.natural_state + target_delta
        readout = decoder_readout(model, target_delta, train)
        theta_real, real_fit = fit_decoder_space_edge(
            design.real_operator, target_delta, design.split, readout
        )
        theta_delayed, delayed_fit = fit_decoder_space_edge(
            design.delayed_operator, target_delta, design.split, readout
        )
        theta_intercept, intercept_fit = fit_decoder_space_edge(
            offset, target_delta, design.split, readout
        )
        predicted = {
            "no_edge": design.natural_state,
            "intercept_matched": design.natural_state + np.einsum(
                "ndp,p->nd", offset, theta_intercept
            ),
            "real": design.natural_state + np.einsum(
                "ndp,p->nd", design.real_operator, theta_real
            ),
            "delayed": design.natural_state + np.einsum(
                "ndp,p->nd", design.delayed_operator, theta_delayed
            ),
        }
        metrics = {
            arm: state_prediction_metrics(value, target, validation, readout)
            for arm, value in predicted.items()
        }
        cosine = None
        if theta_true is not None:
            denominator = float(
                np.linalg.norm(theta_true) * np.linalg.norm(theta_real)
            )
            if denominator > 1e-12:
                cosine = float(np.dot(theta_true, theta_real) / denominator)
        scenarios[label] = {
            "truth_uses_exposure": bool(uses_exposure),
            "theta_true": (
                theta_true.tolist() if theta_true is not None else None
            ),
            "theta_real_fit": theta_real.tolist(),
            "theta_delayed_fit": theta_delayed.tolist(),
            "theta_intercept_fit": theta_intercept.tolist(),
            "theta_cosine": cosine,
            "metrics": metrics,
            "real_minus_intercept_matched": metric_contrast(
                metrics["real"], metrics["intercept_matched"]
            ),
            "real_minus_delayed": metric_contrast(
                metrics["real"], metrics["delayed"]
            ),
            "delayed_minus_intercept_matched": metric_contrast(
                metrics["delayed"], metrics["intercept_matched"]
            ),
            "real_minus_no_edge": metric_contrast(
                metrics["real"], metrics["no_edge"]
            ),
            "intercept_minus_no_edge": metric_contrast(
                metrics["intercept_matched"], metrics["no_edge"]
            ),
            "real_fit": real_fit,
            "delayed_fit": delayed_fit,
            "intercept_fit": intercept_fit,
        }
    mixed = scenarios["mixed_true_edge"]
    occurrence = scenarios["occurrence_only"]
    null = scenarios["null"]
    offset_null = scenarios["null_with_state_offset"]
    drift_null = scenarios["null_with_slow_drift"]

    def total(scenario: dict, key: str) -> float:
        return float(scenario[key]["decoder_total_equal_block_mse"])

    acceptance = {
        "mixed_real_beats_intercept_matched": total(
            mixed, "real_minus_intercept_matched") < 0,
        "mixed_real_beats_delayed": total(mixed, "real_minus_delayed") < 0,
        "mixed_theta_direction_recovered": (
            mixed["theta_cosine"] is not None and mixed["theta_cosine"] > 0.8
        ),
        "occurrence_real_and_delayed_both_beat_intercept_matched": (
            total(occurrence, "real_minus_intercept_matched") < 0
            and total(occurrence, "delayed_minus_intercept_matched") < 0
        ),
        "null_real_gain_small": abs(
            total(null, "real_minus_intercept_matched")) < 0.02,
        # The gates the shipped v1 acceptance set could not fail: an
        # exposure-free target that merely has a mean or a slow drift.  These
        # are one-sided on purpose.  The failure mode being guarded is a false
        # *favourable* (negative) contrast; the real arm paying a small
        # out-of-sample price for its extra parameters is expected.
        "offset_null_real_does_not_beat_intercept": (
            total(offset_null, "real_minus_intercept_matched") > -0.02
        ),
        "offset_null_no_load_timing_gain": (
            total(offset_null, "real_minus_delayed") > -0.02
        ),
        "drift_null_real_does_not_beat_intercept": (
            total(drift_null, "real_minus_intercept_matched") > -0.02
        ),
        "drift_null_no_load_timing_gain": (
            total(drift_null, "real_minus_delayed") > -0.02
        ),
        # Bad-data regression: if this ever stops holding the free-intercept
        # artefact has changed and the demotion of real-minus-no-edge must be
        # re-derived rather than inherited.
        "offset_null_reproduces_intercept_artefact": (
            total(offset_null, "real_minus_no_edge") < -1.0
        ),
    }
    intercept_artefact = {
        "null_with_state_offset_real_minus_no_edge": total(
            offset_null, "real_minus_no_edge"),
        "null_with_slow_drift_real_minus_no_edge": total(
            drift_null, "real_minus_no_edge"),
        "note": (
            "an exposure-free target beats no-edge by a large margin purely "
            "through the free intercept, so real-minus-no-edge must never be "
            "read as cumulative-exposure evidence"
        ),
    }
    payload = {
        "status": "COMPLETE" if all(acceptance.values()) else "FAIL",
        "revision": LONG_TOTAL_REVISION,
        "seed": 20260826,
        "n_events": int(n_event),
        "train_windows": int((design.split == 0).sum()),
        "validation_windows": int((design.split == 1).sum()),
        "median_n10000_hours": float(np.median(design.duration_hours)),
        "occurrence_block_variation": occurrence_block_variation(
            design.real_operator, validation
        ),
        "scenarios": scenarios,
        "acceptance": acceptance,
        "intercept_artefact_demonstration": intercept_artefact,
        "sealed_opened": False,
        "claim_boundary": "instrument calibration only; no human biological claim",
    }
    contract.atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
