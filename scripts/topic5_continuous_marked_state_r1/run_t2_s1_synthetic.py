#!/usr/bin/env python3
"""Synthetic truth/negative-control calibration for the minimal T2-S1 edge."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_s1 import (
    OneStepDesign, SignedExposureEdge, evaluate_edge, fit_edge,
)


REVISION = "t2_s1_synthetic_signed_timing_mark_recovery_v1"


class SyntheticGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(2))

    def matrix(self) -> torch.Tensor:
        return self.mu.new_zeros((2, 2))


class SyntheticT1(torch.nn.Module):
    """Known frozen readout: state[0] controls timing, state[1] the subset."""
    def __init__(self) -> None:
        super().__init__()
        self.state = torch.nn.Module()
        self.state.generator = SyntheticGenerator()

    def timing_log_rate(self, history, state):
        return state[:, 0]

    def mark_terms(self, history, state, group_ids, group_count):
        chosen_first = (group_ids[:, 0] == 0).to(state.dtype)
        subset = -torch.nn.functional.binary_cross_entropy_with_logits(
            state[:, 1], chosen_first, reduction="none"
        )
        zero = state[:, 0] * 0.0
        size_step = torch.stack([zero, zero], dim=1)
        subset_step = torch.stack([subset, zero], dim=1)
        return SimpleNamespace(
            event_log_prob=subset,
            group_size_log_prob=zero,
            subset_log_prob=subset,
            group_size_step_log_prob=size_step,
            subset_step_log_prob=subset_step,
            active_step=torch.ones((len(state), 2), dtype=torch.bool, device=state.device),
            select_step=torch.tensor(
                [True, False], dtype=torch.bool, device=state.device
            ).expand(len(state), 2),
        )


def simulate(seed: int, *, timing_edge: float, mark_edge: float,
             n: int = 4000) -> OneStepDesign:
    rng = np.random.default_rng(int(seed))
    exposure = rng.normal(size=n).astype(np.float32)
    log_rate = timing_edge * exposure
    interval = rng.exponential(1.0 / np.exp(log_rate)).astype(np.float32)
    probability = 1.0 / (1.0 + np.exp(-mark_edge * exposure))
    first = rng.uniform(size=n) < probability
    group_ids = np.where(
        first[:, None], np.asarray([[0, -1]]), np.asarray([[-1, 0]])
    ).astype(np.int64)
    n_train = int(round(0.65 * n))
    result = OneStepDesign(
        current_state=np.zeros((n, 2), dtype=np.float32),
        current_index=np.arange(n, dtype=np.int64),
        next_history=np.zeros((n, 3), dtype=np.float32),
        next_group_ids=group_ids,
        next_group_count=np.ones(n, dtype=np.int64),
        delta_minutes=interval / 60.0,
        quadrature_delta_minutes=np.tile(interval[:, None] / 120.0, (1, 4)),
        quadrature_history=np.zeros((n, 4, 3), dtype=np.float32),
        quadrature_weight_seconds=np.tile(interval[:, None] / 4.0, (1, 4)),
        exposure=exposure,
        split=np.r_[
            np.zeros(n_train, dtype=np.int8), np.ones(n - n_train, dtype=np.int8)
        ],
    )
    result.validate()
    return result


def shuffled_design(value: OneStepDesign, seed: int) -> OneStepDesign:
    rng = np.random.default_rng(int(seed))
    exposure = value.exposure.copy()
    for code in (0, 1):
        rows = np.flatnonzero(value.split == code)
        exposure[rows] = exposure[rng.permutation(rows)]
    return replace(value, exposure=exposure)


def run_one(seed: int, timing_edge: float, mark_edge: float, *,
            n: int = 4000) -> dict:
    model = SyntheticT1()
    design = simulate(seed, timing_edge=timing_edge, mark_edge=mark_edge, n=n)
    shuffled = shuffled_design(design, seed + 1000)
    null = SignedExposureEdge(2)
    null_metrics = asdict(evaluate_edge(
        model, null, design, split="validation", device="cpu"
    ))
    fitted, fit_audit = fit_edge(
        model, design, device="cpu", seed=seed, epochs=30,
        learning_rate=0.03, batch_size=512,
    )
    shuffled_edge, shuffled_audit = fit_edge(
        model, shuffled, device="cpu", seed=seed, epochs=30,
        learning_rate=0.03, batch_size=512,
    )
    fitted_metrics = asdict(evaluate_edge(
        model, fitted, design, split="validation", device="cpu"
    ))
    shuffled_metrics = asdict(evaluate_edge(
        model, shuffled_edge, shuffled, split="validation", device="cpu"
    ))
    return {
        "seed": seed,
        "truth": {"timing_edge": timing_edge, "mark_edge": mark_edge},
        "recovered_vector": fitted.vector.detach().cpu().tolist(),
        "shuffled_vector": shuffled_edge.vector.detach().cpu().tolist(),
        "selected_epoch": fit_audit["selected_epoch"],
        "shuffled_selected_epoch": shuffled_audit["selected_epoch"],
        "validation": {
            "no_edge": null_metrics,
            "real": fitted_metrics,
            "shuffled": shuffled_metrics,
            "real_minus_no_edge_joint": (
                fitted_metrics["joint_nll_per_event"]
                - null_metrics["joint_nll_per_event"]
            ),
            "shuffled_minus_no_edge_joint": (
                shuffled_metrics["joint_nll_per_event"]
                - null_metrics["joint_nll_per_event"]
            ),
        },
    }


def main() -> None:
    truth = [run_one(seed, 0.55, 0.80) for seed in (0, 1, 2)]
    negative = [run_one(seed + 10, 0.0, 0.0) for seed in (0, 1, 2)]
    ladder = {}
    for level in (0.02, 0.05, 0.10, 0.20):
        values = [
            run_one(100 + seed, level, level, n=2200)
            for seed in (0, 1, 2)
        ]
        gain = [
            row["validation"]["real_minus_no_edge_joint"] for row in values
        ]
        shuffled_gain = [
            row["validation"]["shuffled_minus_no_edge_joint"] for row in values
        ]
        ladder[str(level)] = {
            "n_total": 2200,
            "n_validation": 770,
            "runs": values,
            "real_gain_median": float(np.median(gain)),
            "shuffled_gain_median": float(np.median(shuffled_gain)),
            "n_real_favourable": int(sum(value < 0 for value in gain)),
            "n_direction_recovered": int(sum(
                row["recovered_vector"][0] > 0
                and row["recovered_vector"][1] > 0 for row in values
            )),
        }
    truth_gain = [row["validation"]["real_minus_no_edge_joint"] for row in truth]
    truth_shuffle = [
        row["validation"]["shuffled_minus_no_edge_joint"] for row in truth
    ]
    negative_gain = [
        row["validation"]["real_minus_no_edge_joint"] for row in negative
    ]
    summary = {
        "status": "COMPLETE",
        "revision": REVISION,
        "truth_runs": truth,
        "negative_runs": negative,
        "small_edge_sensitivity_ladder": ladder,
        "recovery": {
            "truth_joint_gain_median": float(np.median(truth_gain)),
            "truth_shuffled_gain_median": float(np.median(truth_shuffle)),
            "negative_joint_gain_median": float(np.median(negative_gain)),
            "truth_direction_recovered": int(sum(
                row["recovered_vector"][0] > 0
                and row["recovered_vector"][1] > 0 for row in truth
            )),
            "truth_seeds": 3,
        },
        "acceptance": {
            "truth_gain_negative_all_seeds": bool(all(value < 0 for value in truth_gain)),
            "truth_beats_shuffled_median": bool(
                np.median(truth_gain) < np.median(truth_shuffle)
            ),
            "truth_direction_all_seeds": bool(all(
                row["recovered_vector"][0] > 0
                and row["recovered_vector"][1] > 0 for row in truth
            )),
        },
        "sealed_opened": False,
        "claim_boundary": "instrument calibration only; no human biological claim",
    }
    output = contract.RESULT_ROOT / "t2_s1_long_scale" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(output / "synthetic_recovery.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
