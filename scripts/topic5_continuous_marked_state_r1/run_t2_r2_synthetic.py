#!/usr/bin/env python3
"""Positive, zero and reversed-sign calibration for T2-R2.0."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    T2_R2_REVISION,
    ExposureEdge,
    edge_estimability_audit,
    evaluate_r2_edge,
    exponential_event_exposure,
    fit_r2_edge,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import OneStepDesign


class SyntheticGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(2))

    def matrix(self) -> torch.Tensor:
        return self.mu.new_zeros((2, 2))


class SyntheticT1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state = torch.nn.Module()
        self.state.generator = SyntheticGenerator()

    def timing_log_rate(self, history, state):
        return state[:, 0]

    def mark_terms(self, history, state, group_ids, group_count):
        n = len(state)
        zero = state[:, 0] * 0
        step = zero[:, None].expand(n, 2)
        return SimpleNamespace(
            event_log_prob=zero,
            group_size_log_prob=zero,
            subset_log_prob=zero,
            group_size_step_log_prob=step,
            subset_step_log_prob=step,
            active_step=torch.ones((n, 2), dtype=torch.bool, device=state.device),
            select_step=torch.tensor([True, False], device=state.device).expand(n, 2),
        )


def make_design(exposure: np.ndarray, interval: np.ndarray,
                split: np.ndarray) -> OneStepDesign:
    n = len(exposure)
    return OneStepDesign(
        current_state=np.zeros((n, 2), dtype=np.float32),
        current_index=np.arange(n, dtype=np.int64),
        next_history=np.zeros((n, 3), dtype=np.float32),
        next_group_ids=np.zeros((n, 1), dtype=np.int64),
        next_group_count=np.ones(n, dtype=np.int64),
        delta_minutes=np.asarray(interval / 60.0, dtype=np.float32),
        quadrature_delta_minutes=np.tile(
            np.asarray(interval[:, None] / 120.0, dtype=np.float32), (1, 4)
        ),
        quadrature_history=np.zeros((n, 4, 3), dtype=np.float32),
        quadrature_weight_seconds=np.tile(
            np.asarray(interval[:, None] / 4.0, dtype=np.float32), (1, 4)
        ),
        exposure=np.asarray(exposure, dtype=np.float32),
        split=np.asarray(split, dtype=np.int8),
    )


def one_truth(truth: float, seed: int, *, device: str) -> dict:
    rng = np.random.default_rng(10_000 + int(seed))
    n_total = 4200
    innovation = rng.normal(size=n_total).astype(np.float32)
    accumulated, eligible, exposure_audit = exponential_event_exposure(
        innovation, np.zeros(n_total, dtype=np.int64), 100
    )
    take = np.flatnonzero(eligible)
    accumulated = accumulated[take]
    innovation = innovation[take]
    n = len(take)
    split = np.r_[
        np.zeros(int(.7 * n), dtype=np.int8),
        np.ones(n - int(.7 * n), dtype=np.int8),
    ]
    train = split == 0
    scale = float(accumulated[train].std())
    real = accumulated / scale
    # A donor history separated by 1,500 events cannot overlap the effective
    # 5N history used by the human placebo.
    placebo = np.roll(real, 1500)
    current = innovation / float(innovation[train].std())
    rate = np.exp(float(truth) * .30 * real)
    interval = rng.exponential(1.0 / rate).astype(np.float32)
    exposures = {
        "real_cumulative": real,
        "state_matched_placebo": placebo,
        "current_event_only": current,
    }
    model = SyntheticT1().to(device)
    null_design = make_design(np.zeros_like(real), interval, split)
    null_edge = ExposureEdge(2, 1).to(device).eval()
    metrics = {
        "no_edge": asdict(evaluate_r2_edge(
            model, null_edge, null_design, split="validation", device=device
        ))
    }
    fits = {}
    vectors = {}
    audits = {}
    for label, value in exposures.items():
        design = make_design(value, interval, split)
        audits[label] = edge_estimability_audit(model, design, device=device)
        edge, fit = fit_r2_edge(
            model, design, device=device, seed=seed, epochs=30,
            learning_rate=.03, batch_size=512,
        )
        fits[label] = fit
        vectors[label] = edge.matrix.detach().cpu().numpy().tolist()
        metrics[label] = asdict(evaluate_r2_edge(
            model, edge, design, split="validation", device=device
        ))
    return {
        "truth": float(truth),
        "seed": int(seed),
        "n_train": int(train.sum()),
        "n_validation": int((~train).sum()),
        "exposure": exposure_audit,
        "fits": fits,
        "edge_matrices": vectors,
        "estimability": audits,
        "validation": metrics,
        "contrasts": {
            label: float(
                metrics["real_cumulative"]["joint_nll_per_event"]
                - metrics[label]["joint_nll_per_event"]
            ) for label in (
                "no_edge", "state_matched_placebo", "current_event_only"
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "t2_r2",
    )
    args = parser.parse_args()
    rows = [
        one_truth(truth, seed, device=args.device)
        for truth in (.8, 0.0, -.8) for seed in (0, 1, 2)
    ]
    by_truth = {}
    for truth in (.8, 0.0, -.8):
        take = [row for row in rows if row["truth"] == truth]
        edge = np.asarray([
            row["edge_matrices"]["real_cumulative"][0][0] for row in take
        ])
        by_truth[str(truth)] = {
            "median_real_edge": float(np.median(edge)),
            "real_edge_signs": np.sign(edge).astype(int).tolist(),
            "real_minus_placebo_median": float(np.median([
                row["contrasts"]["state_matched_placebo"] for row in take
            ])),
            "real_minus_current_median": float(np.median([
                row["contrasts"]["current_event_only"] for row in take
            ])),
            "selected_epochs": [
                row["fits"]["real_cumulative"]["selected_epoch"] for row in take
            ],
        }
    passed = {
        "positive_sign": by_truth["0.8"]["median_real_edge"] > 0,
        "positive_beats_placebo": by_truth["0.8"]["real_minus_placebo_median"] < 0,
        "positive_beats_current": by_truth["0.8"]["real_minus_current_median"] < 0,
        "reversed_sign": by_truth["-0.8"]["median_real_edge"] < 0,
        "reversed_beats_placebo": by_truth["-0.8"]["real_minus_placebo_median"] < 0,
        "zero_does_not_beat_placebo": (
            by_truth["0.0"]["real_minus_placebo_median"] >= 0
        ),
        "zero_has_no_consistent_direction": abs(sum(
            by_truth["0.0"]["real_edge_signs"]
        )) < 3,
        "zero_edge_is_small_relative_to_truth": abs(
            by_truth["0.0"]["median_real_edge"]
        ) < .2 * min(
            abs(by_truth["0.8"]["median_real_edge"]),
            abs(by_truth["-0.8"]["median_real_edge"]),
        ),
    }
    result = {
        "status": "COMPLETE" if all(passed.values()) else "FAIL",
        "revision": T2_R2_REVISION,
        "rows": rows,
        "by_truth": by_truth,
        "criteria": passed,
        "all_criteria_pass": bool(all(passed.values())),
        "sealed_opened": False,
        "source_hashes": {
            "t2_r2": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/t2_r2.py"
            ),
            "runner": contract.sha256_file(Path(__file__)),
        },
    }
    output = args.output_root / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(output / "synthetic_recovery.json", result)
    print(json.dumps({
        "status": result["status"],
        "revision": result["revision"],
        "by_truth": result["by_truth"],
        "criteria": result["criteria"],
        "output": str(output / "synthetic_recovery.json"),
    }, indent=2, sort_keys=True))
    if not result["all_criteria_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
