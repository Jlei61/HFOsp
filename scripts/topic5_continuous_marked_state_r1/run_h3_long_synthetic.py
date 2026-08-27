#!/usr/bin/env python3
"""Synthetic recovery for exact-window long H3 affine edges."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.h3_long import (
    H3_LONG_REVISION,
    SCALES,
    SYNTHETIC_TRUTHS,
    affine_estimability_audit,
    chronological_trend_exposure,
    classify_affine_estimability,
    evaluate_affine_edge,
    exact_boxcar_event_exposure,
    exact_previous_block_placebo,
    fit_affine_edge,
    standardise_exposure_on_train,
)
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    state_matched_nonoverlap_placebo,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import OneStepDesign


TRUTHS = SYNTHETIC_TRUTHS
SEEDS = (0, 1, 2)


class SyntheticGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(2))

    def matrix(self) -> torch.Tensor:
        return self.mu.new_zeros((2, 2))


class SyntheticModel(torch.nn.Module):
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
            event_log_prob=zero, group_size_log_prob=zero,
            subset_log_prob=zero, group_size_step_log_prob=step,
            subset_step_log_prob=step,
            active_step=torch.ones((n, 2), dtype=torch.bool, device=state.device),
            select_step=torch.tensor(
                [True, False], device=state.device
            ).expand(n, 2),
        )


def make_design(
    state: np.ndarray,
    history: np.ndarray,
    exposure: np.ndarray,
    log_rate: np.ndarray,
    split: np.ndarray,
    index: np.ndarray,
) -> OneStepDesign:
    interval = np.exp(-np.asarray(log_rate, dtype=np.float64)).astype(np.float32)
    n = len(index)
    return OneStepDesign(
        current_state=np.asarray(state[index], dtype=np.float32),
        current_index=np.asarray(index, dtype=np.int64),
        next_history=np.asarray(history[index], dtype=np.float32),
        next_group_ids=np.zeros((n, 1), dtype=np.int64),
        next_group_count=np.ones(n, dtype=np.int64),
        delta_minutes=interval[index] / 60.0,
        quadrature_delta_minutes=np.tile(
            interval[index, None] / 120.0, (1, 4)
        ),
        quadrature_history=np.zeros((n, 4, 3), dtype=np.float32),
        quadrature_weight_seconds=np.tile(
            interval[index, None] / 4.0, (1, 4)
        ),
        exposure=np.asarray(exposure[index], dtype=np.float32),
        split=np.asarray(split[index], dtype=np.int8),
    )


def run_cell(scale: int, truth: str, seed: int, *, device: str) -> dict:
    rng = np.random.default_rng(8000 + 17 * int(seed) + int(scale))
    total = max(5 * int(scale), 6000)
    train_stop = int(.7 * total)
    split = np.r_[
        np.zeros(train_stop, dtype=np.int8),
        np.ones(total - train_stop, dtype=np.int8),
    ]
    segment = np.zeros(total, dtype=np.int64)
    innovation = rng.normal(size=total).astype(np.float32)
    phase = np.linspace(0.0, 8.0 * np.pi, total, dtype=np.float64)
    state = np.zeros((total, 2), dtype=np.float32)
    state[:, 0] = (.35 * np.sin(phase)).astype(np.float32)
    state[:, 1] = (.20 * np.cos(phase)).astype(np.float32)
    history = np.column_stack([
        np.sin(phase), np.cos(phase), np.sin(.5 * phase),
    ]).astype(np.float32)
    observation = np.column_stack([
        np.sin(phase + .1), np.cos(phase + .1),
    ]).astype(np.float32)
    real_raw, real_eligible, exposure_audit = exact_boxcar_event_exposure(
        innovation, segment, scale_events=int(scale)
    )
    causal_raw, causal_eligible, causal_audit = exact_previous_block_placebo(
        real_raw, segment, scale_events=int(scale)
    )
    train = split == 0
    matched_raw, matched_eligible, matched_audit = (
        state_matched_nonoverlap_placebo(
            real_raw, state, history, observation, train, real_eligible,
            segment, scale_events=int(scale), history_multiples=1,
            neighbours=64,
        )
    )
    common = real_eligible & causal_eligible & matched_eligible
    real, real_scaler = standardise_exposure_on_train(real_raw, train, common)
    causal, causal_scaler = standardise_exposure_on_train(
        causal_raw, train, common
    )
    current, current_scaler = standardise_exposure_on_train(
        innovation, train, common
    )
    state_matched, matched_scaler = standardise_exposure_on_train(
        matched_raw, train, common
    )
    trend_raw = chronological_trend_exposure(
        np.arange(total, dtype=np.float64), segment, 1
    )[:, 0]
    trend, trend_scaler = standardise_exposure_on_train(
        trend_raw, train, common
    )
    beta = 0.0
    offset = 0.0
    if truth == "positive":
        beta = .45
    elif truth == "reversed":
        beta = -.45
    elif truth == "constant":
        offset = .65
    elif truth == "observed_drift":
        # The slow variation is already represented in current_state.
        pass
    elif truth == "unobserved_drift":
        # A difficult exposure-free null omitted from the frozen state.
        offset = np.linspace(-.55, .55, total, dtype=np.float64)
    elif truth != "zero":
        raise ValueError(truth)
    log_rate = state[:, 0].astype(np.float64) + offset + beta * real
    index = np.flatnonzero(common)
    exposures = {
        "real": real,
        "causal_previous_block": causal,
        "current_event": current,
        "state_matched_nonoverlap": state_matched,
        "chronological_trend": trend,
        "intercept_only": np.zeros(total, dtype=np.float32),
    }
    model = SyntheticModel().to(device)
    designs = {
        label: make_design(state, history, value, log_rate, split, index)
        for label, value in exposures.items()
    }
    edges, fits, metrics = {}, {}, {}
    for label, design in designs.items():
        edge, fit = fit_affine_edge(
            model, design, device=device, seed=seed, epochs=20,
            learning_rate=.03, batch_size=4096,
        )
        edges[label] = edge; fits[label] = fit
        metrics[label] = asdict(evaluate_affine_edge(
            model, edge, design, split="validation", device=device,
            batch_size=4096,
        ))
    real_matrix = float(edges["real"].matrix[0, 0].detach().cpu())
    real_minus = {
        label: float(
            metrics["real"]["joint_nll_per_event"]
            - metrics[label]["joint_nll_per_event"]
        ) for label in (
            "causal_previous_block", "state_matched_nonoverlap",
            "current_event", "chronological_trend", "intercept_only"
        )
    }
    audit = affine_estimability_audit(
        model, designs["real"], device=device, batch_size=4096
    )
    estimability_class = classify_affine_estimability(audit, fits["real"])
    if truth in {"positive", "reversed"}:
        passed = bool(
            np.sign(real_matrix) == np.sign(beta)
            and fits["real"]["edge_left_zero_initialisation"]
            and all(value < 0 for value in real_minus.values())
        )
    elif truth == "constant":
        passed = bool(
            (
                abs(real_matrix) < .03
                and fits["real"]["intercept_norm"] > .1
                and abs(real_minus["intercept_only"]) < .002
            )
            or (
                estimability_class == "ZERO_SELECTED"
                and real_minus["intercept_only"] >= 0
            )
        )
    elif truth == "unobserved_drift":
        passed = bool(
            real_minus["chronological_trend"] >= -0.002
            and metrics["chronological_trend"]["joint_nll_per_event"]
            < metrics["intercept_only"]["joint_nll_per_event"]
        )
    else:
        passed = bool(
            abs(real_matrix) < .03
            and abs(real_minus["intercept_only"]) < .002
        )
    return {
        "scale_events": int(scale), "truth": truth, "seed": int(seed),
        "n_train": int((designs["real"].split == 0).sum()),
        "n_validation": int((designs["real"].split == 1).sum()),
        "true_beta": float(beta),
        "true_offset": (
            float(offset) if np.ndim(offset) == 0 else "unobserved_linear_drift"
        ),
        "real_matrix_0_0": real_matrix, "real_minus_controls": real_minus,
        "fits": fits, "metrics": metrics, "estimability": audit,
        "real_estimability_class": estimability_class,
        "exposure_audit": exposure_audit, "causal_audit": causal_audit,
        "state_matched_audit": matched_audit,
        "scalers": {
            "real": real_scaler, "causal": causal_scaler,
            "current": current_scaler, "state_matched": matched_scaler,
            "chronological_trend": trend_scaler,
        },
        "pass": passed,
    }


def _run_cell_tuple(value: tuple[tuple[int, str, int], str]) -> dict:
    task, device = value
    return run_cell(*task, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long/synthetic",
    )
    args = parser.parse_args()
    tasks = [
        (scale, truth, seed) for scale in SCALES
        for truth in TRUTHS for seed in SEEDS
    ]
    if args.device == "cpu" and int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            rows = list(pool.map(
                _run_cell_tuple, [(task, args.device) for task in tasks]
            ))
    else:
        rows = [run_cell(*task, device=args.device) for task in tasks]
    by_truth = {}
    for scale in SCALES:
        for truth in TRUTHS:
            take = [
                row for row in rows
                if row["scale_events"] == scale and row["truth"] == truth
            ]
            by_truth[f"N{scale}_{truth}"] = {
                "passed_seeds": int(sum(row["pass"] for row in take)),
                "n_seeds": len(take),
                "all_pass": bool(all(row["pass"] for row in take)),
            }
    payload = {
        "status": "COMPLETE",
        "revision": H3_LONG_REVISION,
        "scales": list(SCALES), "truths": list(TRUTHS),
        "seeds": list(SEEDS), "rows": rows, "by_truth": by_truth,
        "all_cells_pass": bool(all(value["all_pass"] for value in by_truth.values())),
        "expected_cell_count": len(SCALES) * len(TRUTHS) * len(SEEDS),
        "source_hashes": {
            "producer": contract.sha256_file(Path(__file__)),
            "h3_long": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/h3_long.py"
            ),
        },
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(args.output_root / "synthetic_recovery.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "all_cells_pass": payload["all_cells_pass"],
        "by_truth": by_truth,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
