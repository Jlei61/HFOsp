#!/usr/bin/env python3
"""Synthetic recovery for the exact frozen-baseline regular T1 trainer."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import (
    PreparedRegularT1,
    REGULAR_T1_REVISION,
    RegularT1Model,
    _baseline_metrics,
    _matched_wrong_time_swap,
    _new_history_baseline,
    _optimise_history_baseline,
    _post_anchor_challenge,
    _run_split,
    _target_scales,
)


def latent_at(minutes: np.ndarray) -> np.ndarray:
    decay = np.exp(-minutes / 1000.0)
    angle = 0.04 * minutes
    return np.stack([
        decay * (1.3 * np.cos(angle) - 0.7 * np.sin(angle)),
        decay * (1.3 * np.sin(angle) + 0.7 * np.cos(angle)),
    ], axis=1)


def make_sequence(seed: int = 0, n_events: int = 1400) -> PreparedRegularT1:
    rng = np.random.default_rng(seed)
    event_minutes = [0.0]
    latent = []
    log_iei = []
    for _ in range(n_events):
        state = latent_at(np.asarray([event_minutes[-1]]))[0]
        latent.append(state)
        value = 3.1 + 0.32 * state[0] - 0.18 * state[1] + rng.normal(0, 0.08)
        log_iei.append(value)
        event_minutes.append(event_minutes[-1] + math.exp(value) / 60.0)
    event_minutes = np.asarray(event_minutes[:-1], dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float32)
    log_iei = np.asarray(log_iei, dtype=np.float32)
    next_minutes = event_minutes + np.exp(log_iei) / 60.0
    n_contacts = 5
    weights = np.asarray([
        [1.2, -0.2], [0.8, 0.5], [-0.5, 1.1], [-1.0, -0.3], [0.2, -1.0]
    ], dtype=np.float32)
    logits = latent @ weights.T
    probability = 1.0 / (1.0 + np.exp(-logits))
    participation = (rng.random(probability.shape) < probability).astype(np.float32)
    empty = participation.sum(axis=1) == 0
    participation[empty, np.argmax(probability[empty], axis=1)] = 1.0
    rank = (latent @ (weights * 0.35).T + rng.normal(
        0, 0.05, size=probability.shape
    )).astype(np.float32)
    rank *= participation
    stop = np.clip(
        0.45 + 0.12 * latent[:, 0] - 0.08 * latent[:, 1]
        + rng.normal(0, 0.025, n_events), 0.02, 0.98,
    ).astype(np.float32)
    history = rng.normal(0, 1, size=(n_events, 5)).astype(np.float32)
    cut = int(0.70 * n_events)
    split = np.ones(n_events, dtype=np.int8)
    split[:cut] = 0
    anchors = np.arange(0.0, float(next_minutes[-1]) + 1.0, 1.0)
    observation_latent = latent_at(anchors).astype(np.float32)
    observation = rng.normal(
        0, 0.05, size=(len(anchors), contract.STATE_OBSERVATION_DIM)
    ).astype(np.float32)
    observation[:, :2] += observation_latent
    observation_split = (anchors >= event_minutes[cut]).astype(np.int8)
    return PreparedRegularT1(
        subject="synthetic_damped_rotation",
        history=torch.as_tensor(history),
        observation=torch.as_tensor(observation),
        observation_time=anchors * 60.0,
        observation_split=observation_split,
        event_time=event_minutes * 60.0,
        next_time=next_minutes * 60.0,
        session=np.zeros(n_events, dtype=np.int64),
        split=split,
        log_iei=torch.as_tensor(log_iei),
        participation=torch.as_tensor(participation),
        rank=torch.as_tensor(rank),
        stop=torch.as_tensor(stop),
    )


def fit_seed(sequence: PreparedRegularT1, baseline, scales: dict[str, float],
             seed: int, epochs: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RegularT1Model(
        sequence.history.shape[1], sequence.participation.shape[1],
        scales, baseline, state_dim=8,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=2e-2,
    )
    for _ in range(epochs):
        _run_split(
            model, sequence, 0, correction_enabled=True,
            state_enabled=True, optimizer=optimizer,
        )
    model.eval()
    with torch.no_grad():
        train, train_final = _run_split(
            model, sequence, 0, correction_enabled=True, state_enabled=True
        )
        filtered, _ = _run_split(
            model, sequence, 1, correction_enabled=True,
            state_enabled=True, initial=train_final,
        )
        correction_off, _ = _run_split(
            model, sequence, 1, correction_enabled=False,
            state_enabled=True, initial=train_final,
        )
        post = _post_anchor_challenge(
            model, sequence, train_final,
            correction_enabled=True, state_enabled=True,
        )
        swap = _matched_wrong_time_swap(
            model, sequence, train_final,
            correction_enabled=True, state_enabled=True,
        )
    return {
        "seed": seed, "train": train, "filtered": filtered,
        "correction_off_from_split_start": correction_off,
        "post_anchor": post, "state_swap": swap,
    }


def main() -> None:
    sequence = make_sequence()
    scales = _target_scales(sequence)
    train_idx = torch.as_tensor(np.flatnonzero(sequence.split == 0), dtype=torch.long)
    baseline = _new_history_baseline(sequence, scales)
    _optimise_history_baseline(
        baseline, sequence, train_idx, penalty_weight=1.0, max_iter=200
    )
    baseline_validation = _baseline_metrics(
        baseline, sequence,
        torch.as_tensor(np.flatnonzero(sequence.split == 1), dtype=torch.long),
    )
    torch.manual_seed(0)
    t0_model = RegularT1Model(
        sequence.history.shape[1], sequence.participation.shape[1],
        scales, baseline, state_dim=8,
    )
    t0_model.eval()
    with torch.no_grad():
        _, t0_train_final = _run_split(
            t0_model, sequence, 0, correction_enabled=False,
            state_enabled=False,
        )
        t0_filtered, _ = _run_split(
            t0_model, sequence, 1, correction_enabled=False,
            state_enabled=False, initial=t0_train_final,
        )
        t0_post = _post_anchor_challenge(
            t0_model, sequence, t0_train_final,
            correction_enabled=False, state_enabled=False,
        )
    runs = [fit_seed(sequence, baseline, scales, seed, epochs=40)
            for seed in (0, 1, 2)]
    contrasts = []
    for run in runs:
        contrasts.append({
            "seed": run["seed"],
            "filtered_joint_t1_minus_t0": (
                run["filtered"]["joint_nll"] - t0_filtered["joint_nll"]
            ),
            "correction_off_joint_t1_minus_t0": (
                run["correction_off_from_split_start"]["joint_nll"]
                - t0_filtered["joint_nll"]
            ),
            "post_anchor_joint_t1_minus_t0": {
                horizon: values["joint_nll"] - t0_post[horizon]["joint_nll"]
                for horizon, values in run["post_anchor"].items()
            },
            "state_swap_joint_wrong_minus_correct": run["state_swap"]
            ["endpoints"]["joint_nll"]["wrong_minus_correct"],
        })
    output = {
        "contract": contract.REVISION,
        "regular_t1_revision": REGULAR_T1_REVISION,
        "synthetic_truth": (
            "two-dimensional 1000-min damped rotation observed every minute; "
            "the same latent drives exact next interval and all mark endpoints"
        ),
        "n_train_events": int((sequence.split == 0).sum()),
        "n_validation_events": int((sequence.split == 1).sum()),
        "state_dim": 8,
        "baseline_validation": baseline_validation,
        "t0_validation": t0_filtered,
        "t0_post_anchor": t0_post,
        "runs": runs, "contrasts": contrasts,
        "recovery_summary": {
            "n_filtered_joint_better": int(sum(
                row["filtered_joint_t1_minus_t0"] < 0 for row in contrasts
            )),
            "n_swap_correct_better": int(sum(
                row["state_swap_joint_wrong_minus_correct"] > 0 for row in contrasts
            )),
            "post_anchor_n_better": {
                horizon: int(sum(
                    row["post_anchor_joint_t1_minus_t0"][horizon] < 0
                    for row in contrasts
                )) for horizon in ("5", "10", "20")
            },
        },
        "sealed_opened": False,
        "claim_boundary": "synthetic instrument recovery only; no human-data claim",
    }
    path = contract.RESULT_ROOT / "state_smoke/REGULAR_T1_SYNTHETIC_RECOVERY.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps(output["recovery_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
