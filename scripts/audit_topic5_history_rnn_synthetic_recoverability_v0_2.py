#!/usr/bin/env python3
"""Synthetic positive control for chronology-specific HistoryRNN recovery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_history_rnn import MatchedUnorderedSummary, TimeDecayHistoryGRU


class UnorderedModel(nn.Module):
    def __init__(self, event_dim: int, hidden_dim: int, contacts: int):
        super().__init__()
        self.summary = MatchedUnorderedSummary(event_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, contacts)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.output(self.summary(summary))


class ChronologicalModel(nn.Module):
    def __init__(self, event_dim: int, hidden_dim: int, contacts: int):
        super().__init__()
        self.history = TimeDecayHistoryGRU(
            event_dim, hidden_dim, initial_half_life_hours=2.0
        )
        self.output = nn.Linear(hidden_dim, contacts)

    def forward(
        self, events: torch.Tensor, delta: torch.Tensor, reset: torch.Tensor
    ) -> torch.Tensor:
        state = self.history(events, delta, reset)
        return self.output(state)


def _summary(events: np.ndarray, delta: np.ndarray) -> np.ndarray:
    n_sequence, n_event, event_dim = events.shape
    count = np.arange(1, n_event + 1, dtype=np.float32)[None, :, None]
    mean = np.cumsum(events, axis=1) / count
    maximum = np.maximum.accumulate(events, axis=1)
    span = np.cumsum(delta, axis=1) / 3600.0
    scalar = np.stack([
        np.broadcast_to(np.log1p(np.arange(1, n_event + 1)), (n_sequence, n_event)),
        np.log1p(span),
        np.log1p(delta),
    ], axis=-1)
    return np.concatenate([mean, maximum, events, scalar], axis=-1).astype(np.float32)


def _dataset(seed: int, n_sequence: int, n_event: int, contacts: int):
    rng = np.random.default_rng(seed)
    symbol = rng.integers(0, 4, size=(n_sequence, n_event))
    events = np.eye(4, dtype=np.float32)[symbol]
    events += rng.normal(0, 0.03, events.shape).astype(np.float32)
    delta = rng.uniform(30.0, 7200.0, size=(n_sequence, n_event)).astype(np.float32)
    delta[:, 0] = 0.0
    # The target depends on the ordered pair (t-1,t), which mean/max/last
    # summaries cannot reconstruct.  Four code bits and their complements
    # create a balanced eight-contact field, avoiding a trivial all-negative
    # solution under BCE.
    previous = np.roll(symbol, 1, axis=1)
    code = 4 * previous + symbol
    bit = np.stack([((code >> shift) & 1) for shift in range(4)], axis=-1)
    target = np.concatenate([bit, 1 - bit], axis=-1).astype(np.float32)
    target[:, 0] = 0.0
    return events, delta, target, _summary(events, delta)


def _fit(model, inputs, target, *, epochs: int, seed: int, device):
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*[value.to(device) for value in inputs])
        loss = criterion(logits[:, 1:], target[:, 1:].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _bce(model, inputs, target, device) -> float:
    logits = model(*[value.to(device) for value in inputs])
    return float(nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:], target[:, 1:].to(device)
    ).cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in (20260831, 20260832, 20260833):
        train = _dataset(seed, 256, 32, 8)
        test = _dataset(seed + 100, 128, 32, 8)
        train_event, train_delta, train_target, train_summary = train
        test_event, test_delta, test_target, test_summary = test
        reset_train = np.zeros(train_delta.shape, dtype=bool)
        reset_test = np.zeros(test_delta.shape, dtype=bool)
        reset_train[:, 0] = True
        reset_test[:, 0] = True
        unordered = UnorderedModel(4, 12, 8)
        chronological = ChronologicalModel(4, 12, 8)
        unordered = _fit(
            unordered,
            (torch.as_tensor(train_summary),),
            torch.as_tensor(train_target),
            epochs=args.epochs,
            seed=seed,
            device=device,
        )
        chronological = _fit(
            chronological,
            (
                torch.as_tensor(train_event),
                torch.as_tensor(train_delta),
                torch.as_tensor(reset_train),
            ),
            torch.as_tensor(train_target),
            epochs=args.epochs,
            seed=seed,
            device=device,
        )
        true_bce = _bce(
            chronological,
            (
                torch.as_tensor(test_event),
                torch.as_tensor(test_delta),
                torch.as_tensor(reset_test),
            ),
            torch.as_tensor(test_target),
            device,
        )
        shuffled = test_event.copy()
        rng = np.random.default_rng(seed + 200)
        for sequence in shuffled:
            for index in range(2, len(sequence)):
                if index > 2:
                    prefix = sequence[:index - 1].copy()
                    rng.shuffle(prefix, axis=0)
                    sequence[:index - 1] = prefix
        shuffle_bce = _bce(
            chronological,
            (
                torch.as_tensor(shuffled),
                torch.as_tensor(test_delta),
                torch.as_tensor(reset_test),
            ),
            torch.as_tensor(test_target),
            device,
        )
        unordered_bce = _bce(
            unordered,
            (torch.as_tensor(test_summary),),
            torch.as_tensor(test_target),
            device,
        )
        with torch.no_grad():
            state = chronological.history(
                torch.as_tensor(test_event, device=device),
                torch.as_tensor(test_delta, device=device),
                torch.as_tensor(reset_test, device=device),
            )
        rows.append({
            "seed": seed,
            "unordered_bce": unordered_bce,
            "chronological_bce": true_bce,
            "shuffle_bce": shuffle_bce,
            "unordered_minus_chronological": unordered_bce - true_bce,
            "shuffle_minus_chronological": shuffle_bce - true_bce,
            "state_variance": float(state.var(dim=(0, 1)).mean().cpu()),
            "readout_norm": float(chronological.output.weight.norm().cpu()),
            "decay_gradient_available": True,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "synthetic_seed_metrics.csv", index=False)
    passed = bool(
        np.all(frame.unordered_minus_chronological > 0.02)
        and np.all(frame.shuffle_minus_chronological > 0.02)
        and np.all(frame.state_variance > 1e-5)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "contract": "topic5_history_rnn_synthetic_recoverability_v0_2",
        "n_seeds": len(frame),
        "median_unordered_minus_chronological_bce": float(
            frame.unordered_minus_chronological.median()
        ),
        "median_shuffle_minus_chronological_bce": float(
            frame.shuffle_minus_chronological.median()
        ),
        "all_seed_direction_consistent": bool(
            np.all(frame.unordered_minus_chronological > 0)
            and np.all(frame.shuffle_minus_chronological > 0)
        ),
        "interpretation": (
            "The architecture and optimizer can recover an injected chronology-specific signal."
            if passed else
            "Recoverability was not established; real-data G1 must not be interpreted scientifically."
        ),
    }
    (output / "SYNTHETIC_RECOVERABILITY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
