#!/usr/bin/env python3
"""Fit train-only set-size decoders and evaluate held-out free ECoG fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_ecog_physical_neighborhood_v0_1 import build_fixed_grid_model  # noqa: E402
from src.topic5_rnn_motif_v0_4 import RolloutSizeHead, state_features  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def advance(model: torch.nn.Module, hidden: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    hidden = model._step(hidden, inputs)
    zero = torch.zeros_like(inputs)
    for _ in range(int(model.microsteps) - 1):
        hidden = model._step(hidden, zero)
    return hidden


@torch.no_grad()
def size_examples(
    model: torch.nn.Module,
    ranks: np.ndarray,
    event_indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    maximum_decisions: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(int(seed))
    indices = np.asarray(event_indices, dtype=int)
    if len(indices) > 0:
        indices = rng.permutation(indices)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    collected = 0
    model.eval()
    for begin in range(0, len(indices), int(batch_size)):
        chosen = indices[begin:begin + int(batch_size)]
        tensors = {name: value.to(device) for name, value in build_event_tensors(ranks[chosen]).items()}
        hidden = torch.zeros(len(chosen), model.n_nodes * model.state_dim, device=device)
        feature_grid: list[torch.Tensor] = []
        for step in range(tensors["x"].shape[1]):
            hidden = advance(model, hidden, tensors["x"][:, step])
            feature_grid.append(state_features(
                model, hidden, step, tensors["recruited"][:, step].mean(-1)
            ))
        feature_grid_tensor = torch.stack(feature_grid, dim=1)
        continuing = tensors["valid"] & ~tensors["is_last"]
        features.append(feature_grid_tensor[continuing].cpu())
        targets.append((tensors["target"].sum(-1).long()[continuing] - 1).cpu())
        collected += int(continuing.sum())
        if collected >= int(maximum_decisions):
            break
    if not features:
        raise RuntimeError("no continuing decisions for size decoder")
    x = torch.cat(features)[: int(maximum_decisions)]
    y = torch.cat(targets)[: int(maximum_decisions)]
    return x, y


def fit_size_head(
    model: torch.nn.Module,
    ranks: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[RolloutSizeHead, dict[str, Any]]:
    train_x, train_y = size_examples(
        model, ranks, train_index, device, batch_size=batch_size,
        maximum_decisions=200000, seed=seed + 101,
    )
    validation_x, validation_y = size_examples(
        model, ranks, validation_index, device, batch_size=batch_size,
        maximum_decisions=100000, seed=seed + 103,
    )
    torch.manual_seed(int(seed) + 4242)
    head = RolloutSizeHead(model.n_contacts).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    train_x, train_y = train_x.to(device), train_y.to(device)
    validation_x, validation_y = validation_x.to(device), validation_y.to(device)
    best = float("inf")
    best_state = None
    stale = 0
    curve = []
    rng = np.random.default_rng(int(seed) + 107)
    for epoch in range(200):
        head.train()
        order = rng.permutation(len(train_y))
        loss_sum = 0.0
        for begin in range(0, len(order), 4096):
            chosen = torch.as_tensor(order[begin:begin + 4096], device=device)
            loss = torch.nn.functional.cross_entropy(head(train_x[chosen]), train_y[chosen])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(chosen)
        head.eval()
        with torch.no_grad():
            validation_loss = torch.nn.functional.cross_entropy(head(validation_x), validation_y)
        curve.append({
            "epoch": epoch,
            "train_nll": loss_sum / len(train_y),
            "validation_nll": float(validation_loss),
        })
        if float(validation_loss) < best - 1e-6:
            best = float(validation_loss)
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 20:
                break
    if best_state is None:
        raise RuntimeError("set-size decoder did not produce a finite checkpoint")
    head.load_state_dict(best_state)
    return head, {
        "n_train_decisions": len(train_y),
        "n_validation_decisions": len(validation_y),
        "best_validation_nll": best,
        "epochs": len(curve),
        "curve": curve,
    }


@torch.no_grad()
def batched_free_rollout(
    model: torch.nn.Module,
    size_head: RolloutSizeHead,
    starts: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    size_head.eval()
    n_events, n_contacts = starts.shape
    output = np.full((n_events, n_contacts), -1, dtype=np.int16)
    for begin in range(0, n_events, int(batch_size)):
        end = min(n_events, begin + int(batch_size))
        start = torch.as_tensor(starts[begin:end], dtype=torch.bool, device=device)
        n_batch = len(start)
        hidden = torch.zeros(n_batch, model.n_nodes * model.state_dim, device=device)
        recruited = start.clone()
        inputs = start.float()
        generated = torch.full((n_batch, n_contacts), -1, dtype=torch.int16, device=device)
        generated[start] = 0
        active = torch.ones(n_batch, dtype=torch.bool, device=device)
        for step in range(n_contacts):
            hidden = advance(model, hidden, inputs)
            fraction = recruited.float().mean(-1)
            features = state_features(model, hidden, step, fraction)
            stop = torch.sigmoid(model._stop(hidden, features[:, 2], fraction)) >= 0.5
            active = active & ~stop & ~torch.all(recruited, dim=-1)
            if not bool(active.any()):
                break
            logits = model._readout(hidden).masked_fill(recruited, -1e9)
            requested = size_head(features).argmax(-1) + 1
            remaining = (~recruited).sum(-1)
            requested = torch.minimum(requested, remaining)
            order = torch.argsort(logits, dim=-1, descending=True, stable=True)
            rank_position = torch.empty_like(order)
            positions = torch.arange(n_contacts, device=device).expand_as(order)
            rank_position.scatter_(1, order, positions)
            picked = (rank_position < requested[:, None]) & active[:, None] & ~recruited
            generated[picked] = int(step + 1)
            recruited |= picked
            inputs = picked.float()
        output[begin:end] = generated.cpu().numpy()
    return output


def mean_field(ranks: np.ndarray, remove_start: bool) -> np.ndarray:
    values = np.asarray(ranks)
    scores = np.zeros(values.shape, dtype=float)
    for index, row in enumerate(values):
        observed = row >= 0
        if not np.any(observed):
            continue
        n_sets = int(row[observed].max()) + 1
        scores[index, observed] = (n_sets - row[observed]) / max(1, n_sets)
        if remove_start:
            scores[index, row == 0] = 0.0
    return scores.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
    ))
    args = parser.parse_args()
    summary = json.loads(args.summary_path.read_text())
    unit_dir = args.summary_path.parent
    output_path = unit_dir / "field_metrics.json"
    checkpoint_path = Path(summary["checkpoint_path"])
    event_path = args.cache_root / str(summary["subject"]) / "events.npz"
    if output_path.exists() and not args.force:
        prior = json.loads(output_path.read_text())
        if (
            prior.get("checkpoint_sha256") == sha256_file(checkpoint_path)
            and prior.get("events_sha256") == sha256_file(event_path)
        ):
            print(output_path.read_text())
            return
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_fixed_grid_model(
        checkpoint["channel_names"], checkpoint["mask"], checkpoint["model_seed"],
        state_dim=checkpoint["state_dim"], microsteps=checkpoint["microsteps"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    with np.load(event_path, allow_pickle=False) as events:
        real_ranks = np.asarray(events["ranks"], dtype=np.int16)
        split = np.asarray(events["split"], dtype=np.int8)
    development_ranks = real_ranks.copy()
    if summary["family"] == "SUFFIX_SHUFFLED":
        null_path = Path(summary["null_path"])
        with np.load(null_path, allow_pickle=False) as null:
            null_ranks = np.asarray(null["ranks"], dtype=np.int16)
        development_ranks[split != 2] = null_ranks[split != 2]
    started = time.time()
    head, decoder_metrics = fit_size_head(
        model, development_ranks, np.flatnonzero(split == 0), np.flatnonzero(split == 1),
        device, int(summary["model_seed"]), args.batch_size,
    )
    test_ranks = real_ranks[split == 2]
    starts = test_ranks == 0
    generated = batched_free_rollout(model, head, starts, device, args.batch_size)
    empirical_full = mean_field(test_ranks, remove_start=False)
    generated_full = mean_field(generated, remove_start=False)
    empirical_removed = mean_field(test_ranks, remove_start=True)
    generated_removed = mean_field(generated, remove_start=True)
    result = {
        "schema": "topic5_ecog_free_field_v0.1",
        "subject": summary["subject"],
        "family": summary["family"],
        "graph_id": summary["graph_id"],
        "seed_index": summary["seed_index"],
        "n_test_events": len(test_ranks),
        "full_field_spearman": float(spearmanr(empirical_full, generated_full).statistic),
        "start_removed_field_spearman": float(spearmanr(empirical_removed, generated_removed).statistic),
        "generated_participant_count_median": float(np.median(np.sum(generated >= 0, axis=1))),
        "observed_participant_count_median": float(np.median(np.sum(test_ranks >= 0, axis=1))),
        "decoder": {key: value for key, value in decoder_metrics.items() if key != "curve"},
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "events_sha256": sha256_file(event_path),
        "runtime_sec": time.time() - started,
    }
    torch.save(head.state_dict(), unit_dir / "rollout_size_head.pt")
    np.savez_compressed(
        unit_dir / "heldout_free_fields.npz",
        generated_ranks=generated,
        empirical_full=empirical_full.astype(np.float32),
        generated_full=generated_full.astype(np.float32),
        empirical_start_removed=empirical_removed.astype(np.float32),
        generated_start_removed=generated_removed.astype(np.float32),
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
