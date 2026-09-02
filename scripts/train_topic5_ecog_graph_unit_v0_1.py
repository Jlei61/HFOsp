#!/usr/bin/env python3
"""Train one fixed-graph ECoG RNN unit on frozen full-grid rank sets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_ecog_physical_neighborhood_v0_1 import build_fixed_grid_model  # noqa: E402
from src.topic5_wiring_economy_rnn import (  # noqa: E402
    build_event_tensors,
    cardinality_conditioned_nll,
    next_rank_stop_loss,
)


MODEL_SEEDS = (2026081611, 2026081612, 2026081613)
TOP1_CONTRACT = "top_prediction_is_any_member_of_tied_next_rank_set_v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    # Hash trainable tensors only. The graph mask has its own artifact hash and
    # deliberately differs across arms; including buffers here would make the
    # paired initial-weight audit fail by construction.
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).view(np.uint8))
    return digest.hexdigest()


def batch_from_indices(ranks: np.ndarray, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = build_event_tensors(ranks[np.asarray(indices, dtype=int)])
    return {name: value.to(device, non_blocking=True) for name, value in tensors.items()}


def top1_hits(logits: torch.Tensor, target: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Return one hit per decision when the top contact belongs to the tied next set."""
    masked = logits.masked_fill(~available, -1e9)
    prediction = masked.argmax(-1, keepdim=True)
    return target.gather(-1, prediction).squeeze(-1) > 0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    ranks: np.ndarray | dict[str, torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "next_bce": 0.0, "stop_bce": 0.0, "contact_nll": 0.0, "top1": 0.0}
    contact_decisions = 0.0
    valid_steps = 0.0
    step_nll_sum: dict[int, float] = {}
    step_nll_count: dict[int, int] = {}
    cached = isinstance(ranks, dict)
    for begin in range(0, len(indices), batch_size):
        chosen = indices[begin:begin + batch_size]
        if cached:
            batch = {
                name: value[torch.as_tensor(chosen, dtype=torch.long)].to(device, non_blocking=True)
                for name, value in ranks.items()
            }
        else:
            batch = batch_from_indices(ranks, chosen, device)
        logits, stop = model(batch["x"], batch["recruited"], batch["valid"])
        loss, next_bce, stop_bce = next_rank_stop_loss(
            logits, stop, batch["target"], batch["available"], batch["valid"], batch["is_last"]
        )
        predict = batch["valid"] & ~batch["is_last"]
        nll = cardinality_conditioned_nll(logits, batch["target"], batch["available"], predict)
        masked = logits.masked_fill(~batch["available"], -1e9)
        top1 = (top1_hits(logits, batch["target"], batch["available"]) & predict).float().sum()
        n_decision = float(predict.sum().item())
        n_valid = float(batch["valid"].sum().item())
        totals["loss"] += float(loss) * n_decision
        totals["next_bce"] += float(next_bce) * n_decision
        totals["stop_bce"] += float(stop_bce) * n_decision
        totals["contact_nll"] += float(nll) * n_decision
        totals["top1"] += float(top1)
        contact_decisions += n_decision
        valid_steps += n_valid

        log_prob = torch.log_softmax(masked, dim=-1)
        per_step = -(log_prob * batch["target"]).sum(-1) / batch["target"].sum(-1).clamp_min(1.0)
        for step in range(per_step.shape[1]):
            keep = predict[:, step]
            if bool(keep.any()):
                step_nll_sum[step] = step_nll_sum.get(step, 0.0) + float(per_step[keep, step].sum())
                step_nll_count[step] = step_nll_count.get(step, 0) + int(keep.sum())
    result = {name: value / max(contact_decisions, 1.0) for name, value in totals.items()}
    result["n_events"] = int(len(indices))
    result["n_continue_decisions"] = int(contact_decisions)
    result["n_valid_steps"] = int(valid_steps)
    result["nll_by_rank_step"] = {
        str(step): step_nll_sum[step] / step_nll_count[step] for step in sorted(step_nll_sum)
    }
    result["n_by_rank_step"] = {str(step): step_nll_count[step] for step in sorted(step_nll_count)}
    return result


def train_unit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    event_path = args.cache_root / args.subject / "events.npz"
    with np.load(event_path, allow_pickle=False) as events:
        real_ranks = np.asarray(events["ranks"], dtype=np.int16)
        split = np.asarray(events["split"], dtype=np.int8)
        channel_names = [str(value) for value in events["channel_names"].tolist()]
    ranks = real_ranks.copy()
    null_path: Path | None = None
    if args.family == "SUFFIX_SHUFFLED":
        null_path = args.cache_root / args.subject / f"events_suffix_null_seed{args.seed_index}.npz"
        with np.load(null_path, allow_pickle=False) as null:
            null_ranks = np.asarray(null["ranks"], dtype=np.int16)
            if not np.array_equal(null["split"], split):
                raise RuntimeError("suffix-null split mismatch")
        development = (split == 0) | (split == 1)
        ranks[development] = null_ranks[development]
        if not np.array_equal(ranks[split == 2], real_ranks[split == 2]):
            raise RuntimeError("suffix control changed held-out test")

    graph_path = args.graph_path
    with np.load(graph_path, allow_pickle=False) as graph:
        graph_names = [str(value) for value in graph["channel_names"].tolist()]
        mask = np.asarray(graph["mask"], dtype=np.uint8)
        graph_id = str(graph["graph_id"].item())
    if graph_names != channel_names:
        raise ValueError("graph and event channel order mismatch")

    unit_name = f"{args.family}__{graph_id}__seed{args.seed_index}"
    out_dir = args.output_root / args.subject / unit_name
    summary_path = out_dir / "summary.json"
    checkpoint_path = out_dir / "checkpoint.pt"
    current_contract = {
        "events_sha256": sha256_file(event_path),
        "graph_sha256": sha256_file(graph_path),
        "null_sha256": sha256_file(null_path) if null_path else None,
        "microsteps": int(args.microsteps),
        "state_dim": int(args.state_dim),
        "seed_index": int(args.seed_index),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "events_per_epoch": int(args.events_per_epoch),
        "train_eval_events": int(args.train_eval_events),
        "max_epochs": int(args.max_epochs),
        "min_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "min_relative_improvement": float(args.min_relative_improvement),
        "gradient_clip": float(args.gradient_clip),
        "smoke": bool(args.smoke),
        "torch_compile": bool(args.torch_compile),
        "training_device_type": device.type,
    }
    if not args.force and summary_path.exists() and checkpoint_path.exists():
        try:
            prior = json.loads(summary_path.read_text())
            if all(prior.get(key) == value for key, value in current_contract.items()):
                return prior
        except Exception:
            pass

    model_seed = MODEL_SEEDS[args.seed_index]
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    model = build_fixed_grid_model(
        channel_names,
        mask,
        seed=model_seed,
        state_dim=args.state_dim,
        microsteps=args.microsteps,
    ).to(device)
    execution_model = torch.compile(model, dynamic=False) if args.torch_compile else model
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_index = np.flatnonzero(split == 0)
    validation_index = np.flatnonzero(split == 1)
    test_index = np.flatnonzero(split == 2)
    # The ECoG events contain at most 23 rank sets. A one-time 2.2-GB CPU
    # tensor cache for E958 is cheaper and much faster than rebuilding every
    # event in Python on every epoch; batches are still moved to GPU lazily.
    training_tensors = build_event_tensors(ranks[train_index])
    training_cache_index = np.arange(len(train_index), dtype=int)
    # Validation is revisited every epoch. Building padded rank tensors once
    # removes repeated Python event loops without changing any value.
    validation_tensors = build_event_tensors(ranks[validation_index])
    validation_cache_index = np.arange(len(validation_index), dtype=int)
    rng = np.random.default_rng(model_seed + 991)
    best_metric = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(args.max_epochs):
        model.train()
        permutation = rng.permutation(training_cache_index)
        if args.events_per_epoch > 0:
            permutation = permutation[: min(len(permutation), args.events_per_epoch)]
        train_loss_sum = 0.0
        train_decisions = 0
        for begin in range(0, len(permutation), args.batch_size):
            chosen = permutation[begin:begin + args.batch_size]
            batch = {
                name: value[torch.as_tensor(chosen, dtype=torch.long)].to(device, non_blocking=True)
                for name, value in training_tensors.items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits, stop = execution_model(batch["x"], batch["recruited"], batch["valid"])
            loss, _, _ = next_rank_stop_loss(
                logits, stop, batch["target"], batch["available"], batch["valid"], batch["is_last"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            decisions = int((batch["valid"] & ~batch["is_last"]).sum())
            train_loss_sum += float(loss.detach()) * decisions
            train_decisions += decisions

        validation = evaluate(
            execution_model, validation_tensors, validation_cache_index, device, args.eval_batch_size
        )
        metric = float(validation["contact_nll"])
        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_decisions, 1),
            "train_decisions": train_decisions,
            "validation_contact_nll": metric,
            "validation_top1": validation["top1"],
        })
        improved = metric < best_metric * (1.0 - args.min_relative_improvement)
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= args.min_epochs and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    final_hash = state_hash(model)
    if args.train_eval_events > 0 and len(train_index) > args.train_eval_events:
        stride = max(1, len(train_index) // args.train_eval_events)
        train_evaluation_index = training_cache_index[::stride][:args.train_eval_events]
    else:
        train_evaluation_index = training_cache_index
    train_metrics = evaluate(
        execution_model, training_tensors, train_evaluation_index, device, args.eval_batch_size
    )
    train_metrics["subset_of_train"] = bool(len(train_evaluation_index) < len(train_index))
    train_metrics["n_train_total"] = int(len(train_index))
    validation_metrics = evaluate(
        execution_model, validation_tensors, validation_cache_index, device, args.eval_batch_size
    )
    test_tensors = build_event_tensors(real_ranks[test_index])
    test_metrics = evaluate(
        execution_model, test_tensors, np.arange(len(test_index), dtype=int), device, args.eval_batch_size
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / "checkpoint.tmp.pt"
    torch.save({
        "schema": "topic5_ecog_grid_checkpoint_v0.1",
        "subject": args.subject,
        "family": args.family,
        "graph_id": graph_id,
        "seed_index": args.seed_index,
        "model_seed": model_seed,
        "microsteps": args.microsteps,
        "state_dim": args.state_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "events_per_epoch": args.events_per_epoch,
        "train_eval_events": args.train_eval_events,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_relative_improvement": args.min_relative_improvement,
        "gradient_clip": args.gradient_clip,
        "smoke": bool(args.smoke),
        "torch_compile": bool(args.torch_compile),
        "training_device_type": device.type,
        "top1_contract": TOP1_CONTRACT,
        "channel_names": channel_names,
        "mask": mask,
        "state_dict": best_state,
        "events_sha256": sha256_file(event_path),
        "graph_sha256": sha256_file(graph_path),
        "null_sha256": sha256_file(null_path) if null_path else None,
    }, temporary)
    temporary.replace(checkpoint_path)
    summary = {
        "schema": "topic5_ecog_graph_unit_v0.1",
        "subject": args.subject,
        "family": args.family,
        "graph_id": graph_id,
        "seed_index": args.seed_index,
        "model_seed": model_seed,
        "microsteps": args.microsteps,
        "state_dim": args.state_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "events_per_epoch": args.events_per_epoch,
        "train_eval_events": args.train_eval_events,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_relative_improvement": args.min_relative_improvement,
        "gradient_clip": args.gradient_clip,
        "smoke": bool(args.smoke),
        "torch_compile": bool(args.torch_compile),
        "training_device_type": device.type,
        "top1_contract": TOP1_CONTRACT,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "initial_parameter_sha256": initial_hash,
        "best_parameter_sha256": final_hash,
        "events_path": str(event_path),
        "events_sha256": sha256_file(event_path),
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "null_path": str(null_path) if null_path else None,
        "null_sha256": sha256_file(null_path) if null_path else None,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "runtime_sec": float(time.time() - started),
    }
    temp_summary = out_dir / "summary.tmp.json"
    temp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temp_summary.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=("958", "1084"))
    parser.add_argument("--family", required=True, choices=("TRUE_GRID", "WRONG_GRID", "DEGREE_RANDOM", "SUFFIX_SHUFFLED"))
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--seed-index", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microsteps", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--events-per-epoch", type=int, default=32768)
    parser.add_argument("--train-eval-events", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--cache-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"))
    parser.add_argument("--output-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_epochs = 2
        args.min_epochs = 2
        args.patience = 2
        args.events_per_epoch = 2048
    summary = train_unit(args)
    print(json.dumps({
        "subject": summary["subject"],
        "family": summary["family"],
        "graph_id": summary["graph_id"],
        "seed_index": summary["seed_index"],
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "test_contact_nll": summary["test"]["contact_nll"],
        "test_top1": summary["test"]["top1"],
        "runtime_sec": summary["runtime_sec"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
