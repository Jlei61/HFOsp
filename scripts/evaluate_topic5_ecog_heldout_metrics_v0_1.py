#!/usr/bin/env python3
"""Extended held-out metrics for one frozen ECoG graph checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_ecog_physical_neighborhood_v0_1 import (  # noqa: E402
    build_fixed_grid_model,
    coordinate_array,
)
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


DISTANCE_BINS = (
    ("up_down_left_right", 0.0, 1.01),
    ("diagonal", 1.01, math.sqrt(2.0) + 0.01),
    ("two_grid_steps", math.sqrt(2.0) + 0.01, 2.01),
    ("farther_than_two_grid_steps", 2.01, math.inf),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accumulator() -> dict[str, float | int]:
    return {"n": 0, "nll_sum": 0.0}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    ranks: np.ndarray,
    xy: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    tensors = build_event_tensors(ranks)
    distance = torch.as_tensor(
        np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1),
        dtype=torch.float32,
        device=device,
    )
    distance_stats = {name: _accumulator() for name, _, _ in DISTANCE_BINS}
    remaining_stats = {name: {"n": 0, "stop_probability_sum": 0.0} for name in ("0", "1", "2", "3", "4+")}
    total_target_nll = 0.0
    total_target_contacts = 0
    top1_hits = 0
    topk_recall_sum = 0.0
    continuing_decisions = 0
    stop_brier_sum = 0.0
    stop_bce_sum = 0.0
    valid_steps = 0
    for begin in range(0, len(ranks), int(batch_size)):
        end = min(len(ranks), begin + int(batch_size))
        batch = {name: value[begin:end].to(device) for name, value in tensors.items()}
        logits, stop_logits = model(batch["x"], batch["recruited"], batch["valid"])
        masked = logits.masked_fill(~batch["available"], -1e9)
        log_probability = torch.log_softmax(masked, dim=-1)
        target = batch["target"].bool()
        predict = batch["valid"] & ~batch["is_last"]
        target_log_probability = -log_probability[target]
        total_target_nll += float(target_log_probability.sum())
        total_target_contacts += int(target.sum())

        top_prediction = masked.argmax(-1, keepdim=True)
        top1_hits += int((target.gather(-1, top_prediction).squeeze(-1) & predict).sum())
        order = torch.argsort(masked, dim=-1, descending=True, stable=True)
        target_size = target.sum(-1)
        positions = torch.arange(masked.shape[-1], device=device).view(1, 1, -1)
        picked = positions < target_size.unsqueeze(-1)
        sorted_target = target.gather(-1, order)
        recall = (sorted_target & picked).sum(-1) / target_size.clamp_min(1)
        topk_recall_sum += float(recall[predict].sum())
        continuing_decisions += int(predict.sum())

        # Per-target distance from the current rank set, not from all previously recruited contacts.
        per_contact_nll = -log_probability
        for step in range(batch["x"].shape[1]):
            current = batch["x"][:, step].bool()
            nearest = distance.unsqueeze(0).masked_fill(~current[:, None, :], math.inf).amin(-1)
            future = target[:, step] & predict[:, step, None]
            for name, low, high in DISTANCE_BINS:
                keep = future & (nearest > low) & (nearest <= high)
                if bool(keep.any()):
                    distance_stats[name]["n"] += int(keep.sum())
                    distance_stats[name]["nll_sum"] += float(per_contact_nll[:, step][keep].sum())

        stop_probability = torch.sigmoid(stop_logits)
        stop_target = batch["is_last"].float()
        valid = batch["valid"]
        stop_brier_sum += float(((stop_probability - stop_target).pow(2) * valid).sum())
        stop_bce_sum += float((torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logits, stop_target, reduction="none"
        ) * valid).sum())
        valid_steps += int(valid.sum())
        lengths = valid.sum(-1, keepdim=True)
        step_index = torch.arange(valid.shape[1], device=device).view(1, -1)
        remaining = lengths - step_index - 1
        for key, keep in (
            ("0", remaining == 0), ("1", remaining == 1),
            ("2", remaining == 2), ("3", remaining == 3), ("4+", remaining >= 4),
        ):
            keep = keep & valid
            remaining_stats[key]["n"] += int(keep.sum())
            remaining_stats[key]["stop_probability_sum"] += float(stop_probability[keep].sum())

    for payload in distance_stats.values():
        payload["mean_target_contact_nll"] = float(payload["nll_sum"] / max(int(payload["n"]), 1))
        del payload["nll_sum"]
    for payload in remaining_stats.values():
        payload["mean_stop_probability"] = float(
            payload["stop_probability_sum"] / max(int(payload["n"]), 1)
        )
        del payload["stop_probability_sum"]
    return {
        "contact_nll_per_true_contact": total_target_nll / max(total_target_contacts, 1),
        "n_true_target_contacts": total_target_contacts,
        "n_continuing_decisions": continuing_decisions,
        "top1_any_next_contact": top1_hits / max(continuing_decisions, 1),
        "top_observed_cardinality_recall": topk_recall_sum / max(continuing_decisions, 1),
        "stop_brier": stop_brier_sum / max(valid_steps, 1),
        "stop_bce": stop_bce_sum / max(valid_steps, 1),
        "n_valid_stop_steps": valid_steps,
        "distance_strata": distance_stats,
        "stop_probability_by_remaining_rank_sets": remaining_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_path = args.summary_path.parent / "heldout_extended_metrics.json"
    summary = json.loads(args.summary_path.read_text())
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_fixed_grid_model(
        checkpoint["channel_names"], np.asarray(checkpoint["mask"], dtype=np.uint8),
        int(checkpoint["model_seed"]), state_dim=int(checkpoint["state_dim"]),
        microsteps=int(checkpoint["microsteps"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    with np.load(event_path, allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"], dtype=np.int16)
        split = np.asarray(events["split"], dtype=np.int8)
        channel_names = [str(value) for value in events["channel_names"].tolist()]
    metrics = evaluate(
        model, ranks[split == 2], coordinate_array(channel_names), device, int(args.batch_size)
    )
    result = {
        "schema": "topic5_ecog_heldout_extended_metrics_v0.1",
        "subject": str(summary["subject"]),
        "family": summary["family"],
        "graph_id": summary["graph_id"],
        "seed_index": int(summary["seed_index"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "events_sha256": sha256_file(event_path),
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
