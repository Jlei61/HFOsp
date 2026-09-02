#!/usr/bin/env python3
"""Recompute held-out top-1 with the tied-next-set contract for frozen checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_ecog_graph_unit_v0_1 import (  # noqa: E402
    TOP1_CONTRACT,
    evaluate,
)
from src.topic5_ecog_physical_neighborhood_v0_1 import build_fixed_grid_model  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    parser.add_argument("--cache-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--subjects", nargs="+", default=("958", "1084"), choices=("958", "1084")
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    repaired: list[dict[str, object]] = []
    current = 0
    maximum_nll_difference = 0.0
    for summary_path in sorted(args.training_root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if bool(summary.get("smoke", False)):
            continue
        if str(summary.get("subject")) not in set(args.subjects):
            continue
        if summary.get("top1_contract") == TOP1_CONTRACT:
            current += 1
            continue
        checkpoint_path = Path(summary["checkpoint_path"])
        if sha256_file(checkpoint_path) != summary["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_fixed_grid_model(
            checkpoint["channel_names"], np.asarray(checkpoint["mask"], dtype=np.uint8),
            int(checkpoint["model_seed"]), state_dim=int(checkpoint["state_dim"]),
            microsteps=int(checkpoint["microsteps"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        event_path = args.cache_root / str(summary["subject"]) / "events.npz"
        if sha256_file(event_path) != summary["events_sha256"]:
            raise RuntimeError(f"event hash mismatch: {event_path}")
        with np.load(event_path, allow_pickle=False) as events:
            ranks = np.asarray(events["ranks"], dtype=np.int16)
            split = np.asarray(events["split"], dtype=np.int8)
        test_tensors = build_event_tensors(ranks[split == 2])
        result = evaluate(
            model, test_tensors, np.arange(int(np.sum(split == 2)), dtype=int),
            device, int(args.batch_size),
        )
        nll_difference = abs(float(result["contact_nll"]) - float(summary["test"]["contact_nll"]))
        maximum_nll_difference = max(maximum_nll_difference, nll_difference)
        if nll_difference > 1e-6:
            raise RuntimeError(f"held-out NLL changed by {nll_difference}: {summary_path}")
        old_top1 = float(summary["test"]["top1"])
        original_sha = sha256_file(summary_path)
        summary["test"]["top1"] = float(result["top1"])
        summary["top1_contract"] = TOP1_CONTRACT
        summary["tied_top1_repaired_from_checkpoint"] = True
        summary["pre_tied_top1_repair_summary_sha256"] = original_sha
        temporary = summary_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(summary_path)
        repaired.append({
            "summary_path": str(summary_path),
            "old_test_top1": old_top1,
            "new_test_top1": float(result["top1"]),
            "heldout_nll_absolute_difference": nll_difference,
        })
        del model, test_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "schema": "topic5_ecog_tied_top1_repair_v0.1",
        "top1_contract": TOP1_CONTRACT,
        "n_repaired": len(repaired),
        "n_already_current": current,
        "maximum_heldout_nll_absolute_difference": maximum_nll_difference,
        "model_parameters_changed": False,
        "repaired": repaired,
    }
    output = args.training_root / "TIED_TOP1_REPAIR.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "repaired"}, indent=2))


if __name__ == "__main__":
    main()
