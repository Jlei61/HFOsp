#!/usr/bin/env python3
"""Backfill training metadata from immutable checkpoints without retraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/training")
FIELDS = (
    "lr", "batch_size", "eval_batch_size", "events_per_epoch", "train_eval_events",
    "max_epochs", "min_epochs", "patience", "min_relative_improvement",
    "gradient_clip", "smoke", "torch_compile", "training_device_type",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repaired = []
    already = []
    for summary_path in sorted(ROOT.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if all(field in summary for field in FIELDS):
            already.append(str(summary_path))
            continue
        checkpoint_path = Path(summary["checkpoint_path"])
        if sha256_file(checkpoint_path) != summary["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["events_sha256"] != summary["events_sha256"]:
            raise RuntimeError(f"event hash mismatch: {summary_path}")
        if checkpoint["graph_sha256"] != summary["graph_sha256"]:
            raise RuntimeError(f"graph hash mismatch: {summary_path}")
        missing_checkpoint = [field for field in FIELDS if field not in checkpoint]
        if missing_checkpoint:
            raise RuntimeError(f"checkpoint lacks {missing_checkpoint}: {checkpoint_path}")
        original_sha = sha256_file(summary_path)
        for field in FIELDS:
            summary[field] = checkpoint[field]
        summary["metadata_repaired_from_checkpoint"] = True
        summary["pre_repair_summary_sha256"] = original_sha
        temporary = summary_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(summary_path)
        repaired.append(str(summary_path))
    payload = {
        "schema": "topic5_ecog_training_summary_metadata_repair_v0.1",
        "n_repaired": len(repaired),
        "n_already_complete": len(already),
        "repaired": repaired,
        "model_or_metric_values_changed": False,
    }
    output = ROOT / "SUMMARY_METADATA_REPAIR.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
