#!/usr/bin/env python3
"""Run the small R1.6 optimizer synthetic matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_synthetic import (
    run_optimizer_synthetic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--truths", nargs="+", default=["positive", "zero", "reversed"])
    parser.add_argument("--n-anchors", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    rows = [
        run_optimizer_synthetic(
            seed=seed, truth=truth, n_anchors=args.n_anchors,
            epochs=args.epochs, optimizer_name=args.optimizer,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            grad_clip_norm=(
                None if args.grad_clip_norm <= 0 else args.grad_clip_norm
            ),
            warmup_fraction=args.warmup_fraction,
        )
        for truth in args.truths for seed in args.seeds
    ]
    summary = {
        "status": "COMPLETE",
        "config_id": args.config_id,
        "rows": rows,
        "by_truth": {
            truth: {
                "recovered": int(sum(
                    row["recovered"] for row in rows if row["truth"] == truth
                )),
                "total": int(sum(row["truth"] == truth for row in rows)),
            }
            for truth in args.truths
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    output = args.output_root / "synthetic" / f"{args.config_id}.json"
    contract.atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
