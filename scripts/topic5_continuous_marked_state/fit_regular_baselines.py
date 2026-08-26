#!/usr/bin/env python3
"""Fit deterministic frozen event-history baselines before any T1 run."""
from __future__ import annotations

import argparse
import json

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import (
    fit_regular_history_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True,
                        choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--max-iter", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for subject in args.subjects:
        result = fit_regular_history_baseline(
            subject, max_iter=args.max_iter, overwrite=args.overwrite
        )
        print(json.dumps({
            "subject": subject,
            "selected_weight_decay": result["selected_weight_decay"],
            "train_joint_nll": result["train"]["joint_nll"],
            "sealed_opened": result["sealed_opened"],
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
