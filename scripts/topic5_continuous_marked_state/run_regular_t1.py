#!/usr/bin/env python3
"""Run one full-event regular-observation T0/T1 development fit."""
from __future__ import annotations

import argparse
import json
import os

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import fit_regular_t1
from src.topic5_continuous_marked_state.regular_t1 import REGULAR_T1_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--arm", required=True, choices=(
        "t0_no_observation_state", "t1_regular_observation"
    ))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train-events", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--observation-variant", choices=("spectral", "raw", "both"),
                        default="spectral")
    parser.add_argument("--state-dim", type=int, default=8)
    args = parser.parse_args()
    name = f"{args.subject}__{args.arm}__s{args.seed}.json"
    run_root = (
        contract.RESULT_ROOT / "regular_t1/runs"
        if args.observation_variant == "spectral"
        else contract.RESULT_ROOT / f"regular_t1/{args.observation_variant}_e0/runs"
    )
    output = run_root / name
    if output.exists() and not args.overwrite:
        old = json.loads(output.read_text())
        same_configuration = (
            old.get("regular_t1_revision") == REGULAR_T1_REVISION
            and old.get("observation_variant", "spectral") == args.observation_variant
            and int(old.get("state_dim", -1)) == int(args.state_dim)
            and int(old.get("epochs", -1)) == int(args.epochs)
            and old.get("max_train_events") == args.max_train_events
        )
        if same_configuration:
            print(json.dumps({"status": "SKIPPED", "path": str(output)}))
            return
        raise ValueError(
            f"configuration collision at {output}; use an isolated result path "
            "or --overwrite after explicitly archiving the existing run"
        )
    result = fit_regular_t1(
        args.subject, args.arm, seed=args.seed, epochs=args.epochs,
        max_train_events=args.max_train_events,
        observation_variant=args.observation_variant,
        state_dim=args.state_dim,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, output)
    print(json.dumps({
        "status": "DONE", "path": str(output),
        "filtered": result["validation_filtered"],
        "correction_off": result["validation_correction_off_from_split_start"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
