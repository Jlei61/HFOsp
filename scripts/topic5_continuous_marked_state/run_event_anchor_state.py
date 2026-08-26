#!/usr/bin/env python3
"""Run the event-anchor T1/T2-real/T2-placebo development prototype."""
from __future__ import annotations

import argparse
import json
import os

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import BridgeArrays
from src.topic5_continuous_marked_state.event_anchor import fit_event_anchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--arm", required=True, choices=("t1", "t2_real", "t2_placebo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-minutes", type=float, default=60.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    arrays = BridgeArrays.load(
        contract.RESULT_ROOT / "bridge/features" / f"{args.subject}.npz"
    )
    name = f"{args.subject}__{args.arm}__tau{args.tau_minutes:g}m__s{args.seed}.json"
    output = contract.RESULT_ROOT / "state_smoke" / name
    if output.exists() and not args.overwrite:
        print(json.dumps({"status": "SKIPPED", "path": str(output)}))
        return
    result = fit_event_anchor(
        arrays, args.arm, seed=args.seed, tau_minutes=args.tau_minutes,
        epochs=args.epochs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, output)
    print(json.dumps({"status": "DONE", "path": str(output),
                      "filtered": result["validation_filtered"],
                      "correction_off": result["validation_correction_off_from_start"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
