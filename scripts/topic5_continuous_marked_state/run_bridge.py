#!/usr/bin/env python3
"""Fit B0--B3 with identical heads and write one atomic result per job."""
from __future__ import annotations

import argparse
import json
import os

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import BridgeArrays, fit_bridge_arm

ARMS = ("b0_history", "b1_spectral", "b2_raw", "b3_both")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    feature_path = contract.RESULT_ROOT / "bridge/features" / f"{args.subject}.npz"
    output = contract.RESULT_ROOT / "bridge/runs" / f"{args.subject}__{args.arm}__s{args.seed}.json"
    if output.exists() and not args.overwrite:
        old = json.loads(output.read_text())
        if (old.get("contract") == contract.REVISION
                and old.get("fit_revision") == contract.FIT_REVISION):
            print(json.dumps({"status": "SKIPPED", "path": str(output)}))
            return
    arrays = BridgeArrays.load(feature_path)
    result = fit_bridge_arm(arrays, args.arm, seed=args.seed, epochs=args.epochs)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, output)
    print(json.dumps({"status": "DONE", "path": str(output),
                      "validation": result["validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
