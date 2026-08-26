#!/usr/bin/env python3
"""Run and persist the R1 T1 synthetic recovery panel."""
from __future__ import annotations

import argparse
import json

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.synthetic_recovery import (
    run_synthetic_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=(4, 5, 6))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    rows = [run_synthetic_recovery(seed, args.epochs) for seed in args.seeds]
    result = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "rows": rows,
        "n_recovered": int(sum(row["recovered"] for row in rows)),
        "n_seeds": int(len(rows)),
        "sealed_opened": False,
    }
    output = contract.RESULT_ROOT / "synthetic" / "t1_recovery.json"
    contract.atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
