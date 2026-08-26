#!/usr/bin/env python3
"""Build one denominator-locked frozen-upstream R1.2b node cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2b import (
    R1_2B_SUBJECTS, build_joint_node_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=R1_2B_SUBJECTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-batch-size", type=int, default=16)
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_2b"
    )
    args = parser.parse_args()
    value = build_joint_node_cache(
        args.subject, device=args.device,
        anchor_batch_size=args.anchor_batch_size,
        r1_2_root=args.r1_2_root, output_root=args.output_root,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
