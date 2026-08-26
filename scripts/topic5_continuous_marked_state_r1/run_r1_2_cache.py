#!/usr/bin/env python3
"""Build one subject's frozen-observer full-anchor R1.2 cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import build_full_anchor_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True, choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-batch-size", type=int, default=8)
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    args = parser.parse_args()
    result = build_full_anchor_cache(
        args.subject, device=args.device,
        anchor_batch_size=args.anchor_batch_size,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
