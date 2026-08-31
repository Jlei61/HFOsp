#!/usr/bin/env python3
"""Validate and materialise the immutable H2b v0.3 analysis contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_3_RESULT_ROOT,
)
from src.topic5_continuous_marked_state_h2b.v03_contract import (  # noqa: E402
    DEFAULT_CONTRACT,
    freeze_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--output", type=Path,
        default=CANONICAL_V0_3_RESULT_ROOT / "analysis_contract.json",
    )
    args = parser.parse_args()
    frozen = freeze_contract(args.contract, args.output)
    print(f"FROZEN {frozen['contract_sha256']} {args.output}")


if __name__ == "__main__":
    main()
