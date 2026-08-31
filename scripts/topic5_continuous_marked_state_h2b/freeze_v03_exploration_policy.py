#!/usr/bin/env python3
"""Freeze the user-directed low-gate H2b v0.3 exploration addendum."""
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
from src.topic5_continuous_marked_state_h2b.v03_exploration_policy import (  # noqa: E402
    DEFAULT_POLICY,
    freeze_exploration_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--output", type=Path,
        default=CANONICAL_V0_3_RESULT_ROOT / "exploration_policy.json",
    )
    args = parser.parse_args()
    frozen = freeze_exploration_policy(args.policy, args.output)
    print(f"FROZEN {frozen['policy_sha256']} {args.output}")


if __name__ == "__main__":
    main()
