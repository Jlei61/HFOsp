#!/usr/bin/env python3
"""Build inventory-derived R1.2 coverage for one pilot subject."""
from __future__ import annotations

import argparse
import json

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import write_full_admissible_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.PILOT_SUBJECTS)
    args = parser.parse_args()
    print(json.dumps(write_full_admissible_coverage(args.subject), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
