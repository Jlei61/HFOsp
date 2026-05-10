"""Topic 1 × Topic 5 Bridge CLI driver."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic 1 × Topic 5 Bridge — Q1 + Q1b + Q3 driver",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="Freeze T0/T1 convention + audit-rerun marker")
    sub.add_parser("per-subject", help="Run per-subject Q1 statistics")
    sub.add_parser("cohort", help="Aggregate cohort + 3-state verdict")
    sub.add_parser("sentinel-442", help="Run Q1b 442 binary-outlier exact tests")
    sub.add_parser("figures", help="Render the 5 locked figures")
    args = parser.parse_args()
    raise NotImplementedError(f"subcommand {args.cmd} pending")


if __name__ == "__main__":
    main()
