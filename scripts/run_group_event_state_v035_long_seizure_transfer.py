#!/usr/bin/env python3
"""Run frozen H2b transfer on exploratory 30-min to 24-h horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035 import seizure_transfer as assay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--rate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # This is a separate exploratory process, so changing these module-level
    # assay constants cannot alter the registered core H2b outputs.
    horizon = float(args.horizon_seconds)
    if horizon not in (7200.0, 21600.0, 43200.0, 86400.0):
        raise ValueError("unregistered long seizure horizon")
    assay.LEADS_SECONDS = tuple(sorted(set((horizon, horizon / 2.0, min(7200.0, horizon)))))
    assay.RISK_HORIZONS_SECONDS = (horizon,)
    assay.HAZARD_BIN_SECONDS = min(1800.0, horizon / 4.0)
    assay.HAZARD_MAX_SECONDS = horizon
    assay.STATE_SHIFT_MIN_SECONDS = horizon
    card = assay.run_seizure_transfer(
        args.subject, args.trajectory, args.rate,
        out_dir=args.out_dir, overwrite=args.overwrite,
    )
    print(json.dumps({
        "subject": args.subject, "out": str(args.out_dir),
        "risk_horizons_seconds": list(assay.RISK_HORIZONS_SECONDS),
        "early_field_leads_seconds": list(assay.LEADS_SECONDS),
    }, indent=2))


if __name__ == "__main__":
    main()
