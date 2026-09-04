#!/usr/bin/env python3
"""Run frozen W5 seizure transfer for one exact W3 unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import OUTPUT_ROOT  # noqa: E402
from src.topic5_group_event_state.v035.seizure_transfer import run_seizure_transfer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--decoder-seed", type=int, required=True)
    ap.add_argument("--state-seed", type=int, required=True)
    ap.add_argument("--state-unit", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    tag = f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}"
    unit = args.state_unit or (OUTPUT_ROOT / "full_mark_state" / args.subject / tag)
    card = json.loads((unit / "card.json").read_text(encoding="utf-8"))
    trajectory = Path(card.get("state_trajectory", unit / "state_trajectory.npz"))
    if not trajectory.exists():
        raise FileNotFoundError(f"shared frozen W3 trajectory missing: {trajectory}; W4 exporter must finish first")
    rate = OUTPUT_ROOT / "dynamic_rate" / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    out = args.out_dir or (OUTPUT_ROOT / "seizure_transfer" / args.subject / tag)
    card = run_seizure_transfer(args.subject, trajectory, rate, out_dir=out, overwrite=args.overwrite)
    print(json.dumps({"subject": args.subject, "seed": args.state_seed, "out": str(out),
                      "seizures_by_phase": card["distance_survival"]["seizures_by_phase"]}, indent=2))


if __name__ == "__main__": main()
