#!/usr/bin/env python3
"""Run one W6 long-scale feedback comparison from a frozen W3 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import OUTPUT_ROOT  # noqa: E402
from src.topic5_group_event_state.v035.feedback_models import run_feedback_models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True); ap.add_argument("--decoder-seed", type=int, required=True)
    ap.add_argument("--state-seed", type=int, required=True); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--state-unit", type=Path, default=None,
                    help="Exact registered frozen-state unit; defaults to full_mark_final, then legacy work path")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Exact output unit (default: feedback_models_final/<subject>/<tag>)")
    args = ap.parse_args()
    tag = f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}"
    if args.state_unit is not None:
        unit = args.state_unit
    else:
        final = OUTPUT_ROOT / "full_mark_final" / args.subject / tag
        unit = final if (final / "state_trajectory.npz").exists() else OUTPUT_ROOT / "full_mark_state" / args.subject / tag
    trajectory = unit / "state_trajectory.npz"
    if not trajectory.exists(): raise FileNotFoundError(f"shared frozen W3 trajectory missing: {trajectory}")
    rate = OUTPUT_ROOT / "dynamic_rate" / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    if not rate.exists():
        rate = OUTPUT_ROOT / "dynamic_rate_final" / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    out = args.out_dir or OUTPUT_ROOT / "feedback_models_final" / args.subject / tag
    card = run_feedback_models(args.subject, trajectory, rate, out_dir=out, overwrite=args.overwrite)
    print(json.dumps({"subject": args.subject, "seed": args.state_seed, "out": str(out),
                      "design_status": {k: v["status"] for k, v in card["designs"].items()}}, indent=2))


if __name__ == "__main__": main()
