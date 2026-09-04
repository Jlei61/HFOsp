#!/usr/bin/env python3
"""Train one frozen-decoder per-step lag/energy/waveform assay unit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    DECODER_ROOT,
    OUTPUT_ROOT,
    V035_DECODER_FITS,
)
from src.topic5_group_event_state.v035.full_mark_state import load_full_mark_data  # noqa: E402
from src.topic5_group_event_state.v035.stepwise_auxiliary import AuxiliaryConfig, run_auxiliary_heads  # noqa: E402


FITS = V035_DECODER_FITS
ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", choices=tuple(FITS), required=True)
    ap.add_argument("--decoder-seed", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--state-seed", type=int, required=True)
    ap.add_argument("--batch-events", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    fit = FITS[args.subject]
    bundle = load_frozen_decoder(
        DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
        DECODER_ROOT / "cache" / fit,
        device=device,
    )
    rate = OUTPUT_ROOT / "dynamic_rate" / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    trajectory = (OUTPUT_ROOT / "full_mark_state" / args.subject /
                  f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}" / "state_trajectory.npz")
    data = load_full_mark_data(args.subject, bundle, rate)
    out = (OUTPUT_ROOT / "stepwise_auxiliary" / args.subject /
           f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}")
    config = AuxiliaryConfig(batch_events=args.batch_events, seed=args.state_seed)
    card = run_auxiliary_heads(data, bundle, trajectory, config, device=device, out_dir=out,
                               overwrite=args.overwrite)
    print(json.dumps({"subject": args.subject, "state_seed": args.state_seed,
                      "output": str(out), "stages": card["stages"]}, indent=2))


if __name__ == "__main__":
    main()
