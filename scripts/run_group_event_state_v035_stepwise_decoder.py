#!/usr/bin/env python3
"""Run one real-patient step-wise frozen-decoder unit using its causal q(t)."""

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
from src.topic5_group_event_state.v034_spatial_state.contracts import TrainConfig  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.data import load_human_spatial_data  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v035.contracts import DECODER_ROOT, OUTPUT_ROOT, V035_DECODER_FITS  # noqa: E402
from src.topic5_group_event_state.v035.stepwise_train import StepwiseTrainConfig, run_stepwise_subject  # noqa: E402

FITS = V035_DECODER_FITS
ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", choices=tuple(FITS), required=True)
    ap.add_argument("--decoder-seed", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--state-seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate-root", type=Path, default=OUTPUT_ROOT / "dynamic_rate")
    ap.add_argument("--out-root", type=Path, default=OUTPUT_ROOT / "stepwise_decoder")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    fit = FITS[args.subject]
    device = torch.device(args.device)
    bundle = load_frozen_decoder(DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
                                 DECODER_ROOT / "cache" / fit, device=device)
    data = load_human_spatial_data(args.subject, train_config=TrainConfig(max_steps=900, seed=args.state_seed))
    trajectory = args.rate_root / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    config = StepwiseTrainConfig(seed=args.state_seed)
    out = args.out_root / args.subject / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}"
    card = run_stepwise_subject(data, bundle, trajectory, config, device=device, out_dir=out, overwrite=args.overwrite)
    print(json.dumps({"subject": args.subject, "out": str(out), "selection_means": card["selection_means"],
                      "stages": {k: {x: v[x] for x in ("selected_step", "steps_run", "selected_at_init", "selected_at_budget_edge")}
                                 for k, v in card["stages"].items()},
                      "elapsed_seconds": card["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
