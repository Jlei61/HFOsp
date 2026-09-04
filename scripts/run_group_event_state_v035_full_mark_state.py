#!/usr/bin/env python3
"""Train one v0.3.5 full-event m(t) unit through the step-wise frozen decoder."""

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
from src.topic5_group_event_state.v035.contracts import DECODER_ROOT, OUTPUT_ROOT, V035_DECODER_FITS  # noqa: E402
from src.topic5_group_event_state.v035.full_mark_state import (  # noqa: E402
    INPUT_VIEWS, FullMarkTrainConfig, load_full_mark_data, train_full_mark_subject,
)

FITS = V035_DECODER_FITS
ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", choices=tuple(FITS), required=True)
    ap.add_argument("--decoder-seed", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--state-seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate-root", type=Path, default=OUTPUT_ROOT / "dynamic_rate")
    ap.add_argument("--adapter-root", type=Path, default=OUTPUT_ROOT / "stepwise_decoder")
    ap.add_argument("--out-root", type=Path, default=OUTPUT_ROOT / "full_mark_state")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--chunk-events", type=int, default=256)
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--input-view", choices=INPUT_VIEWS)
    ap.add_argument("--hold-selection", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device); fit = FITS[args.subject]
    bundle = load_frozen_decoder(DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
                                 DECODER_ROOT / "cache" / fit, device=device)
    rate = args.rate_root / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    adapter = args.adapter_root / args.subject / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}" / "adapter.pt"
    overrides = {}
    if args.config_json is not None:
        overrides = json.loads(args.config_json.read_text(encoding="utf-8"))
        allowed = set(FullMarkTrainConfig.__dataclass_fields__)
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"unknown FullMarkTrainConfig keys: {unknown}")
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.input_view is not None:
        overrides["input_view"] = args.input_view
    for key in ("state_taus_seconds", "offset_weights", "event_offsets"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    overrides.update(chunk_events=args.chunk_events, seed=args.state_seed,
                     report_selection=not args.hold_selection)
    cfg = FullMarkTrainConfig(**overrides)
    data = load_full_mark_data(
        args.subject, bundle, rate,
        event_offsets=tuple(int(value) for value in cfg.event_offsets),
    )
    out = args.out_root / args.subject / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}"
    card = train_full_mark_subject(data, bundle, adapter, cfg, device=device, out_dir=out, overwrite=args.overwrite)
    print(json.dumps({"subject": args.subject, "seed": args.state_seed, "out": str(out),
                      "selected_epoch": card["selected_epoch"], "best_inner_loss": card["best_inner_loss"],
                      "selection": card["selection"], "elapsed_seconds": card["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
