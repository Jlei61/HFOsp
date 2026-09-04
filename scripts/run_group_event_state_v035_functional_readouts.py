#!/usr/bin/env python3
"""Run one full W4 functional-readout unit from a frozen W3 trajectory."""

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
    export_state_trajectory, load_full_mark_data, restore_full_mark_model,
)
from src.topic5_group_event_state.v035.functional_readouts import run_functional_readouts  # noqa: E402


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
    ap.add_argument("--state-unit", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    tag = f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}"
    unit = args.state_unit or (OUTPUT_ROOT / "full_mark_state" / args.subject / tag)
    card = json.loads((unit / "card.json").read_text(encoding="utf-8"))
    fit = FITS[args.subject]
    bundle = load_frozen_decoder(
        DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
        DECODER_ROOT / "cache" / fit,
        device=device,
    )
    rate = args.rate_root / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    adapter = args.adapter_root / args.subject / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}" / "adapter.pt"
    cfg = card.get("config", {})
    event_offsets = tuple(int(value) for value in cfg.get("event_offsets", (1, 5, 20)))
    data = load_full_mark_data(args.subject, bundle, rate, event_offsets=event_offsets)
    # Final rescoring cards may intentionally reference the locked search
    # trajectory instead of copying it.  The card is the provenance contract;
    # a coincidental local file can be stale even when its basename matches.
    trajectory = Path(card.get("state_trajectory", unit / "state_trajectory.npz"))
    if not trajectory.exists():
        model, config = restore_full_mark_model(data, bundle, adapter, Path(card["checkpoint"]), device)
        trajectory = unit / "state_trajectory.npz"
        export_state_trajectory(model, data, config, device, trajectory)
    out = args.out_dir or (OUTPUT_ROOT / "functional_readouts" / args.subject / tag)
    result = run_functional_readouts(data, trajectory, rate, out_dir=out, overwrite=args.overwrite)
    print(json.dumps({"subject": args.subject, "seed": args.state_seed, "out": str(out),
                      "event_horizons": list(result["event_horizons"]),
                      "physical_horizons": list(result["physical_horizons"])}, indent=2))


if __name__ == "__main__":
    main()
