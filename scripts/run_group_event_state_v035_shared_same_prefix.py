#!/usr/bin/env python3
"""Score state-dependent continuation through the frozen contact decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import (  # noqa: E402
    decoder_tensors, load_frozen_decoder,
)
from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    DECODER_ROOT, FORMAT_PREFIX, V035_DECODER_FITS, atomic_json,
)
from src.topic5_group_event_state.v035.full_mark_state import (  # noqa: E402
    FullMarkStateModel, FullMarkTrainConfig, evaluate_selection, load_full_mark_data,
)

ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--producer-unit", type=Path, required=True)
    ap.add_argument("--decoder-seed", type=int, required=True)
    ap.add_argument("--rate-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    checkpoint_path = args.producer_unit / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["family"] != "S_G":
        raise ValueError("same-prefix continuation is an S_G evaluator")
    subject = checkpoint["subject"]
    raw = dict(checkpoint["config"]["base"])
    for key in ("state_taus_seconds", "offset_weights", "event_offsets"):
        raw[key] = tuple(raw[key])
    cfg = FullMarkTrainConfig(**raw)
    device = torch.device(args.device)
    fit = V035_DECODER_FITS[subject]
    bundle = load_frozen_decoder(
        DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
        DECODER_ROOT / "cache" / fit, device=device,
    )
    rate = args.rate_root / subject / f"seed{cfg.seed}" / "trajectory_and_scores.npz"
    data = load_full_mark_data(subject, bundle, rate, event_offsets=cfg.event_offsets)
    model = FullMarkStateModel(data, bundle, Path(checkpoint["base_adapter"]), cfg, device).to(device)
    current = model.state_dict()
    for key, value in checkpoint["producer"].items():
        current[key] = value.to(device)
    model.load_state_dict(current); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = evaluate_selection(model, data, decoder_tensors(bundle, device), cfg, device)
    card = {
        "format": f"{FORMAT_PREFIX}_shared_same_prefix_v1",
        "subject": subject, "family": "S_G", "producer_checkpoint": str(checkpoint_path),
        "producer_frozen": True, "conditional_continuation": result,
        "scientific_scope": "later recruitment and STOP conditioned on the observed first tied group; not an unconditional next-event score",
        "selection_targets_read": True, "development_targets_read": False,
        "sealed_partition_opened": False, "seizure_outcomes_read": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "card.json", card)
    print(json.dumps({"subject": subject, "offsets": list(result["arms"]),
                      "n_shift_valid": result["n_shift_valid"]}, indent=2))


if __name__ == "__main__":
    main()
