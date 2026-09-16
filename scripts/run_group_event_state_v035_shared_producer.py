#!/usr/bin/env python3
"""Train one shared multi-horizon S_N or S_G producer without reading SELECTION."""

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
from src.topic5_group_event_state.v035.contracts import DECODER_ROOT, V035_DECODER_FITS  # noqa: E402
from src.topic5_group_event_state.v035.full_mark_state import FullMarkTrainConfig, load_full_mark_data  # noqa: E402
from src.topic5_group_event_state.v035.shared_state import (  # noqa: E402
    SharedProducerConfig, train_shared_producer, update_checkpoint_registry,
)

ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", choices=tuple(V035_DECODER_FITS), required=True)
    ap.add_argument("--decoder-seed", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--state-seed", type=int, required=True)
    ap.add_argument("--family", choices=("S_N", "S_G"), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate-root", type=Path, required=True)
    ap.add_argument("--adapter-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--chunk-events", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    overrides: dict = {}
    shared_overrides: dict = {}
    if args.config_json is not None:
        raw = json.loads(args.config_json.read_text(encoding="utf-8"))
        overrides = dict(raw.get("base", raw))
        shared_overrides = dict(raw.get("shared", {}))
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    overrides.update(
        objective_family="sn" if args.family == "S_N" else "sg",
        event_innovation_mode="marked_event_only",
        seed=args.state_seed,
        chunk_events=args.chunk_events,
        report_selection=False,
    )
    for key in ("state_taus_seconds", "offset_weights", "event_offsets"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    unknown = sorted(set(overrides) - set(FullMarkTrainConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown FullMarkTrainConfig keys: {unknown}")

    fit = V035_DECODER_FITS[args.subject]
    device = torch.device(args.device)
    bundle = load_frozen_decoder(
        DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
        DECODER_ROOT / "cache" / fit,
        device=device,
    )
    rate = args.rate_root / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    adapter = (
        args.adapter_root / args.subject
        / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}" / "adapter.pt"
    )
    if not rate.exists():
        raise FileNotFoundError(f"shared-split rate prerequisite missing: {rate}")
    if not adapter.exists():
        raise FileNotFoundError(f"causal frozen-decoder adapter missing: {adapter}")
    base = FullMarkTrainConfig(**overrides)
    cfg = SharedProducerConfig(family=args.family, base=base, **shared_overrides)
    data = load_full_mark_data(args.subject, bundle, rate, event_offsets=base.event_offsets)
    out = args.out_root / args.subject / args.family / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}"
    card = train_shared_producer(data, bundle, adapter, cfg, device=device,
                                 out_dir=out, overwrite=args.overwrite)
    registry = update_checkpoint_registry(args.out_root / "checkpoint_registry.json", card)
    print(json.dumps({
        "subject": args.subject, "family": args.family, "state_seed": args.state_seed,
        "selected_epoch": card["selected_epoch"], "best_inner_loss": card["best_inner_loss"],
        "elapsed_seconds": card["elapsed_seconds"], "registry_entries": len(registry["entries"]),
        "selection_targets_read": card["selection_targets_read"],
    }, indent=2))


if __name__ == "__main__":
    main()
