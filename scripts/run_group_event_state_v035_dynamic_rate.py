#!/usr/bin/env python3
"""Run one real-patient v0.3.5 dynamic-rate unit."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from src.topic5_group_event_state.v035.contracts import INPUT_ROOT, OUTPUT_ROOT, RateTrainConfig  # noqa: E402
from src.topic5_group_event_state.v035.dynamic_rate import load_rate_data, run_rate_subject  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    available = tuple(sorted(path.parent.name for path in INPUT_ROOT.glob("*/manifest_v3.json")))
    ap.add_argument("--subject", choices=available, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-root", type=Path, default=OUTPUT_ROOT / "dynamic_rate")
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--hold-selection", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    overrides = {}
    if args.config_json is not None:
        overrides = json.loads(args.config_json.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(RateTrainConfig)} - {"seed"}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"unknown RateTrainConfig fields: {unknown}")
        for key in ("horizons_seconds", "taus_seconds"):
            if key in overrides:
                overrides[key] = tuple(float(v) for v in overrides[key])
    cfg = RateTrainConfig(seed=args.seed, **overrides)
    data = load_rate_data(args.subject, cfg)
    out = args.out_root / args.subject / f"seed{args.seed}"
    card = run_rate_subject(
        data, cfg, device=torch.device(args.device), out_dir=out,
        overwrite=args.overwrite, report_selection=not args.hold_selection,
    )
    print(json.dumps({"subject": args.subject, "seed": args.seed, "out": str(out),
                      "selection_arms": card["selection_arms"], "stages": {
                          k: {x: v[x] for x in ("selected_step", "steps_run", "selected_at_init", "selected_at_budget_edge")}
                          for k, v in card["stages"].items()},
                      "elapsed_seconds": card["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
