#!/usr/bin/env python3
"""Run one bounded, synthetic-only S_G Level-2 recovery card."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.v032_eval.contract import atomic_json, load_eval_config  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import dgp as D  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import power as P  # noqa: E402
from src.topic5_group_event_state.v033_evaluator.grammar_recovery import RECIPES, recipe, run_recovery  # noqa: E402
from src.topic5_group_event_state.v033_evaluator.scaffold import load_real_scaffold  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", required=True, choices=tuple(RECIPES))
    ap.add_argument("--kind", choices=("D0", "D3"), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--output-root", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_3/training_lab/sg_synthetic_recovery"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.replicate < 0:
        ap.error("replicate must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = args.output_root / "cards" / f"{args.kind}_rep{args.replicate:03d}_{args.recipe}.json"
    if out.exists() and not args.overwrite:
        print(out)
        return
    cfg = load_eval_config(args.config)
    scaffold = load_real_scaffold(args.subject, cfg, carry="session")
    bc, bg = ((0.0, 0.0) if args.kind == "D0" else (0.7, 2.5))
    spec = P.ReplicateSpec(args.kind, bc, bg, args.replicate).resolved()
    data = D.generate(scaffold, args.kind, beta_count=bc, beta_grammar=bg,
                      generator_seed=spec["generator_seed"], noise_seed=spec["noise_seed"])
    card = run_recovery(scaffold, data, cfg=recipe(args.recipe), horizon=args.horizon,
                        seed=spec["estimator_seed"], device=device)
    card.update({"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "threads": {name: os.environ.get(name) for name in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}})
    atomic_json(out, card)
    print(json.dumps({"output": str(out), "gain": card["gain_level2"],
                      "ci": [card["ci_lower"], card["ci_upper"]],
                      "selected_step": card["selected_step"], "resources": card["resources"]}, indent=2))


if __name__ == "__main__":
    main()
