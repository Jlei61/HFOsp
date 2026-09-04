#!/usr/bin/env python3
"""Train and score the S_P state on the frozen wiring-economy tissue decoder (one subject, one seed).

Decoder unit ``k`` (retrained under the recorded-time split) is paired with
state seed ``k`` so that the seed spread contains decoder variance as well.
TRAIN + STATE_SELECTION only; nothing here reads development targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.v034_spatial_state.contracts import TrainConfig, seed_before_model_construction  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.data import load_human_spatial_data  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_state import WEStateConfig, run_subject  # noqa: E402

DECODER_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/we_decoder")
OUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/we_state")
FITS = {
    "epilepsiae_253": "epilepsiae_253__own_a",
    "epilepsiae_1146": "epilepsiae_1146__shared",
    "epilepsiae_548": "epilepsiae_548__shared",
    "epilepsiae_583": "epilepsiae_583__shared",
    "epilepsiae_922": "epilepsiae_922__own_a",
}
ARM = "L3_LOCAL_PLUS_LEARNED_LR"
BASE_SEED = 20260903


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", choices=tuple(FITS), required=True)
    ap.add_argument("--decoder-seed", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    state_seed = BASE_SEED + args.decoder_seed
    seed_before_model_construction(state_seed)
    fit = FITS[args.subject]
    bundle = load_frozen_decoder(DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
                                 DECODER_ROOT / "cache" / fit, device=device)
    data = load_human_spatial_data(args.subject, train_config=TrainConfig(max_steps=900, seed=state_seed))
    config = WEStateConfig(seed=state_seed, max_steps=args.max_steps)
    out_dir = args.out_root / (args.tag or "main") / args.subject / f"decoder_seed{args.decoder_seed}"
    card = run_subject(data=data, bundle=bundle, config=config, device=device, out_dir=out_dir, overwrite=args.overwrite)
    m = card["selection_means"]
    print(json.dumps({
        "subject": args.subject, "decoder_seed": args.decoder_seed, "out": str(out_dir),
        "coverage": card["coverage"]["coverage_fraction"],
        "selection_grammar": {k: round(v["grammar"], 4) for k, v in m.items()},
        "stages": {k: {kk: v[kk] for kk in ("best_inner_val", "selected_step", "selected_at_init", "selected_at_budget_edge")}
                   for k, v in card["stages"].items()},
        "elapsed_seconds": round(card["elapsed_seconds"], 1),
    }, indent=1))


if __name__ == "__main__":
    main()
