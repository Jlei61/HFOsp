#!/usr/bin/env python3
"""Run the v0.3.2 frozen-grammar H2a transfer probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.v032_eval.contract import EvalPaths, load_eval_config  # noqa: E402
from src.topic5_group_event_state.v032_eval.h2a_probe import evaluate_h2a_patient_seed  # noqa: E402
from src.topic5_group_event_state.v032_eval.state_registry import (  # noqa: E402
    complete_seed_entries, load_registry, load_state_bundle,
)
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    p.add_argument("--registry", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_2/shared/frozen_state_registry.json"))
    p.add_argument("--subject", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = load_eval_config(args.config)
    paths = EvalPaths.from_config(cfg)
    tl = load_eval_timeline(args.subject, cfg)
    registry = load_registry(args.registry)
    spec = complete_seed_entries(registry, args.subject)[str(args.seed)]
    state = load_state_bundle(
        spec, subject=args.subject, seed=str(args.seed),
        grid_times=tl.grid.t_anchor, grid_segment=tl.grid.segment_index,
        event_times=tl.event_times, event_segment=tl.event_segment,
    )
    report = evaluate_h2a_patient_seed(
        tl, state, cfg, out_dir=paths.evaluation / "h2a" / args.subject,
        device=torch.device(args.device),
    )
    print(args.subject, args.seed, report["status"], flush=True)


if __name__ == "__main__":
    main()

