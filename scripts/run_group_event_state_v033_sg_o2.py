#!/usr/bin/env python3
"""Run one staged human S_G O2 architecture cell or emit its search plan."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v033_training_lab.sg_o2 import (  # noqa: E402
    DEPTHS,
    INITS,
    NORMS,
    O2_ROOT,
    ROUTINGS,
    TUNING_SUBJECTS,
    WIDTHS,
    SGO2ArchConfig,
    SGO2TrainConfig,
    run_sg_o2_cell,
    staged_o2_plan,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-stage-plan", type=Path)
    ap.add_argument("--subject", choices=TUNING_SUBJECTS)
    ap.add_argument("--lease", type=Path)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--width", type=int, choices=WIDTHS, default=32)
    ap.add_argument("--depth", type=int, choices=DEPTHS, default=2)
    ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--norm", choices=NORMS, default="pre")
    ap.add_argument("--init", choices=INITS, default="xavier")
    ap.add_argument("--update-gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--input-routing", choices=ROUTINGS, default="joint")
    ap.add_argument("--write-width", type=int, default=4)
    ap.add_argument("--adapter-rank", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--pair-batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--smoke-train-anchors", type=int)
    ap.add_argument("--smoke-inner-anchors", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.emit_stage_plan is not None:
        atomic_write_json(args.emit_stage_plan, staged_o2_plan())
        print(args.emit_stage_plan)
        return
    if args.subject is None or args.lease is None:
        ap.error("--subject and --lease are required unless --emit-stage-plan is used")
    if args.smoke_train_anchors is None or args.smoke_inner_anchors is None:
        ap.error("current grant is smoke-only; both smoke anchor caps are required")

    arch = SGO2ArchConfig(
        width=args.width, depth=args.depth, residual=args.residual,
        norm=args.norm, init=args.init, update_gate=args.update_gate,
        input_routing=args.input_routing, write_width=args.write_width,
        adapter_rank=args.adapter_rank,
    )
    cfg = SGO2TrainConfig(
        max_steps=args.max_steps, patience=args.patience,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip, pair_batch_size=args.pair_batch_size,
        seed=args.seed, smoke_train_anchors=args.smoke_train_anchors,
        smoke_inner_anchors=args.smoke_inner_anchors,
    )
    cell = (
        f"w{arch.width}_d{arch.depth}_r{int(arch.residual)}_{arch.norm}_"
        f"{arch.init}_g{int(arch.update_gate)}_{arch.input_routing}_seed{cfg.seed}"
    )
    out = args.output_dir or O2_ROOT / args.subject / "resource_smoke" / cell
    card = run_sg_o2_cell(
        args.subject, arch=arch, train_cfg=cfg,
        device=torch.device(args.device), output_dir=out,
        lease_path=args.lease, overwrite=args.overwrite,
    )
    print(card["status"], card["training"]["selected_inner_gain"], out)


if __name__ == "__main__":
    main()

