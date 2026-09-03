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
    RUN_KINDS,
    ROUTINGS,
    STAGES,
    TUNING_SUBJECTS,
    WIDTHS,
    SGO2ArchConfig,
    SGO2TrainConfig,
    freeze_o1_optimizer_recipe,
    run_sg_o2_cell,
    staged_o2_plan,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-stage-plan", type=Path)
    ap.add_argument("--freeze-o1-recipe", type=Path)
    ap.add_argument("--o1-study-manifest", type=Path)
    ap.add_argument("--o1-cell-manifest", type=Path)
    ap.add_argument("--subject", choices=TUNING_SUBJECTS)
    ap.add_argument("--lease", type=Path)
    ap.add_argument("--o1-recipe", type=Path)
    ap.add_argument("--run-kind", choices=RUN_KINDS, default="resource_smoke")
    ap.add_argument("--stage", choices=STAGES, default="S0")
    ap.add_argument("--pairing-id")
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
    ap.add_argument("--pair-batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--smoke-train-anchors", type=int)
    ap.add_argument("--smoke-inner-anchors", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.emit_stage_plan is not None:
        atomic_write_json(args.emit_stage_plan, staged_o2_plan())
        print(args.emit_stage_plan)
        return
    if args.freeze_o1_recipe is not None:
        if args.o1_study_manifest is None or args.o1_cell_manifest is None:
            ap.error("freezing O1 requires --o1-study-manifest and --o1-cell-manifest")
        freeze_o1_optimizer_recipe(
            study_manifest_path=args.o1_study_manifest,
            cell_manifest_path=args.o1_cell_manifest,
            output_path=args.freeze_o1_recipe,
        )
        print(args.freeze_o1_recipe)
        return
    if args.subject is None or args.lease is None or args.o1_recipe is None:
        ap.error("--subject, --lease and --o1-recipe are required for a cell")
    if not args.pairing_id:
        ap.error("--pairing-id is required so all cells in a stage share one seed contract")
    if args.run_kind == "resource_smoke" and (
        args.smoke_train_anchors is None or args.smoke_inner_anchors is None
    ):
        ap.error("resource_smoke requires both smoke anchor caps")
    if args.run_kind == "full_training" and (
        args.smoke_train_anchors is not None or args.smoke_inner_anchors is not None
    ):
        ap.error("full_training forbids smoke caps")

    arch = SGO2ArchConfig(
        width=args.width, depth=args.depth, residual=args.residual,
        norm=args.norm, init=args.init, update_gate=args.update_gate,
        input_routing=args.input_routing, write_width=args.write_width,
        adapter_rank=args.adapter_rank,
    )
    cfg = SGO2TrainConfig(
        max_steps=args.max_steps, patience=args.patience,
        pair_batch_size=args.pair_batch_size, run_kind=args.run_kind,
        seed=args.seed, smoke_train_anchors=args.smoke_train_anchors,
        smoke_inner_anchors=args.smoke_inner_anchors,
    )
    cell = (
        f"w{arch.width}_d{arch.depth}_r{int(arch.residual)}_{arch.norm}_"
        f"{arch.init}_g{int(arch.update_gate)}_{arch.input_routing}_seed{cfg.seed}"
    )
    out = args.output_dir or (
        O2_ROOT / args.subject / args.run_kind / args.pairing_id / args.stage / cell
    )
    card = run_sg_o2_cell(
        args.subject, stage=args.stage, pairing_id=args.pairing_id,
        arch=arch, train_cfg=cfg, o1_recipe_path=args.o1_recipe,
        device=torch.device(args.device), output_dir=out,
        lease_path=args.lease, resume=args.resume, overwrite=args.overwrite,
    )
    print(card["status"], card["training"]["selected_inner_gain"], out)


if __name__ == "__main__":
    main()
