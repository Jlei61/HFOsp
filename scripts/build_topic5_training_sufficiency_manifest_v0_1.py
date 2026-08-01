#!/usr/bin/env python3
"""Build the frozen run manifests for the Topic 5 sufficiency audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_ROOT = Path("results/topic5_rnn_training_sufficiency_v0_1")
DATASET_AUDIT = (
    ROOT
    / "results/topic5_interictal_rank_distribution/dataset_v0_4/subject_audit.csv"
)
SEEDS = (20260725, 20260726, 20260727)
OBJECTIVES = (
    "teacher_forced_one_step",
    "self_fed_2step",
    "self_fed_3step",
    "scheduled_sampling",
)


def _subjects() -> list[str]:
    frame = pd.read_csv(DATASET_AUDIT)
    subjects = sorted(frame.loc[frame.status.eq("ok"), "subject"].astype(str))
    if len(subjects) != 34:
        raise RuntimeError(f"expected 34 sealed subjects, found {len(subjects)}")
    return subjects


def _dev_cell(*, cycles, updates, hidden, lr, optimizer, weight_decay, batch, seed, objective="teacher_forced_one_step"):
    name = (
        f"c{cycles}_u{updates}_h{hidden}_lr{lr:g}_{optimizer}"
        f"_wd{weight_decay:g}_b{batch}_{objective}/seed_{seed}"
    )
    return {
        "cell": name,
        "script": "development",
        "args": {
            "cycles": cycles,
            "updates_per_patient": updates,
            "hidden_size": hidden,
            "learning_rate": lr,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "batch_size": batch,
            "seed": seed,
            "objective": objective,
        },
    }


def _loso_cell(
    *,
    mode,
    condition,
    subject,
    seed,
    cycles,
    updates,
    objective,
    hidden=32,
    lr=1e-3,
    optimizer="adamw",
    weight_decay=1e-4,
    batch=1024,
    offset_cycles=8,
    offset_snapshots=(4, 8),
    rollout=True,
    save_checkpoint=False,
    from_checkpoint=None,
    gpu_memory_fraction=None,
):
    extra = {} if from_checkpoint is None else {"from_checkpoint": from_checkpoint}
    if gpu_memory_fraction is not None:
        extra["gpu_memory_fraction"] = float(gpu_memory_fraction)
    return {
        "cell": f"{condition}/seed_{seed}/{subject}",
        "script": "loso",
        "args": {
            **extra,
            "heldout_subject": subject,
            "mode": mode,
            "condition": condition,
            "cycles": cycles,
            "updates_per_patient": updates,
            "offset_cycles": offset_cycles,
            "offset_snapshot_cycles": list(offset_snapshots),
            "hidden_size": hidden,
            "learning_rate": lr,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "batch_size": batch,
            "seed": seed,
            "objective": objective,
            "rollout": bool(rollout),
            "save_checkpoint": bool(save_checkpoint),
        },
    }


def build_b1(args) -> dict:
    cells = []
    for hidden in (32, 64):
        for updates in (8, 32):
            for seed in SEEDS:
                cells.append(
                    _dev_cell(
                        cycles=args.max_cycles,
                        updates=updates,
                        hidden=hidden,
                        lr=1e-3,
                        optimizer="adamw",
                        weight_decay=1e-4,
                        batch=1024,
                        seed=seed,
                    )
                )
    return {
        "phase": "b1_training_budget",
        "root": str(RESULT_ROOT / "development/b1_budget"),
        "notes": (
            "coverage cycles {1,2,4} are read from one run of max_cycles; "
            "learning rate fixed at 1e-3; hidden 32 primary, 64 sensitivity"
        ),
        "cells": cells,
    }


def build_b1x(args) -> dict:
    """Extend only the current best budget when 4 cycles are still improving."""
    cells = [
        _dev_cell(
            cycles=args.max_cycles,
            updates=args.updates,
            hidden=hidden,
            lr=1e-3,
            optimizer="adamw",
            weight_decay=1e-4,
            batch=1024,
            seed=seed,
        )
        for hidden in args.hidden
        for seed in SEEDS
    ]
    return {
        "phase": "b1x_extended_budget",
        "root": str(RESULT_ROOT / "development/b1_budget_extended"),
        "notes": (
            "cycles 1-4 of this run reproduce the 4-cycle run exactly because "
            "coverage cycles are nested under the same seed"
        ),
        "cells": cells,
    }


def build_b2(args) -> dict:
    grid = [
        ("adamw", 0.0),
        ("adamw", 1e-4),
        ("adam", 0.0),
    ]
    cells = []
    for optimizer, weight_decay in grid:
        for lr in (3e-4, 1e-3, 3e-3):
            for seed in SEEDS:
                cells.append(
                    _dev_cell(
                        cycles=args.cycles,
                        updates=args.updates,
                        hidden=32,
                        lr=lr,
                        optimizer=optimizer,
                        weight_decay=weight_decay,
                        batch=1024,
                        seed=seed,
                    )
                )
    return {
        "phase": "b2_learning_rate",
        "root": str(RESULT_ROOT / "development/b2_learning_rate"),
        "notes": "AdamW primary, Adam is the single optimizer sensitivity at wd=0",
        "cells": cells,
    }


def build_b3(args) -> dict:
    cells = [
        _dev_cell(
            cycles=args.cycles,
            updates=args.updates,
            hidden=32,
            lr=args.learning_rate,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            batch=batch,
            seed=SEEDS[0],
        )
        for batch in (512, 1024)
    ]
    return {
        "phase": "b3_chunk_parity",
        "root": str(RESULT_ROOT / "development/b3_chunk_parity"),
        "notes": (
            "engineering parity only: identical seed and identical update "
            "boundaries, teacher-forced objective"
        ),
        "cells": cells,
    }


def build_b1c(args) -> dict:
    subjects = _subjects()
    cells = []
    for budget in args.budget:
        cycles, updates = (int(value) for value in budget.split(":"))
        condition = f"loso_dev_c{cycles}_u{updates}"
        for seed in SEEDS:
            for subject in subjects:
                cells.append(
                    _loso_cell(
                        mode="development",
                        condition=condition,
                        subject=subject,
                        seed=seed,
                        cycles=cycles,
                        updates=updates,
                        objective="teacher_forced_one_step",
                        gpu_memory_fraction=args.gpu_memory_fraction,
                        lr=args.learning_rate,
                        optimizer=args.optimizer,
                        weight_decay=args.weight_decay,
                        rollout=False,
                    )
                )
    return {
        "phase": "b1c_loso_development_confirmation",
        "root": str(RESULT_ROOT / "development/b1c_loso_confirm"),
        "notes": (
            "structure-faithful confirmation of the shared budget; local-offset "
            "calibration budgets 4 and 8 are read from one calibration run"
        ),
        "cells": cells,
    }


def build_c(args) -> dict:
    subjects = _subjects()
    cells = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for subject in subjects:
                cells.append(
                    _loso_cell(
                        mode="development",
                        condition=f"objective_{objective}",
                        subject=subject,
                        seed=seed,
                        cycles=args.cycles,
                        updates=args.updates,
                        objective=objective,
                        gpu_memory_fraction=args.gpu_memory_fraction,
                        lr=args.learning_rate,
                        optimizer=args.optimizer,
                        weight_decay=args.weight_decay,
                        offset_cycles=args.offset_cycles,
                        offset_snapshots=(args.offset_cycles,),
                        rollout=True,
                    )
                )
    return {
        "phase": "c_objective_sufficiency",
        "root": str(RESULT_ROOT / "development/c_objectives"),
        "notes": (
            "identical parameter count, train events, patient weighting and "
            "optimizer budget across objectives; only the fed history token differs"
        ),
        "cells": cells,
    }


ARCHIVED_LINEAR_STATE = (
    "results/topic5_ordered_history_architecture_audit/formal/"
    "architecture_controls_formal_20260729/linear_state"
)


def build_d(args) -> dict:
    subjects = _subjects()
    frozen = {
        "cycles": args.cycles,
        "updates": args.updates,
        "offset_cycles": args.offset_cycles,
        "lr": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
    }
    conditions = {
        # the published reference keeps the published hyperparameters; it is
        # loaded from its archived checkpoint rather than retrained
        "current_teacher_forced_reference": {
            "cycles": 1,
            "updates": 8,
            "objective": "teacher_forced_one_step",
            "offset_cycles": 4,
            "lr": 1e-3,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "from_archive": True,
        },
        "converged_teacher_forced": {
            **frozen,
            "objective": "teacher_forced_one_step",
        },
        "best_rollout_aware": {
            **frozen,
            "objective": args.objective,
        },
    }
    cells = []
    for condition, setting in conditions.items():
        from_archive = bool(setting.get("from_archive"))
        for seed in SEEDS:
            for subject in subjects:
                checkpoint = (
                    f"{ARCHIVED_LINEAR_STATE}/seed_{seed}/{subject}/"
                    "linear_state_checkpoint.pt"
                    if from_archive
                    else None
                )
                if from_archive and not (ROOT / checkpoint).is_file():
                    raise RuntimeError(f"archived checkpoint missing: {checkpoint}")
                cells.append(
                    _loso_cell(
                        mode="formal",
                        condition=condition,
                        subject=subject,
                        seed=seed,
                        cycles=setting["cycles"],
                        updates=setting["updates"],
                        objective=setting["objective"],
                        gpu_memory_fraction=args.gpu_memory_fraction,
                        lr=setting["lr"],
                        optimizer=setting["optimizer"],
                        weight_decay=setting["weight_decay"],
                        offset_cycles=setting["offset_cycles"],
                        offset_snapshots=(setting["offset_cycles"],),
                        rollout=True,
                        save_checkpoint=not from_archive,
                        from_checkpoint=checkpoint,
                    )
                )
    return {
        "phase": "d_formal_confirmation",
        "root": str(RESULT_ROOT / "formal"),
        "notes": (
            "outer heldout20 read once; the published reference condition "
            "evaluates the archived frozen checkpoint itself rather than a "
            "retrained copy; the static_only rollout is produced inside every "
            "cell and must be byte-identical across conditions"
        ),
        "cells": cells,
    }


BUILDERS = {
    "b1": build_b1,
    "b1x": build_b1x,
    "b2": build_b2,
    "b3": build_b3,
    "b1c": build_b1c,
    "c": build_c,
    "d": build_d,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--offset-cycles", type=int, default=8)
    parser.add_argument("--objective", default="scheduled_sampling")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "adam"))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=None,
        help="per-process GPU cap; keep workers x fraction below 1.0",
    )
    parser.add_argument(
        "--budget",
        nargs="+",
        default=["4:32", "2:32"],
        help="cycles:updates pairs for the LOSO development confirmation",
    )
    args = parser.parse_args()

    manifest = BUILDERS[args.phase](args)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": manifest["phase"],
                "root": manifest["root"],
                "n_cells": len(manifest["cells"]),
                "manifest": str(
                    out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
