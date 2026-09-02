#!/usr/bin/env python3
"""Pre-registered calibration-only E253 offset optimiser grid.

The three trials use independent output directories and the same 0--16% fit /
16--20% inner-validation rows.  No state-training, development, seizure or
sealed outcome is loaded by the underlying calibrator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.contact_grammar import (  # noqa: E402
    LegacyGrammarCalibrationConfig,
    calibrate_legacy_contact_grammar,
    select_offset_optimizer_trial,
)
from src.topic5_group_event_state.v033_training_lab.paths import (  # noqa: E402
    AGENT_B_ROOT,
    atomic_write_json,
    current_commit,
    file_hash,
)


SUBJECT = "epilepsiae_253"
OFFSET_LR_GRID = (0.01, 0.03, 0.1)
MAX_EPOCHS = 600
PATIENCE = 30
INNER_NLL_TOLERANCE = 1e-4


def _trial_name(value: float) -> str:
    return f"offset_lr_{value:g}".replace(".", "p")


def _expected_config(lr: float, args: argparse.Namespace) -> LegacyGrammarCalibrationConfig:
    return LegacyGrammarCalibrationConfig(
        batch_size=int(args.batch_size),
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        base_learning_rate=1e-3,  # irrelevant because E253 base is frozen
        offset_learning_rate=float(lr),
        weight_decay=float(args.weight_decay),
        gradient_clip=5.0,
        seed=int(args.seed),
    )


def _validate_existing(report: Mapping[str, Any], cfg: LegacyGrammarCalibrationConfig) -> None:
    if report.get("subject") != SUBJECT or report.get("scientific_use") is not True:
        raise ValueError("existing trial is not a scientific-use E253 calibration")
    if report.get("status") != "COMPLETE_CALIBRATION_PREFIX_ONLY":
        raise ValueError("existing trial is incomplete")
    observed = report.get("history")
    if not isinstance(observed, list) or not observed:
        raise ValueError("existing trial has no complete history")
    checkpoint = report.get("checkpoint")
    if not checkpoint or not Path(checkpoint).is_file() \
            or file_hash(Path(checkpoint)) != report.get("checkpoint_sha256"):
        raise ValueError("existing trial checkpoint is missing or changed")
    # The checkpoint carries the exact config; validation must not trust only a
    # directory name or a rounded LR in the report.
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if payload.get("training", {}).get("config") != asdict(cfg):
        raise ValueError("existing trial config differs from the pre-registered grid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root", type=Path,
        default=AGENT_B_ROOT / "contact_grammar_optimizer_grid_e253_v1",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--overwrite-trials", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    spec = {
        "format": "group_event_contact_grammar_v033_e253_offset_grid",
        "status": "PRE_REGISTERED_CALIBRATION_ONLY",
        "subject": SUBJECT,
        "offset_learning_rates": list(OFFSET_LR_GRID),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "inner_nll_tolerance": INNER_NLL_TOLERANCE,
        "optimizer": "AdamW_offset_only",
        "fit_phase": "recorded_time_0_to_16_percent",
        "selection_phase": "recorded_time_16_to_20_percent",
        "later_phase_read": False,
        "decoder": "frozen_old_next_set_or_STOP",
        "base_trainable": False,
        "source_commit": current_commit(),
    }
    spec_path = root / "grid_spec.json"
    if spec_path.exists():
        existing = json.loads(spec_path.read_text())
        # Source commit is provenance, not an optimisation hyperparameter; an
        # exact resume after a cherry-pick may differ only in this field.
        comparable = dict(existing)
        comparable["source_commit"] = spec["source_commit"]
        if comparable != spec:
            raise ValueError("existing grid spec differs; refusing to mix trials")
    else:
        atomic_write_json(spec_path, spec)

    trials: list[dict[str, Any]] = []
    for lr in OFFSET_LR_GRID:
        trial = _trial_name(lr)
        cfg = _expected_config(lr, args)
        report_path = root / trial / "legacy_contact_grammar_v033.json"
        if report_path.exists() and not args.overwrite_trials:
            report = json.loads(report_path.read_text())
            _validate_existing(report, cfg)
        else:
            report = calibrate_legacy_contact_grammar(
                SUBJECT,
                out_dir=root / trial,
                device=torch.device(args.device),
                cfg=cfg,
                overwrite=bool(args.overwrite_trials),
            )
        trials.append({
            "trial": trial,
            "offset_learning_rate": float(lr),
            "best_inner_validation_event_nll": float(report["best_inner_validation_event_nll"]),
            "training_adequacy": dict(report["training_adequacy"]),
            "checkpoint": str(report["checkpoint"]),
            "checkpoint_sha256": str(report["checkpoint_sha256"]),
            "report": str(report_path),
            "report_sha256": file_hash(report_path),
        })
    selection = select_offset_optimizer_trial(
        trials, tolerance=INNER_NLL_TOLERANCE
    )
    output = {
        "format": "group_event_contact_grammar_v033_e253_offset_grid_selection",
        **selection,
        "trials": trials,
        "grid_spec": str(spec_path),
        "grid_spec_sha256": file_hash(spec_path),
        "development_or_outcome_read": False,
        "source_commit": current_commit(),
    }
    atomic_write_json(root / "selection.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
