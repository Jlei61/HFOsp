#!/usr/bin/env python3
"""Freeze within-event models in one dataset and confirm in the other.

The shared core sees only the source dataset.  In each target patient, only
contact-local nuisance offsets are calibrated on chronological train80;
heldout20 remains evaluation-only.  This is a new-endpoint cross-dataset
confirmation, not an untouched external validation, because architecture
selection previously inspected both datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from scripts.analyze_topic5_rank_tolerance_subject_v0_2 import _metrics  # noqa: E402
from scripts.train_topic5_fir_h3_v0_2 import (  # noqa: E402
    _train_ordered_residual_coverage,
    _train_ordered_residual_smoke,
)
from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _seed_everything,
    calibrate_offset,
    calibrate_offset_coverage,
    load_records,
    train_shared,
    train_shared_coverage,
)
from src.topic5_minimal_sequence_kernel import (  # noqa: E402
    ResidualFIRH3SequenceModel,
)
from src.topic5_rank_distribution import (  # noqa: E402
    LinearStateSequenceRNN,
    StaticSequenceContactQuery,
)


DATASETS = ("epilepsiae", "yuquan")


def _fit_shared(
    model,
    records,
    *,
    history: dict,
    device: torch.device,
    seed: int,
    smoke: bool,
    smoke_steps: int,
):
    if smoke:
        return train_shared(
            model,
            records,
            steps=int(smoke_steps),
            batch_size=min(int(history["batch_events"]), 256),
            learning_rate=float(history["learning_rate"]),
            local_learning_rate=float(history["local_learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            local_offset_dim=int(history["local_offset_dim"]),
            device=device,
            seed=int(seed),
            rank_shuffle=False,
        )
    return train_shared_coverage(
        model,
        records,
        coverage_cycles=int(history["coverage_shared_cycles"]),
        updates_per_patient=int(history["coverage_updates_per_patient"]),
        batch_size=int(history["batch_events"]),
        learning_rate=float(history["learning_rate"]),
        local_learning_rate=float(history["local_learning_rate"]),
        weight_decay=float(history["weight_decay"]),
        gradient_clip=float(history["gradient_clip"]),
        local_offset_dim=int(history["local_offset_dim"]),
        device=device,
        seed=int(seed),
        rank_shuffle=False,
    )


def _calibrate(
    model,
    record,
    *,
    history: dict,
    device: torch.device,
    seed: int,
    smoke: bool,
    smoke_steps: int,
):
    if smoke:
        return calibrate_offset(
            model,
            record,
            steps=int(smoke_steps),
            batch_size=min(int(history["batch_events"]), 256),
            local_learning_rate=float(history["local_learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            local_offset_dim=int(history["local_offset_dim"]),
            device=device,
            seed=int(seed),
            rank_shuffle=False,
        )
    return calibrate_offset_coverage(
        model,
        record,
        coverage_cycles=int(history["coverage_calibration_cycles"]),
        updates_per_cycle=int(history["coverage_updates_per_patient"]),
        batch_size=int(history["batch_events"]),
        local_learning_rate=float(history["local_learning_rate"]),
        weight_decay=float(history["weight_decay"]),
        gradient_clip=float(history["gradient_clip"]),
        local_offset_dim=int(history["local_offset_dim"]),
        device=device,
        seed=int(seed),
        rank_shuffle=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "config/topic5_static_scaffold_reliability_history_necessity_v0_1.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.18)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    records = load_records(ROOT / cfg["inputs"]["dataset"])
    target_dataset = next(name for name in DATASETS if name != args.source_dataset)
    source = sorted(
        (record for record in records.values() if record.dataset == args.source_dataset),
        key=lambda record: record.subject,
    )
    target = sorted(
        (record for record in records.values() if record.dataset == target_dataset),
        key=lambda record: record.subject,
    )
    expected = {"epilepsiae": 18, "yuquan": 16}
    if len(source) != expected[args.source_dataset] or len(target) != expected[target_dataset]:
        raise RuntimeError("cross-dataset cohort does not match the frozen 18/16 contract")

    history = cfg["history_necessity"]
    model_kwargs = {
        "hidden_size": int(history["hidden_size"]),
        "contact_embedding_dim": int(history["contact_embedding_dim"]),
        "contact_encoder_hidden": int(history["contact_encoder_hidden"]),
        "local_offset_dim": int(history["local_offset_dim"]),
    }
    feature_dim = source[0].contact_features.shape[1]
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction))
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(args.cpu_threads))
    _seed_everything(args.seed)
    started = time.time()

    baseline = StaticSequenceContactQuery(
        feature_dim, mode="unordered", **model_kwargs
    )
    baseline_state, source_offsets, baseline_log, baseline_coverage = _fit_shared(
        baseline,
        source,
        history=history,
        device=device,
        seed=args.seed,
        smoke=args.smoke,
        smoke_steps=args.smoke_steps,
    )
    baseline.load_state_dict(baseline_state)

    linear = LinearStateSequenceRNN(feature_dim, **model_kwargs)
    linear_state, _, linear_log, linear_coverage = _fit_shared(
        linear,
        source,
        history=history,
        device=device,
        seed=args.seed + 100_000,
        smoke=args.smoke,
        smoke_steps=args.smoke_steps,
    )
    linear.load_state_dict(linear_state)

    fir = ResidualFIRH3SequenceModel(feature_dim, **model_kwargs)
    missing, unexpected = fir.load_state_dict(baseline_state, strict=False)
    expected_missing = {f"lag_projections.{lag}.weight" for lag in range(3)}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"baseline-to-FIR mismatch: missing={missing}, unexpected={unexpected}"
        )
    if args.smoke:
        fir_log = _train_ordered_residual_smoke(
            fir,
            source,
            source_offsets,
            steps=args.smoke_steps,
            batch_size=min(int(history["batch_events"]), 256),
            learning_rate=float(history["learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            device=device,
            seed=args.seed + 200_000,
        )
    else:
        fir_log = _train_ordered_residual_coverage(
            fir,
            source,
            source_offsets,
            coverage_cycles=int(history["coverage_shared_cycles"]),
            updates_per_patient=int(history["coverage_updates_per_patient"]),
            batch_size=int(history["batch_events"]),
            learning_rate=float(history["learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            device=device,
            seed=args.seed + 200_000,
        )

    torch.save(
        {
            "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
            "source_dataset": args.source_dataset,
            "target_dataset": target_dataset,
            "seed": int(args.seed),
            "model_kwargs": model_kwargs,
            "unordered_state": baseline_state,
            "linear_state": linear_state,
            "fir_state": {
                key: value.detach().cpu() for key, value in fir.state_dict().items()
            },
            "target_values_read": False,
        },
        run_dir / "source_frozen_models.pt",
    )

    metric_rows = []
    calibration_rows = []
    calibration_coverage = {}
    for patient_index, record in enumerate(target):
        baseline_offset, baseline_calibration, baseline_patient_coverage = _calibrate(
            baseline,
            record,
            history=history,
            device=device,
            seed=args.seed + 300_000 + patient_index,
            smoke=args.smoke,
            smoke_steps=args.smoke_steps,
        )
        linear_offset, linear_calibration, linear_patient_coverage = _calibrate(
            linear,
            record,
            history=history,
            device=device,
            seed=args.seed + 400_000 + patient_index,
            smoke=args.smoke,
            smoke_steps=args.smoke_steps,
        )
        for condition, model, offset in (
            ("unordered_source_frozen", baseline, baseline_offset),
            ("linear_source_frozen", linear, linear_offset),
            ("fir_h3_source_frozen", fir, baseline_offset),
        ):
            metric_rows.append(
                {
                    "subject": record.subject,
                    "dataset": record.dataset,
                    "source_dataset": args.source_dataset,
                    "target_dataset": target_dataset,
                    "seed": int(args.seed),
                    "condition": condition,
                    **_metrics(
                        model,
                        record,
                        offset.to(device),
                        device=device,
                        batch_size=512,
                    ),
                }
            )
        calibration_rows.extend(
            {"condition": "unordered", **row} for row in baseline_calibration
        )
        calibration_rows.extend(
            {"condition": "linear", **row} for row in linear_calibration
        )
        calibration_coverage[record.subject] = {
            "unordered": baseline_patient_coverage,
            "linear": linear_patient_coverage,
        }

    metrics = pd.DataFrame(metric_rows)
    if float(metrics.maximum_reconstruction_error.max()) > 2e-5:
        raise RuntimeError("cross-dataset likelihood reconstruction failed")
    metrics.to_csv(run_dir / "target_patient_metrics.csv", index=False)
    pd.DataFrame(
        [
            *({"stage": "source_unordered", **row} for row in baseline_log),
            *({"stage": "source_linear", **row} for row in linear_log),
            *({"stage": "source_fir", **row} for row in fir_log),
            *({"stage": "target_offset", **row} for row in calibration_rows),
        ]
    ).to_csv(run_dir / "training_log.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "source_dataset": args.source_dataset,
        "target_dataset": target_dataset,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "source_patients": len(source),
        "target_patients": len(target),
        "source_shared_coverage": {
            "unordered": baseline_coverage,
            "linear": linear_coverage,
            "fir_events": int(sum(row["n_events"] for row in fir_log)),
        },
        "target_calibration_coverage": calibration_coverage,
        "baseline_frozen_during_fir": True,
        "target_values_read": False,
        "interpretation": "new_endpoint_cross_dataset_confirmation_not_untouched_external_validation",
        "resources": {
            "runtime_seconds": time.time() - started,
            "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024**2,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0,
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
            if device.type == "cuda"
            else 0,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
