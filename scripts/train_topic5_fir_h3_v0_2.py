#!/usr/bin/env python3
"""Train one LOSO FIR-H3 residual over a frozen unordered baseline."""
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

from scripts.analyze_topic5_minimal_sequence_kernel_cell_v0_2 import (  # noqa: E402
    _evaluate,
)
from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _batch,
    _dataset_balanced_patient_order,
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
    StaticSequenceContactQuery,
    next_set_stop_loss,
)


def _train_ordered_residual_coverage(
    model: ResidualFIRH3SequenceModel,
    records,
    outer_offsets: dict[str, torch.Tensor],
    *,
    coverage_cycles: int,
    updates_per_patient: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    device: torch.device,
    seed: int,
) -> list[dict]:
    """Fit only the three FIR projections with all baseline terms frozen."""

    model.to(device).train()
    model.freeze_unordered_baseline()
    parameters = model.ordered_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    offsets = {
        subject: value.to(device=device, dtype=torch.float32)
        for subject, value in outer_offsets.items()
    }
    rng = np.random.default_rng(int(seed))
    rows = []
    global_update = 0
    started = time.time()
    for cycle in range(int(coverage_cycles)):
        for record in _dataset_balanced_patient_order(records, rng):
            indices = rng.permutation(record.train_indices)
            segments = [
                segment
                for segment in np.array_split(indices, int(updates_per_patient))
                if len(segment)
            ]
            for segment_index, segment in enumerate(segments):
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                for start in range(0, len(segment), int(batch_size)):
                    chunk = segment[start : start + int(batch_size)]
                    batch = _batch(
                        record,
                        chunk,
                        device,
                        rank_shuffle=False,
                        rng=rng,
                    )
                    output = model(
                        **batch, local_offset=offsets[record.subject]
                    )
                    loss = next_set_stop_loss(
                        output, batch["group_ids"], batch["group_count"]
                    )
                    weight = len(chunk) / len(segment)
                    (loss["total"] * weight).backward()
                    weighted_loss += float(loss["total"].detach().cpu()) * weight
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(gradient_clip)
                )
                optimizer.step()
                global_update += 1
                rows.append(
                    {
                        "phase": "fir_residual_full_coverage",
                        "coverage_cycle": cycle + 1,
                        "patient_update": segment_index + 1,
                        "global_update": global_update,
                        "subject": record.subject,
                        "dataset": record.dataset,
                        "n_events": int(len(segment)),
                        "loss": weighted_loss,
                        "gradient_norm": float(grad_norm.detach().cpu()),
                        "elapsed_seconds": time.time() - started,
                    }
                )
    return rows


def _train_ordered_residual_smoke(
    model: ResidualFIRH3SequenceModel,
    records,
    outer_offsets: dict[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    device: torch.device,
    seed: int,
) -> list[dict]:
    model.to(device).train()
    model.freeze_unordered_baseline()
    parameters = model.ordered_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    offsets = {
        subject: value.to(device=device, dtype=torch.float32)
        for subject, value in outer_offsets.items()
    }
    rng = np.random.default_rng(int(seed))
    rows = []
    ordered = list(records)
    for step in range(int(steps)):
        record = ordered[step % len(ordered)]
        size = min(int(batch_size), len(record.train_indices))
        indices = rng.choice(record.train_indices, size=size, replace=False)
        batch = _batch(
            record,
            indices,
            device,
            rank_shuffle=False,
            rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch, local_offset=offsets[record.subject])
        loss = next_set_stop_loss(
            output, batch["group_ids"], batch["group_count"]
        )
        loss["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(gradient_clip)
        )
        optimizer.step()
        rows.append(
            {
                "phase": "fir_residual_smoke",
                "global_update": step + 1,
                "subject": record.subject,
                "dataset": record.dataset,
                "n_events": int(size),
                "loss": float(loss["total"].detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
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
    parser.add_argument("--smoke-steps", type=int, default=12)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    cfg = yaml.safe_load(config_path.read_text())
    records = load_records(ROOT / cfg["inputs"]["dataset"])
    if args.heldout_subject not in records:
        raise RuntimeError(f"unknown heldout subject: {args.heldout_subject}")
    heldout = records[args.heldout_subject]
    outer = [
        record for subject, record in records.items()
        if subject != heldout.subject
    ]
    history = cfg["history_necessity"]
    model_kwargs = {
        "hidden_size": int(history["hidden_size"]),
        "contact_embedding_dim": int(history["contact_embedding_dim"]),
        "contact_encoder_hidden": int(history["contact_encoder_hidden"]),
        "local_offset_dim": int(history["local_offset_dim"]),
    }
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_memory_fraction)
        )
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(args.cpu_threads))
    _seed_everything(args.seed)
    started = time.time()

    baseline = StaticSequenceContactQuery(
        heldout.contact_features.shape[1],
        mode="unordered",
        **model_kwargs,
    )
    if args.smoke:
        baseline_state, outer_offsets, baseline_log, baseline_coverage = (
            train_shared(
                baseline,
                outer,
                steps=int(args.smoke_steps),
                batch_size=min(int(history["batch_events"]), 256),
                learning_rate=float(history["learning_rate"]),
                local_learning_rate=float(history["local_learning_rate"]),
                weight_decay=float(history["weight_decay"]),
                gradient_clip=float(history["gradient_clip"]),
                local_offset_dim=int(history["local_offset_dim"]),
                device=device,
                seed=int(args.seed),
                rank_shuffle=False,
            )
        )
    else:
        baseline_state, outer_offsets, baseline_log, baseline_coverage = (
            train_shared_coverage(
                baseline,
                outer,
                coverage_cycles=int(history["coverage_shared_cycles"]),
                updates_per_patient=int(
                    history["coverage_updates_per_patient"]
                ),
                batch_size=int(history["batch_events"]),
                learning_rate=float(history["learning_rate"]),
                local_learning_rate=float(history["local_learning_rate"]),
                weight_decay=float(history["weight_decay"]),
                gradient_clip=float(history["gradient_clip"]),
                local_offset_dim=int(history["local_offset_dim"]),
                device=device,
                seed=int(args.seed),
                rank_shuffle=False,
            )
        )
    baseline.load_state_dict(baseline_state)
    if args.smoke:
        heldout_offset, calibration_log, calibration_coverage = calibrate_offset(
            baseline,
            heldout,
            steps=int(args.smoke_steps),
            batch_size=min(int(history["batch_events"]), 256),
            local_learning_rate=float(history["local_learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            local_offset_dim=int(history["local_offset_dim"]),
            device=device,
            seed=int(args.seed) + 500_000,
            rank_shuffle=False,
        )
    else:
        heldout_offset, calibration_log, calibration_coverage = (
            calibrate_offset_coverage(
                baseline,
                heldout,
                coverage_cycles=int(history["coverage_calibration_cycles"]),
                updates_per_cycle=int(
                    history["coverage_updates_per_patient"]
                ),
                batch_size=int(history["batch_events"]),
                local_learning_rate=float(history["local_learning_rate"]),
                weight_decay=float(history["weight_decay"]),
                gradient_clip=float(history["gradient_clip"]),
                local_offset_dim=int(history["local_offset_dim"]),
                device=device,
                seed=int(args.seed) + 500_000,
                rank_shuffle=False,
            )
        )

    fir = ResidualFIRH3SequenceModel(
        heldout.contact_features.shape[1], **model_kwargs
    )
    missing, unexpected = fir.load_state_dict(baseline_state, strict=False)
    expected_missing = {
        f"lag_projections.{lag}.weight" for lag in range(3)
    }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"baseline-to-FIR state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if args.smoke:
        fir_log = _train_ordered_residual_smoke(
            fir,
            outer,
            outer_offsets,
            steps=int(args.smoke_steps),
            batch_size=min(int(history["batch_events"]), 256),
            learning_rate=float(history["learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            device=device,
            seed=int(args.seed) + 900_000,
        )
    else:
        fir_log = _train_ordered_residual_coverage(
            fir,
            outer,
            outer_offsets,
            coverage_cycles=int(history["coverage_shared_cycles"]),
            updates_per_patient=int(
                history["coverage_updates_per_patient"]
            ),
            batch_size=int(history["batch_events"]),
            learning_rate=float(history["learning_rate"]),
            weight_decay=float(history["weight_decay"]),
            gradient_clip=float(history["gradient_clip"]),
            device=device,
            seed=int(args.seed) + 900_000,
        )
    torch.save(
        {
            "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
            "condition": "fir_h3_residual",
            "model_kwargs": model_kwargs,
            "model_state": {
                key: value.detach().cpu()
                for key, value in fir.state_dict().items()
            },
            "heldout_local_offset": heldout_offset.cpu(),
            "heldout_subject": heldout.subject,
            "seed": int(args.seed),
            "baseline_frozen": True,
            "history_lags": 3,
            "ictal_target_read": False,
        },
        run_dir / "fir_h3_checkpoint.pt",
    )
    metrics = [
        _evaluate(
            baseline,
            heldout,
            heldout_offset.to(device),
            condition="unordered_retrained",
            seed=args.seed,
            device=device,
            output_path=run_dir / "unordered_retrained_decisions.csv.gz",
            batch_size=512,
        ),
        _evaluate(
            fir,
            heldout,
            heldout_offset.to(device),
            condition="fir_h3_residual",
            seed=args.seed,
            device=device,
            output_path=run_dir / "fir_h3_residual_decisions.csv.gz",
            batch_size=512,
        ),
    ]
    pd.DataFrame(metrics).to_csv(
        run_dir / "component_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            *({"stage": "unordered_baseline", **row} for row in baseline_log),
            *({"stage": "heldout_offset", **row} for row in calibration_log),
            *({"stage": "fir_residual", **row} for row in fir_log),
        ]
    ).to_csv(run_dir / "training_log.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "subject": heldout.subject,
        "dataset": heldout.dataset,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "baseline_frozen_during_fir": True,
        "trainable_fir_parameters": [
            name for name, parameter in fir.named_parameters()
            if parameter.requires_grad
        ],
        "dataset_npz_sha256": heldout.input_sha256,
        "target_values_read": False,
        "coverage": {
            "baseline_shared": baseline_coverage,
            "heldout_calibration": calibration_coverage,
            "fir_outer_training_events": int(
                sum(row["n_events"] for row in fir_log)
            ),
        },
        "resources": {
            "runtime_seconds": float(time.time() - started),
            "peak_rss_gb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            ),
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda"
                else 0
            ),
            "gpu_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if device.type == "cuda"
                else 0
            ),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
