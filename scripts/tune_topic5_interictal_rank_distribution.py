#!/usr/bin/env python3
"""Select the minimal v0.4 GRU hyperparameters using interictal data only."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _jsonable,
    _seed_everything,
    evaluate_model,
    load_records,
    train_shared,
)
from src.topic5_rank_distribution import FullHistorySequenceGRU  # noqa: E402


def _inner_records(records, fraction: float):
    """Split only the chronological first 80%; keep the outer 20% sealed."""
    out = {}
    audit = []
    for subject, record in records.items():
        first80 = record.train_indices
        n_validation = max(1, int(round(len(first80) * float(fraction))))
        n_training = len(first80) - n_validation
        if n_training < 1:
            raise RuntimeError(f"{subject}: insufficient inner-training events")
        split = np.full(record.event_split.shape, 2, dtype=np.uint8)
        split[first80[:n_training]] = 0
        split[first80[n_training:]] = 1
        out[subject] = replace(record, event_split=split)
        audit.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_inner_train": n_training,
                "n_inner_validation": n_validation,
                "n_outer_eval_sealed": int(len(record.eval_indices)),
            }
        )
    return out, pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    tuning = cfg["hyperparameter_tuning"]
    records = load_records(ROOT / cfg["outputs"]["dataset"])
    inner, split_audit = _inner_records(
        records,
        float(tuning["inner_validation_fraction_of_first80"]),
    )
    split_audit.to_csv(run_dir / "inner_split_audit.csv", index=False)

    seed = int(tuning["seed"])
    _seed_everything(seed)
    device = torch.device(args.device or cfg["resources"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(0.70)
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(min(16, int(cfg["resources"]["cpu_threads_per_process"]) * 2))
    stage = cfg["stage_a"]
    grid = list(
        itertools.product(
            tuning["hidden_sizes"],
            tuning["learning_rates"],
            tuning["local_offset_dims"],
        )
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_configurations": len(grid),
                "ictal_target_read": False,
                "outer_last20_read": False,
            },
            indent=2,
        )
    )
    summary_rows = []
    patient_rows = []
    all_logs = []
    for config_index, (hidden_size, learning_rate, offset_dim) in enumerate(grid):
        config_id = (
            f"h{int(hidden_size)}_lr{float(learning_rate):g}_o{int(offset_dim)}"
        )
        config_seed = seed + config_index * 100_003
        print(
            json.dumps(
                {
                    "config": config_id,
                    "index": config_index + 1,
                    "total": len(grid),
                    "status": "training",
                }
            ),
            flush=True,
        )
        model_kwargs = {
            "hidden_size": int(hidden_size),
            "contact_embedding_dim": int(stage["contact_embedding_dim"]),
            "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
            "local_offset_dim": int(offset_dim),
        }
        model = FullHistorySequenceGRU(
            next(iter(inner.values())).contact_features.shape[1],
            **model_kwargs,
        )
        state, offsets, training_log, coverage = train_shared(
            model,
            list(inner.values()),
            steps=int(tuning["shared_steps"]),
            batch_size=int(tuning["batch_events"]),
            learning_rate=float(learning_rate),
            local_learning_rate=float(stage["local_learning_rate"]),
            weight_decay=float(stage["weight_decay"]),
            gradient_clip=float(stage["gradient_clip"]),
            local_offset_dim=int(offset_dim),
            device=device,
            seed=config_seed,
            rank_shuffle=False,
            log_every=64,
        )
        model.load_state_dict(state)
        n_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        config_patient_rows = []
        for subject, record in inner.items():
            metrics, _, _ = evaluate_model(
                model,
                record,
                offsets[subject].to(device),
                device=device,
                batch_size=256,
                max_events=None,
            )
            row = {
                "config_id": config_id,
                "subject": subject,
                "dataset": record.dataset,
                "inner_validation_nll": metrics["heldout_event_nll"],
                "top1_next_set_accuracy": metrics["top1_next_set_accuracy"],
                "stop_brier": metrics["stop_brier"],
            }
            config_patient_rows.append(row)
            patient_rows.append(row)
        patient_frame = pd.DataFrame(config_patient_rows)
        values = patient_frame.inner_validation_nll.to_numpy(float)
        summary_rows.append(
            {
                "config_id": config_id,
                "hidden_size": int(hidden_size),
                "learning_rate": float(learning_rate),
                "local_offset_dim": int(offset_dim),
                "n_parameters": n_parameters,
                "patient_mean_validation_nll": float(np.mean(values)),
                "patient_median_validation_nll": float(np.median(values)),
                "patient_se_validation_nll": float(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                ),
                "mean_top1_next_set_accuracy": float(
                    patient_frame.top1_next_set_accuracy.mean()
                ),
                "mean_stop_brier": float(patient_frame.stop_brier.mean()),
                "min_patient_first_cycle_fraction": float(
                    min(
                        value["fraction_of_first_cycle"]
                        for value in coverage.values()
                    )
                ),
            }
        )
        for row in training_log:
            row["config_id"] = config_id
            all_logs.append(row)
        torch.save(
            {
                "config_id": config_id,
                "model_kwargs": model_kwargs,
                "model_state": state,
                "patient_local_offsets": offsets,
                "seed": config_seed,
                "ictal_target_read": False,
                "outer_last20_read": False,
            },
            run_dir / f"{config_id}_checkpoint.pt",
        )
        print(
            json.dumps(
                {
                    "config": config_id,
                    "status": "evaluated",
                    "patient_mean_validation_nll": summary_rows[-1][
                        "patient_mean_validation_nll"
                    ],
                }
            ),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    patients = pd.DataFrame(patient_rows)
    best_index = summary.patient_mean_validation_nll.idxmin()
    best = summary.loc[best_index]
    threshold = float(
        best.patient_mean_validation_nll + best.patient_se_validation_nll
    )
    eligible = summary[
        summary.patient_mean_validation_nll <= threshold
    ].copy()
    selected = eligible.sort_values(
        ["n_parameters", "patient_mean_validation_nll", "local_offset_dim"]
    ).iloc[0]
    summary["within_one_se"] = (
        summary.patient_mean_validation_nll <= threshold
    )
    summary["selected"] = summary.config_id == selected.config_id
    summary.to_csv(run_dir / "tuning_summary.csv", index=False)
    patients.to_csv(run_dir / "tuning_per_subject.csv", index=False)
    pd.DataFrame(all_logs).to_csv(run_dir / "training_log.csv", index=False)
    result = {
        "status": "complete",
        "selection_rule": tuning["selection_rule"],
        "best_mean_config": str(best.config_id),
        "one_se_threshold": threshold,
        "selected_config": str(selected.config_id),
        "selected_hyperparameters": {
            "hidden_size": int(selected.hidden_size),
            "learning_rate": float(selected.learning_rate),
            "local_offset_dim": int(selected.local_offset_dim),
        },
        "selected_patient_mean_validation_nll": float(
            selected.patient_mean_validation_nll
        ),
        "n_patients": len(records),
        "n_configurations": len(grid),
        "ictal_target_read": False,
        "outer_last20_read": False,
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
    }
    (run_dir / "selected_hyperparameters.json").write_text(
        json.dumps(_jsonable(result), indent=2)
    )
    (run_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_config": result["selected_config"],
                "ictal_target_read": False,
                "outer_last20_read": False,
            },
            indent=2,
        )
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_config": result["selected_config"],
                "ictal_target_read": False,
                "outer_last20_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(_jsonable(result)), flush=True)


if __name__ == "__main__":
    main()
