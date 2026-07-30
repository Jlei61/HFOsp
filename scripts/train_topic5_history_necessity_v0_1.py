#!/usr/bin/env python3
"""Train one LOSO fold for fixed history windows 1, 2, and 3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from scripts.train_topic5_interictal_rank_distribution import (
    _seed_everything,
    calibrate_offset,
    calibrate_offset_coverage,
    evaluate_model,
    load_records,
    train_shared,
    train_shared_coverage,
)
from src.topic5_rank_distribution import WindowedHistorySequenceGRU


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "config/topic5_static_scaffold_reliability_history_necessity_v0_1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shared-steps", type=int, default=12)
    parser.add_argument("--calibration-steps", type=int, default=8)
    parser.add_argument("--max-eval-events", type=int, default=None)
    parser.add_argument(
        "--rank-shuffle-window",
        type=int,
        default=None,
        help="Train only the matched finite-window within-event rank-shuffle control.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    dataset_dir = ROOT / cfg["inputs"]["dataset"]
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(config_path)
    snapshot_path = run_dir / "config_snapshot.yaml"
    if snapshot_path.exists():
        previous = yaml.safe_load(snapshot_path.read_text())
        if previous != cfg:
            raise RuntimeError("partial run config differs from requested config")
    else:
        snapshot_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    records = load_records(dataset_dir)
    if args.heldout_subject not in records:
        raise RuntimeError(f"held-out subject is absent: {args.heldout_subject}")
    heldout = records[args.heldout_subject]
    outer = [
        record
        for subject, record in records.items()
        if subject != args.heldout_subject
    ]
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        previous_state = json.loads(state_path.read_text())
        if (
            str(previous_state.get("heldout_subject")) != heldout.subject
            or int(previous_state.get("seed", -1)) != int(args.seed)
        ):
            raise RuntimeError("partial run belongs to another fold or seed")

    history_cfg = cfg["history_necessity"]
    resource_cfg = cfg["resources"]
    device = torch.device(args.device or resource_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_per_process_memory_fraction(
            float(resource_cfg["gpu_memory_fraction_per_process"]),
            device=device_index,
        )
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.set_num_threads(int(resource_cfg["cpu_threads_per_process"]))

    model_kwargs = {
        "hidden_size": int(history_cfg["hidden_size"]),
        "contact_embedding_dim": int(history_cfg["contact_embedding_dim"]),
        "contact_encoder_hidden": int(history_cfg["contact_encoder_hidden"]),
        "local_offset_dim": int(history_cfg["local_offset_dim"]),
    }
    batch_size = int(history_cfg["batch_events"])
    state_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "heldout_subject": heldout.subject,
                "dataset": heldout.dataset,
                "seed": int(args.seed),
                "smoke": bool(args.smoke),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )

    if args.rank_shuffle_window is None:
        windows = [int(value) for value in history_cfg["history_windows"]]
        required_conditions = [
            f"history_{window}_gru" for window in windows
        ]
        rank_shuffle = False
    else:
        windows = [int(args.rank_shuffle_window)]
        if windows[0] < 1:
            raise ValueError("--rank-shuffle-window must be positive")
        required_conditions = [
            f"history_{windows[0]}_rank_shuffle_gru"
        ]
        rank_shuffle = True
    for window, condition in zip(windows, required_conditions):
        condition_dir = run_dir / condition
        metric_path = condition_dir / "metrics.json"
        done_path = condition_dir / "DONE.json"
        if metric_path.exists() and done_path.exists():
            done = json.loads(done_path.read_text())
            if done.get("status") == "complete":
                print(
                    json.dumps(
                        {"condition": condition, "status": "resume_skip"}
                    ),
                    flush=True,
                )
                continue
        condition_dir.mkdir(parents=True, exist_ok=True)
        # Architecture and data order are matched across windows.
        _seed_everything(int(args.seed))
        model = WindowedHistorySequenceGRU(
            heldout.contact_features.shape[1],
            history_window=int(window),
            **model_kwargs,
        )
        print(
            json.dumps(
                {
                    "condition": condition,
                    "status": "training",
                    "heldout_subject": heldout.subject,
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                }
            ),
            flush=True,
        )
        if args.smoke:
            shared_state, _, shared_log, shared_coverage = train_shared(
                model,
                outer,
                steps=int(args.shared_steps),
                batch_size=min(batch_size, 256),
                learning_rate=float(history_cfg["learning_rate"]),
                local_learning_rate=float(history_cfg["local_learning_rate"]),
                weight_decay=float(history_cfg["weight_decay"]),
                gradient_clip=float(history_cfg["gradient_clip"]),
                local_offset_dim=int(history_cfg["local_offset_dim"]),
                device=device,
                seed=int(args.seed),
                rank_shuffle=rank_shuffle,
            )
        else:
            shared_state, _, shared_log, shared_coverage = train_shared_coverage(
                model,
                outer,
                coverage_cycles=int(history_cfg["coverage_shared_cycles"]),
                updates_per_patient=int(
                    history_cfg["coverage_updates_per_patient"]
                ),
                batch_size=batch_size,
                learning_rate=float(history_cfg["learning_rate"]),
                local_learning_rate=float(history_cfg["local_learning_rate"]),
                weight_decay=float(history_cfg["weight_decay"]),
                gradient_clip=float(history_cfg["gradient_clip"]),
                local_offset_dim=int(history_cfg["local_offset_dim"]),
                device=device,
                seed=int(args.seed),
                rank_shuffle=rank_shuffle,
            )
        model.load_state_dict(shared_state)
        if args.smoke:
            offset, calibration_log, calibration_coverage = calibrate_offset(
                model,
                heldout,
                steps=int(args.calibration_steps),
                batch_size=min(batch_size, 256),
                local_learning_rate=float(history_cfg["local_learning_rate"]),
                weight_decay=float(history_cfg["weight_decay"]),
                gradient_clip=float(history_cfg["gradient_clip"]),
                local_offset_dim=int(history_cfg["local_offset_dim"]),
                device=device,
                seed=int(args.seed) + 500_000,
                rank_shuffle=rank_shuffle,
            )
        else:
            offset, calibration_log, calibration_coverage = (
                calibrate_offset_coverage(
                    model,
                    heldout,
                    coverage_cycles=int(
                        history_cfg["coverage_calibration_cycles"]
                    ),
                    updates_per_cycle=int(
                        history_cfg["coverage_updates_per_patient"]
                    ),
                    batch_size=batch_size,
                    local_learning_rate=float(
                        history_cfg["local_learning_rate"]
                    ),
                    weight_decay=float(history_cfg["weight_decay"]),
                    gradient_clip=float(history_cfg["gradient_clip"]),
                    local_offset_dim=int(history_cfg["local_offset_dim"]),
                    device=device,
                    seed=int(args.seed) + 500_000,
                    rank_shuffle=rank_shuffle,
                )
            )
        metrics, event_frame, eval_indices = evaluate_model(
            model,
            heldout,
            offset.to(device),
            device=device,
            batch_size=min(batch_size, 256),
            max_events=(
                int(args.max_eval_events)
                if args.max_eval_events is not None
                else 256
                if args.smoke
                else None
            ),
        )
        metric_row = {
            "subject": heldout.subject,
            "dataset": heldout.dataset,
            "condition": condition,
            "history_window": int(window),
            "seed": int(args.seed),
            "smoke": bool(args.smoke),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "n_local_offset_parameters": int(offset.numel()),
            "n_train_calibration_events": int(heldout.train_indices.size),
            "n_eval_events_available": int(heldout.eval_indices.size),
            "n_eval_events_used": int(len(eval_indices)),
            **metrics,
            "ictal_target_read": False,
        }
        metric_path.write_text(
            json.dumps(_jsonable(metric_row), indent=2, allow_nan=False)
        )
        event_frame["condition"] = condition
        event_frame["seed"] = int(args.seed)
        event_frame.to_csv(condition_dir / "heldout_event_nll.csv", index=False)
        pd.DataFrame([*shared_log, *calibration_log]).to_csv(
            condition_dir / "training_log.csv", index=False
        )
        coverage = {
            "shared": shared_coverage,
            "heldout_calibration": calibration_coverage,
        }
        (condition_dir / "coverage.json").write_text(
            json.dumps(_jsonable(coverage), indent=2)
        )
        torch.save(
            {
                "contract": cfg["contract"],
                "condition": condition,
                "history_window": int(window),
                "model_kwargs": model_kwargs,
                "model_state": shared_state,
                "heldout_local_offset": offset.cpu(),
                "heldout_subject": heldout.subject,
                "seed": int(args.seed),
                "input_sha256": heldout.input_sha256,
                "ictal_target_read": False,
            },
            condition_dir / "checkpoint.pt",
        )
        done_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "condition": condition,
                    "heldout_subject": heldout.subject,
                    "seed": int(args.seed),
                    "ictal_target_read": False,
                },
                indent=2,
            )
        )
        print(
            json.dumps(
                {
                    "condition": condition,
                    "status": "evaluated",
                    "heldout_event_nll": metrics["heldout_event_nll"],
                    "n_eval_events": metrics["n_eval_events"],
                }
            ),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = [
        json.loads((run_dir / condition / "metrics.json").read_text())
        for condition in required_conditions
    ]
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(run_dir / "heldout_metrics.csv", index=False)
    engineering_pass = bool(
        len(metrics_frame) == len(required_conditions)
        and np.all(np.isfinite(metrics_frame.heldout_event_nll))
        and np.all(metrics_frame.n_eval_events > 0)
    )
    resource = {
        "cpu_threads": int(torch.get_num_threads()),
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "gpu_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
        ),
    }
    summary = {
        "status": "complete" if engineering_pass else "engineering_gate_failed",
        "heldout_subject": heldout.subject,
        "dataset": heldout.dataset,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "conditions": required_conditions,
        "engineering_pass": engineering_pass,
        "resource": resource,
        "config_sha256": config_hash,
        "input_fingerprints": {
            subject: record.input_sha256
            for subject, record in sorted(records.items())
        },
        "ictal_target_read": False,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    state_path.write_text(
        json.dumps(
            {
                "status": summary["status"],
                "heldout_subject": heldout.subject,
                "dataset": heldout.dataset,
                "seed": int(args.seed),
                "smoke": bool(args.smoke),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    (run_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "heldout_subject": heldout.subject,
                "seed": int(args.seed),
                "engineering_pass": engineering_pass,
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
