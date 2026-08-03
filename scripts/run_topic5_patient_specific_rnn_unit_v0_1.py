#!/usr/bin/env python3
"""Train one patient x seed x architecture target-free bridge unit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _distribution_frame,
    _seed_everything,
    evaluate_model,
    load_records,
    train_shared_coverage,
)
from src.topic5_patient_specific_rnn_bridge import (  # noqa: E402
    chronological_60_20_20,
    distribution_fields,
    record_with_split,
    train_only_contact_features,
)
from src.topic5_rank_distribution import (  # noqa: E402
    FullHistorySequenceGRU,
    LinearStateSequenceRNN,
    contact_rank_distribution,
    distribution_errors,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_model(name: str, feature_dim: int, kwargs: dict):
    if name in {"full_history_gru", "rank_shuffle_gru"}:
        return FullHistorySequenceGRU(feature_dim, **kwargs)
    if name == "linear_state":
        return LinearStateSequenceRNN(feature_dim, **kwargs)
    raise ValueError(f"unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model", choices=("full_history_gru", "linear_state", "rank_shuffle_gru"), required=True
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    artifact_root = Path(config["artifact_root"]).resolve()
    dataset_root = artifact_root / config["dataset_root"]
    output_root = ROOT / config["output_root"]
    run_dir = output_root / "units" / args.subject / args.model / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "DONE.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get("status") == "COMPLETE":
            print(json.dumps(done), flush=True)
            return

    _seed_everything(args.seed)
    device = torch.device(args.device or config["resources"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(
            float(config["resources"]["gpu_memory_fraction_per_process"]), device=device_index
        )
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.set_num_threads(int(config["resources"]["torch_num_threads"]))

    records = load_records(dataset_root)
    raw = records[args.subject]
    fit60, validation20, test20 = chronological_60_20_20(raw)
    features = train_only_contact_features(raw.group_ids, fit60)
    train_record = record_with_split(raw, fit60, validation20, features)
    test_record = record_with_split(raw, fit60, test20, features)

    model_cfg = config["models"]
    model_kwargs = {
        "hidden_size": int(model_cfg["hidden_size"]),
        "contact_embedding_dim": int(model_cfg["contact_embedding_dim"]),
        "contact_encoder_hidden": int(model_cfg["contact_encoder_hidden"]),
        "local_offset_dim": int(model_cfg["local_offset_dim"]),
    }
    model = build_model(args.model, features.shape[1], model_kwargs).to(device)
    training = config["training"]
    started = time.time()
    model_state, offset_states, training_log, coverage = train_shared_coverage(
        model,
        [train_record],
        coverage_cycles=int(training["coverage_cycles"]),
        updates_per_patient=int(training["updates_per_cycle"]),
        batch_size=int(training["batch_events"]),
        learning_rate=float(training["learning_rate"]),
        local_learning_rate=float(training["local_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        local_offset_dim=int(model_cfg["local_offset_dim"]),
        device=device,
        seed=args.seed,
        rank_shuffle=args.model == "rank_shuffle_gru",
    )
    model.load_state_dict(model_state)
    offset = offset_states[args.subject].to(device)
    validation_metrics, validation_events, _ = evaluate_model(
        model,
        train_record,
        offset,
        device=device,
        batch_size=min(int(training["batch_events"]), 256),
        max_events=None,
    )
    test_metrics, test_events, evaluated_test = evaluate_model(
        model,
        test_record,
        offset,
        device=device,
        batch_size=min(int(training["batch_events"]), 256),
        max_events=None,
    )

    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    contact_mask = torch.ones((1, features.shape[0]), dtype=torch.bool, device=device)
    rollout_groups, rollout_count = model.rollout(
        feature_tensor,
        contact_mask,
        offset,
        n_events=int(config["readout"]["free_rollouts"]),
        seed=args.seed + 700_000,
        batch_size=512,
    )
    observed_groups = raw.group_ids[evaluated_test]
    observed_count = raw.group_count[evaluated_test]
    errors = distribution_errors(
        rollout_groups,
        rollout_count,
        observed_groups,
        observed_count,
        bins=int(config["readout"]["rank_bins"]),
    )
    fields = distribution_fields(
        rollout_groups, rollout_count, bins=int(config["readout"]["rank_bins"])
    )

    torch.save(
        {
            "contract": config["contract"],
            "subject": args.subject,
            "seed": args.seed,
            "model": args.model,
            "model_kwargs": model_kwargs,
            "model_state": {key: value.cpu() for key, value in model_state.items()},
            "local_offset": offset.cpu(),
            "contact_features": features,
            "contact_names": raw.contact_names,
            "fit_indices": fit60,
            "validation_indices": validation20,
            "test_indices": test20,
            "ictal_target_read": False,
        },
        run_dir / "checkpoint.pt",
    )
    np.savez_compressed(
        run_dir / "free_rollouts.npz",
        event_group_ids=rollout_groups,
        event_group_count=rollout_count,
        contact_names=raw.contact_names,
        **{f"field__{name}": value.astype(np.float32) for name, value in fields.items()},
    )
    _distribution_frame(
        test_record,
        args.model,
        rollout_groups,
        rollout_count,
        observed_groups,
        observed_count,
        int(config["readout"]["rank_bins"]),
    ).to_csv(run_dir / "contact_rank_distribution.csv", index=False)
    pd.DataFrame(training_log).to_csv(run_dir / "training_log.csv", index=False)
    validation_events.to_csv(run_dir / "validation_event_nll.csv", index=False)
    test_events.to_csv(run_dir / "test_event_nll.csv", index=False)

    fit_distribution = contact_rank_distribution(
        raw.group_ids[fit60], raw.group_count[fit60], bins=int(config["readout"]["rank_bins"])
    )
    test_distribution = contact_rank_distribution(
        raw.group_ids[test20], raw.group_count[test20], bins=int(config["readout"]["rank_bins"])
    )
    np.savez_compressed(
        run_dir / "empirical_references.npz",
        contact_names=raw.contact_names,
        fit_participation=fit_distribution["participation_probability"].astype(np.float32),
        fit_mean_rank=fit_distribution["mean_rank"].astype(np.float32),
        fit_rank_histogram=fit_distribution["rank_histogram"].astype(np.float32),
        test_participation=test_distribution["participation_probability"].astype(np.float32),
        test_mean_rank=test_distribution["mean_rank"].astype(np.float32),
        test_rank_histogram=test_distribution["rank_histogram"].astype(np.float32),
    )
    summary = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "subject": args.subject,
        "seed": args.seed,
        "model": args.model,
        "n_contacts": int(len(raw.contact_names)),
        "n_events": {
            "fit60": int(len(fit60)),
            "validation20": int(len(validation20)),
            "test20": int(len(test20)),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "rollout_errors": errors,
        "rollout_participant_count_mean": float(np.mean(rollout_count)),
        "rollout_zero_length_fraction": float(np.mean(rollout_count == 0)),
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_index) / 1024**2) if device.type == "cuda" else 0.0
        ),
        "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2),
        "coverage": coverage,
        "dataset_sha256": sha256(raw.path),
        "config_sha256": sha256(config_path),
        "other_patient_events_used": False,
        "empirical_ab_used": False,
        "ictal_target_read": False,
    }
    atomic_json(run_dir / "run_summary.json", summary)
    atomic_json(done_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
