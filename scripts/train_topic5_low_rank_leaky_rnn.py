#!/usr/bin/env python3
"""Train one formal LOSO fold of the structured low-rank leaky RNN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _batch,
    _distribution_frame,
    _jsonable,
    _seed_everything,
    evaluate_model,
    load_records,
    train_shared_coverage,
    calibrate_offset_coverage,
)
from src.topic5_rank_distribution import (  # noqa: E402
    LowRankLeakySequenceRNN,
    distribution_errors,
)


@torch.no_grad()
def _trajectory_artifact(
    model,
    record,
    offset: torch.Tensor,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    n_events = len(indices)
    max_steps = int(np.max(record.group_count[indices])) + 1
    hidden = np.full(
        (n_events, max_steps, model.hidden_size), np.nan, np.float32
    )
    modes = np.full(
        (n_events, max_steps, model.recurrent_rank), np.nan, np.float32
    )
    mask = np.zeros((n_events, max_steps), bool)
    for start in range(0, n_events, int(batch_size)):
        chunk = indices[start : start + int(batch_size)]
        batch = _batch(
            record,
            chunk,
            device,
            rank_shuffle=False,
            rng=np.random.default_rng(0),
        )
        output = model.hidden_trajectory(
            **batch, local_offset=offset
        )
        width = output["hidden_states"].shape[1]
        hidden[start : start + len(chunk), :width] = (
            output["hidden_states"].cpu().numpy()
        )
        modes[start : start + len(chunk), :width] = (
            output["mode_coordinates"].cpu().numpy()
        )
        mask[start : start + len(chunk), :width] = (
            output["state_mask"].cpu().numpy()
        )
    return {
        "event_index": np.asarray(indices, np.int64),
        "event_source_index": record.event_source_index[indices],
        "event_group_count": record.group_count[indices],
        "hidden_states": hidden,
        "mode_coordinates": modes,
        "state_mask": mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--recurrent-rank", type=int, choices=range(5), required=True)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--local-offset-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--shared-cycles", type=int, default=1)
    parser.add_argument("--calibration-cycles", type=int, default=4)
    parser.add_argument("--updates-per-patient", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=5000)
    parser.add_argument("--trajectory-events", type=int, default=500)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    records = load_records(ROOT / cfg["outputs"]["dataset"])
    if args.heldout_subject not in records:
        raise RuntimeError(f"held-out subject absent: {args.heldout_subject}")
    heldout = records[args.heldout_subject]
    outer = [
        record for subject, record in records.items()
        if subject != heldout.subject
    ]
    _seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal low-rank run requires CUDA")
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(
        float(cfg["resources"]["gpu_memory_fraction_per_process"])
    )
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(cfg["resources"]["cpu_threads_per_process"]))
    stage = cfg["stage_a"]
    model_kwargs = {
        "contact_feature_dim": heldout.contact_features.shape[1],
        "recurrent_rank": int(args.recurrent_rank),
        "hidden_size": int(args.hidden_size),
        "contact_embedding_dim": int(stage["contact_embedding_dim"]),
        "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
        "local_offset_dim": int(args.local_offset_dim),
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "subject": heldout.subject,
                "seed": int(args.seed),
                "recurrent_rank": int(args.recurrent_rank),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    model = LowRankLeakySequenceRNN(**model_kwargs)
    state, _, shared_log, shared_coverage = train_shared_coverage(
        model,
        outer,
        coverage_cycles=int(args.shared_cycles),
        updates_per_patient=int(args.updates_per_patient),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        local_learning_rate=float(stage["local_learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
        gradient_clip=float(stage["gradient_clip"]),
        local_offset_dim=int(args.local_offset_dim),
        device=device,
        seed=int(args.seed) + int(args.recurrent_rank) * 1_000_003,
        rank_shuffle=False,
    )
    model.load_state_dict(state)
    offset, calibration_log, calibration_coverage = calibrate_offset_coverage(
        model,
        heldout,
        coverage_cycles=int(args.calibration_cycles),
        updates_per_cycle=int(args.updates_per_patient),
        batch_size=int(args.batch_size),
        local_learning_rate=float(stage["local_learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
        gradient_clip=float(stage["gradient_clip"]),
        local_offset_dim=int(args.local_offset_dim),
        device=device,
        seed=int(args.seed) + 500_000 + int(args.recurrent_rank) * 1_000_003,
        rank_shuffle=False,
    )
    model.eval()
    control = f"low_rank_leaky_r{int(args.recurrent_rank)}"
    metrics, event_frame, eval_indices = evaluate_model(
        model,
        heldout,
        offset.to(device),
        device=device,
        batch_size=256,
        max_events=None,
    )
    features = torch.as_tensor(
        heldout.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    contact_mask = torch.ones(
        (1, heldout.contact_features.shape[0]),
        dtype=torch.bool,
        device=device,
    )
    rollout_groups, rollout_count = model.rollout(
        features,
        contact_mask,
        offset.to(device),
        n_events=int(args.rollouts),
        seed=int(args.seed) + 700_000 + int(args.recurrent_rank) * 1_000_003,
    )
    observed_groups = heldout.group_ids[eval_indices]
    observed_count = heldout.group_count[eval_indices]
    bins = int(cfg["event_encoding"]["rank_distribution_bins"])
    distribution = distribution_errors(
        rollout_groups,
        rollout_count,
        observed_groups,
        observed_count,
        bins=bins,
    )
    empirical_groups = heldout.group_ids[heldout.train_indices]
    empirical_count = heldout.group_count[heldout.train_indices]
    empirical_error = distribution_errors(
        empirical_groups,
        empirical_count,
        observed_groups,
        observed_count,
        bins=bins,
    )
    split_at = max(1, len(empirical_groups) // 2)
    split_half_error = distribution_errors(
        empirical_groups[:split_at],
        empirical_count[:split_at],
        empirical_groups[split_at:],
        empirical_count[split_at:],
        bins=bins,
    )
    metric_rows = [
        {
            "subject": heldout.subject,
            "dataset": heldout.dataset,
            "control": control,
            "seed": int(args.seed),
            "recurrent_rank": int(args.recurrent_rank),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "rollout_participant_count_mean": float(np.mean(rollout_count)),
            **metrics,
            **distribution,
        },
        {
            "subject": heldout.subject,
            "dataset": heldout.dataset,
            "control": "empirical_rank_distribution",
            "seed": int(args.seed),
            "recurrent_rank": int(args.recurrent_rank),
            "n_parameters": 0,
            "rollout_participant_count_mean": float(
                np.mean(np.sum(empirical_groups >= 0, axis=1))
            ),
            "heldout_event_nll": np.nan,
            **empirical_error,
        },
    ]
    pd.DataFrame(metric_rows).to_csv(run_dir / "heldout_metrics.csv", index=False)
    event_frame["control"] = control
    event_frame.to_csv(run_dir / "heldout_event_nll.csv", index=False)
    pd.concat(
        [
            _distribution_frame(
                heldout,
                control,
                rollout_groups,
                rollout_count,
                observed_groups,
                observed_count,
                bins,
            ),
            _distribution_frame(
                heldout,
                "empirical_rank_distribution",
                empirical_groups,
                empirical_count,
                observed_groups,
                observed_count,
                bins,
            ),
        ],
        ignore_index=True,
    ).to_csv(run_dir / "contact_rank_distributions.csv", index=False)
    pd.DataFrame([*shared_log, *calibration_log]).to_csv(
        run_dir / "training_log.csv", index=False
    )
    np.savez_compressed(
        run_dir / "free_rollouts.npz",
        event_group_ids=rollout_groups,
        event_group_count=rollout_count,
    )
    loading = model.contact_mode_loadings(
        torch.as_tensor(
            heldout.contact_features, dtype=torch.float32, device=device
        ),
        offset.to(device),
    )
    np.savez_compressed(
        run_dir / "mode_artifacts.npz",
        recurrent_rank=np.asarray(args.recurrent_rank),
        mode_u=(
            np.empty((args.hidden_size, 0), np.float32)
            if model.mode_u is None
            else model.mode_u.detach().cpu().numpy()
        ),
        mode_v=(
            np.empty((args.hidden_size, 0), np.float32)
            if model.mode_v is None
            else model.mode_v.detach().cpu().numpy()
        ),
        decay=model.decay.detach().cpu().numpy(),
        alpha=np.asarray(float(model.alpha.detach().cpu())),
        contact_names=heldout.contact_names,
        u_output_loading=loading["u_output_loading"].cpu().numpy(),
        v_output_loading=loading["v_output_loading"].cpu().numpy(),
    )
    trajectory_indices = eval_indices
    if len(trajectory_indices) > int(args.trajectory_events):
        take = np.linspace(
            0, len(trajectory_indices) - 1, int(args.trajectory_events)
        ).round().astype(int)
        trajectory_indices = trajectory_indices[np.unique(take)]
    real_trajectory = _trajectory_artifact(
        model,
        heldout,
        offset.to(device),
        trajectory_indices,
        device=device,
    )
    generated_record = heldout
    generated_groups = rollout_groups[: len(trajectory_indices)]
    generated_count = rollout_count[: len(trajectory_indices)]
    original_groups = generated_record.group_ids
    original_count = generated_record.group_count
    original_split = generated_record.event_split
    original_source_index = generated_record.event_source_index
    try:
        generated_record.group_ids = generated_groups
        generated_record.group_count = generated_count
        generated_record.event_split = np.ones(len(generated_groups), np.uint8)
        generated_record.event_source_index = np.arange(len(generated_groups))
        generated_trajectory = _trajectory_artifact(
            model,
            generated_record,
            offset.to(device),
            np.arange(len(generated_groups)),
            device=device,
        )
    finally:
        generated_record.group_ids = original_groups
        generated_record.group_count = original_count
        generated_record.event_split = original_split
        generated_record.event_source_index = original_source_index
    np.savez_compressed(run_dir / "real_event_trajectories.npz", **real_trajectory)
    np.savez_compressed(
        run_dir / "generated_event_trajectories.npz", **generated_trajectory
    )
    torch.save(
        {
            "contract": "topic5_structured_low_rank_rnn_cross_state_v0_5",
            "model_kwargs": model_kwargs,
            "model_state": state,
            "heldout_local_offset": offset.cpu(),
            "heldout_subject": heldout.subject,
            "seed": int(args.seed),
            "ictal_target_read": False,
        },
        run_dir / "checkpoint.pt",
    )
    empirical_margin = float(split_half_error["rank_wasserstein"])
    summary = {
        "status": "complete",
        "contract": "topic5_structured_low_rank_rnn_cross_state_v0_5",
        "subject": heldout.subject,
        "dataset": heldout.dataset,
        "seed": int(args.seed),
        "recurrent_rank": int(args.recurrent_rank),
        "hidden_size": int(args.hidden_size),
        "learning_rate": float(args.learning_rate),
        "local_offset_dim": int(args.local_offset_dim),
        "heldout_event_nll": float(metrics["heldout_event_nll"]),
        "distribution_errors": distribution,
        "empirical_distribution_errors": empirical_error,
        "empirical_split_half_variability": split_half_error,
        "rank_wasserstein_excess_over_empirical_variability": float(
            distribution["rank_wasserstein"]
            - empirical_error["rank_wasserstein"]
            - empirical_margin
        ),
        "shared_coverage": shared_coverage,
        "heldout_calibration_coverage": calibration_coverage,
        "ictal_target_read": False,
        "input_fingerprints": {
            subject: record.input_sha256 for subject, record in records.items()
        },
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True)
    )
    done = {
        "status": "complete",
        "subject": heldout.subject,
        "seed": int(args.seed),
        "recurrent_rank": int(args.recurrent_rank),
        "ictal_target_read": False,
    }
    (run_dir / "DONE.json").write_text(json.dumps(done, indent=2))
    (run_dir / "run_state.json").write_text(json.dumps(done, indent=2))
    print(json.dumps(_jsonable(summary)), flush=True)


if __name__ == "__main__":
    main()
