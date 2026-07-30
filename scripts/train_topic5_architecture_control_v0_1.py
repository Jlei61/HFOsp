#!/usr/bin/env python3
"""Train one target-sealed LOSO architecture-control fold for Topic 5."""
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

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _distribution_frame,
    _jsonable,
    _seed_everything,
    _sha256,
    calibrate_offset_coverage,
    evaluate_model,
    load_records,
    train_shared_coverage,
)
from src.topic5_rank_distribution import (  # noqa: E402
    LinearStateSequenceRNN,
    LowRankLeakySequenceRNN,
    VanillaRateSequenceRNN,
    distribution_errors,
)


ARCHITECTURES = {
    "linear_state": LinearStateSequenceRNN,
    "vanilla_rnn": VanillaRateSequenceRNN,
    **{
        f"low_rank_r{rank}": LowRankLeakySequenceRNN
        for rank in range(5)
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=(
            ROOT
            / "results/topic5_interictal_rank_distribution/runs/"
            "tuning_20260725_v1/selection/selected_hyperparameters.json"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument(
        "--hidden-size-override",
        type=int,
        default=None,
        help=(
            "Optional target-sealed capacity sensitivity. The primary ladder "
            "uses the frozen selected hidden size."
        ),
    )
    parser.add_argument(
        "--control-name",
        default=None,
        help="Optional explicit label for a capacity-matched sensitivity cell.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank-shuffle", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--shared-cycles", type=int, default=1)
    parser.add_argument("--calibration-cycles", type=int, default=4)
    parser.add_argument("--updates-per-patient", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.24)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    selection_path = (
        args.selection if args.selection.is_absolute() else ROOT / args.selection
    )
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(config_path.read_text())
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_hyperparameters"]
    records = load_records(ROOT / cfg["outputs"]["dataset"])
    if args.heldout_subject not in records:
        raise RuntimeError(f"held-out subject absent: {args.heldout_subject}")
    heldout = records[args.heldout_subject]
    outer = [
        record
        for subject, record in records.items()
        if subject != heldout.subject
    ]

    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction))
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(args.cpu_threads))
    stage = cfg["stage_a"]
    hidden_size = (
        int(args.hidden_size_override)
        if args.hidden_size_override is not None
        else int(selected["hidden_size"])
    )
    model_kwargs = {
        "contact_feature_dim": int(heldout.contact_features.shape[1]),
        "hidden_size": hidden_size,
        "contact_embedding_dim": int(stage["contact_embedding_dim"]),
        "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
        "local_offset_dim": int(selected["local_offset_dim"]),
    }
    model_class = ARCHITECTURES[args.architecture]
    if args.architecture.startswith("low_rank_r"):
        model_kwargs["recurrent_rank"] = int(args.architecture.rsplit("r", 1)[1])
    model = model_class(**model_kwargs)
    architecture_seed_offset = {
        "linear_state": 11_000_033,
        "vanilla_rnn": 17_000_051,
    }.get(
        args.architecture,
        19_000_057 + 1_000_003 * int(model_kwargs.get("recurrent_rank", 0)),
    )
    control_seed = (
        int(args.seed)
        + architecture_seed_offset
        + (23_000_069 if args.rank_shuffle else 0)
    )
    control = (
        str(args.control_name)
        if args.control_name is not None
        else args.architecture
    ) + ("_rank_shuffle" if args.rank_shuffle else "")
    state_payload = {
        "status": "RUNNING",
        "subject": heldout.subject,
        "architecture": args.architecture,
        "control": control,
        "seed": int(args.seed),
        "rank_shuffle": bool(args.rank_shuffle),
        "ictal_target_read": False,
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(state_payload, indent=2) + "\n"
    )
    started = time.time()
    shared_state, _, shared_log, shared_coverage = train_shared_coverage(
        model,
        outer,
        coverage_cycles=int(args.shared_cycles),
        updates_per_patient=int(args.updates_per_patient),
        batch_size=int(args.batch_size),
        learning_rate=float(selected["learning_rate"]),
        local_learning_rate=float(stage["local_learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
        gradient_clip=float(stage["gradient_clip"]),
        local_offset_dim=int(selected["local_offset_dim"]),
        device=device,
        seed=control_seed,
        rank_shuffle=bool(args.rank_shuffle),
    )
    model.load_state_dict(shared_state)
    offset, calibration_log, calibration_coverage = calibrate_offset_coverage(
        model,
        heldout,
        coverage_cycles=int(args.calibration_cycles),
        updates_per_cycle=int(args.updates_per_patient),
        batch_size=int(args.batch_size),
        local_learning_rate=float(stage["local_learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
        gradient_clip=float(stage["gradient_clip"]),
        local_offset_dim=int(selected["local_offset_dim"]),
        device=device,
        seed=control_seed + 500_000,
        rank_shuffle=bool(args.rank_shuffle),
    )
    checkpoint = run_dir / f"{control}_checkpoint.pt"
    torch.save(
        {
            "contract": "topic5_ordered_history_architecture_audit_v0_1",
            "architecture": args.architecture,
            "control": control,
            "model_kwargs": model_kwargs,
            "model_state": shared_state,
            "heldout_local_offset": offset.cpu(),
            "heldout_subject": heldout.subject,
            "seed": int(args.seed),
            "rank_shuffle": bool(args.rank_shuffle),
            "ictal_target_read": False,
        },
        checkpoint,
    )
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
        seed=control_seed + 700_000,
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
    metric_row = {
        "subject": heldout.subject,
        "dataset": heldout.dataset,
        "control": control,
        "architecture": args.architecture,
        "seed": int(args.seed),
        "rank_shuffle": bool(args.rank_shuffle),
        "hidden_size": hidden_size,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_local_offset_parameters": int(offset.numel()),
        "rollout_participant_count_mean": float(np.mean(rollout_count)),
        "rollout_participant_count_sd": float(np.std(rollout_count)),
        **metrics,
        **distribution,
    }
    pd.DataFrame([metric_row]).to_csv(
        run_dir / "heldout_metrics.csv", index=False
    )
    event_frame["control"] = control
    event_frame["architecture"] = args.architecture
    event_frame["seed"] = int(args.seed)
    event_frame.to_csv(run_dir / "heldout_event_nll.csv", index=False)
    _distribution_frame(
        heldout,
        control,
        rollout_groups,
        rollout_count,
        observed_groups,
        observed_count,
        bins,
    ).to_csv(run_dir / "contact_rank_distributions.csv", index=False)
    training_log = pd.DataFrame([*shared_log, *calibration_log])
    training_log["control"] = control
    training_log.to_csv(run_dir / "training_log.csv", index=False)
    np.savez_compressed(
        run_dir / f"{control}_free_rollouts.npz",
        event_group_ids=rollout_groups,
        event_group_count=rollout_count,
        seed=np.asarray(control_seed + 700_000),
    )

    engineering_pass = bool(
        np.isfinite(metric_row["heldout_event_nll"])
        and np.isfinite(metric_row["participation_mae"])
        and np.isfinite(metric_row["rank_wasserstein"])
        and 0.25
        < metric_row["rollout_participant_count_mean"]
        < heldout.contact_features.shape[0]
    )
    summary = {
        "status": "COMPLETE" if engineering_pass else "ENGINEERING_GATE_FAILED",
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "subject": heldout.subject,
        "dataset": heldout.dataset,
        "architecture": args.architecture,
        "control": control,
        "seed": int(args.seed),
        "rank_shuffle": bool(args.rank_shuffle),
        "model_kwargs": model_kwargs,
        "n_parameters": metric_row["n_parameters"],
        "n_outer_subjects": len(outer),
        "n_train_calibration_events": int(heldout.train_indices.size),
        "n_eval_events": int(len(eval_indices)),
        "metrics": metric_row,
        "coverage": {
            "shared": shared_coverage,
            "heldout_calibration": calibration_coverage,
        },
        "resources": {
            "runtime_seconds": float(time.time() - started),
            "cpu_threads": int(torch.get_num_threads()),
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
        "input_fingerprints": {
            "heldout_dataset_npz": heldout.input_sha256,
            "dataset_manifest": _sha256(
                ROOT
                / cfg["outputs"]["dataset"]
                / "dataset_manifest.json"
            ),
            "config": _sha256(config_path),
            "selected_hyperparameters": _sha256(selection_path),
        },
        "ictal_target_read": False,
        "early_ictal_target_arrays_deserialized": False,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n"
    )
    done = {
        "status": summary["status"],
        "subject": heldout.subject,
        "control": control,
        "seed": int(args.seed),
        "engineering_pass": engineering_pass,
        "ictal_target_read": False,
    }
    (run_dir / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    (run_dir / "run_state.json").write_text(
        json.dumps(done, indent=2) + "\n"
    )
    print(json.dumps(_jsonable(summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
