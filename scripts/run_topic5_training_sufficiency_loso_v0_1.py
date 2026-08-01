#!/usr/bin/env python3
"""One leave-one-patient-out cell of the Topic 5 training-sufficiency audit.

Two structurally identical modes:

``development``
    shared core on 33 patients' inner-training events, local offsets on the
    held-out patient's inner-training events, evaluation and free rollout on the
    held-out patient's inner-validation events.  The outer heldout 20% stays
    sealed.

``formal``
    shared core on 33 patients' train80, local offsets on the held-out
    patient's train80, evaluation and free rollout on the held-out patient's
    chronological heldout 20%.  Only run after Phase B and Phase C are frozen.

Free generation reuses the frozen constructive generator: the true first rank
set is revealed, every later contact comes from the model, and STOP comes from
the train-only rank-progress hazard.  Paired uniforms are derived from
``(subject, seed)`` alone, so every condition and every compared model shares
identical random numbers, identical source ranks and an identical rollout count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _seed_everything,
    _sha256,
    load_records,
)
from src.topic5_constructive_event_generator import (  # noqa: E402
    constant_stop_hazard,
    event_length_wasserstein,
    remove_revealed_source,
    source_conditioned_rollout,
    stop_hazard_curve,
    train_progress_hazard,
    train_static_log_scaffold,
)
from src.topic5_constructive_readback import transition_errors  # noqa: E402
from src.topic5_rank_distribution import (  # noqa: E402
    LinearStateSequenceRNN,
    distribution_errors,
)
from src.topic5_training_sufficiency import (  # noqa: E402
    calibrate_offset_instrumented,
    development_records,
    evaluate_decomposed,
    objective_from_name,
    paired_native_rollout,
    run_environment,
    train_coverage_instrumented,
)

#: ``native_model`` is a secondary diagnostic: the constructive generator
#: samples from the static scaffold plus the frozen ordered residual, whereas a
#: self-fed training objective makes the model robust to samples from its own
#: next-contact head.  Reporting both keeps a null result from being explained
#: away by a training/evaluation sampling mismatch.
ROLLOUT_CONDITIONS = ("full_constructive", "static_only", "native_model")


def _paired_uniform_seed(subject: str, seed: int) -> int:
    """Identical across conditions and across compared models by construction."""
    token = hashlib.sha256(
        f"{subject}:{int(seed)}:paired_uniforms_v0_1".encode()
    ).hexdigest()
    return int(token[:8], 16)


def _stop_curve_mae(predicted, observed, *, n_contacts: int) -> float:
    left = stop_hazard_curve(predicted, max_groups=n_contacts)
    right = stop_hazard_curve(observed, max_groups=n_contacts)
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.mean(np.abs(left[valid] - right[valid])))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--condition", required=True, help="label for this cell")
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--updates-per-patient", type=int, required=True)
    parser.add_argument("--offset-cycles", type=int, default=8)
    parser.add_argument(
        "--offset-snapshot-cycles",
        type=int,
        nargs="+",
        default=(4, 8),
        help="calibration budgets read from one calibration run",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "adam"))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--objective", default="teacher_forced_one_step")
    parser.add_argument("--local-offset-dim", type=int, default=4)
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help=(
            "evaluate an already frozen checkpoint instead of retraining; used "
            "for the published reference condition so the comparison uses the "
            "published model itself rather than a numerically equivalent copy"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    args = parser.parse_args()

    started = time.time()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(config_path.read_text())
    stage = cfg["stage_a"]

    configuration = {
        "mode": str(args.mode),
        "condition": str(args.condition),
        "cycles": int(args.cycles),
        "updates_per_patient": int(args.updates_per_patient),
        "offset_cycles": int(args.offset_cycles),
        "offset_snapshot_cycles": sorted(int(v) for v in args.offset_snapshot_cycles),
        "hidden_size": int(args.hidden_size),
        "learning_rate": float(args.learning_rate),
        "optimizer": str(args.optimizer),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "objective": str(args.objective),
        "local_offset_dim": int(args.local_offset_dim),
    }
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "configuration": configuration,
                "heldout_subject": args.heldout_subject,
                "source_config": str(config_path.relative_to(ROOT)),
                "source_config_sha256": _sha256(config_path),
                "environment": run_environment(),
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "contract": f"topic5_rnn_training_sufficiency_v0_1_{args.mode}",
                "subject": args.heldout_subject,
                "condition": args.condition,
                "seed": int(args.seed),
                "ictal_target_read": False,
                "outer_heldout_read": args.mode == "formal",
            },
            indent=2,
        )
        + "\n"
    )

    _seed_everything(int(args.seed))
    torch.set_num_threads(int(args.cpu_threads))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction))
        torch.cuda.reset_peak_memory_stats()

    frozen = load_records(ROOT / cfg["outputs"]["dataset"])
    if args.heldout_subject not in frozen:
        raise RuntimeError(f"held-out subject absent: {args.heldout_subject}")
    if args.mode == "development":
        fraction = float(
            cfg["hyperparameter_tuning"]["inner_validation_fraction_of_first80"]
        )
        records, _ = development_records(frozen, fraction)
    else:
        records = frozen
    heldout = records[args.heldout_subject]
    outer = [
        record for subject, record in records.items() if subject != heldout.subject
    ]
    if len(outer) != len(records) - 1:
        raise RuntimeError("leave-one-out split is malformed")

    cell_seed = (
        int(args.seed)
        + 31_000_013
        + int(
            hashlib.sha256(
                f"{args.condition}:{args.heldout_subject}".encode()
            ).hexdigest()[:6],
            16,
        )
    )
    objective = objective_from_name(args.objective)
    if args.from_checkpoint is not None:
        checkpoint_path = (
            args.from_checkpoint
            if args.from_checkpoint.is_absolute()
            else ROOT / args.from_checkpoint
        )
        frozen_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if frozen_checkpoint.get("heldout_subject") != heldout.subject:
            raise RuntimeError("frozen checkpoint subject mismatch")
        if int(frozen_checkpoint.get("seed")) != int(args.seed):
            raise RuntimeError("frozen checkpoint seed mismatch")
        if frozen_checkpoint.get("ictal_target_read") is not False:
            raise RuntimeError("frozen checkpoint is not target sealed")
        if frozen_checkpoint.get("architecture") not in (None, "linear_state"):
            raise RuntimeError("frozen checkpoint is not the linear-state model")
        model = LinearStateSequenceRNN(**frozen_checkpoint["model_kwargs"])
        shared_state = frozen_checkpoint["model_state"]
        model.load_state_dict(shared_state)
        model.to(device)
        offset_snapshots = {
            int(args.offset_cycles): frozen_checkpoint["heldout_local_offset"].to(device)
        }
        shared_rows, calibration_rows = [], []
        shared_coverage = {}
        calibration_coverage = {"frozen_checkpoint": True}
        provenance = {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": _sha256(checkpoint_path),
            "contract": frozen_checkpoint.get("contract"),
            "control": frozen_checkpoint.get("control"),
        }
        archived_metrics = checkpoint_path.parent / "heldout_metrics.csv"
        if archived_metrics.is_file():
            archived = pd.read_csv(archived_metrics)
            provenance["archived_heldout_event_nll"] = float(
                archived.heldout_event_nll.iloc[0]
            )
    else:
        provenance = None
        model = LinearStateSequenceRNN(
            int(heldout.contact_features.shape[1]),
            hidden_size=int(args.hidden_size),
            contact_embedding_dim=int(stage["contact_embedding_dim"]),
            contact_encoder_hidden=int(stage["contact_encoder_hidden"]),
            local_offset_dim=int(args.local_offset_dim),
        )
    if args.from_checkpoint is None:
        snapshots, shared_rows, shared_coverage = train_coverage_instrumented(
            model,
            outer,
            coverage_cycles=int(args.cycles),
            updates_per_patient=int(args.updates_per_patient),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            local_learning_rate=float(stage["local_learning_rate"]),
            weight_decay=float(args.weight_decay),
            gradient_clip=float(stage["gradient_clip"]),
            local_offset_dim=int(args.local_offset_dim),
            device=device,
            seed=cell_seed,
            objective=objective,
            optimizer_name=str(args.optimizer),
        )
        shared_state = snapshots[int(args.cycles)]["model_state"]
        model.load_state_dict(shared_state)

        offset_snapshots, calibration_rows, calibration_coverage = (
            calibrate_offset_instrumented(
                model,
                heldout,
                coverage_cycles=int(args.offset_cycles),
                updates_per_cycle=int(args.updates_per_patient),
                batch_size=int(args.batch_size),
                local_learning_rate=float(stage["local_learning_rate"]),
                weight_decay=float(args.weight_decay),
                gradient_clip=float(stage["gradient_clip"]),
                local_offset_dim=int(args.local_offset_dim),
                device=device,
                seed=cell_seed + 500_000,
                # the whole model is trained under one objective: a hybrid whose
                # core is rollout-aware but whose local offsets were fitted under
                # teacher forcing was never trained end to end under either
                objective=objective,
                snapshot_cycles=tuple(int(v) for v in args.offset_snapshot_cycles),
            )
        )

    eval_indices = np.asarray(heldout.eval_indices, dtype=int)
    observed_groups = np.asarray(heldout.group_ids[eval_indices], dtype=np.int16)
    observed_count = np.asarray(heldout.group_count[eval_indices], dtype=np.int16)
    source_mask = observed_groups == 0
    if not np.all(source_mask.any(axis=1)):
        raise RuntimeError("an evaluation event has no rank-zero contact")
    n_events, n_contacts = observed_groups.shape

    static = train_static_log_scaffold(heldout.group_ids, heldout.train_indices)
    hazard = train_progress_hazard(
        heldout.group_count, heldout.train_indices, max_groups=n_contacts
    )
    constant = constant_stop_hazard(heldout.group_count, heldout.train_indices)
    uniform_seed = _paired_uniform_seed(heldout.subject, int(args.seed))
    uniforms = np.random.default_rng(uniform_seed).random(
        (n_events, n_contacts), dtype=np.float64
    )
    uniforms_sha256 = hashlib.sha256(uniforms.tobytes()).hexdigest()
    suffix_observed = remove_revealed_source(observed_groups, source_mask)
    suffix_observed_count = np.maximum(observed_count - 1, 0)

    features = torch.as_tensor(
        heldout.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    contact_mask = torch.ones(
        (1, n_contacts), dtype=torch.bool, device=device
    )

    metric_rows: list[dict] = []
    payload: dict[str, np.ndarray] = {
        "observed_group_ids": observed_groups,
        "observed_group_count": observed_count,
        "revealed_source_mask": source_mask.astype(np.uint8),
        "eval_indices": eval_indices,
    }
    for offset_cycle in sorted(offset_snapshots):
        offset = offset_snapshots[offset_cycle].to(device)
        likelihood = evaluate_decomposed(
            model, heldout, offset, device=device, batch_size=256
        )
        row = {
            "subject": heldout.subject,
            "dataset": heldout.dataset,
            "mode": args.mode,
            "condition": args.condition,
            "objective": args.objective,
            "seed": int(args.seed),
            "cycles": int(args.cycles),
            "updates_per_patient": int(args.updates_per_patient),
            "offset_cycles": int(offset_cycle),
            "hidden_size": int(args.hidden_size),
            "learning_rate": float(args.learning_rate),
            "optimizer": str(args.optimizer),
            "weight_decay": float(args.weight_decay),
            "n_contacts": int(n_contacts),
            "n_train_events": int(heldout.train_indices.size),
            "n_eval_events": int(n_events),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "rollout_condition": "none",
            **{f"likelihood_{key}": value for key, value in likelihood.items()},
        }
        metric_rows.append(row)

        if not args.rollout:
            continue
        for rollout_condition in ROLLOUT_CONDITIONS:
            if rollout_condition == "native_model":
                generated, generated_count = paired_native_rollout(
                    model,
                    features,
                    contact_mask,
                    offset,
                    source_mask,
                    uniforms,
                )
            else:
                rollout = source_conditioned_rollout(
                    model,
                    features,
                    contact_mask,
                    offset,
                    source_mask,
                    uniforms,
                    static,
                    hazard,
                    condition=rollout_condition,
                    constant_hazard=constant,
                    batch_size=int(args.batch_size),
                    uniforms_sha256=uniforms_sha256,
                )
                generated = rollout.event_group_ids
                generated_count = rollout.event_group_count
            suffix_generated = remove_revealed_source(generated, source_mask)
            whole = distribution_errors(
                generated, generated_count, observed_groups, observed_count, bins=10
            )
            suffix = distribution_errors(
                suffix_generated,
                np.maximum(generated_count - 1, 0),
                suffix_observed,
                suffix_observed_count,
                bins=10,
            )
            transition = transition_errors(observed_groups, generated)
            metric_rows.append(
                {
                    **{key: value for key, value in row.items() if not key.startswith("likelihood_")},
                    **{f"likelihood_{key}": value for key, value in likelihood.items()},
                    "rollout_condition": rollout_condition,
                    **{f"whole_{key}": value for key, value in whole.items()},
                    **{f"suffix_{key}": value for key, value in suffix.items()},
                    **transition,
                    "event_length_wasserstein": event_length_wasserstein(
                        generated_count, observed_count
                    ),
                    "stop_hazard_mae": _stop_curve_mae(
                        generated_count, observed_count, n_contacts=n_contacts
                    ),
                    "generated_zero_suffix_fraction": float(
                        np.mean(generated_count == 1)
                    ),
                    "uniforms_sha256": uniforms_sha256,
                }
            )
            if offset_cycle == max(offset_snapshots):
                payload[f"{rollout_condition}__event_group_ids"] = generated
                payload[f"{rollout_condition}__event_group_count"] = generated_count

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "cell_metrics.csv", index=False)
    training_frame = pd.DataFrame([*shared_rows, *calibration_rows])
    training_frame.to_csv(run_dir / "training_log.csv", index=False)
    if args.rollout:
        np.savez_compressed(run_dir / "rollouts.npz", **payload)
    if args.save_checkpoint:
        torch.save(
            {
                "contract": f"topic5_rnn_training_sufficiency_v0_1_{args.mode}",
                "configuration": configuration,
                "model_kwargs": {
                    "contact_feature_dim": int(heldout.contact_features.shape[1]),
                    "hidden_size": int(args.hidden_size),
                    "contact_embedding_dim": int(stage["contact_embedding_dim"]),
                    "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
                    "local_offset_dim": int(args.local_offset_dim),
                },
                "model_state": shared_state,
                "heldout_local_offsets": {
                    int(cycle): value.cpu()
                    for cycle, value in offset_snapshots.items()
                },
                "heldout_subject": heldout.subject,
                "seed": int(args.seed),
                "ictal_target_read": False,
            },
            run_dir / "checkpoint.pt",
        )

    trained_here = args.from_checkpoint is None
    if provenance is not None and "archived_heldout_event_nll" in provenance:
        # reproducing the published number from the published checkpoint under
        # this pipeline's evaluator is the audit that the reference condition
        # really is the published model
        reproduced = float(
            metrics.loc[
                metrics.rollout_condition == "none", "likelihood_event_total_nll"
            ].iloc[0]
        )
        provenance["reproduced_heldout_event_nll"] = reproduced
        provenance["absolute_difference"] = abs(
            reproduced - provenance["archived_heldout_event_nll"]
        )
        provenance["reproduction_pass"] = bool(
            provenance["absolute_difference"] < 1e-5
        )
    shared_updates = pd.DataFrame(shared_rows)
    summary = {
        "status": "COMPLETE",
        "contract": f"topic5_rnn_training_sufficiency_v0_1_{args.mode}",
        "subject": heldout.subject,
        "dataset": heldout.dataset,
        "condition": args.condition,
        "configuration": configuration,
        "trained_in_this_cell": trained_here,
        "frozen_checkpoint_provenance": provenance,
        "n_outer_subjects": len(outer),
        "n_train_events": int(heldout.train_indices.size),
        "n_eval_events": int(n_events),
        "n_shared_optimizer_steps": int(len(shared_updates)),
        "expected_shared_optimizer_steps": (
            int(int(args.cycles) * len(outer) * int(args.updates_per_patient))
            if trained_here
            else 0
        ),
        "n_shared_backward_chunks": (
            int(shared_updates.n_backward_chunks.sum()) if trained_here else 0
        ),
        "n_offset_optimizer_steps": int(len(calibration_rows)),
        "gradient_clip_fraction_shared": (
            float(shared_updates.clipped.mean()) if trained_here else None
        ),
        "parameter_update_norm_median_shared": (
            float(shared_updates.parameter_update_norm.median())
            if trained_here
            else None
        ),
        "self_fed_steps": (
            int(shared_updates.n_model_fed_steps.sum()) if trained_here else 0
        ),
        "self_fed_eligible_steps": (
            int(shared_updates.n_self_feed_eligible_steps.sum())
            if trained_here
            else 0
        ),
        "tie_fallback_steps": (
            int(shared_updates.n_tie_fallback_steps.sum()) if trained_here else 0
        ),
        "coverage": {
            "shared_complete": bool(
                all(
                    entry["fraction_of_first_cycle"] == 1.0
                    for entry in shared_coverage.values()
                )
            )
            if trained_here
            else None,
            "heldout_calibration": calibration_coverage,
        },
        "rollout": {
            "enabled": bool(args.rollout),
            "conditions": list(ROLLOUT_CONDITIONS) if args.rollout else [],
            "n_rollout_events": int(n_events),
            "uniforms_seed": int(uniform_seed),
            "uniforms_sha256": uniforms_sha256,
            "source_ranks_revealed": True,
        },
        "environment": run_environment(),
        "input_fingerprints": {
            "heldout_dataset_npz": heldout.input_sha256,
            "dataset_manifest": _sha256(
                ROOT / cfg["outputs"]["dataset"] / "dataset_manifest.json"
            ),
            "config": _sha256(config_path),
        },
        "resources": {
            "runtime_seconds": float(time.time() - started),
            "peak_rss_gb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            ),
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "gpu_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
            ),
        },
        "ictal_target_read": False,
        "outer_heldout_read": args.mode == "formal",
    }
    if (
        trained_here
        and summary["n_shared_optimizer_steps"]
        != summary["expected_shared_optimizer_steps"]
    ):
        raise RuntimeError("shared optimizer step count does not match the budget")
    (run_dir / "run_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n"
    )
    done = {
        "status": "COMPLETE",
        "subject": heldout.subject,
        "condition": args.condition,
        "seed": int(args.seed),
        "mode": args.mode,
        "ictal_target_read": False,
    }
    (run_dir / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    (run_dir / "run_state.json").write_text(json.dumps(done, indent=2) + "\n")
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
