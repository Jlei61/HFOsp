#!/usr/bin/env python3
"""One non-LOSO development run of the Topic 5 training-budget audit.

The shared model is trained on the chronological first 90% of every patient's
train80 and validated on the remaining 10% of train80.  The outer heldout 20%
is relabelled ``2`` by :func:`development_records` and is never read.

Coverage cycles are nested: the run trains to ``--cycles`` and evaluates at the
end of every cycle, so budgets {1, 2, 4} come from a single 4-cycle run.
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
from src.topic5_rank_distribution import LinearStateSequenceRNN  # noqa: E402
from src.topic5_training_sufficiency import (  # noqa: E402
    aggregate_patient_metric,
    development_records,
    evaluate_decomposed,
    objective_from_name,
    plateau_verdict,
    run_environment,
)

TRAIN_PROBE_EVENTS = 2000


def _probe_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    """Deterministic evenly spaced subsample for the train-validation gap."""
    indices = np.asarray(indices, int)
    if indices.size <= int(limit):
        return indices
    take = np.linspace(0, indices.size - 1, int(limit)).round().astype(int)
    return indices[np.unique(take)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--updates-per-patient", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "adam"))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--objective", default="teacher_forced_one_step")
    parser.add_argument("--local-offset-dim", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.24)
    args = parser.parse_args()

    started = time.time()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(config_path.read_text())
    stage = cfg["stage_a"]
    fraction = float(cfg["hyperparameter_tuning"]["inner_validation_fraction_of_first80"])

    configuration = {
        "cycles": int(args.cycles),
        "updates_per_patient": int(args.updates_per_patient),
        "hidden_size": int(args.hidden_size),
        "learning_rate": float(args.learning_rate),
        "optimizer": str(args.optimizer),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "objective": str(args.objective),
        "local_offset_dim": int(args.local_offset_dim),
        "inner_validation_fraction_of_first80": fraction,
    }
    config_id = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()[:12]
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "config_id": config_id,
                "configuration": configuration,
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
                "contract": "topic5_rnn_training_sufficiency_v0_1_development",
                "config_id": config_id,
                "ictal_target_read": False,
                "outer_heldout_read": False,
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

    records = load_records(ROOT / cfg["outputs"]["dataset"])
    inner, split_audit = development_records(records, fraction)
    pd.DataFrame(split_audit).to_csv(run_dir / "inner_split_audit.csv", index=False)
    ordered = [inner[key] for key in sorted(inner)]

    model = LinearStateSequenceRNN(
        int(ordered[0].contact_features.shape[1]),
        hidden_size=int(args.hidden_size),
        contact_embedding_dim=int(stage["contact_embedding_dim"]),
        contact_encoder_hidden=int(stage["contact_encoder_hidden"]),
        local_offset_dim=int(args.local_offset_dim),
    )
    objective = objective_from_name(args.objective)
    cycle_rows: list[dict] = []

    def _on_cycle_end(cycle: int, trained_model, offsets) -> None:
        for record in ordered:
            offset = offsets[record.subject].detach()
            validation = evaluate_decomposed(
                trained_model,
                record,
                offset,
                device=device,
                batch_size=256,
            )
            train_probe = evaluate_decomposed(
                trained_model,
                record,
                offset,
                device=device,
                batch_size=256,
                indices=_probe_indices(record.train_indices, TRAIN_PROBE_EVENTS),
            )
            cycle_rows.append(
                {
                    "config_id": config_id,
                    "coverage_cycle": int(cycle),
                    "subject": record.subject,
                    "dataset": record.dataset,
                    "seed": int(args.seed),
                    "n_inner_train_events": int(record.train_indices.size),
                    "n_validation_events": int(validation["n_events"]),
                    "validation_contact_choice_nll": validation["contact_choice_nll"],
                    "validation_stop_contribution_nll": validation[
                        "stop_contribution_nll"
                    ],
                    "validation_total_nll": validation["event_total_nll"],
                    "train_contact_choice_nll": train_probe["contact_choice_nll"],
                    "train_total_nll": train_probe["event_total_nll"],
                    "train_validation_gap_contact_choice": (
                        validation["contact_choice_nll"]
                        - train_probe["contact_choice_nll"]
                    ),
                }
            )
        trained_model.train()

    from src.topic5_training_sufficiency import train_coverage_instrumented

    snapshots, training_rows, coverage = train_coverage_instrumented(
        model,
        ordered,
        coverage_cycles=int(args.cycles),
        updates_per_patient=int(args.updates_per_patient),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        local_learning_rate=float(stage["local_learning_rate"]),
        weight_decay=float(args.weight_decay),
        gradient_clip=float(stage["gradient_clip"]),
        local_offset_dim=int(args.local_offset_dim),
        device=device,
        seed=int(args.seed),
        objective=objective,
        optimizer_name=str(args.optimizer),
        on_cycle_end=_on_cycle_end,
    )

    cycle_frame = pd.DataFrame(cycle_rows)
    cycle_frame.to_csv(run_dir / "cycle_patient_metrics.csv", index=False)
    training_frame = pd.DataFrame(training_rows)
    training_frame.to_csv(run_dir / "training_log.csv", index=False)

    per_cycle = []
    for cycle, group in cycle_frame.groupby("coverage_cycle"):
        rows = group.to_dict("records")
        contact = aggregate_patient_metric(
            rows, value_key="validation_contact_choice_nll"
        )
        total = aggregate_patient_metric(rows, value_key="validation_total_nll")
        stop = aggregate_patient_metric(
            rows, value_key="validation_stop_contribution_nll"
        )
        gap = aggregate_patient_metric(
            rows, value_key="train_validation_gap_contact_choice"
        )
        cycle_updates = training_frame[training_frame.coverage_cycle == cycle]
        per_cycle.append(
            {
                "coverage_cycle": int(cycle),
                "patient_median_validation_contact_choice_nll": contact["median"],
                "patient_mean_validation_contact_choice_nll": contact["mean"],
                "patient_median_validation_total_nll": total["median"],
                "patient_median_validation_stop_nll": stop["median"],
                "patient_median_train_validation_gap": gap["median"],
                "optimizer_steps_in_cycle": int(len(cycle_updates)),
                "gradient_clip_fraction": float(cycle_updates.clipped.mean()),
                "parameter_update_norm_median": float(
                    cycle_updates.parameter_update_norm.median()
                ),
                "self_feed_probability_max": float(
                    cycle_updates.self_feed_probability.max()
                ),
                "model_fed_steps": int(cycle_updates.n_model_fed_steps.sum()),
                "tie_fallback_steps": int(cycle_updates.n_tie_fallback_steps.sum()),
            }
        )
    per_cycle_frame = pd.DataFrame(per_cycle).sort_values("coverage_cycle")
    per_cycle_frame.to_csv(run_dir / "cycle_summary.csv", index=False)

    plateau = plateau_verdict(
        per_cycle_frame.patient_median_validation_contact_choice_nll.tolist()
    )
    final_cycle = int(per_cycle_frame.coverage_cycle.max())
    torch.save(
        {
            "contract": "topic5_rnn_training_sufficiency_v0_1_development",
            "config_id": config_id,
            "configuration": configuration,
            "model_kwargs": {
                "contact_feature_dim": int(ordered[0].contact_features.shape[1]),
                "hidden_size": int(args.hidden_size),
                "contact_embedding_dim": int(stage["contact_embedding_dim"]),
                "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
                "local_offset_dim": int(args.local_offset_dim),
            },
            "cycle_states": {
                cycle: snapshot["model_state"]
                for cycle, snapshot in snapshots.items()
            },
            "cycle_offsets": {
                cycle: snapshot["offsets"] for cycle, snapshot in snapshots.items()
            },
            "ictal_target_read": False,
            "outer_heldout_read": False,
        },
        run_dir / "development_checkpoint.pt",
    )

    summary = {
        "status": "COMPLETE",
        "contract": "topic5_rnn_training_sufficiency_v0_1_development",
        "config_id": config_id,
        "configuration": configuration,
        "n_patients": len(ordered),
        "n_inner_train_events": int(
            sum(record.train_indices.size for record in ordered)
        ),
        "n_validation_events": int(
            sum(record.eval_indices.size for record in ordered)
        ),
        "n_optimizer_steps": int(len(training_frame)),
        "n_backward_chunks": int(training_frame.n_backward_chunks.sum()),
        "expected_optimizer_steps": int(
            int(args.cycles) * len(ordered) * int(args.updates_per_patient)
        ),
        "coverage_complete": bool(
            all(
                entry["fraction_of_first_cycle"] == 1.0 for entry in coverage.values()
            )
        ),
        "gradient_clip_fraction": float(training_frame.clipped.mean()),
        "parameter_update_norm_median": float(
            training_frame.parameter_update_norm.median()
        ),
        "per_cycle": per_cycle,
        "plateau": plateau,
        "final_cycle": final_cycle,
        "final_patient_median_validation_contact_choice_nll": float(
            per_cycle_frame.loc[
                per_cycle_frame.coverage_cycle == final_cycle,
                "patient_median_validation_contact_choice_nll",
            ].iloc[0]
        ),
        "environment": run_environment(),
        "input_fingerprints": {
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
        "outer_heldout_read": False,
    }
    if summary["n_optimizer_steps"] != summary["expected_optimizer_steps"]:
        raise RuntimeError("optimizer step count does not match the frozen budget")
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    done = {
        "status": "COMPLETE",
        "config_id": config_id,
        "seed": int(args.seed),
        "final_patient_median_validation_contact_choice_nll": summary[
            "final_patient_median_validation_contact_choice_nll"
        ],
        "ictal_target_read": False,
        "outer_heldout_read": False,
    }
    (run_dir / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    (run_dir / "run_state.json").write_text(json.dumps(done, indent=2) + "\n")
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
