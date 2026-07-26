#!/usr/bin/env python3
"""Post-hoc, target-sealed internal-dynamics analysis for formal K=2 runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records
from scripts.train_topic5_persistent_path_rnn import (
    _batch,
    load_path_mode_priors,
)
from src.topic5_persistent_path_rnn import (
    PersistentPathModeRNN,
    persistent_mixture_loss,
)


SEEDS = (20260726, 20260727, 20260728)
PROGRESS_BINS = np.linspace(0.0, 1.0, 11)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _sample_eval_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministic coverage of the full heldout chronology."""
    indices = np.asarray(indices, int)
    if len(indices) <= int(maximum):
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, int(maximum))
    positions = np.rint(positions).astype(int)
    if len(np.unique(positions)) != int(maximum):
        raise RuntimeError("deterministic heldout sampling produced duplicates")
    return indices[positions]


def _parameter_row(
    model: PersistentPathModeRNN,
    offset: torch.Tensor,
    *,
    subject: str,
    dataset: str,
    seed: int,
) -> dict:
    row = {
        "subject": subject,
        "dataset": dataset,
        "seed": int(seed),
        "alpha": float(model.alpha.detach()),
        "input_gain": float(model.input_gain.detach()),
        "propagation_gain": float(model.propagation_gain.detach()),
        "decay": float(model.decay.detach()),
        "inhibition_gain": float(model.inhibition_gain.detach()),
        "output_gain": float(model.output_gain.detach()),
        "inhibitory_alpha": float(model.inhibitory_alpha.detach()),
        "inhibitory_drive": float(model.inhibitory_drive.detach()),
        "endpoint_gain": float(model.endpoint_gain.detach()),
        "stop_bias": float(model.stop_bias.detach()),
        "stop_progress_gain": float(
            torch.nn.functional.softplus(
                model.raw_stop_progress_gain
            ).detach()
        ),
        "stop_inhibition_gain": float(
            torch.nn.functional.softplus(
                model.raw_stop_inhibition_gain
            ).detach()
        ),
        "continue_state_gain": float(
            torch.nn.functional.softplus(
                model.raw_continue_state_gain
            ).detach()
        ),
        "local_offset_mean": float(offset.mean()),
        "local_offset_sd": float(offset.std(unbiased=False)),
    }
    return row


@torch.no_grad()
def _trajectory_rows(
    model: PersistentPathModeRNN,
    record,
    prior,
    offset: torch.Tensor,
    cfg: dict,
    *,
    seed: int,
    max_events: int,
    batch_size: int,
) -> list[dict]:
    model.eval()
    device = torch.device("cpu")
    selected = _sample_eval_indices(record.eval_indices, int(max_events))
    rows = []
    mode = torch.as_tensor(prior.component_mode, dtype=torch.long)
    direction = torch.as_tensor(prior.component_direction, dtype=torch.long)
    n_components = len(prior.component_prior)
    entropy_scale = float(np.log(max(n_components, 2)))
    for start in range(0, len(selected), int(batch_size)):
        indices = selected[start : start + int(batch_size)]
        batch = _batch(record, prior, indices, device)
        output = model(**batch, local_offset=offset)
        loss = persistent_mixture_loss(
            output,
            batch["group_ids"],
            batch["group_count"],
            stop_calibration_weight=float(
                cfg["model"]["stop_calibration_weight"]
            ),
            endpoint_source_weight=float(
                cfg["model"]["endpoint_source_weight"]
            ),
        )
        posterior = loss["component_posterior_trajectory"]
        state = output["latent_state"]
        inhibition = output["inhibitory_state"]
        valid_contact = batch["contact_mask"][:, None, None, :].to(
            state.dtype
        )
        mean_excitation = (
            torch.relu(state) * valid_contact
        ).sum(3) / valid_contact.sum(3).clamp_min(1.0)
        weighted_excitation = torch.sum(
            posterior * mean_excitation.permute(0, 2, 1), dim=2
        )
        weighted_inhibition = torch.sum(
            posterior * inhibition.permute(0, 2, 1), dim=2
        )
        entropy = -torch.sum(
            posterior * torch.log(posterior.clamp_min(1e-12)), dim=2
        ) / entropy_scale
        posterior_max = posterior.max(2).values
        mode0 = torch.sum(posterior[:, :, mode == 0], dim=2)
        forward = torch.sum(posterior[:, :, direction > 0], dim=2)
        counts = batch["group_count"].cpu().numpy()
        for local, event_index in enumerate(indices):
            count = int(counts[local])
            denominator = max(count, 1)
            for step in range(count + 1):
                progress = float(step / denominator)
                bin_index = int(np.argmin(np.abs(PROGRESS_BINS - progress)))
                rows.append(
                    {
                        "subject": record.subject,
                        "dataset": record.dataset,
                        "seed": int(seed),
                        "event_index": int(event_index),
                        "step": int(step),
                        "terminal": bool(step == count),
                        "progress": progress,
                        "progress_bin": float(PROGRESS_BINS[bin_index]),
                        "posterior_entropy_normalized": float(
                            entropy[local, step]
                        ),
                        "posterior_max": float(
                            posterior_max[local, step]
                        ),
                        "posterior_weighted_excitation": float(
                            weighted_excitation[local, step]
                        ),
                        "posterior_weighted_inhibition": float(
                            weighted_inhibition[local, step]
                        ),
                        "mode0_probability": float(mode0[local, step]),
                        "forward_probability": float(
                            forward[local, step]
                        ),
                    }
                )
    return rows


def _summarize_trajectories(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "posterior_entropy_normalized",
        "posterior_max",
        "posterior_weighted_excitation",
        "posterior_weighted_inhibition",
        "mode0_probability",
        "forward_probability",
    ]
    patient = (
        frame.groupby(
            ["subject", "dataset", "seed", "progress_bin"], as_index=False
        )
        .agg(
            **{metric: (metric, "mean") for metric in metrics},
            n_event_steps=("event_index", "size"),
            n_events=("event_index", "nunique"),
        )
    )
    seed_median = (
        patient.groupby(
            ["subject", "dataset", "progress_bin"], as_index=False
        )[metrics]
        .median()
    )
    cohort_rows = []
    for progress_bin, current in seed_median.groupby("progress_bin"):
        for metric in metrics:
            values = current[metric].to_numpy(float)
            cohort_rows.append(
                {
                    "progress_bin": float(progress_bin),
                    "metric": metric,
                    "n_patients": int(len(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return patient, pd.DataFrame(cohort_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_persistent_path_mode_rnn_v1_0.yaml",
    )
    parser.add_argument("--max-events-per-run", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    cfg = yaml.safe_load(config_path.read_text())
    records = load_records(ROOT / cfg["inputs"]["dataset"])
    analysis = root / "analysis"
    gate_path = analysis / "formal_gate_summary.json"
    if not gate_path.exists():
        raise RuntimeError("formal cohort analysis must complete first")
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("formal run manifest is not complete")
    if manifest.get("ictal_target_read") is not False:
        raise RuntimeError("ictal target seal was violated")

    parameter_rows = []
    trajectory_rows = []
    for seed in SEEDS:
        priors = load_path_mode_priors(
            ROOT / cfg["inputs"]["path_mode_prior"],
            records,
            mode_count=2,
            control="intact",
            seed=int(seed),
            axis_floor=float(cfg["prior"]["axis_floor"]),
            neighbors=int(cfg["prior"]["neighbors"]),
        )
        for subject, record in records.items():
            run_dir = root / f"seed_{seed}" / "k_2" / "intact" / subject
            checkpoint = _load_checkpoint(run_dir / "checkpoint.pt")
            if checkpoint.get("ictal_target_read") is not False:
                raise RuntimeError(f"{subject}: ictal target entered checkpoint")
            model = PersistentPathModeRNN(
                record.contact_features.shape[1],
                local_offset_dim=int(cfg["model"]["local_offset_dim"]),
                use_recurrence=True,
            )
            model.load_state_dict(checkpoint["model_state"])
            offset = checkpoint["heldout_local_offset"].float()
            parameter_rows.append(
                _parameter_row(
                    model,
                    offset,
                    subject=subject,
                    dataset=record.dataset,
                    seed=int(seed),
                )
            )
            trajectory_rows.extend(
                _trajectory_rows(
                    model,
                    record,
                    priors[subject],
                    offset,
                    cfg,
                    seed=int(seed),
                    max_events=int(args.max_events_per_run),
                    batch_size=int(args.batch_size),
                )
            )

    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(analysis / "intact_k2_parameter_estimates.csv", index=False)
    trajectories = pd.DataFrame(trajectory_rows)
    patient, cohort = _summarize_trajectories(trajectories)
    patient.to_csv(
        analysis / "intact_k2_internal_dynamics_patient.csv", index=False
    )
    cohort.to_csv(
        analysis / "intact_k2_internal_dynamics_cohort.csv", index=False
    )
    summary = {
        "status": "complete",
        "contract": "topic5_persistent_path_mode_rnn_v1_0",
        "n_patients": int(parameters.subject.nunique()),
        "n_seeds": int(parameters.seed.nunique()),
        "max_events_per_run": int(args.max_events_per_run),
        "trajectory_progress_bins": PROGRESS_BINS.tolist(),
        "ictal_target_read": False,
    }
    (analysis / "internal_dynamics_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
