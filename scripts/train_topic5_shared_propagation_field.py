#!/usr/bin/env python3
"""Train one patient-specific SPF-RNN v2.0-core development run.

SUPERSEDED as the comparison entry point -- kept for the provenance of the
first smoke run only.  Use ``scripts/run_topic5_spf_model_ladder.py`` instead.

This runner fits the latent field by gradient descent but takes the static
scaffold and the first-order residual from moment estimators, so an
M4-vs-M0/M1 gap produced here is confounded with the estimator rather than the
mechanism (spec v0.1 section 5.2).  It also has no convergence criterion: its
recorded smoke run reached only six gradient updates.

This runner is deliberately development-only by default.  It splits the old
train80 chronologically into inner train/validation and never scores the
previously read outer heldout20.  Formal cohort orchestration remains locked
until the SNN identifiability dataset and all gates are frozen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required; use the cuda_env environment") from exc

from src.topic5_rank_distribution import distribution_errors  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    CONTRACT_NAME,
    SharedPropagationFieldRNN,
    baseline_conditioned_log_likelihood,
    estimate_static_participation_bias,
    generate_first_order_markov_conditioned,
    generate_static_conditioned,
    load_subject_rank_events,
    sha256_file,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_pair_residual,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _device(name: str) -> torch.device:
    requested = str(name)
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _batch(
    groups: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(
            groups[np.asarray(indices, dtype=int)],
            dtype=torch.long,
            device=device,
        ),
        torch.as_tensor(
            counts[np.asarray(indices, dtype=int)],
            dtype=torch.long,
            device=device,
        ),
    )


def _subsample_chronological(indices: np.ndarray, limit: int | None) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if limit is None or len(values) <= int(limit):
        return values
    # Even chronological coverage is more representative than taking only the
    # earliest block, while selection remains target-blind.
    positions = np.linspace(0, len(values) - 1, int(limit)).astype(int)
    return values[positions]


@torch.no_grad()
def _evaluate(
    model: SharedPropagationFieldRNN,
    groups: np.ndarray,
    counts: np.ndarray,
    validation_indices: np.ndarray,
    static_bias: np.ndarray,
    transition_residual: np.ndarray,
    *,
    device: torch.device,
    prior_samples: int,
    importance_samples: int,
    seed: int,
) -> dict[str, Any]:
    batch_groups, batch_counts = _batch(
        groups, counts, validation_indices, device
    )
    static_tensor = torch.as_tensor(static_bias, dtype=torch.float32, device=device)
    residual_tensor = torch.as_tensor(
        transition_residual, dtype=torch.float32, device=device
    )

    prior = model.prior_predictive_log_likelihood(
        batch_groups,
        batch_counts,
        n_samples=int(prior_samples),
        seed=int(seed) + 101,
    )
    importance = model.importance_weighted_log_likelihood(
        batch_groups,
        batch_counts,
        n_samples=int(importance_samples),
        seed=int(seed) + 211,
    )
    static_likelihood = baseline_conditioned_log_likelihood(
        static_tensor, batch_groups, batch_counts
    )
    markov_likelihood = baseline_conditioned_log_likelihood(
        static_tensor,
        batch_groups,
        batch_counts,
        transition_residual=residual_tensor,
    )

    generated_spf = model.generate_conditioned(
        batch_groups, batch_counts, seed=int(seed) + 307
    )
    generated_static = generate_static_conditioned(
        static_tensor,
        batch_groups,
        batch_counts,
        seed=int(seed) + 307,
    )
    generated_markov = generate_first_order_markov_conditioned(
        static_tensor,
        residual_tensor,
        batch_groups,
        batch_counts,
        seed=int(seed) + 307,
    )
    observed_groups = batch_groups.cpu().numpy().astype(np.int16)
    observed_counts = batch_counts.cpu().numpy().astype(np.int16)
    generated = {
        "spf": generated_spf.cpu().numpy().astype(np.int16),
        "static": generated_static.cpu().numpy().astype(np.int16),
        "markov": generated_markov.cpu().numpy().astype(np.int16),
    }
    repertoire = {
        name: distribution_errors(
            prediction,
            observed_counts,
            observed_groups,
            observed_counts,
        )
        for name, prediction in generated.items()
    }
    return {
        "prior_predictive_nll_per_event": prior["nll_per_event"],
        "prior_predictive_nll_per_decision": prior["nll_per_decision"],
        "iwae_nll_per_event_diagnostic": importance["nll_per_event"],
        "static_nll_per_event": static_likelihood["nll_per_event"],
        "static_nll_per_decision": static_likelihood["nll_per_decision"],
        "markov_nll_per_event": markov_likelihood["nll_per_event"],
        "markov_nll_per_decision": markov_likelihood["nll_per_decision"],
        "repertoire": repertoire,
        "generated": generated,
        "observed_groups": observed_groups,
        "observed_counts": observed_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one SPF-RNN patient development run"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_propagation_field_v0_1.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-validation-events", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Only allowed for an explicitly targeted development run directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["contract"]["name"] != CONTRACT_NAME:
        raise SystemExit("config is not the SPF-RNN v0.1 contract")
    if not bool(config["contract"]["recurrent_teacher_forcing_forbidden"]):
        raise SystemExit("config does not forbid recurrent teacher forcing")

    dataset_dir = ROOT / config["data"]["dataset_dir"]
    record = load_subject_rank_events(dataset_dir, args.subject)
    train_indices, validation_indices = record.inner_split(
        float(config["data"]["inner_validation_fraction"])
    )
    train_indices = _subsample_chronological(
        train_indices, args.max_train_events
    )
    configured_validation_limit = int(
        config["evaluation"]["max_validation_events_for_development"]
    )
    validation_limit = (
        configured_validation_limit
        if args.max_validation_events is None
        else int(args.max_validation_events)
    )
    validation_indices = _subsample_chronological(
        validation_indices, validation_limit
    )
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("inner train/validation overlap")
    if np.intersect1d(
        np.r_[train_indices, validation_indices],
        record.old_heldout20_indices,
    ).size:
        raise RuntimeError("old outer heldout20 leaked into SPF development")

    training = config["training"]
    model_config = config["model"]
    epochs = int(training["epochs"] if args.epochs is None else args.epochs)
    device = _device(
        args.device
        or str(config.get("resources", {}).get("device", "cuda"))
    )
    _seed_everything(args.seed)

    static_bias = estimate_static_participation_bias(
        record.group_ids,
        train_indices,
        alpha=float(model_config["static_bias_alpha"]),
    )
    pair = estimate_pair_residual(record.group_ids, train_indices)
    transition_residual = np.asarray(pair.residual, dtype=np.float32)
    model = SharedPropagationFieldRNN(
        len(record.contact_names),
        static_bias,
        latent_dim=int(model_config["latent_dim"]),
        encoder_hidden=int(model_config["encoder_hidden"]),
        jacobian_soft_cap=float(model_config["jacobian_soft_cap"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay_optimizer"]),
    )

    if args.output_dir is None:
        run_name = (
            f"{record.subject}_seed{args.seed}_d{model_config['latent_dim']}"
        )
        output_dir = ROOT / config["outputs"]["development"] / run_name
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; refusing to overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(args.config, output_dir / "config_snapshot.yaml")

    start = time.time()
    run_state = {
        "contract": CONTRACT_NAME,
        "state": "RUNNING",
        "subject": record.subject,
        "dataset": record.dataset,
        "seed": int(args.seed),
        "device": str(device),
        "input_sha256": record.input_sha256,
        "config_sha256": sha256_file(args.config),
        "n_inner_train_events": int(len(train_indices)),
        "n_inner_validation_events": int(len(validation_indices)),
        "old_heldout20_scored": False,
        "ictal_target_read": False,
        "ab_or_axis_label_read": False,
        "geometry_input_read": False,
    }
    _write_json(output_dir / "run_state.json", run_state)

    batch_size = int(training["batch_events"])
    history = []
    rng = np.random.default_rng(int(args.seed))
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        beta = float(training["beta_final"]) * min(
            1.0,
            (epoch + 1) / max(1, int(training["kl_warmup_epochs"])),
        )
        aggregate: dict[str, float] = {}
        examples = 0
        clipped = 0
        updates = 0
        for start_index in range(0, len(order), batch_size):
            indices = order[start_index : start_index + batch_size]
            batch_groups, batch_counts = _batch(
                record.group_ids, record.group_count, indices, device
            )
            optimizer.zero_grad(set_to_none=True)
            losses = model.elbo_loss(
                batch_groups,
                batch_counts,
                beta=beta,
                free_bits=float(training["free_bits_per_dimension"]),
                jacobian_weight=float(training["jacobian_weight"]),
                weight_decay=float(training["explicit_weight_penalty"]),
            )
            losses["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            clipped += int(
                float(gradient_norm) > float(training["gradient_clip"])
            )
            optimizer.step()
            updates += 1
            examples += len(indices)
            for key, value in losses.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(
                    value.detach().cpu()
                ) * len(indices)
        epoch_row = {
            "epoch": epoch + 1,
            "beta": beta,
            "updates": updates,
            "examples": examples,
            "gradient_clipping_fraction": clipped / max(updates, 1),
            **{
                key: value / max(examples, 1)
                for key, value in aggregate.items()
            },
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False))

    model.eval()
    evaluation = _evaluate(
        model,
        record.group_ids,
        record.group_count,
        validation_indices,
        static_bias,
        transition_residual,
        device=device,
        prior_samples=int(config["evaluation"]["prior_predictive_samples"]),
        importance_samples=int(config["evaluation"]["importance_samples"]),
        seed=int(args.seed),
    )
    generated = evaluation.pop("generated")
    observed_groups = evaluation.pop("observed_groups")
    observed_counts = evaluation.pop("observed_counts")

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "contract": CONTRACT_NAME,
            "subject": record.subject,
            "seed": int(args.seed),
            "model_state": model.state_dict(),
            "model_config": dict(model_config),
            "static_bias": static_bias,
            "transition_residual": transition_residual,
            "input_sha256": record.input_sha256,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        output_dir / "conditioned_generation.npz",
        observed_group_ids=observed_groups,
        observed_group_count=observed_counts,
        spf_group_ids=generated["spf"],
        static_group_ids=generated["static"],
        markov_group_ids=generated["markov"],
        validation_event_source_index=record.event_source_index[
            validation_indices
        ],
    )
    summary = {
        "contract": CONTRACT_NAME,
        "status": "DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        "subject": record.subject,
        "dataset": record.dataset,
        "seed": int(args.seed),
        "device": str(device),
        "input_sha256": record.input_sha256,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "n_inner_train_events": int(len(train_indices)),
        "n_inner_validation_events": int(len(validation_indices)),
        "old_heldout20_scored": False,
        "ictal_target_read": False,
        "model_selection_scope": "old_train80_inner_split_only",
        "evaluation": evaluation,
        "training_history": history,
        "elapsed_seconds": time.time() - start,
        "gate_status": {
            "g0_snn_identifiability": "LOCKED_NOT_RUN",
            "g1_full_event_generation": "NOT_JUDGED_BY_ONE_DEVELOPMENT_RUN",
            "g2_stable_structure": "LOCKED_NOT_RUN",
            "g3_one_structure_many_trajectories": "LOCKED_NOT_RUN",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    run_state.update(
        {
            "state": "COMPLETE",
            "elapsed_seconds": summary["elapsed_seconds"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
        }
    )
    _write_json(output_dir / "run_state.json", run_state)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
