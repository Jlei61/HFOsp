#!/usr/bin/env python3
"""Fit the SPF-RNN comparison ladder for one patient and one seed.

Every model uses the same exact conditional k-subset observation likelihood,
the same train/monitor/test partitions, and the same fixed nuisance schedule.
Non-latent controls maximize that likelihood directly; latent models optimize
an ELBO and are evaluated with repeated future-blind prior-predictive estimates.
The estimator difference is therefore explicit and its Monte-Carlo uncertainty
is reported, rather than hidden inside a single likelihood number.

The monitor partition selects checkpoints. The development-test partition is
read only after selection. Each fit carries a training-adequacy verdict; an
inadequate run is a development artifact and cannot enter a comparison.

This runner never scores the previously read outer heldout20, never reads
ictal targets, A/B or axis labels, geometry, or inter-event intervals.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
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
    LatentTemplateModel,
    MarkovMixtureModel,
    PhaseConditionedPropagationFieldRNN,
    SharedPropagationFieldRNN,
    fit_static_scaffold_ml,
    load_subject_rank_events,
    sha256_file,
    training_adequacy_verdict,
)

LADDER = (
    "m0_static",
    "m1_markov",
    "m1_markov_phase",
    "m2_markov_mixture",
    "m2_markov_mixture_phase",
    "m3_template",
    "m4_field",
    "m4_field_phase",
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


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
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


def _model_seed(run_seed: int, model_name: str) -> int:
    """Stable per-model seed, independent of ladder construction order."""
    offset = int.from_bytes(
        hashlib.sha256(model_name.encode("utf-8")).digest()[:4], "little"
    )
    return int(run_seed) + offset % 1_000_000


def _subsample_chronological(indices: np.ndarray, limit: int | None) -> np.ndarray:
    """Target-blind even chronological coverage of an index pool."""
    values = np.asarray(indices, dtype=int)
    if limit is None or len(values) <= int(limit):
        return values
    return values[np.linspace(0, len(values) - 1, int(limit)).astype(int)]


def _batch(groups, counts, indices, device):
    selected = np.asarray(indices, dtype=int)
    return (
        torch.as_tensor(groups[selected], dtype=torch.long, device=device),
        torch.as_tensor(counts[selected], dtype=torch.long, device=device),
    )


def _trainable(model: torch.nn.Module) -> int:
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


def _indices_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _source_provenance(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    tracked = [
        ROOT / "src/topic5_shared_propagation_field.py",
        ROOT / "scripts/run_topic5_spf_model_ladder.py",
        config_path,
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head, dirty = "UNAVAILABLE", True
    return {
        "git_head": git_head,
        "git_worktree_dirty": dirty,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in tracked
            if path.exists() and path.is_file()
        },
    }


def _build(name: str, n_contacts: int, scaffold: np.ndarray, model_config: Mapping):
    latent_dim = int(model_config["latent_dim"])
    encoder_hidden = int(model_config["encoder_hidden"])
    if name == "m0_static":
        return MarkovMixtureModel(
            n_contacts, scaffold, n_components=1, use_transition=False
        )
    if name == "m1_markov":
        return MarkovMixtureModel(n_contacts, scaffold, n_components=1)
    if name == "m1_markov_phase":
        return MarkovMixtureModel(
            n_contacts,
            scaffold,
            n_components=1,
            phase_order=int(model_config["phase_order"]),
        )
    if name == "m2_markov_mixture":
        return MarkovMixtureModel(
            n_contacts,
            scaffold,
            n_components=int(model_config["mixture_components"]),
        )
    if name == "m2_markov_mixture_phase":
        return MarkovMixtureModel(
            n_contacts,
            scaffold,
            n_components=int(model_config["mixture_components"]),
            phase_order=int(model_config["phase_order"]),
        )
    if name == "m3_template":
        return LatentTemplateModel(
            n_contacts,
            scaffold,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
        )
    if name == "m4_field":
        return SharedPropagationFieldRNN(
            n_contacts,
            scaffold,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            jacobian_soft_cap=float(model_config["jacobian_soft_cap"]),
        )
    if name == "m4_field_phase":
        return PhaseConditionedPropagationFieldRNN(
            n_contacts,
            scaffold,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            jacobian_soft_cap=float(model_config["jacobian_soft_cap"]),
            phase_order=int(model_config["phase_order"]),
        )
    raise ValueError(f"unknown ladder member: {name}")


@torch.no_grad()
def _score_once(
    model: torch.nn.Module,
    groups: torch.Tensor,
    counts: torch.Tensor,
    *,
    prior_samples: int,
    importance_samples: int,
    seed: int,
    primary_estimator: str = "importance",
) -> dict[str, Any]:
    """Score one untouched batch: exact for Markov, MC for latent models.

    The latent models are never scored from the full-event posterior; only the
    future-blind initial-state prior is admissible.
    """
    if isinstance(model, MarkovMixtureModel):
        score = model.conditional_nll(groups, counts)
        prior_score = score
        estimator = "exact"
    else:
        prior_score = model.prior_predictive_log_likelihood(
            groups, counts, n_samples=int(prior_samples), seed=int(seed)
        )
        if primary_estimator == "prior":
            score = prior_score
            estimator = "prior_predictive_monte_carlo"
        elif primary_estimator == "importance":
            score = model.importance_weighted_log_likelihood(
                groups,
                counts,
                n_samples=int(importance_samples),
                seed=int(seed) + 499,
            )
            estimator = "importance_weighted_posterior_proposal"
        else:
            raise ValueError(f"unknown primary estimator: {primary_estimator}")
    return {
        "estimator": estimator,
        "nll_per_event": float(score["nll_per_event"]),
        "nll_per_decision": float(score["nll_per_decision"]),
        "prior_predictive_nll_per_event": float(prior_score["nll_per_event"]),
        "prior_predictive_nll_per_decision": float(
            prior_score["nll_per_decision"]
        ),
        "step_nll_per_decision_diagnostic": [
            float(value)
            for value in prior_score["step_nll_per_decision_diagnostic"]
            .detach()
            .cpu()
        ],
    }


@torch.no_grad()
def _score_repeated(
    model: torch.nn.Module,
    groups: torch.Tensor,
    counts: torch.Tensor,
    *,
    prior_samples: int,
    importance_samples: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Repeat latent MC scoring and expose estimator uncertainty."""
    n_repeats = 1 if isinstance(model, MarkovMixtureModel) else int(repeats)
    if n_repeats < 1:
        raise ValueError("score repeats must be positive")
    rows = [
        _score_once(
            model,
            groups,
            counts,
            prior_samples=prior_samples,
            importance_samples=importance_samples,
            seed=int(seed) + repeat * 1009,
        )
        for repeat in range(n_repeats)
    ]
    event = np.asarray([row["nll_per_event"] for row in rows], dtype=float)
    decision = np.asarray([row["nll_per_decision"] for row in rows], dtype=float)
    prior_event = np.asarray(
        [row["prior_predictive_nll_per_event"] for row in rows], dtype=float
    )
    prior_decision = np.asarray(
        [row["prior_predictive_nll_per_decision"] for row in rows], dtype=float
    )
    max_steps = max(len(row["step_nll_per_decision_diagnostic"]) for row in rows)
    step = np.full((n_repeats, max_steps), np.nan, dtype=float)
    for index, row in enumerate(rows):
        values = row["step_nll_per_decision_diagnostic"]
        step[index, : len(values)] = values
    return {
        "estimator": rows[0]["estimator"],
        "nll_per_event": float(np.mean(event)),
        "nll_per_event_mc_sd": float(np.std(event, ddof=1)) if n_repeats > 1 else 0.0,
        "nll_per_decision": float(np.mean(decision)),
        "nll_per_decision_mc_sd": (
            float(np.std(decision, ddof=1)) if n_repeats > 1 else 0.0
        ),
        "prior_predictive_nll_per_event": float(np.mean(prior_event)),
        "prior_predictive_nll_per_event_mc_sd": (
            float(np.std(prior_event, ddof=1)) if n_repeats > 1 else 0.0
        ),
        "prior_predictive_nll_per_decision": float(np.mean(prior_decision)),
        "prior_predictive_nll_per_decision_mc_sd": (
            float(np.std(prior_decision, ddof=1)) if n_repeats > 1 else 0.0
        ),
        "step_nll_per_decision_diagnostic": np.nanmean(step, axis=0).tolist(),
        "step_nll_mc_sd_diagnostic": (
            np.nanstd(step, axis=0, ddof=1).tolist()
            if n_repeats > 1
            else np.zeros(max_steps, dtype=float).tolist()
        ),
        "mc_repeats": n_repeats,
        "repeat_scores": rows,
    }


def _train_one(
    name: str,
    model: torch.nn.Module,
    groups: np.ndarray,
    counts: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    device: torch.device,
    training: Mapping,
    evaluation: Mapping,
    seed: int,
) -> dict[str, Any]:
    """Fit one ladder member with an inner-validation early-stopping curve."""
    validation_groups, validation_counts = _batch(
        groups, counts, validation_indices, device
    )
    monitor_samples = int(evaluation["monitor_prior_samples"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    history: list[dict[str, Any]] = []

    if not parameters:
        # M0 carries no free parameter: the frozen scaffold is already the
        # maximum-likelihood static solution, so there is nothing to fit.
        value = _score_once(
            model,
            validation_groups,
            validation_counts,
            prior_samples=monitor_samples,
            importance_samples=int(evaluation["importance_samples"]),
            seed=seed + 101,
            primary_estimator="prior",
        )["nll_per_event"]
        history.append(
            {
                "epoch": 0,
                "validation_nll_per_event": value,
                "learning_rate": 0.0,
            }
        )
        return {
            "history": history,
            "best_state": copy.deepcopy(model.state_dict()),
            "best_optimizer_state": None,
            "adequacy": {
                "converged": True,
                "verdict": "NO_FREE_PARAMETERS",
                "n_epochs": 1,
                "best_epoch": 0,
                "best_validation_nll": value,
                "final_validation_nll": value,
                "epochs_since_best": 0,
                "recent_relative_improvement": 0.0,
                "initial_validation_nll": value,
                "relative_improvement_from_initial": 0.0,
                "stopped_by_patience": False,
            },
        }

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay_optimizer"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["lr_scheduler_factor"]),
        patience=int(training["lr_scheduler_patience"]),
        min_lr=float(training["minimum_learning_rate"]),
    )
    batch_size = int(training["batch_events"])
    epochs = int(training["epochs"])
    rng = np.random.default_rng(int(seed))
    model.eval()
    initial_value = _score_once(
        model,
        validation_groups,
        validation_counts,
        prior_samples=monitor_samples,
        importance_samples=int(evaluation["importance_samples"]),
        seed=seed + 101,
        primary_estimator="prior",
    )["nll_per_event"]
    history.append(
        {
            "epoch": 0,
            "beta": 0.0,
            "train_objective_per_event": None,
            "validation_nll_per_event": initial_value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_mean": None,
            "gradient_clip_fraction": None,
        }
    )
    best_value = float(initial_value)
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    patience = int(training["early_stopping_patience"])
    relative_min_delta = float(training["early_stopping_relative_min_delta"])
    material_best_value = float(initial_value)
    last_material_improvement_epoch = 0
    is_latent = not isinstance(model, MarkovMixtureModel)
    stopped_by_patience = False

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        beta = float(training["beta_final"]) * min(
            1.0, (epoch + 1) / max(1, int(training["kl_warmup_epochs"]))
        )
        total = 0.0
        examples = 0
        component_totals: dict[str, float] = {}
        gradient_norms: list[float] = []
        clipped = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_groups, batch_counts = _batch(groups, counts, indices, device)
            optimizer.zero_grad(set_to_none=True)
            if is_latent:
                losses = model.elbo_loss(
                    batch_groups,
                    batch_counts,
                    beta=beta,
                    free_bits=float(training["free_bits_per_dimension"]),
                    jacobian_weight=float(training["jacobian_weight"]),
                    weight_decay=float(training["explicit_weight_penalty"]),
                )
                loss = losses["loss"]
            else:
                losses = model.conditional_nll(batch_groups, batch_counts)
                loss = losses["nll_per_event"]
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip"])
            )
            gradient_norms.append(float(gradient_norm))
            clipped += int(float(gradient_norm) > float(training["gradient_clip"]))
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
            examples += len(indices)
            if is_latent:
                for key in (
                    "reconstruction_nll_per_event",
                    "reconstruction_nll_per_decision",
                    "raw_kl",
                    "effective_kl",
                    "jacobian_penalty",
                    "weight_penalty",
                ):
                    component_totals[key] = component_totals.get(key, 0.0) + (
                        float(losses[key].detach().cpu()) * len(indices)
                    )

        model.eval()
        value = _score_once(
            model,
            validation_groups,
            validation_counts,
            prior_samples=monitor_samples,
            importance_samples=int(evaluation["importance_samples"]),
            seed=seed + 101,
            primary_estimator="prior",
        )["nll_per_event"]
        scheduler.step(value)
        row = {
            "epoch": epoch + 1,
            "beta": beta,
            "train_objective_per_event": total / max(examples, 1),
            "validation_nll_per_event": value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_clip_fraction": float(clipped / max(len(gradient_norms), 1)),
        }
        if is_latent:
            row.update(
                {
                    key: value_sum / max(examples, 1)
                    for key, value_sum in component_totals.items()
                }
            )
        history.append(row)
        if value < best_value - 1e-9:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        material_delta = max(
            abs(material_best_value) * relative_min_delta, 1e-9
        )
        if value < material_best_value - material_delta:
            material_best_value = float(value)
            last_material_improvement_epoch = epoch + 1
        curve = [row["validation_nll_per_event"] for row in history]
        if (epoch + 1) % 25 == 0:
            print(
                f"    [{name}] epoch={epoch + 1} val_nll={value:.4f} "
                f"best={best_value:.4f}",
                flush=True,
            )
        if epoch + 1 - last_material_improvement_epoch >= patience:
            stopped_by_patience = True
            break

    curve = [row["validation_nll_per_event"] for row in history]
    return {
        "history": history,
        "best_state": best_state,
        "best_optimizer_state": best_optimizer_state,
        "adequacy": training_adequacy_verdict(
            curve,
            patience=patience,
            tolerance=float(training["convergence_tolerance"]),
            initial_validation_nll=initial_value,
            minimum_relative_improvement=float(
                training["minimum_relative_improvement"]
            ),
            minimum_epochs=int(training["minimum_training_epochs"]),
            minimum_best_epoch=int(training["minimum_best_epoch"]),
            stopped_by_patience=stopped_by_patience,
        ),
    }


def run_subject_seed(
    subject: str,
    seed: int,
    config: Mapping,
    *,
    device: torch.device,
    output_dir: Path,
    max_train_events: int | None,
    max_validation_events: int | None,
    max_test_events: int | None,
    config_path: Path,
) -> dict[str, Any]:
    """Fit and score the whole ladder for one patient / seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _source_provenance(config_path)
    config_sha256 = sha256_file(config_path)
    _write_json(
        output_dir / "run_state.json",
        {
            "contract": CONTRACT_NAME,
            "status": "RUNNING",
            "subject": subject,
            "seed": int(seed),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha256,
            **provenance,
        },
    )
    dataset_dir = ROOT / config["data"]["dataset_dir"]
    record = load_subject_rank_events(dataset_dir, subject)
    train_indices, validation_indices, test_indices = record.development_split(
        float(config["data"]["inner_validation_fraction"]),
        float(config["data"]["inner_test_fraction"]),
    )
    ladder = config["ladder"]
    train_indices = _subsample_chronological(
        train_indices,
        max_train_events
        if max_train_events is not None
        else int(ladder["max_train_events"]),
    )
    validation_indices = _subsample_chronological(
        validation_indices,
        max_validation_events
        if max_validation_events is not None
        else int(ladder["max_validation_events"]),
    )
    test_indices = _subsample_chronological(
        test_indices,
        max_test_events
        if max_test_events is not None
        else int(ladder["max_test_events"]),
    )
    if (
        np.intersect1d(train_indices, validation_indices).size
        or np.intersect1d(train_indices, test_indices).size
        or np.intersect1d(validation_indices, test_indices).size
    ):
        raise RuntimeError("inner train/validation/test overlap")
    if np.intersect1d(
        np.r_[train_indices, validation_indices, test_indices],
        record.old_heldout20_indices,
    ).size:
        raise RuntimeError("old outer heldout20 leaked into the SPF ladder")

    _seed_everything(seed)
    training = config["training"]
    evaluation = config["evaluation"]
    model_config = dict(config["model"])
    model_config.setdefault(
        "mixture_components", int(ladder["mixture_components"])
    )

    scaffold = fit_static_scaffold_ml(
        record.group_ids,
        record.group_count,
        train_indices,
        steps=int(ladder["scaffold_steps"]),
        learning_rate=float(ladder["scaffold_learning_rate"]),
        seed=seed,
        device=device,
    )

    results: dict[str, Any] = {}
    generated_all: dict[str, np.ndarray] = {}
    checkpoints: dict[str, Any] = {}
    trained: dict[str, tuple[torch.nn.Module, dict[str, Any], float]] = {}
    model_roles = {
        "m0_static": "static_scaffold",
        "m1_markov": "stationary_first_order",
        "m1_markov_phase": "progress_matched_first_order_control",
        "m2_markov_mixture": "stationary_discrete_route_control",
        "m2_markov_mixture_phase": "progress_matched_discrete_route_control",
        "m3_template": "time_indexed_latent_template_control",
        "m4_field": "primary_autonomous_shared_field",
        "m4_field_phase": "nonautonomous_clock_diagnostic",
    }
    for name in LADDER:
        started = time.time()
        model_seed = _model_seed(seed, name)
        _seed_everything(model_seed)
        model = _build(
            name, len(record.contact_names), scaffold, model_config
        ).to(device)
        initial_state = copy.deepcopy(model.state_dict())
        fitted = _train_one(
            name,
            model,
            record.group_ids,
            record.group_count,
            train_indices,
            validation_indices,
            device=device,
            training=training,
            evaluation=evaluation,
            seed=model_seed,
        )
        attempts = [
            {
                "label": "primary",
                "learning_rate": float(training["learning_rate"]),
                "adequacy": copy.deepcopy(fitted["adequacy"]),
                "history": copy.deepcopy(fitted["history"]),
            }
        ]
        if fitted["adequacy"]["verdict"] == "EARLY_OPTIMUM_UNVERIFIED":
            rescue_training = dict(training)
            rescue_training["learning_rate"] = float(
                training["learning_rate"]
            ) * float(training["learning_rate_rescue_factor"])
            model.load_state_dict(initial_state)
            _seed_everything(model_seed)
            fitted = _train_one(
                name,
                model,
                record.group_ids,
                record.group_count,
                train_indices,
                validation_indices,
                device=device,
                training=rescue_training,
                evaluation=evaluation,
                seed=model_seed,
            )
            attempts.append(
                {
                    "label": "lower_learning_rate_rescue",
                    "learning_rate": float(
                        rescue_training["learning_rate"]
                    ),
                    "adequacy": copy.deepcopy(fitted["adequacy"]),
                    "history": copy.deepcopy(fitted["history"]),
                }
            )
        fitted["training_attempts"] = attempts
        fitted["adequacy"] = {
            **fitted["adequacy"],
            "rescue_used": len(attempts) > 1,
            "n_training_attempts": len(attempts),
            "primary_attempt_verdict": attempts[0]["adequacy"]["verdict"],
        }
        model.load_state_dict(fitted["best_state"])
        model.eval()
        trained[name] = (model, fitted, time.time() - started)
        checkpoints[name] = {
            "model_class": type(model).__name__,
            "model_state": _cpu_tree(fitted["best_state"]),
            "optimizer_state_at_best": _cpu_tree(
                fitted["best_optimizer_state"]
            ),
            "training_adequacy": fitted["adequacy"],
            "training_attempts": fitted["training_attempts"],
        }

    # The development-test values are first materialized only after every
    # checkpoint has been selected from the monitor partition.
    test_groups, test_counts = _batch(
        record.group_ids, record.group_count, test_indices, device
    )
    observed_groups = test_groups.cpu().numpy().astype(np.int16)
    observed_counts = test_counts.cpu().numpy().astype(np.int16)
    for name in LADDER:
        model, fitted, training_seconds = trained[name]
        started = time.time()
        formal = _score_repeated(
            model,
            test_groups,
            test_counts,
            prior_samples=int(evaluation["prior_predictive_samples"]),
            importance_samples=int(evaluation["importance_samples"]),
            repeats=int(evaluation["prior_predictive_repeats"]),
            seed=seed + 211,
        )
        rollout_rows = []
        first_generated = None
        for repeat in range(int(evaluation["rollout_repeats"])):
            generated = model.generate_conditioned(
                test_groups,
                test_counts,
                seed=seed + 307 + repeat * 1013,
            )
            generated_np = generated.cpu().numpy().astype(np.int16)
            if first_generated is None:
                first_generated = generated_np
            rollout_rows.append(
                distribution_errors(
                    generated_np,
                    observed_counts,
                    observed_groups,
                    observed_counts,
                )
            )
        assert first_generated is not None
        generated_all[name] = first_generated
        repertoire = {
            key: float(np.mean([row[key] for row in rollout_rows]))
            for key in rollout_rows[0]
        }
        repertoire_sd = {
            key: (
                float(np.std([row[key] for row in rollout_rows], ddof=1))
                if len(rollout_rows) > 1
                else 0.0
            )
            for key in rollout_rows[0]
        }
        results[name] = {
            "scientific_role": model_roles[name],
            "n_trainable_parameters": _trainable(model),
            "complete_event_likelihood_estimator": formal["estimator"],
            "development_test_nll_per_event": formal["nll_per_event"],
            "development_test_nll_per_event_mc_sd": formal[
                "nll_per_event_mc_sd"
            ],
            "development_test_nll_per_decision": formal["nll_per_decision"],
            "development_test_nll_per_decision_mc_sd": formal[
                "nll_per_decision_mc_sd"
            ],
            "prior_predictive_nll_per_event": formal[
                "prior_predictive_nll_per_event"
            ],
            "prior_predictive_nll_per_event_mc_sd": formal[
                "prior_predictive_nll_per_event_mc_sd"
            ],
            "prior_predictive_nll_per_decision": formal[
                "prior_predictive_nll_per_decision"
            ],
            "prior_predictive_nll_per_decision_mc_sd": formal[
                "prior_predictive_nll_per_decision_mc_sd"
            ],
            "step_nll_per_decision_diagnostic": formal[
                "step_nll_per_decision_diagnostic"
            ],
            "step_nll_mc_sd_diagnostic": formal[
                "step_nll_mc_sd_diagnostic"
            ],
            "likelihood_estimator_repeats": formal["mc_repeats"],
            "likelihood_repeat_scores": formal["repeat_scores"],
            "training_adequacy": fitted["adequacy"],
            "training_attempts": fitted["training_attempts"],
            "history": fitted["history"],
            "repertoire": repertoire,
            "repertoire_rollout_sd": repertoire_sd,
            "repertoire_rollout_repeats": rollout_rows,
            "training_elapsed_seconds": training_seconds,
            "evaluation_elapsed_seconds": time.time() - started,
        }

    np.savez_compressed(
        output_dir / "conditioned_generation.npz",
        observed_group_ids=observed_groups,
        observed_group_count=observed_counts,
        test_event_source_index=record.event_source_index[test_indices],
        static_scaffold_ml=scaffold,
        **{f"{name}_group_ids": value for name, value in generated_all.items()},
    )
    split_provenance = {
        "train_indices_sha256": _indices_sha256(train_indices),
        "monitor_validation_indices_sha256": _indices_sha256(validation_indices),
        "development_test_indices_sha256": _indices_sha256(test_indices),
        "old_heldout20_indices_sha256": _indices_sha256(
            record.old_heldout20_indices
        ),
    }
    torch.save(
        {
            "contract": CONTRACT_NAME,
            "subject": record.subject,
            "dataset": record.dataset,
            "seed": int(seed),
            "input_sha256": record.input_sha256,
            "config_sha256": config_sha256,
            "source_provenance": provenance,
            "split_provenance": split_provenance,
            "static_scaffold_ml": torch.as_tensor(scaffold),
            "models": checkpoints,
        },
        output_dir / "checkpoint.pt",
    )
    snn_audit_state = (
        ROOT
        / "results/topic5_shared_propagation_field/snn_positive_control"
        / "existing_artifact_reuse/ARTIFACT_AUDIT_STATE.json"
    )
    g0_input_status = (
        "EXISTING_ARTIFACTS_AUDITED_IDENTIFIABILITY_NOT_SCORED"
        if snn_audit_state.exists()
        and json.loads(snn_audit_state.read_text()).get("status") == "COMPLETE"
        else "EXISTING_ARTIFACT_AUDIT_PENDING"
    )
    summary = {
        "contract": CONTRACT_NAME,
        "status": "DEVELOPMENT_PILOT_NO_GATE_VERDICT",
        "subject": record.subject,
        "dataset": record.dataset,
        "seed": int(seed),
        "device": str(device),
        "input_sha256": record.input_sha256,
        "config_sha256": config_sha256,
        "source_provenance": provenance,
        "split_provenance": split_provenance,
        "n_contacts": int(len(record.contact_names)),
        "n_inner_train_events": int(len(train_indices)),
        "n_monitor_validation_events": int(len(validation_indices)),
        "n_development_test_events": int(len(test_indices)),
        "development_test_mean_decisions": float(
            np.mean(np.maximum(observed_counts - 1, 0))
        ),
        "scaffold_estimator": "maximum_likelihood_under_scoring_objective",
        "old_heldout20_scored": False,
        "ictal_target_read": False,
        "ab_or_axis_label_read": False,
        "geometry_input_read": False,
        "models": results,
        "checkpoint_path": str(output_dir / "checkpoint.pt"),
        "gate_status": {
            "g0_snn_identifiability": g0_input_status,
            "g1_full_event_generation": "PILOT_ONLY_NOT_JUDGED",
            "g2_stable_structure": "LOCKED_NOT_RUN",
            "g3_one_structure_many_trajectories": "LOCKED_NOT_RUN",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "run_state.json",
        {
            "contract": CONTRACT_NAME,
            "status": "COMPLETE",
            "subject": record.subject,
            "seed": int(seed),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha256,
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "checkpoint_sha256": sha256_file(output_dir / "checkpoint.pt"),
            **provenance,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPF-RNN model ladder")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the frozen target-blind six-patient pilot from Phase 0",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_propagation_field_v0_1.yaml",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-validation-events", type=int, default=None)
    parser.add_argument("--max-test-events", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["contract"]["name"] != CONTRACT_NAME:
        raise SystemExit("config is not the SPF-RNN v0.1 contract")
    if not bool(config["contract"]["recurrent_teacher_forcing_forbidden"]):
        raise SystemExit("config does not forbid recurrent teacher forcing")
    if tuple(config["ladder"]["members"]) != LADDER:
        raise SystemExit("config ladder members do not match the frozen runner")

    requested = str(
        args.device or config.get("resources", {}).get("device", "cuda")
    )
    device = torch.device(
        "cpu" if requested == "cuda" and not torch.cuda.is_available() else requested
    )

    if args.pilot:
        pilot_path = (
            ROOT
            / config["outputs"]["phase0"]
            / "pilot_subjects_target_blind.csv"
        )
        rows = pilot_path.read_text().strip().splitlines()[1:]
        subjects = [row.split(",")[0] for row in rows]
    elif args.subject:
        subjects = [args.subject]
    else:
        raise SystemExit("pass --subject or --pilot")
    seeds = args.seeds or [int(s) for s in config["training"]["seeds"]]

    root = args.output_root or (
        ROOT / config["outputs"]["development"] / "ladder_pilot_v0_4"
    )
    started = time.time()
    index: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in seeds:
            output_dir = Path(root) / f"{subject}_seed{seed}"
            summary = run_subject_seed(
                subject,
                int(seed),
                config,
                device=device,
                output_dir=output_dir,
                max_train_events=args.max_train_events,
                max_validation_events=args.max_validation_events,
                max_test_events=args.max_test_events,
                config_path=args.config,
            )
            for name, payload in summary["models"].items():
                index.append(
                    {
                        "subject": summary["subject"],
                        "dataset": summary["dataset"],
                        "seed": summary["seed"],
                        "model": name,
                        "n_contacts": summary["n_contacts"],
                        "n_inner_train_events": summary["n_inner_train_events"],
                        "n_monitor_validation_events": summary[
                            "n_monitor_validation_events"
                        ],
                        "n_development_test_events": summary[
                            "n_development_test_events"
                        ],
                        "n_trainable_parameters": payload[
                            "n_trainable_parameters"
                        ],
                        "development_test_nll_per_event": payload[
                            "development_test_nll_per_event"
                        ],
                        "development_test_nll_per_decision": payload[
                            "development_test_nll_per_decision"
                        ],
                        "training_adequacy": payload["training_adequacy"][
                            "verdict"
                        ],
                        "best_epoch": payload["training_adequacy"]["best_epoch"],
                        **{
                            f"repertoire_{key}": value
                            for key, value in payload["repertoire"].items()
                        },
                    }
                )
            print(
                json.dumps(
                    {
                        "subject": summary["subject"],
                        "seed": summary["seed"],
                        "nll": {
                            name: payload["development_test_nll_per_event"]
                            for name, payload in summary["models"].items()
                        },
                        "adequacy": {
                            name: payload["training_adequacy"]["verdict"]
                            for name, payload in summary["models"].items()
                        },
                    },
                    ensure_ascii=False,
                )
            )

    import pandas as pd

    frame = pd.DataFrame(index)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    # Shards of one pilot run in parallel and must not race on one filename;
    # aggregate_topic5_spf_ladder.py reconciles them afterwards.
    stem = (
        "ladder_runs"
        if len(subjects) > 1 or len(seeds) > 1
        else f"ladder_runs_{subjects[0]}_seed{seeds[0]}"
    )
    frame.to_csv(root_path / f"{stem}.csv", index=False)
    _write_json(
        root_path / f"{stem}_index.json",
        {
            "contract": CONTRACT_NAME,
            "status": "DEVELOPMENT_PILOT_NO_GATE_VERDICT",
            "config_sha256": sha256_file(args.config),
            "subjects": subjects,
            "seeds": [int(s) for s in seeds],
            "models": list(LADDER),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"wrote {root_path / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
