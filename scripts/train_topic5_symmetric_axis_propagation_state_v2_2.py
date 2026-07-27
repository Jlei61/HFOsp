#!/usr/bin/env python3
"""Train one v2.2 development run on real interictal rank-set events.

One invocation fits the symmetric-axis full model and its exactly matched
local-isotropic control.  The ictal target tree is never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import Tensor
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    SymmetricAxisPropagationStateRNN,
    canonicalize_axis,
    estimate_node_hazard_bias,
    node_bias_fingerprint,
)


OBJECTIVE_HORIZON = {
    "next_only": 0,
    "next_plus_rollout_h3": 3,
    "next_plus_rollout_h5": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _events_to_numpy(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        counts = np.asarray(data["event_group_count"], dtype=np.int64)
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
        names = [str(value) for value in data["contact_names"]]
        times = np.asarray(data["event_abs_time"], dtype=np.float64)
    if groups.ndim != 2 or counts.shape != (groups.shape[0],):
        raise ValueError("invalid event group schema")
    if coords.shape != (groups.shape[1], 3) or not np.all(np.isfinite(coords)):
        raise ValueError("development physical-axis training requires full geometry")
    if not np.all(np.diff(times) >= 0):
        raise ValueError("events are not chronological")
    return {
        "groups": groups,
        "counts": counts,
        "coords": coords,
        "names": names,
        "times": times,
    }


def _partition_indices(n_events: int) -> dict[str, np.ndarray]:
    end_fit = int(np.floor(0.60 * n_events))
    end_validation = int(np.floor(0.80 * n_events))
    if end_fit <= 0 or end_validation <= end_fit or end_validation >= n_events:
        raise ValueError("development 60/20/20 partition is empty")
    return {
        "fit60": np.arange(0, end_fit, dtype=np.int64),
        "validation20": np.arange(end_fit, end_validation, dtype=np.int64),
        "confirmation20": np.arange(end_validation, n_events, dtype=np.int64),
    }


def _tensor_events(
    groups: np.ndarray, counts: np.ndarray, device: torch.device
) -> tuple[Tensor, Tensor]:
    return (
        torch.as_tensor(groups, dtype=torch.long, device=device),
        torch.as_tensor(counts, dtype=torch.long, device=device),
    )


def _batch_auxiliary_first_arrival_nll(
    *,
    model: SymmetricAxisPropagationStateRNN,
    state: Tensor,
    eligible: Tensor,
    groups: Tensor,
    current_step: int,
    horizon: int,
    horizon_limits: Tensor | None = None,
) -> Tensor:
    """Per-prefix eligible-contact-normalized mean-field first-arrival NLL."""
    batch_size, n_contacts = groups.shape
    dtype = state.dtype
    operator = model.operator_components()["W"]
    not_arrived = eligible.to(dtype)
    survival = torch.ones(batch_size, device=state.device, dtype=dtype)
    q_steps: list[Tensor] = []
    rollout_state = state
    base_seen = n_contacts - eligible.sum(dim=1).to(dtype)
    if horizon_limits is None:
        horizon_limits = torch.full(
            (batch_size,), horizon, device=state.device, dtype=torch.long
        )

    for future_index in range(1, horizon + 1):
        remaining_weight = not_arrived.sum(dim=1)
        weighted_mean = (not_arrived * rollout_state).sum(dim=1) / torch.clamp(
            remaining_weight, min=model.eps
        )
        expected_seen = (
            base_seen + (eligible.to(dtype) - not_arrived).sum(dim=1)
        ) / float(n_contacts)
        stop_logit = (
            model.c0 + model.c_p * weighted_mean + model.c_n * expected_seen
        )
        p_stop = torch.sigmoid(stop_logit)
        hazard = torch.sigmoid(model.node_bias[None, :] + rollout_state)
        hazard = hazard * eligible.to(dtype)
        log_empty = torch.where(
            eligible,
            torch.log1p(-torch.clamp(hazard, max=1.0 - model.eps)),
            torch.zeros_like(hazard),
        ).sum(dim=1)
        z = -torch.expm1(log_empty)
        active_horizon = future_index <= horizon_limits
        forced_stop = (remaining_weight <= model.eps) | (z <= model.eps)
        conditional_hazard = torch.where(
            (forced_stop | ~active_horizon)[:, None],
            torch.zeros_like(hazard),
            hazard / torch.clamp(z[:, None], min=model.eps),
        )
        p_stop = torch.where(
            forced_stop,
            torch.ones_like(p_stop),
            p_stop,
        )
        activation = not_arrived * conditional_hazard
        q = survival[:, None] * (1.0 - p_stop[:, None]) * activation
        q = q * active_horizon[:, None].to(dtype)
        q_steps.append(q)
        survival = torch.where(
            active_horizon,
            survival * (1.0 - p_stop),
            survival,
        )
        not_arrived = torch.where(
            active_horizon[:, None],
            not_arrived * (1.0 - conditional_hazard),
            not_arrived,
        )
        rollout_state = torch.where(
            active_horizon[:, None],
            model.rho_p * rollout_state + activation @ operator.T,
            rollout_state,
        )

    q_stack = torch.stack(q_steps, dim=1)
    q_sum = q_stack.sum(dim=1)
    offset = groups - int(current_step)
    target_arrives = (
        eligible
        & (offset >= 1)
        & (offset <= horizon_limits[:, None])
        & (offset <= horizon)
    )
    gather_index = torch.clamp(offset - 1, min=0, max=horizon - 1)
    arrival_probability = torch.gather(
        q_stack.permute(0, 2, 1), 2, gather_index[..., None]
    ).squeeze(-1)
    class_probability = torch.where(
        target_arrives,
        arrival_probability,
        1.0 - q_sum,
    )
    contact_nll = -torch.log(torch.clamp(class_probability, min=model.eps))
    eligible_count = torch.clamp(eligible.sum(dim=1), min=1)
    return (contact_nll * eligible.to(dtype)).sum(dim=1) / eligible_count


def batch_event_losses(
    *,
    model: SymmetricAxisPropagationStateRNN,
    groups: Tensor,
    counts: Tensor,
    training_horizon: int,
    evaluate_full_future: bool,
) -> dict[str, Tensor]:
    """Return event-first next and future losses for one event batch."""
    batch_size, n_contacts = groups.shape
    max_steps = int(torch.max(counts).item())
    dtype = model.node_bias.dtype
    operator = model.operator_components()["W"]
    state = torch.zeros(
        (batch_size, n_contacts), device=groups.device, dtype=dtype
    )
    next_sum = torch.zeros(batch_size, device=groups.device, dtype=dtype)
    future_sum = torch.zeros_like(next_sum)
    decision_count = torch.zeros_like(next_sum)
    future_count = torch.zeros_like(next_sum)
    rollout_horizon = n_contacts if evaluate_full_future else training_horizon

    for step in range(max_steps):
        active = counts > step
        current = (groups == step).to(dtype)
        state = model.rho_p * state + current @ operator.T
        seen = (groups >= 0) & (groups <= step)
        eligible = ~seen
        eligible_count = eligible.sum(dim=1)
        mean_drive = (state * eligible.to(dtype)).sum(dim=1) / torch.clamp(
            eligible_count, min=1
        )
        seen_fraction = seen.to(dtype).mean(dim=1)
        stop_logit = model.c0 + model.c_p * mean_drive + model.c_n * seen_fraction
        stop_logit = torch.where(
            eligible_count > 0,
            stop_logit,
            torch.full_like(stop_logit, torch.inf),
        )

        logits = model.node_bias[None, :] + state
        target = groups == (step + 1)
        terminal = counts == (step + 1)
        log_hazard = F.logsigmoid(logits)
        log_one_minus = F.logsigmoid(-logits)
        target_float = target.to(dtype)
        eligible_float = eligible.to(dtype)
        bernoulli = (
            target_float * log_hazard
            + (eligible_float - target_float) * log_one_minus
        ).sum(dim=1)
        log_empty = (eligible_float * log_one_minus).sum(dim=1)
        log_empty = torch.clamp(
            log_empty, max=-torch.finfo(log_empty.dtype).eps
        )
        log_z = torch.log(-torch.expm1(log_empty))
        log_probability = torch.where(
            terminal,
            F.logsigmoid(stop_logit),
            F.logsigmoid(-stop_logit) + bernoulli - log_z,
        )
        normalized_nll = -log_probability / torch.clamp(eligible_count, min=1)
        next_sum = next_sum + torch.where(
            active, normalized_nll, torch.zeros_like(normalized_nll)
        )
        decision_count = decision_count + active.to(dtype)

        if rollout_horizon > 0:
            horizon_limits = (
                eligible_count
                if evaluate_full_future
                else torch.full_like(eligible_count, rollout_horizon)
            )
            future_nll = _batch_auxiliary_first_arrival_nll(
                model=model,
                state=state,
                eligible=eligible,
                groups=groups,
                current_step=step,
                horizon=rollout_horizon,
                horizon_limits=horizon_limits,
            )
            future_sum = future_sum + torch.where(
                active, future_nll, torch.zeros_like(future_nll)
            )
            future_count = future_count + active.to(dtype)

    event_next = next_sum / torch.clamp(decision_count, min=1.0)
    event_future = torch.where(
        future_count > 0,
        future_sum / torch.clamp(future_count, min=1.0),
        torch.full_like(future_sum, torch.nan),
    )
    objective = event_next
    if training_horizon > 0 and not evaluate_full_future:
        objective = objective + event_future
    return {
        "event_next_nll": event_next,
        "event_future_nll": event_future,
        "event_objective": objective,
    }


@torch.no_grad()
def evaluate_partition(
    *,
    model: SymmetricAxisPropagationStateRNN,
    groups: Tensor,
    counts: Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    next_values: list[np.ndarray] = []
    future_values: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch = torch.as_tensor(
            indices[start : start + batch_size],
            device=groups.device,
            dtype=torch.long,
        )
        losses = batch_event_losses(
            model=model,
            groups=groups[batch],
            counts=counts[batch],
            training_horizon=0,
            evaluate_full_future=True,
        )
        next_values.append(losses["event_next_nll"].cpu().numpy())
        future_values.append(losses["event_future_nll"].cpu().numpy())
    next_array = np.concatenate(next_values)
    future_array = np.concatenate(future_values)
    return {
        "n_events": int(len(indices)),
        "next_nll": float(np.mean(next_array)),
        "future_nll": float(np.mean(future_array)),
        "next_nll_median_event": float(np.median(next_array)),
        "future_nll_median_event": float(np.median(future_array)),
        "finite": bool(
            np.all(np.isfinite(next_array)) and np.all(np.isfinite(future_array))
        ),
    }


def fit_variant(
    *,
    variant: str,
    coords: np.ndarray,
    node_bias: np.ndarray,
    groups: Tensor,
    counts: Tensor,
    partitions: dict[str, np.ndarray],
    training_horizon: int,
    seed: int,
    optimizer_config: dict[str, Any],
    batch_size: int,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
    log_path: Path,
) -> tuple[SymmetricAxisPropagationStateRNN, dict[str, Any]]:
    set_determinism(seed)
    model = SymmetricAxisPropagationStateRNN(
        coords=coords,
        node_bias=node_bias,
        isotropic=variant == "local_isotropic",
    ).to(groups.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    best_validation = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    generator = np.random.default_rng(seed)
    clip = float(optimizer_config["gradient_clip"])
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    for epoch in range(max_epochs):
        model.train()
        order = generator.permutation(partitions["fit60"])
        training_event_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_numpy = order[start : start + batch_size]
            batch = torch.as_tensor(
                batch_numpy, device=groups.device, dtype=torch.long
            )
            optimizer.zero_grad(set_to_none=True)
            losses = batch_event_losses(
                model=model,
                groups=groups[batch],
                counts=counts[batch],
                training_horizon=training_horizon,
                evaluate_full_future=False,
            )
            loss = losses["event_objective"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant}: non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
            optimizer.step()
            training_event_losses.append(float(loss.detach().cpu()))

        validation = evaluate_partition(
            model=model,
            groups=groups,
            counts=counts,
            indices=partitions["validation20"],
            batch_size=max(batch_size, 1024),
        )
        score = validation["next_nll"]
        if training_horizon > 0:
            score += validation["future_nll"]
        improved = score < best_validation - 1.0e-7
        if improved:
            best_validation = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_objective": score,
                    "variant": variant,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "variant": variant,
            "training_objective": float(np.mean(training_event_losses)),
            "validation_objective": float(score),
            "validation_next_nll": validation["next_nll"],
            "validation_future_nll": validation["future_nll"],
            "gradient_norm_last_batch": float(gradient_norm.detach().cpu()),
            "best_epoch": best_epoch,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if epochs_without_improvement >= patience:
            break

    best = torch.load(checkpoint_path, map_location=groups.device, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.eval()
    metrics = {
        partition: evaluate_partition(
            model=model,
            groups=groups,
            counts=counts,
            indices=indices,
            batch_size=max(batch_size, 1024),
        )
        for partition, indices in partitions.items()
    }
    axis = canonicalize_axis(model.axis.detach().cpu().numpy())
    summary = {
        "variant": variant,
        "seed": seed,
        "training_horizon": training_horizon,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epoch + 1),
        "early_stopped": bool(epoch + 1 < max_epochs),
        "metrics": metrics,
        "parameters": {
            "axis": axis.tolist(),
            "gamma": float(model.gamma.detach().cpu()),
            "gain": float(model.gain.detach().cpu()),
            "anisotropy_ratio": float(model.anisotropy_ratio.detach().cpu()),
            "rho_p": float(model.rho_p.detach().cpu()),
            "c0": float(model.c0.detach().cpu()),
            "c_p": float(model.c_p.detach().cpu()),
            "c_n": float(model.c_n.detach().cpu()),
        },
        "runtime_seconds": float(time.time() - started),
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--objective", choices=sorted(OBJECTIVE_HORIZON), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-events", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    development = list(map(str, cfg["cohort"]["development"]))
    if args.subject not in development:
        raise SystemExit(f"{args.subject} is not in the frozen development cohort")
    if args.seed not in list(map(int, cfg["optimizer"]["seeds"])):
        raise SystemExit(f"seed {args.seed} is outside the frozen seed set")

    device_name = args.device or cfg["resources"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    set_determinism(args.seed)
    dataset_root = ROOT / cfg["inputs"]["rank_dataset"]
    npz_path = dataset_root / "per_subject" / f"{args.subject}.npz"
    data = _events_to_numpy(npz_path)
    partitions = _partition_indices(len(data["groups"]))
    if args.smoke:
        limit = int(args.smoke_events)
        partitions = {
            name: indices[: min(limit, len(indices))]
            for name, indices in partitions.items()
        }
    bias_result = estimate_node_hazard_bias(
        data["groups"][partitions["fit60"]]
    )
    bias_fingerprint = node_bias_fingerprint(bias_result["bias"])
    groups, counts = _tensor_events(data["groups"], data["counts"], device)

    run_kind = "smoke" if args.smoke else "runs"
    run_root = (
        ROOT
        / cfg["outputs"]["root"]
        / "development"
        / run_kind
        / args.subject
        / args.objective
        / f"seed_{args.seed}"
    )
    complete_path = run_root / "COMPLETE"
    if complete_path.exists() and not args.force:
        print(f"already complete: {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_root / "run_state.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "subject": args.subject,
            "objective": args.objective,
            "seed": args.seed,
            "smoke": args.smoke,
            "ictal_target_values_read": False,
            "started_unix": time.time(),
        },
    )
    resolved = {
        "subject": args.subject,
        "objective": args.objective,
        "training_horizon": OBJECTIVE_HORIZON[args.objective],
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "partitions": {key: len(value) for key, value in partitions.items()},
        "optimizer": cfg["optimizer"],
        "input_npz": str(npz_path.relative_to(ROOT)),
        "input_sha256": sha256(npz_path),
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "core_sha256": sha256(
            ROOT / "src/topic5_symmetric_axis_propagation_state_v2_2.py"
        ),
        "git_commit": git_commit(),
        "node_bias_sha256": bias_fingerprint,
        "target_values_read": False,
    }
    atomic_json(run_root / "resolved_config.json", resolved)

    max_epochs = 3 if args.smoke else int(cfg["optimizer"]["max_epochs"])
    patience = 2 if args.smoke else int(cfg["optimizer"]["patience"])
    summaries: dict[str, Any] = {}
    try:
        for variant in ("full", "local_isotropic"):
            _, summaries[variant] = fit_variant(
                variant=variant,
                coords=data["coords"],
                node_bias=bias_result["bias"],
                groups=groups,
                counts=counts,
                partitions=partitions,
                training_horizon=OBJECTIVE_HORIZON[args.objective],
                seed=args.seed,
                optimizer_config=cfg["optimizer"],
                batch_size=args.batch_size,
                max_epochs=max_epochs,
                patience=patience,
                checkpoint_path=run_root / f"{variant}_best.pt",
                log_path=run_root / f"{variant}_epochs.jsonl",
            )
        full_validation = summaries["full"]["metrics"]["validation20"]
        isotropic_validation = summaries["local_isotropic"]["metrics"]["validation20"]
        full_confirmation = summaries["full"]["metrics"]["confirmation20"]
        isotropic_confirmation = summaries["local_isotropic"]["metrics"]["confirmation20"]
        result = {
            "contract": cfg["contract"]["name"],
            "version": cfg["contract"]["version"],
            "status": "complete",
            "subject": args.subject,
            "objective": args.objective,
            "seed": args.seed,
            "smoke": args.smoke,
            "node_bias_sha256": bias_fingerprint,
            "full_control_bias_identical": True,
            "models": summaries,
            "comparison": {
                "validation_future_benefit": (
                    isotropic_validation["future_nll"]
                    - full_validation["future_nll"]
                ),
                "validation_next_benefit": (
                    isotropic_validation["next_nll"]
                    - full_validation["next_nll"]
                ),
                "confirmation_future_benefit": (
                    isotropic_confirmation["future_nll"]
                    - full_confirmation["future_nll"]
                ),
                "confirmation_next_benefit": (
                    isotropic_confirmation["next_nll"]
                    - full_confirmation["next_nll"]
                ),
            },
            "resource": {
                "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.0**2),
                "peak_cuda_allocated_gb": (
                    torch.cuda.max_memory_allocated(device) / (1024.0**3)
                    if device.type == "cuda"
                    else 0.0
                ),
                "peak_cuda_reserved_gb": (
                    torch.cuda.max_memory_reserved(device) / (1024.0**3)
                    if device.type == "cuda"
                    else 0.0
                ),
            },
            "ictal_target_values_read": False,
        }
        atomic_json(run_root / "metrics.json", result)
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "COMPLETE",
                "pid": os.getpid(),
                "finished_unix": time.time(),
                "metrics": "metrics.json",
                "ictal_target_values_read": False,
            },
        )
        complete_path.write_text("COMPLETE\n", encoding="utf-8")
        print(json.dumps(result["comparison"], indent=2))
        print(json.dumps(result["resource"], indent=2))
    except Exception as exc:
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "FAILED",
                "pid": os.getpid(),
                "finished_unix": time.time(),
                "error": repr(exc),
                "ictal_target_values_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
