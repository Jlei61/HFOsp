#!/usr/bin/env python3
"""Train one target-sealed LOSO fold of the axis-structured graph RNN."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from scripts.audit_topic5_structured_axis_prior import (  # noqa: E402
    _directed_axis_graph,
)
from scripts.build_topic5_transition_skeleton_prior import (  # noqa: E402
    _blend_graph,
)
from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    BalancedSubjectSampler,
    EventQueue,
    SubjectRecord,
    _jsonable,
    _seed_everything,
    load_records,
)
from src.topic5_axis_graph_rnn import (  # noqa: E402
    AxisStructuredGraphRNN,
    structured_next_set_stop_loss,
)
from src.topic5_rank_distribution import distribution_errors  # noqa: E402


@dataclass(frozen=True)
class AxisPrior:
    subject: str
    axis: np.ndarray
    forward: np.ndarray
    reverse: np.ndarray
    left: np.ndarray
    right: np.ndarray
    source_sha256: str
    control: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shaft_key(name: str) -> str:
    first = str(name).split("-")[0]
    match = re.match(r"^([^0-9]+)", first)
    return match.group(1).upper() if match else first.upper()


def _shuffled_axis_prior(
    axis: np.ndarray,
    contact_names: np.ndarray,
    *,
    seed: int,
    neighbors: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within-shaft axis permutation, with a global fallback if immobile."""
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(axis, np.float32).copy()
    changed = False
    keys = np.asarray([_shaft_key(str(name)) for name in contact_names])
    for key in np.unique(keys):
        index = np.flatnonzero(keys == key)
        if len(index) >= 2:
            original = shuffled[index].copy()
            shuffled[index] = rng.permutation(original)
            changed |= not np.array_equal(original, shuffled[index])
    if not changed:
        shuffled = rng.permutation(shuffled)
    forward = _directed_axis_graph(shuffled, neighbors=int(neighbors))
    reverse = forward.T.copy()
    denominator = reverse.sum(1, keepdims=True)
    reverse = np.divide(
        reverse,
        denominator,
        out=np.zeros_like(reverse),
        where=denominator > 0,
    )
    endpoint_k = max(1, min(3, len(shuffled) // 4))
    order = np.argsort(shuffled, kind="stable")
    left = np.zeros(len(shuffled), bool)
    right = np.zeros(len(shuffled), bool)
    left[order[:endpoint_k]] = True
    right[order[-endpoint_k:]] = True
    return shuffled, forward, reverse, left, right


def _weight_shuffled_transition_graph(
    skeleton: np.ndarray,
    axis: np.ndarray,
    *,
    seed: int,
    axis_floor: float = 0.20,
    neighbors: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle edge weights at fixed density and fixed patient axis."""
    rng = np.random.default_rng(int(seed))
    shuffled = np.zeros_like(skeleton)
    allowed = np.argwhere(
        (axis[:, None] > axis[None, :])
        & ~np.eye(len(axis), dtype=bool)
    )
    weights = skeleton[allowed[:, 0], allowed[:, 1]].copy()
    rng.shuffle(weights)
    shuffled[allowed[:, 0], allowed[:, 1]] = weights
    tied = np.isclose(axis[:, None], axis[None, :]) & ~np.eye(
        len(axis), dtype=bool
    )
    tied_values = skeleton[tied].copy()
    rng.shuffle(tied_values)
    shuffled[tied] = tied_values
    forward, reverse, _ = _blend_graph(
        shuffled,
        axis,
        axis_floor=float(axis_floor),
        neighbors=int(neighbors),
    )
    return forward, reverse


def load_axis_priors(
    prior_root: Path,
    records: Dict[str, SubjectRecord],
    *,
    control: str,
    seed: int,
) -> Dict[str, AxisPrior]:
    if control not in {"intact", "axis_shuffle", "weight_shuffle"}:
        raise ValueError(f"unknown prior control: {control}")
    priors = {}
    for subject, record in records.items():
        path = prior_root / "per_subject" / f"{subject}.npz"
        if not path.exists():
            raise RuntimeError(f"{subject}: axis prior is missing")
        with np.load(path, allow_pickle=False) as z:
            if str(z["source_event_split"]) != "chronological_train80_only":
                raise RuntimeError(f"{subject}: axis prior is not train-only")
            if bool(z["ictal_target_read"]):
                raise RuntimeError(f"{subject}: axis prior read ictal data")
            if str(z["input_record_sha256"]) != record.input_sha256:
                raise RuntimeError(f"{subject}: record fingerprint changed")
            names = np.asarray(z["contact_names"])
            if not np.array_equal(names.astype(str), record.contact_names.astype(str)):
                raise RuntimeError(f"{subject}: contact order mismatch")
            axis = np.asarray(z["axis_coordinate"], np.float32)
            forward = np.asarray(z["forward_graph"], np.float32)
            reverse = np.asarray(z["reverse_graph"], np.float32)
            left = np.asarray(z["left_endpoint"], bool)
            right = np.asarray(z["right_endpoint"], bool)
            skeleton = (
                np.asarray(z["transition_skeleton_raw"], np.float32)
                if "transition_skeleton_raw" in z.files
                else None
            )
        if control == "axis_shuffle":
            subject_seed = int(
                hashlib.sha256(f"{subject}:{seed}".encode()).hexdigest()[:8],
                16,
            )
            axis, forward, reverse, left, right = _shuffled_axis_prior(
                axis, record.contact_names, seed=subject_seed
            )
        elif control == "weight_shuffle":
            if skeleton is None:
                raise RuntimeError(
                    f"{subject}: weight shuffle requires transition skeleton"
                )
            subject_seed = int(
                hashlib.sha256(f"{subject}:{seed}".encode()).hexdigest()[:8],
                16,
            )
            forward, reverse = _weight_shuffled_transition_graph(
                skeleton, axis, seed=subject_seed
            )
        priors[subject] = AxisPrior(
            subject=subject,
            axis=axis,
            forward=forward,
            reverse=reverse,
            left=left,
            right=right,
            source_sha256=_sha256(path),
            control=control,
        )
    if len(priors) != 34:
        raise RuntimeError(f"expected 34 axis priors, found {len(priors)}")
    return priors


def _batch(
    record: SubjectRecord,
    prior: AxisPrior,
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    n_events = len(indices)
    n_contacts = record.contact_features.shape[0]
    return {
        "contact_features": torch.as_tensor(
            record.contact_features, dtype=torch.float32, device=device
        ).unsqueeze(0).expand(n_events, -1, -1),
        "contact_mask": torch.ones(
            (n_events, n_contacts), dtype=torch.bool, device=device
        ),
        "group_ids": torch.as_tensor(
            record.group_ids[indices], dtype=torch.long, device=device
        ),
        "group_count": torch.as_tensor(
            record.group_count[indices], dtype=torch.long, device=device
        ),
        "axis_coordinate": torch.as_tensor(
            prior.axis, dtype=torch.float32, device=device
        ),
        "forward_graph": torch.as_tensor(
            prior.forward, dtype=torch.float32, device=device
        ),
        "reverse_graph": torch.as_tensor(
            prior.reverse, dtype=torch.float32, device=device
        ),
        "left_endpoint": torch.as_tensor(
            prior.left, dtype=torch.bool, device=device
        ),
        "right_endpoint": torch.as_tensor(
            prior.right, dtype=torch.bool, device=device
        ),
    }


def _loss(model, batch: dict, offset: torch.Tensor, cfg: dict) -> dict:
    output = model(**batch, local_offset=offset)
    return structured_next_set_stop_loss(
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


def train_shared(
    model: AxisStructuredGraphRNN,
    records: Sequence[SubjectRecord],
    priors: Dict[str, AxisPrior],
    cfg: dict,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict, list[dict], dict]:
    model.to(device)
    model.train()
    rng = np.random.default_rng(int(seed))
    sampler = BalancedSubjectSampler(records, rng)
    queues = {
        record.subject: EventQueue(
            record.train_indices,
            np.random.default_rng(
                int(seed)
                ^ int(
                    hashlib.sha256(record.subject.encode()).hexdigest()[:8],
                    16,
                )
            ),
        )
        for record in records
    }
    offset_dim = int(cfg["model"]["local_offset_dim"])
    offsets = {
        record.subject: torch.nn.Parameter(
            torch.zeros(
                (record.contact_features.shape[0], offset_dim),
                dtype=torch.float32,
                device=device,
            )
        )
        for record in records
    }
    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": float(training["learning_rate"]),
                "weight_decay": float(training["weight_decay"]),
            },
            {
                "params": list(offsets.values()),
                "lr": float(training["local_learning_rate"]),
                "weight_decay": float(training["weight_decay"]),
            },
        ]
    )
    rows = []
    started = time.time()
    window = []
    for step in range(int(steps)):
        record = sampler.draw(step)
        index = queues[record.subject].draw(
            min(int(batch_size), len(record.train_indices))
        )
        batch = _batch(record, priors[record.subject], index, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, offsets[record.subject], cfg)
        loss["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), *offsets.values()],
            float(training["gradient_clip"]),
        )
        optimizer.step()
        value = float(loss["total"].detach().cpu())
        window.append(value)
        rows.append(
            {
                "phase": "shared",
                "step": step + 1,
                "subject": record.subject,
                "dataset": record.dataset,
                "total_loss": value,
                "next_set_stop_loss": float(
                    loss["next_set_stop"].detach().cpu()
                ),
                "stop_calibration_loss": float(
                    loss["stop_calibration"].detach().cpu()
                ),
                "endpoint_source_loss": float(
                    loss["endpoint_source"].detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "elapsed_seconds": time.time() - started,
            }
        )
        if (step + 1) % 32 == 0 or step + 1 == int(steps):
            print(
                json.dumps(
                    {
                        "phase": "shared",
                        "step": step + 1,
                        "steps": int(steps),
                        "loss_window": float(np.mean(window)),
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                ),
                flush=True,
            )
            window.clear()
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    coverage = {
        record.subject: {
            "events_available": int(len(record.train_indices)),
            "events_drawn": int(queues[record.subject].drawn),
            "completed_cycles": int(queues[record.subject].cycles),
        }
        for record in records
    }
    return state, rows, coverage


def calibrate_heldout_offset(
    model: AxisStructuredGraphRNN,
    record: SubjectRecord,
    prior: AxisPrior,
    cfg: dict,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, list[dict], dict]:
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    offset = torch.nn.Parameter(
        torch.zeros(
            (
                record.contact_features.shape[0],
                int(cfg["model"]["local_offset_dim"]),
            ),
            dtype=torch.float32,
            device=device,
        )
    )
    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        [offset],
        lr=float(training["local_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(int(seed))
    queue = EventQueue(record.train_indices, rng)
    rows = []
    started = time.time()
    for step in range(int(steps)):
        index = queue.draw(min(int(batch_size), len(record.train_indices)))
        batch = _batch(record, prior, index, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, offset, cfg)
        loss["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [offset], float(training["gradient_clip"])
        )
        optimizer.step()
        rows.append(
            {
                "phase": "heldout_calibration",
                "step": step + 1,
                "subject": record.subject,
                "dataset": record.dataset,
                "total_loss": float(loss["total"].detach().cpu()),
                "next_set_stop_loss": float(
                    loss["next_set_stop"].detach().cpu()
                ),
                "stop_calibration_loss": float(
                    loss["stop_calibration"].detach().cpu()
                ),
                "endpoint_source_loss": float(
                    loss["endpoint_source"].detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "elapsed_seconds": time.time() - started,
            }
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    coverage = {
        "events_available": int(len(record.train_indices)),
        "events_drawn": int(queue.drawn),
        "completed_cycles": int(queue.cycles),
    }
    return offset.detach(), rows, coverage


@torch.no_grad()
def evaluate_teacher_forced(
    model: AxisStructuredGraphRNN,
    record: SubjectRecord,
    prior: AxisPrior,
    offset: torch.Tensor,
    cfg: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    stop_probability = []
    stop_target = []
    top1 = []
    for start in range(0, len(record.eval_indices), int(batch_size)):
        index = record.eval_indices[start : start + int(batch_size)]
        batch = _batch(record, prior, index, device)
        output = model(**batch, local_offset=offset)
        loss = structured_next_set_stop_loss(
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
        event_nll = loss["event_nll"].cpu().numpy()
        contact_logits = output["contact_logits"].cpu().numpy()
        stop_logits = output["stop_logits"].cpu().numpy()
        groups = record.group_ids[index]
        counts = record.group_count[index]
        for local, event_index in enumerate(index):
            rows.append(
                {
                    "subject": record.subject,
                    "event_index": int(event_index),
                    "event_source_index": int(
                        record.event_source_index[event_index]
                    ),
                    "event_nll": float(event_nll[local]),
                }
            )
            for step in range(int(counts[local]) + 1):
                probability = float(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(stop_logits[local, step], -60.0, 60.0)
                        )
                    )
                )
                terminal = step == int(counts[local])
                stop_probability.append(probability)
                stop_target.append(float(terminal))
                if not terminal:
                    choice = int(np.argmax(contact_logits[local, step]))
                    top1.append(
                        float(
                            stop_logits[local, step]
                            < contact_logits[local, step, choice]
                            and groups[local, choice] == step
                        )
                    )
    frame = pd.DataFrame(rows)
    probability = np.asarray(stop_probability)
    target = np.asarray(stop_target)
    return {
        "heldout_event_nll": float(frame.event_nll.mean()),
        "top1_next_set_accuracy": float(np.mean(top1)),
        "stop_brier": float(np.mean((probability - target) ** 2)),
        "terminal_stop_probability": float(
            np.mean(probability[target == 1])
        ),
        "nonterminal_stop_probability": float(
            np.mean(probability[target == 0])
        ),
        "n_eval_events": int(len(frame)),
    }, frame


def _event_feature_matrix(
    group_ids: np.ndarray, group_count: np.ndarray
) -> np.ndarray:
    participating = np.asarray(group_ids) >= 0
    denominator = np.maximum(np.asarray(group_count) - 1, 1)[:, None]
    normalized = np.where(participating, group_ids / denominator, 0.0)
    return np.concatenate(
        [participating.astype(np.float32), normalized.astype(np.float32)],
        axis=1,
    )


def _projected_quantiles(
    values: np.ndarray,
    directions: np.ndarray,
    *,
    rng: np.random.Generator,
    max_events: int,
    n_quantiles: int,
) -> np.ndarray:
    if len(values) > int(max_events):
        values = values[
            rng.choice(len(values), int(max_events), replace=False)
        ]
    return np.quantile(
        values @ directions,
        np.linspace(0.0, 1.0, int(n_quantiles)),
        axis=0,
    )


def _whole_path_distance(
    generated_groups: np.ndarray,
    generated_count: np.ndarray,
    observed_groups: np.ndarray,
    observed_count: np.ndarray,
    empirical_groups: np.ndarray,
    empirical_count: np.ndarray,
    cfg: dict,
    *,
    seed: int,
) -> dict:
    evaluation = cfg["evaluation"]
    generated = _event_feature_matrix(generated_groups, generated_count)
    observed = _event_feature_matrix(observed_groups, observed_count)
    empirical = _event_feature_matrix(empirical_groups, empirical_count)
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(
        size=(observed.shape[1], int(evaluation["random_path_projections"]))
    )
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    kwargs = {
        "max_events": int(evaluation["path_max_events"]),
        "n_quantiles": int(evaluation["path_quantiles"]),
    }
    observed_q = _projected_quantiles(
        observed, directions, rng=rng, **kwargs
    )
    generated_q = _projected_quantiles(
        generated, directions, rng=rng, **kwargs
    )
    empirical_q = _projected_quantiles(
        empirical, directions, rng=rng, **kwargs
    )
    split = max(1, len(empirical) // 2)
    first_q = _projected_quantiles(
        empirical[:split], directions, rng=rng, **kwargs
    )
    second_q = _projected_quantiles(
        empirical[split:], directions, rng=rng, **kwargs
    )
    return {
        "path_sliced_wasserstein": float(
            np.mean(np.abs(generated_q - observed_q))
        ),
        "path_empirical_distance": float(
            np.mean(np.abs(empirical_q - observed_q))
        ),
        "path_split_half_distance": float(
            np.mean(np.abs(first_q - second_q))
        ),
    }


def _axis_path_rho(
    group_ids: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    values = []
    for event in np.asarray(group_ids):
        valid = event >= 0
        if int(valid.sum()) < 3:
            continue
        statistic = spearmanr(axis[valid], event[valid]).statistic
        if np.isfinite(statistic):
            values.append(float(statistic))
    return np.asarray(values)


def _axis_path_metrics(
    generated_groups: np.ndarray,
    observed_groups: np.ndarray,
    axis: np.ndarray,
) -> dict:
    generated = _axis_path_rho(generated_groups, axis)
    observed = _axis_path_rho(observed_groups, axis)

    def summary(values: np.ndarray, prefix: str) -> dict:
        nonzero = values[np.abs(values) > 1e-12]
        balance = (
            float(min(np.mean(nonzero > 0), np.mean(nonzero < 0)))
            if len(nonzero)
            else 0.0
        )
        return {
            f"{prefix}_axis_rho_median": (
                float(np.median(values)) if len(values) else np.nan
            ),
            f"{prefix}_axis_rho_abs_median": (
                float(np.median(np.abs(values))) if len(values) else np.nan
            ),
            f"{prefix}_axis_direction_balance": balance,
            f"{prefix}_axis_rho_n": int(len(values)),
        }

    distance = (
        float(wasserstein_distance(generated, observed))
        if len(generated) and len(observed)
        else np.nan
    )
    return {
        **summary(generated, "generated"),
        **summary(observed, "observed"),
        "axis_rho_wasserstein": distance,
    }


def _rollout(
    model: AxisStructuredGraphRNN,
    record: SubjectRecord,
    prior: AxisPrior,
    offset: torch.Tensor,
    *,
    device: torch.device,
    n_events: int,
    seed: int,
    lesion: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.ones(
        (1, record.contact_features.shape[0]),
        dtype=torch.bool,
        device=device,
    )
    return model.rollout(
        features,
        mask,
        offset,
        torch.as_tensor(prior.axis, dtype=torch.float32, device=device),
        torch.as_tensor(prior.forward, dtype=torch.float32, device=device),
        torch.as_tensor(prior.reverse, dtype=torch.float32, device=device),
        torch.as_tensor(prior.left, dtype=torch.bool, device=device),
        torch.as_tensor(prior.right, dtype=torch.bool, device=device),
        n_events=int(n_events),
        seed=int(seed),
        lesion=lesion,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_axis_graph_rnn_v0_6.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument(
        "--structured-rank", type=int, choices=range(5), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--prior-control",
        choices=("intact", "axis_shuffle", "weight_shuffle"),
        default="intact",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shared-steps", type=int, default=None)
    parser.add_argument("--calibration-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Evaluate only the intact rollout; defer lesion diagnostics.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset_root = ROOT / cfg["inputs"]["dataset"]
    prior_root = ROOT / cfg["inputs"]["axis_prior"]
    records = load_records(dataset_root)
    if args.heldout_subject not in records:
        raise RuntimeError(f"heldout subject absent: {args.heldout_subject}")
    priors = load_axis_priors(
        prior_root,
        records,
        control=args.prior_control,
        seed=int(args.seed),
    )
    heldout = records[args.heldout_subject]
    outer = [
        record
        for subject, record in records.items()
        if subject != heldout.subject
    ]
    _seed_everything(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_device(index)
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["resources"]["gpu_memory_fraction_per_process"]),
            device=index,
        )
        torch.cuda.reset_peak_memory_stats(index)
    torch.set_num_threads(int(cfg["resources"]["cpu_threads_per_process"]))
    shared_steps = int(
        args.shared_steps
        if args.shared_steps is not None
        else cfg["training"]["pilot_shared_steps"]
    )
    calibration_steps = int(
        args.calibration_steps
        if args.calibration_steps is not None
        else cfg["training"]["pilot_calibration_steps"]
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else cfg["training"]["batch_events"]
    )
    n_rollouts = int(
        args.rollouts
        if args.rollouts is not None
        else cfg["evaluation"]["pilot_rollouts"]
    )
    state_path = run_dir / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "subject": heldout.subject,
                "structured_rank": int(args.structured_rank),
                "prior_control": args.prior_control,
                "seed": int(args.seed),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    started = time.time()
    model = AxisStructuredGraphRNN(
        heldout.contact_features.shape[1],
        structured_rank=int(args.structured_rank),
        local_offset_dim=int(cfg["model"]["local_offset_dim"]),
    )
    model_state, shared_log, shared_coverage = train_shared(
        model,
        outer,
        priors,
        cfg,
        steps=shared_steps,
        batch_size=batch_size,
        device=device,
        seed=int(args.seed) + int(args.structured_rank) * 1_000_003,
    )
    model.load_state_dict(model_state)
    offset, calibration_log, calibration_coverage = calibrate_heldout_offset(
        model,
        heldout,
        priors[heldout.subject],
        cfg,
        steps=calibration_steps,
        batch_size=batch_size,
        device=device,
        seed=int(args.seed) + 500_000,
    )
    teacher_metrics, event_frame = evaluate_teacher_forced(
        model,
        heldout,
        priors[heldout.subject],
        offset.to(device),
        cfg,
        batch_size=batch_size,
        device=device,
    )
    observed_groups = heldout.group_ids[heldout.eval_indices]
    observed_count = heldout.group_count[heldout.eval_indices]
    empirical_groups = heldout.group_ids[heldout.train_indices]
    empirical_count = heldout.group_count[heldout.train_indices]
    lesions = ["none"]
    if not args.primary_only:
        lesions.extend(["endpoints", "inhibition"])
        if int(args.structured_rank) >= 2:
            lesions.extend(["direction_forward", "direction_reverse"])
    metric_rows = []
    primary_groups = None
    primary_count = None
    for lesion_index, lesion in enumerate(lesions):
        groups, count = _rollout(
            model,
            heldout,
            priors[heldout.subject],
            offset.to(device),
            device=device,
            n_events=n_rollouts,
            seed=(
                int(args.seed)
                + 700_000
                + int(args.structured_rank) * 1_000_003
            ),
            lesion=lesion,
        )
        if lesion == "none":
            primary_groups, primary_count = groups, count
        distribution = distribution_errors(
            groups,
            count,
            observed_groups,
            observed_count,
            bins=int(cfg["evaluation"]["rank_distribution_bins"]),
        )
        path = _whole_path_distance(
            groups,
            count,
            observed_groups,
            observed_count,
            empirical_groups,
            empirical_count,
            cfg,
            seed=int(args.seed),
        )
        axis_path = _axis_path_metrics(
            groups, observed_groups, priors[heldout.subject].axis
        )
        metric_rows.append(
            {
                "subject": heldout.subject,
                "dataset": heldout.dataset,
                "structured_rank": int(args.structured_rank),
                "prior_control": args.prior_control,
                "lesion": lesion,
                "seed": int(args.seed),
                "n_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "rollout_participant_count_mean": float(
                    np.mean(np.sum(groups >= 0, axis=1))
                ),
                "rollout_rank_set_count_mean": float(np.mean(count)),
                "rollout_zero_length_fraction": float(np.mean(count == 0)),
                **teacher_metrics,
                **distribution,
                **path,
                **axis_path,
            }
        )
    split = max(1, len(empirical_groups) // 2)
    empirical_error = distribution_errors(
        empirical_groups,
        empirical_count,
        observed_groups,
        observed_count,
        bins=int(cfg["evaluation"]["rank_distribution_bins"]),
    )
    split_half_error = distribution_errors(
        empirical_groups[:split],
        empirical_count[:split],
        empirical_groups[split:],
        empirical_count[split:],
        bins=int(cfg["evaluation"]["rank_distribution_bins"]),
    )
    pd.DataFrame(metric_rows).to_csv(
        run_dir / "heldout_metrics.csv", index=False
    )
    event_frame.to_csv(run_dir / "heldout_event_nll.csv", index=False)
    pd.DataFrame(shared_log + calibration_log).to_csv(
        run_dir / "training_log.csv", index=False
    )
    np.savez_compressed(
        run_dir / "free_rollouts.npz",
        event_group_ids=primary_groups,
        event_group_count=primary_count,
        contact_names=heldout.contact_names,
        axis_coordinate=priors[heldout.subject].axis,
        left_endpoint=priors[heldout.subject].left,
        right_endpoint=priors[heldout.subject].right,
        ictal_target_read=np.asarray(False),
    )
    torch.save(
        {
            "contract": cfg["contract"],
            "model_state": model_state,
            "heldout_local_offset": offset.cpu(),
            "subject": heldout.subject,
            "structured_rank": int(args.structured_rank),
            "prior_control": args.prior_control,
            "seed": int(args.seed),
            "ictal_target_read": False,
        },
        run_dir / "checkpoint.pt",
    )
    summary = {
        "status": "COMPLETE",
        "contract": cfg["contract"],
        "subject": heldout.subject,
        "dataset": heldout.dataset,
        "structured_rank": int(args.structured_rank),
        "prior_control": args.prior_control,
        "seed": int(args.seed),
        "n_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "shared_steps": shared_steps,
        "calibration_steps": calibration_steps,
        "rollouts": n_rollouts,
        "elapsed_seconds": time.time() - started,
        "shared_coverage": shared_coverage,
        "calibration_coverage": calibration_coverage,
        "empirical_distribution_errors": empirical_error,
        "split_half_distribution_errors": split_half_error,
        "input_fingerprints": {
            subject: {
                "record_sha256": record.input_sha256,
                "axis_prior_sha256": priors[subject].source_sha256,
            }
            for subject, record in records.items()
        },
        "dataset_root": str(dataset_root),
        "prior_root": str(prior_root),
        "ictal_target_read": False,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True)
    )
    state_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "subject": heldout.subject,
                "structured_rank": int(args.structured_rank),
                "prior_control": args.prior_control,
                "seed": int(args.seed),
                "ictal_target_read": False,
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "subject": heldout.subject,
                "structured_rank": int(args.structured_rank),
                "prior_control": args.prior_control,
                "elapsed_seconds": round(summary["elapsed_seconds"], 2),
                "ictal_target_read": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
