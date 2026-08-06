#!/usr/bin/env python3
"""Train one v0.4 outer-patient fold and its three fixed target seeds."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from src.topic5_history_rnn import TimeDecayHistoryGRU  # noqa: E402
from src.topic5_static_anchored_history_residual import (  # noqa: E402
    DualCandidateResidualHead,
    TimeAwareNonrecurrentResidual,
    patient_balanced_soft_maxab,
    unit_eps,
)


@dataclass
class Example:
    subject: str
    seizure_id: str
    seizure_idx: int
    event_embedding: np.ndarray
    event_time: np.ndarray
    cutoff_time: float
    time_summary: np.ndarray
    contact_embedding: np.ndarray
    static_a: np.ndarray
    static_b: np.ndarray
    target_1_45: np.ndarray
    target_rank_1_45: np.ndarray
    target_1_150: np.ndarray
    contact_names: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _load_examples(cache: Path) -> tuple[list[Example], dict]:
    index = json.loads((cache / "INDEX.json").read_text())
    hashes = {entry["encoder_checkpoint_sha256"] for entry in index["entries"]}
    expected = index["outer_fold_shared_encoder"]["event_checkpoint_sha256"]
    if hashes != {expected}:
        raise RuntimeError("outer-fold cache mixes encoder coordinates")
    examples = []
    for entry in index["entries"]:
        with np.load(cache / entry["cache_file"], allow_pickle=False) as data:
            examples.append(
                Example(
                    subject=str(entry["subject"]),
                    seizure_id=str(entry["seizure_id"]),
                    seizure_idx=int(entry["seizure_idx"]),
                    event_embedding=np.asarray(data["event_embedding"], np.float32),
                    event_time=np.asarray(data["event_time"], np.float64),
                    cutoff_time=float(data["cutoff_time"]),
                    time_summary=np.asarray(data["time_summary"], np.float32),
                    contact_embedding=np.asarray(data["contact_embedding"], np.float32),
                    static_a=np.asarray(data["static_a"], np.float32),
                    static_b=np.asarray(data["static_b"], np.float32),
                    target_1_45=np.asarray(data["target_1_45"], np.float32),
                    target_rank_1_45=np.asarray(data["target_rank_1_45"], np.float32),
                    target_1_150=np.asarray(data["target_1_150"], np.float32),
                    contact_names=np.asarray(data["contact_names"]).astype(str),
                )
            )
    return examples, index


def _load_history_initialization(path: Path, device: torch.device) -> TimeDecayHistoryGRU:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if bool(payload.get("ictal_target_read", True)):
        raise RuntimeError("history initialization is not target blind")
    config = payload["config"]
    history = TimeDecayHistoryGRU(
        int(payload["event_embedding_mean"].numel()),
        int(config["history_dim"]),
        initial_half_life_hours=float(config["initial_half_life_hours"]),
    )
    state = {
        key.removeprefix("history."): value
        for key, value in payload["history_state"].items()
        if key.startswith("history.")
    }
    history.load_state_dict(state, strict=True)
    return history.to(device)


def _batch_history_states(
    history: TimeDecayHistoryGRU,
    examples: list[Example],
    *,
    device: torch.device,
    chunk_events: int,
    orders: list[np.ndarray] | None = None,
) -> torch.Tensor:
    if not examples:
        raise ValueError("empty patient history batch")
    batch = len(examples)
    maximum = max(len(example.event_time) for example in examples)
    feature_dim = examples[0].event_embedding.shape[1]
    embedding = torch.zeros((batch, maximum, feature_dim), dtype=torch.float32, device=device)
    delta = torch.zeros((batch, maximum), dtype=torch.float32, device=device)
    reset = torch.zeros((batch, maximum), dtype=torch.bool, device=device)
    mask = torch.zeros((batch, maximum), dtype=torch.bool, device=device)
    final_gap = torch.zeros(batch, dtype=torch.float32, device=device)
    for row, example in enumerate(examples):
        order = np.arange(len(example.event_time)) if orders is None else np.asarray(orders[row])
        if sorted(order.tolist()) != list(range(len(example.event_time))):
            raise RuntimeError("history order is not a full permutation")
        value = example.event_embedding[order]
        size = len(value)
        embedding[row, :size] = torch.as_tensor(value, device=device)
        local_delta = np.zeros(size, dtype=np.float32)
        if size > 1:
            local_delta[1:] = np.diff(example.event_time).astype(np.float32)
        delta[row, :size] = torch.as_tensor(local_delta, device=device)
        reset[row, 0] = True
        mask[row, :size] = True
        final_gap[row] = float(example.cutoff_time - example.event_time[-1])
    state = None
    for start in range(0, maximum, int(chunk_events)):
        stop = min(start + int(chunk_events), maximum)
        _, state = history.forward_masked(
            embedding[:, start:stop],
            delta[:, start:stop],
            reset[:, start:stop],
            mask[:, start:stop],
            initial_state=state,
        )
    return history.decay(state, final_gap)


def _example_tensors(example: Example, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "contact": torch.as_tensor(example.contact_embedding, device=device),
        "static_a": torch.as_tensor(example.static_a, device=device),
        "static_b": torch.as_tensor(example.static_b, device=device),
        "target_rank": torch.as_tensor(example.target_rank_1_45, device=device),
        "summary": torch.as_tensor(example.time_summary, device=device),
    }


def _patient_score_from_states(
    head: DualCandidateResidualHead,
    states: torch.Tensor,
    examples: list[Example],
    *,
    device: torch.device,
    rank_temperature: float,
    max_temperature: float,
) -> torch.Tensor:
    predictions = []
    for row, example in enumerate(examples):
        tensor = _example_tensors(example, device)
        output = head(
            states[row], tensor["contact"], tensor["static_a"], tensor["static_b"]
        )
        predictions.append((output["candidate_a"], output["candidate_b"], tensor["target_rank"]))
    return patient_balanced_soft_maxab(
        predictions,
        rank_temperature=rank_temperature,
        max_temperature=max_temperature,
    )


def _gain_penalty(head: DualCandidateResidualHead, weight: float) -> torch.Tensor:
    return float(weight) * head.gains.square().sum()


def _anchor_penalty(
    history: TimeDecayHistoryGRU,
    initial: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    total = None
    count = 0
    for name, parameter in history.named_parameters():
        value = (parameter - initial[name]).square().sum()
        total = value if total is None else total + value
        count += parameter.numel()
    if total is None:
        raise RuntimeError("history has no trainable parameters")
    return float(weight) * total / max(count, 1)


def _gradient_norm(parameters) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def _history_half_life(history: TimeDecayHistoryGRU) -> float:
    rate = history.decay_rate_per_second.detach().cpu().numpy()
    return float(np.median(np.log(2.0) / np.maximum(rate, 1e-12) / 3600.0))


def _patient_groups(examples: list[Example]) -> dict[str, list[Example]]:
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.subject, []).append(example)
    return {key: sorted(value, key=lambda row: row.seizure_id) for key, value in groups.items()}


def _frozen_states(
    history: TimeDecayHistoryGRU,
    groups: dict[str, list[Example]],
    *,
    device: torch.device,
    chunk_events: int,
) -> dict[str, torch.Tensor]:
    history.eval()
    for parameter in history.parameters():
        parameter.requires_grad_(False)
    output = {}
    with torch.no_grad():
        for subject, examples in groups.items():
            output[subject] = _batch_history_states(
                history, examples, device=device, chunk_events=chunk_events
            ).detach()
    return output


def _epoch_orders(subjects: list[str], seed: int, epochs: int) -> list[list[str]]:
    rows = []
    for epoch in range(int(epochs)):
        order = list(subjects)
        np.random.default_rng(int(seed) + 7000 + epoch).shuffle(order)
        rows.append(order)
    return rows


def _train_head_stage(
    head: DualCandidateResidualHead,
    optimizer: torch.optim.Optimizer,
    frozen: dict[str, torch.Tensor],
    groups: dict[str, list[Example]],
    orders: list[list[str]],
    *,
    stage: str,
    epoch_offset: int,
    config: dict,
    device: torch.device,
    started: float,
) -> list[dict]:
    head.train()
    logs = []
    for local_epoch, order in enumerate(orders, start=1):
        losses = []
        scores = []
        gradients = []
        for subject in order:
            optimizer.zero_grad(set_to_none=True)
            score = _patient_score_from_states(
                head,
                frozen[subject],
                groups[subject],
                device=device,
                rank_temperature=config["soft_rank_temperature"],
                max_temperature=config["soft_max_temperature"],
            )
            loss = 1.0 - score + _gain_penalty(head, config["lambda_gain"])
            loss.backward()
            gradients.append(_gradient_norm(head.parameters()))
            torch.nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.append(float(score.detach().cpu()))
        row = {
            "stage": stage,
            "epoch": int(epoch_offset + local_epoch),
            "loss": float(np.mean(losses)),
            "soft_maxab": float(np.mean(scores)),
            "gradient_norm": float(np.mean(gradients)),
            "gain_a": float(head.gains[0].detach().cpu()),
            "gain_b": float(head.gains[1].detach().cpu()),
            "history_half_life_hours": None,
            "gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
            "elapsed_seconds": time.time() - started,
        }
        logs.append(row)
        print(json.dumps(row), flush=True)
    return logs


def _train_joint_stage(
    head: DualCandidateResidualHead,
    history: TimeDecayHistoryGRU,
    optimizer: torch.optim.Optimizer,
    groups: dict[str, list[Example]],
    orders: list[list[str]],
    *,
    epoch_offset: int,
    initial_history: dict[str, torch.Tensor],
    config: dict,
    device: torch.device,
    started: float,
) -> list[dict]:
    head.train()
    history.train()
    for parameter in history.parameters():
        parameter.requires_grad_(True)
    parameters = list(head.parameters()) + list(history.parameters())
    logs = []
    for local_epoch, order in enumerate(orders, start=1):
        losses = []
        scores = []
        gradients = []
        for subject in order:
            optimizer.zero_grad(set_to_none=True)
            states = _batch_history_states(
                history,
                groups[subject],
                device=device,
                chunk_events=config["chunk_events"],
            )
            score = _patient_score_from_states(
                head,
                states,
                groups[subject],
                device=device,
                rank_temperature=config["soft_rank_temperature"],
                max_temperature=config["soft_max_temperature"],
            )
            loss = (
                1.0
                - score
                + _gain_penalty(head, config["lambda_gain"])
                + _anchor_penalty(history, initial_history, config["lambda_anchor_recurrent_only"])
            )
            loss.backward()
            gradients.append(_gradient_norm(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.append(float(score.detach().cpu()))
        row = {
            "stage": "M3_joint",
            "epoch": int(epoch_offset + local_epoch),
            "loss": float(np.mean(losses)),
            "soft_maxab": float(np.mean(scores)),
            "gradient_norm": float(np.mean(gradients)),
            "gain_a": float(head.gains[0].detach().cpu()),
            "gain_b": float(head.gains[1].detach().cpu()),
            "history_half_life_hours": _history_half_life(history),
            "gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
            "elapsed_seconds": time.time() - started,
        }
        logs.append(row)
        print(json.dumps(row), flush=True)
    return logs


def _train_m2(
    model: TimeAwareNonrecurrentResidual,
    optimizer: torch.optim.Optimizer,
    groups: dict[str, list[Example]],
    orders: list[list[str]],
    *,
    config: dict,
    device: torch.device,
    started: float,
) -> list[dict]:
    model.train()
    logs = []
    for epoch, order in enumerate(orders, start=1):
        losses = []
        scores = []
        gradients = []
        for subject in order:
            optimizer.zero_grad(set_to_none=True)
            predictions = []
            for example in groups[subject]:
                tensor = _example_tensors(example, device)
                output = model(
                    tensor["summary"], tensor["contact"], tensor["static_a"], tensor["static_b"]
                )
                predictions.append((output["candidate_a"], output["candidate_b"], tensor["target_rank"]))
            score = patient_balanced_soft_maxab(
                predictions,
                rank_temperature=config["soft_rank_temperature"],
                max_temperature=config["soft_max_temperature"],
            )
            loss = 1.0 - score + _gain_penalty(model.head, config["lambda_gain"])
            loss.backward()
            gradients.append(_gradient_norm(model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.append(float(score.detach().cpu()))
        row = {
            "stage": "M2_time_aware_nonrecurrent",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "soft_maxab": float(np.mean(scores)),
            "gradient_norm": float(np.mean(gradients)),
            "gain_a": float(model.head.gains[0].detach().cpu()),
            "gain_b": float(model.head.gains[1].detach().cpu()),
            "history_half_life_hours": None,
            "gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
            "elapsed_seconds": time.time() - started,
        }
        logs.append(row)
        print(json.dumps(row), flush=True)
    return logs


def _order_shuffle_permutation(example: Example, *, seed: int, draw: int) -> np.ndarray:
    """Permute event identities across the *complete* causal prefix.

    Contract v0.4 §7.3 requires the whole prefix to be permuted; v0.2 shuffled
    only the most recent 64 events, which left the shuffled arm and the true
    arm sharing almost the same history.  The permutation length is therefore
    always ``len(example.event_time)``.
    """

    rng = np.random.default_rng(
        _stable_seed(f"v0.4-order:{seed}:{example.subject}:{example.seizure_id}:{draw}")
    )
    return rng.permutation(len(example.event_time))


def _abs_spearman(candidate: np.ndarray, target: np.ndarray) -> float:
    if len(candidate) < 3 or np.std(candidate) <= 0 or np.std(target) <= 0:
        return float("nan")
    value = spearmanr(candidate, target).statistic
    return abs(float(value)) if np.isfinite(value) else float("nan")


def _prediction_rows(
    example: Example,
    model: str,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    *,
    seed: int,
    draw: int = -1,
    donor_seizure_id: str = "",
) -> tuple[list[dict], dict]:
    rows = []
    for index, contact in enumerate(example.contact_names):
        rows.append(
            {
                "subject": example.subject,
                "seizure_id": example.seizure_id,
                "seizure_idx": example.seizure_idx,
                "contact": str(contact),
                "model": model,
                "seed": int(seed),
                "draw": int(draw),
                "donor_seizure_id": donor_seizure_id,
                "prediction_a": float(candidate_a[index]),
                "prediction_b": float(candidate_b[index]),
                "target_1_45": float(example.target_1_45[index]),
                "target_1_150": float(example.target_1_150[index]),
            }
        )
    rho45_a = _abs_spearman(candidate_a, example.target_1_45)
    rho45_b = _abs_spearman(candidate_b, example.target_1_45)
    rho150_a = _abs_spearman(candidate_a, example.target_1_150)
    rho150_b = _abs_spearman(candidate_b, example.target_1_150)
    metric = {
        "subject": example.subject,
        "seizure_id": example.seizure_id,
        "model": model,
        "seed": int(seed),
        "draw": int(draw),
        "donor_seizure_id": donor_seizure_id,
        "n_contacts": len(example.contact_names),
        "maxab_1_45": float(max(rho45_a, rho45_b)),
        "maxab_1_150_no_retrain": float(max(rho150_a, rho150_b)),
    }
    return rows, metric


@torch.no_grad()
def _evaluate(
    heldout: list[Example],
    frozen_history: TimeDecayHistoryGRU,
    m1_head: DualCandidateResidualHead,
    m2: TimeAwareNonrecurrentResidual,
    m3_history: TimeDecayHistoryGRU,
    m3_head: DualCandidateResidualHead,
    *,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for model in (frozen_history, m1_head, m2, m3_history, m3_head):
        model.eval()
    frozen_states = _batch_history_states(
        frozen_history, heldout, device=device, chunk_events=config["chunk_events"]
    )
    m3_states = _batch_history_states(
        m3_history, heldout, device=device, chunk_events=config["chunk_events"]
    )
    prediction_rows = []
    metric_rows = []
    for row, example in enumerate(heldout):
        tensor = _example_tensors(example, device)
        outputs = {
            "M0_STATIC_AB": {
                "candidate_a": unit_eps(tensor["static_a"]),
                "candidate_b": unit_eps(tensor["static_b"]),
            },
            "M1_FROZEN_HISTORY_HEAD": m1_head(
                frozen_states[row], tensor["contact"], tensor["static_a"], tensor["static_b"]
            ),
            "M2_TIME_AWARE_NONRECURRENT": m2(
                tensor["summary"], tensor["contact"], tensor["static_a"], tensor["static_b"]
            ),
            "M3_JOINT_RNN": m3_head(
                m3_states[row], tensor["contact"], tensor["static_a"], tensor["static_b"]
            ),
        }
        for model_name, output in outputs.items():
            rows, metric = _prediction_rows(
                example,
                model_name,
                output["candidate_a"].cpu().numpy(),
                output["candidate_b"].cpu().numpy(),
                seed=seed,
            )
            prediction_rows.extend(rows)
            metric_rows.append(metric)

    for example_index, example in enumerate(heldout):
        for draw in range(int(config["order_shuffle_draws"])):
            order = _order_shuffle_permutation(example, seed=seed, draw=draw)
            state = _batch_history_states(
                m3_history,
                [example],
                device=device,
                chunk_events=config["chunk_events"],
                orders=[order],
            )[0]
            tensor = _example_tensors(example, device)
            output = m3_head(
                state, tensor["contact"], tensor["static_a"], tensor["static_b"]
            )
            rows, metric = _prediction_rows(
                example,
                "M3_ORDER_SHUFFLE_FULL_HISTORY",
                output["candidate_a"].cpu().numpy(),
                output["candidate_b"].cpu().numpy(),
                seed=seed,
                draw=draw,
            )
            prediction_rows.extend(rows)
            metric_rows.append(metric)

    if len(heldout) >= 2:
        for target_index, target in enumerate(heldout):
            tensor = _example_tensors(target, device)
            for donor_index, donor in enumerate(heldout):
                if donor_index == target_index:
                    continue
                output = m3_head(
                    m3_states[donor_index],
                    tensor["contact"],
                    tensor["static_a"],
                    tensor["static_b"],
                )
                rows, metric = _prediction_rows(
                    target,
                    "M3_WITHIN_PATIENT_HISTORY_SWAP",
                    output["candidate_a"].cpu().numpy(),
                    output["candidate_b"].cpu().numpy(),
                    seed=seed,
                    donor_seizure_id=donor.seizure_id,
                )
                prediction_rows.extend(rows)
                metric_rows.append(metric)
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def _initial_static_deviation(
    head: DualCandidateResidualHead,
    frozen: dict[str, torch.Tensor],
    groups: dict[str, list[Example]],
    device: torch.device,
) -> dict:
    distances = []
    angles = []
    with torch.no_grad():
        for subject, examples in groups.items():
            for row, example in enumerate(examples):
                tensor = _example_tensors(example, device)
                output = head(
                    frozen[subject][row],
                    tensor["contact"],
                    tensor["static_a"],
                    tensor["static_b"],
                )
                for branch in ("a", "b"):
                    base = unit_eps(tensor[f"static_{branch}"])
                    prediction = output[f"candidate_{branch}"]
                    distances.append(float(torch.linalg.vector_norm(prediction - base).cpu()))
                    cosine = torch.sum(prediction * base) / (
                        torch.linalg.vector_norm(prediction) * torch.linalg.vector_norm(base)
                    ).clamp_min(1e-12)
                    angles.append(float(torch.rad2deg(torch.arccos(cosine.clamp(-1, 1))).cpu()))
    return {
        "median_l2_difference": float(np.median(distances)),
        "max_l2_difference": float(np.max(distances)),
        "median_angle_degrees": float(np.median(angles)),
        "max_angle_degrees": float(np.max(angles)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4/cache",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4/per_subject",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_history_conditioned_field_refinement_v0_4.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve() / f"seed_{args.seed}" / args.heldout_subject
    if (output / "DONE.json").exists():
        print((output / "DONE.json").read_text(), end="")
        return
    if output.exists():
        archive = (
            args.output_root.resolve()
            / "diagnostic_archives"
            / f"{args.heldout_subject}_seed{args.seed}_incomplete_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(output, archive)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.resolve().read_text())
    if args.smoke:
        config = dict(config)
        config["common_frozen_recurrent_head_epochs"] = 1
        config["m1_frozen_recurrent_continuation_epochs"] = 1
        config["m2_total_epochs"] = 1
        config["m3_joint_epochs_after_common_stage"] = 1
        config["order_shuffle_draws"] = 2
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cache = args.cache_root.resolve() / f"outer_{args.heldout_subject}"
    examples, cache_index = _load_examples(cache)
    train = [example for example in examples if example.subject != args.heldout_subject]
    heldout = [example for example in examples if example.subject == args.heldout_subject]
    if not train or not heldout:
        raise RuntimeError("outer train/test split is empty")
    if args.heldout_subject != "epilepsiae_1146" and len(set(row.subject for row in train)) != 14:
        raise RuntimeError("formal outer fold must contain 14 target-training patients")
    history_checkpoint = Path(cache_index["outer_fold_shared_encoder"]["history_checkpoint"])
    frozen_history = _load_history_initialization(history_checkpoint, device)
    train_groups = _patient_groups(train)
    frozen_train_states = _frozen_states(
        frozen_history,
        train_groups,
        device=device,
        chunk_events=config["chunk_events"],
    )
    event_dim = train[0].event_embedding.shape[1]
    contact_dim = train[0].contact_embedding.shape[1]
    state_dim = int(config["history_dim"])
    head_kwargs = {
        "initial_gain": config["initial_gain"],
        "epsilon": config["unit_epsilon"],
        "norm_threshold": config["residual_norm_threshold"],
    }
    common_head = DualCandidateResidualHead(state_dim, contact_dim, **head_kwargs).to(device)
    initial_deviation = _initial_static_deviation(
        common_head, frozen_train_states, train_groups, device
    )
    common_optimizer = torch.optim.AdamW(
        common_head.parameters(), lr=config["head_gain_lr"], weight_decay=0.0
    )
    common_orders = _epoch_orders(
        sorted(train_groups), args.seed, config["common_frozen_recurrent_head_epochs"]
    )
    log = _train_head_stage(
        common_head,
        common_optimizer,
        frozen_train_states,
        train_groups,
        common_orders,
        stage="M1_M3_common_frozen_recurrent",
        epoch_offset=0,
        config=config,
        device=device,
        started=started,
    )
    common_head_state = copy.deepcopy(common_head.state_dict())
    common_optimizer_state = copy.deepcopy(common_optimizer.state_dict())
    continuation_orders = _epoch_orders(
        sorted(train_groups),
        args.seed + 100000,
        config["m1_frozen_recurrent_continuation_epochs"],
    )

    m1_head = copy.deepcopy(common_head)
    m1_optimizer = torch.optim.AdamW(
        m1_head.parameters(), lr=config["head_gain_lr"], weight_decay=0.0
    )
    m1_optimizer.load_state_dict(common_optimizer_state)
    log.extend(
        _train_head_stage(
            m1_head,
            m1_optimizer,
            frozen_train_states,
            train_groups,
            continuation_orders,
            stage="M1_frozen_recurrent_continuation",
            epoch_offset=config["common_frozen_recurrent_head_epochs"],
            config=config,
            device=device,
            started=started,
        )
    )

    m3_head = DualCandidateResidualHead(state_dim, contact_dim, **head_kwargs).to(device)
    m3_head.load_state_dict(common_head_state)
    m3_history = _load_history_initialization(history_checkpoint, device)
    initial_history = {
        name: parameter.detach().clone() for name, parameter in m3_history.named_parameters()
    }
    m3_optimizer = torch.optim.AdamW(
        m3_head.parameters(), lr=config["head_gain_lr"], weight_decay=0.0
    )
    m3_optimizer.load_state_dict(common_optimizer_state)
    m3_optimizer.add_param_group(
        {"params": list(m3_history.parameters()), "lr": config["recurrent_decay_lr"]}
    )
    log.extend(
        _train_joint_stage(
            m3_head,
            m3_history,
            m3_optimizer,
            train_groups,
            continuation_orders,
            epoch_offset=config["common_frozen_recurrent_head_epochs"],
            initial_history=initial_history,
            config=config,
            device=device,
            started=started,
        )
    )

    m2 = TimeAwareNonrecurrentResidual(
        summary_dim=len(train[0].time_summary),
        state_dim=state_dim,
        contact_dim=contact_dim,
        **head_kwargs,
    ).to(device)
    m2_optimizer = torch.optim.AdamW(
        m2.parameters(), lr=config["head_gain_lr"], weight_decay=0.0
    )
    m2_orders = _epoch_orders(sorted(train_groups), args.seed + 200000, config["m2_total_epochs"])
    log.extend(
        _train_m2(
            m2,
            m2_optimizer,
            train_groups,
            m2_orders,
            config=config,
            device=device,
            started=started,
        )
    )
    predictions, metrics = _evaluate(
        heldout,
        frozen_history,
        m1_head,
        m2,
        m3_history,
        m3_head,
        config=config,
        device=device,
        seed=args.seed,
    )
    pd.DataFrame(log).to_csv(output / "training_log.csv", index=False)
    predictions.to_csv(output / "heldout_candidate_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output / "heldout_seizure_metrics.csv", index=False)
    torch.save(
        {
            "contract": config["contract"],
            "heldout_subject": args.heldout_subject,
            "seed": args.seed,
            "common_head_state": common_head_state,
            "m1_head_state": m1_head.state_dict(),
            "m2_state": m2.state_dict(),
            "m3_head_state": m3_head.state_dict(),
            "m3_history_state": m3_history.state_dict(),
            "history_initialization": str(history_checkpoint),
            "outer_encoder_sha256": cache_index["outer_fold_shared_encoder"]["event_checkpoint_sha256"],
            "config": config,
        },
        output / "checkpoint.pt",
    )
    true_metrics = metrics.loc[metrics.draw == -1]
    aggregate = {
        model: float(group.maxab_1_45.mean())
        for model, group in true_metrics.groupby("model", sort=True)
    }
    done = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "heldout_subject": args.heldout_subject,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "n_train_patients": len(train_groups),
        "n_train_seizures": len(train),
        "n_heldout_seizures": len(heldout),
        "outer_fold_encoder_sha256": cache_index["outer_fold_shared_encoder"]["event_checkpoint_sha256"],
        "target_values_used_for_training": "1-45Hz_outer_training_patients_only",
        "heldout_target_used_for_training": False,
        "initial_output_deviation_from_static": initial_deviation,
        "heldout_mean_maxab_1_45": aggregate,
        "final_gains": {
            "m1": m1_head.gains.detach().cpu().tolist(),
            "m2": m2.head.gains.detach().cpu().tolist(),
            "m3": m3_head.gains.detach().cpu().tolist(),
        },
        "final_m3_half_life_hours": _history_half_life(m3_history),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
        "elapsed_seconds": time.time() - started,
    }
    (output / "DONE.json").write_text(
        json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
