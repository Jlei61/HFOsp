#!/usr/bin/env python3
"""Train one patient-specific shared-axis Figure 6 unit (v0.4).

The runner is deliberately target-sealed.  It reads one patient's masked
interictal rank events, a fit60 conditional-hazard bias, and contact shaft
names.  No empirical A/B field, mean-rank feature, clinical label, or ictal
array is loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_patient_specific_rnn_bridge import chronological_60_20_20  # noqa: E402
from src.topic5_shared_axis_rnn import (  # noqa: E402
    SharedAxisPropagationRNN,
    axis_smoothness_penalty,
)
from src.topic5_shared_scaffold_rnn import (  # noqa: E402
    OrdinaryDenseGRUBaseline,
    build_fixed_local_shaft_adjacency,
    estimate_node_hazard_bias,
)


MODELS = ("shared_axis", "ordinary_gru", "shared_axis_rank_shuffle")


@dataclass(frozen=True)
class PatientRecord:
    subject: str
    path: Path
    contact_names: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    event_split: np.ndarray

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 0)

    @property
    def eval_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_one_patient_record(dataset_root: Path, subject: str) -> PatientRecord:
    """Load only the requested patient; never deserialize another patient."""

    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("target_values_read", True)):
        raise RuntimeError("rank dataset does not certify sealed ictal targets")
    cohort = list(map(str, manifest.get("cohort_subjects", ())))
    if len(cohort) != 34 or subject not in cohort:
        raise KeyError(f"{subject!r} is not in the frozen 34-patient cohort")
    path = dataset_root / "per_subject" / f"{subject}.npz"
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if sha256_file(path) != str(sidecar["dataset_npz_sha256"]):
        raise RuntimeError(f"{subject}: input fingerprint mismatch")
    with np.load(path, allow_pickle=False) as data:
        record = PatientRecord(
            subject=subject,
            path=path,
            contact_names=np.asarray(data["contact_names"]),
            group_ids=np.asarray(data["event_group_ids"], dtype=np.int16),
            group_count=np.asarray(data["event_group_count"], dtype=np.int16),
            event_split=np.asarray(data["event_split"], dtype=np.uint8),
        )
    if record.group_ids.shape[0] != record.group_count.size:
        raise RuntimeError(f"{subject}: group arrays are misaligned")
    if record.group_ids.shape[0] != record.event_split.size:
        raise RuntimeError(f"{subject}: event split is misaligned")
    return record


def within_event_rank_shuffle(groups: np.ndarray, *, seed: int) -> np.ndarray:
    """Freeze one participation-preserving rank-label shuffle per unit."""

    output = np.asarray(groups, dtype=np.int16).copy()
    rng = np.random.default_rng(int(seed) + 47_000_003)
    for row in output:
        participating = np.flatnonzero(row >= 0)
        if participating.size > 1:
            row[participating] = rng.permutation(row[participating])
    return output


def make_model(
    model_name: str,
    *,
    participation_bias: np.ndarray,
    contact_names: np.ndarray,
    config: Mapping[str, Any],
) -> torch.nn.Module:
    if model_name in {"shared_axis", "shared_axis_rank_shuffle"}:
        adjacency = build_fixed_local_shaft_adjacency(
            channel_names=[str(item) for item in contact_names]
        )
        operator = config["axis_operator"]
        return SharedAxisPropagationRNN(
            fixed_adjacency=adjacency,
            participation_bias=participation_bias,
            length_scale=float(operator["length_scale"]),
            delta=float(operator["delta"]),
            smoothness_weight=float(operator["smoothness_weight"]),
            direction_gain=float(operator["direction_gain"]),
        )
    if model_name == "ordinary_gru":
        return OrdinaryDenseGRUBaseline(
            participation_bias=participation_bias,
            hidden_size=int(config["models"]["ordinary_hidden_size"]),
        )
    raise ValueError(f"unknown model {model_name!r}")


def _tensor_batch(
    groups: np.ndarray, counts: np.ndarray, indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(groups[indices], dtype=torch.long, device=device),
        torch.as_tensor(counts[indices], dtype=torch.long, device=device),
    )


def weighted_training_loss(
    model: torch.nn.Module,
    groups: torch.Tensor,
    counts: torch.Tensor,
    *,
    n_macro_events: int,
    n_macro_transition_events: int,
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact event-first loss contribution for one gradient micro-batch."""

    likelihood = model.batched_event_log_likelihood(groups, counts)
    decision_count = likelihood["decision_count"].to(torch.float32)
    continuation_count = likelihood["nonterminal_decision_count"].to(torch.float32)
    stop_sum = (-likelihood["stop"] / decision_count).sum()
    has_transition = continuation_count > 0
    if torch.any(has_transition):
        contact_sum = (
            -likelihood["conditional_contacts"][has_transition]
            / continuation_count[has_transition]
        ).sum()
        cardinality_sum = (
            -likelihood["cardinality"][has_transition]
            / continuation_count[has_transition]
        ).sum()
    else:
        contact_sum = stop_sum * 0.0
        cardinality_sum = stop_sum * 0.0
    stop = stop_sum / float(n_macro_events)
    if n_macro_transition_events:
        contacts = contact_sum / float(n_macro_transition_events)
        cardinality = cardinality_sum / float(n_macro_transition_events)
    else:
        contacts = stop_sum * 0.0
        cardinality = stop_sum * 0.0
    joint = (
        float(weights["contact"]) * contacts
        + float(weights["cardinality"]) * cardinality
        + float(weights["stop"]) * stop
    )
    # Same-shaft axis smoothness.  It only touches the learned coordinate and
    # only through the fixed shaft graph, so it imports no target structure.
    smoothness = (
        model.smoothness_penalty()
        if hasattr(model, "smoothness_penalty")
        else joint * 0.0
    )
    joint = joint + smoothness
    return joint, {
        "axis_smoothness": float(smoothness.detach().cpu()),
        "joint": float(joint.detach().cpu()),
        "conditional_contacts": float(contacts.detach().cpu()),
        "cardinality": float(cardinality.detach().cpu()),
        "stop": float(stop.detach().cpu()),
    }


@torch.no_grad()
def evaluate_groups(
    model: torch.nn.Module,
    groups: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Decision-first decomposed NLL plus tied-set-aware top-1 accuracy."""

    model.eval()
    totals = {name: 0.0 for name in ("conditional_contacts", "cardinality", "stop")}
    n_decisions = 0
    n_continue = 0
    top1_hits = 0
    top1_total = 0
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        group_tensor, count_tensor = _tensor_batch(groups, counts, batch_indices, device)
        likelihood = model.batched_event_log_likelihood(group_tensor, count_tensor)
        for name in totals:
            totals[name] += float((-likelihood[name].sum()).detach().cpu())
        n_decisions += int(likelihood["decision_count"].sum().item())
        n_continue += int(likelihood["nonterminal_decision_count"].sum().item())

        state = model.reset_state(batch_size=len(batch_indices))
        seen = torch.zeros_like(group_tensor, dtype=torch.bool)
        max_steps = int(count_tensor.max().item())
        for step in range(max_steps):
            active = count_tensor > step
            current = group_tensor == step
            state = model.observe(state, current, active=active)
            seen = seen | current
            continuing = count_tensor > step + 1
            if torch.any(continuing):
                decision = model.decision(state, seen)
                predicted = torch.argmax(decision["node_logits"], dim=1)
                target = group_tensor == step + 1
                rows = torch.where(continuing)[0]
                top1_hits += int(target[rows, predicted[rows]].sum().item())
                top1_total += int(rows.numel())

    if not n_decisions or not n_continue:
        raise RuntimeError("evaluation split has no scorable continuation decisions")
    contact_nll = totals["conditional_contacts"] / n_continue
    cardinality_nll = totals["cardinality"] / n_continue
    stop_nll = totals["stop"] / n_decisions
    return {
        "contact_nll_per_continue_decision": contact_nll,
        "cardinality_nll_per_continue_decision": cardinality_nll,
        "stop_nll_per_decision": stop_nll,
        "joint_weighted_nll": (
            float(weights["contact"]) * contact_nll
            + float(weights["cardinality"]) * cardinality_nll
            + float(weights["stop"]) * stop_nll
        ),
        "top1_next_contact_accuracy": top1_hits / top1_total,
        "top1_hits": top1_hits,
        "n_continue_decisions": n_continue,
        "n_all_decisions": n_decisions,
        "n_events": int(len(indices)),
    }


def _state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(key): value.detach().cpu().clone() for key, value in state.items()}


def save_operator(
    path: Path,
    model: torch.nn.Module,
    *,
    model_name: str,
    participation_bias: np.ndarray,
    contact_names: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {
        "contact_names": np.asarray(contact_names),
        "participation_bias": np.asarray(participation_bias, dtype=np.float32),
        "model_name": np.asarray(model_name),
    }
    if isinstance(model, SharedAxisPropagationRNN):
        for key, value in model.operator_components().items():
            payload[key] = value.detach().cpu().numpy()
        payload.update(
            rho_p=np.asarray(float(model.rho_p.detach().cpu())),
            rho_r=np.asarray(float(model.rho_r.detach().cpu())),
            propagation_weight=np.asarray(
                float(model.propagation_weight.detach().cpu())
            ),
            restraint_weight=np.asarray(float(model.restraint_weight.detach().cpu())),
            gamma=np.asarray(float(model.gamma.detach().cpu())),
            gain=np.asarray(float(model.gain.detach().cpu())),
            flow_weight=np.asarray(float(model.flow_weight.detach().cpu())),
            direction_gain=np.asarray(float(model.direction_gain)),
        )
    else:
        for key, value in model.state_dict().items():
            payload[f"state__{key}"] = value.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.model not in MODELS or args.model not in config["models"]["names"]:
        raise ValueError(f"model must be one of {MODELS}")
    dataset_artifact_root = Path(config["dataset_artifact_root"]).resolve()
    dataset_root = dataset_artifact_root / config["dataset_root"]
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    if args.smoke:
        output_root = output_root / "smoke"
    unit_name = args.model if args.fit_half is None else f"{args.model}__{args.fit_half}_half"
    run_dir = output_root / "per_subject" / args.subject / unit_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "DONE.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get("status") == "COMPLETE":
            print(json.dumps(done), flush=True)
            return done
    (run_dir / "FAILED.json").unlink(missing_ok=True)
    progress_path = run_dir / "resume_state.pt"
    if progress_path.exists() and not args.resume:
        raise RuntimeError(f"resume state exists; rerun with --resume: {progress_path}")

    seed_everything(args.seed)
    resources = config["resources"]
    torch.set_num_threads(int(resources["torch_num_threads"]))
    device = torch.device(args.device or resources["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_index = None
    if device.type == "cuda":
        device_index = int(device.index or 0)
        torch.cuda.set_device(device_index)
        fraction = float(resources.get("gpu_memory_fraction_per_process", 0.0))
        if 0.0 < fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

    record = load_one_patient_record(dataset_root, args.subject)
    fit60, validation20, test20 = chronological_60_20_20(record)
    if args.fit_half is not None:
        midpoint = len(fit60) // 2
        fit60 = fit60[:midpoint] if args.fit_half == "first" else fit60[midpoint:]
        if len(fit60) < 1:
            raise RuntimeError(f"{args.subject}: fit half is empty")
    if args.smoke:
        smoke = config["smoke"]
        fit60 = fit60[: int(smoke["fit_events"])]
        validation20 = validation20[: int(smoke["validation_events"])]
        test20 = test20[: int(smoke["test_events"])]
    groups = np.asarray(record.group_ids, dtype=np.int16)
    counts = np.asarray(record.group_count, dtype=np.int16)
    hazard = estimate_node_hazard_bias(
        groups[fit60], pseudocount=float(config["training"]["hazard_pseudocount"])
    )
    participation_bias = np.asarray(hazard["bias"], dtype=np.float32)
    contact_names = np.asarray(record.contact_names)
    model = make_model(
        args.model,
        participation_bias=participation_bias,
        contact_names=contact_names,
        config=config,
    ).to(device)

    training = dict(config["training"])
    if args.learning_rate is not None:
        training["learning_rate"] = float(args.learning_rate)
    if args.coverage_cycles is not None:
        training["coverage_cycles"] = int(args.coverage_cycles)
    if args.optimizer_updates_per_cycle is not None:
        training["optimizer_updates_per_cycle"] = int(
            args.optimizer_updates_per_cycle
        )
    if args.smoke:
        training["coverage_cycles"] = int(config["smoke"]["coverage_cycles"])
        training["optimizer_updates_per_cycle"] = int(
            config["smoke"]["optimizer_updates_per_cycle"]
        )
        training["micro_batch_events"] = int(config["smoke"]["micro_batch_events"])
    weights = {
        "contact": float(training["contact_weight"]),
        "cardinality": float(training["cardinality_weight"]),
        "stop": float(training["stop_weight"]),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if args.model == "structured_rank_shuffle":
        train_groups = groups.copy()
        train_groups[fit60] = within_event_rank_shuffle(
            groups[fit60], seed=args.seed
        )
    else:
        train_groups = groups
    shuffle_hash = (
        sha256_array(train_groups[fit60])
        if args.model == "structured_rank_shuffle"
        else None
    )
    resolved = {
        "contract": config["contract"],
        "subject": args.subject,
        "model": args.model,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "device": str(device),
        "training": training,
        "loss_weights": weights,
        "allowed_inputs_used": [
            "fit60_conditional_hazard_bias",
            "contact_shaft_names",
            "current_and_past_rank_sets",
        ],
        "forbidden_inputs_used": [],
        "target_values_read": False,
        "other_patient_events_used": False,
        "n_contacts": int(len(contact_names)),
        "n_events": {
            "fit60": int(len(fit60)),
            "validation20": int(len(validation20)),
            "test20": int(len(test20)),
        },
        "input_hashes": {
            "dataset_npz": sha256_file(record.path),
            "fit_indices": sha256_array(fit60.astype("<i8")),
            "validation_indices": sha256_array(validation20.astype("<i8")),
            "test_indices": sha256_array(test20.astype("<i8")),
            "participation_bias": sha256_array(participation_bias.astype("<f4")),
            "rank_shuffle_fit60": shuffle_hash,
            "config": sha256_file(config_path),
            "core_code": sha256_file(ROOT / "src/topic5_shared_scaffold_rnn.py"),
            "runner_code": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(run_dir / "resolved_config.json", resolved)

    completed_cycle = 0
    training_log: list[dict[str, Any]] = []
    best_cycle = 0
    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    if progress_path.exists() and args.resume:
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress["config_sha256"] != resolved["input_hashes"]["config"]:
            raise RuntimeError("resume config hash differs from current config")
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        completed_cycle = int(progress["completed_cycle"])
        training_log = list(progress["training_log"])
        best_cycle = int(progress["best_cycle"])
        best_metric = float(progress["best_metric"])
        best_state = progress["best_state"]
    atomic_text(
        run_dir / "train_log.jsonl",
        "".join(json.dumps(_jsonable(row), allow_nan=False) + "\n" for row in training_log),
    )

    started = time.time()
    cycles = int(training["coverage_cycles"])
    updates_per_cycle = int(training["optimizer_updates_per_cycle"])
    micro_batch = int(training["micro_batch_events"])
    if updates_per_cycle < 1 or micro_batch < 1 or len(fit60) < updates_per_cycle:
        raise ValueError("fit60 must support the frozen optimizer-update schedule")
    for cycle_index in range(completed_cycle, cycles):
        model.train()
        cycle_rng = np.random.default_rng(
            int(args.seed) * 1_000_003 + (cycle_index + 1) * 10_007
        )
        order = cycle_rng.permutation(fit60)
        update_chunks = np.array_split(order, updates_per_cycle)
        update_log = []
        for update_index, macro_indices in enumerate(update_chunks):
            optimizer.zero_grad(set_to_none=True)
            n_transition_events = int(np.sum(counts[macro_indices] > 1))
            aggregate = {
                name: 0.0
                for name in (
                    "joint", "conditional_contacts", "cardinality", "stop",
                    "axis_smoothness",
                )
            }
            for micro_start in range(0, len(macro_indices), micro_batch):
                micro_indices = macro_indices[micro_start : micro_start + micro_batch]
                group_tensor, count_tensor = _tensor_batch(
                    train_groups, counts, micro_indices, device
                )
                loss, pieces = weighted_training_loss(
                    model,
                    group_tensor,
                    count_tensor,
                    n_macro_events=len(macro_indices),
                    n_macro_transition_events=n_transition_events,
                    weights=weights,
                )
                loss.backward()
                for name, value in pieces.items():
                    aggregate[name] += value
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("non-finite model parameter after optimizer step")
            update_log.append(
                {
                    "update": update_index + 1,
                    "n_events": int(len(macro_indices)),
                    "n_micro_batches": int(math.ceil(len(macro_indices) / micro_batch)),
                    "n_transition_events": n_transition_events,
                    "gradient_norm_before_clip": float(torch.as_tensor(gradient_norm).cpu()),
                    **aggregate,
                }
            )

        validation = evaluate_groups(
            model,
            groups,
            counts,
            validation20,
            device=device,
            batch_size=micro_batch,
            weights=weights,
        )
        metric = float(validation["contact_nll_per_continue_decision"])
        if metric < best_metric:
            best_metric = metric
            best_cycle = cycle_index + 1
            best_state = _state_to_cpu(model.state_dict())
        cycle_row = {
            "cycle": cycle_index + 1,
            "fit_events_seen_exactly_once": int(len(order)),
            "fit_index_unique_count": int(np.unique(order).size),
            "optimizer_updates": updates_per_cycle,
            "train_update_mean": {
                name: float(np.mean([row[name] for row in update_log]))
                for name in ("joint", "conditional_contacts", "cardinality", "stop")
            },
            "updates": update_log,
            "validation": validation,
            "selected_as_best": best_cycle == cycle_index + 1,
            "best_cycle_so_far": best_cycle,
            "best_validation_contact_nll": best_metric,
            "elapsed_seconds": time.time() - started,
        }
        training_log.append(cycle_row)
        progress = {
            "config_sha256": resolved["input_hashes"]["config"],
            "completed_cycle": cycle_index + 1,
            "model_state": _state_to_cpu(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "training_log": training_log,
            "best_cycle": best_cycle,
            "best_metric": best_metric,
            "best_state": best_state,
        }
        atomic_torch_save(progress_path, progress)
        atomic_text(
            run_dir / "train_log.jsonl",
            "".join(
                json.dumps(_jsonable(row), allow_nan=False) + "\n"
                for row in training_log
            ),
        )
        print(
            f"{args.subject} {args.model} seed={args.seed} cycle={cycle_index + 1}/{cycles} "
            f"val_contact_nll={metric:.6f} best={best_metric:.6f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_metrics = evaluate_groups(
        model,
        groups,
        counts,
        validation20,
        device=device,
        batch_size=micro_batch,
        weights=weights,
    )
    test_metrics = evaluate_groups(
        model,
        groups,
        counts,
        test20,
        device=device,
        batch_size=micro_batch,
        weights=weights,
    )
    checkpoint = {
        "contract": config["contract"],
        "subject": args.subject,
        "model": args.model,
        "fit_half": args.fit_half,
        "seed": int(args.seed),
        "best_cycle": best_cycle,
        "model_hyperparameters": (
            dict(config["axis_operator"])
            if args.model in {"shared_axis", "shared_axis_rank_shuffle"}
            else {"hidden_size": int(config["models"]["ordinary_hidden_size"])}
        ),
        "model_state": best_state,
        "participation_bias": participation_bias,
        "hazard_counts": {
            key: value
            for key, value in hazard.items()
            if key in {"n_next", "n_eligible", "hazard_probability"}
        },
        "contact_names": contact_names,
        "fit_indices": fit60,
        "validation_indices": validation20,
        "test_indices": test20,
        "target_values_read": False,
        "empirical_ab_used": False,
    }
    atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
    save_operator(
        run_dir / "operator.npz",
        model,
        model_name=args.model,
        participation_bias=participation_bias,
        contact_names=contact_names,
    )
    atomic_json(run_dir / "validation_metrics.json", validation_metrics)
    atomic_json(run_dir / "test_metrics.json", test_metrics)
    artifact_paths = [
        run_dir / "resolved_config.json",
        run_dir / "checkpoint.pt",
        run_dir / "train_log.jsonl",
        run_dir / "validation_metrics.json",
        run_dir / "test_metrics.json",
        run_dir / "operator.npz",
    ]
    summary = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "subject": args.subject,
        "model": args.model,
        "fit_half": args.fit_half,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "best_cycle": best_cycle,
        "validation": validation_metrics,
        "test": test_metrics,
        "n_events": resolved["n_events"],
        "n_contacts": int(len(contact_names)),
        "coverage": {
            "completed_cycles": cycles,
            "events_per_cycle": int(len(fit60)),
            "optimizer_updates_per_cycle": updates_per_cycle,
            "micro_batch_events": micro_batch,
            "all_fit_events_seen_per_cycle": True,
        },
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_index) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2),
        "target_values_read": False,
        "other_patient_events_used": False,
        "empirical_ab_used": False,
        "rank_shuffle_fit60_sha256": shuffle_hash,
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    atomic_json(done_path, summary)
    progress_path.unlink(missing_ok=True)
    (run_dir / "FAILED.json").unlink(missing_ok=True)
    print(json.dumps(_jsonable(summary), allow_nan=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_ictal_shared_axis_rnn_v0_4.yaml",
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    # Axis-stability leg: refit on one chronological half of fit60.
    # validation20 / test20 are untouched, so the halves stay comparable
    # to each other and to the full-fit model on the same held-out data.
    parser.add_argument("--fit-half", choices=("first", "second"), default=None)
    parser.add_argument("--coverage-cycles", type=int, default=None)
    parser.add_argument("--optimizer-updates-per-cycle", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        try:
            config = yaml.safe_load(args.config.resolve().read_text())
            output_root = (
                Path(args.output_root).resolve()
                if args.output_root
                else ROOT / config["output_root"]
            )
            if args.smoke:
                output_root = output_root / "smoke"
            failure = (
                output_root
                / "per_subject"
                / args.subject
                / args.model
                / f"seed_{args.seed}"
                / "FAILED.json"
            )
            atomic_json(
                failure,
                {
                    "status": "FAILED",
                    "subject": args.subject,
                    "model": args.model,
                    "seed": args.seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "target_values_read": False,
                    "updated_unix": time.time(),
                },
            )
        finally:
            raise


if __name__ == "__main__":
    main()
