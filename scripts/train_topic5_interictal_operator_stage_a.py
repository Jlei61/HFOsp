#!/usr/bin/env python3
"""Train and evaluate the v0.3 within-event contact-query GRU (Stage A).

This runner never opens an ictal target. Shared initialization uses other
patients; held-out-patient calibration uses only its chronological first 80%
interictal events. The last 20% are read for evaluation only after calibration
is complete.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required; run this script in conda env cuda_env") from exc

from src.topic5_interictal_operator import (  # noqa: E402
    ContactQueryGRU,
    StaticContactQuery,
    contact_query_loss,
    fit_empirical_template_baseline,
    fit_first_order_markov,
    pairwise_rank_concordance,
    prefix_targets,
)


@dataclass
class SubjectRecord:
    subject: str
    dataset: str
    path: Path
    contact_features: np.ndarray
    contact_names: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    event_split: np.ndarray
    event_source_index: np.ndarray
    support: np.ndarray
    input_sha256: str

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 0)

    @property
    def eval_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_records(dataset_dir: Path) -> Dict[str, SubjectRecord]:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    if bool(manifest.get("target_values_read", True)):
        raise RuntimeError("dataset manifest does not certify target isolation")
    audit = pd.read_csv(dataset_dir / "subject_audit.csv")
    audit = audit[audit.status.astype(str) == "ok"]
    records = {}
    for row in audit.itertuples():
        subject = str(row.subject)
        path = dataset_dir / "per_subject" / f"{subject}.npz"
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        expected = str(metadata["dataset_npz_sha256"])
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{subject}: dataset fingerprint mismatch")
        with np.load(path, allow_pickle=False) as z:
            record = SubjectRecord(
                subject=subject,
                dataset=str(row.dataset),
                path=path,
                contact_features=np.asarray(z["contact_features"], np.float32),
                contact_names=np.asarray(z["contact_names"]),
                group_ids=np.asarray(z["event_group_ids"], np.int16),
                group_count=np.asarray(z["event_group_count"], np.int16),
                event_split=np.asarray(z["event_split"], np.uint8),
                event_source_index=np.asarray(z["event_source_index"], np.int64),
                support=np.asarray(z["prefix_participation_support"], np.float32),
                input_sha256=actual,
            )
        if record.group_ids.shape[0] != record.event_split.size:
            raise RuntimeError(f"{subject}: event array shape mismatch")
        records[subject] = record
    if not records:
        raise RuntimeError("no Stage-A subject record is available")
    return records


def shuffled_group_matrix(groups: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle group labels within each event while preserving participation."""
    groups = np.asarray(groups, np.int16)
    out = groups.copy()
    rng = np.random.default_rng(int(seed))
    for event_index, event in enumerate(groups):
        participating = np.flatnonzero(event >= 0)
        if participating.size:
            out[event_index, participating] = rng.permutation(event[participating])
    return out


def _balanced_record_choice(
    records: Sequence[SubjectRecord], rng: np.random.Generator
) -> SubjectRecord:
    by_dataset: Dict[str, list[SubjectRecord]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)
    dataset = str(rng.choice(sorted(by_dataset)))
    candidates = by_dataset[dataset]
    return candidates[int(rng.integers(0, len(candidates)))]


def _example(
    record: SubjectRecord,
    event_index: int,
    tau: int,
    *,
    groups_override: Optional[np.ndarray] = None,
) -> dict:
    group_ids = (
        record.group_ids[event_index]
        if groups_override is None
        else groups_override[event_index]
    )
    target = prefix_targets(group_ids, int(tau))
    return {
        "record": record,
        "event_index": int(event_index),
        "tau": int(tau),
        "group_ids": np.asarray(group_ids, np.int16),
        **target,
    }


def sample_examples(
    records: Sequence[SubjectRecord],
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle_groups: Optional[Mapping[str, np.ndarray]] = None,
) -> list[dict]:
    examples = []
    for _ in range(int(batch_size)):
        record = _balanced_record_choice(records, rng)
        indices = record.train_indices
        event_index = int(indices[int(rng.integers(0, len(indices)))])
        override = None if shuffle_groups is None else shuffle_groups[record.subject]
        n_groups = int(record.group_count[event_index])
        tau = int(rng.integers(1, n_groups + 1))
        examples.append(
            _example(record, event_index, tau, groups_override=override)
        )
    return examples


def collate_examples(examples: Sequence[dict], device: torch.device) -> dict:
    batch_size = len(examples)
    max_contacts = max(example["record"].contact_features.shape[0] for example in examples)
    max_steps = max(int(example["tau"]) for example in examples)
    n_features = examples[0]["record"].contact_features.shape[1]
    arrays = {
        "contact_features": np.zeros(
            (batch_size, max_contacts, n_features), np.float32
        ),
        "contact_mask": np.zeros((batch_size, max_contacts), bool),
        "prefix_sets": np.zeros((batch_size, max_steps, max_contacts), np.float32),
        "step_mask": np.zeros((batch_size, max_steps), bool),
        "recruited": np.zeros((batch_size, max_contacts), bool),
        "next_set": np.zeros((batch_size, max_contacts), bool),
        "terminal": np.zeros(batch_size, bool),
        "remaining": np.zeros((batch_size, max_contacts), bool),
        "suffix_group": np.full((batch_size, max_contacts), -1, np.int16),
    }
    for sample, example in enumerate(examples):
        record = example["record"]
        n_contacts = record.contact_features.shape[0]
        arrays["contact_features"][sample, :n_contacts] = record.contact_features
        arrays["contact_mask"][sample, :n_contacts] = True
        group_ids = example["group_ids"]
        for step in range(int(example["tau"])):
            arrays["prefix_sets"][sample, step, :n_contacts] = group_ids == step
            arrays["step_mask"][sample, step] = True
        for key in ("recruited", "next_set", "remaining", "suffix_group"):
            arrays[key][sample, :n_contacts] = example[key]
        arrays["terminal"][sample] = bool(example["terminal"])
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in arrays.items()
    }


def _forward_loss(model, batch, loss_weights):
    outputs = model(
        batch["contact_features"],
        batch["contact_mask"],
        batch["prefix_sets"],
        batch["step_mask"],
        batch["recruited"],
    )
    return outputs, contact_query_loss(outputs, batch, loss_weights=loss_weights)


def _validation_examples(
    records: Sequence[SubjectRecord], max_per_subject: int
) -> list[dict]:
    out = []
    for record in records:
        candidates = []
        for event_index in record.eval_indices:
            for tau in range(1, int(record.group_count[event_index]) + 1):
                candidates.append(_example(record, int(event_index), tau))
        if len(candidates) > max_per_subject:
            take = np.linspace(0, len(candidates) - 1, max_per_subject).round().astype(int)
            candidates = [candidates[index] for index in np.unique(take)]
        out.extend(candidates)
    return out


@torch.no_grad()
def validation_loss(
    model,
    examples: Sequence[dict],
    device: torch.device,
    loss_weights,
    *,
    batch_size: int = 512,
) -> float:
    model.eval()
    subject_losses: Dict[str, list[float]] = {}
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate_examples(chunk, device)
        _, loss = _forward_loss(model, batch, loss_weights)
        # A batch scalar is assigned to its represented patients only for
        # early-stopping stability; formal metrics are computed separately.
        value = float(loss["total"].detach().cpu())
        for subject in {example["record"].subject for example in chunk}:
            subject_losses.setdefault(subject, []).append(value)
    if not subject_losses:
        return float("nan")
    return float(np.mean([np.mean(values) for values in subject_losses.values()]))


def train_model(
    model,
    records: Sequence[SubjectRecord],
    *,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    loss_weights: Mapping[str, float],
    seed: int,
    validation_examples: Optional[Sequence[dict]] = None,
    patience: Optional[int] = None,
    shuffle_groups: Optional[Mapping[str, np.ndarray]] = None,
    phase: str,
    restore_best: bool = True,
) -> tuple[dict, list[dict]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed))
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    stale = 0
    log = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        running: Dict[str, list[float]] = {
            key: [] for key in ("total", "next_set", "stop", "remaining_participation", "suffix_rank")
        }
        epoch_start = time.time()
        for _ in range(int(steps_per_epoch)):
            examples = sample_examples(
                records,
                int(batch_size),
                rng,
                shuffle_groups=shuffle_groups,
            )
            batch = collate_examples(examples, device)
            optimizer.zero_grad(set_to_none=True)
            _, losses = _forward_loss(model, batch, loss_weights)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            for key in running:
                running[key].append(float(losses[key].detach().cpu()))

        if validation_examples is not None:
            valid = validation_loss(
                model, validation_examples, device, loss_weights, batch_size=512
            )
        else:
            valid = float(np.mean(running["total"]))
        if valid < best_value - 1e-6:
            best_value = valid
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        row = {
            "phase": phase,
            "epoch": epoch,
            **{f"train_{key}": float(np.mean(value)) for key, value in running.items()},
            "selection_loss": valid,
            "best_selection_loss": best_value,
            "elapsed_sec": time.time() - epoch_start,
            "gpu_allocated_mb": (
                float(
                    torch.cuda.max_memory_allocated(
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                    / 2**20
                )
                if device.type == "cuda"
                else 0.0
            ),
        }
        log.append(row)
        print(
            f"[{phase}] epoch={epoch:03d} train={row['train_total']:.4f} "
            f"select={valid:.4f} best={best_value:.4f}",
            flush=True,
        )
        if patience is not None and stale >= int(patience):
            print(f"[{phase}] early stop after {stale} stale epochs", flush=True)
            break
    if restore_best:
        model.load_state_dict(best_state)
        selected_state = best_state
    else:
        selected_state = copy.deepcopy(model.state_dict())
    return selected_state, log


def _all_eval_examples(record: SubjectRecord) -> list[dict]:
    examples = []
    for event_index in record.eval_indices:
        for tau in range(1, int(record.group_count[event_index]) + 1):
            examples.append(_example(record, int(event_index), tau))
    return examples


def _set_nll(scores: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    values = np.maximum(np.asarray(scores, float), 1e-12)
    target = np.asarray(target, bool)
    valid = np.asarray(valid, bool)
    denominator = float(np.sum(values[valid]))
    numerator = float(np.sum(values[target]))
    if denominator <= 0 or numerator <= 0:
        return float("nan")
    return float(-np.log(numerator / denominator))


@torch.no_grad()
def evaluate_heldout(
    model,
    record: SubjectRecord,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> tuple[dict, pd.DataFrame]:
    """First and only held-out-last-20% model read after calibration."""
    examples = _all_eval_examples(record)
    markov = fit_first_order_markov(record.group_ids[record.train_indices])
    empirical_templates = fit_empirical_template_baseline(
        record.group_ids[record.train_indices]
    )
    rows = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate_examples(chunk, device)
        outputs = model(
            batch["contact_features"],
            batch["contact_mask"],
            batch["prefix_sets"],
            batch["step_mask"],
            batch["recruited"],
        )
        next_logits = outputs["next_logits"].detach().cpu().numpy()
        utility = outputs["suffix_utility"].detach().cpu().numpy()
        stop_probability = torch.sigmoid(outputs["stop_logit"]).detach().cpu().numpy()
        remaining_probability = (
            torch.sigmoid(outputs["remaining_participation_logits"]).detach().cpu().numpy()
        )
        for local_index, example in enumerate(chunk):
            n_contacts = example["record"].contact_features.shape[0]
            recruited = np.asarray(example["recruited"], bool)
            target = np.asarray(example["next_set"], bool)
            terminal = bool(example["terminal"])
            valid = ~recruited
            last_set = example["group_ids"] == int(example["tau"]) - 1
            model_nll = (
                _set_nll(np.exp(next_logits[local_index, :n_contacts] - np.max(next_logits[local_index, :n_contacts])), target, valid)
                if not terminal
                else np.nan
            )
            support_nll = (
                _set_nll(record.support, target, valid) if not terminal else np.nan
            )
            markov_score = markov.scores(last_set, recruited)
            markov_nll = (
                _set_nll(markov_score, target, valid) if not terminal else np.nan
            )
            template_score, template_utility = empirical_templates.scores(
                example["group_ids"], int(example["tau"])
            )
            template_nll = (
                _set_nll(template_score, target, valid) if not terminal else np.nan
            )
            suffix_group = np.asarray(example["suffix_group"], int)
            rows.append(
                {
                    "subject": record.subject,
                    "event_index": int(example["event_index"]),
                    "event_source_index": int(
                        record.event_source_index[example["event_index"]]
                    ),
                    "tau": int(example["tau"]),
                    "n_groups": int(record.group_count[example["event_index"]]),
                    "terminal": terminal,
                    "n_participants": int(np.sum(example["group_ids"] >= 0)),
                    "model_next_set_nll": model_nll,
                    "support_next_set_nll": support_nll,
                    "markov_next_set_nll": markov_nll,
                    "empirical_template_next_set_nll": template_nll,
                    "model_suffix_concordance": pairwise_rank_concordance(
                        utility[local_index, :n_contacts], suffix_group
                    ),
                    "support_suffix_concordance": pairwise_rank_concordance(
                        record.support, suffix_group
                    ),
                    "markov_suffix_concordance": pairwise_rank_concordance(
                        markov_score, suffix_group
                    ),
                    "empirical_template_suffix_concordance": pairwise_rank_concordance(
                        template_utility, suffix_group
                    ),
                    "stop_probability": float(stop_probability[local_index]),
                    "stop_correct": bool(
                        (stop_probability[local_index] >= 0.5) == terminal
                    ),
                    "remaining_brier": float(
                        np.mean(
                            (
                                remaining_probability[local_index, :n_contacts][valid]
                                - np.asarray(example["remaining"], float)[valid]
                            )
                            ** 2
                        )
                    )
                    if np.any(valid)
                    else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    nonterminal = frame[~frame.terminal]
    summary = {
        "subject": record.subject,
        "n_eval_events": int(len(record.eval_indices)),
        "n_prefix_examples": int(len(frame)),
        "n_nonterminal_examples": int(len(nonterminal)),
        "model_next_set_nll": float(nonterminal.model_next_set_nll.mean()),
        "support_next_set_nll": float(nonterminal.support_next_set_nll.mean()),
        "markov_next_set_nll": float(nonterminal.markov_next_set_nll.mean()),
        "empirical_template_next_set_nll": float(
            nonterminal.empirical_template_next_set_nll.mean()
        ),
        "model_suffix_concordance": float(frame.model_suffix_concordance.mean()),
        "support_suffix_concordance": float(frame.support_suffix_concordance.mean()),
        "markov_suffix_concordance": float(frame.markov_suffix_concordance.mean()),
        "empirical_template_suffix_concordance": float(
            frame.empirical_template_suffix_concordance.mean()
        ),
        "stop_accuracy": float(frame.stop_correct.mean()),
        "remaining_brier": float(frame.remaining_brier.mean()),
    }
    summary["model_minus_best_baseline_next_nll"] = float(
        summary["model_next_set_nll"]
        - min(
            summary["support_next_set_nll"],
            summary["markov_next_set_nll"],
            summary["empirical_template_next_set_nll"],
        )
    )
    summary["model_minus_best_baseline_suffix_concordance"] = float(
        summary["model_suffix_concordance"]
        - max(
            summary["support_suffix_concordance"],
            summary["markov_suffix_concordance"],
            summary["empirical_template_suffix_concordance"],
        )
    )
    return summary, frame


def _select_training_records(
    records: Mapping[str, SubjectRecord],
    heldout: str,
    max_train_subjects: Optional[int],
) -> list[SubjectRecord]:
    selected = [record for subject, record in records.items() if subject != heldout]
    if max_train_subjects is None or len(selected) <= int(max_train_subjects):
        return selected
    limit = int(max_train_subjects)
    by_dataset: Dict[str, list[SubjectRecord]] = {}
    for record in selected:
        by_dataset.setdefault(record.dataset, []).append(record)
    chosen = []
    datasets = sorted(by_dataset)
    cursor = {dataset: 0 for dataset in datasets}
    while len(chosen) < limit:
        progressed = False
        for dataset in datasets:
            pool = sorted(by_dataset[dataset], key=lambda record: record.subject)
            index = cursor[dataset]
            if index < len(pool) and len(chosen) < limit:
                chosen.append(pool[index])
                cursor[dataset] += 1
                progressed = True
        if not progressed:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_operator_static_readout.yaml",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", default="epilepsiae_1084")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--calibration-epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--calibration-steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-subjects", type=int, default=None)
    parser.add_argument("--include-rank-shuffle-control", action="store_true")
    parser.add_argument("--include-static-neural-controls", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir is not None and args.dataset_dir.is_absolute()
        else ROOT / args.dataset_dir
        if args.dataset_dir is not None
        else ROOT / cfg["outputs"]["dataset"]
    )
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    records = load_records(dataset_dir)
    if args.heldout_subject not in records:
        raise RuntimeError(f"heldout subject absent from dataset: {args.heldout_subject}")
    heldout = records[args.heldout_subject]
    train_records = _select_training_records(
        records, args.heldout_subject, args.max_train_subjects
    )
    if not train_records:
        raise RuntimeError("outer training pool is empty")

    _seed_everything(args.seed)
    device_name = args.device or cfg["resources"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    device_index = None
    if device.type == "cuda":
        fraction = float(cfg["resources"]["gpu_memory_fraction"])
        device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.set_num_threads(int(cfg["resources"]["cpu_threads"]))

    stage = cfg["stage_a"]
    epochs = int(
        args.epochs
        if args.epochs is not None
        else stage["smoke_epochs"]
        if args.smoke
        else stage["max_epochs"]
    )
    calibration_epochs = int(
        args.calibration_epochs
        if args.calibration_epochs is not None
        else min(3, stage["calibration_epochs"])
        if args.smoke
        else stage["calibration_epochs"]
    )
    steps = int(
        args.steps_per_epoch
        if args.steps_per_epoch is not None
        else min(32, stage["steps_per_epoch"])
        if args.smoke
        else stage["steps_per_epoch"]
    )
    calibration_steps = int(
        args.calibration_steps_per_epoch
        if args.calibration_steps_per_epoch is not None
        else min(16, stage["calibration_steps_per_epoch"])
        if args.smoke
        else stage["calibration_steps_per_epoch"]
    )
    batch_size = int(args.batch_size or stage["batch_size"])
    model_kwargs = {
        "contact_feature_dim": heldout.contact_features.shape[1],
        "hidden_size": int(args.hidden_size),
        "contact_embedding_dim": int(stage["contact_embedding_dim"]),
        "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
    }
    model = ContactQueryGRU(**model_kwargs)
    inner_examples = _validation_examples(train_records, max_per_subject=128 if args.smoke else 512)
    shared_state, shared_log = train_model(
        model,
        train_records,
        device=device,
        epochs=epochs,
        steps_per_epoch=steps,
        batch_size=batch_size,
        learning_rate=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
        gradient_clip=float(stage["gradient_clip"]),
        loss_weights=stage["loss_weights"],
        seed=args.seed,
        validation_examples=inner_examples,
        patience=None if args.smoke else int(stage["early_stopping_patience"]),
        phase="shared_initialization",
    )
    torch.save(
        {
            "model_state": shared_state,
            "model_kwargs": model_kwargs,
            "heldout_subject": args.heldout_subject,
            "seed": args.seed,
        },
        run_dir / "shared_checkpoint.pt",
    )

    # Fixed target-free patient calibration. The held-out last 20% is not
    # touched until this loop has ended and the checkpoint is locked.
    calibrated = ContactQueryGRU(**model_kwargs)
    calibrated.load_state_dict(shared_state)
    calibrated_state, calibration_log = train_model(
        calibrated,
        [heldout],
        device=device,
        epochs=calibration_epochs,
        steps_per_epoch=calibration_steps,
        batch_size=min(batch_size, 256),
        learning_rate=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
        gradient_clip=float(stage["gradient_clip"]),
        loss_weights=stage["loss_weights"],
        seed=args.seed + 100_000,
        validation_examples=None,
        patience=None,
        phase="heldout_target_free_calibration",
        restore_best=False,
    )
    torch.save(
        {
            "model_state": calibrated_state,
            "model_kwargs": model_kwargs,
            "heldout_subject": args.heldout_subject,
            "seed": args.seed,
            "heldout_eval_read_before_checkpoint_lock": False,
        },
        run_dir / "calibrated_checkpoint.pt",
    )
    summary, predictions = evaluate_heldout(calibrated, heldout, device)
    summary["control"] = "true_order_core"
    summary["n_parameters"] = int(
        sum(parameter.numel() for parameter in calibrated.parameters())
    )
    summaries = [summary]
    predictions["control"] = "true_order_core"
    prediction_frames = [predictions]

    if args.include_rank_shuffle_control:
        shuffled = {
            record.subject: shuffled_group_matrix(
                record.group_ids,
                seed=args.seed ^ (int(hashlib.sha256(record.subject.encode()).hexdigest()[:8], 16)),
            )
            for record in [*train_records, heldout]
        }
        shuffled_model = ContactQueryGRU(**model_kwargs)
        shuffled_state, shuffled_shared_log = train_model(
            shuffled_model,
            train_records,
            device=device,
            epochs=epochs,
            steps_per_epoch=steps,
            batch_size=batch_size,
            learning_rate=float(stage["learning_rate"]),
            weight_decay=float(stage["weight_decay"]),
            gradient_clip=float(stage["gradient_clip"]),
            loss_weights=stage["loss_weights"],
            seed=args.seed + 200_000,
            validation_examples=None,
            patience=None,
            shuffle_groups=shuffled,
            phase="rank_shuffle_shared",
        )
        shuffled_calibrated = ContactQueryGRU(**model_kwargs)
        shuffled_calibrated.load_state_dict(shuffled_state)
        shuffled_calibrated_state, shuffled_calibration_log = train_model(
            shuffled_calibrated,
            [heldout],
            device=device,
            epochs=calibration_epochs,
            steps_per_epoch=calibration_steps,
            batch_size=min(batch_size, 256),
            learning_rate=float(stage["learning_rate"]),
            weight_decay=float(stage["weight_decay"]),
            gradient_clip=float(stage["gradient_clip"]),
            loss_weights=stage["loss_weights"],
            seed=args.seed + 300_000,
            validation_examples=None,
            patience=None,
            shuffle_groups=shuffled,
            phase="rank_shuffle_heldout_calibration",
            restore_best=False,
        )
        torch.save(
            {
                "model_state": shuffled_calibrated_state,
                "model_kwargs": model_kwargs,
                "heldout_subject": args.heldout_subject,
                "seed": args.seed,
                "control": "participation_preserving_within_event_rank_shuffle",
            },
            run_dir / "rank_shuffle_checkpoint.pt",
        )
        shuffle_summary, shuffle_predictions = evaluate_heldout(
            shuffled_calibrated, heldout, device
        )
        shuffle_summary["control"] = "rank_shuffle_core"
        shuffle_summary["n_parameters"] = int(
            sum(parameter.numel() for parameter in shuffled_calibrated.parameters())
        )
        summaries.append(shuffle_summary)
        shuffle_predictions["control"] = "rank_shuffle_core"
        prediction_frames.append(shuffle_predictions)
        shared_log.extend(shuffled_shared_log)
        calibration_log.extend(shuffled_calibration_log)

    if args.include_static_neural_controls:
        for offset, (control_name, use_last_set) in enumerate(
            (
                ("unordered_deepsets", False),
                ("matched_feedforward_contact_query", True),
            ),
            start=1,
        ):
            static_model = StaticContactQuery(
                **model_kwargs, use_last_set=use_last_set
            )
            static_state, static_shared_log = train_model(
                static_model,
                train_records,
                device=device,
                epochs=epochs,
                steps_per_epoch=steps,
                batch_size=batch_size,
                learning_rate=float(stage["learning_rate"]),
                weight_decay=float(stage["weight_decay"]),
                gradient_clip=float(stage["gradient_clip"]),
                loss_weights=stage["loss_weights"],
                seed=args.seed + offset * 400_000,
                validation_examples=inner_examples,
                patience=None if args.smoke else int(stage["early_stopping_patience"]),
                phase=f"{control_name}_shared",
            )
            static_calibrated = StaticContactQuery(
                **model_kwargs, use_last_set=use_last_set
            )
            static_calibrated.load_state_dict(static_state)
            static_calibrated_state, static_calibration_log = train_model(
                static_calibrated,
                [heldout],
                device=device,
                epochs=calibration_epochs,
                steps_per_epoch=calibration_steps,
                batch_size=min(batch_size, 256),
                learning_rate=float(stage["learning_rate"]),
                weight_decay=float(stage["weight_decay"]),
                gradient_clip=float(stage["gradient_clip"]),
                loss_weights=stage["loss_weights"],
                seed=args.seed + offset * 500_000,
                validation_examples=None,
                patience=None,
                phase=f"{control_name}_heldout_calibration",
                restore_best=False,
            )
            torch.save(
                {
                    "model_state": static_calibrated_state,
                    "model_kwargs": model_kwargs,
                    "use_last_set": use_last_set,
                    "heldout_subject": args.heldout_subject,
                    "seed": args.seed,
                    "control": control_name,
                },
                run_dir / f"{control_name}_checkpoint.pt",
            )
            static_summary, static_predictions = evaluate_heldout(
                static_calibrated, heldout, device
            )
            static_summary["control"] = control_name
            static_summary["n_parameters"] = int(
                sum(parameter.numel() for parameter in static_calibrated.parameters())
            )
            summaries.append(static_summary)
            static_predictions["control"] = control_name
            prediction_frames.append(static_predictions)
            shared_log.extend(static_shared_log)
            calibration_log.extend(static_calibration_log)

    epoch_log = pd.DataFrame([*shared_log, *calibration_log])
    epoch_log.to_csv(run_dir / "epoch_log.csv", index=False)
    pd.DataFrame(summaries).to_csv(run_dir / "heldout_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        run_dir / "heldout_prefix_predictions.csv", index=False
    )
    input_fingerprint = {
        record.subject: record.input_sha256
        for record in [*train_records, heldout]
    }
    run_manifest = {
        "contract": cfg["contract"]["name"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "dataset_manifest_sha256": _sha256(dataset_dir / "dataset_manifest.json"),
        "input_subject_sha256": input_fingerprint,
        "heldout_subject": args.heldout_subject,
        "heldout_train_events": int(len(heldout.train_indices)),
        "heldout_eval_events": int(len(heldout.eval_indices)),
        "outer_training_subjects": [record.subject for record in train_records],
        "outer_training_dataset_counts": pd.Series(
            [record.dataset for record in train_records]
        )
        .value_counts()
        .astype(int)
        .to_dict(),
        "model_kwargs": model_kwargs,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "epochs": epochs,
        "calibration_epochs": calibration_epochs,
        "steps_per_epoch": steps,
        "calibration_steps_per_epoch": calibration_steps,
        "batch_size": batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_name": (
            torch.cuda.get_device_name(device_index) if device.type == "cuda" else None
        ),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_index) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "ictal_target_opened": False,
        "heldout_eval_read_before_calibration_checkpoint_lock": False,
        "controls": [summary["control"] for summary in summaries],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(_jsonable(run_manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    done = {
        "status": "engineering_smoke_complete" if args.smoke else "stage_a_cell_complete",
        "scientific_gate": "not_evaluated_formally" if args.smoke else "requires_patient_level_aggregation",
        "heldout_subject": args.heldout_subject,
        "seed": int(args.seed),
        "hidden_size": int(args.hidden_size),
        "n_controls": len(summaries),
        "peak_gpu_memory_mb": run_manifest["peak_gpu_memory_mb"],
        "ictal_target_opened": False,
        "finished_epoch": int(time.time()),
    }
    (run_dir / "DONE.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"done": done, "metrics": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
