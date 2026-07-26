#!/usr/bin/env python3
"""Train one v0.4 held-out-patient rank-distribution fold.

Stage A is target-sealed.  A shared contact-query model is learned from the
other patients, its core is frozen, and only a held-out patient's local
contact offsets are calibrated on that patient's chronological first 80%.
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
from typing import Dict, Mapping, Optional, Sequence

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
    raise SystemExit("PyTorch is required; use the cuda_env conda environment") from exc

from src.topic5_rank_distribution import (  # noqa: E402
    FullHistorySequenceGRU,
    StaticSequenceContactQuery,
    contact_rank_distribution,
    distribution_errors,
    next_set_stop_loss,
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
    input_sha256: str

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 0)

    @property
    def eval_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 1)


class EventQueue:
    """Without-replacement queue over one patient's event indices."""

    def __init__(self, indices: np.ndarray, rng: np.random.Generator):
        self.indices = np.asarray(indices, int)
        if not self.indices.size:
            raise ValueError("event queue cannot be empty")
        self.rng = rng
        self.order = self.rng.permutation(self.indices)
        self.cursor = 0
        self.cycles = 0
        self.drawn = 0

    def draw(self, size: int) -> np.ndarray:
        chunks = []
        remaining = int(size)
        while remaining:
            available = self.order.size - self.cursor
            take = min(available, remaining)
            chunks.append(self.order[self.cursor : self.cursor + take])
            self.cursor += take
            self.drawn += take
            remaining -= take
            if self.cursor == self.order.size:
                self.cycles += 1
                self.order = self.rng.permutation(self.indices)
                self.cursor = 0
        return np.concatenate(chunks)


class BalancedSubjectSampler:
    """Dataset-alternating, patient-without-replacement training sampler."""

    def __init__(
        self, records: Sequence[SubjectRecord], rng: np.random.Generator
    ):
        self.rng = rng
        self.pools: Dict[str, list[SubjectRecord]] = {}
        for record in records:
            self.pools.setdefault(record.dataset, []).append(record)
        self.datasets = sorted(self.pools)
        self.orders = {
            dataset: self.rng.permutation(len(pool))
            for dataset, pool in self.pools.items()
        }
        self.cursors = {dataset: 0 for dataset in self.datasets}

    def draw(self, step: int) -> SubjectRecord:
        dataset = self.datasets[int(step) % len(self.datasets)]
        cursor = self.cursors[dataset]
        if cursor == len(self.orders[dataset]):
            self.orders[dataset] = self.rng.permutation(len(self.pools[dataset]))
            cursor = 0
        record = self.pools[dataset][int(self.orders[dataset][cursor])]
        self.cursors[dataset] = cursor + 1
        return record


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_records(dataset_dir: Path) -> Dict[str, SubjectRecord]:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    if bool(manifest.get("target_values_read", True)):
        raise RuntimeError("dataset manifest does not certify sealed ictal targets")
    if int(manifest.get("n_subjects_ok", -1)) != 34:
        raise RuntimeError("v0.4 Stage-A dataset is not the frozen 34-patient cohort")
    audit = pd.read_csv(dataset_dir / "subject_audit.csv")
    records = {}
    for row in audit[audit.status.astype(str) == "ok"].itertuples():
        path = dataset_dir / "per_subject" / f"{row.subject}.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        expected = str(metadata["dataset_npz_sha256"])
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{row.subject}: input fingerprint mismatch")
        with np.load(path, allow_pickle=False) as z:
            record = SubjectRecord(
                subject=str(row.subject),
                dataset=str(row.dataset),
                path=path,
                contact_features=np.asarray(z["contact_features"], np.float32),
                contact_names=np.asarray(z["contact_names"]),
                group_ids=np.asarray(z["event_group_ids"], np.int16),
                group_count=np.asarray(z["event_group_count"], np.int16),
                event_split=np.asarray(z["event_split"], np.uint8),
                event_source_index=np.asarray(z["event_source_index"], np.int64),
                input_sha256=actual,
            )
        if record.group_ids.shape[0] != record.event_split.size:
            raise RuntimeError(f"{row.subject}: event arrays are misaligned")
        records[record.subject] = record
    if len(records) != 34:
        raise RuntimeError(f"expected 34 valid records, found {len(records)}")
    return records


def _shuffled_groups(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve participation while permuting within-event rank assignment."""
    out = np.asarray(groups, np.int16).copy()
    for event_index, event in enumerate(out):
        participant = np.flatnonzero(event >= 0)
        if participant.size:
            out[event_index, participant] = rng.permutation(event[participant])
    return out


def _batch(
    record: SubjectRecord,
    indices: np.ndarray,
    device: torch.device,
    *,
    rank_shuffle: bool,
    rng: np.random.Generator,
) -> dict:
    groups = record.group_ids[indices]
    if rank_shuffle:
        groups = _shuffled_groups(groups, rng)
    count = record.group_count[indices]
    batch_size = len(indices)
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(batch_size, -1, -1)
    contact_mask = torch.ones(
        (batch_size, record.contact_features.shape[0]),
        dtype=torch.bool,
        device=device,
    )
    return {
        "contact_features": features,
        "contact_mask": contact_mask,
        "group_ids": torch.as_tensor(groups, dtype=torch.long, device=device),
        "group_count": torch.as_tensor(count, dtype=torch.long, device=device),
    }


def _model(
    control: str,
    feature_dim: int,
    model_kwargs: Mapping[str, int],
):
    kwargs = dict(model_kwargs)
    if control in {"full_history_gru", "rank_shuffle_gru"}:
        return FullHistorySequenceGRU(feature_dim, **kwargs)
    mode = {
        "static_contact_hazard": "static",
        "unordered_prefix": "unordered",
        "last_set_first_order": "last_set",
    }.get(control)
    if mode is None:
        raise ValueError(f"unknown control: {control}")
    return StaticSequenceContactQuery(feature_dim, mode=mode, **kwargs)


def train_shared(
    model,
    records: Sequence[SubjectRecord],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    local_learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    local_offset_dim: int,
    device: torch.device,
    seed: int,
    rank_shuffle: bool,
    log_every: int = 16,
) -> tuple[dict, dict[str, torch.Tensor], list[dict], dict]:
    """Jointly learn a shared core and outer-patient nuisance offsets."""
    model.to(device)
    rng = np.random.default_rng(int(seed))
    subject_sampler = BalancedSubjectSampler(records, rng)
    queues = {
        record.subject: EventQueue(
            record.train_indices,
            np.random.default_rng(
                int(seed)
                ^ int(hashlib.sha256(record.subject.encode()).hexdigest()[:8], 16)
            ),
        )
        for record in records
    }
    offsets = {
        record.subject: torch.nn.Parameter(
            torch.zeros(
                (record.contact_features.shape[0], int(local_offset_dim)),
                dtype=torch.float32,
                device=device,
            )
        )
        for record in records
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            },
            {
                "params": list(offsets.values()),
                "lr": float(local_learning_rate),
                "weight_decay": float(weight_decay),
            },
        ]
    )
    rows = []
    start = time.time()
    window = []
    for step in range(int(steps)):
        record = subject_sampler.draw(step)
        indices = queues[record.subject].draw(
            min(int(batch_size), record.train_indices.size)
        )
        batch = _batch(
            record, indices, device, rank_shuffle=rank_shuffle, rng=rng
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, local_offset=offsets[record.subject])
        loss = next_set_stop_loss(
            outputs, batch["group_ids"], batch["group_count"]
        )
        loss["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), *offsets.values()], float(gradient_clip)
        )
        optimizer.step()
        value = float(loss["total"].detach().cpu())
        window.append(value)
        row = {
            "phase": "shared",
            "step": step + 1,
            "subject": record.subject,
            "dataset": record.dataset,
            "loss": value,
            "gradient_norm": float(grad_norm.detach().cpu()),
            "elapsed_seconds": time.time() - start,
        }
        rows.append(row)
        if (step + 1) % int(log_every) == 0 or step + 1 == int(steps):
            print(
                json.dumps(
                    {
                        "phase": "shared",
                        "step": step + 1,
                        "steps": int(steps),
                        "loss_window": float(np.mean(window)),
                        "elapsed_seconds": round(time.time() - start, 2),
                    }
                ),
                flush=True,
            )
            window.clear()
    coverage = {
        record.subject: {
            "events_available": int(record.train_indices.size),
            "drawn": int(queues[record.subject].drawn),
            "completed_cycles": int(queues[record.subject].cycles),
            "fraction_of_first_cycle": float(
                min(queues[record.subject].drawn, record.train_indices.size)
                / record.train_indices.size
            ),
        }
        for record in records
    }
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    offset_state = {
        subject: value.detach().cpu().clone()
        for subject, value in offsets.items()
    }
    return state, offset_state, rows, coverage


def _dataset_balanced_patient_order(
    records: Sequence[SubjectRecord],
    rng: np.random.Generator,
) -> list[SubjectRecord]:
    """Return every patient once, interleaving datasets when possible."""
    pools: Dict[str, list[SubjectRecord]] = {}
    for record in records:
        pools.setdefault(record.dataset, []).append(record)
    for dataset, pool in pools.items():
        order = rng.permutation(len(pool))
        pools[dataset] = [pool[int(index)] for index in order]
    datasets = sorted(pools)
    ordered = []
    for position in range(max(len(pool) for pool in pools.values())):
        for dataset in datasets:
            if position < len(pools[dataset]):
                ordered.append(pools[dataset][position])
    return ordered


def train_shared_coverage(
    model,
    records: Sequence[SubjectRecord],
    *,
    coverage_cycles: int,
    updates_per_patient: int,
    batch_size: int,
    learning_rate: float,
    local_learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    local_offset_dim: int,
    device: torch.device,
    seed: int,
    rank_shuffle: bool,
) -> tuple[dict, dict[str, torch.Tensor], list[dict], dict]:
    """Train with exact event coverage and equal optimizer updates per patient."""
    model.to(device)
    rng = np.random.default_rng(int(seed))
    offsets = {
        record.subject: torch.nn.Parameter(
            torch.zeros(
                (record.contact_features.shape[0], int(local_offset_dim)),
                dtype=torch.float32,
                device=device,
            )
        )
        for record in records
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            },
            {
                "params": list(offsets.values()),
                "lr": float(local_learning_rate),
                "weight_decay": float(weight_decay),
            },
        ]
    )
    rows = []
    global_update = 0
    start = time.time()
    for cycle in range(int(coverage_cycles)):
        patient_order = _dataset_balanced_patient_order(records, rng)
        for record in patient_order:
            indices = rng.permutation(record.train_indices)
            segments = [
                segment
                for segment in np.array_split(indices, int(updates_per_patient))
                if len(segment)
            ]
            for segment_index, segment in enumerate(segments):
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                for batch_start in range(0, len(segment), int(batch_size)):
                    chunk = segment[batch_start : batch_start + int(batch_size)]
                    batch = _batch(
                        record,
                        chunk,
                        device,
                        rank_shuffle=rank_shuffle,
                        rng=rng,
                    )
                    outputs = model(
                        **batch, local_offset=offsets[record.subject]
                    )
                    loss = next_set_stop_loss(
                        outputs, batch["group_ids"], batch["group_count"]
                    )
                    weight = len(chunk) / len(segment)
                    (loss["total"] * weight).backward()
                    weighted_loss += float(loss["total"].detach().cpu()) * weight
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [*model.parameters(), offsets[record.subject]],
                    float(gradient_clip),
                )
                optimizer.step()
                global_update += 1
                rows.append(
                    {
                        "phase": "shared_full_coverage",
                        "coverage_cycle": cycle + 1,
                        "patient_update": segment_index + 1,
                        "global_update": global_update,
                        "subject": record.subject,
                        "dataset": record.dataset,
                        "n_events": int(len(segment)),
                        "loss": weighted_loss,
                        "gradient_norm": float(grad_norm.detach().cpu()),
                        "elapsed_seconds": time.time() - start,
                    }
                )
            print(
                json.dumps(
                    {
                        "phase": "shared_full_coverage",
                        "cycle": cycle + 1,
                        "subject": record.subject,
                        "global_update": global_update,
                        "elapsed_seconds": round(time.time() - start, 2),
                    }
                ),
                flush=True,
            )
    coverage = {
        record.subject: {
            "events_available": int(record.train_indices.size),
            "drawn": int(record.train_indices.size * int(coverage_cycles)),
            "completed_cycles": int(coverage_cycles),
            "fraction_of_first_cycle": 1.0,
        }
        for record in records
    }
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    offset_state = {
        subject: value.detach().cpu().clone()
        for subject, value in offsets.items()
    }
    return state, offset_state, rows, coverage


def calibrate_offset(
    model,
    record: SubjectRecord,
    *,
    steps: int,
    batch_size: int,
    local_learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    local_offset_dim: int,
    device: torch.device,
    seed: int,
    rank_shuffle: bool,
    log_every: int = 16,
) -> tuple[torch.Tensor, list[dict], dict]:
    """Freeze the core and fit only held-out patient local contact offsets."""
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    offset = torch.nn.Parameter(
        torch.zeros(
            (record.contact_features.shape[0], int(local_offset_dim)),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        [offset],
        lr=float(local_learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    queue = EventQueue(record.train_indices, rng)
    rows = []
    start = time.time()
    window = []
    for step in range(int(steps)):
        indices = queue.draw(min(int(batch_size), record.train_indices.size))
        batch = _batch(
            record, indices, device, rank_shuffle=rank_shuffle, rng=rng
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, local_offset=offset)
        loss = next_set_stop_loss(
            outputs, batch["group_ids"], batch["group_count"]
        )
        loss["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([offset], float(gradient_clip))
        optimizer.step()
        value = float(loss["total"].detach().cpu())
        window.append(value)
        rows.append(
            {
                "phase": "heldout_offset_calibration",
                "step": step + 1,
                "subject": record.subject,
                "dataset": record.dataset,
                "loss": value,
                "gradient_norm": float(grad_norm.detach().cpu()),
                "elapsed_seconds": time.time() - start,
            }
        )
        if (step + 1) % int(log_every) == 0 or step + 1 == int(steps):
            print(
                json.dumps(
                    {
                        "phase": "calibration",
                        "step": step + 1,
                        "steps": int(steps),
                        "loss_window": float(np.mean(window)),
                        "elapsed_seconds": round(time.time() - start, 2),
                    }
                ),
                flush=True,
            )
            window.clear()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    coverage = {
        "events_available": int(record.train_indices.size),
        "drawn": int(queue.drawn),
        "completed_cycles": int(queue.cycles),
        "fraction_of_first_cycle": float(
            min(queue.drawn, record.train_indices.size)
            / record.train_indices.size
        ),
    }
    return offset.detach(), rows, coverage


def calibrate_offset_coverage(
    model,
    record: SubjectRecord,
    *,
    coverage_cycles: int,
    updates_per_cycle: int,
    batch_size: int,
    local_learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    local_offset_dim: int,
    device: torch.device,
    seed: int,
    rank_shuffle: bool,
) -> tuple[torch.Tensor, list[dict], dict]:
    """Fit held-out offsets with complete, event-balanced calibration cycles."""
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    offset = torch.nn.Parameter(
        torch.zeros(
            (record.contact_features.shape[0], int(local_offset_dim)),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        [offset],
        lr=float(local_learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    rows = []
    global_update = 0
    start = time.time()
    for cycle in range(int(coverage_cycles)):
        indices = rng.permutation(record.train_indices)
        segments = [
            segment
            for segment in np.array_split(indices, int(updates_per_cycle))
            if len(segment)
        ]
        for segment_index, segment in enumerate(segments):
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for batch_start in range(0, len(segment), int(batch_size)):
                chunk = segment[batch_start : batch_start + int(batch_size)]
                batch = _batch(
                    record,
                    chunk,
                    device,
                    rank_shuffle=rank_shuffle,
                    rng=rng,
                )
                outputs = model(**batch, local_offset=offset)
                loss = next_set_stop_loss(
                    outputs, batch["group_ids"], batch["group_count"]
                )
                weight = len(chunk) / len(segment)
                (loss["total"] * weight).backward()
                weighted_loss += float(loss["total"].detach().cpu()) * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [offset], float(gradient_clip)
            )
            optimizer.step()
            global_update += 1
            rows.append(
                {
                    "phase": "heldout_offset_full_coverage",
                    "coverage_cycle": cycle + 1,
                    "patient_update": segment_index + 1,
                    "global_update": global_update,
                    "subject": record.subject,
                    "dataset": record.dataset,
                    "n_events": int(len(segment)),
                    "loss": weighted_loss,
                    "gradient_norm": float(grad_norm.detach().cpu()),
                    "elapsed_seconds": time.time() - start,
                }
            )
        print(
            json.dumps(
                {
                    "phase": "heldout_offset_full_coverage",
                    "cycle": cycle + 1,
                    "subject": record.subject,
                    "global_update": global_update,
                    "elapsed_seconds": round(time.time() - start, 2),
                }
            ),
            flush=True,
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    coverage = {
        "events_available": int(record.train_indices.size),
        "drawn": int(record.train_indices.size * int(coverage_cycles)),
        "completed_cycles": int(coverage_cycles),
        "fraction_of_first_cycle": 1.0,
    }
    return offset.detach(), rows, coverage


@torch.no_grad()
def evaluate_model(
    model,
    record: SubjectRecord,
    offset: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    max_events: Optional[int],
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    model.eval()
    indices = record.eval_indices
    if max_events is not None and indices.size > int(max_events):
        take = np.linspace(0, indices.size - 1, int(max_events)).round().astype(int)
        indices = indices[np.unique(take)]
    rows = []
    stop_probabilities = []
    stop_targets = []
    top1 = []
    for start in range(0, indices.size, int(batch_size)):
        chunk = indices[start : start + int(batch_size)]
        batch = _batch(
            record,
            chunk,
            device,
            rank_shuffle=False,
            rng=np.random.default_rng(0),
        )
        outputs = model(**batch, local_offset=offset)
        loss = next_set_stop_loss(
            outputs, batch["group_ids"], batch["group_count"]
        )
        event_nll = loss["event_nll"].detach().cpu().numpy()
        contact_logits = outputs["contact_logits"].detach().cpu().numpy()
        stop_logits = outputs["stop_logits"].detach().cpu().numpy()
        groups = record.group_ids[chunk]
        counts = record.group_count[chunk]
        for local, event_index in enumerate(chunk):
            rows.append(
                {
                    "subject": record.subject,
                    "event_index": int(event_index),
                    "event_source_index": int(
                        record.event_source_index[event_index]
                    ),
                    "event_nll": float(event_nll[local]),
                    "n_groups": int(counts[local]),
                    "n_participants": int(np.sum(groups[local] >= 0)),
                }
            )
            for step in range(int(counts[local]) + 1):
                stop_probability = float(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(stop_logits[local, step], -60.0, 60.0)
                        )
                    )
                )
                terminal = step == int(counts[local])
                stop_probabilities.append(stop_probability)
                stop_targets.append(float(terminal))
                if not terminal:
                    contact_argmax = int(
                        np.argmax(contact_logits[local, step])
                    )
                    combined_stop_wins = (
                        stop_logits[local, step]
                        >= contact_logits[local, step, contact_argmax]
                    )
                    top1.append(
                        float(
                            (not combined_stop_wins)
                            and groups[local, contact_argmax] == step
                        )
                    )
    frame = pd.DataFrame(rows)
    stop_probability = np.asarray(stop_probabilities)
    stop_target = np.asarray(stop_targets)
    summary = {
        "heldout_event_nll": float(frame.event_nll.mean()),
        "n_eval_events": int(len(frame)),
        "top1_next_set_accuracy": float(np.mean(top1)),
        "stop_brier": float(np.mean((stop_probability - stop_target) ** 2)),
        "stop_accuracy": float(
            np.mean((stop_probability >= 0.5) == stop_target.astype(bool))
        ),
        "terminal_stop_probability": float(
            np.mean(stop_probability[stop_target == 1])
        ),
        "nonterminal_stop_probability": float(
            np.mean(stop_probability[stop_target == 0])
        ),
    }
    return summary, frame, indices


def _distribution_frame(
    record: SubjectRecord,
    control: str,
    generated_groups: np.ndarray,
    generated_count: np.ndarray,
    observed_groups: np.ndarray,
    observed_count: np.ndarray,
    bins: int,
) -> pd.DataFrame:
    generated = contact_rank_distribution(
        generated_groups, generated_count, bins=bins
    )
    observed = contact_rank_distribution(
        observed_groups, observed_count, bins=bins
    )
    rows = []
    for contact, name in enumerate(record.contact_names):
        row = {
            "subject": record.subject,
            "control": control,
            "contact_index": contact,
            "contact_name": str(name),
            "predicted_participation": float(
                generated["participation_probability"][contact]
            ),
            "observed_participation": float(
                observed["participation_probability"][contact]
            ),
            "predicted_mean_rank": float(generated["mean_rank"][contact]),
            "observed_mean_rank": float(observed["mean_rank"][contact]),
            "predicted_rank_variance": float(generated["rank_variance"][contact]),
            "observed_rank_variance": float(observed["rank_variance"][contact]),
        }
        for bin_index in range(int(bins)):
            row[f"predicted_rank_bin_{bin_index}"] = float(
                generated["rank_histogram"][contact, bin_index]
            )
            row[f"observed_rank_bin_{bin_index}"] = float(
                observed["rank_histogram"][contact, bin_index]
            )
        for quantile_index, label in enumerate(("q10", "q50", "q90")):
            row[f"predicted_rank_{label}"] = float(
                generated["rank_quantiles"][contact, quantile_index]
            )
            row[f"observed_rank_{label}"] = float(
                observed["rank_quantiles"][contact, quantile_index]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--hidden-size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--local-learning-rate", type=float, default=None)
    parser.add_argument("--local-offset-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--shared-steps", type=int, default=None)
    parser.add_argument("--calibration-steps", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--max-eval-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--formal-coverage", action="store_true")
    parser.add_argument("--coverage-shared-cycles", type=int, default=1)
    parser.add_argument("--coverage-calibration-cycles", type=int, default=4)
    parser.add_argument("--coverage-updates-per-patient", type=int, default=8)
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
        raise RuntimeError(f"held-out subject is absent: {args.heldout_subject}")
    heldout = records[args.heldout_subject]
    outer = [
        record for subject, record in records.items()
        if subject != args.heldout_subject
    ]

    _seed_everything(args.seed)
    device = torch.device(args.device or cfg["resources"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["resources"]["gpu_memory_fraction_per_process"]),
            device=device_index,
        )
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.set_num_threads(int(cfg["resources"]["cpu_threads_per_process"]))

    stage = cfg["stage_a"]
    shared_steps = int(
        args.shared_steps
        if args.shared_steps is not None
        else stage["pilot_shared_steps"]
        if args.pilot
        else stage["screen_shared_steps"]
    )
    calibration_steps = int(
        args.calibration_steps
        if args.calibration_steps is not None
        else stage["pilot_calibration_steps"]
        if args.pilot
        else stage["screen_calibration_steps"]
    )
    n_rollouts = int(
        args.rollouts
        if args.rollouts is not None
        else stage["pilot_rollouts"]
        if args.pilot
        else stage["formal_rollouts"]
    )
    max_eval_events = (
        args.max_eval_events
        if args.max_eval_events is not None
        else 1000
        if args.pilot
        else None
    )
    batch_size = int(args.batch_size or stage["batch_events"])
    bins = int(cfg["event_encoding"]["rank_distribution_bins"])
    model_kwargs = {
        "hidden_size": int(args.hidden_size),
        "contact_embedding_dim": int(stage["contact_embedding_dim"]),
        "contact_encoder_hidden": int(stage["contact_encoder_hidden"]),
        "local_offset_dim": int(
            args.local_offset_dim
            if args.local_offset_dim is not None
            else stage["local_offset_dim"]
        ),
    }
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else stage["learning_rate"]
    )
    local_learning_rate = float(
        args.local_learning_rate
        if args.local_learning_rate is not None
        else stage["local_learning_rate"]
    )
    controls = (
        "full_history_gru",
        "static_contact_hazard",
        "unordered_prefix",
        "last_set_first_order",
        "rank_shuffle_gru",
    )
    all_metrics = []
    all_event_nll = []
    all_contact_distributions = []
    all_training_logs = []
    coverage_contract = {}
    input_fingerprints = {
        record.subject: record.input_sha256 for record in records.values()
    }

    print(
        json.dumps(
            {
                "status": "RUNNING",
                "heldout_subject": heldout.subject,
                "n_outer_subjects": len(outer),
                "shared_steps": shared_steps,
                "calibration_steps": calibration_steps,
                "n_rollouts": n_rollouts,
                "device": str(device),
            }
        ),
        flush=True,
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "heldout_subject": heldout.subject,
                "seed": int(args.seed),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    for control_index, control in enumerate(controls):
        control_seed = int(args.seed + control_index * 1_000_003)
        rank_shuffle = control == "rank_shuffle_gru"
        print(
            json.dumps(
                {"control": control, "status": "training", "seed": control_seed}
            ),
            flush=True,
        )
        model = _model(
            control, heldout.contact_features.shape[1], model_kwargs
        )
        if args.formal_coverage:
            shared_state, _, shared_log, shared_coverage = train_shared_coverage(
                model,
                outer,
                coverage_cycles=int(args.coverage_shared_cycles),
                updates_per_patient=int(args.coverage_updates_per_patient),
                batch_size=batch_size,
                learning_rate=learning_rate,
                local_learning_rate=local_learning_rate,
                weight_decay=float(stage["weight_decay"]),
                gradient_clip=float(stage["gradient_clip"]),
                local_offset_dim=int(model_kwargs["local_offset_dim"]),
                device=device,
                seed=control_seed,
                rank_shuffle=rank_shuffle,
            )
        else:
            shared_state, _, shared_log, shared_coverage = train_shared(
                model,
                outer,
                steps=shared_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                local_learning_rate=local_learning_rate,
                weight_decay=float(stage["weight_decay"]),
                gradient_clip=float(stage["gradient_clip"]),
                local_offset_dim=int(model_kwargs["local_offset_dim"]),
                device=device,
                seed=control_seed,
                rank_shuffle=rank_shuffle,
            )
        model.load_state_dict(shared_state)
        if args.formal_coverage:
            offset, calibration_log, calibration_coverage = (
                calibrate_offset_coverage(
                    model,
                    heldout,
                    coverage_cycles=int(args.coverage_calibration_cycles),
                    updates_per_cycle=int(args.coverage_updates_per_patient),
                    batch_size=batch_size,
                    local_learning_rate=local_learning_rate,
                    weight_decay=float(stage["weight_decay"]),
                    gradient_clip=float(stage["gradient_clip"]),
                    local_offset_dim=int(model_kwargs["local_offset_dim"]),
                    device=device,
                    seed=control_seed + 500_000,
                    rank_shuffle=rank_shuffle,
                )
            )
        else:
            offset, calibration_log, calibration_coverage = calibrate_offset(
                model,
                heldout,
                steps=calibration_steps,
                batch_size=batch_size,
                local_learning_rate=local_learning_rate,
                weight_decay=float(stage["weight_decay"]),
                gradient_clip=float(stage["gradient_clip"]),
                local_offset_dim=int(model_kwargs["local_offset_dim"]),
                device=device,
                seed=control_seed + 500_000,
                rank_shuffle=rank_shuffle,
            )
        torch.save(
            {
                "contract": cfg["contract"],
                "control": control,
                "model_kwargs": model_kwargs,
                "model_state": shared_state,
                "heldout_local_offset": offset.cpu(),
                "heldout_subject": heldout.subject,
                "seed": int(args.seed),
                "ictal_target_read": False,
            },
            run_dir / f"{control}_checkpoint.pt",
        )
        metrics, event_frame, eval_indices = evaluate_model(
            model,
            heldout,
            offset.to(device),
            device=device,
            batch_size=min(batch_size, 256),
            max_events=max_eval_events,
        )
        feature = torch.as_tensor(
            heldout.contact_features, dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask = torch.ones(
            (1, heldout.contact_features.shape[0]),
            dtype=torch.bool,
            device=device,
        )
        generated_groups, generated_count = model.rollout(
            feature,
            mask,
            offset.to(device),
            n_events=n_rollouts,
            seed=control_seed + 700_000,
        )
        observed_groups = heldout.group_ids[eval_indices]
        observed_count = heldout.group_count[eval_indices]
        distribution = distribution_errors(
            generated_groups,
            generated_count,
            observed_groups,
            observed_count,
            bins=bins,
        )
        row = {
            "subject": heldout.subject,
            "dataset": heldout.dataset,
            "control": control,
            "seed": int(args.seed),
            "hidden_size": int(args.hidden_size),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "n_local_offset_parameters": int(offset.numel()),
            "rollout_participant_count_mean": float(np.mean(generated_count)),
            "rollout_participant_count_sd": float(np.std(generated_count)),
            **metrics,
            **distribution,
        }
        all_metrics.append(row)
        event_frame["control"] = control
        all_event_nll.append(event_frame)
        all_contact_distributions.append(
            _distribution_frame(
                heldout,
                control,
                generated_groups,
                generated_count,
                observed_groups,
                observed_count,
                bins,
            )
        )
        np.savez_compressed(
            run_dir / f"{control}_free_rollouts.npz",
            event_group_ids=generated_groups,
            event_group_count=generated_count,
            seed=np.asarray(control_seed + 700_000),
        )
        for row_log in [*shared_log, *calibration_log]:
            row_log["control"] = control
            all_training_logs.append(row_log)
        coverage_contract[control] = {
            "shared": shared_coverage,
            "heldout_calibration": calibration_coverage,
        }
        print(
            json.dumps(
                {
                    "control": control,
                    "status": "evaluated",
                    "heldout_event_nll": metrics["heldout_event_nll"],
                    "participation_mae": distribution["participation_mae"],
                    "rank_wasserstein": distribution["rank_wasserstein"],
                    "rollout_participant_count_mean": row[
                        "rollout_participant_count_mean"
                    ],
                }
            ),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eval_indices = (
        heldout.eval_indices
        if max_eval_events is None or heldout.eval_indices.size <= int(max_eval_events)
        else heldout.eval_indices[
            np.unique(
                np.linspace(
                    0,
                    heldout.eval_indices.size - 1,
                    int(max_eval_events),
                ).round().astype(int)
            )
        ]
    )
    empirical_groups = heldout.group_ids[heldout.train_indices]
    empirical_count = heldout.group_count[heldout.train_indices]
    observed_groups = heldout.group_ids[eval_indices]
    observed_count = heldout.group_count[eval_indices]
    empirical_error = distribution_errors(
        empirical_groups,
        empirical_count,
        observed_groups,
        observed_count,
        bins=bins,
    )
    split_at = max(1, len(empirical_groups) // 2)
    split_half_error = distribution_errors(
        empirical_groups[:split_at],
        empirical_count[:split_at],
        empirical_groups[split_at:],
        empirical_count[split_at:],
        bins=bins,
    )
    all_metrics.append(
        {
            "subject": heldout.subject,
            "dataset": heldout.dataset,
            "control": "empirical_rank_distribution",
            "seed": int(args.seed),
            "hidden_size": int(args.hidden_size),
            "n_parameters": 0,
            "n_local_offset_parameters": 0,
            "rollout_participant_count_mean": float(
                np.mean(np.sum(empirical_groups >= 0, axis=1))
            ),
            "rollout_participant_count_sd": float(
                np.std(np.sum(empirical_groups >= 0, axis=1))
            ),
            "heldout_event_nll": np.nan,
            "n_eval_events": int(len(eval_indices)),
            "top1_next_set_accuracy": np.nan,
            "stop_brier": np.nan,
            "stop_accuracy": np.nan,
            "terminal_stop_probability": np.nan,
            "nonterminal_stop_probability": np.nan,
            **empirical_error,
        }
    )
    all_contact_distributions.append(
        _distribution_frame(
            heldout,
            "empirical_rank_distribution",
            empirical_groups,
            empirical_count,
            observed_groups,
            observed_count,
            bins,
        )
    )

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(run_dir / "heldout_metrics.csv", index=False)
    pd.concat(all_event_nll, ignore_index=True).to_csv(
        run_dir / "heldout_event_nll.csv", index=False
    )
    pd.concat(all_contact_distributions, ignore_index=True).to_csv(
        run_dir / "contact_rank_distributions.csv", index=False
    )
    pd.DataFrame(all_training_logs).to_csv(
        run_dir / "training_log.csv", index=False
    )

    nll = metrics_frame.set_index("control").heldout_event_nll
    strongest_nonrecurrent = float(
        nll[
            [
                "static_contact_hazard",
                "unordered_prefix",
                "last_set_first_order",
            ]
        ].min()
    )
    gru_nll = float(nll["full_history_gru"])
    main_row = metrics_frame[
        metrics_frame.control == "full_history_gru"
    ].iloc[0]
    required_controls = metrics_frame[
        metrics_frame.control != "empirical_rank_distribution"
    ]
    engineering_pass = bool(
        np.all(np.isfinite(required_controls["heldout_event_nll"]))
        and np.all(np.isfinite(required_controls["participation_mae"]))
        and np.all(np.isfinite(required_controls["rank_wasserstein"]))
        and np.all(
            required_controls["rollout_participant_count_mean"].to_numpy()
            > 0.25
        )
        and np.all(
            required_controls["rollout_participant_count_mean"].to_numpy()
            < heldout.contact_features.shape[0]
        )
    )
    resource = {
        "cpu_threads": int(torch.get_num_threads()),
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "gpu_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
        ),
    }
    summary = {
        "status": "complete" if engineering_pass else "engineering_gate_failed",
        "contract": cfg["contract"],
        "heldout_subject": heldout.subject,
        "dataset": heldout.dataset,
        "pilot": bool(args.pilot),
        "seed": int(args.seed),
        "hidden_size": int(args.hidden_size),
        "learning_rate": learning_rate,
        "local_learning_rate": local_learning_rate,
        "local_offset_dim": int(model_kwargs["local_offset_dim"]),
        "n_outer_subjects": len(outer),
        "n_train_calibration_events": int(heldout.train_indices.size),
        "n_eval_events_available": int(heldout.eval_indices.size),
        "n_eval_events_used": int(len(eval_indices)),
        "shared_steps": shared_steps,
        "calibration_steps": calibration_steps,
        "n_rollouts": n_rollouts,
        "formal_coverage": bool(args.formal_coverage),
        "coverage_shared_cycles": (
            int(args.coverage_shared_cycles) if args.formal_coverage else 0
        ),
        "coverage_calibration_cycles": (
            int(args.coverage_calibration_cycles) if args.formal_coverage else 0
        ),
        "full_history_gru_nll": gru_nll,
        "strongest_nonrecurrent_nll": strongest_nonrecurrent,
        "ordered_history_nll_gain": strongest_nonrecurrent - gru_nll,
        "empirical_split_half_variability": split_half_error,
        "distribution_noninferiority_margin_rank_wasserstein": float(
            split_half_error["rank_wasserstein"]
        ),
        "engineering_pass": engineering_pass,
        "ictal_target_read": False,
        "coverage": coverage_contract,
        "resources": resource,
        "input_fingerprints": input_fingerprints,
        "config_sha256": _sha256(config_path),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True)
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )
    (run_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "heldout_subject": heldout.subject,
                "ictal_target_read": False,
                "engineering_pass": engineering_pass,
            },
            indent=2,
        )
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "heldout_subject": heldout.subject,
                "seed": int(args.seed),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(_jsonable(summary)), flush=True)


if __name__ == "__main__":
    main()
