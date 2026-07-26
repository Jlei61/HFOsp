#!/usr/bin/env python3
"""Train one target-sealed LOSO fold of the persistent path-mode RNN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from scripts.build_topic5_transition_skeleton_prior import _blend_graph
from scripts.train_topic5_axis_graph_rnn import (
    _axis_path_metrics,
    _whole_path_distance,
)
from scripts.train_topic5_interictal_rank_distribution import (
    BalancedSubjectSampler,
    EventQueue,
    SubjectRecord,
    _jsonable,
    _seed_everything,
    load_records,
)
from src.topic5_persistent_path_rnn import (
    PersistentPathModeRNN,
    persistent_mixture_loss,
)
from src.topic5_rank_distribution import distribution_errors


@dataclass(frozen=True)
class PathModePrior:
    subject: str
    axis: np.ndarray
    component_graphs: np.ndarray
    component_prior: np.ndarray
    component_mode: np.ndarray
    component_direction: np.ndarray
    aggregate_forward: np.ndarray
    aggregate_reverse: np.ndarray
    left: np.ndarray
    right: np.ndarray
    source_sha256: str
    control: str
    mode_count: int
    use_recurrence: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_seed(subject: str, seed: int, suffix: str) -> int:
    return int(
        hashlib.sha256(f"{subject}:{seed}:{suffix}".encode()).hexdigest()[:8],
        16,
    )


def _shuffle_mode_graphs(
    forward_graphs: np.ndarray,
    mode_prior: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Destroy cross-edge mode coherence while preserving the mean graph.

    Each edge is independently reassigned across modes. A per-edge rescaling
    then preserves its prior-weighted mean exactly, so this null changes only
    which edges travel together as one persistent event mode.
    """
    rng = np.random.default_rng(int(seed))
    original = np.asarray(forward_graphs, np.float64)
    prior = np.asarray(mode_prior, np.float64)
    prior = prior / prior.sum()
    shuffled = original.copy()
    for target in range(shuffled.shape[1]):
        for source in range(shuffled.shape[2]):
            shuffled[:, target, source] = rng.permutation(
                shuffled[:, target, source]
            )
            expected = float(prior @ original[:, target, source])
            actual = float(prior @ shuffled[:, target, source])
            if expected <= 1e-12:
                shuffled[:, target, source] = 0.0
            elif actual > 1e-12:
                shuffled[:, target, source] *= expected / actual
            else:  # Defensive only: a permutation cannot create this case.
                shuffled[:, target, source] = expected
    return shuffled.astype(np.float32)


def _shuffle_weights_within_mode(
    raw_modes: np.ndarray,
    axis: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Shuffle canonical edge weights at fixed per-mode density."""
    rng = np.random.default_rng(int(seed))
    shuffled = np.zeros_like(raw_modes)
    increasing = np.argwhere(
        (axis[:, None] > axis[None, :])
        & ~np.eye(len(axis), dtype=bool)
    )
    tied = np.argwhere(
        np.isclose(axis[:, None], axis[None, :])
        & ~np.eye(len(axis), dtype=bool)
    )
    for mode in range(len(raw_modes)):
        values = raw_modes[
            mode, increasing[:, 0], increasing[:, 1]
        ].copy()
        rng.shuffle(values)
        shuffled[mode, increasing[:, 0], increasing[:, 1]] = values
        if len(tied):
            values = raw_modes[
                mode, tied[:, 0], tied[:, 1]
            ].copy()
            rng.shuffle(values)
            shuffled[mode, tied[:, 0], tied[:, 1]] = values
    return shuffled


def _component_graphs(
    forward: np.ndarray,
    reverse: np.ndarray,
    mode_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode_count = len(mode_prior)
    graphs = np.concatenate([forward, reverse], axis=0).astype(np.float32)
    prior = np.concatenate([mode_prior, mode_prior]).astype(np.float32) * 0.5
    mode = np.concatenate(
        [np.arange(mode_count), np.arange(mode_count)]
    ).astype(np.int16)
    direction = np.concatenate(
        [np.ones(mode_count), -np.ones(mode_count)]
    ).astype(np.int8)
    return graphs, prior, mode, direction


def load_path_mode_priors(
    prior_root: Path,
    records: Dict[str, SubjectRecord],
    *,
    mode_count: int,
    control: str,
    seed: int,
    axis_floor: float,
    neighbors: int,
) -> Dict[str, PathModePrior]:
    valid_controls = {
        "no_history",
        "merged_path",
        "intact",
        "weight_shuffle",
        "mode_shuffle",
    }
    if control not in valid_controls:
        raise ValueError(f"unknown control: {control}")
    if control == "no_history" and mode_count != 0:
        raise ValueError("no_history requires mode_count=0")
    if control == "merged_path" and mode_count != 1:
        raise ValueError("merged_path requires mode_count=1")
    if control in {"intact", "weight_shuffle", "mode_shuffle"}:
        if mode_count not in {1, 2, 3, 4}:
            raise ValueError("structured controls require K in [1, 4]")
    if control == "mode_shuffle" and mode_count < 2:
        raise ValueError("mode_shuffle requires at least two path modes")
    priors = {}
    source_k = max(1, int(mode_count))
    for subject, record in records.items():
        path = prior_root / f"k_{source_k}" / "per_subject" / f"{subject}.npz"
        if not path.exists():
            raise RuntimeError(f"{subject}: missing K={source_k} path prior")
        with np.load(path, allow_pickle=False) as z:
            if str(z["source_event_split"]) != "chronological_train80_only":
                raise RuntimeError(f"{subject}: prior is not train80-only")
            if bool(z["heldout_used_for_construction"]):
                raise RuntimeError(f"{subject}: heldout entered path prior")
            if bool(z["ab_labels_used"]) or bool(z["iei_used"]):
                raise RuntimeError(f"{subject}: forbidden input entered prior")
            if bool(z["ictal_target_read"]):
                raise RuntimeError(f"{subject}: ictal target entered prior")
            if str(z["input_record_sha256"]) != record.input_sha256:
                raise RuntimeError(f"{subject}: record fingerprint mismatch")
            if not np.array_equal(
                np.asarray(z["contact_names"]).astype(str),
                record.contact_names.astype(str),
            ):
                raise RuntimeError(f"{subject}: contact order mismatch")
            axis = np.asarray(z["axis_coordinate"], np.float32)
            raw_modes = np.asarray(z["mode_skeleton_raw"], np.float32)
            forward = np.asarray(z["mode_forward_graphs"], np.float32)
            reverse = np.asarray(z["mode_reverse_graphs"], np.float32)
            mode_prior = np.asarray(z["mode_prior"], np.float32)
            aggregate_forward = np.asarray(
                z["aggregate_forward_graph"], np.float32
            )
            aggregate_reverse = np.asarray(
                z["aggregate_reverse_graph"], np.float32
            )
            left = np.asarray(z["left_endpoint"], bool)
            right = np.asarray(z["right_endpoint"], bool)
        if control == "no_history":
            graphs = np.zeros(
                (1, len(axis), len(axis)), dtype=np.float32
            )
            prior = np.ones(1, np.float32)
            component_mode = np.zeros(1, np.int16)
            direction = np.zeros(1, np.int8)
            use_recurrence = False
        elif control == "merged_path":
            graphs = (
                0.5 * (aggregate_forward + aggregate_reverse)
            )[None].astype(np.float32)
            prior = np.ones(1, np.float32)
            component_mode = np.zeros(1, np.int16)
            direction = np.zeros(1, np.int8)
            use_recurrence = True
        else:
            if control == "weight_shuffle":
                raw_modes = _shuffle_weights_within_mode(
                    raw_modes,
                    axis,
                    seed=_subject_seed(subject, seed, control),
                )
                forward = []
                reverse = []
                for raw in raw_modes:
                    fwd, rev, _ = _blend_graph(
                        raw,
                        axis,
                        axis_floor=float(axis_floor),
                        neighbors=int(neighbors),
                    )
                    forward.append(fwd)
                    reverse.append(rev)
                forward = np.stack(forward)
                reverse = np.stack(reverse)
            elif control == "mode_shuffle":
                forward = _shuffle_mode_graphs(
                    forward,
                    mode_prior,
                    seed=_subject_seed(subject, seed, control),
                )
                reverse = np.transpose(forward, (0, 2, 1)).copy()
            graphs, prior, component_mode, direction = _component_graphs(
                np.asarray(forward, np.float32),
                np.asarray(reverse, np.float32),
                mode_prior,
            )
            use_recurrence = True
        priors[subject] = PathModePrior(
            subject=subject,
            axis=axis,
            component_graphs=graphs,
            component_prior=prior / prior.sum(),
            component_mode=component_mode,
            component_direction=direction,
            aggregate_forward=aggregate_forward,
            aggregate_reverse=aggregate_reverse,
            left=left,
            right=right,
            source_sha256=_sha256(path),
            control=control,
            mode_count=int(mode_count),
            use_recurrence=use_recurrence,
        )
    if len(priors) != 34:
        raise RuntimeError(f"expected 34 priors, found {len(priors)}")
    return priors


def select_outer_records(
    records: Dict[str, SubjectRecord], heldout_subject: str
) -> list[SubjectRecord]:
    """Return each non-held-out patient exactly once."""
    outer = [
        record
        for subject, record in records.items()
        if subject != heldout_subject
    ]
    if len(outer) != len(records) - 1:
        raise RuntimeError("LOSO outer cohort is not exactly N-1")
    if len({record.subject for record in outer}) != len(outer):
        raise RuntimeError("LOSO outer cohort contains duplicate patients")
    return outer


def _batch(
    record: SubjectRecord,
    prior: PathModePrior,
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    n_events = len(indices)
    n_contacts = len(record.contact_names)
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
        "component_graphs": torch.as_tensor(
            prior.component_graphs, dtype=torch.float32, device=device
        ),
        "component_prior": torch.as_tensor(
            prior.component_prior, dtype=torch.float32, device=device
        ),
        "left_endpoint": torch.as_tensor(
            prior.left, dtype=torch.bool, device=device
        ),
        "right_endpoint": torch.as_tensor(
            prior.right, dtype=torch.bool, device=device
        ),
    }


def _loss(
    model: PersistentPathModeRNN,
    batch: dict,
    offset: torch.Tensor,
    cfg: dict,
) -> dict:
    output = model(**batch, local_offset=offset)
    return persistent_mixture_loss(
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
    model: PersistentPathModeRNN,
    records: Sequence[SubjectRecord],
    priors: Dict[str, PathModePrior],
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
                (len(record.contact_names), offset_dim),
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
                "marginal_sequence_loss": float(
                    loss["next_set_stop"].detach().cpu()
                ),
                "stop_calibration_loss": float(
                    loss["stop_calibration"].detach().cpu()
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


def _dataset_balanced_patient_order(
    records: Sequence[SubjectRecord],
    rng: np.random.Generator,
) -> list[SubjectRecord]:
    """Return every patient once while interleaving datasets when possible."""
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
    model: PersistentPathModeRNN,
    records: Sequence[SubjectRecord],
    priors: Dict[str, PathModePrior],
    cfg: dict,
    *,
    coverage_cycles: int,
    updates_per_patient: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict, list[dict], dict]:
    """Train with exact event coverage and equal updates per patient."""
    if int(coverage_cycles) < 1 or int(updates_per_patient) < 1:
        raise ValueError("coverage cycles and updates per patient must be positive")
    model.to(device)
    model.train()
    rng = np.random.default_rng(int(seed))
    offset_dim = int(cfg["model"]["local_offset_dim"])
    offsets = {
        record.subject: torch.nn.Parameter(
            torch.zeros(
                (len(record.contact_names), offset_dim),
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
    global_update = 0
    started = time.time()
    for cycle in range(int(coverage_cycles)):
        for record in _dataset_balanced_patient_order(records, rng):
            indices = rng.permutation(record.train_indices)
            segments = [
                segment
                for segment in np.array_split(indices, int(updates_per_patient))
                if len(segment)
            ]
            for segment_index, segment in enumerate(segments):
                optimizer.zero_grad(set_to_none=True)
                weighted = {
                    "total": 0.0,
                    "next_set_stop": 0.0,
                    "stop_calibration": 0.0,
                }
                for batch_start in range(0, len(segment), int(batch_size)):
                    chunk = segment[
                        batch_start : batch_start + int(batch_size)
                    ]
                    batch = _batch(
                        record, priors[record.subject], chunk, device
                    )
                    loss = _loss(
                        model, batch, offsets[record.subject], cfg
                    )
                    weight = len(chunk) / len(segment)
                    (loss["total"] * weight).backward()
                    for key in weighted:
                        weighted[key] += (
                            float(loss[key].detach().cpu()) * weight
                        )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [*model.parameters(), offsets[record.subject]],
                    float(training["gradient_clip"]),
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
                        "total_loss": weighted["total"],
                        "marginal_sequence_loss": weighted[
                            "next_set_stop"
                        ],
                        "stop_calibration_loss": weighted[
                            "stop_calibration"
                        ],
                        "gradient_norm": float(
                            gradient_norm.detach().cpu()
                        ),
                        "elapsed_seconds": time.time() - started,
                    }
                )
            print(
                json.dumps(
                    {
                        "phase": "shared_full_coverage",
                        "cycle": cycle + 1,
                        "subject": record.subject,
                        "global_update": global_update,
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                ),
                flush=True,
            )
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    coverage = {
        record.subject: {
            "events_available": int(len(record.train_indices)),
            "events_drawn": int(
                len(record.train_indices) * int(coverage_cycles)
            ),
            "completed_cycles": int(coverage_cycles),
            "fraction_of_first_cycle": 1.0,
            "optimizer_updates": int(
                int(coverage_cycles)
                * min(int(updates_per_patient), len(record.train_indices))
            ),
        }
        for record in records
    }
    return state, rows, coverage


def calibrate_heldout_offset(
    model: PersistentPathModeRNN,
    record: SubjectRecord,
    prior: PathModePrior,
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
                len(record.contact_names),
                int(cfg["model"]["local_offset_dim"]),
            ),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        [offset],
        lr=float(cfg["training"]["local_learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    queue = EventQueue(
        record.train_indices, np.random.default_rng(int(seed))
    )
    rows = []
    started = time.time()
    for step in range(int(steps)):
        index = queue.draw(min(int(batch_size), len(record.train_indices)))
        batch = _batch(record, prior, index, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, offset, cfg)
        loss["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [offset], float(cfg["training"]["gradient_clip"])
        )
        optimizer.step()
        rows.append(
            {
                "phase": "heldout_calibration",
                "step": step + 1,
                "subject": record.subject,
                "dataset": record.dataset,
                "total_loss": float(loss["total"].detach().cpu()),
                "marginal_sequence_loss": float(
                    loss["next_set_stop"].detach().cpu()
                ),
                "stop_calibration_loss": float(
                    loss["stop_calibration"].detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "elapsed_seconds": time.time() - started,
            }
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return offset.detach(), rows, {
        "events_available": int(len(record.train_indices)),
        "events_drawn": int(queue.drawn),
        "completed_cycles": int(queue.cycles),
    }


def calibrate_heldout_offset_coverage(
    model: PersistentPathModeRNN,
    record: SubjectRecord,
    prior: PathModePrior,
    cfg: dict,
    *,
    coverage_cycles: int,
    updates_per_cycle: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, list[dict], dict]:
    """Fit the held-out offset using complete train80 coverage cycles."""
    if int(coverage_cycles) < 1 or int(updates_per_cycle) < 1:
        raise ValueError("coverage cycles and updates per cycle must be positive")
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    offset = torch.nn.Parameter(
        torch.zeros(
            (
                len(record.contact_names),
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
    rows = []
    global_update = 0
    started = time.time()
    for cycle in range(int(coverage_cycles)):
        indices = rng.permutation(record.train_indices)
        segments = [
            segment
            for segment in np.array_split(indices, int(updates_per_cycle))
            if len(segment)
        ]
        for segment_index, segment in enumerate(segments):
            optimizer.zero_grad(set_to_none=True)
            weighted = {
                "total": 0.0,
                "next_set_stop": 0.0,
                "stop_calibration": 0.0,
            }
            for batch_start in range(0, len(segment), int(batch_size)):
                chunk = segment[
                    batch_start : batch_start + int(batch_size)
                ]
                batch = _batch(record, prior, chunk, device)
                loss = _loss(model, batch, offset, cfg)
                weight = len(chunk) / len(segment)
                (loss["total"] * weight).backward()
                for key in weighted:
                    weighted[key] += (
                        float(loss[key].detach().cpu()) * weight
                    )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [offset], float(training["gradient_clip"])
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
                    "total_loss": weighted["total"],
                    "marginal_sequence_loss": weighted[
                        "next_set_stop"
                    ],
                    "stop_calibration_loss": weighted[
                        "stop_calibration"
                    ],
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "elapsed_seconds": time.time() - started,
                }
            )
        print(
            json.dumps(
                {
                    "phase": "heldout_offset_full_coverage",
                    "cycle": cycle + 1,
                    "subject": record.subject,
                    "global_update": global_update,
                    "elapsed_seconds": round(time.time() - started, 2),
                }
            ),
            flush=True,
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return offset.detach(), rows, {
        "events_available": int(len(record.train_indices)),
        "events_drawn": int(
            len(record.train_indices) * int(coverage_cycles)
        ),
        "completed_cycles": int(coverage_cycles),
        "fraction_of_first_cycle": 1.0,
        "optimizer_updates": int(
            int(coverage_cycles)
            * min(int(updates_per_cycle), len(record.train_indices))
        ),
    }


@torch.no_grad()
def evaluate_teacher_forced(
    model: PersistentPathModeRNN,
    record: SubjectRecord,
    prior: PathModePrior,
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
        probability = loss["predictive_action_probability"].cpu().numpy()
        posterior = loss["final_component_posterior"].cpu().numpy()
        groups = record.group_ids[index]
        counts = record.group_count[index]
        event_nll = loss["event_nll"].cpu().numpy()
        for local, event_index in enumerate(index):
            component = int(np.argmax(posterior[local]))
            mode = int(prior.component_mode[component])
            direction = int(prior.component_direction[component])
            entropy = float(
                -np.sum(
                    posterior[local]
                    * np.log(np.clip(posterior[local], 1e-12, 1.0))
                )
            )
            rows.append(
                {
                    "subject": record.subject,
                    "event_index": int(event_index),
                    "event_source_index": int(
                        record.event_source_index[event_index]
                    ),
                    "event_nll": float(event_nll[local]),
                    "map_component": component,
                    "map_mode": mode,
                    "map_direction": direction,
                    "posterior_max": float(np.max(posterior[local])),
                    "posterior_entropy": entropy,
                }
            )
            for step in range(int(counts[local]) + 1):
                terminal = step == int(counts[local])
                stop_probability.append(float(probability[local, step, 0]))
                stop_target.append(float(terminal))
                action = int(np.argmax(probability[local, step]))
                if not terminal:
                    top1.append(
                        float(
                            action > 0
                            and groups[local, action - 1] == step
                        )
                    )
    frame = pd.DataFrame(rows)
    stop_probability = np.asarray(stop_probability)
    stop_target = np.asarray(stop_target)
    occupancy = (
        frame.groupby(["map_mode", "map_direction"]).size() / len(frame)
    )
    return {
        "heldout_event_nll": float(frame.event_nll.mean()),
        "top1_next_set_accuracy": float(np.mean(top1)),
        "stop_brier": float(
            np.mean((stop_probability - stop_target) ** 2)
        ),
        "terminal_stop_probability": float(
            np.mean(stop_probability[stop_target == 1])
        ),
        "nonterminal_stop_probability": float(
            np.mean(stop_probability[stop_target == 0])
        ),
        "posterior_max_mean": float(frame.posterior_max.mean()),
        "posterior_entropy_mean": float(frame.posterior_entropy.mean()),
        "map_occupancy": {
            f"mode_{mode}:direction_{direction}": float(value)
            for (mode, direction), value in occupancy.items()
        },
        "n_eval_events": int(len(frame)),
    }, frame


def _lesioned_prior(
    prior: PathModePrior,
    lesion: str,
) -> PathModePrior:
    if lesion in {"none", "inhibition", "graph"}:
        return prior
    component_prior = prior.component_prior.copy()
    graphs = prior.component_graphs.copy()
    if lesion == "drop_dominant_mode":
        if prior.mode_count < 2:
            raise ValueError("dominant-mode lesion requires K>=2")
        mode_mass = np.bincount(
            prior.component_mode,
            weights=component_prior,
            minlength=prior.mode_count,
        )
        component_prior[prior.component_mode == int(np.argmax(mode_mass))] = 0
    elif lesion == "drop_forward":
        component_prior[prior.component_direction > 0] = 0
    elif lesion == "drop_reverse":
        component_prior[prior.component_direction < 0] = 0
    elif lesion == "mode_collapse":
        for index, direction in enumerate(prior.component_direction):
            if direction > 0:
                graphs[index] = prior.aggregate_forward
            elif direction < 0:
                graphs[index] = prior.aggregate_reverse
            else:
                graphs[index] = 0.5 * (
                    prior.aggregate_forward + prior.aggregate_reverse
                )
    else:
        raise ValueError(f"unknown lesion: {lesion}")
    if component_prior.sum() <= 0:
        raise ValueError(f"lesion removed all component mass: {lesion}")
    component_prior /= component_prior.sum()
    return replace(
        prior,
        component_graphs=graphs,
        component_prior=component_prior,
    )


def _rollout(
    model: PersistentPathModeRNN,
    record: SubjectRecord,
    prior: PathModePrior,
    offset: torch.Tensor,
    *,
    device: torch.device,
    n_events: int,
    seed: int,
    lesion: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effective = _lesioned_prior(prior, lesion)
    model_lesion = lesion if lesion in {"inhibition", "graph"} else "none"
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.ones(
        (1, len(record.contact_names)), dtype=torch.bool, device=device
    )
    return model.rollout(
        features,
        mask,
        offset,
        torch.as_tensor(
            effective.component_graphs, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            effective.component_prior, dtype=torch.float32, device=device
        ),
        torch.as_tensor(effective.left, dtype=torch.bool, device=device),
        torch.as_tensor(effective.right, dtype=torch.bool, device=device),
        n_events=int(n_events),
        seed=int(seed),
        lesion=model_lesion,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_persistent_path_mode_rnn_v0_9.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--mode-count", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--control",
        choices=(
            "no_history",
            "merged_path",
            "intact",
            "weight_shuffle",
            "mode_shuffle",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shared-steps", type=int, default=None)
    parser.add_argument("--calibration-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--formal-coverage", action="store_true")
    parser.add_argument("--coverage-shared-cycles", type=int, default=None)
    parser.add_argument(
        "--coverage-calibration-cycles", type=int, default=None
    )
    parser.add_argument("--coverage-updates-per-patient", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    records = load_records(ROOT / cfg["inputs"]["dataset"])
    if args.heldout_subject not in records:
        raise RuntimeError(f"heldout subject absent: {args.heldout_subject}")
    priors = load_path_mode_priors(
        ROOT / cfg["inputs"]["path_mode_prior"],
        records,
        mode_count=int(args.mode_count),
        control=args.control,
        seed=int(args.seed),
        axis_floor=float(cfg["prior"]["axis_floor"]),
        neighbors=int(cfg["prior"]["neighbors"]),
    )
    heldout = records[args.heldout_subject]
    outer = select_outer_records(records, heldout.subject)
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
                "mode_count": int(args.mode_count),
                "control": args.control,
                "seed": int(args.seed),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    started = time.time()
    model = PersistentPathModeRNN(
        heldout.contact_features.shape[1],
        local_offset_dim=int(cfg["model"]["local_offset_dim"]),
        use_recurrence=priors[heldout.subject].use_recurrence,
    )
    coverage_shared_cycles = int(
        args.coverage_shared_cycles
        if args.coverage_shared_cycles is not None
        else cfg["training"]["formal_shared_coverage_cycles"]
    )
    coverage_calibration_cycles = int(
        args.coverage_calibration_cycles
        if args.coverage_calibration_cycles is not None
        else cfg["training"]["formal_calibration_coverage_cycles"]
    )
    coverage_updates_per_patient = int(
        args.coverage_updates_per_patient
        if args.coverage_updates_per_patient is not None
        else cfg["training"]["formal_updates_per_patient"]
    )
    if args.formal_coverage:
        model_state, shared_log, shared_coverage = train_shared_coverage(
            model,
            outer,
            priors,
            cfg,
            coverage_cycles=coverage_shared_cycles,
            updates_per_patient=coverage_updates_per_patient,
            batch_size=batch_size,
            device=device,
            seed=int(args.seed),
        )
    else:
        model_state, shared_log, shared_coverage = train_shared(
            model,
            outer,
            priors,
            cfg,
            steps=shared_steps,
            batch_size=batch_size,
            device=device,
            seed=int(args.seed),
        )
    model.load_state_dict(model_state)
    if args.formal_coverage:
        offset, calibration_log, calibration_coverage = (
            calibrate_heldout_offset_coverage(
                model,
                heldout,
                priors[heldout.subject],
                cfg,
                coverage_cycles=coverage_calibration_cycles,
                updates_per_cycle=coverage_updates_per_patient,
                batch_size=batch_size,
                device=device,
                seed=int(args.seed) + 500_000,
            )
        )
    else:
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
    if not args.primary_only and args.control == "intact":
        lesions.extend(
            ["graph", "inhibition", "drop_forward", "drop_reverse", "mode_collapse"]
        )
        if int(args.mode_count) >= 2:
            lesions.append("drop_dominant_mode")
    metric_rows = []
    primary_groups = None
    primary_count = None
    primary_components = None
    for lesion in lesions:
        groups, count, components = _rollout(
            model,
            heldout,
            priors[heldout.subject],
            offset.to(device),
            device=device,
            n_events=n_rollouts,
            seed=(
                int(args.seed)
                + 700_000
            ),
            lesion=lesion,
        )
        if lesion == "none":
            primary_groups = groups
            primary_count = count
            primary_components = components
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
        occupancy = np.bincount(
            components,
            minlength=len(priors[heldout.subject].component_prior),
        ) / len(components)
        metric_rows.append(
            {
                "subject": heldout.subject,
                "dataset": heldout.dataset,
                "mode_count": int(args.mode_count),
                "control": args.control,
                "lesion": lesion,
                "seed": int(args.seed),
                "n_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "rollout_participant_count_mean": float(
                    np.mean(np.sum(groups >= 0, axis=1))
                ),
                "rollout_zero_length_fraction": float(np.mean(count == 0)),
                "rollout_component_entropy": float(
                    -np.sum(
                        occupancy * np.log(np.clip(occupancy, 1e-12, 1.0))
                    )
                ),
                **{
                    key: value
                    for key, value in teacher_metrics.items()
                    if key != "map_occupancy"
                },
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
    event_frame.to_csv(run_dir / "heldout_event_modes.csv", index=False)
    pd.DataFrame(shared_log + calibration_log).to_csv(
        run_dir / "training_log.csv", index=False
    )
    selected_modes = priors[heldout.subject].component_mode[
        primary_components
    ]
    selected_directions = priors[heldout.subject].component_direction[
        primary_components
    ]
    np.savez_compressed(
        run_dir / "free_rollouts.npz",
        event_group_ids=primary_groups,
        event_group_count=primary_count,
        selected_component=primary_components,
        selected_mode=selected_modes,
        selected_direction=selected_directions,
        contact_names=heldout.contact_names,
        axis_coordinate=priors[heldout.subject].axis,
        ictal_target_read=np.asarray(False),
    )
    torch.save(
        {
            "contract": cfg["contract"],
            "model_state": model_state,
            "heldout_local_offset": offset.cpu(),
            "subject": heldout.subject,
            "mode_count": int(args.mode_count),
            "control": args.control,
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
        "mode_count": int(args.mode_count),
        "control": args.control,
        "seed": int(args.seed),
        "n_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "shared_steps": shared_steps,
        "calibration_steps": calibration_steps,
        "formal_coverage": bool(args.formal_coverage),
        "coverage_shared_cycles": (
            coverage_shared_cycles if args.formal_coverage else 0
        ),
        "coverage_calibration_cycles": (
            coverage_calibration_cycles if args.formal_coverage else 0
        ),
        "coverage_updates_per_patient": (
            coverage_updates_per_patient if args.formal_coverage else 0
        ),
        "rollouts": n_rollouts,
        "elapsed_seconds": time.time() - started,
        "shared_coverage": shared_coverage,
        "calibration_coverage": calibration_coverage,
        "teacher_map_occupancy": teacher_metrics["map_occupancy"],
        "empirical_distribution_errors": empirical_error,
        "split_half_distribution_errors": split_half_error,
        "input_fingerprints": {
            subject: {
                "record_sha256": record.input_sha256,
                "path_mode_prior_sha256": priors[subject].source_sha256,
            }
            for subject, record in records.items()
        },
        "dataset_root": str(ROOT / cfg["inputs"]["dataset"]),
        "prior_root": str(ROOT / cfg["inputs"]["path_mode_prior"]),
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
                "mode_count": int(args.mode_count),
                "control": args.control,
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
                "mode_count": int(args.mode_count),
                "control": args.control,
                "elapsed_seconds": round(summary["elapsed_seconds"], 2),
                "ictal_target_read": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
