"""Target-sealed data construction for the cross-event HistoryRNN.

This module contains only the common, causally valid data path used by the
strict sequential trainer.  It deliberately excludes the deprecated
fixed-window trainer, whose artificial window resets do not represent a
continuous patient history state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from scripts.build_topic5_interictal_operator_dataset import _raw_subject_dir
from src.interictal_propagation import load_subject_propagation_events
from src.topic5_history_rnn import build_continuous_segment_ids, encode_within_event
from src.topic5_rank_distribution import LinearStateSequenceRNN


@dataclass
class HistoryRecord:
    subject: str
    dataset: str
    event_embedding: np.ndarray
    contact_embedding: np.ndarray
    participation: np.ndarray
    relative_rank: np.ndarray
    event_time: np.ndarray
    segment_id: np.ndarray
    event_split: np.ndarray
    static_logit: np.ndarray

    def target_indices(self, split: int) -> np.ndarray:
        previous_same_segment = np.zeros(len(self.segment_id), dtype=bool)
        previous_same_segment[1:] = self.segment_id[1:] == self.segment_id[:-1]
        return np.flatnonzero(
            (self.event_split == int(split)) & previous_same_segment
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_metadata(
    subject: str,
    dataset: str,
    artifact_root: Path,
) -> dict[str, dict]:
    short = subject.split("_", 1)[1]
    if dataset == "epilepsiae":
        frame = pd.read_csv(
            artifact_root / "results/epilepsiae_block_inventory.csv",
            dtype={"subject": str, "recording_id": str},
        )
        frame = frame.loc[frame.subject.astype(str) == short].copy()
    elif dataset == "yuquan":
        frame = pd.read_csv(
            artifact_root / "results/dataset_inventory/yuquan_block_inventory.csv",
            dtype={"subject": str, "recording_id": str},
        )
        frame = frame.loc[frame.subject.astype(str) == short].copy()
        frame["block_no"] = np.arange(len(frame), dtype=int)
    else:  # pragma: no cover
        raise ValueError(dataset)
    frame = frame.sort_values(["block_start_epoch", "block_stem"]).reset_index(
        drop=True
    )
    frame["sequence_index"] = np.arange(len(frame), dtype=int)
    if frame.block_stem.astype(str).duplicated().any():
        raise RuntimeError(f"{subject}: duplicate block stems in inventory")
    return {
        str(row.block_stem): {
            "recording_id": str(row.recording_id),
            "block_no": int(row.block_no),
            "sequence_index": int(row.sequence_index),
            "block_start_epoch": float(row.block_start_epoch),
            "block_end_epoch": float(row.block_end_epoch),
        }
        for row in frame.itertuples()
    }


def _segment_ids(
    subject: str,
    dataset: str,
    source_index: np.ndarray,
    artifact_root: Path,
) -> np.ndarray:
    raw = load_subject_propagation_events(_raw_subject_dir(subject))
    raw_block_id = np.asarray(raw["block_ids"], np.int64)
    raw_names = np.asarray([str(value) for value in raw["record_names"]])
    block_id = raw_block_id[source_index]
    stems = raw_names[block_id]
    metadata = _block_metadata(subject, dataset, artifact_root)

    # The Yuquan inventory is incomplete.  Missing recordings fail closed at
    # every file boundary; their start labels are never used to infer duration
    # or continuity.
    raw_starts = np.asarray(raw["block_start_times"], np.float64)
    for sequence_index, stem in enumerate(raw_names):
        if str(stem) in metadata:
            continue
        start = float(raw_starts[sequence_index])
        metadata[str(stem)] = {
            "recording_id": f"inventory_missing:{stem}",
            "block_no": int(sequence_index),
            "sequence_index": int(sequence_index),
            "block_start_epoch": start,
            "block_end_epoch": start,
            "inventory_missing_fail_closed": True,
        }
    segment, _ = build_continuous_segment_ids(
        stems,
        metadata,
        allow_cross_recording_contiguous=True,
    )
    return segment


@torch.no_grad()
def encode_subject(
    event_model: LinearStateSequenceRNN,
    record,
    *,
    artifact_root: Path,
    device: torch.device,
    batch_size: int,
    within_event_rank_shuffle_seed: int | None = None,
) -> HistoryRecord:
    with np.load(record.path, allow_pickle=False) as data:
        group_ids = np.asarray(data["event_group_ids"], np.int16)
        group_count = np.asarray(data["event_group_count"], np.int16)
        participation = np.asarray(data["event_participation"], np.uint8)
        event_time = np.asarray(data["event_abs_time"], np.float64)
        event_split = np.asarray(data["event_split"], np.uint8)
        source_index = np.asarray(data["event_source_index"], np.int64)
        contact_features = np.asarray(data["contact_features"], np.float32)

    if within_event_rank_shuffle_seed is not None:
        generator = np.random.default_rng(int(within_event_rank_shuffle_seed))
        group_ids = group_ids.copy()
        for event_index in range(len(group_ids)):
            member = np.flatnonzero(group_ids[event_index] >= 0)
            if len(member) > 1:
                group_ids[event_index, member] = group_ids[
                    event_index, generator.permutation(member)
                ]

    relative_rank = np.full(group_ids.shape, np.nan, dtype=np.float32)
    for index, count in enumerate(group_count):
        member = group_ids[index] >= 0
        denominator = max(int(count) - 1, 1)
        relative_rank[index, member] = group_ids[index, member] / denominator
        if np.any(member):
            relative_rank[index, member] -= np.mean(relative_rank[index, member])

    features = torch.as_tensor(contact_features, device=device).unsqueeze(0)
    zero_offset = torch.zeros(
        (contact_features.shape[0], event_model.local_offset_dim),
        dtype=torch.float32,
        device=device,
    )
    event_states: list[np.ndarray] = []
    contact_embedding = None
    for start in range(0, len(group_ids), int(batch_size)):
        stop = min(start + int(batch_size), len(group_ids))
        state, embedding = encode_within_event(
            event_model,
            features.expand(stop - start, -1, -1),
            torch.as_tensor(
                group_ids[start:stop], dtype=torch.long, device=device
            ),
            torch.as_tensor(
                group_count[start:stop], dtype=torch.long, device=device
            ),
            local_offset=zero_offset,
        )
        event_states.append(state.cpu().numpy().astype(np.float32))
        if contact_embedding is None:
            contact_embedding = embedding[0].cpu().numpy().astype(np.float32)

    train = event_split == 0
    prior = (participation[train].sum(0) + 0.5) / (np.sum(train) + 1.0)
    static_logit = np.log(np.clip(prior, 1e-5, 1 - 1e-5)) - np.log(
        np.clip(1 - prior, 1e-5, 1 - 1e-5)
    )
    return HistoryRecord(
        subject=record.subject,
        dataset=record.dataset,
        event_embedding=np.row_stack(event_states),
        contact_embedding=np.asarray(contact_embedding),
        participation=participation,
        relative_rank=relative_rank,
        event_time=event_time,
        segment_id=_segment_ids(
            record.subject, record.dataset, source_index, artifact_root
        ),
        event_split=event_split,
        static_logit=static_logit.astype(np.float32),
    )


def training_normalization(
    records: Sequence[HistoryRecord],
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_value = None
    sum_square = None
    for record in records:
        value = record.event_embedding[record.event_split == 0].astype(np.float64)
        if not len(value):
            continue
        current_sum = value.sum(0)
        current_square = np.square(value).sum(0)
        sum_value = current_sum if sum_value is None else sum_value + current_sum
        sum_square = (
            current_square if sum_square is None else sum_square + current_square
        )
        total += len(value)
    if total == 0 or sum_value is None or sum_square is None:
        raise RuntimeError("no outer-training events available for normalization")
    mean = sum_value / total
    variance = np.maximum(sum_square / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
