#!/usr/bin/env python3
"""Strict G1 fold with persistent state over complete continuous segments.

Unlike the engineering window smoke, this trainer never resets at an
arbitrary training window.  It carries state across truncated-BPTT chunks and
detaches (but does not zero) the state at chunk boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required; use cuda_env") from exc

from src.topic5_history_data import (  # noqa: E402
    HistoryRecord,
    encode_subject,
    sha256,
    training_normalization,
)
from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_history_rnn import (  # noqa: E402
    MatchedUnorderedSummary,
    NextEventFieldHeads,
    TimeDecayHistoryGRU,
    next_event_field_loss,
)
from src.topic5_rank_distribution import LinearStateSequenceRNN  # noqa: E402


@dataclass
class Segment:
    subject: str
    dataset: str
    original_index: np.ndarray
    embedding: np.ndarray
    unordered_summary: np.ndarray
    participation: np.ndarray
    relative_rank: np.ndarray
    event_time: np.ndarray
    event_split: np.ndarray
    contact_embedding: np.ndarray
    static_logit: np.ndarray
    train_decision_weight: float

    @property
    def length(self) -> int:
        return int(len(self.embedding))


class MatchedSequentialModel(nn.Module):
    def __init__(self, event_dim: int, state_dim: int, contact_dim: int):
        super().__init__()
        self.history = MatchedUnorderedSummary(event_dim, state_dim)
        self.heads = NextEventFieldHeads(state_dim, contact_dim)

    def forward(
        self,
        summary: torch.Tensor,
        contact_embedding: torch.Tensor,
        static_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prediction = self.heads(self.history(summary), contact_embedding)
        prediction["participation_logits"] = (
            prediction["participation_logits"] + static_logit[:, None]
        )
        return prediction


class UnorderedResidualSequentialModel(nn.Module):
    """Capacity-matched residual branch with no chronological recurrence."""

    def __init__(self, event_dim: int, state_dim: int, contact_dim: int):
        super().__init__()
        self.history = MatchedUnorderedSummary(event_dim, state_dim)
        self.heads = NextEventFieldHeads(state_dim, contact_dim)

    def forward(
        self, summary: torch.Tensor, contact_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.heads(self.history(summary), contact_embedding)


class ResidualSequentialModel(nn.Module):
    def __init__(
        self,
        event_dim: int,
        state_dim: int,
        contact_dim: int,
        *,
        initial_half_life_hours: float,
    ):
        super().__init__()
        self.history = TimeDecayHistoryGRU(
            event_dim,
            state_dim,
            initial_half_life_hours=initial_half_life_hours,
        )
        self.heads = NextEventFieldHeads(state_dim, contact_dim)

    def forward(
        self,
        embedding: torch.Tensor,
        delta_t: torch.Tensor,
        reset_mask: torch.Tensor,
        event_mask: torch.Tensor,
        contact_embedding: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        state, final = self.history.forward_masked(
            embedding,
            delta_t,
            reset_mask,
            event_mask,
            initial_state=initial_state,
        )
        return self.heads(state, contact_embedding), final


def _capacity_matched_unordered_dim(
    event_dim: int, contact_dim: int, history_dim: int
) -> int:
    """Choose the M1 residual width closest to the HistoryRNN parameter count."""

    # MatchedUnorderedSummary + two bias-free contact queries.
    def unordered_parameters(width: int) -> int:
        return (
            (3 * event_dim + 3) * width
            + width
            + width * width
            + width
            + 2 * width * contact_dim
        )

    # GRUCell + dimension-wise decay + two bias-free contact queries.
    history_parameters = (
        3 * history_dim * event_dim
        + 3 * history_dim * history_dim
        + 6 * history_dim
        + history_dim
        + 2 * history_dim * contact_dim
    )
    candidates = range(1, max(4 * int(history_dim), 2))
    return min(candidates, key=lambda width: abs(unordered_parameters(width) - history_parameters))


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


def _causal_unordered_summary(
    embedding: np.ndarray, event_time: np.ndarray
) -> np.ndarray:
    count = np.arange(1, len(embedding) + 1, dtype=np.float32)
    mean = np.cumsum(embedding, axis=0, dtype=np.float64) / count[:, None]
    maximum = np.maximum.accumulate(embedding, axis=0)
    span_hours = (event_time - event_time[0]) / 3600.0
    previous_iei = np.diff(event_time, prepend=event_time[0])
    scalar = np.column_stack(
        [
            np.log1p(count),
            np.log1p(np.maximum(span_hours, 0.0)),
            np.log1p(np.maximum(previous_iei, 0.0)),
        ]
    )
    return np.column_stack([mean, maximum, embedding, scalar]).astype(np.float32)


def _segments(
    records: Sequence[HistoryRecord],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> list[Segment]:
    train_decisions = {
        record.subject: max(int(record.target_indices(0).size), 1)
        for record in records
    }
    out = []
    for record in records:
        for segment_id in np.unique(record.segment_id):
            index = np.flatnonzero(record.segment_id == segment_id)
            if len(index) < 2:
                continue
            embedding = ((record.event_embedding[index] - mean) / scale).astype(
                np.float32
            )
            out.append(
                Segment(
                    subject=record.subject,
                    dataset=record.dataset,
                    original_index=index,
                    embedding=embedding,
                    unordered_summary=_causal_unordered_summary(
                        embedding, record.event_time[index]
                    ),
                    participation=record.participation[index],
                    relative_rank=record.relative_rank[index],
                    event_time=record.event_time[index],
                    event_split=record.event_split[index],
                    contact_embedding=record.contact_embedding,
                    static_logit=record.static_logit,
                    train_decision_weight=1.0 / train_decisions[record.subject],
                )
            )
    return out


def _groups(
    segments: Sequence[Segment], batch_segments: int, rng: np.random.Generator
) -> list[list[Segment]]:
    ordered = sorted(segments, key=lambda segment: segment.length)
    groups = [
        ordered[start : start + int(batch_segments)]
        for start in range(0, len(ordered), int(batch_segments))
    ]
    rng.shuffle(groups)
    return groups


def _contact_arrays(
    segments: Sequence[Segment], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(segments)
    max_contacts = max(len(segment.contact_embedding) for segment in segments)
    contact_dim = segments[0].contact_embedding.shape[1]
    embedding = np.zeros((batch, max_contacts, contact_dim), dtype=np.float32)
    static = np.zeros((batch, max_contacts), dtype=np.float32)
    mask = np.zeros((batch, max_contacts), dtype=bool)
    for row, segment in enumerate(segments):
        size = len(segment.contact_embedding)
        embedding[row, :size] = segment.contact_embedding
        static[row, :size] = segment.static_logit
        mask[row, :size] = True
    return (
        torch.as_tensor(embedding, device=device),
        torch.as_tensor(static, device=device),
        torch.as_tensor(mask, device=device),
    )


def _chunk(
    segments: Sequence[Segment],
    start: int,
    chunk_length: int,
    *,
    target_split: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = len(segments)
    stop = min(max(segment.length for segment in segments), start + int(chunk_length))
    length = stop - start
    event_dim = segments[0].embedding.shape[1]
    summary_dim = segments[0].unordered_summary.shape[1]
    max_contacts = max(len(segment.contact_embedding) for segment in segments)
    embedding = np.zeros((batch, length, event_dim), dtype=np.float32)
    summary = np.zeros((batch, length, summary_dim), dtype=np.float32)
    delta_t = np.zeros((batch, length), dtype=np.float32)
    reset = np.zeros((batch, length), dtype=bool)
    event_mask = np.zeros((batch, length), dtype=bool)
    decision_mask = np.zeros((batch, length), dtype=bool)
    event_weight = np.zeros((batch, length), dtype=np.float32)
    target_participation = np.zeros((batch, length, max_contacts), dtype=np.float32)
    target_rank = np.full(
        (batch, length, max_contacts), np.nan, dtype=np.float32
    )
    target_original_index = np.full((batch, length), -1, dtype=np.int64)
    for row, segment in enumerate(segments):
        local_stop = min(stop, segment.length)
        if start >= local_stop:
            continue
        size = local_stop - start
        embedding[row, :size] = segment.embedding[start:local_stop]
        summary[row, :size] = segment.unordered_summary[start:local_stop]
        event_mask[row, :size] = True
        if start == 0:
            reset[row, 0] = True
        positions = np.arange(start, local_stop)
        if start == 0:
            delta_t[row, 0] = 0.0
        else:
            delta_t[row, 0] = float(
                segment.event_time[start] - segment.event_time[start - 1]
            )
        if size > 1:
            delta_t[row, 1:size] = np.diff(
                segment.event_time[start:local_stop]
            ).astype(np.float32)
        has_target = positions + 1 < segment.length
        valid = np.zeros(size, dtype=bool)
        valid[has_target] = (
            segment.event_split[positions[has_target] + 1] == int(target_split)
        )
        decision_mask[row, :size] = valid
        event_weight[row, :size] = valid * float(segment.train_decision_weight)
        for local, position in enumerate(positions):
            target = position + 1
            if target >= segment.length:
                continue
            contacts = segment.participation.shape[1]
            target_participation[row, local, :contacts] = segment.participation[
                target
            ]
            target_rank[row, local, :contacts] = segment.relative_rank[target]
            target_original_index[row, local] = segment.original_index[target]
    contact_embedding, static_logit, contact_mask = _contact_arrays(segments, device)
    return {
        "embedding": torch.as_tensor(embedding, device=device),
        "summary": torch.as_tensor(summary, device=device),
        "delta_t": torch.as_tensor(delta_t, device=device),
        "reset_mask": torch.as_tensor(reset, device=device),
        "event_mask": torch.as_tensor(event_mask, device=device),
        "decision_mask": torch.as_tensor(decision_mask, device=device),
        "event_weight": torch.as_tensor(event_weight, device=device),
        "participation": torch.as_tensor(target_participation, device=device),
        "relative_rank": torch.as_tensor(target_rank, device=device),
        "target_original_index": torch.as_tensor(target_original_index, device=device),
        "contact_embedding": contact_embedding,
        "static_logit": static_logit,
        "contact_mask": contact_mask,
    }


def _add_predictions(base: dict, residual: dict) -> dict:
    return {
        "participation_logits": base["participation_logits"]
        + residual["participation_logits"],
        "relative_rank": base["relative_rank"] + residual["relative_rank"],
    }


def _train(
    matched: MatchedSequentialModel,
    unordered_residual: UnorderedResidualSequentialModel,
    residual: ResidualSequentialModel,
    segments: Sequence[Segment],
    *,
    stage: str,
    cycles: int,
    batch_segments: int,
    chunk_length: int,
    learning_rate: float,
    rank_weight: float,
    seed: int,
    device: torch.device,
) -> list[dict]:
    model = {
        "matched_base": matched,
        "unordered_residual": unordered_residual,
        "chronological_residual": residual,
    }[stage]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    rows = []
    global_update = 0
    start_time = time.time()
    for cycle in range(int(cycles)):
        for group_index, group in enumerate(_groups(segments, batch_segments, rng)):
            state = None
            for start in range(0, max(segment.length for segment in group), int(chunk_length)):
                batch = _chunk(
                    group,
                    start,
                    chunk_length,
                    target_split=0,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(stage == "matched_base"):
                    base = matched(
                        batch["summary"],
                        batch["contact_embedding"],
                        batch["static_logit"],
                    )
                if stage == "matched_base":
                    prediction = base
                    final = None
                elif stage == "unordered_residual":
                    increment = unordered_residual(
                        batch["summary"], batch["contact_embedding"]
                    )
                    prediction = _add_predictions(base, increment)
                    final = None
                else:
                    increment, final = residual(
                        batch["embedding"],
                        batch["delta_t"],
                        batch["reset_mask"],
                        batch["event_mask"],
                        batch["contact_embedding"],
                        state,
                    )
                    prediction = _add_predictions(base, increment)
                if not torch.any(batch["decision_mask"]):
                    if final is not None:
                        state = final.detach()
                    continue
                loss = next_event_field_loss(
                    prediction,
                    batch["participation"],
                    batch["relative_rank"],
                    rank_weight=rank_weight,
                    event_weight=batch["event_weight"],
                    contact_mask=batch["contact_mask"],
                )
                loss["total"].backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                global_update += 1
                if final is not None:
                    state = final.detach()
                rows.append(
                    {
                        "stage": stage,
                        "cycle": cycle + 1,
                        "group": group_index + 1,
                        "chunk_start": start,
                        "global_update": global_update,
                        "total": float(loss["total"].detach().cpu()),
                        "participation_bce": float(
                            loss["participation_bce"].detach().cpu()
                        ),
                        "relative_rank_huber": float(
                            loss["relative_rank_huber"].detach().cpu()
                        ),
                        "gradient_norm": float(gradient.detach().cpu()),
                        "elapsed_seconds": time.time() - start_time,
                    }
                )
        recent = rows[-min(100, len(rows)) :]
        print(
            json.dumps(
                {
                    "stage": stage,
                    "cycle": cycle + 1,
                    "cycles": int(cycles),
                    "updates": global_update,
                    "recent_loss": float(np.mean([row["total"] for row in recent])),
                    "elapsed_seconds": round(time.time() - start_time, 2),
                }
            ),
            flush=True,
        )
    return rows


@torch.no_grad()
def _evaluate_condition(
    matched: MatchedSequentialModel,
    unordered_residual: UnorderedResidualSequentialModel,
    residual: ResidualSequentialModel,
    segments: Sequence[Segment],
    *,
    condition: str,
    batch_segments: int,
    chunk_length: int,
    rank_weight: float,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in _groups(segments, batch_segments, np.random.default_rng(0)):
        state = None
        shuffled = []
        if condition == "across_event_order_shuffle":
            for segment in group:
                order = rng.permutation(segment.length)
                shuffled.append(segment.embedding[order])
        for start in range(0, max(segment.length for segment in group), int(chunk_length)):
            batch = _chunk(
                group,
                start,
                chunk_length,
                target_split=1,
                device=device,
            )
            base = matched(
                batch["summary"], batch["contact_embedding"], batch["static_logit"]
            )
            unordered_increment = unordered_residual(
                batch["summary"], batch["contact_embedding"]
            )
            matched_capacity = _add_predictions(base, unordered_increment)
            if condition == "static_prior":
                prediction = {
                    "participation_logits": batch["static_logit"][:, None].expand(
                        -1, batch["summary"].shape[1], -1
                    ),
                    "relative_rank": torch.zeros(
                        (*batch["summary"].shape[:2], batch["contact_embedding"].shape[1]),
                        device=device,
                    ),
                }
                final = None
            elif condition == "base_unordered":
                prediction = base
                final = None
            elif condition == "matched_unordered":
                prediction = matched_capacity
                final = None
            else:
                if condition == "across_event_order_shuffle":
                    for row, segment in enumerate(group):
                        local_stop = min(start + int(chunk_length), segment.length)
                        if start < local_stop:
                            batch["embedding"][row, : local_stop - start] = torch.as_tensor(
                                shuffled[row][start:local_stop], device=device
                            )
                increment, final = residual(
                    batch["embedding"],
                    batch["delta_t"],
                    batch["reset_mask"],
                    batch["event_mask"],
                    batch["contact_embedding"],
                    state,
                )
                prediction = _add_predictions(base, increment)
                state = final
            if not torch.any(batch["decision_mask"]):
                continue
            loss = next_event_field_loss(
                prediction,
                batch["participation"],
                batch["relative_rank"],
                rank_weight=rank_weight,
                event_weight=batch["decision_mask"].to(torch.float32),
                contact_mask=batch["contact_mask"],
            )
            decision = batch["decision_mask"].cpu().numpy()
            event_bce = loss["event_participation_bce"].cpu().numpy()
            event_rank = loss["event_relative_rank_huber"].cpu().numpy()
            target_index = batch["target_original_index"].cpu().numpy()
            for row, segment in enumerate(group):
                for column in np.flatnonzero(decision[row]):
                    rows.append(
                        {
                            "subject": segment.subject,
                            "dataset": segment.dataset,
                            "event_index": int(target_index[row, column]),
                            "model": condition,
                            "participation_bce": float(event_bce[row, column]),
                            "relative_rank_huber": float(event_rank[row, column]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--history-dim", type=int, default=32)
    parser.add_argument("--initial-half-life-hours", type=float, default=2.0)
    parser.add_argument("--matched-cycles", type=int, default=3)
    parser.add_argument("--history-cycles", type=int, default=3)
    parser.add_argument("--segment-batch-size", type=int, default=16)
    parser.add_argument("--bptt-chunk", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = args.artifact_root.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)
    dataset = artifact / "results/topic5_interictal_rank_distribution/dataset_v0_4"
    event_checkpoint = (
        artifact
        / "results/topic5_rnn_training_sufficiency_v0_1/formal/"
        f"converged_teacher_forced/seed_{args.seed}/{args.heldout_subject}/checkpoint.pt"
    )
    payload = torch.load(event_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("heldout_subject") != args.heldout_subject:
        raise RuntimeError("EventRNN outer fold mismatch")
    if bool(payload.get("ictal_target_read", True)):
        raise RuntimeError("EventRNN checkpoint is not target sealed")
    event_model = LinearStateSequenceRNN(**payload["model_kwargs"])
    event_model.load_state_dict(payload["model_state"])
    event_model.to(device).eval()
    source_records = load_records(dataset)
    encoded = {}
    encode_start = time.time()
    for position, subject in enumerate(sorted(source_records), start=1):
        encoded[subject] = encode_subject(
            event_model,
            source_records[subject],
            artifact_root=artifact,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        print(
            json.dumps(
                {
                    "phase": "event_embedding",
                    "position": position,
                    "total": len(source_records),
                    "subject": subject,
                    "elapsed_seconds": round(time.time() - encode_start, 2),
                }
            ),
            flush=True,
        )
    outer_records = [
        record for subject, record in encoded.items() if subject != args.heldout_subject
    ]
    heldout_record = encoded[args.heldout_subject]
    mean, scale = training_normalization(outer_records)
    outer_segments = _segments(outer_records, mean=mean, scale=scale)
    heldout_segments = _segments([heldout_record], mean=mean, scale=scale)
    event_dim = heldout_record.event_embedding.shape[1]
    contact_dim = heldout_record.contact_embedding.shape[1]
    matched = MatchedSequentialModel(event_dim, args.history_dim, contact_dim).to(device)
    unordered_dim = _capacity_matched_unordered_dim(
        event_dim, contact_dim, args.history_dim
    )
    unordered_residual = UnorderedResidualSequentialModel(
        event_dim, unordered_dim, contact_dim
    ).to(device)
    residual = ResidualSequentialModel(
        event_dim,
        args.history_dim,
        contact_dim,
        initial_half_life_hours=args.initial_half_life_hours,
    ).to(device)
    log = _train(
        matched,
        unordered_residual,
        residual,
        outer_segments,
        stage="matched_base",
        cycles=args.matched_cycles,
        batch_segments=args.segment_batch_size,
        chunk_length=args.bptt_chunk,
        learning_rate=args.learning_rate,
        rank_weight=args.rank_weight,
        seed=args.seed + 1000,
        device=device,
    )
    matched.eval()
    for parameter in matched.parameters():
        parameter.requires_grad_(False)
    log.extend(
        _train(
            matched,
            unordered_residual,
            residual,
            outer_segments,
            stage="unordered_residual",
            cycles=args.history_cycles,
            batch_segments=args.segment_batch_size,
            chunk_length=args.bptt_chunk,
            learning_rate=args.learning_rate,
            rank_weight=args.rank_weight,
            seed=args.seed + 1500,
            device=device,
        )
    )
    unordered_residual.eval()
    for parameter in unordered_residual.parameters():
        parameter.requires_grad_(False)
    log.extend(
        _train(
            matched,
            unordered_residual,
            residual,
            outer_segments,
            stage="chronological_residual",
            cycles=args.history_cycles,
            batch_segments=args.segment_batch_size,
            chunk_length=args.bptt_chunk,
            learning_rate=args.learning_rate,
            rank_weight=args.rank_weight,
            seed=args.seed + 2000,
            device=device,
        )
    )
    residual.eval()
    frames = [
        _evaluate_condition(
            matched,
            unordered_residual,
            residual,
            heldout_segments,
            condition=condition,
            batch_segments=args.segment_batch_size,
            chunk_length=args.bptt_chunk,
            rank_weight=args.rank_weight,
            seed=args.seed + 3000,
            device=device,
        )
        for condition in (
            "static_prior",
            "base_unordered",
            "matched_unordered",
            "chronological_history",
            "across_event_order_shuffle",
        )
    ]
    metrics = pd.concat(frames, ignore_index=True)
    aggregate = {
        model: {
            "n_events": int(len(group)),
            "participation_bce": float(group.participation_bce.mean()),
            "relative_rank_huber": float(group.relative_rank_huber.mean()),
        }
        for model, group in metrics.groupby("model")
    }
    aggregate["contrasts"] = {
        "static_minus_matched_participation_bce": (
            aggregate["static_prior"]["participation_bce"]
            - aggregate["matched_unordered"]["participation_bce"]
        ),
        "static_minus_chronological_participation_bce": (
            aggregate["static_prior"]["participation_bce"]
            - aggregate["chronological_history"]["participation_bce"]
        ),
        "matched_minus_chronological_participation_bce": (
            aggregate["matched_unordered"]["participation_bce"]
            - aggregate["chronological_history"]["participation_bce"]
        ),
        "matched_minus_chronological_relative_rank_huber": (
            aggregate["matched_unordered"]["relative_rank_huber"]
            - aggregate["chronological_history"]["relative_rank_huber"]
        ),
        "shuffle_minus_chronological_participation_bce": (
            aggregate["across_event_order_shuffle"]["participation_bce"]
            - aggregate["chronological_history"]["participation_bce"]
        ),
    }
    pd.DataFrame(log).to_csv(output / "training_log.csv", index=False)
    metrics.to_csv(output / "heldout_event_metrics.csv", index=False)
    torch.save(
        {
            "contract": "topic5_history_rnn_early_ictal_field_v0_1_g1_sequential",
            "heldout_subject": args.heldout_subject,
            "seed": args.seed,
            "matched_state": matched.state_dict(),
            "unordered_residual_state": unordered_residual.state_dict(),
            "history_state": residual.state_dict(),
            "event_embedding_mean": torch.as_tensor(mean),
            "event_embedding_scale": torch.as_tensor(scale),
            "config": vars(args),
            "ictal_target_read": False,
        },
        output / "checkpoint.pt",
    )
    done = {
        "status": "COMPLETE",
        "contract": "topic5_history_rnn_early_ictal_field_v0_1_g1_sequential",
        "heldout_subject": args.heldout_subject,
        "seed": args.seed,
        "n_outer_patients": len(outer_records),
        "n_outer_segments": len(outer_segments),
        "n_heldout_segments": len(heldout_segments),
        "metrics": aggregate,
        "parameter_counts": {
            "base_unordered": int(sum(p.numel() for p in matched.parameters())),
            "unordered_residual": int(
                sum(p.numel() for p in unordered_residual.parameters())
            ),
            "chronological_residual": int(
                sum(p.numel() for p in residual.parameters())
            ),
            "unordered_residual_state_dim": int(unordered_dim),
            "chronological_state_dim": int(args.history_dim),
        },
        "state_reset_contract": "segment_boundaries_only; BPTT detaches but does not zero",
        "target_values_read": False,
        "event_checkpoint": str(event_checkpoint),
        "event_checkpoint_sha256": sha256(event_checkpoint),
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output / "DONE.json").write_text(
        json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
