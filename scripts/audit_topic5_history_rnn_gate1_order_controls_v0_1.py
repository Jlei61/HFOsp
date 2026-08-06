#!/usr/bin/env python3
"""Run causal-prefix-matched and within-event order controls for one G1 fold."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from scripts.run_topic5_history_rnn_gate1_sequential_fold_v0_1 import (  # noqa: E402
    MatchedSequentialModel,
    ResidualSequentialModel,
    UnorderedResidualSequentialModel,
    _add_predictions,
    _capacity_matched_unordered_dim,
    _evaluate_condition,
    _segments,
)
from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_history_data import encode_subject  # noqa: E402
from src.topic5_history_rnn import (  # noqa: E402
    next_event_field_loss,
    prefix_matched_order_indices,
)
from src.topic5_rank_distribution import LinearStateSequenceRNN  # noqa: E402


@torch.no_grad()
def _true_states(
    model: ResidualSequentialModel,
    embedding: np.ndarray,
    event_time: np.ndarray,
    *,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    state = None
    for start in range(0, len(embedding), int(chunk)):
        stop = min(start + int(chunk), len(embedding))
        current = torch.as_tensor(embedding[start:stop], device=device).unsqueeze(0)
        delta = np.zeros(stop - start, dtype=np.float32)
        if start > 0:
            delta[0] = float(event_time[start] - event_time[start - 1])
        if stop - start > 1:
            delta[1:] = np.diff(event_time[start:stop]).astype(np.float32)
        reset = torch.zeros((1, stop - start), dtype=torch.bool, device=device)
        if start == 0:
            reset[:, 0] = True
        states, state = model.history.forward_masked(
            current,
            torch.as_tensor(delta, device=device).unsqueeze(0),
            reset,
            torch.ones((1, stop - start), dtype=torch.bool, device=device),
            initial_state=state,
        )
        outputs.append(states[0])
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def _prefix_matched_shuffle(
    matched: MatchedSequentialModel,
    chronological: ResidualSequentialModel,
    segments,
    *,
    window: int,
    batch_size: int,
    rank_weight: float,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    generator = np.random.default_rng(int(seed))
    rows = []
    for segment_index, segment in enumerate(segments):
        true_state = _true_states(
            chronological,
            segment.embedding,
            segment.event_time,
            chunk=512,
            device=device,
        )
        targets = np.flatnonzero(segment.event_split == 1)
        targets = targets[targets > 0]
        for offset in range(0, len(targets), int(batch_size)):
            current_targets = targets[offset : offset + int(batch_size)]
            if not len(current_targets):
                continue
            lengths = np.minimum(current_targets, int(window)).astype(int)
            max_length = int(lengths.max())
            embeddings = np.zeros(
                (len(current_targets), max_length, segment.embedding.shape[1]),
                dtype=np.float32,
            )
            delta = np.zeros((len(current_targets), max_length), dtype=np.float32)
            mask = np.zeros((len(current_targets), max_length), dtype=bool)
            reset = np.zeros((len(current_targets), max_length), dtype=bool)
            initial = torch.zeros(
                (len(current_targets), chronological.history.history_dim),
                dtype=torch.float32,
                device=device,
            )
            for row, target in enumerate(current_targets):
                start, shuffled_indices = prefix_matched_order_indices(
                    int(target), window=window, rng=generator
                )
                size = len(shuffled_indices)
                embeddings[row, :size] = segment.embedding[shuffled_indices]
                mask[row, :size] = True
                slots = np.arange(start, int(target))
                if start == 0:
                    reset[row, 0] = True
                else:
                    initial[row] = true_state[start - 1]
                    delta[row, 0] = float(
                        segment.event_time[start] - segment.event_time[start - 1]
                    )
                if size > 1:
                    delta[row, 1:size] = np.diff(
                        segment.event_time[slots]
                    ).astype(np.float32)
            _, final = chronological.history.forward_masked(
                torch.as_tensor(embeddings, device=device),
                torch.as_tensor(delta, device=device),
                torch.as_tensor(reset, device=device),
                torch.as_tensor(mask, device=device),
                initial_state=initial,
            )
            contact = torch.as_tensor(
                segment.contact_embedding, device=device
            ).unsqueeze(0).expand(len(current_targets), -1, -1)
            static = torch.as_tensor(
                segment.static_logit, device=device
            ).unsqueeze(0).expand(len(current_targets), -1)
            summary = torch.as_tensor(
                segment.unordered_summary[current_targets - 1], device=device
            )
            base_sequence = matched(summary[:, None], contact, static)
            base = {key: value[:, 0] for key, value in base_sequence.items()}
            # The control is contrasted against M2 = frozen base + chronological
            # residual.  Adding the unordered residual here would make the
            # shuffled arm a different model, so the contrast would measure
            # composition rather than event order.
            prediction = _add_predictions(base, chronological.heads(final, contact))
            target_participation = torch.as_tensor(
                segment.participation[current_targets], device=device
            )
            target_rank = torch.as_tensor(
                segment.relative_rank[current_targets], device=device
            )
            loss = next_event_field_loss(
                prediction,
                target_participation,
                target_rank,
                rank_weight=rank_weight,
            )
            bce = loss["event_participation_bce"].cpu().numpy()
            rank = loss["event_relative_rank_huber"].cpu().numpy()
            for row, target in enumerate(current_targets):
                rows.append(
                    {
                        "subject": segment.subject,
                        "dataset": segment.dataset,
                        "segment_index": int(segment_index),
                        "event_index": int(segment.original_index[target]),
                        "model": f"prefix_matched_order_shuffle_k{int(window)}",
                        "participation_bce": float(bce[row]),
                        "relative_rank_huber": float(rank[row]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    fold = args.fold_dir.resolve()
    done = json.loads((fold / "DONE.json").read_text())
    if bool(done.get("target_values_read", True)):
        raise RuntimeError("G1 target seal violated")
    checkpoint = torch.load(fold / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    subject = checkpoint["heldout_subject"]
    seed = int(checkpoint["seed"])
    artifact = args.artifact_root.resolve()
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    event_payload = torch.load(
        Path(done["event_checkpoint"]), map_location="cpu", weights_only=False
    )
    event_model = LinearStateSequenceRNN(**event_payload["model_kwargs"])
    event_model.load_state_dict(event_payload["model_state"])
    event_model.to(device).eval()
    source = load_records(
        artifact / "results/topic5_interictal_rank_distribution/dataset_v0_4"
    )[subject]
    record = encode_subject(
        event_model,
        source,
        artifact_root=artifact,
        device=device,
        batch_size=int(config["embedding_batch_size"]),
    )
    mean = checkpoint["event_embedding_mean"].numpy()
    scale = checkpoint["event_embedding_scale"].numpy()
    segments = _segments([record], mean=mean, scale=scale)
    event_dim = int(record.event_embedding.shape[1])
    contact_dim = int(record.contact_embedding.shape[1])
    history_dim = int(config["history_dim"])
    unordered_dim = _capacity_matched_unordered_dim(
        event_dim, contact_dim, history_dim
    )
    matched = MatchedSequentialModel(event_dim, history_dim, contact_dim).to(device)
    unordered = UnorderedResidualSequentialModel(
        event_dim, unordered_dim, contact_dim
    ).to(device)
    chronological = ResidualSequentialModel(
        event_dim,
        history_dim,
        contact_dim,
        initial_half_life_hours=float(config["initial_half_life_hours"]),
    ).to(device)
    matched.load_state_dict(checkpoint["matched_state"])
    unordered.load_state_dict(checkpoint["unordered_residual_state"])
    chronological.load_state_dict(checkpoint["history_state"])
    matched.eval()
    unordered.eval()
    chronological.eval()

    exact = _prefix_matched_shuffle(
        matched,
        chronological,
        segments,
        window=args.window,
        batch_size=args.batch_size,
        rank_weight=float(config["rank_weight"]),
        seed=seed + 4100,
        device=device,
    )

    shuffled_record = encode_subject(
        event_model,
        source,
        artifact_root=artifact,
        device=device,
        batch_size=int(config["embedding_batch_size"]),
        within_event_rank_shuffle_seed=seed + 4200,
    )
    # Rank shuffle is an input perturbation only; scoring targets remain the
    # original next-event participation and relative-rank fields.
    shuffled_record.relative_rank = record.relative_rank
    shuffled_segments = _segments([shuffled_record], mean=mean, scale=scale)
    within = _evaluate_condition(
        matched,
        unordered,
        chronological,
        shuffled_segments,
        condition="chronological_history",
        batch_segments=16,
        chunk_length=256,
        rank_weight=float(config["rank_weight"]),
        seed=seed + 4300,
        device=device,
    )
    within["model"] = "within_event_rank_shuffle"
    controls = pd.concat([exact, within], ignore_index=True)
    controls.to_csv(fold / "heldout_order_controls.csv", index=False)
    exact_mean = exact[["participation_bce", "relative_rank_huber"]].mean()
    within_mean = within[["participation_bce", "relative_rank_huber"]].mean()
    chronological_mean = done["metrics"]["chronological_history"]
    rate = chronological.history.decay_rate_per_second.detach().cpu().numpy()
    half_life_hours = np.log(2.0) / np.maximum(rate, 1e-12) / 3600.0
    result = {
        "status": "COMPLETE",
        "contract": "topic5_history_rnn_early_ictal_field_v0_1_g1_order_controls",
        "subject": subject,
        "target_values_read": False,
        "prefix_order_window_events": int(args.window),
        "prefix_control": "same causal prefix event multiset and timestamp slots; earlier true state frozen",
        "n_prefix_matched_decisions": int(len(exact)),
        "n_within_event_rank_shuffle_decisions": int(len(within)),
        "learned_event_history_half_life_hours": {
            "minimum": float(np.min(half_life_hours)),
            "median": float(np.median(half_life_hours)),
            "maximum": float(np.max(half_life_hours)),
        },
        "metrics": {
            "chronological_history": chronological_mean,
            "prefix_matched_order_shuffle": {
                "participation_bce": float(exact_mean.participation_bce),
                "relative_rank_huber": float(exact_mean.relative_rank_huber),
            },
            "within_event_rank_shuffle": {
                "participation_bce": float(within_mean.participation_bce),
                "relative_rank_huber": float(within_mean.relative_rank_huber),
            },
            "contrasts": {
                "prefix_matched_shuffle_minus_chronological_bce": float(
                    exact_mean.participation_bce
                    - chronological_mean["participation_bce"]
                ),
                "within_event_rank_shuffle_minus_chronological_bce": float(
                    within_mean.participation_bce
                    - chronological_mean["participation_bce"]
                ),
            },
        },
    }
    (fold / "ORDER_CONTROLS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
