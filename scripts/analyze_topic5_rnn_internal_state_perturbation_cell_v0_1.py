#!/usr/bin/env python3
"""Matched order and hidden-direction perturbations for one subject/seed cell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_topic5_rnn_internal_states_v0_1 import (  # noqa: E402
    CONTROLS,
    SEED_DIRS,
    load_model,
    load_subject,
)
from src.topic5_rnn_internal_state import event_first_mean  # noqa: E402


BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
AMPLITUDES = (0.25, 0.5, 1.0)
N_ORDER_SHUFFLES = 5


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def decoder_inputs(model, record, offset, device):
    features = torch.as_tensor(
        record["contact_features"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.ones(
        (1, features.shape[1]), dtype=torch.bool, device=device
    )
    embedding, encoder_input = model._encode(features, offset)
    return embedding, encoder_input, mask


@torch.no_grad()
def action_probability(
    model,
    embedding,
    encoder_input,
    hidden: np.ndarray,
    candidate: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    device = next(model.parameters()).device
    result = []
    states = np.asarray(hidden, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=bool)
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        state = torch.as_tensor(states[start:stop], device=device)
        mask = torch.as_tensor(candidate[start:stop], device=device)
        logits, stop_logits = model._decode(
            embedding.expand(len(state), -1, -1),
            encoder_input.expand(len(state), -1, -1),
            state,
            mask,
        )
        probability = torch.softmax(
            torch.cat([stop_logits[:, None], logits], dim=1), dim=1
        )
        result.append(probability.cpu().numpy())
    return np.row_stack(result).astype(np.float64)


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(left, dtype=np.float64), 1.0e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=np.float64), 1.0e-12, 1.0)
    middle = 0.5 * (left + right)
    return 0.5 * np.sum(left * np.log(left / middle), axis=1) + 0.5 * np.sum(
        right * np.log(right / middle), axis=1
    )


def event_first_vector(values: np.ndarray, events: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    event_ids = np.asarray(events, dtype=np.int64)
    unique, inverse = np.unique(event_ids, return_inverse=True)
    event_mean = np.zeros((len(unique), matrix.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    for column in range(matrix.shape[1]):
        event_mean[:, column] = (
            np.bincount(inverse, weights=matrix[:, column]) / counts
        )
    return event_mean.mean(axis=0)


def prefix_arrays(record, events: np.ndarray, steps: np.ndarray):
    groups = record["group_ids"][events]
    counts = record["group_count"][events]
    recruited = (groups >= 0) & (groups < steps[:, None])
    candidate = ~recruited
    progress = recruited.sum(1) / float(groups.shape[1])
    target = np.zeros(len(steps), dtype=np.int64)
    nonterminal = steps < counts
    target[nonterminal] = 1 + np.argmax(
        groups[nonterminal] == steps[nonterminal, None], axis=1
    )
    return groups, counts, candidate, progress, target


def probability_metrics(
    actual: np.ndarray,
    perturbed: np.ndarray,
    target: np.ndarray,
    event: np.ndarray,
    progress: np.ndarray,
) -> list[dict]:
    actual_loss = -np.log(
        np.clip(actual[np.arange(len(target)), target], 1.0e-12, 1.0)
    )
    perturbed_loss = -np.log(
        np.clip(perturbed[np.arange(len(target)), target], 1.0e-12, 1.0)
    )
    difference = perturbed_loss - actual_loss
    js = js_divergence(actual, perturbed)
    stop_shift = perturbed[:, 0] - actual[:, 0]
    bins = {
        "all": np.ones(len(progress), dtype=bool),
        "early": progress <= (1.0 / 3.0),
        "middle": (progress > (1.0 / 3.0)) & (progress <= (2.0 / 3.0)),
        "late": progress > (2.0 / 3.0),
    }
    rows = []
    for prefix_bin, mask in bins.items():
        if not np.any(mask):
            continue
        for metric, values in (
            ("nll_loss", difference),
            ("js_divergence", js),
            ("stop_probability_shift", stop_shift),
        ):
            rows.append(
                {
                    "prefix_bin": prefix_bin,
                    "metric": metric,
                    "value": event_first_mean(values[mask], event[mask]),
                    "n_prefixes": int(np.count_nonzero(mask)),
                    "n_events": int(len(np.unique(event[mask]))),
                }
            )
    return rows


@torch.no_grad()
def matched_order_hidden(
    model,
    record,
    offset,
    selected_events: np.ndarray,
    reference_event: np.ndarray,
    reference_step: np.ndarray,
    *,
    mode: str,
    shuffle_index: int,
) -> np.ndarray:
    """Recompute each prefix while preserving its observed member set."""
    device = next(model.parameters()).device
    embedding_one, _, mask_one = decoder_inputs(model, record, offset, device)
    groups_all = record["group_ids"]
    counts_all = record["group_count"]
    hidden_lookup: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(
        2026072800 + int(shuffle_index) * 1009 + int(np.sum(selected_events) % 100003)
    )
    for start in range(0, len(selected_events), 256):
        events = selected_events[start : start + 256]
        groups_np = np.asarray(groups_all[events], dtype=np.int64)
        counts_np = np.asarray(counts_all[events], dtype=np.int64)
        batch = len(events)
        embedding = embedding_one.expand(batch, -1, -1)
        contact_mask = mask_one.expand(batch, -1)
        maximum = int(counts_np.max())
        for step in range(maximum + 1):
            active_np = counts_np >= step
            if not np.any(active_np):
                continue
            active_events = events[active_np]
            active_groups_np = groups_np[active_np]
            active_embedding = embedding[torch.as_tensor(active_np, device=device)]
            active_mask = contact_mask[torch.as_tensor(active_np, device=device)]
            hidden = model._initial_hidden(active_embedding, active_mask)
            recruited = torch.zeros_like(active_mask)
            if step:
                if mode == "reverse":
                    order = np.tile(
                        np.arange(step - 1, -1, -1, dtype=np.int64),
                        (len(active_events), 1),
                    )
                elif mode == "shuffle":
                    order = np.row_stack(
                        [rng.permutation(step) for _ in active_events]
                    )
                else:
                    raise ValueError(f"unknown order mode: {mode}")
                active_groups = torch.as_tensor(
                    active_groups_np, dtype=torch.long, device=device
                )
                for position in range(step):
                    rank = torch.as_tensor(
                        order[:, position], dtype=torch.long, device=device
                    )
                    current = active_groups == rank[:, None]
                    recruited = recruited | current
                    hidden = model._advance(
                        active_embedding,
                        current,
                        recruited,
                        hidden,
                        active_mask,
                    )
            for row, event in enumerate(active_events):
                hidden_lookup[(int(event), int(step))] = (
                    hidden[row].cpu().numpy().astype(np.float32)
                )
    return np.row_stack(
        [
            hidden_lookup[(int(event), int(step))]
            for event, step in zip(reference_event, reference_step)
        ]
    )


def align_direction(
    direction: np.ndarray,
    contact_loading: np.ndarray,
    participation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    vector = np.asarray(direction, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    field = contact_loading @ vector
    correlation = float(np.corrcoef(field, participation)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        vector = -vector
        field = -field
        correlation = -correlation
    return vector, field, correlation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed-dir", required=True, choices=SEED_DIRS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-order-events", type=int, default=512)
    args = parser.parse_args()
    started = time.time()
    out = BASE / "interictal/perturbation_cells" / args.seed_dir / args.subject
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "CELL_STATUS.json"
    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "subject": args.subject,
            "seed_dir": args.seed_dir,
            "target_values_read": False,
        },
    )
    device = torch.device(args.device)
    record = load_subject(args.subject)
    hidden_path = (
        BASE
        / "interictal/cells"
        / args.seed_dir
        / args.subject
        / "hidden_states.npz"
    )
    with np.load(hidden_path, allow_pickle=False) as data:
        cell = {key: np.asarray(data[key]) for key in data.files}
    event = cell["heldout20_event_index"].astype(np.int64)
    step = cell["heldout20_step"].astype(np.int64)
    groups, counts, candidate, progress, target = prefix_arrays(
        record, event, step
    )
    selected_events = np.unique(event)[: int(args.max_order_events)]
    order_mask = np.isin(event, selected_events)
    order_rows = []
    direction_rows = []
    contact_rows = []
    train_groups = record["group_ids"][np.flatnonzero(record["event_split"] == 0)]
    participation = np.mean(train_groups >= 0, axis=0)
    for control in CONTROLS:
        model, offset, _ = load_model(
            args.subject,
            args.seed_dir,
            control,
            record["contact_features"].shape[1],
            device,
        )
        embedding, encoder_input, _ = decoder_inputs(
            model, record, offset, device
        )
        hidden = cell[f"{control}_heldout20_hidden"].astype(np.float32)
        actual_probability = action_probability(
            model, embedding, encoder_input, hidden, candidate
        )

        for mode in ("reverse", "shuffle"):
            repeats = 1 if mode == "reverse" else N_ORDER_SHUFFLES
            repeated_metrics = []
            for repeat in range(repeats):
                perturbed_hidden = matched_order_hidden(
                    model,
                    record,
                    offset,
                    selected_events,
                    event[order_mask],
                    step[order_mask],
                    mode=mode,
                    shuffle_index=repeat,
                )
                probability = action_probability(
                    model,
                    embedding,
                    encoder_input,
                    perturbed_hidden,
                    candidate[order_mask],
                )
                metrics = probability_metrics(
                    actual_probability[order_mask],
                    probability,
                    target[order_mask],
                    event[order_mask],
                    progress[order_mask],
                )
                for row in metrics:
                    row["repeat"] = repeat
                    repeated_metrics.append(row)
            frame = pd.DataFrame(repeated_metrics)
            for (prefix_bin, metric), group in frame.groupby(
                ["prefix_bin", "metric"]
            ):
                order_rows.append(
                    {
                        "subject": args.subject,
                        "seed_dir": args.seed_dir,
                        "control": control,
                        "order_perturbation": mode,
                        "prefix_bin": prefix_bin,
                        "metric": metric,
                        "value": float(group.value.median()),
                        "n_prefixes": int(group.n_prefixes.max()),
                        "n_events": int(group.n_events.max()),
                        "n_repeats": repeats,
                    }
                )

        hidden_dim = hidden.shape[1]
        centered_loading = (
            embedding[0].cpu().numpy()
            @ model.action_query.weight.detach().cpu().numpy()
            / np.sqrt(float(model.contact_embedding_dim))
        )
        centered_loading = centered_loading - centered_loading.mean(
            axis=0, keepdims=True
        )
        _, _, right = np.linalg.svd(centered_loading, full_matrices=False)
        pca_components = cell[f"{control}_pca_components"].astype(np.float64)
        pca_eigenvalues = cell[f"{control}_pca_eigenvalues"].astype(np.float64)
        train_hidden = cell[f"{control}_train60_hidden"].astype(np.float64)
        direction_specs = []
        for index in range(min(2, hidden_dim)):
            direction_specs.append(
                (
                    "pca",
                    index + 1,
                    pca_components[index],
                    np.sqrt(max(pca_eigenvalues[index], 1.0e-12)),
                )
            )
        for index in range(min(2, right.shape[0])):
            vector = right[index]
            scale = float(np.std(train_hidden @ vector, ddof=1))
            direction_specs.append(("output_coupled", index + 1, vector, scale))
        for direction_type, direction_index, vector, scale in direction_specs:
            vector, field, sign_correlation = align_direction(
                vector, centered_loading, participation
            )
            for amplitude in AMPLITUDES:
                delta = float(amplitude * scale) * vector
                plus = action_probability(
                    model,
                    embedding,
                    encoder_input,
                    hidden + delta[None, :],
                    candidate,
                )
                minus = action_probability(
                    model,
                    embedding,
                    encoder_input,
                    hidden - delta[None, :],
                    candidate,
                )
                js_plus = event_first_mean(
                    js_divergence(actual_probability, plus), event
                )
                js_minus = event_first_mean(
                    js_divergence(actual_probability, minus), event
                )
                stop_contrast = event_first_mean(
                    plus[:, 0] - minus[:, 0], event
                )
                direction_rows.append(
                    {
                        "subject": args.subject,
                        "seed_dir": args.seed_dir,
                        "control": control,
                        "direction_type": direction_type,
                        "direction_index": direction_index,
                        "amplitude_sd": amplitude,
                        "direction_state_sd": scale,
                        "sign_alignment_interictal_participation_r": sign_correlation,
                        "mean_js_plus": js_plus,
                        "mean_js_minus": js_minus,
                        "stop_probability_plus_minus": stop_contrast,
                    }
                )
                contrast = plus[:, 1:] - minus[:, 1:]
                unique_events = np.unique(event)
                cut = len(unique_events) // 2
                half_masks = {
                    "all": np.ones(len(event), dtype=bool),
                    "first": np.isin(event, unique_events[:cut]),
                    "second": np.isin(event, unique_events[cut:]),
                }
                for event_half, half_mask in half_masks.items():
                    probability_field = event_first_vector(
                        contrast[half_mask], event[half_mask]
                    )
                    for contact_index, contact_name in enumerate(
                        record["contact_names"]
                    ):
                        contact_rows.append(
                            {
                                "subject": args.subject,
                                "seed_dir": args.seed_dir,
                                "control": control,
                                "direction_type": direction_type,
                                "direction_index": direction_index,
                                "amplitude_sd": amplitude,
                                "event_half": event_half,
                                "contact_index": contact_index,
                                "contact_name": str(contact_name),
                                "decoder_loading": float(field[contact_index]),
                                "probability_contrast": float(
                                    probability_field[contact_index]
                                ),
                                "train80_participation": float(
                                    participation[contact_index]
                                ),
                            }
                        )
        del model

    pd.DataFrame(order_rows).to_csv(out / "order_perturbation_metrics.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(
        out / "direction_perturbation_metrics.csv", index=False
    )
    pd.DataFrame(contact_rows).to_csv(
        out / "direction_contact_fields.csv", index=False
    )
    atomic_json(
        status_path,
        {
            "contract": "topic5_rnn_internal_state_reduction_v0_1",
            "status": "COMPLETE",
            "subject": args.subject,
            "seed_dir": args.seed_dir,
            "n_order_rows": len(order_rows),
            "n_direction_rows": len(direction_rows),
            "n_contact_rows": len(contact_rows),
            "runtime_seconds": float(time.time() - started),
            "target_values_read": False,
            "early_ictal_arrays_deserialized": False,
        },
    )


if __name__ == "__main__":
    main()
