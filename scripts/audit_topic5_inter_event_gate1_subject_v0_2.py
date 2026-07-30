#!/usr/bin/env python3
"""Gate 1 feasibility: can causal inter-event context predict the next event?

This is a fixed linear-probability screening audit, not a new RNN.  The target
is the next event's contact-participation vector.  Every fitted baseline uses
chronological train80 only; heldout20 is evaluated only for within-recording
pairs separated by at most 10 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
MAX_GAP_SECONDS = 600.0
RECENT_EVENTS = 8
EWMA_HALF_LIFE_SECONDS = 3600.0
PRIOR_STRENGTH = 2.0
NULL_DRAWS = 200
NULL_MAX_EVENTS = 5000


def _causal_features(participation: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    n_events, n_contacts = participation.shape
    last = np.zeros((n_events, n_contacts), dtype=np.float32)
    recent = np.zeros_like(last)
    ewma = np.zeros_like(last)
    log_iei = np.zeros(n_events, dtype=np.float32)
    log_rate = np.zeros(n_events, dtype=np.float32)
    phase_sin = np.sin(2 * np.pi * ((times % 86400.0) / 86400.0)).astype(np.float32)
    phase_cos = np.cos(2 * np.pi * ((times % 86400.0) / 86400.0)).astype(np.float32)
    cumulative = np.row_stack(
        [np.zeros((1, n_contacts), dtype=np.float64), np.cumsum(participation, axis=0)]
    )
    state = np.zeros(n_contacts, dtype=np.float64)
    state_weight = 0.0
    segment_start = 0
    decay_constant = np.log(2.0) / EWMA_HALF_LIFE_SECONDS
    for event_index in range(1, n_events):
        gap = max(float(times[event_index] - times[event_index - 1]), 0.0)
        if gap <= 0 or gap > MAX_GAP_SECONDS:
            # Recording discontinuities are not latent biological time.  Start
            # a new causal segment and do not carry participation history
            # across the gap.
            segment_start = event_index
            state.fill(0.0)
            state_weight = 0.0
            log_iei[event_index] = np.log1p(gap)
            continue
        last[event_index] = participation[event_index - 1]
        start = max(segment_start, event_index - RECENT_EVENTS)
        recent[event_index] = (
            cumulative[event_index] - cumulative[start]
        ) / max(event_index - start, 1)
        decay = np.exp(-decay_constant * gap)
        state = decay * state + participation[event_index - 1]
        state_weight = decay * state_weight + 1.0
        ewma[event_index] = state / max(state_weight, 1e-8)
        log_iei[event_index] = np.log1p(gap)
        left = max(
            segment_start,
            int(np.searchsorted(times, times[event_index] - 600.0, side="left")),
        )
        log_rate[event_index] = np.log1p(event_index - left)
    return {
        "last": last,
        "recent": recent,
        "ewma": ewma,
        "scalar": np.column_stack(
            [log_iei, log_rate, phase_sin, phase_cos]
        ).astype(np.float32),
    }


def _fit_contact_models(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    static_prior: np.ndarray,
) -> np.ndarray:
    x_train = np.asarray(features[train_mask], np.float64)
    x_eval = np.asarray(features[eval_mask], np.float64)
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-6] = 1.0
    x_train = (x_train - mean) / scale
    x_eval = (x_eval - mean) / scale
    predictions = np.zeros((len(x_eval), targets.shape[1]), dtype=np.float64)
    for contact in range(targets.shape[1]):
        y = targets[train_mask, contact].astype(np.uint8)
        if np.unique(y).size < 2:
            predictions[:, contact] = static_prior[contact]
            continue
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=100,
            tol=1e-6,
            random_state=20260730,
        )
        model.fit(x_train, y)
        predictions[:, contact] = model.predict_proba(x_eval)[:, 1]
    return np.clip(predictions, 1e-6, 1 - 1e-6)


def _contactwise_features(
    context: dict[str, np.ndarray],
    contact: int,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "last_event":
        return context["last"][:, [contact]]
    if mode == "recent_unordered":
        return context["recent"][:, [contact]]
    if mode == "scalar_context":
        return context["scalar"]
    if mode == "time_state":
        return np.column_stack(
            [
                context["last"][:, contact],
                context["recent"][:, contact],
                context["ewma"][:, contact],
                context["scalar"],
            ]
        )
    raise ValueError(mode)


def _fit_mode(
    context: dict[str, np.ndarray],
    targets: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    static_prior: np.ndarray,
    mode: str,
) -> np.ndarray:
    predictions = np.zeros((int(np.sum(eval_mask)), targets.shape[1]), dtype=np.float64)
    for contact in range(targets.shape[1]):
        x = _contactwise_features(context, contact, mode=mode)
        predictions[:, contact] = _fit_contact_models(
            x,
            targets[:, [contact]],
            train_mask,
            eval_mask,
            static_prior=static_prior[[contact]],
        )[:, 0]
    return predictions


def _event_bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    y = np.asarray(target, float)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    records = load_records(DATASET)
    record = records[args.subject]
    with np.load(record.path, allow_pickle=False) as data:
        participation = np.asarray(data["event_participation"], np.uint8)
        times = np.asarray(data["event_abs_time"], np.float64)
        split = np.asarray(data["event_split"], np.uint8)
    if np.any(np.diff(times) < 0):
        raise RuntimeError("event time order drifted")
    gaps = np.diff(times, prepend=np.nan)
    valid_pair = np.isfinite(gaps) & (gaps > 0) & (gaps <= MAX_GAP_SECONDS)
    train_mask = valid_pair & (split == 0)
    eval_mask = valid_pair & (split == 1)
    if np.sum(train_mask) < 50 or np.sum(eval_mask) < 20:
        raise RuntimeError("insufficient continuous train/eval event pairs")

    context = _causal_features(participation, times)
    train_targets = participation[train_mask]
    static_prior = (
        train_targets.sum(axis=0) + 0.5
    ) / (len(train_targets) + 1.0)
    target_eval = participation[eval_mask]
    predictions = {
        "static_prior": np.broadcast_to(
            static_prior, (len(target_eval), len(static_prior))
        ).copy()
    }
    for mode in ("last_event", "recent_unordered", "scalar_context", "time_state"):
        predictions[mode] = _fit_mode(
            context,
            participation,
            train_mask,
            eval_mask,
            static_prior=static_prior,
            mode=mode,
        )

    rows = []
    losses = {}
    for mode, probability in predictions.items():
        event_loss = _event_bce(probability, target_eval)
        losses[mode] = event_loss
        rows.append(
            {
                "subject": record.subject,
                "dataset": record.dataset,
                "model": mode,
                "n_train_pairs": int(np.sum(train_mask)),
                "n_eval_pairs": int(np.sum(eval_mask)),
                "mean_contact_bce": float(np.mean(event_loss)),
                "median_event_contact_bce": float(np.median(event_loss)),
            }
        )
    static_loss = float(np.mean(losses["static_prior"]))
    observed_gain = static_loss - float(np.mean(losses["time_state"]))

    rng = np.random.default_rng(
        20260730 + sum(map(ord, record.subject))
    )
    probability = predictions["time_state"]
    target = target_eval
    if len(target) > NULL_MAX_EVENTS:
        audit_index = np.sort(
            rng.choice(len(target), size=NULL_MAX_EVENTS, replace=False)
        )
        probability = probability[audit_index]
        target = target[audit_index]
    static_probability = np.broadcast_to(
        static_prior, (len(target), len(static_prior))
    )
    static_null_reference = float(np.mean(_event_bce(static_probability, target)))
    observed_null_subset_gain = (
        static_null_reference - float(np.mean(_event_bce(probability, target)))
    )
    circular = []
    block = []
    for _ in range(NULL_DRAWS):
        shift = int(rng.integers(1, max(len(target), 2)))
        shifted = np.roll(probability, shift, axis=0)
        circular.append(static_null_reference - float(np.mean(_event_bce(shifted, target))))
        permutation = np.arange(len(target))
        for start in range(0, len(target), 256):
            stop = min(start + 256, len(target))
            permutation[start:stop] = rng.permutation(permutation[start:stop])
        block.append(
            static_null_reference
            - float(np.mean(_event_bce(probability[permutation], target)))
        )

    metric_frame = pd.DataFrame(rows)
    metric_frame["gain_over_static_nats_per_contact"] = (
        static_loss - metric_frame.mean_contact_bce
    )
    metric_frame.to_csv(output / "model_metrics.csv", index=False)
    np.savez_compressed(
        output / "pairing_nulls.npz",
        circular_gain=np.asarray(circular, np.float32),
        block_gain=np.asarray(block, np.float32),
    )
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "subject": record.subject,
        "dataset": record.dataset,
        "target": "next_event_contact_participation",
        "n_train_pairs": int(np.sum(train_mask)),
        "n_eval_pairs": int(np.sum(eval_mask)),
        "pair_gap_max_seconds": MAX_GAP_SECONDS,
        "recent_unordered_events": RECENT_EVENTS,
        "time_state": "fixed one-hour half-life EWMA plus last/recent/scalar context",
        "observed_time_state_gain_over_static": observed_gain,
        "observed_null_subset_gain_over_static": observed_null_subset_gain,
        "circular_null_p_greater": float(
            (1 + np.sum(np.asarray(circular) >= observed_null_subset_gain))
            / (NULL_DRAWS + 1)
        ),
        "block_null_p_greater": float(
            (1 + np.sum(np.asarray(block) >= observed_null_subset_gain))
            / (NULL_DRAWS + 1)
        ),
        "early_ictal_target_read": False,
        "interpretation": "self_supervised_feasibility_only_not_recurrent_state_identification",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
