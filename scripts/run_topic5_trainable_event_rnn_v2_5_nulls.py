#!/usr/bin/env python3
"""Coherent chronology controls for the frozen trainable event RNN v2.5.4."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_trainable_event_rnn_v2_5 import (  # noqa: E402
    _build_dataset,
    fit_baseline,
    jsonable,
    load_subject,
    prepare_subject,
    score_dict,
)
from src.topic5_stable_repertoire_event_history_v2_4 import (  # noqa: E402
    build_event_history_dataset,
    chronological_sequences,
    family_scales_from_train,
    score_v24,
    source_coherent_block_shuffle,
)
from src.topic5_trainable_event_rnn_v2_5 import (  # noqa: E402
    fit_trainable_residual_rnn,
    partition_indices,
    profile_from_mapping,
    window_balanced_source_partition,
)


DEFAULT_CONFIG = ROOT / "config/topic5_trainable_event_rnn_v2_5.yaml"


def split_sequences(subject: str, config: dict[str, Any]):
    raw, encoder, true_datasets, partition, _, audit = prepare_subject(
        subject, config, final_fit=False
    )
    indices = {
        split: partition_indices(raw["source"], raw["eligible"], partition, split)
        for split in ("train", "validation", "test")
    }
    if len(indices["validation"]) == 0:
        ordered = indices["train"][
            np.argsort(raw["event_time"][indices["train"]], kind="mergesort")
        ]
        cut = int(np.floor(0.7 * len(ordered)))
        indices["train"] = ordered[:cut]
        indices["validation"] = ordered[cut:]
    sequences = {
        split: chronological_sequences(raw["source"], raw["event_time"], indices[split])
        for split in indices
    }
    tokens, modes = encoder.event_tokens(raw["rank"], raw["participation"])
    return raw, encoder, tokens, modes, true_datasets, sequences, audit


def build_from_sequences(raw, encoder, tokens, modes, sequences, config):
    output = {}
    for split in ("train", "validation", "test"):
        output[split], _ = _build_dataset(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            raw["event_time"],
            encoder,
            sequences[split],
            config,
            train=split == "train",
        )
    return output


def safe_dense_circular_pairing(recipient, donor_pool, *, horizon: int, fraction: float):
    keep = []
    donors = []
    for row in range(len(recipient)):
        source_id = recipient.source_ids[row]
        pool = np.flatnonzero(donor_pool.source_ids == source_id)
        pool = pool[np.argsort(donor_pool.target_start[pool], kind="mergesort")]
        if len(pool) < 2:
            continue
        offset = max(1, int(round(float(fraction) * len(pool)))) % len(pool)
        if offset == 0:
            offset = 1
        start = int(np.searchsorted(donor_pool.target_start[pool], recipient.target_start[row]))
        chosen = None
        for attempt in range(len(pool)):
            donor = int(pool[(start + offset + attempt) % len(pool)])
            history = recipient.history_event_indices[row]
            target = donor_pool.target_event_indices[donor]
            if np.array_equal(target, recipient.target_event_indices[row]):
                continue
            if np.intersect1d(history, target).size:
                continue
            h0, h1 = int(recipient.history_start[row]), int(recipient.history_stop[row])
            t0, t1 = int(donor_pool.target_start[donor]), int(donor_pool.target_stop[donor])
            gap = h0 - t1 if t1 <= h0 else (t0 - h1 if t0 >= h1 else -1)
            if gap < int(horizon):
                continue
            chosen = donor
            break
        if chosen is not None:
            keep.append(row)
            donors.append(chosen)
    if not keep:
        raise ValueError("no safe dense circular pairing")
    rows = np.asarray(keep, int)
    donor_rows = np.asarray(donors, int)
    paired = recipient.take(rows)
    return replace(
        paired,
        targets=donor_pool.targets[donor_rows].copy(),
        target_start=donor_pool.target_start[donor_rows].copy(),
        target_stop=donor_pool.target_stop[donor_rows].copy(),
        target_event_indices=donor_pool.target_event_indices[donor_rows].copy(),
        target_positions=donor_pool.target_positions[donor_rows].copy(),
        target_event_times=donor_pool.target_event_times[donor_rows].copy(),
        donor_rows=donor_pool.origin_rows[donor_rows].copy(),
        surrogate_kind=f"safe_dense_circular_{fraction:.3f}",
    )


def circular_datasets(raw, encoder, tokens, modes, true_datasets, sequences, config, fraction):
    output = {}
    for split in ("train", "validation", "test"):
        pool = build_event_history_dataset(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            raw["event_time"],
            encoder,
            sequences[split],
            history_length=int(config["history_length"]),
            horizon=int(config["horizon"]),
            stride=1,
            surrogate_kind="dense_circular_donor_pool",
        )
        output[split] = safe_dense_circular_pairing(
            true_datasets[split],
            pool,
            horizon=int(config["horizon"]),
            fraction=float(fraction),
        )
    return output


def null_contract(datasets, raw):
    checks = {}
    eligible = set(raw["eligible"].tolist())
    for split, dataset in datasets.items():
        all_indices = np.concatenate(
            [dataset.history_event_indices.ravel(), dataset.target_event_indices.ravel()]
        )
        checks[split] = {
            "all_indices_train80_only": bool(all(int(value) in eligible for value in all_indices)),
            "history_target_disjoint": bool(
                all(
                    np.intersect1d(history, target).size == 0
                    for history, target in zip(
                        dataset.history_event_indices, dataset.target_event_indices
                    )
                )
            ),
            "finite_targets": bool(np.all(np.isfinite(dataset.targets))),
        }
    return checks


def fit_condition(datasets, raw, encoder, config, frozen, scales):
    baseline = fit_baseline(
        datasets["train"], frozen["selected_baseline"], encoder, int(config["n_modes"])
    )
    baseline_prediction = baseline.predict(datasets["test"])
    baseline_score = score_v24(
        datasets["test"].targets,
        baseline_prediction,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    profile = profile_from_mapping(frozen["recurrent_profile"])
    runs = []
    for seed in map(int, config["final_seeds"]):
        fitted = fit_trainable_residual_rnn(
            datasets["train"],
            baseline=baseline,
            profile=profile,
            scales=scales,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            seed=seed,
            maximum_epochs=int(config["maximum_epochs"]),
            patience=int(config["patience"]),
            minimum_epochs=int(config["minimum_epochs"]),
            validation=datasets["validation"],
        )
        score = score_v24(
            datasets["test"].targets,
            fitted.predict(datasets["test"]),
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        runs.append({
            "seed": seed,
            "score": score_dict(score),
            "best_epoch": fitted.trace.best_epoch,
            "fallback": fitted.trace.best_is_untrained_baseline,
        })
    median = {
        key: float(np.median([run["score"][key] for run in runs]))
        for key in runs[0]["score"]
    }
    gain = {
        key: float(score_dict(baseline_score)[key] - median[key]) for key in median
    }
    return {
        "n_windows": {split: len(dataset) for split, dataset in datasets.items()},
        "baseline_test_score": score_dict(baseline_score),
        "recurrent_median_test_score": median,
        "baseline_minus_rnn_gain": gain,
        "runs": runs,
        "contract_checks": null_contract(datasets, raw),
    }


def run_subject(subject, config, frozen):
    raw, encoder, tokens, modes, true_datasets, sequences, audit = split_sequences(
        subject, config
    )
    scales = family_scales_from_train(
        true_datasets["train"].targets,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    true_path = ROOT / config["output_root"] / "per_subject" / f"{subject}.json"
    true = json.load(true_path.open())
    true_gain = -float(true["rnn_minus_baseline"]["propagation"])

    block = []
    for seed in (101, 211):
        shuffled = {}
        metadata = {}
        for split in sequences:
            shuffled[split], metadata[split] = source_coherent_block_shuffle(
                sequences[split], block_size=5, seed=seed
            )
        datasets = build_from_sequences(raw, encoder, tokens, modes, shuffled, config)
        fitted = fit_condition(datasets, raw, encoder, config, frozen, scales)
        fitted.update({"seed": seed, "permutations": metadata})
        block.append(fitted)

    circular = []
    circular_failures = []
    for fraction in (1.0 / 3.0, 0.5):
        try:
            datasets = circular_datasets(
                raw, encoder, tokens, modes, true_datasets, sequences, config, fraction
            )
            fitted = fit_condition(datasets, raw, encoder, config, frozen, scales)
            fitted["fraction"] = fraction
            circular.append(fitted)
        except ValueError as error:
            circular_failures.append({"fraction": fraction, "reason": str(error)})
    block_gain = float(
        np.median([item["baseline_minus_rnn_gain"]["propagation"] for item in block])
    )
    circular_gain = (
        float(
            np.median(
                [item["baseline_minus_rnn_gain"]["propagation"] for item in circular]
            )
        )
        if circular
        else None
    )
    return {
        "contract": config["contract"],
        "subject": subject,
        "development_subject": subject in config["development_subjects"],
        "true_baseline_minus_rnn_gain": true_gain,
        "block_shuffle": block,
        "safe_circular": circular,
        "safe_circular_failures": circular_failures,
        "chronology_specificity": {
            "true_minus_block_gain": true_gain - block_gain,
            "true_minus_circular_gain": None if circular_gain is None else true_gain - circular_gain,
        },
        "true_contract_checks": audit,
        "old_heldout20_entered": False,
    }


def inference(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    positive = int(np.sum(x > 0))
    try:
        p = float(wilcoxon(x, alternative="greater").pvalue)
    except ValueError:
        p = float("nan")
    return {
        "n": len(x),
        "median": float(np.median(x)),
        "n_positive": positive,
        "wilcoxon_one_sided_greater_p": p,
        "sign_test_one_sided_p": float(
            binomtest(positive, len(x), 0.5, alternative="greater").pvalue
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.open())
    root = ROOT / config["output_root"]
    frozen = json.load((root / "development_screen/FROZEN_PROFILE.json").open())
    subjects = args.subjects or sorted(
        path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz")
    )
    output = root / "chronology_nulls"
    (output / "per_subject").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["torch_num_threads"]))
    results = []
    failures = []
    for index, subject in enumerate(subjects, 1):
        print(f"[v2.5 nulls {index}/{len(subjects)}] {subject}", flush=True)
        try:
            result = run_subject(subject, config, frozen)
            results.append(result)
            with (output / "per_subject" / f"{subject}.json").open("w") as stream:
                json.dump(jsonable(result), stream, indent=2, sort_keys=True)
        except Exception as error:
            failures.append(
                {"subject": subject, "error_type": type(error).__name__, "reason": str(error)}
            )
            print(f"[null failure] {subject}: {type(error).__name__}: {error}", flush=True)
    rows = []
    for result in results:
        row = {
            "subject": result["subject"],
            "development_subject": result["development_subject"],
            "true_gain": result["true_baseline_minus_rnn_gain"],
            **result["chronology_specificity"],
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    extension = frame[~frame["development_subject"]]
    state = {
        "contract": config["contract"],
        "status": "CHRONOLOGY_NULLS_COMPLETE",
        "n_attempted": len(subjects),
        "n_completed": len(results),
        "n_failed": len(failures),
        "extension_true_gain": inference(extension["true_gain"]),
        "extension_true_minus_block": inference(extension["true_minus_block_gain"]),
        "extension_true_minus_circular": inference(extension["true_minus_circular_gain"]),
        "old_heldout20_entered": False,
    }
    with (output / "CHRONOLOGY_NULL_STATE.json").open("w") as stream:
        json.dump(jsonable(state), stream, indent=2, sort_keys=True)
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
