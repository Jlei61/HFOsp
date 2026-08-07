#!/usr/bin/env python3
"""Source-level time-reversal null for Topic 5 stateful RNN v2.6."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    DEFAULT_CONFIG,
    fit_profile,
    jsonable,
    prepare_subject,
    score_dict,
    sha256,
)
from scripts.run_topic5_stateful_event_rnn_v2_6_cohort_worker import verify_frozen  # noqa: E402
from src.topic5_stable_repertoire_event_history_v2_4 import score_v24  # noqa: E402
from src.topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    build_stateful_sequences,
    family_scales_from_sequences,
    fit_continuous_ewma_ridge,
    profile_from_mapping,
)


def reversed_datasets(raw, encoder, true_datasets, config):
    tokens, modes = encoder.event_tokens(raw["rank"], raw["participation"])
    output = {}
    hashes = {}
    for split, sequences in true_datasets.items():
        mapping = {}
        for sequence in sequences:
            reversed_indices = sequence.event_indices[::-1].copy()
            if not np.array_equal(
                np.sort(reversed_indices), np.sort(sequence.event_indices)
            ):
                raise RuntimeError("reversal changed source event set")
            if len(reversed_indices) > 1 and np.array_equal(
                reversed_indices, sequence.event_indices
            ):
                raise RuntimeError("reversal did not change source order")
            mapping[sequence.source_id] = reversed_indices
            digest = hashlib.sha256(np.ascontiguousarray(reversed_indices).tobytes())
            hashes[f"{split}::{sequence.source_id}"] = digest.hexdigest()
        output[split] = build_stateful_sequences(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            encoder,
            mapping,
            horizon=int(config["horizon"]),
            warmup_events=int(config["warmup_events"]),
        )
    return output, hashes


def run_subject(subject, config, output):
    profile_record = json.load(
        (output / "validation_screen/per_subject" / f"{subject}.json").open()
    )
    primary = json.load((output / "per_subject" / f"{subject}.json").open())
    raw, encoder, true_datasets, _, _, audit = prepare_subject(subject, config)
    datasets, reversal_hashes = reversed_datasets(
        raw, encoder, true_datasets, config
    )
    profile = profile_from_mapping(profile_record["selected_profile"])
    training_config = dict(config)
    training_config.update(profile_record["selected_training_budget"])
    scales = family_scales_from_sequences(
        true_datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    ewma = fit_continuous_ewma_ridge(
        datasets["train"],
        decay=float(config["ewma_decay"]),
        alpha=float(config["ewma_alpha"]),
        n_modes=int(config["n_modes"]),
    )
    ewma_prediction, target, _ = ewma.predict(datasets["test"], formal=True)
    ewma_score = score_v24(
        target,
        ewma_prediction,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    runs = []
    for seed in map(int, config["final_seeds"]):
        started = time.time()
        fitted, _ = fit_profile(
            subject,
            profile,
            datasets,
            encoder,
            training_config,
            scales,
            seed,
        )
        prediction, recurrent_target, _ = fitted.predict(
            datasets["test"], checkpoint="trained", formal=True
        )
        if not np.array_equal(target, recurrent_target):
            raise RuntimeError("reversal-null target mismatch")
        recurrent_score = score_v24(
            target,
            prediction,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        gain = {
            key: score_dict(recurrent_score)[key] - score_dict(ewma_score)[key]
            for key in score_dict(recurrent_score)
        }
        runs.append(
            {
                "seed": seed,
                "recurrent_score": score_dict(recurrent_score),
                "ewma_score": score_dict(ewma_score),
                "rnn_minus_ewma": gain,
                "trace": asdict(fitted.trace),
                "runtime_seconds": float(time.time() - started),
            }
        )
    null_median = {
        key: float(np.median([row["rnn_minus_ewma"][key] for row in runs]))
        for key in runs[0]["rnn_minus_ewma"]
    }
    true_gain = primary["trained_rnn_minus_ewma"]
    result = {
        "contract": config["contract"],
        "null": "source_level_time_reversal",
        "subject": subject,
        "dataset": primary["dataset"],
        "selected_profile": profile_record["selected_profile"],
        "selected_training_budget": profile_record["selected_training_budget"],
        "true_rnn_minus_ewma": true_gain,
        "reversal_rnn_minus_ewma": null_median,
        "true_minus_reversal_gain": {
            key: float(true_gain[key] - null_median[key]) for key in null_median
        },
        "runs": runs,
        "contract_checks": {
            **audit,
            "same_event_set_within_each_source": True,
            "source_time_arrow_reversed": True,
            "targets_rebuilt_from_reversed_sequence": True,
            "test_results_not_used_for_model_selection": True,
        },
        "provenance": {
            "reversed_index_sha256": reversal_hashes,
            "primary_result_sha256": sha256(output / "per_subject" / f"{subject}.json"),
            "old_heldout20_entered": False,
        },
    }
    root = output / "chronology_null/time_reversal/per_subject"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{subject}.json.tmp"
    with temporary.open("w") as stream:
        json.dump(jsonable(result), stream, indent=2, sort_keys=True)
    temporary.replace(root / f"{subject}.json")
    return result


def inference(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(20260802)
    boot = np.median(rng.choice(values, (10000, len(values)), replace=True), axis=1)
    return {
        "n": int(len(values)),
        "median_true_minus_reversal_gain": float(np.median(values)),
        "bootstrap_median_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_true_more_favorable": int(np.sum(values < 0)),
        "wilcoxon_one_sided_less_p": float(wilcoxon(values, alternative="less").pvalue),
        "sign_p": float(
            binomtest(int(np.sum(values < 0)), len(values), 0.5, alternative="greater").pvalue
        ),
    }


def aggregate(config, output):
    subjects = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    records = []
    failures = []
    root = output / "chronology_null/time_reversal/per_subject"
    for subject in subjects:
        path = root / f"{subject}.json"
        if path.exists():
            records.append(json.load(path.open()))
        else:
            failures.append({"subject": subject, "error_type": "MissingArtifact", "reason": str(path)})
    rows = [
        {
            "subject": row["subject"],
            "dataset": row["dataset"],
            "true_gain_propagation": row["true_rnn_minus_ewma"]["propagation"],
            "reversal_gain_propagation": row["reversal_rnn_minus_ewma"]["propagation"],
            "true_minus_reversal_propagation": row["true_minus_reversal_gain"]["propagation"],
        }
        for row in records
    ]
    frame = pd.DataFrame(rows)
    destination = output / "chronology_null/time_reversal"
    frame.to_csv(destination / "patient_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        destination / "failures.csv", index=False
    )
    state = {
        "contract": config["contract"],
        "status": "TIME_REVERSAL_34_COMPLETE" if len(records) == 34 else "INCOMPLETE",
        "n_completed": int(len(records)),
        "n_failed": int(len(failures)),
        "propagation": inference(frame["true_minus_reversal_propagation"])
        if len(frame)
        else {},
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (destination / "TIME_REVERSAL_STATE.json").open("w") as stream:
        json.dump(jsonable(state), stream, indent=2, sort_keys=True)
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("patients", "aggregate"), required=True)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.open())
    output = ROOT / config["output_root"]
    verify_frozen(config_path, output)
    torch.set_num_threads(int(config["torch_num_threads"]))
    if args.phase == "aggregate":
        aggregate(config, output)
        return
    if not args.subjects:
        raise ValueError("patients phase requires --subjects")
    for subject in args.subjects:
        print(f"[v2.6 reversal null] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()
