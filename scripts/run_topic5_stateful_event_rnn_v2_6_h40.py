#!/usr/bin/env python3
"""Frozen-profile H=40 sensitivity for Topic 5 stateful RNN v2.6."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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


HORIZON = 40


def h40_datasets(raw, encoder, primary_datasets, config):
    tokens, modes = encoder.event_tokens(raw["rank"], raw["participation"])
    output = {}
    for split, sequences in primary_datasets.items():
        mapping = {sequence.source_id: sequence.event_indices for sequence in sequences}
        output[split] = build_stateful_sequences(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            encoder,
            mapping,
            horizon=HORIZON,
            warmup_events=int(config["warmup_events"]),
        )
    return output


def run_subject(subject, config, output):
    raw, encoder, primary_datasets, partition, _, audit = prepare_subject(subject, config)
    root = output / "h40_sensitivity/per_subject"
    root.mkdir(parents=True, exist_ok=True)
    try:
        datasets = h40_datasets(raw, encoder, primary_datasets, config)
    except ValueError as error:
        result = {
            "contract": config["contract"],
            "sensitivity": "future_horizon_40",
            "status": "INELIGIBLE_H40",
            "subject": subject,
            "dataset": subject.split("_", 1)[0],
            "reason": str(error),
            "old_heldout20_entered": False,
        }
        temporary = root / f"{subject}.json.tmp"
        with temporary.open("w") as stream:
            json.dump(jsonable(result), stream, indent=2, sort_keys=True)
        temporary.replace(root / f"{subject}.json")
        return result
    profile_record = json.load(
        (output / "validation_screen/per_subject" / f"{subject}.json").open()
    )
    profile = profile_from_mapping(profile_record["selected_profile"])
    training_config = dict(config)
    training_config.update(profile_record["selected_training_budget"])
    scales = family_scales_from_sequences(
        datasets["train"],
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
            raise RuntimeError("H40 target mismatch")
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
    median_gain = {
        key: float(np.median([run["rnn_minus_ewma"][key] for run in runs]))
        for key in runs[0]["rnn_minus_ewma"]
    }
    result = {
        "contract": config["contract"],
        "sensitivity": "future_horizon_40",
        "status": "ELIGIBLE_COMPLETE",
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "partition_strategy": partition.strategy,
        "horizon": HORIZON,
        "warmup_events": int(config["warmup_events"]),
        "n_formal_test_targets": int(len(target)),
        "selected_profile": profile_record["selected_profile"],
        "selected_training_budget": profile_record["selected_training_budget"],
        "median_rnn_minus_ewma": median_gain,
        "runs": runs,
        "contract_checks": {
            **audit,
            "same_source_split_as_primary_h20": True,
            "profile_and_training_budget_frozen_from_h20_validation": True,
            "test_not_used_for_h40_parameter_selection": True,
        },
        "provenance": {
            "primary_profile_sha256": sha256(
                output / "validation_screen/per_subject" / f"{subject}.json"
            ),
            "old_heldout20_entered": False,
        },
    }
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
        "median_rnn_minus_ewma": float(np.median(values)),
        "bootstrap_median_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_rnn_better": int(np.sum(values < 0)),
        "wilcoxon_one_sided_less_p": float(wilcoxon(values, alternative="less").pvalue),
        "sign_p": float(
            binomtest(int(np.sum(values < 0)), len(values), 0.5, alternative="greater").pvalue
        ),
    }


def aggregate(config, output):
    subjects = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    records = []
    failures = []
    root = output / "h40_sensitivity/per_subject"
    for subject in subjects:
        path = root / f"{subject}.json"
        if path.exists():
            records.append(json.load(path.open()))
        else:
            failures.append({"subject": subject, "error_type": "MissingArtifact", "reason": str(path)})
    eligible = [row for row in records if row.get("status") == "ELIGIBLE_COMPLETE"]
    ineligible = [row for row in records if row.get("status") == "INELIGIBLE_H40"]
    rows = [
        {
            "subject": row["subject"],
            "dataset": row["dataset"],
            "n_formal_test_targets": row["n_formal_test_targets"],
            "rnn_minus_ewma_propagation": row["median_rnn_minus_ewma"]["propagation"],
            "rnn_minus_ewma_recruitment": row["median_rnn_minus_ewma"]["recruitment"],
        }
        for row in eligible
    ]
    frame = pd.DataFrame(rows)
    destination = output / "h40_sensitivity"
    frame.to_csv(destination / "patient_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        destination / "failures.csv", index=False
    )
    state = {
        "contract": config["contract"],
        "status": "H40_COHORT_AUDIT_COMPLETE" if len(records) == 34 else "INCOMPLETE",
        "n_artifacts": int(len(records)),
        "n_eligible": int(len(eligible)),
        "n_ineligible": int(len(ineligible)),
        "n_failed": int(len(failures)),
        "propagation": inference(frame["rnn_minus_ewma_propagation"])
        if len(frame)
        else {},
        "recruitment": inference(frame["rnn_minus_ewma_recruitment"])
        if len(frame)
        else {},
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (destination / "H40_STATE.json").open("w") as stream:
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
        print(f"[v2.6 H40] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()
