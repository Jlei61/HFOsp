#!/usr/bin/env python3
"""Inference-only recurrent memory-length curve for Topic 5 v2.6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_topic5_stateful_event_rnn_v2_6_dense import load_model  # noqa: E402
from scripts.evaluate_topic5_stateful_event_rnn_v2_6_state_reset import evaluate  # noqa: E402
from scripts.run_topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    DEFAULT_CONFIG,
    jsonable,
    prepare_subject,
    sha256,
)
from scripts.run_topic5_stateful_event_rnn_v2_6_cohort_worker import verify_frozen  # noqa: E402
from src.topic5_stateful_event_rnn_v2_6 import family_scales_from_sequences  # noqa: E402


RESET_INTERVALS = (1, 5, 10, 20, 50, 100)


def run_subject(subject, config, output):
    _, encoder, datasets, _, _, audit = prepare_subject(subject, config)
    scales = family_scales_from_sequences(
        datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    primary = json.load((output / "per_subject" / f"{subject}.json").open())
    seed_rows = []
    for seed in map(int, config["final_seeds"]):
        checkpoint = torch.load(
            output / "checkpoints" / subject / f"seed_{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model, profile = load_model(
            checkpoint,
            input_dim=datasets["train"][0].tokens.shape[1],
            target_dim=datasets["train"][0].targets.shape[1],
            n_modes=int(config["n_modes"]),
        )
        continuous, reference_target = evaluate(
            model,
            datasets["test"],
            checkpoint,
            profile,
            scales,
            config,
            chunk=profile.tbptt_length,
            carry=True,
        )
        interval_scores = {}
        interval_penalties = {}
        for interval in RESET_INTERVALS:
            score, target = evaluate(
                model,
                datasets["test"],
                checkpoint,
                profile,
                scales,
                config,
                chunk=interval,
                carry=False,
            )
            if not np.array_equal(target, reference_target):
                raise RuntimeError("memory-curve target mismatch")
            interval_scores[str(interval)] = score
            interval_penalties[str(interval)] = {
                key: score[key] - continuous[key] for key in continuous
            }
        seed_rows.append(
            {
                "seed": seed,
                "continuous": continuous,
                "reset_interval_score": interval_scores,
                "reset_interval_minus_continuous": interval_penalties,
            }
        )
    median_penalty = {
        str(interval): {
            key: float(
                np.median(
                    [
                        row["reset_interval_minus_continuous"][str(interval)][key]
                        for row in seed_rows
                    ]
                )
            )
            for key in seed_rows[0]["continuous"]
        }
        for interval in RESET_INTERVALS
    }
    result = {
        "contract": config["contract"],
        "subject": subject,
        "dataset": primary["dataset"],
        "n_formal_test_targets": primary["n_formal_test_targets"],
        "reset_intervals_events": list(RESET_INTERVALS),
        "median_reset_interval_minus_continuous": median_penalty,
        "seed_rows": seed_rows,
        "contract_checks": {
            **audit,
            "same_frozen_checkpoint_all_intervals": True,
            "identical_test_targets_all_intervals": True,
        },
        "provenance": {
            "primary_result_sha256": sha256(output / "per_subject" / f"{subject}.json"),
            "old_heldout20_entered": False,
        },
    }
    root = output / "memory_curve/per_subject"
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
        "median_reset_penalty": float(np.median(values)),
        "bootstrap_median_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_reset_worse": int(np.sum(values > 0)),
        "wilcoxon_one_sided_greater_p": float(wilcoxon(values, alternative="greater").pvalue),
        "sign_p": float(
            binomtest(int(np.sum(values > 0)), len(values), 0.5, alternative="greater").pvalue
        ),
    }


def aggregate(config, output):
    subjects = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    records = []
    failures = []
    root = output / "memory_curve/per_subject"
    for subject in subjects:
        path = root / f"{subject}.json"
        if path.exists():
            records.append(json.load(path.open()))
        else:
            failures.append({"subject": subject, "error_type": "MissingArtifact", "reason": str(path)})
    rows = []
    for record in records:
        for interval in RESET_INTERVALS:
            rows.append(
                {
                    "subject": record["subject"],
                    "dataset": record["dataset"],
                    "n_formal_test_targets": record["n_formal_test_targets"],
                    "reset_interval_events": interval,
                    "reset_penalty_propagation": record[
                        "median_reset_interval_minus_continuous"
                    ][str(interval)]["propagation"],
                }
            )
    frame = pd.DataFrame(rows)
    destination = output / "memory_curve"
    frame.to_csv(destination / "patient_interval_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        destination / "failures.csv", index=False
    )
    curves = {
        str(interval): inference(
            frame.loc[
                frame.reset_interval_events == interval, "reset_penalty_propagation"
            ]
        )
        for interval in RESET_INTERVALS
    } if len(frame) else {}
    state = {
        "contract": config["contract"],
        "status": "MEMORY_CURVE_34_COMPLETE" if len(records) == 34 else "INCOMPLETE",
        "n_completed": int(len(records)),
        "n_failed": int(len(failures)),
        "propagation_reset_curve": curves,
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (destination / "MEMORY_CURVE_STATE.json").open("w") as stream:
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
        print(f"[v2.6 memory curve] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()
