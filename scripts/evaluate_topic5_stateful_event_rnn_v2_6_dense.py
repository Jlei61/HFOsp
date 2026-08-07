#!/usr/bin/env python3
"""Dense-anchor test and validation-generalization audit for v2.6 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    DEFAULT_CONFIG,
    jsonable,
    prepare_subject,
    score_dict,
    sha256,
)
from scripts.run_topic5_stateful_event_rnn_v2_6_cohort_worker import (  # noqa: E402
    verify_frozen,
)
from src.topic5_stable_repertoire_event_history_v2_4 import score_v24  # noqa: E402
from src.topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    StatefulEventRNN,
    family_scales_from_sequences,
    fit_continuous_ewma_ridge,
    profile_from_mapping,
    rollout_sequences,
)


def load_model(checkpoint, input_dim, target_dim, n_modes):
    profile = profile_from_mapping(checkpoint["profile"])
    model = StatefulEventRNN(
        input_dim,
        target_dim,
        n_modes,
        profile,
        np.full(target_dim, 0.5, float),
    )
    model.load_state_dict(checkpoint["trained_model_state_dict"])
    return model, profile


def run_subject(subject, config, output):
    raw, encoder, datasets, _, _, audit = prepare_subject(subject, config)
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
    validation_ewma, validation_target, _ = ewma.predict(
        datasets["validation"], formal=False
    )
    test_ewma, test_target, _ = ewma.predict(datasets["test"], formal=False)
    validation_ewma_score = score_v24(
        validation_target,
        validation_ewma,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    test_ewma_score = score_v24(
        test_target,
        test_ewma,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    profile_record = json.load(
        (output / "validation_screen/per_subject" / f"{subject}.json").open()
    )
    test_scores = []
    for seed in map(int, config["final_seeds"]):
        checkpoint_path = output / "checkpoints" / subject / f"seed_{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model, profile = load_model(
            checkpoint,
            input_dim=datasets["train"][0].tokens.shape[1],
            target_dim=datasets["train"][0].targets.shape[1],
            n_modes=int(config["n_modes"]),
        )
        prediction, target, _ = rollout_sequences(
            model,
            datasets["test"],
            mean=checkpoint["feature_mean"],
            scale=checkpoint["feature_scale"],
            chunk_length=int(profile.tbptt_length),
            carry_state=True,
            formal=False,
            return_states=False,
        )
        if not np.array_equal(target, test_target):
            raise RuntimeError("dense test target mismatch")
        test_scores.append(
            score_dict(
                score_v24(
                    target,
                    prediction,
                    n_modes=int(config["n_modes"]),
                    n_contacts=len(encoder.rank_prior),
                    scales=scales,
                )
            )
        )
    recurrent_median = {
        key: float(np.median([row[key] for row in test_scores]))
        for key in test_scores[0]
    }
    dense_gain = {
        key: recurrent_median[key] - score_dict(test_ewma_score)[key]
        for key in recurrent_median
    }
    validation_gain = float(
        profile_record["selected_validation_propagation"]
        - validation_ewma_score.propagation
    )
    formal_primary = json.load((output / "per_subject" / f"{subject}.json").open())
    result = {
        "contract": config["contract"],
        "subject": subject,
        "dataset": formal_primary["dataset"],
        "n_dense_test_targets": int(len(test_target)),
        "n_formal_test_targets": int(formal_primary["n_formal_test_targets"]),
        "validation_selected_rnn_minus_ewma_propagation": validation_gain,
        "dense_test_ewma_score": score_dict(test_ewma_score),
        "dense_test_recurrent_median_score": recurrent_median,
        "dense_test_rnn_minus_ewma": dense_gain,
        "formal_test_rnn_minus_ewma": formal_primary["trained_rnn_minus_ewma"],
        "seed_test_scores": test_scores,
        "contract_checks": audit,
        "provenance": {
            "formal_primary_sha256": sha256(
                output / "per_subject" / f"{subject}.json"
            ),
            "old_heldout20_entered": False,
        },
    }
    root = output / "dense_test_sensitivity/per_subject"
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
        "median": float(np.median(values)),
        "bootstrap_median_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_favorable": int(np.sum(values < 0)),
        "wilcoxon_one_sided_less_p": float(wilcoxon(values, alternative="less").pvalue),
        "sign_p": float(
            binomtest(int(np.sum(values < 0)), len(values), 0.5, alternative="greater").pvalue
        ),
    }


def aggregate(config, output):
    subjects = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    records = []
    failures = []
    root = output / "dense_test_sensitivity/per_subject"
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
            "n_dense_test_targets": row["n_dense_test_targets"],
            "n_formal_test_targets": row["n_formal_test_targets"],
            "validation_gain_propagation": row[
                "validation_selected_rnn_minus_ewma_propagation"
            ],
            "dense_test_gain_propagation": row["dense_test_rnn_minus_ewma"]["propagation"],
            "formal_test_gain_propagation": row["formal_test_rnn_minus_ewma"]["propagation"],
        }
        for row in records
    ]
    frame = pd.DataFrame(rows)
    destination = output / "dense_test_sensitivity"
    frame.to_csv(destination / "patient_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        destination / "failures.csv", index=False
    )
    rho, rho_p = spearmanr(
        frame["validation_gain_propagation"], frame["dense_test_gain_propagation"]
    ) if len(frame) else (np.nan, np.nan)
    state = {
        "contract": config["contract"],
        "status": "DENSE_TEST_34_COMPLETE" if len(records) == 34 else "INCOMPLETE",
        "n_completed": int(len(records)),
        "n_failed": int(len(failures)),
        "dense_test_rnn_minus_ewma_propagation": inference(
            frame["dense_test_gain_propagation"]
        ) if len(frame) else {},
        "formal_test_rnn_minus_ewma_propagation": inference(
            frame["formal_test_gain_propagation"]
        ) if len(frame) else {},
        "validation_to_dense_test_spearman": {
            "rho": float(rho),
            "p": float(rho_p),
        },
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (destination / "DENSE_TEST_STATE.json").open("w") as stream:
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
        print(f"[v2.6 dense sensitivity] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()
