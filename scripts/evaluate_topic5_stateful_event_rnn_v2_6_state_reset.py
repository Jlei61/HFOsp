#!/usr/bin/env python3
"""Hidden-state carry ablation for frozen Topic 5 v2.6 checkpoints."""
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
from scripts.run_topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    DEFAULT_CONFIG,
    jsonable,
    prepare_subject,
    score_dict,
    sha256,
)
from scripts.run_topic5_stateful_event_rnn_v2_6_cohort_worker import verify_frozen  # noqa: E402
from src.topic5_stable_repertoire_event_history_v2_4 import score_v24  # noqa: E402
from src.topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    family_scales_from_sequences,
    rollout_sequences,
)


def evaluate(model, sequences, checkpoint, profile, scales, config, *, chunk, carry):
    prediction, target, _ = rollout_sequences(
        model,
        sequences,
        mean=checkpoint["feature_mean"],
        scale=checkpoint["feature_scale"],
        chunk_length=int(chunk),
        carry_state=bool(carry),
        formal=True,
        return_states=False,
    )
    score = score_v24(
        target,
        prediction,
        n_modes=int(config["n_modes"]),
        n_contacts=(target.shape[1] - int(config["n_modes"])) // 2,
        scales=scales,
    )
    return score_dict(score), target


def run_subject(subject, config, output):
    _, encoder, datasets, _, _, audit = prepare_subject(subject, config)
    scales = family_scales_from_sequences(
        datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    primary = json.load((output / "per_subject" / f"{subject}.json").open())
    runs = []
    for seed in map(int, config["final_seeds"]):
        path = output / "checkpoints" / subject / f"seed_{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model, profile = load_model(
            checkpoint,
            input_dim=datasets["train"][0].tokens.shape[1],
            target_dim=datasets["train"][0].targets.shape[1],
            n_modes=int(config["n_modes"]),
        )
        continuous, target = evaluate(
            model,
            datasets["test"],
            checkpoint,
            profile,
            scales,
            config,
            chunk=profile.tbptt_length,
            carry=True,
        )
        reset_chunk, chunk_target = evaluate(
            model,
            datasets["test"],
            checkpoint,
            profile,
            scales,
            config,
            chunk=profile.tbptt_length,
            carry=False,
        )
        reset_event, event_target = evaluate(
            model,
            datasets["test"],
            checkpoint,
            profile,
            scales,
            config,
            chunk=1,
            carry=False,
        )
        if not (
            np.array_equal(target, chunk_target)
            and np.array_equal(target, event_target)
        ):
            raise RuntimeError("state-reset ablation target mismatch")
        runs.append(
            {
                "seed": seed,
                "continuous": continuous,
                "reset_tbptt_chunk": reset_chunk,
                "reset_every_event": reset_event,
                "reset_chunk_minus_continuous": {
                    key: reset_chunk[key] - continuous[key] for key in continuous
                },
                "reset_event_minus_continuous": {
                    key: reset_event[key] - continuous[key] for key in continuous
                },
            }
        )
    continuous_median = float(
        np.median([row["continuous"]["propagation"] for row in runs])
    )
    expected = float(primary["trained_recurrent_median_test_score"]["propagation"])
    if not np.isclose(continuous_median, expected, rtol=0.0, atol=1e-7):
        raise RuntimeError("checkpoint carry score does not reproduce primary artifact")
    result = {
        "contract": config["contract"],
        "subject": subject,
        "dataset": primary["dataset"],
        "n_formal_test_targets": primary["n_formal_test_targets"],
        "runs": runs,
        "median_reset_chunk_minus_continuous": {
            key: float(np.median([row["reset_chunk_minus_continuous"][key] for row in runs]))
            for key in runs[0]["continuous"]
        },
        "median_reset_event_minus_continuous": {
            key: float(np.median([row["reset_event_minus_continuous"][key] for row in runs]))
            for key in runs[0]["continuous"]
        },
        "contract_checks": {
            **audit,
            "continuous_checkpoint_reproduces_primary": True,
            "identical_test_targets_across_ablations": True,
        },
        "provenance": {
            "primary_result_sha256": sha256(output / "per_subject" / f"{subject}.json"),
            "old_heldout20_entered": False,
        },
    }
    root = output / "state_reset_ablation/per_subject"
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
        "median_reset_penalty": float(np.median(values)),
        "bootstrap_median_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_reset_worse": int(np.sum(values > 0)),
        "wilcoxon_one_sided_greater_p": float(
            wilcoxon(values, alternative="greater").pvalue
        ),
        "sign_p": float(
            binomtest(int(np.sum(values > 0)), len(values), 0.5, alternative="greater").pvalue
        ),
    }


def aggregate(config, output):
    subjects = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    records = []
    failures = []
    root = output / "state_reset_ablation/per_subject"
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
            "n_formal_test_targets": row["n_formal_test_targets"],
            "reset_chunk_penalty_propagation": row[
                "median_reset_chunk_minus_continuous"
            ]["propagation"],
            "reset_event_penalty_propagation": row[
                "median_reset_event_minus_continuous"
            ]["propagation"],
        }
        for row in records
    ]
    frame = pd.DataFrame(rows)
    destination = output / "state_reset_ablation"
    frame.to_csv(destination / "patient_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        destination / "failures.csv", index=False
    )
    state = {
        "contract": config["contract"],
        "status": "STATE_RESET_34_COMPLETE" if len(records) == 34 else "INCOMPLETE",
        "n_completed": int(len(records)),
        "n_failed": int(len(failures)),
        "reset_tbptt_chunk": inference(frame["reset_chunk_penalty_propagation"])
        if len(frame)
        else {},
        "reset_every_event": inference(frame["reset_event_penalty_propagation"])
        if len(frame)
        else {},
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (destination / "STATE_RESET_STATE.json").open("w") as stream:
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
        print(f"[v2.6 state reset] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()
