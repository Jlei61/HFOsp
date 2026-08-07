#!/usr/bin/env python3
"""Run frozen v2.5.4 recurrent profile at longer event-history lengths.

This is a sensitivity analysis, not a second hyperparameter screen.  The
baseline, recurrent architecture, optimizer, checkpoint rule, and seeds are
read from the frozen L=20 development profile.  Only ``history_length`` changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_trainable_event_rnn_v2_5 import (  # noqa: E402
    jsonable,
    patient_inference,
    run_subject,
    sha256,
)


DEFAULT_CONFIG = ROOT / "config/topic5_trainable_event_rnn_v2_5.yaml"
PRIMARY_RUNNER = ROOT / "scripts/run_topic5_trainable_event_rnn_v2_5.py"
MODEL_MODULE = ROOT / "src/topic5_trainable_event_rnn_v2_5.py"


def verify_primary_freeze(
    config_path: Path,
    frozen: dict[str, Any],
) -> None:
    expected = {
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(MODEL_MODULE),
        "runner_sha256": sha256(PRIMARY_RUNNER),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "observed": value}
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"primary v2.5.4 freeze mismatch: {mismatches}")


def aggregate_length(
    results: list[dict[str, Any]],
    failures: list[dict[str, str]],
    *,
    history_length: int,
    config: dict[str, Any],
    config_path: Path,
    frozen: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    rows = [
        {
            "subject": result["subject"],
            "dataset": result["dataset"],
            "development_subject": result["development_subject"],
            "support_grade": result["support_grade"],
            "split_strategy": result["partition"]["strategy"],
            "n_events_train80": result["n_events_train80"],
            "n_train_windows": result["n_windows"]["train_dense"],
            "n_test_windows": result["n_windows"]["test_formal"],
            "baseline_propagation": result["baseline_test_score"]["propagation"],
            "rnn_propagation": result["recurrent_median_test_score"]["propagation"],
            "rnn_minus_baseline_propagation": result["rnn_minus_baseline"]["propagation"],
            "rnn_minus_baseline_recruitment": result["rnn_minus_baseline"]["recruitment"],
            "rnn_minus_baseline_repertoire": result["rnn_minus_baseline"]["repertoire"],
            "all_runs_finite": all(
                run["trace"]["finite"] for run in result["recurrent_runs"]
            ),
        }
        for result in results
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    pd.DataFrame(
        failures,
        columns=("subject", "error_type", "reason"),
    ).to_csv(output / "denominator_failures.csv", index=False)
    extension = frame[~frame["development_subject"]] if len(frame) else frame
    state = {
        "contract": config["contract"],
        "status": "FROZEN_HISTORY_LENGTH_SENSITIVITY_COMPLETE",
        "history_length": int(history_length),
        "horizon": int(config["horizon"]),
        "n_subjects_attempted": 34,
        "n_subjects_completed": int(len(frame)),
        "n_subjects_failed": int(len(failures)),
        "failed_subjects": [item["subject"] for item in failures],
        "n_development_completed": int(frame["development_subject"].sum()) if len(frame) else 0,
        "n_extension_completed": int((~frame["development_subject"]).sum()) if len(frame) else 0,
        "all_completed_descriptive_propagation": patient_inference(
            frame["rnn_minus_baseline_propagation"].to_numpy()
        ) if len(frame) else {},
        "extension_propagation": patient_inference(
            extension["rnn_minus_baseline_propagation"].to_numpy()
        ) if len(extension) else {},
        "extension_recruitment": patient_inference(
            extension["rnn_minus_baseline_recruitment"].to_numpy()
        ) if len(extension) else {},
        "frozen_selected_baseline": frozen["selected_baseline"],
        "frozen_recurrent_profile": frozen["recurrent_profile"],
        "frozen_final_seeds": list(map(int, config["final_seeds"])),
        "selection_or_retuning_at_this_length": False,
        "primary_l20_config_sha256": sha256(config_path),
        "primary_l20_module_sha256": sha256(MODEL_MODULE),
        "primary_l20_runner_sha256": sha256(PRIMARY_RUNNER),
        "sensitivity_runner_sha256": sha256(Path(__file__)),
        "execution_environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_num_threads": int(config["torch_num_threads"]),
        },
        "old_heldout20_entered": False,
    }
    with (output / "HISTORY_LENGTH_STATE.json").open("w") as stream:
        json.dump(jsonable(state), stream, indent=2, sort_keys=True)
    return state


def run_length(
    base_config: dict[str, Any],
    config_path: Path,
    frozen: dict[str, Any],
    history_length: int,
    subjects: list[str],
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["history_length"] = int(history_length)
    config["contract"] = (
        f"topic5_trainable_event_rnn_v2_5_4_history_length_sensitivity_l{history_length}"
    )
    output = (
        ROOT
        / base_config["output_root"]
        / "history_length_sensitivity"
        / f"l{history_length}"
    )
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, subject in enumerate(subjects, 1):
        print(
            f"[v2.5.4 L={history_length} {index}/{len(subjects)}] {subject}",
            flush=True,
        )
        try:
            result = run_subject(subject, config, frozen, output)
            result["history_length_sensitivity"] = {
                "history_length": int(history_length),
                "primary_history_length": int(base_config["history_length"]),
                "retuned": False,
            }
            per_subject = output / "per_subject" / f"{subject}.json"
            with per_subject.open("w") as stream:
                json.dump(jsonable(result), stream, indent=2, sort_keys=True)
            results.append(result)
        except Exception as error:
            failures.append(
                {
                    "subject": subject,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            print(
                f"[v2.5.4 L={history_length} failure] {subject}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    state = aggregate_length(
        results,
        failures,
        history_length=history_length,
        config=config,
        config_path=config_path,
        frozen=frozen,
        output=output,
    )
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--history-lengths", type=int, nargs="+", default=(40, 80))
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.open())
    primary_output = ROOT / config["output_root"]
    frozen_path = primary_output / "development_screen/FROZEN_PROFILE.json"
    frozen = json.load(frozen_path.open())
    verify_primary_freeze(config_path, frozen)
    torch.set_num_threads(int(config["torch_num_threads"]))
    subjects = (
        list(args.subjects)
        if args.subjects
        else sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    )
    for history_length in args.history_lengths:
        if int(history_length) <= int(config["history_length"]):
            raise ValueError("sensitivity history lengths must exceed primary L=20")
        run_length(config, config_path, frozen, int(history_length), subjects)


if __name__ == "__main__":
    main()
