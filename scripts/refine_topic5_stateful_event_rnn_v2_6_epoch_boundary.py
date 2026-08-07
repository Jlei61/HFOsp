#!/usr/bin/env python3
"""Validation-only epoch-boundary audit for Topic 5 stateful RNN v2.6.

This script never evaluates the test split.  It revisits only patients whose
best validation checkpoint is close to the 40-epoch screening ceiling, then
re-trains the leading validation profiles with a longer budget.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pandas as pd
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
    sha256,
)
from src.topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    StatefulProfile,
    family_scales_from_sequences,
    profile_from_mapping,
)


PROFILE_KEYS = tuple(StatefulProfile.__dataclass_fields__)


def _profile_key(row: dict) -> tuple:
    return tuple(row[key] for key in PROFILE_KEYS)


def _leading_unique_candidates(record: dict, top_n: int) -> list[dict]:
    rows = [*record["architecture_screen"], *record["refinement_screen"]]
    rows = sorted(
        rows,
        key=lambda item: (
            float(item["trained_validation_propagation"]),
            int(item["n_parameters"]),
            str(item["stage"]),
            str(item["profile"]),
        ),
    )
    selected = []
    seen = set()
    for row in rows:
        key = _profile_key(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= int(top_n):
            break
    return selected


def _needs_extension(record: dict, *, trigger_epoch: int, top_n: int) -> bool:
    leaders = _leading_unique_candidates(record, top_n)
    selected_epoch = min(
        int(row["best_trained_epoch"])
        for row in record["refinement_screen"]
        if _profile_key(row) == _profile_key(record["selected_profile"])
    )
    return selected_epoch >= int(trigger_epoch) or any(
        int(row["best_trained_epoch"]) >= int(trigger_epoch) for row in leaders
    )


def refine_subject(
    subject: str,
    config: dict,
    output: Path,
    *,
    top_n: int,
    trigger_epoch: int,
    maximum_epochs: int,
    patience: int,
) -> dict:
    path = output / "validation_screen/per_subject" / f"{subject}.json"
    record = json.load(path.open())
    existing_audit = record.get("epoch_boundary_audit")
    if existing_audit is not None:
        return {
            "subject": subject,
            "status": f"ALREADY_{existing_audit['status']}",
            "original_score": float(
                existing_audit.get("original_selection", {}).get(
                    "selected_validation_propagation",
                    record["selected_validation_propagation"],
                )
            ),
            "final_score": float(record["selected_validation_propagation"]),
            "profile_changed": bool(
                existing_audit.get("extended_selection_adopted", False)
            ),
        }
    record.setdefault(
        "selected_training_budget",
        {
            "maximum_epochs": int(config["maximum_epochs"]),
            "minimum_epochs": int(config["minimum_epochs"]),
            "patience": int(config["patience"]),
        },
    )
    leaders = _leading_unique_candidates(record, top_n)
    original = {
        "selected_refinement": record["selected_refinement"],
        "selected_profile": record["selected_profile"],
        "selected_validation_propagation": record[
            "selected_validation_propagation"
        ],
    }
    if not _needs_extension(
        record, trigger_epoch=trigger_epoch, top_n=top_n
    ):
        record["epoch_boundary_audit"] = {
            "status": "NOT_TRIGGERED",
            "trigger_epoch": int(trigger_epoch),
            "top_n": int(top_n),
            "maximum_epochs": int(maximum_epochs),
            "test_results_read": False,
        }
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w") as stream:
            json.dump(jsonable(record), stream, indent=2, sort_keys=True)
        temporary.replace(path)
        return {
            "subject": subject,
            "status": "NOT_TRIGGERED",
            "original_score": float(original["selected_validation_propagation"]),
            "final_score": float(original["selected_validation_propagation"]),
            "profile_changed": False,
        }

    raw, encoder, datasets, _, _, _ = prepare_subject(subject, config)
    scales = family_scales_from_sequences(
        datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    extended_config = dict(config)
    extended_config["maximum_epochs"] = int(maximum_epochs)
    extended_config["patience"] = int(patience)
    rows = []
    for source_row in leaders:
        profile = profile_from_mapping(
            {key: source_row[key] for key in PROFILE_KEYS}
        )
        started = time.time()
        fitted, _ = fit_profile(
            subject,
            profile,
            datasets,
            encoder,
            extended_config,
            scales,
            int(config["screen_seed"]),
        )
        rows.append(
            {
                "source_stage": source_row["stage"],
                "source_profile": source_row["profile"],
                "trained_validation_propagation": float(
                    fitted.trained_validation_score.propagation
                ),
                "nested_validation_propagation": float(
                    fitted.nested_validation_score.propagation
                ),
                "best_trained_epoch": int(fitted.trace.best_trained_epoch),
                "best_nested_epoch": int(fitted.trace.best_nested_epoch),
                "stopped_epoch": int(fitted.trace.stopped_epoch),
                "finite": bool(fitted.trace.finite),
                "n_parameters": int(fitted.n_parameters),
                "runtime_seconds": float(time.time() - started),
                **asdict(profile),
            }
        )

    best_extended = min(
        rows,
        key=lambda item: (
            item["trained_validation_propagation"],
            item["n_parameters"],
            item["source_stage"],
            item["source_profile"],
        ),
    )
    extended_wins = (
        float(best_extended["trained_validation_propagation"])
        < float(original["selected_validation_propagation"]) - 1e-12
    )
    if extended_wins:
        record["selected_refinement"] = (
            "epoch100::"
            f"{best_extended['source_stage']}::{best_extended['source_profile']}"
        )
        record["selected_profile"] = {
            key: jsonable(best_extended[key]) for key in PROFILE_KEYS
        }
        record["selected_validation_propagation"] = float(
            best_extended["trained_validation_propagation"]
        )
        record["selected_training_budget"] = {
            "maximum_epochs": int(maximum_epochs),
            "minimum_epochs": int(config["minimum_epochs"]),
            "patience": int(patience),
        }
    record["epoch_boundary_audit"] = {
        "status": "EXTENDED",
        "trigger_epoch": int(trigger_epoch),
        "top_n": int(top_n),
        "maximum_epochs": int(maximum_epochs),
        "patience": int(patience),
        "original_selection": original,
        "extended_candidates": rows,
        "extended_selection_adopted": bool(extended_wins),
        "test_results_read": False,
    }
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(jsonable(record), stream, indent=2, sort_keys=True)
    temporary.replace(path)
    return {
        "subject": subject,
        "status": "EXTENDED",
        "original_score": float(original["selected_validation_propagation"]),
        "final_score": float(record["selected_validation_propagation"]),
        "profile_changed": bool(extended_wins),
        "n_contacts": int(raw["rank"].shape[1]),
        "n_events_train80": int(len(raw["eligible"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--trigger-epoch", type=int, default=35)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=16)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve().open())
    output = ROOT / config["output_root"]
    screen_root = output / "validation_screen/per_subject"
    subjects = (
        sorted(args.subjects)
        if args.subjects
        else sorted(path.stem for path in screen_root.glob("*.json"))
    )
    torch.set_num_threads(int(config["torch_num_threads"]))
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"[epoch-boundary {index}/{len(subjects)}] {subject}", flush=True)
        rows.append(
            refine_subject(
                subject,
                config,
                output,
                top_n=args.top_n,
                trigger_epoch=args.trigger_epoch,
                maximum_epochs=args.maximum_epochs,
                patience=args.patience,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_screen/epoch_boundary_summary.csv", index=False)
    summary = {
        "contract": config["contract"],
        "status": "VALIDATION_EPOCH_BOUNDARY_AUDIT_COMPLETE",
        "n_subjects": int(len(rows)),
        "n_triggered": int(
            sum(row["status"] in {"EXTENDED", "ALREADY_EXTENDED"} for row in rows)
        ),
        "n_profile_changed": int(sum(row["profile_changed"] for row in rows)),
        "test_results_read": False,
        "trigger_epoch": int(args.trigger_epoch),
        "maximum_epochs": int(args.maximum_epochs),
        "patience": int(args.patience),
        "top_n": int(args.top_n),
        "config_sha256": sha256(args.config.resolve()),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_6.py"),
        "primary_runner_sha256": sha256(
            ROOT / "scripts/run_topic5_stateful_event_rnn_v2_6.py"
        ),
        "boundary_runner_sha256": sha256(Path(__file__)),
    }
    with (output / "validation_screen/EPOCH_BOUNDARY_STATE.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
