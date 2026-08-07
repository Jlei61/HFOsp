#!/usr/bin/env python3
"""Validation-only runner for the repair-only stateful event RNN v2.7.

The data contract and profile grid are imported from frozen v2.6.  The only
training change is that ``fit_profile`` calls the repaired v2.7 fit function.
This runner does not execute or score the human test partition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_resource_guard import (  # noqa: E402
    configure_torch_threads,
    pin_thread_environment,
)

pin_thread_environment(1)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from scripts.run_topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    array_sha256,
    jsonable,
    merge_profile,
    prepare_subject,
    sha256,
)
from src.topic5_stateful_event_rnn_v2_7 import (  # noqa: E402
    StatefulProfile,
    family_scales_from_sequences,
    fit_stateful_event_rnn,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stateful_event_rnn_v2_7.yaml"
PARENT_CONFIG = ROOT / "config/topic5_stateful_event_rnn_v2_6.yaml"
PARENT_MODULE = ROOT / "src/topic5_stateful_event_rnn_v2_6.py"
PARENT_RUNNER = ROOT / "scripts/run_topic5_stateful_event_rnn_v2_6.py"
PARENT_FROZEN_STATE = (
    ROOT
    / "results/topic5_stateful_event_sequence_rnn/v2_6/validation_screen"
    / "FROZEN_VALIDATION_STATE.json"
)
PARENT_TEST_STATE = (
    ROOT
    / "results/topic5_stateful_event_sequence_rnn/v2_6/STATEFUL_TEST_STATE.json"
)
V27_MODULE = ROOT / "src/topic5_stateful_event_rnn_v2_7.py"
V27_RUNNER = Path(__file__).resolve()
V27_WORKER = ROOT / "scripts/run_topic5_stateful_event_rnn_v2_7_cohort_worker.py"


def assert_repair_only_config(config_path: Path) -> dict[str, Any]:
    """Fail closed if v2.7 changes anything except namespace and output root."""

    config = yaml.safe_load(config_path.open())
    parent = yaml.safe_load(PARENT_CONFIG.open())
    expected_contract = "topic5_stateful_event_sequence_rnn_v2_7"
    expected_output = "results/topic5_stateful_event_sequence_rnn/v2_7"
    if config.get("contract") != expected_contract:
        raise RuntimeError("v2.7 contract namespace mismatch")
    if config.get("output_root") != expected_output:
        raise RuntimeError("v2.7 output root must be the parallel v2_7 directory")
    child_science = dict(config)
    parent_science = dict(parent)
    child_science.pop("contract", None)
    child_science.pop("output_root", None)
    parent_science.pop("contract", None)
    parent_science.pop("output_root", None)
    if child_science != parent_science:
        raise RuntimeError("v2.7 scientific/training grid drifted from frozen v2.6")
    return config


def provenance_manifest(config_path: Path) -> dict[str, Any]:
    """Return verified parent and v2.7 hashes for every frozen state."""

    parent_state = json.load(PARENT_FROZEN_STATE.open())
    parent_actual = {
        "config_sha256": sha256(PARENT_CONFIG),
        "module_sha256": sha256(PARENT_MODULE),
        "runner_sha256": sha256(PARENT_RUNNER),
    }
    for key, value in parent_actual.items():
        if parent_state.get(key) != value:
            raise RuntimeError(f"v2.6 parent frozen hash mismatch: {key}")
    if parent_state.get("status") != "ALL_PATIENT_VALIDATION_PROFILES_FROZEN":
        raise RuntimeError("v2.6 parent validation state is not frozen")
    if not PARENT_TEST_STATE.exists():
        raise RuntimeError("v2.6 parent primary test state is missing")
    return {
        "parent_v2_6": {
            **parent_actual,
            "frozen_validation_state_sha256": sha256(PARENT_FROZEN_STATE),
            "primary_test_state_sha256": sha256(PARENT_TEST_STATE),
            "frozen_status": parent_state["status"],
        },
        "v2_7": {
            "config_sha256": sha256(config_path),
            "module_sha256": sha256(V27_MODULE),
            "runner_sha256": sha256(V27_RUNNER),
            "cohort_worker_sha256": sha256(V27_WORKER),
        },
        "repair_only_grid_match": True,
    }


def fit_profile(profile, datasets, encoder, config, scales, seed):
    """Fit one frozen profile through the repaired v2.7 implementation."""

    started = time.time()
    fitted = fit_stateful_event_rnn(
        datasets["train"],
        datasets["validation"],
        profile=profile,
        scales=scales,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        seed=int(seed),
        maximum_epochs=int(config["maximum_epochs"]),
        minimum_epochs=int(config["minimum_epochs"]),
        patience=int(config["patience"]),
        carry_state=True,
    )
    return fitted, time.time() - started


def _screen_row(stage: str, name: str, profile, fitted, runtime: float) -> dict:
    return {
        "stage": stage,
        "profile": name,
        "trained_validation_propagation": (
            fitted.trained_validation_score.propagation
        ),
        "nested_validation_propagation": (
            fitted.nested_validation_score.propagation
        ),
        "best_trained_epoch": fitted.trace.best_trained_epoch,
        "best_nested_epoch": fitted.trace.best_nested_epoch,
        "stopped_epoch": fitted.trace.stopped_epoch,
        "n_parameters": fitted.n_parameters,
        "runtime_seconds": runtime,
        **asdict(profile),
    }


def screen_subject(subject: str, config: dict[str, Any], output: Path) -> dict:
    """Run the frozen architecture/refinement screen for one patient."""

    raw, encoder, datasets, partition, _, audit = prepare_subject(subject, config)
    scales = family_scales_from_sequences(
        datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    architecture_rows = []
    for values in config["architecture_profiles"]:
        name, profile = merge_profile(config["base_profile"], values)
        fitted, runtime = fit_profile(
            profile, datasets, encoder, config, scales, config["screen_seed"]
        )
        architecture_rows.append(
            _screen_row("architecture", name, profile, fitted, runtime)
        )
    architecture_frame = pd.DataFrame(architecture_rows).sort_values(
        ["trained_validation_propagation", "n_parameters", "profile"]
    )
    best_architecture = architecture_frame.iloc[0]
    selected_architecture = {
        key: best_architecture[key] for key in StatefulProfile.__dataclass_fields__
    }

    refinement_rows = []
    for values in config["training_refinements"]:
        name, profile = merge_profile(selected_architecture, values)
        fitted, runtime = fit_profile(
            profile, datasets, encoder, config, scales, config["screen_seed"]
        )
        refinement_rows.append(
            _screen_row("refinement", name, profile, fitted, runtime)
        )
    refinement_frame = pd.DataFrame(refinement_rows).sort_values(
        ["trained_validation_propagation", "n_parameters", "profile"]
    )
    best = refinement_frame.iloc[0]
    record = {
        "contract": config["contract"],
        "subject": subject,
        "status": "PATIENT_VALIDATION_PROFILE_SCREENED",
        "n_contacts": int(raw["rank"].shape[1]),
        "n_events_train80": int(len(raw["eligible"])),
        "partition_strategy": partition.strategy,
        "n_sequences": {key: len(value) for key, value in datasets.items()},
        "n_dense_targets": {
            key: int(sum(item.valid_mask.sum() for item in value))
            for key, value in datasets.items()
        },
        "n_formal_targets": {
            key: int(sum(item.formal_mask.sum() for item in value))
            for key, value in datasets.items()
        },
        "selected_architecture": str(best_architecture["profile"]),
        "selected_refinement": str(best["profile"]),
        "selected_profile": {
            key: jsonable(best[key]) for key in StatefulProfile.__dataclass_fields__
        },
        "selected_training_budget": {
            "maximum_epochs": int(config["maximum_epochs"]),
            "minimum_epochs": int(config["minimum_epochs"]),
            "patience": int(config["patience"]),
        },
        "selected_validation_propagation": float(
            best["trained_validation_propagation"]
        ),
        "architecture_screen": architecture_frame.to_dict("records"),
        "refinement_screen": refinement_frame.to_dict("records"),
        "contract_checks": audit,
        "provenance": {
            "dataset_sha256": sha256(raw["data_path"]),
            "source_mapping_sha256": sha256(raw["mapping_path"]),
            "eligible_indices_sha256": array_sha256(
                raw["eligible"].astype("int64")
            ),
            "old_heldout20_entered": False,
            "v2_7_module_sha256": sha256(V27_MODULE),
            "v2_7_runner_sha256": sha256(V27_RUNNER),
        },
    }
    subject_root = output / "validation_screen/per_subject"
    subject_root.mkdir(parents=True, exist_ok=True)
    destination = subject_root / f"{subject}.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(jsonable(record), stream, indent=2, sort_keys=True)
    temporary.replace(destination)
    return record


def freeze_screen(
    config: dict[str, Any], config_path: Path, output: Path, subjects=None
) -> dict:
    """Freeze validation selection only after the boundary audit is present."""

    available = sorted(
        path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz")
    )
    selected = available if not subjects else sorted(subjects)
    records = []
    failures = []
    for subject in selected:
        path = output / "validation_screen/per_subject" / f"{subject}.json"
        if not path.exists():
            failures.append(
                {"subject": subject, "error_type": "MissingArtifact", "reason": str(path)}
            )
            continue
        record = json.load(path.open())
        boundary = record.get("epoch_boundary_audit")
        if boundary is None or boundary.get("test_results_read") is not False:
            failures.append(
                {
                    "subject": subject,
                    "error_type": "BoundaryAuditMissing",
                    "reason": "validation-only epoch boundary audit is incomplete",
                }
            )
            continue
        if not all(record.get("contract_checks", {}).values()):
            failures.append(
                {
                    "subject": subject,
                    "error_type": "DataContractFailure",
                    "reason": "validation artifact has a failed contract check",
                }
            )
            continue
        records.append(record)

    root = output / "validation_screen"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "subject": item["subject"],
            "n_contacts": item["n_contacts"],
            "n_events_train80": item["n_events_train80"],
            "selected_architecture": item["selected_architecture"],
            "selected_refinement": item["selected_refinement"],
            "selected_cell": item["selected_profile"]["cell"],
            "selected_hidden_size": item["selected_profile"]["hidden_size"],
            "selected_tbptt_length": item["selected_profile"]["tbptt_length"],
            "selected_maximum_epochs": item["selected_training_budget"][
                "maximum_epochs"
            ],
            "selected_validation_propagation": item[
                "selected_validation_propagation"
            ],
        }
        for item in records
    ]
    pd.DataFrame(rows).to_csv(root / "patient_profile_summary.csv", index=False)
    pd.DataFrame(
        failures, columns=("subject", "error_type", "reason")
    ).to_csv(root / "failures.csv", index=False)
    provenance = provenance_manifest(config_path)
    state = {
        "contract": config["contract"],
        "status": (
            "ALL_PATIENT_VALIDATION_PROFILES_FROZEN"
            if len(records) == 34 and len(selected) == 34
            else "INCOMPLETE"
        ),
        "n_attempted": len(selected),
        "n_completed": len(records),
        "n_failed": len(failures),
        "test_results_read_during_selection": False,
        "old_heldout20_entered": False,
        **provenance["v2_7"],
        "parent_v2_6": provenance["parent_v2_6"],
        "repair_only_grid_match": provenance["repair_only_grid_match"],
    }
    with (root / "FROZEN_VALIDATION_STATE.json").open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=("screen-patients", "freeze-screen"), required=True
    )
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = assert_repair_only_config(config_path)
    provenance_manifest(config_path)
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    configure_torch_threads(torch, int(config["torch_num_threads"]))
    if args.phase == "freeze-screen":
        freeze_screen(config, config_path, output, args.subjects)
        return
    if not args.subjects:
        raise ValueError("screen-patients requires --subjects")
    for subject in args.subjects:
        print(f"[v2.7 validation screen] {subject}", flush=True)
        screen_subject(subject, config, output)


if __name__ == "__main__":
    main()
